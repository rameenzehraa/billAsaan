"""Batch pipeline: bill images + template CSVs -> output CSVs.

Usage: python run.py --bills <dir> --templates <dir> --out <dir> [--refresh]
"""
import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cache
from answer import LOG_DIR as ANSWER_LOG_DIR
from answer import answer_bill
from checks import run_checks
from config import load_settings, provider_for
from extract import SETTINGS_ID, extract_bill
from images import IMAGE_EXTS, load_image
from schema import ExtractionResult, empty_level1, to_level1_json

log = logging.getLogger("run")
MAX_WORKERS = 3


def find_image(bills_dir: Path, bill_id: str) -> Path | None:
    for p in sorted(bills_dir.glob(f"{bill_id}.*")):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def process_bill(bill_id: str, bills_dir: Path, refresh: bool) -> tuple[ExtractionResult, dict]:
    """Never raises. Returns (result, info) where info has cache/meta/check fields for the summary."""
    provider = provider_for(bill_id)
    info = {"cache": "-", "latency_s": None, "input_tokens": None, "output_tokens": None,
            "reasked": False, "kept": "-", "reask_reason": None, "checks": [], "error": None}
    try:
        path = find_image(bills_dir, bill_id)
        if path is None:
            raise FileNotFoundError(f"no image for {bill_id} in {bills_dir}")
        image = load_image(path)
        model = load_settings().model_name
        key = cache.extract_key(image[0], model, SETTINGS_ID)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        log.error("bill=%s %s", bill_id, info["error"])
        return ExtractionResult(level1=empty_level1(bill_id, provider)), info

    if not refresh:
        entry = cache.get(key)
        if entry is not None:
            try:
                result = ExtractionResult.model_validate(entry["result"])
                # Checks are cheap and deterministic: recompute so they reflect the current checks.py.
                info.update(cache="hit", latency_s=0.0, input_tokens=0, output_tokens=0,
                            reasked=entry["meta"].get("reasked", False), kept=entry["meta"].get("kept") or "-",
                            reask_reason=entry["meta"].get("reask_reason"), checks=run_checks(result.level1, result.internal))
                return result, info
            except Exception:  # noqa: BLE001 (corrupt entry: re-extract)
                log.warning("bill=%s corrupt cache entry, re-extracting", bill_id)

    result, meta = extract_bill(bill_id, provider, path, image=image)
    info.update(cache="miss", latency_s=meta["latency_s"], input_tokens=meta["input_tokens"],
                output_tokens=meta["output_tokens"], reasked=meta["reasked"], kept=meta["kept"] or "-",
                reask_reason=meta["reask_reason"], checks=meta["checks"], error=meta["error"])
    if meta["error"] is None:  # only successful, validated results are cached
        entry = {"bill_id": bill_id, "key": key, "model": model, "result": result.model_dump(mode="json"),
                 "checks": meta["checks"], "meta": {k: v for k, v in meta.items() if k != "checks"}}
        cache.put(key, entry)
        cache.save_result(bill_id, entry)
    return result, info


def read_template(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def short_status(checks: list[dict]) -> str:
    abbrev = {"Matched": "M", "Mismatch": "X", "Insufficient Data": "?"}
    return " ".join(f"{c['name'].split('_')[0]}:{abbrev.get(c['status'], c['status'])}" for c in checks) or "-"


class InputError(ValueError):
    """Bad or missing input folders/templates. The message is shown to the user as is."""


L1_COLUMNS = ("bill_id", "json")
L2_COLUMNS = ("bill_id", "question_id", "question", "answer")


def check_bills_dir(bills_dir: Path) -> list[Path]:
    """Raises InputError if the folder is missing or holds no bill images. Returns the images."""
    bills_dir = Path(bills_dir)
    if not bills_dir.is_dir():
        raise InputError(f"Bills folder not found: {bills_dir}")
    images = [p for p in sorted(bills_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise InputError(f"No bill images ({', '.join(sorted(IMAGE_EXTS))}) in {bills_dir}")
    return images


def _read_checked(path: Path, columns: tuple[str, ...]) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise InputError(f"Template not found: {path}")
    try:
        fields, rows = read_template(path)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        raise InputError(f"Could not read template {path}: {e}") from e
    missing = [c for c in columns if c not in fields]
    if missing:
        raise InputError(f"Template {path} is missing column(s): {', '.join(missing)} (needs {', '.join(columns)})")
    if not rows:
        raise InputError(f"Template {path} has no rows")
    if any(not (r.get("bill_id") or "").strip() for r in rows):
        raise InputError(f"Template {path} has a row with an empty bill_id")
    return fields, rows


def load_templates(templates_dir: Path) -> dict:
    """Parse the level1/level2 templates. Raises InputError on a missing or bad template."""
    templates_dir = Path(templates_dir)
    if not templates_dir.is_dir():
        raise InputError(f"Templates folder not found: {templates_dir}")
    fields, rows = _read_checked(templates_dir / "level1.csv", L1_COLUMNS)
    l2_fields, l2_rows = _read_checked(templates_dir / "level2.csv", L2_COLUMNS)
    questions: dict[str, list[tuple[str, str]]] = {}
    for r in l2_rows:
        qs = questions.setdefault(r["bill_id"], [])
        if r["question_id"] not in {q for q, _ in qs}:
            qs.append((r["question_id"], r["question"]))
    # unique, template order (level1 first, then any bill that appears only in level2)
    bill_ids = list(dict.fromkeys([r["bill_id"] for r in rows] + list(questions)))
    return {"fields": fields, "rows": rows, "l2_fields": l2_fields, "l2_rows": l2_rows,
            "questions": questions, "bill_ids": bill_ids}


def missing_images(bills_dir: Path, bill_ids: list[str]) -> list[str]:
    return [b for b in bill_ids if find_image(Path(bills_dir), b) is None]


def answer_flags(bill_id: str, answers: dict[str, str], answer_meta: dict | None) -> dict[str, dict]:
    """{question_id: {"status": "unverified"|"failed", "detail"}} for answers that need a human look.

    unverified = kept although some numbers could not be traced to a calc (from the answer log, written on hits too);
    failed = the "Processing failed: ..." text was written.
    """
    flags: dict[str, dict] = {}
    for q, text in answers.items():
        if text.startswith("Processing failed"):
            flags[q] = {"status": "failed", "detail": text}
    if answer_meta and answer_meta.get("cache") in ("hit", "miss"):  # this run wrote the log
        try:
            entries = json.loads((ANSWER_LOG_DIR / f"answers_{bill_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = {}
        for q, e in entries.items():
            if q in answers and q not in flags and isinstance(e, dict) and e.get("unverified"):
                untraced = [x.split(": ", 1)[-1] for x in e.get("errors") or [] if x.startswith("numbers typed directly")]
                flags[q] = {"status": "unverified", "detail": f"untraced numbers: {', '.join(untraced) or '?'}"}
    return flags


def process_with_answers(bill_id: str, bills_dir: Path, questions: list[tuple[str, str]] | None,
                         refresh: bool = False) -> tuple[ExtractionResult, dict, dict[str, str], dict | None]:
    """Extraction + Level 2 for one bill. Never raises. Returns (result, info, answers, answer_meta)."""
    result, info = process_bill(bill_id, Path(bills_dir), refresh)
    if not questions:
        return result, info, {}, None
    answers, ameta = answer_bill(bill_id, questions, result, info["error"] is None, refresh)
    return result, info, answers, ameta


def run_pipeline(bills_dir: Path, templates_dir: Path, out_dir: Path, refresh: bool = False) -> dict:
    """Process every template bill, write level1.csv and level2.csv.

    Returns {"bill_ids", "outcomes": {bill_id: (result, info, answers, answer_meta)},
    "flags": {bill_id: answer_flags(...)}, "level1_path", "level2_path", "level1_rows", "level2_rows"}.
    Raises InputError on bad input folders/templates (checked before any processing).
    """
    out_dir = Path(out_dir)
    check_bills_dir(bills_dir)
    t = load_templates(templates_dir)
    for b in missing_images(bills_dir, t["bill_ids"]):  # not an error: the bill gets a fallback row
        log.warning("bill=%s has no image in %s; writing a fallback row", b, bills_dir)
    rows, l2_rows, questions = t["rows"], t["l2_rows"], t["questions"]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {b: pool.submit(process_with_answers, b, bills_dir, questions.get(b), refresh) for b in t["bill_ids"]}
        outcomes = {b: f.result() for b, f in futures.items()}

    for r in rows:
        r["json"] = to_level1_json(outcomes[r["bill_id"]][0].level1)
    write_csv(out_dir / "level1.csv", t["fields"], rows)
    l2_out = None
    if l2_rows:
        for r in l2_rows:
            r["answer"] = outcomes[r["bill_id"]][2][r["question_id"]]
        l2_out = out_dir / "level2.csv"
        write_csv(l2_out, t["l2_fields"], l2_rows)
    flags = {b: answer_flags(b, o[2], o[3]) for b, o in outcomes.items()}
    return {"bill_ids": t["bill_ids"], "outcomes": outcomes, "flags": flags, "level1_path": out_dir / "level1.csv",
            "level2_path": l2_out, "level1_rows": len(rows), "level2_rows": len(l2_rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Utility Bill Decoder batch run")
    ap.add_argument("--bills", required=True, type=Path)
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--refresh", action="store_true", help="skip cache reads (still writes successful results)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        load_settings()  # fail fast on a disallowed model
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    try:
        res = run_pipeline(args.bills, args.templates, args.out, args.refresh)
    except InputError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    bill_ids, outcomes = res["bill_ids"], res["outcomes"]

    print(f"\nLEVEL 1\n{'bill_id':<12}{'cache':<6}{'latency':>8}{'in_tok':>8}{'out_tok':>8}  {'reask':<6}{'kept':<9}checks / error")
    for b in bill_ids:
        i = outcomes[b][1]
        lat = "" if i["latency_s"] is None else f"{i['latency_s']}s"
        print(f"{b:<12}{i['cache']:<6}{lat:>8}{str(i['input_tokens'] or 0):>8}{str(i['output_tokens'] or 0):>8}  "
              f"{'y' if i['reasked'] else 'n':<6}{i['kept']:<9}{short_status(i['checks'])}{'  ERROR: ' + i['error'] if i['error'] else ''}")
        if i["reask_reason"]:
            print(f"{'':<12}re-ask: {i['reask_reason']}")
    print("checks: M=Matched X=Mismatch ?=Insufficient Data")
    failed = sum(1 for b in bill_ids if outcomes[b][1]["error"])
    api_calls = sum(1 for b in bill_ids if outcomes[b][1]["cache"] == "miss")
    print(f"Wrote {args.out / 'level1.csv'} ({res['level1_rows']} rows). cache misses={api_calls}, failures={failed}")

    if res["level2_path"]:
        print(f"\nLEVEL 2\n{'bill_id':<12}{'cache':<6}{'latency':>8}{'in_tok':>8}{'out_tok':>8}  {'reask':<6}{'failed':<7}error")
        for b in bill_ids:
            m = outcomes[b][3]
            if m is None:
                continue
            lat = "" if m["latency_s"] is None else f"{m['latency_s']}s"
            print(f"{b:<12}{m['cache']:<6}{lat:>8}{str(m['input_tokens'] or 0):>8}{str(m['output_tokens'] or 0):>8}  "
                  f"{'y' if m['reasked'] else 'n':<6}{m['failed']:<7}{m['error'] or ''}")
        a_calls = sum(1 for b in bill_ids if outcomes[b][3] and outcomes[b][3]["cache"] == "miss")
        print(f"Wrote {args.out / 'level2.csv'} ({res['level2_rows']} rows). answer cache misses={a_calls}. "
              f"Expressions logged in .cache/logs/answers_<bill_id>.json")

    flagged = [(b, q, f) for b in bill_ids for q, f in res["flags"].get(b, {}).items()]
    print("\nUNVERIFIED ANSWERS (unverified numbers or processing failed)")
    if not flagged:
        print("none")
    else:
        print(f"{'bill_id':<12}{'question':<10}{'status':<12}detail")
        for b, q, f in flagged:
            print(f"{b:<12}{q:<10}{f['status']:<12}{f['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
