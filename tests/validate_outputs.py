"""Validate generated CSVs against their templates (guide §7.2) and summarise checks / answer quality flags.

Usage: python tests/validate_outputs.py <templates_dir> <output_dir>
Exit 0 = all format checks pass (check statuses and answer flags are reported, not failed on).
"""
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from answer import LOG_DIR  # noqa: E402
from cache import RESULTS_DIR  # noqa: E402
from checks import run_checks  # noqa: E402
from schema import ExtractionResult, Level1Bill  # noqa: E402

L1_KEYS = list(Level1Bill.model_fields)


def read_utf8(path: Path, errors: list[str]) -> list[dict] | None:
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        errors.append(f"{path}: missing")
        return None
    except UnicodeDecodeError as e:
        errors.append(f"{path}: not UTF-8 ({e})")
        return None
    return list(csv.DictReader(io.StringIO(text.lstrip("﻿"), newline="")))


def empty_cells(name: str, rows: list[dict], errors: list[str]) -> None:
    for i, r in enumerate(rows, start=2):
        for k, v in r.items():
            if v is None or not str(v).strip():
                errors.append(f"{name} line {i}: empty cell in column {k!r}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tdir, odir = Path(sys.argv[1]), Path(sys.argv[2])
    errors: list[str] = []

    t1 = read_utf8(tdir / "level1.csv", errors) or []
    t2 = read_utf8(tdir / "level2.csv", errors) or []
    o1 = read_utf8(odir / "level1.csv", errors)
    o2 = read_utf8(odir / "level2.csv", errors)

    bills: dict[str, Level1Bill] = {}
    if o1 is not None:
        if len(o1) != len(t1):
            errors.append(f"level1: {len(o1)} rows, template has {len(t1)}")
        if [r["bill_id"] for r in o1] != [r["bill_id"] for r in t1]:
            errors.append("level1: bill_id order differs from template")
        if t1 and o1 and list(o1[0]) != list(t1[0]):
            errors.append(f"level1: columns {list(o1[0])} differ from template {list(t1[0])}")
        empty_cells("level1", o1, errors)
        for r in o1:
            b, cell = r.get("bill_id"), r.get("json") or ""
            if "\n" in cell or "\r" in cell:
                errors.append(f"level1 {b}: json cell spans multiple lines")
            try:
                obj = json.loads(cell)
            except json.JSONDecodeError as e:
                errors.append(f"level1 {b}: json does not parse ({e})")
                continue
            missing = [k for k in L1_KEYS if k not in obj]
            if missing:
                errors.append(f"level1 {b}: missing keys {missing}")
            try:
                bills[b] = Level1Bill.model_validate(obj)
            except Exception as e:  # noqa: BLE001
                errors.append(f"level1 {b}: schema validation failed: {e}")
                continue

    if o2 is not None:
        if len(o2) != len(t2):
            errors.append(f"level2: {len(o2)} rows, template has {len(t2)}")
        if t2 and o2 and list(o2[0]) != list(t2[0]):
            errors.append(f"level2: columns {list(o2[0])} differ from template {list(t2[0])}")
        for i, (o, t) in enumerate(zip(o2, t2), start=2):
            for k in ("bill_id", "question_id", "question"):
                if o.get(k) != t.get(k):
                    errors.append(f"level2 line {i}: {k} {o.get(k)!r} != template {t.get(k)!r}")
        empty_cells("level2", o2, errors)

    # ---- report: per-bill checks (with internal facts from the results cache when it matches the output) ----
    print(f"{'bill_id':<12}{'bill_month':<11}{'extract_reask':<14}checks")
    reasked_bills = 0
    for b, bill in bills.items():
        internal, reask = None, "-"
        rp = RESULTS_DIR / f"{b}.json"
        if rp.exists():
            entry = json.loads(rp.read_text(encoding="utf-8"))
            res = ExtractionResult.model_validate(entry["result"])
            if res.level1 == bill:  # cache entry is the one that produced this output
                internal = res.internal
                reask = entry["meta"].get("kept") if entry["meta"].get("reasked") else "n"
                reasked_bills += bool(entry["meta"].get("reasked"))
        checks = run_checks(bill, internal)
        status = " ".join(f"{c['name'].split('_')[0]}:{c['status']}" for c in checks)
        print(f"{b:<12}{str(bill.bill_month):<11}{reask:<14}{status}")
        for c in checks:
            if c["status"] != "Matched":
                print(f"{'':<12}  {c['name']}: {c['detail']}")

    # ---- report: answer flags ----
    failed = [(r["bill_id"], r["question_id"]) for r in (o2 or []) if r["answer"].startswith("Processing failed")]
    unverified, answer_reasks = [], 0
    for b in dict.fromkeys(r["bill_id"] for r in (o2 or [])):
        lp = LOG_DIR / f"answers_{b}.json"
        if not lp.exists():
            continue
        for q, e in json.loads(lp.read_text(encoding="utf-8")).items():
            if isinstance(e, dict):
                answer_reasks += "first_pass" in e
                if e.get("unverified"):
                    unverified.append((b, q))
    print(f"\nextraction re-asks: {reasked_bills} bills | answer re-asks: {answer_reasks} questions | "
          f"unverified: {len(unverified)} {unverified or ''} | processing failed: {len(failed)} {failed or ''}")

    if errors:
        print(f"\nFAIL ({len(errors)} problems)")
        for e in errors:
            print(" -", e)
        return 1
    print(f"\nVALIDATE OK: level1 {len(o1 or [])} rows, level2 {len(o2 or [])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
