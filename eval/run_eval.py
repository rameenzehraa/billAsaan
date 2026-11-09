"""BillAsaan accuracy eval: Level 1 extraction fields + chat number accuracy.

  python eval/run_eval.py --draft   # write UNVERIFIED draft labels from the current extraction
  python eval/run_eval.py           # score verified labels, write eval/RESULTS.md

Bills (eval/bills/) and labels (eval/labels/) are private and gitignored. Extractions use the run.py disk cache and
chat answers use .cache/eval_chat/, so reruns make no API calls. RESULTS.md contains aggregates only, never bill values.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calc import compute_facts  # noqa: E402
from config import load_settings  # noqa: E402
from schema import ExtractionResult, Level1Bill  # noqa: E402

EVAL = ROOT / "eval"
BILLS, LABELS, RESULTS = EVAL / "bills", EVAL / "labels", EVAL / "RESULTS.md"
CHAT_CACHE = ROOT / ".cache" / "eval_chat"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# The one hand-verified label shipped with the competition data; its image may live in data/train/bills.
SEED_LABELS = {"KESC_0008": ROOT / "data" / "mock" / "kesc_0008_expected.json"}
EXTRA_IMAGE_DIRS = [ROOT / "data" / "train" / "bills"]

TEXT_FIELDS = ["provider", "tariff", "bill_month", "reading_date", "issue_date", "due_date"]
NUMBER_FIELDS = ["sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed", "total_charges",
                 "total_taxes", "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date"]
FIELDS = TEXT_FIELDS + NUMBER_FIELDS
TOLERANCE = 1.0
FREE_TIER_PAUSE_S = 5  # between uncached API calls: stays under ~15 requests/minute

# Fixed chat questions, each checked against one calc fact (first available name wins).
CHAT_QUESTIONS = [
    ("How much do I have to pay by the due date?", ["payable_within_due_date"]),
    ("How much more will I pay if I pay after the due date?", ["late_surcharge"]),
    ("How many units did I use this month?", ["units_consumed"]),
    ("What is my average monthly usage over the previous months?", ["prev12_avg_units"]),
    ("How much tax is on this bill?", ["total_taxes", "taxes_sum"]),
]

WARNING = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  DRAFT LABELS ARE UNVERIFIED. They are copies of the model's own extraction, so      !!
!!  scoring them would measure nothing. Open each eval/labels/<bill_id>.json, correct    !!
!!  EVERY field and every chat "expected" value against the bill image, then set         !!
!!  "verified": true. Unverified labels are skipped by the scorer.                       !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


# ---------- inputs ----------

def find_image(bill_id: str, bills_dir: Path) -> Path | None:
    for d in [bills_dir, *EXTRA_IMAGE_DIRS]:
        for ext in IMAGE_EXTS:
            if (p := d / f"{bill_id}{ext}").exists():
                return p
    return None


def list_bills(bills_dir: Path) -> list[str]:
    return sorted(p.stem for p in bills_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS) if bills_dir.is_dir() else []


def extract(bill_id: str, image: Path) -> tuple[ExtractionResult | None, str]:
    """Cached extraction via run.process_bill (disk cache keyed on image+model+settings). Returns (result, cache)."""
    from run import process_bill
    result, info = process_bill(bill_id, image.parent, refresh=False)
    if info["error"]:
        print(f"  {bill_id}: extraction failed ({info['error'][:80]})")
        return None, info["cache"]
    return result, info["cache"]


def chat_expectations(level1: dict, internal: dict | None = None) -> list[dict]:
    result = ExtractionResult.model_validate({"level1": level1, "internal": internal or {}})
    facts = compute_facts(result)
    out = []
    for question, names in CHAT_QUESTIONS:
        name = next((n for n in names if n in facts), None)
        out.append({"question": question, "fact": name or names[0],
                    "expected": facts[name]["value"] if name else None})
    return out


def seed_labels(labels_dir: Path) -> None:
    for bill_id, path in SEED_LABELS.items():
        target = labels_dir / f"{bill_id}.json"
        if target.exists() or not path.exists():
            continue
        level1 = Level1Bill.model_validate(json.loads(path.read_text(encoding="utf-8"))).model_dump(mode="json")
        write_json(target, {"bill_id": bill_id, "verified": True, "source": "data/mock/kesc_0008_expected.json "
                            "(hand-verified during WTQ 2026)", "level1": level1, "chat": chat_expectations(level1)})
        print(f"  seeded verified label {_rel(target)}")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------- draft ----------

def draft(bills_dir: Path, labels_dir: Path) -> int:
    print(WARNING)
    bills_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    seed_labels(labels_dir)
    ids = list_bills(bills_dir)
    if not ids:
        print(f"No bill images in {bills_dir}. Add .png/.jpg bills there (they stay gitignored) and rerun.")
    written = 0
    for bill_id in ids:
        target = labels_dir / f"{bill_id}.json"
        if target.exists():
            print(f"  {bill_id}: label exists, not overwritten")
            continue
        result, cache = extract(bill_id, find_image(bill_id, bills_dir))
        if cache == "miss":
            time.sleep(FREE_TIER_PAUSE_S)
        if result is None:
            continue
        level1 = result.level1.model_dump(mode="json")
        write_json(target, {"bill_id": bill_id, "verified": False,
                            "source": "DRAFT from model extraction - verify every field against the image",
                            "level1": level1,
                            "chat": chat_expectations(level1, result.internal.model_dump(mode="json"))})
        written += 1
        print(f"  {bill_id}: draft written (extraction cache {cache})")
    print(f"\n{written} draft label(s) written to {labels_dir}. NONE ARE VERIFIED YET." if written else "")
    print(WARNING if written else "")
    return 0


# ---------- scoring ----------

def compare_field(field: str, truth, pred) -> str:
    """'correct' | 'invented' (label null, value given) | 'missed' (label value, null given) | 'wrong'."""
    if truth is None and pred is None:
        return "correct"
    if truth is None:
        return "invented"
    if pred is None:
        return "missed"
    if field in NUMBER_FIELDS:
        try:
            return "correct" if abs(float(truth) - float(pred)) <= TOLERANCE else "wrong"
        except (TypeError, ValueError):
            return "wrong"
    return "correct" if str(truth).strip() == str(pred).strip() else "wrong"


def match_items(truth: list[dict], pred: list[dict]) -> tuple[int, int, int]:
    """Pair items by type in printed order. Returns (correct pairs, predicted items, labelled items)."""
    correct = 0
    for kind in {i["type"] for i in truth} | {i["type"] for i in pred}:
        t = [i["amount"] for i in truth if i["type"] == kind]
        p = [i["amount"] for i in pred if i["type"] == kind]
        correct += sum(1 for a, b in zip(t, p) if abs(a - b) <= TOLERANCE)
    return correct, len(pred), len(truth)


def number_matches(expected: float, answer: str) -> bool:
    import chat
    for raw in chat._NUM.findall(answer):
        try:
            n = float(raw.replace(",", ""))
        except ValueError:
            continue
        if any(abs(abs(n) - abs(v)) <= 0.01 for v in (expected, round(expected), round(expected, 1), round(expected, 2))):
            return True
    return False


def cached_chat(result: ExtractionResult, question: str, model: str) -> tuple[dict, bool]:
    import chat
    context = chat.build_context(result, [], question)
    key = hashlib.sha256(json.dumps([model, chat.SYSTEM_PROMPT, context], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()
    path = CHAT_CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    reply = chat.ask(result, [], question)
    if reply["kind"] == "answer":  # never cache failures
        write_json(path, {"kind": reply["kind"], "content": reply["content"]})
    return reply, False


def score(labels_dir: Path, bills_dir: Path, out: Path, with_chat: bool) -> int:
    labels = sorted(labels_dir.glob("*.json")) if labels_dir.is_dir() else []
    verified, skipped = [], []
    for path in labels:
        data = json.loads(path.read_text(encoding="utf-8"))
        (verified if data.get("verified") is True else skipped).append(data)
    if skipped:
        print(f"WARNING: skipping {len(skipped)} UNVERIFIED label(s): {', '.join(d['bill_id'] for d in skipped)}")
    if not verified:
        print("No verified labels. Run --draft, verify the labels, set \"verified\": true, then rerun.")
        return 1

    model = load_settings().model_name
    fields = {f: {"correct": 0, "invented": 0, "missed": 0, "wrong": 0} for f in FIELDS}
    items = {"charges": [0, 0, 0], "taxes": [0, 0, 0]}
    chat_stats = {"correct": 0, "scored": 0, "na": 0, "failed": 0}
    n, api_calls = 0, 0
    for label in verified:
        bill_id = label["bill_id"]
        image = find_image(bill_id, bills_dir)
        if image is None:
            print(f"  {bill_id}: no image found, skipped")
            continue
        result, cache = extract(bill_id, image)
        api_calls += cache == "miss"
        if result is None:
            continue
        n += 1
        pred, truth = result.level1.model_dump(mode="json"), label["level1"]
        for f in FIELDS:
            fields[f][compare_field(f, truth.get(f), pred.get(f))] += 1
        for kind in items:
            for i, v in enumerate(match_items(truth.get(kind) or [], pred.get(kind) or [])):
                items[kind][i] += v
        if not with_chat:
            continue
        for q in label.get("chat", []):
            if q.get("expected") is None:
                chat_stats["na"] += 1
                continue
            reply, hit = cached_chat(result, q["question"], model)
            if not hit:
                api_calls += 1
                time.sleep(FREE_TIER_PAUSE_S)
            if reply["kind"] != "answer":
                chat_stats["failed"] += 1
                continue
            chat_stats["scored"] += 1
            chat_stats["correct"] += number_matches(float(q["expected"]), reply["content"])
        print(f"  {bill_id}: scored (extraction cache {cache})")

    write_results(out, n, model, fields, items, chat_stats if with_chat else None, len(skipped),
                  {lbl["bill_id"] for lbl in verified if lbl["bill_id"] in SEED_LABELS})
    print(f"\nWrote {_rel(out)} (n={n} bills, {api_calls} API call(s) this run).")
    return 0


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "n/a"


def write_results(out: Path, n: int, model: str, fields: dict, items: dict, chat_stats: dict | None,
                  unverified: int, seen_dev: set[str] = frozenset()) -> None:
    total = {k: sum(f[k] for f in fields.values()) for k in ("correct", "invented", "missed", "wrong")}
    cells = n * len(FIELDS)
    lines = [
        "# BillAsaan eval results", "",
        f"- **Date:** {date.today().isoformat()}",
        f"- **Model:** `{model}` (extraction and chat)",
        f"- **Bills scored (n):** {n} verified label(s)" + (f"; {unverified} unverified skipped" if unverified else ""),
        "- **Scoring:** text and dates exact; numbers within ±1. *Invented* = value given where the bill shows none; "
        "*missed* = null where the bill shows a value. Generated by `eval/run_eval.py`; no bill values are included.",
        "", f"**Overall Level 1 field accuracy: {pct(total['correct'], cells)}** "
        f"({total['correct']}/{cells} field values) · invented {total['invented']} · missed {total['missed']} · "
        f"wrong value {total['wrong']}", "",
        "## Per-field accuracy", "", "| Field | Accuracy | Invented | Missed | Wrong |", "|---|---:|---:|---:|---:|",
    ]
    for f, s in fields.items():
        lines.append(f"| `{f}` | {pct(s['correct'], n)} | {s['invented']} | {s['missed']} | {s['wrong']} |")
    lines += ["", "## Charge and tax lines (matched by type, amount within ±1)", "",
              "| List | Correct | Precision | Recall |", "|---|---:|---:|---:|"]
    for kind, (c, p, t) in items.items():
        lines.append(f"| {kind} | {c} | {pct(c, p)} | {pct(c, t)} |")
    if chat_stats is not None:
        lines += ["", "## Chat number accuracy", "",
                  f"{len(CHAT_QUESTIONS)} fixed questions per bill; an answer is correct when it states the value "
                  "that `calc.py` computes from the verified label.", "",
                  f"- **Correct:** {chat_stats['correct']}/{chat_stats['scored']} ({pct(chat_stats['correct'], chat_stats['scored'])})",
                  f"- **Not applicable** (no labelled value for that fact): {chat_stats['na']}",
                  f"- **Failed calls:** {chat_stats['failed']}"]
    if n < 10:
        lines += ["", f"> Small sample (n={n}). Treat these numbers as indicative until more verified labels are added."]
    if seen_dev:
        lines += ["", f"> Not held out: {', '.join(sorted(seen_dev))} was used while developing the extraction prompt "
                      "during WTQ 2026, so its score is optimistic. Held-out accuracy needs bills the prompt never saw."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draft", action="store_true", help="write unverified draft labels for bills without one")
    ap.add_argument("--bills", type=Path, default=BILLS, help="bill images folder (default eval/bills)")
    ap.add_argument("--labels", type=Path, default=LABELS, help="labels folder (default eval/labels)")
    ap.add_argument("--out", type=Path, default=RESULTS, help="results file (default eval/RESULTS.md)")
    ap.add_argument("--no-chat", action="store_true", help="score extraction only (no chat calls)")
    args = ap.parse_args(argv)
    if args.draft:
        return draft(args.bills, args.labels)
    return score(args.labels, args.bills, args.out, with_chat=not args.no_chat)


if __name__ == "__main__":
    sys.exit(main())
