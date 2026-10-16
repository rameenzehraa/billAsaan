"""Milestone 2 check. Makes 2 API calls: KESC_0008 and LESCO_0006 (once each).

Run: python tests/check_extract.py
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import provider_for  # noqa: E402
from extract import extract_bill  # noqa: E402
from schema import Level1Bill, to_level1_json  # noqa: E402

BILLS = ROOT / "data" / "train" / "bills"
LEVEL1_KEYS = list(Level1Bill.model_fields)


def diff(path: str, got, exp, out: list[str]) -> None:
    """Strings/dates exact, numbers within ±1, lists in order."""
    if isinstance(exp, dict):
        for k in exp:
            diff(f"{path}.{k}" if path else k, (got or {}).get(k) if isinstance(got, dict) else None, exp[k], out)
    elif isinstance(exp, list):
        got = got if isinstance(got, list) else []
        if len(got) != len(exp):
            out.append(f"{path}: length {len(got)} != expected {len(exp)}  got={got}")
        for i, e in enumerate(exp):
            diff(f"{path}[{i}]", got[i] if i < len(got) else None, e, out)
    elif isinstance(exp, (int, float)) and not isinstance(exp, bool):
        if not isinstance(got, (int, float)) or isinstance(got, bool) or abs(got - exp) > 1:
            out.append(f"{path}: got {got!r}, expected {exp!r}")
    elif got != exp:
        out.append(f"{path}: got {got!r}, expected {exp!r}")


def run(bill_id: str):
    path = next(BILLS.glob(f"{bill_id}.*"))
    result, meta = extract_bill(bill_id, provider_for(bill_id), path)
    print(f"[{bill_id}] model={meta['model']} latency={meta['latency_s']}s in_tokens={meta['input_tokens']} "
          f"out_tokens={meta['output_tokens']} attempts={meta['attempts']} error={meta['error']}")
    return result, meta


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ok = True

    result, meta = run("KESC_0008")
    expected = json.loads((ROOT / "data" / "mock" / "kesc_0008_expected.json").read_text(encoding="utf-8"))
    got = json.loads(to_level1_json(result.level1))
    mismatches: list[str] = []
    diff("", got, expected, mismatches)
    print(f"KESC_0008 mismatches: {len(mismatches)}")
    for m in mismatches:
        print("  -", m)
    print(f"KESC_0008 internal: history={len(result.internal.usage_history)} rates={len(result.internal.rates)} "
          f"split={len(result.internal.split)} late={len(result.internal.late_payment_amounts)} "
          f"flags={result.internal.validation_flags}")
    ok &= not mismatches and meta["error"] is None

    result, meta = run("LESCO_0006")
    line = to_level1_json(result.level1)
    keys_ok = list(json.loads(line)) == LEVEL1_KEYS
    print(f"LESCO_0006 schema-valid={meta['error'] is None} all_keys={keys_ok}")
    print(f"LESCO_0006 level1: {line}")
    ok &= keys_ok and meta["error"] is None

    print("EXTRACT OK" if ok else "EXTRACT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
