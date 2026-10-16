"""Milestone 1 smoke test. Run: python tests/smoke_scaffold.py (no API calls)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_settings, provider_for, validate_model_name  # noqa: E402
from images import list_bills, load_image  # noqa: E402
from schema import Level1Bill, empty_level1, to_level1_json  # noqa: E402

LEVEL1_KEYS = list(Level1Bill.model_fields)


def same(a, b) -> bool:
    """Deep equality that also requires matching numeric types (3 vs 3.0)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def main() -> None:
    # 1. Images
    bills = list_bills(ROOT / "data" / "train" / "bills")
    check(len(bills) == 5, f"expected 5 train bills, got {len(bills)}")
    print(f"{'bill_id':<12}{'provider':<10}{'original':<12}{'sent':<12}{'mime':<11}bytes")
    for bill_id, path in bills:
        data, mime, orig, sent = load_image(path)
        provider = provider_for(bill_id)
        check(provider is not None, f"no provider for {bill_id}")
        if bill_id.startswith("KESC"):
            check(provider == "KE", f"{bill_id} should map to KE")
        check(max(sent) <= 2000, f"{bill_id} sent size {sent} exceeds limit")
        if max(orig) <= 2000:
            check(data == path.read_bytes(), f"{bill_id} bytes changed without downscale")
        print(f"{bill_id:<12}{provider:<10}{'x'.join(map(str, orig)):<12}{'x'.join(map(str, sent)):<12}{mime:<11}{len(data)}")

    # 2. Expected KESC_0008 JSON validates and round-trips unchanged
    expected = json.loads((ROOT / "data" / "mock" / "kesc_0008_expected.json").read_text(encoding="utf-8"))
    check(list(expected) == LEVEL1_KEYS, "expected JSON keys/order differ from schema")
    bill = Level1Bill.model_validate(expected)
    line = to_level1_json(bill)
    check("\n" not in line, "to_level1_json output is not one line")
    check(same(json.loads(line), expected), "round-trip changed keys, order or values")
    print("KESC_0008 expected JSON: valid, round-trip OK")

    # 3. Fallback is fully keyed
    empty = json.loads(to_level1_json(empty_level1("KESC_0002", None)))
    check(list(empty) == LEVEL1_KEYS, "empty_level1 missing keys")
    check(empty["provider"] == "KE" and empty["charges"] == [] and empty["taxes"] == [], "empty_level1 values wrong")
    check(all(empty[k] is None for k in LEVEL1_KEYS if k not in ("provider", "charges", "taxes")), "non-null scalar")
    print("empty_level1: all keys present")

    # 4. Invalid model name raises
    try:
        validate_model_name("gpt-4o")
        check(False, "validate_model_name accepted a disallowed model")
    except ValueError:
        pass
    saved = os.environ.get("MODEL_NAME")
    os.environ["MODEL_NAME"] = "gemini-2.5-pro"
    try:
        load_settings()
        check(False, "load_settings accepted a disallowed MODEL_NAME")
    except ValueError:
        pass
    finally:
        if saved is None:
            os.environ.pop("MODEL_NAME")
        else:
            os.environ["MODEL_NAME"] = saved
    print("invalid MODEL_NAME: raises")

    print("SCAFFOLD OK")


if __name__ == "__main__":
    main()
