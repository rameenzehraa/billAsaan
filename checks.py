"""Local consistency checks on extracted Level 1 fields.

Checks only compare. They never modify, round or recalculate stored values.
Status: Matched / Mismatch / Insufficient Data. A missing component is never a Mismatch.
"""
from datetime import datetime

from schema import InternalFacts, Level1Bill

MATCHED, MISMATCH, INSUFFICIENT = "Matched", "Mismatch", "Insufficient Data"

# Bills round the payable amount to whole rupees (KESC_0008: 3430.24 - 0.53 = 3429.71 printed as 3430),
# and separately rounded line items can add up to about a rupee or more, so allow ±2 PKR.
TOLERANCE = 2.0

MULTIPLIER_HINTS = ("multiplier", "mf", "adjust", "net meter", "net-meter")


def _r(name: str, status: str, detail: str, fields: list[str]) -> dict:
    return {"name": name, "status": status, "detail": detail, "fields": fields}


def check_date_formats(b: Level1Bill) -> dict:
    fields = ["bill_month", "reading_date", "issue_date", "due_date"]
    present = {f: getattr(b, f) for f in fields if getattr(b, f) is not None}
    if not present:
        return _r("date_formats", INSUFFICIENT, "no dates extracted", fields)
    bad = []
    for f, v in present.items():
        fmt = "%Y-%m" if f == "bill_month" else "%Y-%m-%d"
        try:
            datetime.strptime(v, fmt)
            if len(v) != (7 if f == "bill_month" else 10):
                bad.append(f)
        except ValueError:
            bad.append(f)
    if bad:
        return _r("date_formats", MISMATCH, f"invalid format: {', '.join(bad)}", bad)
    return _r("date_formats", MATCHED, f"{len(present)} valid", list(present))


def check_reading_diff(b: Level1Bill, internal: InternalFacts | None = None) -> dict:
    fields = ["previous_reading", "current_reading", "units_consumed"]
    prev, cur, units = b.previous_reading, b.current_reading, b.units_consumed
    if prev is None or cur is None or units is None:
        return _r("reading_diff", INSUFFICIENT, "a reading or units value is missing", fields)
    flags = " ".join(f for f in internal.validation_flags if not f.startswith("reask_rejected")).lower() if internal else ""
    if any(h in flags for h in MULTIPLIER_HINTS):
        return _r("reading_diff", INSUFFICIENT, "multiplier/adjustment noted on bill", fields)
    diff = cur - prev
    if diff < 0:
        return _r("reading_diff", INSUFFICIENT, "current < previous (meter rollover/replacement?)", fields)
    if diff == units:
        return _r("reading_diff", MATCHED, f"{cur} - {prev} = {units}", fields)
    # Rounding tolerance for decimal readings: bills print fractional kWh readings but whole-unit consumption.
    if (prev % 1 or cur % 1) and abs(diff - units) <= 1:
        return _r("reading_diff", MATCHED, f"{cur} - {prev} = {round(diff, 2)} ~= {units} (decimal readings, units rounded)", fields)
    if diff > 0 and units % diff == 0 and units != 0:
        return _r("reading_diff", INSUFFICIENT, f"units {units} = {units // diff} x diff {diff} (possible multiplier)", fields)
    return _r("reading_diff", MISMATCH, f"{cur} - {prev} = {diff}, but units_consumed = {units}", fields)


def check_charges_taxes_vs_current_bill(b: Level1Bill) -> dict:
    fields = ["total_charges", "total_taxes", "current_bill"]
    charges = b.total_charges if b.total_charges is not None else (sum(c.amount for c in b.charges) if b.charges else None)
    taxes = b.total_taxes if b.total_taxes is not None else (sum(t.amount for t in b.taxes) if b.taxes else None)
    if charges is None or taxes is None or b.current_bill is None:
        return _r("charges_taxes_vs_current_bill", INSUFFICIENT, "charges, taxes or current_bill missing", fields)
    gap = round(charges + taxes - b.current_bill, 2)
    status = MATCHED if abs(gap) <= TOLERANCE else MISMATCH
    return _r("charges_taxes_vs_current_bill", status, f"{charges} + {taxes} vs {b.current_bill} (gap {gap})", fields)


def check_current_plus_arrears_vs_payable(b: Level1Bill) -> dict:
    fields = ["current_bill", "arrears", "payable_within_due_date"]
    if b.current_bill is None or b.arrears is None or b.payable_within_due_date is None:
        return _r("current_plus_arrears_vs_payable", INSUFFICIENT, "current_bill, arrears or payable missing", fields)
    name, payable = "current_plus_arrears_vs_payable", b.payable_within_due_date
    base = b.current_bill + b.arrears
    gap = round(base - payable, 2)
    detail = f"{b.current_bill} + {b.arrears} vs {payable} (gap {gap})"
    if abs(gap) <= TOLERANCE:
        return _r(name, MATCHED, detail, fields)
    # Zero payable is explained only by an explicitly extracted credit (negative arrears) that covers the bill.
    if payable == 0 and b.arrears < 0 and base <= TOLERANCE:
        return _r(name, MATCHED, f"{detail}; credit arrears cover the bill, nothing payable", fields)
    if b.charges and b.total_charges is not None:
        extra = round(sum(c.amount for c in b.charges) - b.total_charges, 2)  # X = charges_sum - total_charges
        # Evidence that X lies outside current_bill: the printed subtotals already make up current_bill
        # (total_charges + total_taxes ≈ current_bill), and adding X reconciles the payable.
        if (extra > 0 and b.total_taxes is not None
                and abs(b.total_charges + b.total_taxes - b.current_bill) <= TOLERANCE
                and abs(base + extra - payable) <= TOLERANCE):
            return _r(name, MATCHED, f"{detail}; total_charges {b.total_charges} + total_taxes {b.total_taxes} ~= current_bill, "
                                     f"so the {extra} of charge lines beyond the subtotal sit outside current_bill; "
                                     f"{b.current_bill} + {b.arrears} + {extra} ~= {payable}", fields)
        # Otherwise no extracted fact says whether those lines are inside current_bill: never force Matched or Mismatch.
        if abs(extra) > TOLERANCE:
            return _r(name, INSUFFICIENT, f"{detail}; {extra} of charge lines outside the subtotal, "
                                          "inclusion in current_bill not established", fields)
    return _r(name, MISMATCH, detail, fields)


def check_late_ge_within(b: Level1Bill) -> dict:
    fields = ["payable_within_due_date", "payable_after_due_date"]
    if b.payable_within_due_date is None or b.payable_after_due_date is None:
        return _r("late_ge_within", INSUFFICIENT, "a payable amount is missing", fields)
    ok = b.payable_after_due_date >= b.payable_within_due_date
    return _r("late_ge_within", MATCHED if ok else MISMATCH,
              f"after {b.payable_after_due_date} vs within {b.payable_within_due_date}", fields)


def run_checks(b: Level1Bill, internal: InternalFacts | None = None) -> list[dict]:
    return [
        check_date_formats(b),
        check_reading_diff(b, internal),
        check_charges_taxes_vs_current_bill(b),
        check_current_plus_arrears_vs_payable(b),
        check_late_ge_within(b),
    ]


def count_mismatches(checks: list[dict]) -> int:
    return sum(c["status"] == MISMATCH for c in checks)
