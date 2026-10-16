"""Code-computed facts for Level 2 answers, plus a whitelisted arithmetic evaluator.

Every fact is {"value": number, "method": str}. A fact whose inputs are missing is absent, never 0.
"""
import ast
import operator
import re
from datetime import date, datetime

from schema import ExtractionResult

THRESHOLDS = (100, 200, 300, 400, 500)
NET_METER_RE = re.compile(r"net[\s-]?meter|export|solar|green\s*meter", re.I)
MONTH_FORMATS = ("%Y-%m", "%b %y", "%b-%y", "%b%y", "%b %Y", "%b-%Y", "%B %Y", "%m/%y", "%m-%y", "%Y/%m")


def _month(s: str | None) -> str | None:
    """Normalise a history month label to YYYY-MM, or None if unparseable."""
    if not s:
        return None
    s = s.strip()
    for fmt in MONTH_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return None


def _shift(ym: str, months: int) -> str:
    y, m = map(int, ym.split("-"))
    total = y * 12 + (m - 1) + months
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _date(s: str | None) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date() if s else None
    except ValueError:
        return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "item"


def _r(v: float) -> float:
    return round(v, 4)


def compute_facts(result: ExtractionResult) -> dict:
    b, internal = result.level1, result.internal
    facts: dict[str, dict] = {}

    def add(name: str, value, method: str) -> None:
        if value is not None and not isinstance(value, bool):
            facts[name] = {"value": value, "method": method}

    for f in ("units_consumed", "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date",
              "total_charges", "total_taxes", "sanctioned_load_kw", "previous_reading", "current_reading"):
        add(f, getattr(b, f), f"printed on bill ({f})")

    # Line items, numbered per type in printed order
    for kind, items in (("charge", b.charges), ("tax", b.taxes)):
        seen: dict[str, int] = {}
        for it in items:
            seen[it.type] = seen.get(it.type, 0) + 1
            add(f"{kind}_{it.type}_{seen[it.type]}", it.amount, f"printed {kind} line #{seen[it.type]} of type {it.type}")
    if b.charges:
        add("charges_sum", _r(sum(c.amount for c in b.charges)), "sum of all printed charge lines")
    if b.taxes:
        add("taxes_sum", _r(sum(t.amount for t in b.taxes)), "sum of all printed tax lines")

    # History: current month separate; previous months strictly before bill_month
    rows = {}
    for h in internal.usage_history:
        ym = _month(h.month)
        if ym:
            rows[ym] = h
    cur = b.bill_month
    if cur and cur in rows:
        add("current_month_hist_units", rows[cur].units, f"history table row for current month {cur}")
    prev = sorted((ym for ym in rows if cur is None or ym < cur), reverse=True)[:12]
    for ym in prev:
        h, key = rows[ym], ym.replace("-", "_")
        add(f"hist_{key}_units", h.units, f"history table units for {ym}")
        add(f"hist_{key}_bill", h.bill_amount, f"history table bill amount for {ym}")
        add(f"hist_{key}_payment", h.payment, f"history table payment for {ym}")

    # prev12_*: only months inside the 12 calendar months before bill_month that print units.
    # Missing months are skipped (never counted as 0); every method states the available count.
    window = [ym for ym in prev if cur and ym >= _shift(cur, -12)]
    units = [(ym, rows[ym].units) for ym in window if rows[ym].units is not None]
    if cur and units:
        n = len(units)
        avail = f"{n} available previous month{'s' if n != 1 else ''}"
        months_txt = f"{units[-1][0]} to {units[0][0]}, excluding current month {cur}"
        vals = [u for _, u in units]
        add("prev12_count", n, f"{n} of the 12 previous months have units on the bill ({months_txt})")
        add("prev12_total_units", _r(sum(vals)), f"sum of units over {avail} ({months_txt})")
        add("prev12_avg_units", _r(sum(vals) / n), f"average of {avail} ({months_txt})")
        mn = min(units, key=lambda x: x[1]); mx = max(units, key=lambda x: x[1])
        add("prev12_min_units", mn[1], f"lowest of {avail}: {mn[1]} units in {mn[0]}")
        add("prev12_max_units", mx[1], f"highest of {avail}: {mx[1]} units in {mx[0]}")
        for t in THRESHOLDS:
            above = [ym for ym, u in units if u > t]
            add(f"prev12_months_above_{t}", len(above),
                f"of {avail}, months with units > {t}: {', '.join(f'{ym} ({dict(units)[ym]})' for ym in above) or 'none'}")
        bills = [rows[ym].bill_amount for ym in window if rows[ym].bill_amount is not None]
        if bills:
            nb = len(bills)
            add("prev12_avg_bill", _r(sum(bills) / nb),
                f"average history bill amount of {nb} available previous month{'s' if nb != 1 else ''}")

    if cur and b.units_consumed is not None:
        for label, ym in (("last_month", _shift(cur, -1)), ("last_year", _shift(cur, -12))):
            h = rows.get(ym)
            if h is None or h.units is None:
                continue
            base_name = "last_month_units" if label == "last_month" else "same_month_last_year_units"
            add(base_name, h.units, f"history table units for {ym}")
            diff = _r(b.units_consumed - h.units)
            add(f"change_vs_{label}_units", diff, f"units_consumed {b.units_consumed} - {h.units} ({ym})")
            if h.units:
                add(f"change_vs_{label}_pct", _r(diff / h.units * 100), f"({b.units_consumed} - {h.units}) / {h.units} * 100")

    # This month vs the previous-window average (absolute difference; the method says higher/lower)
    avg = facts.get("prev12_avg_units", {}).get("value")
    n_avg = facts.get("prev12_count", {}).get("value")
    if avg and b.units_consumed is not None:
        d = _r(b.units_consumed - avg)
        word = "higher than" if d > 0 else "lower than" if d < 0 else "equal to"
        add("this_month_vs_avg_units", abs(d), f"current month units_consumed {b.units_consumed} is {abs(d)} units {word} the "
            f"average of {n_avg} previous months ({avg}); the average excludes the current month")
        add("this_month_vs_avg_pct", _r(abs(d) / avg * 100),
            f"|units_consumed - prev12_avg_units| / prev12_avg_units * 100; this month is {word} the average")

    # Seasons from the previous-12-month window (current month excluded)
    win = dict(units)
    season_avg: dict[str, tuple[float, int]] = {}
    for name, months, label in (("summer", (6, 7, 8), "June-August"), ("winter", (12, 1, 2), "December-February")):
        sel = sorted(ym for ym in win if int(ym[5:]) in months)
        if not sel:
            continue
        vals = [win[ym] for ym in sel]
        txt = (f"{len(sel)} {name} month{'s' if len(sel) != 1 else ''} ({name} = {label}) in the previous-12-month history, "
               f"current month excluded: {', '.join(f'{ym} ({win[ym]})' for ym in sel)}")
        add(f"{name}_count", len(sel), txt)
        add(f"{name}_total_units", _r(sum(vals)), f"sum of units over {txt}")
        add(f"{name}_avg_units", _r(sum(vals) / len(sel)), f"average units per month over {txt}")
        season_avg[name] = (sum(vals) / len(sel), len(sel))
    if len(season_avg) == 2:
        d = _r(season_avg["summer"][0] - season_avg["winter"][0])
        who = "summer higher" if d > 0 else "winter higher" if d < 0 else "equal"
        add("summer_minus_winter_avg_units", abs(d),
            f"|summer average - winter average| per month ({who}); summer = June-August ({season_avg['summer'][1]} months), "
            f"winter = December-February ({season_avg['winter'][1]} months)")

    # Payment history: only what the history records (absent if the bill prints no payments)
    if any(rows[ym].payment is not None for ym in window):
        zero = [ym for ym in window if rows[ym].payment in (None, 0) and (rows[ym].bill_amount is None or rows[ym].bill_amount > 0)]
        less = [ym for ym in window if rows[ym].payment not in (None, 0) and rows[ym].bill_amount is not None
                and rows[ym].payment < rows[ym].bill_amount]
        note = "as recorded in the history table (reporting timing can differ; not proof of non-payment)"
        add("hist_payment_zero_or_missing_count", len(zero), f"of {len(window)} previous history months, months with payment 0 or "
            f"blank: {', '.join(f'{ym} (bill {rows[ym].bill_amount})' for ym in zero) or 'none'}; {note}")
        add("hist_paid_less_than_billed_count", len(less), f"of {len(window)} previous history months, months with payment below "
            f"the bill amount: {', '.join(f'{ym} (paid {rows[ym].payment} of {rows[ym].bill_amount})' for ym in less) or 'none'}; {note}")

    # Trend over exactly 3 consecutive months
    def _trend(seq: list[tuple[str, float]]) -> tuple[str, float, str]:
        steps = [b2 - a2 for (_, a2), (_, b2) in zip(seq, seq[1:])]
        direction = "rising" if all(s > 0 for s in steps) else "falling" if all(s < 0 for s in steps) else "mixed"
        change = seq[-1][1] - seq[0][1]
        return direction, _r(abs(change)), "higher" if change > 0 else "lower" if change < 0 else "the same"

    if cur:
        variants = (("last3_prev", [_shift(cur, k) for k in (-3, -2, -1)], "3 most recent previous months, current month excluded"),
                    ("last3_incl_current", [_shift(cur, k) for k in (-2, -1, 0)], "last 3 months including the current month"))
        for name, months, desc in variants:
            seq = [(ym, b.units_consumed if ym == cur else (rows[ym].units if ym in rows else None)) for ym in months]
            if any(u is None for _, u in seq):
                continue
            direction, change, rel = _trend(seq)
            add(f"{name}_change_units", change, f"{desc}: {', '.join(f'{ym} ({u})' for ym, u in seq)}; direction: {direction}; "
                f"last month of the three is {change} units {rel} than the first")

    # Average cost per unit (not a tariff rate) and the estimates built on it
    if b.current_bill is not None and b.units_consumed:
        cpu = _r(b.current_bill / b.units_consumed)
        add("effective_avg_cost_per_unit", cpu, "average cost per unit = current_bill / units_consumed (all charges and taxes "
            "divided by units; not a tariff rate)")
        caveat = "fixed charges and slab rates mean the real amount can differ"
        if avg:
            add("est_monthly_cost", _r(cpu * avg), f"estimate: effective_avg_cost_per_unit {cpu} x average units of {n_avg} "
                f"previous months ({avg}); {caveat}")
        lm = facts.get("last_month_units", {}).get("value")
        if lm:
            add("est_bill_at_last_month_units", _r(cpu * lm), f"estimate: effective_avg_cost_per_unit {cpu} x last month's "
                f"units ({lm}); {caveat}")
    if b.total_taxes is not None and b.current_bill:
        add("tax_share_pct", _r(b.total_taxes / b.current_bill * 100), "total_taxes / current_bill * 100")
    if b.payable_after_due_date is not None and b.payable_within_due_date is not None:
        add("late_surcharge", _r(b.payable_after_due_date - b.payable_within_due_date),
            "payable_after_due_date - payable_within_due_date")

    rd, isd, dd = _date(b.reading_date), _date(b.issue_date), _date(b.due_date)
    for name, a, c, txt in (("days_reading_to_issue", rd, isd, "issue_date - reading_date"),
                            ("days_issue_to_due", isd, dd, "due_date - issue_date"),
                            ("days_reading_to_due", rd, dd, "due_date - reading_date")):
        if a and c:
            add(name, (c - a).days, txt)

    for s in internal.split:
        if s.percent is None:
            continue
        slug = _slug(s.label)
        add(f"split_{slug}_pct", s.percent, f"printed split percentage '{s.label}'")
        if b.current_bill is not None:
            add(f"split_{slug}_amount", _r(s.percent / 100 * b.current_bill), f"{s.percent}% of current_bill")

    for i, rate in enumerate(internal.rates, 1):
        desc = rate.description or f"rate line {i}"
        add(f"rate_{i}_units", rate.units, f"units on printed rate line '{desc}'")
        add(f"rate_{i}_rate", rate.rate, f"rate per unit on printed rate line '{desc}'")
        add(f"rate_{i}_amount", rate.amount, f"amount on printed rate line '{desc}'")

    pw = b.payable_within_due_date
    for i, lp in enumerate(internal.late_payment_amounts, 1):
        add(f"late_payment_{i}", lp.amount, f"printed late-payment amount '{lp.label or i}'")
        if lp.amount is not None and pw is not None:
            add(f"late_surcharge_tier_{i}", _r(lp.amount - pw), f"late-payment tier '{lp.label or i}': printed amount "
                f"{lp.amount} - payable_within_due_date {pw}")

    # Net metering: evidence only from extracted flags / rate lines / split labels
    texts = [f for f in internal.validation_flags if not f.startswith("reask_rejected")]
    texts += [r.description or "" for r in internal.rates] + [s.label or "" for s in internal.split]
    hits = [t for t in texts if NET_METER_RE.search(t)]
    if hits:
        add("net_metering_evidence", 1, f"evidence found: {'; '.join(hits)}")
    else:
        add("net_metering_evidence", 0, "no evidence: no export reading, net-metering line or solar note was extracted "
            f"(one reading pair: previous {b.previous_reading}, current {b.current_reading})")

    return facts


# ---- Safe arithmetic ----

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs}


def safe_eval(expr: str, facts: dict) -> float:
    """Evaluate arithmetic over fact names. Only numbers, fact names, + - * /, parentheses,
    round/min/max/abs. Anything else raises ValueError."""
    if len(expr) > 500:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"syntax error in {expr!r}") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in facts:
                raise ValueError(f"unknown fact '{node.id}'")
            return facts[node.id]["value"]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("division by zero")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords and node.args):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    value = ev(tree)
    if not isinstance(value, (int, float)):
        raise ValueError("result is not a number")
    return value
