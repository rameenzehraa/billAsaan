"""Level 2: one text-only answer call per bill.

The model only chooses calculations (expressions over code-computed facts) and phrases answers.
Code evaluates every expression with safe_eval, fills the template, and checks that every number
in the final text traces to a fact, an evaluated expression, the question, or a bill date.
"""
import hashlib
import json
import logging
import re
import time
from datetime import datetime

from google.genai import types
from pydantic import BaseModel, ValidationError

import cache
from calc import compute_facts, safe_eval
from config import ANSWER_PROMPT_VERSION, CACHE_DIR, CALC_VERSION, ROOT, TEMPERATURE, load_settings
from extract import _call, _client
from schema import ExtractionResult

log = logging.getLogger("answer")

ANSWER_DIR = ROOT / CACHE_DIR / "answer"
LOG_DIR = ROOT / CACHE_DIR / "logs"
MAX_OUTPUT_TOKENS = 16384
MAX_INPUT_CHARS = 60_000
SETTINGS_ID = f"answer|thinking=LOW|max_out={MAX_OUTPUT_TOKENS}|transport=v1"

KINDS = {"answered", "not_on_bill", "estimate"}
FMTS = {"pkr", "units", "pct", "number", "count", "days"}
EXTRACTION_FAILED = "Processing failed: this bill could not be read."
ANSWER_FAILED = "Processing failed: an answer could not be generated for this question."


class TCalc(BaseModel):
    name: str
    expr: str
    fmt: str


class TAnswer(BaseModel):
    question_id: str
    kind: str
    calcs: list[TCalc]
    answer_template: str


class TAnswers(BaseModel):
    answers: list[TAnswer]


SYSTEM_PROMPT = """You answer customer questions about one electricity bill, in plain English, using ONLY the data provided:
"bill" (fields extracted from the bill as printed) and "facts" (numbers computed by code, each with its method).

HOW TO ANSWER (for every question_id, exactly once)
- answer_template: 1-4 short sentences. Put every number (amount, units, count, percentage, days, rate) in a placeholder {r1}, {r2}, ...
  Never type a number yourself. Bill dates are filled by code: write {reading_date}, {issue_date}, {due_date} or {bill_month}
  (never type a day-month date). The only digits you may type are years in history month names from fact methods (e.g. "July 2025").
- Placeholders are ONLY {r1}, {r2}, ... and the four bill dates above. Never put a fact name in braces (not {units_consumed}):
  to show a fact, add a calc whose expr is the fact name.
- Month lists: give every month's units its own placeholder; never copy a number from a fact method. Example for
  prev12_months_above_200 whose method lists "2025-07 (...), 2025-08 (...)":
  template "{r1} previous months went above 200 units: July 2025 ({r2} units) and August 2025 ({r3} units)."
  calcs r1 = prev12_months_above_200 (count), r2 = hist_2025_07_units (units), r3 = hist_2025_08_units (units).
  Write month names from the fact name's year and month exactly (hist_2025_07 -> July 2025).
- calcs: one entry per placeholder: name ("r1"), expr, fmt.
  * expr is arithmetic over fact names only: fact names, numbers, + - * /, parentheses, round(), min(), max(), abs().
    No comparisons, no other functions. To quote a fact directly, expr is just its name (e.g. "current_bill").
  * fmt: pkr (money), units (kWh), pct (percent value, already x100), number (averages/rates), count, days.
  * Code computes the values; do not do arithmetic in your head.
- kind: "answered", "estimate" or "not_on_bill".

RULES
- Use only the bill. No outside tariff, rate or tax knowledge.
- If a needed value is missing from bill and facts, set kind "not_on_bill" and say "The bill does not show <what>." If part of the
  question can be answered, answer that part and say what the bill does not show.
- Estimates: set kind "estimate". The answer MUST begin with "This is an estimate." and then explain the method in words
  (e.g. "This is an estimate. Assuming all charges scale with units, ...").
- Ambiguous questions: state the meaning you used, or answer both meanings (e.g. with and without the current month).
- History facts named prev12_* and hist_* cover previous months only and exclude the current month; the current month's units are
  units_consumed. prev12_months_above_<N> counts previous months with units above N; its method lists the months.
- Partial history: prev12_* facts use only the previous months the bill shows (prev12_count of them; missing months are
  never counted). If prev12_count is below 12, every answer that uses a prev12_* fact or history months must say how many
  months were used, via a placeholder (e.g. "Based on the {r1} previous months shown on your bill, ..." with r1 = prev12_count).
- Negative amounts are credits: say "credit" and use abs() in the expression.
- taxes = [] with a total_taxes value means the bill prints only a combined tax amount.
- Be specific and complete, like a helpful billing agent: give the direct answer first, then the supporting numbers.
  * Write "PKR" before every money amount and "units" after every unit count.
  * When the bill prints directly related figures (line items, printed percentages or splits such as split_*_pct), include
    them via placeholders.
  * Cite supporting values with their own placeholders: the individual line items behind a total (e.g. each tax line
    tax_gst_1, tax_electricity_duty_1), the share of the bill, any printed split percentage, and for month counts or
    min/max, name each month and give its units (via hist_YYYY_MM_units facts; month names come from fact methods).
  * When the current month matters to a history question, say whether it is included and give units_consumed.
  * Averages and rates use fmt "number" (2 decimals), not "units".
  * Example style (values are placeholders): "Taxes and duties total PKR {r1} out of your current bill of PKR {r2},
    about {r3}%. The tax lines are sales tax (PKR {r4}) and electricity duty (PKR {r5})."
QUESTION HANDLING (match the question's meaning, not its wording)
- Use the most specific fact that matches the question; read its method text for months, labels and direction.
- Comparison with the average: compare with the average (this_month_vs_avg_units / this_month_vs_avg_pct and prev12_avg_units),
  not with last month. Use the method's "higher than" / "lower than" wording.
- History questions: always say whether the current month is included and how many months were used.
- Season questions: state the definition used (summer = June-August, winter = December-February) and the months used, from
  summer_* / winter_* facts. Never say the bill does not show this when those facts exist.
- Payment-history questions: describe what the history records (hist_payment_zero_or_missing_count,
  hist_paid_less_than_billed_count and the months in their methods). Say that a zero or lower recorded payment does not by
  itself prove non-payment, since reporting timing can differ. If no payment facts exist, the bill does not show payments.
- Trend questions: use the 3-month facts (last3_prev_change_units by default, last3_incl_current_change_units when the
  current month matters). Use only their direction word (rising / falling / mixed) and their month labels, give each month's
  units via hist_YYYY_MM_units (and units_consumed for the current month), and say which of the two windows you used.
- Never write a signed change such as "down by -9": wrap signed facts in abs() and say "up" or "down" in words.
- Cost estimates (set-aside per month, bill at another number of units): start "This is an estimate.", use est_monthly_cost or
  est_bill_at_last_month_units, name the method (average cost per unit x units, with effective_avg_cost_per_unit) and say that
  fixed charges and slab rates can change the real amount. Never present the current bill as the estimate for other units.
- Late payment: list every tier (late_surcharge_tier_<n> and late_payment_<n>), not only the highest. Describe each tier in
  words, e.g. "if paid after the due date ({due_date})" and "the higher late amount (later payment)". Never copy a printed
  tier label (it contains dates and codes) and never put a date placeholder inside or next to label text.
- Percentages: every percentage is its own placeholder with fmt "pct" followed by "%" in the template (e.g. "{r3}%").
- Every number in the text goes through a {rN} placeholder, including values taken from history (hist_YYYY_MM_units,
  same_month_last_year_units, last_month_units). Never retype a number from a fact or method.
- Metering type: if net_metering_evidence is 1, say net metering is indicated and cite the evidence. If it is 0, say the bill
  shows no export or net-metering readings, so it appears to be a standard connection, but this cannot be confirmed from the bill.
- Units must match the fact: "PKR" only with money facts (fmt pkr), "units" only with unit facts. Never write "PKR ... units".
- kind must match the text: use "not_on_bill" only when the answer says the bill does not show the value.
- English only. Never include names, addresses, account, reference, consumer or meter numbers.
"""

REASK_PROMPT = """Some of your answers failed validation. Fix ONLY these questions and return them in the same schema.
Errors:
{errors}

Your previous answers for them:
{previous}
"""


# ---- Formatting and validation ----

def _trim(s: str) -> str:
    return s.rstrip("0").rstrip(".") if "." in s else s


def format_value(v: float, fmt: str) -> str:
    if fmt == "pkr":
        s = f"{v:,.2f}"
        return s[:-3] if s.endswith(".00") else s
    if fmt in ("units", "count", "days"):
        return f"{round(v):,}"
    if fmt == "pct":
        return _trim(f"{v:.2f}")
    return _trim(f"{v:,.2f}")


_NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for tok in _NUM_RE.findall(text):
        tok = tok.rstrip(",").replace(",", "")
        try:
            out.append(abs(float(tok)))
        except ValueError:
            pass
    return out


DATE_FIELDS = ("reading_date", "issue_date", "due_date", "bill_month")
_TYPED_DATE_RE = re.compile(
    r"\b\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(st|nd|rd|th)?,"
    r"|\b\d{4}-\d{2}(-\d{2})?\b", re.I)


def date_values(bill: dict) -> dict[str, str]:
    """Code-formatted bill dates for {reading_date} etc.: '3 April 2026', bill_month 'April 2026'."""
    out = {}
    for f in DATE_FIELDS:
        v = bill.get(f)
        try:
            if v and f == "bill_month":
                out[f] = datetime.strptime(v, "%Y-%m").strftime("%B %Y")
            elif v:
                d = datetime.strptime(v, "%Y-%m-%d")
                out[f] = f"{d.day} {d.strftime('%B %Y')}"
        except ValueError:
            pass
    return out


def _allowed_numbers(facts: dict, question: str, values: list[float], dates: dict[str, str]) -> set[float]:
    """Numbers that may appear in a final answer: evaluated calcs, the question, code-filled dates,
    history years, and the 12-month window. Fact values typed directly are not allowed."""
    allowed: set[float] = {12.0}
    for v in values:
        for d in (0, 1, 2):
            allowed.add(abs(round(float(v), d)))
    for text in [question] + list(dates.values()):
        allowed.update(_numbers(text))
    for name in facts:
        m = re.match(r"hist_(\d{4})_\d{2}_", name)
        if m:
            allowed.add(float(m.group(1)))
    return allowed


def render(ans: TAnswer, question: str, facts: dict, dates: dict[str, str]) -> tuple[str | None, dict, list[str]]:
    """Evaluate calcs, fill template, validate. Returns (text | None, log entry, errors)."""
    errors: list[str] = []
    entry = {"kind": ans.kind, "template": ans.answer_template, "calcs": []}
    if ans.kind not in KINDS:
        errors.append(f"kind must be one of {sorted(KINDS)}")
    values: dict[str, str] = {}
    raw_values: list[float] = []
    for c in ans.calcs:
        if not re.fullmatch(r"r\d+", c.name):
            errors.append(f"calc name {c.name!r} must look like r1")
            continue
        if c.fmt not in FMTS:
            errors.append(f"{c.name}: fmt must be one of {sorted(FMTS)}")
            continue
        try:
            v = safe_eval(c.expr, facts)
        except (ValueError, TypeError, OverflowError) as e:
            errors.append(f"{c.name}: expr {c.expr!r} failed: {e}")
            continue
        values[c.name] = format_value(v, c.fmt)
        raw_values.append(v)
        entry["calcs"].append({"name": c.name, "expr": c.expr, "fmt": c.fmt, "value": v, "text": values[c.name]})

    template = ans.answer_template or ""
    if _TYPED_DATE_RE.search(template):
        errors.append("dates must be written with {reading_date}/{issue_date}/{due_date}/{bill_month}, not typed")
    missing = [p for p in re.findall(r"\{(\w+)\}", template) if p not in values and p not in dates]
    if missing:
        errors.append(f"placeholders without a valid calc or bill date: {missing}")
    fill = {**dates, **values}
    text = re.sub(r"\{(\w+)\}", lambda m: fill.get(m.group(1), m.group(0)), template).strip()
    if "{" in text or "}" in text:
        errors.append("unresolved braces in answer")
    if not text:
        errors.append("empty answer")
    if ans.kind == "not_on_bill" and "does not show" not in text.lower():
        errors.append('kind not_on_bill must say "The bill does not show ..."')
    if ans.kind == "estimate" and "estimat" not in text.lower():
        errors.append("kind estimate must say it is an estimate")
    if not errors:
        allowed = _allowed_numbers(facts, question, raw_values + [float(x.replace(",", "")) for x in values.values()], dates)
        untraced = sorted({n for n in _numbers(text) if n not in allowed})
        if untraced:
            errors.append(f"numbers typed directly instead of via a calc: {untraced}")
    entry["text"] = text
    return (None if errors else text), entry, errors


# ---- Cache ----

def answer_key(result: ExtractionResult, facts: dict, questions: list[tuple[str, str]], model: str) -> str:
    payload = json.dumps({"r": result.model_dump(mode="json"), "f": facts, "q": questions}, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(payload.encode())
    h.update(f"|model={model}|temp={TEMPERATURE}|{SETTINGS_ID}|prompt={ANSWER_PROMPT_VERSION}|calc={CALC_VERSION}".encode())
    return h.hexdigest()


def _cache_get(key: str) -> dict | None:
    try:
        return json.loads((ANSWER_DIR / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_log(bill_id: str, entries: dict) -> None:
    try:
        cache._atomic_write(LOG_DIR / f"answers_{bill_id}.json", entries)
    except OSError:
        pass


# ---- Main entry ----

def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=TAnswers,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


def _parse(raw: str) -> tuple[dict[str, TAnswer], str | None]:
    try:
        parsed = TAnswers.model_validate_json(raw)
    except ValidationError as e:
        return {}, f"invalid JSON: {str(e)[:200]}"
    out: dict[str, TAnswer] = {}
    for a in parsed.answers:
        out.setdefault(a.question_id.strip(), a)  # first answer wins; duplicates are ignored
    return out, None


def answer_bill(bill_id: str, questions: list[tuple[str, str]], result: ExtractionResult,
                extraction_ok: bool, refresh: bool = False) -> tuple[dict[str, str], dict]:
    """Returns ({question_id: answer text}, meta). Never raises."""
    meta = {"cache": "-", "latency_s": None, "input_tokens": None, "output_tokens": None, "attempts": 0,
            "reasked": False, "failed": 0, "error": None}
    qids = [q for q, _ in questions]
    if not extraction_ok:
        meta.update(failed=len(qids), error="extraction failed; no answer call")
        return {q: EXTRACTION_FAILED for q in qids}, meta

    start = time.perf_counter()
    try:
        settings = load_settings()
        facts = compute_facts(result)
        key = answer_key(result, facts, questions, settings.model_name)
        if not refresh and (entry := _cache_get(key)) is not None and set(entry.get("answers", {})) == set(qids):
            meta.update(cache="hit", latency_s=0.0, input_tokens=0, output_tokens=0, reasked=entry["meta"].get("reasked", False))
            _write_log(bill_id, entry["log"])
            return entry["answers"], meta
        meta["cache"] = "miss"
        dates = date_values(result.level1.model_dump(mode="json"))
        payload = {"questions": [{"question_id": q, "question": t} for q, t in questions],
                   "bill": result.level1.model_dump(mode="json"), "facts": facts}
        user = json.dumps(payload, ensure_ascii=False)
        if len(user) > MAX_INPUT_CHARS:
            raise ValueError(f"answer input too large ({len(user)} chars)")
        client = _client(settings.api_key)
    except Exception as e:  # noqa: BLE001
        meta.update(failed=len(qids), error=f"{type(e).__name__}: {e}")
        return {q: ANSWER_FAILED for q in qids}, meta

    qtext = dict(questions)
    answers: dict[str, str] = {}
    logs: dict[str, dict] = {}
    errors: dict[str, list[str]] = {}
    proposed: dict[str, TAnswer] = {}
    unverified: dict[str, str] = {}  # rendered text whose only problem is untraced numbers

    def validate(batch: dict[str, TAnswer], wanted: list[str]) -> None:
        for q in wanted:
            a = batch.get(q)
            if a is None:
                errors[q] = ["no answer returned for this question_id"]
                logs.setdefault(q, {"template": None, "errors": errors[q]})
                continue
            proposed[q] = a
            text, entry, errs = render(a, qtext[q], facts, dates)
            prior = logs.get(q)
            logs[q] = entry | {"errors": errs}
            if prior is not None:  # keep the first-pass attempt for diagnosis
                logs[q]["first_pass"] = {"template": prior.get("template"), "errors": prior.get("errors")}
            if text is None:
                errors[q] = errs
                if entry.get("text") and all(e.startswith("numbers typed directly") for e in errs):
                    unverified[q] = entry["text"]
            else:
                answers[q] = text
                errors.pop(q, None)

    resp, err = _call(client, settings.model_name, [user], bill_id, meta, config=_config())
    if resp is not None:
        batch, perr = _parse(resp.text or "")
        validate(batch, qids)
        if perr:
            errors.update({q: [perr] for q in qids if q not in answers})
        extra = set(batch) - set(qids)
        if extra:
            log.warning("bill=%s ignored unknown question_ids %s", bill_id, sorted(extra))

        pending = [q for q in qids if q not in answers]
        if pending:  # one re-ask with the errors, for the failed questions only
            meta["reasked"] = True
            prev = {q: proposed[q].model_dump() for q in pending if q in proposed}
            missing_q = [{"question_id": q, "question": qtext[q]} for q in pending]
            reask = REASK_PROMPT.format(
                errors="\n".join(f"- {q}: {'; '.join(errors.get(q, []))}" for q in pending),
                previous=json.dumps(prev, ensure_ascii=False))
            resp2, err2 = _call(client, settings.model_name, [user, json.dumps({"fix_questions": missing_q}), reask],
                                bill_id, meta, config=_config())
            if resp2 is not None:
                batch2, perr2 = _parse(resp2.text or "")
                validate(batch2, pending)
                if perr2:
                    errors.update({q: [perr2] for q in pending if q not in answers})
            else:
                err = err2
    meta["latency_s"] = round(time.perf_counter() - start, 2)

    meta["unverified"] = []
    for q in qids:
        if q not in answers and q in unverified:  # renderable; numbers not traceable -> keep, flag in log
            answers[q] = unverified[q]
            meta["unverified"].append(q)
            logs.setdefault(q, {})["unverified"] = True
            log.warning("bill=%s %s kept unverified answer: %s", bill_id, q, errors.get(q))
        elif q not in answers:
            answers[q] = ANSWER_FAILED
            meta["failed"] += 1
            logs.setdefault(q, {})["errors"] = errors.get(q, [err or "unknown"])
    if meta["failed"]:
        meta["error"] = err or f"{meta['failed']} question(s) failed validation"
    ordered = {q: answers[q] for q in qids}
    ordered_logs = {q: logs.get(q, {}) for q in qids}
    _write_log(bill_id, ordered_logs)
    if meta["failed"] == 0:  # cache successes only
        cache._atomic_write(ANSWER_DIR / f"{key}.json", {"bill_id": bill_id, "key": key, "answers": ordered,
                                                          "log": ordered_logs, "meta": {k: v for k, v in meta.items()}})
    log.info("bill=%s answers latency=%ss in=%s out=%s attempts=%s reasked=%s failed=%s error=%s", bill_id,
             meta["latency_s"], meta["input_tokens"], meta["output_tokens"], meta["attempts"], meta["reasked"],
             meta["failed"], meta["error"])
    return ordered, meta
