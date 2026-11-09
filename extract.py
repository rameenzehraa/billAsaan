"""Level 1 + internal-facts extraction: one Gemini vision call per bill.

extract_bill() never raises. On failure it returns empty_level1() with meta["error"] set.
"""
import json
import logging
import re
import time
from pathlib import Path

import httpx

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from config import MAX_RETRIES, PROMPT_VERSION, REQUEST_TIMEOUT_S, ROOT, TEMPERATURE, load_settings
from images import ImageLoadError, load_image
from failures import safe_failure
from checks import MISMATCH, count_mismatches, run_checks
from schema import ExtractionResult, InternalFacts, empty_level1

log = logging.getLogger("extract")

MAX_OUTPUT_TOKENS = 16384  # headroom for thinking + JSON (KESC_0008 used ~1.6k)
# Part of the cache key: any change to these settings invalidates cached extractions.
SETTINGS_ID = f"thinking=LOW|media=HIGH|max_out={MAX_OUTPUT_TOKENS}|schema_transport=v1"
DEV_DUMP_DIR = ROOT / ".cache" / "dev"  # raw responses for inspection only, not a cache

CHARGE_TYPES = {"energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other"}
TAX_TYPES = {"gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"}
LEVEL1_NUMBER_FIELDS = {
    "sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed", "total_charges",
    "total_taxes", "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date",
}


# ---- Transport schema sent to Gemini: plain types only; strict validation happens locally ----

class TItem(BaseModel):
    type: str
    amount: float


class TLevel1(BaseModel):
    provider: str | None
    tariff: str | None
    sanctioned_load_kw: float | None
    bill_month: str | None
    reading_date: str | None
    issue_date: str | None
    due_date: str | None
    previous_reading: float | None
    current_reading: float | None
    units_consumed: float | None
    charges: list[TItem]
    total_charges: float | None
    taxes: list[TItem]
    total_taxes: float | None
    current_bill: float | None
    arrears: float | None
    payable_within_due_date: float | None
    payable_after_due_date: float | None


class TTransport(BaseModel):
    level1: TLevel1
    internal: InternalFacts


SYSTEM_PROMPT = """You extract data from one Pakistani electricity bill image (K-Electric, LESCO or IESCO) into JSON.
Return only the JSON object matching the response schema: {"level1": {...}, "internal": {...}}.

GENERAL RULES
- Extract values exactly as printed. Never recalculate, round, sum or correct any amount.
- Amounts are plain numbers in PKR: remove "Rs", currency labels and thousands commas (Rs. 19,964 -> 19964). Keep printed decimals.
- Credits and subsidies are negative: 38209CR -> -38209; a credit/negative previous balance stays negative.
- Dates: YYYY-MM-DD (22nd Sep 2026 -> 2026-09-22). bill_month: YYYY-MM (MAR 2026 -> 2026-03). Two-digit years are 20xx.
- Use null when a value is not printed, blank, covered by a grey box, or unreadable. Do not guess. Use [] for an empty list.
- Never output personal identifiers anywhere (names, addresses, CNIC, account, reference, consumer or meter numbers), including in "internal".
- Parts of the bill may be in Urdu; read the numbers and use English labels.

LEVEL1 FIELDS
- provider: "KE", "LESCO" or "IESCO".
- tariff: tariff code exactly as printed (e.g. A1-R, A-1a(01)).
- sanctioned_load_kw: sanctioned load (Sanc Load / S.Load / San Load) in kW.
- bill_month, reading_date, issue_date, due_date: as printed. due_date is the last date to pay without late payment surcharge.
- previous_reading, current_reading: printed meter readings. units_consumed: units billed for the current month as printed.
- charges: one entry per printed line in the electricity charges section (the section that holds energy cost, fixed charges, fuel/tariff adjustments, surcharges, rent).
  * Skip lines that are blank or printed as 0. Do not include subtotal/total lines as entries.
  * Include every printed charge line exactly once, wherever it sits on the bill, including side boxes next to the
    charges breakdown (e.g. a "Total FPA" amount printed outside the Current Bill box is an fpa entry). Never list subtotals
    or totals (Total/Net Electricity Charges, Current Bill, Grand Total, Payable) as charge lines.
  * Keep separate entries when the same type appears more than once (e.g. two energy lines -> two "energy" entries), in printed order.
  * type must be one of: energy (cost of units: Variable Upto ... Units, Cost of Electricity, Energy Charges), fixed (fixed charges),
    fpa (fuel price adjustment, FPA or FCA), quarterly_adjustment (QTA, Uniform Quarterly Adjustment), surcharge (e.g. Additional Surcharge (PHL), F.C Surcharge),
    meter_rent (meter or service rent), subsidy (government subsidy/relief, negative amount), other (any other charge line).
- total_charges: the printed charges subtotal, or null if no subtotal is printed.
- taxes: one entry per printed line in the tax / government charges section. Classify a line by the section it is printed in, not by its name.
  * type must be one of: gst (sales tax/GST, including GST on FPA), electricity_duty (ED), income_tax, municipal_tax (municipal/local charges printed under taxes, e.g. KE MUCT (KMC)), other_tax (extra tax, further tax, TV fee, any other).
  * Skip lines that are blank or 0.
  * COMBINED TAX: if the bill prints only one combined tax line (e.g. "Taxes 15.24% 498" or "Taxes 1,723.51") with no per-tax breakdown,
    set taxes: [] and put that amount in total_taxes. Never create an other_tax (or any) entry for a combined amount.
- total_taxes: the printed tax subtotal, or null if no subtotal is printed.
- current_bill: the printed amount for this month's bill.
- arrears: previous dues / arrears carried forward, with the printed sign (credit -> negative); null if blank.
- payable_within_due_date: the amount payable by the due date.
- payable_after_due_date: the amount payable after the due date. If several late-payment amounts are shown, use the highest.

INTERNAL FIELDS (used later to answer questions; same no-recalculation and no-PII rules)
- usage_history: every month in the printed billing/consumption history table: month (YYYY-MM if possible, else as printed), units, bill_amount, payment (null if not shown). Do not include the current month unless it is printed in the table.
- rates: printed tariff slabs or rate lines: description as printed, units, rate per unit, amount (null if not shown).
- split: any printed percentage breakdown of the bill (e.g. energy charges 83.39%, taxes 16.61%): label and percent.
- late_payment_amounts: every printed late-payment amount with its label/date (e.g. "after 24 Apr").
- validation_flags: short notes on values that were unreadable or ambiguous (no personal data). [] if none.
"""


def _client(api_key: str, force_ipv4: bool | None = None) -> genai.Client:
    """force_ipv4: pass settings.force_ipv4 from the caller; None reads settings (standalone use)."""
    if force_ipv4 is None:
        force_ipv4 = load_settings().force_ipv4
    client_args = {}
    if force_ipv4:
        # Bind direct connections to IPv4 on networks with a broken IPv6 TLS path.
        # HTTPTransport retains certificate/hostname verification and environment CA trust.
        client_args["transport"] = httpx.HTTPTransport(local_address="0.0.0.0", retries=0)
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=REQUEST_TIMEOUT_S * 1000, client_args=client_args))


def _is_transient(e: Exception) -> bool:
    if isinstance(e, errors.APIError):
        return e.code == 429 or (e.code or 0) >= 500
    name = type(e).__name__.lower()
    return "timeout" in name or "connect" in name or "timeout" in str(e).lower()


_NUM_RE = re.compile(r"^\(?-?\d+(\.\d+)?\)?$")


def _to_number(v):
    """Format-only conversion: "1,234.50" -> 1234.5, "38209CR" -> -38209, 900.0 -> 900. Value unchanged."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        s = re.sub(r"^(rs\.?|pkr)\s*", "", s, flags=re.I).strip()
        if not s:
            return None
        negative = False
        if s.upper().endswith("CR"):
            negative, s = True, s[:-2].strip()
        if not _NUM_RE.match(s):
            return v  # leave as is; Pydantic will reject it
        if s.startswith("(") and s.endswith(")"):
            negative, s = True, s[1:-1]
        n = float(s)
        v = -abs(n) if negative else n
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _norm_type(t, allowed: set[str], fallback: str) -> str:
    key = re.sub(r"[\s\-]+", "_", str(t or "").strip().lower())
    return key if key in allowed else fallback


def _strip(obj):
    if isinstance(obj, str):
        s = obj.strip()
        return s or None
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()}
    return obj


def normalize(data: dict) -> dict:
    """Pre-validation cleanup. Never changes a numeric value."""
    data = _strip(data) or {}
    l1 = data.get("level1") or {}
    for k in LEVEL1_NUMBER_FIELDS:
        if k in l1:
            l1[k] = _to_number(l1[k])
    if isinstance(l1.get("provider"), str):
        l1["provider"] = {"KESC": "KE", "K-ELECTRIC": "KE"}.get(l1["provider"].upper(), l1["provider"].upper())
    for key, allowed, fallback in (("charges", CHARGE_TYPES, "other"), ("taxes", TAX_TYPES, "other_tax")):
        items = l1.get(key) or []
        l1[key] = [
            {"type": _norm_type(i.get("type"), allowed, fallback), "amount": _to_number(i.get("amount"))}
            for i in items if isinstance(i, dict) and _to_number(i.get("amount")) not in (None, 0)
        ]
    # §5.2: "If only a combined tax amount is printed, return [] and put that amount in total_taxes."
    # A lone other_tax entry equal to total_taxes is that combined amount, not a classified tax line.
    t, total = l1["taxes"], l1.get("total_taxes")
    if (len(t) == 1 and t[0]["type"] == "other_tax" and isinstance(total, (int, float))
            and isinstance(t[0]["amount"], (int, float)) and abs(t[0]["amount"] - total) <= 1):
        l1["taxes"] = []
    internal = data.get("internal") or {}
    for key, num_fields in (
        ("usage_history", ("units", "bill_amount", "payment")),
        ("rates", ("units", "rate", "amount")),
        ("split", ("percent",)),
        ("late_payment_amounts", ("amount",)),
    ):
        for row in internal.get(key) or []:
            if isinstance(row, dict):
                for f in num_fields:
                    if f in row:
                        row[f] = _to_number(str(row[f]).rstrip("%") if isinstance(row[f], str) else row[f])
    internal["validation_flags"] = [f for f in internal.get("validation_flags") or [] if f]
    return {"level1": l1, "internal": internal}


def _dump(bill_id: str, raw: str, parsed: dict | None) -> None:
    try:
        DEV_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        (DEV_DUMP_DIR / f"{bill_id}.raw.txt").write_text(raw, encoding="utf-8")
        if parsed is not None:
            (DEV_DUMP_DIR / f"{bill_id}.parsed.json").write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    except OSError:
        pass


REASK_PROMPT = """Your previous JSON for this bill is below, followed by the fields that failed a local consistency check.
Look at the image again and re-read ONLY the flagged fields exactly as printed.
Do not calculate values to make the check pass: if the printed numbers genuinely do not add up, keep them as printed.
Keep every other field as it was. Return the complete JSON object again in the same schema.

PREVIOUS JSON:
{previous}

FLAGGED:
{flagged}
"""


def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=TTransport,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


RATE_RETRY_MAX_S = 5.0  # longest wait for the single per-minute 429 retry


def _error_details(e: Exception) -> list:
    """The google.rpc details list of an APIError (QuotaFailure, RetryInfo, ...), or []."""
    body = getattr(e, "details", None)
    if isinstance(body, dict):
        body = body.get("error", body)
        details = body.get("details") if isinstance(body, dict) else None
        return details if isinstance(details, list) else []
    return []


def _daily_quota(e: Exception) -> bool:
    """True when a 429 names a per-day quota (e.g. quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier)."""
    for d in _error_details(e):
        for v in (d.get("violations") or []) if isinstance(d, dict) else []:
            if isinstance(v, dict) and "perday" in str(v.get("quotaId", "")).lower():
                return True
    return False


def _retry_delay(e: Exception) -> float:
    """Server RetryInfo delay (e.g. "23s"), capped at RATE_RETRY_MAX_S; 2 s when absent."""
    for d in _error_details(e):
        delay = d.get("retryDelay") if isinstance(d, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(0.0, min(float(delay[:-1]), RATE_RETRY_MAX_S))
            except ValueError:
                pass
    return 2.0


def _call(client: genai.Client, model: str, contents: list, bill_id: str, meta: dict,
          config: types.GenerateContentConfig | None = None):
    """One generate_content call with retries on transient errors only. Returns (response | None, error | None)."""
    rate_retried = False
    for attempt in range(1, MAX_RETRIES + 2):
        meta["attempts"] += 1
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config or _config())
        except Exception as e:  # noqa: BLE001
            if getattr(e, "code", None) == 429:
                # Daily quota: retrying the same model cannot succeed today -> fail now (limits.py falls back).
                # Per-minute / unspecified rate limit: one short retry only.
                if not _daily_quota(e) and not rate_retried and attempt <= MAX_RETRIES:
                    rate_retried = True
                    wait = _retry_delay(e)
                    log.warning("bill=%s attempt=%d rate limited; one retry in %.1fs", bill_id, attempt, wait)
                    time.sleep(wait)
                    continue
            elif attempt <= MAX_RETRIES and _is_transient(e):
                wait = 2 ** attempt
                log.warning("bill=%s attempt=%d transient error %s; retrying in %ds", bill_id, attempt, type(e).__name__, wait)
                time.sleep(wait)
                continue
            meta["failure"] = safe_failure(e, "api")
            msg = f"HTTP {e.code}: {(e.message or '')[:120]}" if isinstance(e, errors.APIError) else f"{type(e).__name__}: {str(e)[:120]}"
            return None, f"API call failed: {msg}"
        u = resp.usage_metadata
        meta["input_tokens"] = (meta["input_tokens"] or 0) + (getattr(u, "prompt_token_count", None) or 0)
        meta["output_tokens"] = ((meta["output_tokens"] or 0) + (getattr(u, "candidates_token_count", None) or 0)
                                 + (getattr(u, "thoughts_token_count", None) or 0))
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        meta["finish_reason"] = getattr(finish, "name", str(finish)) if finish is not None else None
        if finish == types.FinishReason.MAX_TOKENS:  # truncated JSON: not retryable
            meta["failure"] = safe_failure(ValueError(), "validation")
            return None, f"Output truncated at max_output_tokens={MAX_OUTPUT_TOKENS} (finish_reason=MAX_TOKENS)"
        return resp, None
    return None, "API call failed: retries exhausted"


def _parse(bill_id: str, raw: str, tag: str, private: bool = False, meta: dict | None = None):
    """Returns (ExtractionResult | None, error | None). Dumps raw/parsed for inspection."""
    parsed = None
    try:
        parsed = normalize(json.loads(raw))
        return ExtractionResult.model_validate(parsed), None
    except (json.JSONDecodeError, ValidationError, TypeError, AttributeError) as e:
        if meta is not None:
            meta["failure"] = safe_failure(e, "validation")
        return None, f"Invalid model output: {type(e).__name__}: {str(e)[:300]}"
    finally:
        if not private:
            _dump(f"{bill_id}{tag}", raw, parsed)


def _reject_reason(orig: ExtractionResult, checks: list[dict], new: ExtractionResult | None, err: str | None) -> str | None:
    """Why a re-ask result must not replace the original, or None if it may."""
    if new is None:
        return f"invalid ({(err or 'no response')[:80]})"
    new_checks = run_checks(new.level1, new.internal)
    if count_mismatches(new_checks) >= count_mismatches(checks):
        return f"not fewer mismatches ({count_mismatches(new_checks)} vs {count_mismatches(checks)})"
    o, n = orig.level1.model_dump(), new.level1.model_dump()
    dropped = [k for k in o if o[k] not in (None, []) and n[k] in (None, [])]
    if dropped:
        return f"dropped populated fields: {', '.join(dropped)}"
    return None


def extract_bill(bill_id: str, provider: str | None, image_path: str | Path,
                 image: tuple | None = None, *, private: bool = False) -> tuple[ExtractionResult, dict]:
    """image: optional preloaded load_image() output, to avoid loading twice."""
    meta = {"latency_s": None, "input_tokens": None, "output_tokens": None, "model": None, "attempts": 0,
            "error": None, "finish_reason": None, "reasked": False, "kept": None, "reask_reason": None, "checks": []}
    fallback = ExtractionResult(level1=empty_level1(bill_id, provider))
    start = time.perf_counter()
    try:
        settings = load_settings()
        meta["model"] = settings.model_name
        if not settings.api_key:
            raise RuntimeError("API_KEY is not set in .env")
        data, mime, _, _ = image or load_image(image_path)
    except Exception as e:  # noqa: BLE001
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["failure"] = safe_failure(e, "image" if isinstance(e, ImageLoadError) else "configuration")
        if not private:
            log.error("bill=%s setup failed: %s", bill_id, meta["error"])
        return fallback, meta

    client = _client(settings.api_key, settings.force_ipv4)
    image_part = types.Part.from_bytes(data=data, mime_type=mime)
    contents = [
        image_part,
        f"Bill provider: {provider or 'unknown'}. Extract the bill into the JSON schema following every rule. "
        f"(prompt {PROMPT_VERSION})",
    ]

    resp, err = _call(client, settings.model_name, contents, bill_id, meta)
    if resp is None:  # API failure or truncation: no re-ask
        return _finish(bill_id, fallback, meta, start, err, private)

    raw = resp.text or ""
    result, err = _parse(bill_id, raw, "", private, meta)
    checks = run_checks(result.level1, result.internal) if result else []
    flagged = [c for c in checks if c["status"] == MISMATCH]

    if result is None or flagged:  # at most one re-ask, only on validation failure or a Mismatch
        meta["reasked"] = True
        if result is None:
            previous, why = raw[:8000], f"The previous output failed validation: {err}"
        else:
            previous = result.model_dump_json()
            why = "\n".join(f"- {c['name']} ({', '.join(c['fields'])}): {c['detail']}" for c in flagged)
        reask_contents = [image_part, REASK_PROMPT.format(previous=previous, flagged=why)]
        resp2, err2 = _call(client, settings.model_name, reask_contents, bill_id, meta)
        result2, err2 = _parse(bill_id, resp2.text or "", ".reask", private, meta) if resp2 is not None else (None, err2)
        if result is None:  # original invalid: any valid re-ask wins
            if result2 is not None:
                result, checks, err = result2, run_checks(result2.level1, result2.internal), None
                meta.update(kept="reask", reask_reason="original failed validation")
            else:
                err = err2 or err
        else:
            reason = _reject_reason(result, checks, result2, err2)
            if reason is None:
                result, checks = result2, run_checks(result2.level1, result2.internal)
                meta.update(kept="reask", reask_reason="fewer mismatches, no fields dropped")
            else:
                result.internal.validation_flags.append(f"reask_rejected: {reason}")
                meta.update(kept="original", reask_reason=f"reask_rejected: {reason}")
    else:
        meta["kept"] = "original"

    if result is None:
        return _finish(bill_id, fallback, meta, start, err, private)
    meta["checks"] = checks
    return _finish(bill_id, result, meta, start, None, private)


def _finish(bill_id: str, result: ExtractionResult, meta: dict, start: float, error: str | None, private: bool = False):
    meta["latency_s"] = round(time.perf_counter() - start, 2)
    meta["error"] = error
    if error is None:
        meta.pop("failure", None)
    if private:
        return result, meta
    log.info(
        "bill=%s model=%s latency=%ss in_tokens=%s out_tokens=%s attempts=%s reasked=%s kept=%s error=%s",
        bill_id, meta["model"], meta["latency_s"], meta["input_tokens"], meta["output_tokens"], meta["attempts"],
        meta["reasked"], meta["kept"], meta["error"],
    )
    return result, meta

