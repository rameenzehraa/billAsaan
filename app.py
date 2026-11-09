"""BillAsaan public page: upload one electricity bill -> Process -> key stats.

Batch/CSV work stays in run.py (CLI). Uploads live only in memory and session_state.
Run: streamlit run app.py
"""
import json
import logging

import streamlit as st

import limits
import ui_styles
import viewer
from config import load_limits
from failures import ProcessingFailure, safe_failure
from telemetry import stats

log = logging.getLogger("app")
if not log.handlers:  # Streamlit reruns the script; attach the console handler once
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

BADGE = {"Matched": ("green", ":material/check_circle:"), "Mismatch": ("red", ":material/cancel:"),
         "Insufficient Data": ("gray", ":material/help:")}
CHECK_LABELS = {"date_formats": "Date formats", "reading_diff": "Readings match units",
                "charges_taxes_vs_current_bill": "Charges + taxes = current bill",
                "current_plus_arrears_vs_payable": "Current bill + arrears = payable",
                "late_ge_within": "Late amount ≥ amount due"}

st.set_page_config(page_title="BillAsaan", page_icon="⚡", layout="wide")
ss = st.session_state
ss.setdefault("results", {})   # image sha256 -> build_view() output; reruns read this, never the API
ss.setdefault("errors", {})    # image sha256 -> safe user-facing message
ss.setdefault("busy", None)    # image sha256 being processed, or None
ss.setdefault("use_sample", False)


def _use_upload():
    ss.use_sample = False


def _use_sample():
    ss.use_sample = True


def _process(key: str):
    if ss.busy is None and key not in ss.results:
        ss.errors.pop(key, None)
        ss.busy = key


CONSENT_NOTE = "This free demo uses Google's Gemini free tier, so Google may use uploaded bills to improve its products."
CONSENT = "I've covered my name, address and reference numbers"
PAID_NOTE = "Your bill is processed in memory and not stored."
THEME = getattr(getattr(st.context, "theme", None), "type", None) or "light"
# Results fade/rise once: only on the run right after processing (flag popped), never on later reruns.
st.html(ui_styles.css(THEME, animate_results=ss.pop("animate_results", False)))

try:
    LIMITS = load_limits()
except ValueError as error:  # message names the setting only
    st.error(f"Configuration error: {error}")
    st.stop()
ss.setdefault("bills_used", 0)  # successful new extractions this session (cached samples don't count)

with st.sidebar, st.expander("System stats", expanded=False, icon=":material/monitoring:"):
    snap = stats.snapshot()
    st.caption("This server since its last restart. Counts and timings only; no bill content.")
    st.markdown(f"Extractions **{snap['extractions']}** · Chat calls **{snap['chat_calls']}** · "
                f"Cache hits **{snap['cache_hits']}** · Tokens **{snap['tokens']['total']:,}**")
    for kind in ("extraction", "chat"):
        lat = snap["latency_s"][kind]
        st.caption(f"{kind.capitalize()} latency: p50 {lat['p50'] if lat['p50'] is not None else '–'} s · "
                   f"p95 {lat['p95'] if lat['p95'] is not None else '–'} s ({lat['n']} calls)")
    st.caption("Failures: " + (", ".join(f"{k} {v}" for k, v in sorted(snap["failures"].items())) or "none")
               + f" · fallbacks {snap['fallbacks']} · limit blocks {snap['limit_blocks']}")
    st.caption(f"Today: {limits.extractions.count}/{LIMITS.daily_extraction_cap} extractions, "
               f"{limits.chats.count}/{LIMITS.daily_chat_cap} chat questions")

with st.container(key="brand"):
    st.title("BillAsaan", anchor=False)
    with st.container(key="tagline_row"):
        st.html('<div class="ba-tagline"><p>Bijli ka bill, ab aasaan.</p></div>')
    st.caption("Upload a photo or PDF of your electricity bill to see what you owe, when it's due, and ask about any "
               "line on it.")

# ---------- privacy + choose a bill ----------
if LIMITS.free_tier:
    with st.container(key="consent_box"):
        st.caption(CONSENT_NOTE)
        consent = st.checkbox(CONSENT, key="consent")
else:
    st.caption(PAID_NOTE)
    consent = True
samples = viewer.verified_samples()
upload = st.file_uploader("Upload your bill", type=viewer.UPLOAD_TYPES, max_upload_size=8,
                          on_change=_use_upload, key="upload", disabled=not consent,
                          help="JPG, PNG, PDF (first page) or HEIC, up to 8 MB. Kept in memory for this session only.")
if samples:
    st.button("Try a sample bill", icon=":material/description:", on_click=_use_sample, disabled=not consent)

data = mime = key = None
if not consent:
    pass  # nothing is read or sent until the notice is accepted
elif ss.use_sample and samples:
    data, mime = samples[0]["data"], samples[0]["mime"]
elif upload is not None:
    try:
        data, mime = viewer.prepare_upload(upload.name, upload.getvalue())
    except ValueError as error:
        st.error(str(error))
if data is not None:
    key = viewer.image_key(data)

is_sample = bool(ss.use_sample and samples and consent)
needs_api = key is not None and key not in ss.results and not (is_sample and limits.cached_sample(key))
block = None  # why Process is unavailable for this bill, if a limit applies
if needs_api and limits.bills_left(ss, LIMITS.max_bills_per_session) == 0:
    block = limits.BILLS_LIMIT.format(n=LIMITS.max_bills_per_session)
elif needs_api and limits.extractions.remaining(LIMITS.daily_extraction_cap) == 0:
    block = limits.DAILY_LIMIT
st.button("Process", type="primary", icon=":material/bolt:", on_click=_process, args=(key,),
          disabled=key is None or ss.busy is not None or key in ss.results or block is not None)
if block:
    st.warning(block, icon=":material/block:")


def _extract_once(image: bytes, image_mime: str):
    """One extraction attempt on the current model; telemetry records it whatever the outcome."""
    meta: dict = {}
    try:
        result = viewer.process_image(image, image_mime, meta_out=meta)
    except ProcessingFailure as failure:
        stats.record_call("extraction", meta, failure.diagnostic["category"])
        raise
    stats.record_call("extraction", meta)
    return result


# ---------- at most one API call (plus one quota fallback) per new image ----------
if ss.busy is not None:
    busy_key = ss.busy
    try:
        if is_sample and (cached := limits.cached_sample(busy_key)) is not None:
            ss.results[busy_key] = cached  # reviewed sample already processed on this server: no API call
            ss.animate_results = True
            stats.bump("cache_hits")
        elif limits.bills_left(ss, LIMITS.max_bills_per_session) == 0:
            ss.errors[busy_key] = limits.BILLS_LIMIT.format(n=LIMITS.max_bills_per_session)
            stats.bump("limit_blocks")
        elif not limits.extractions.try_acquire(LIMITS.daily_extraction_cap):
            ss.errors[busy_key] = limits.DAILY_LIMIT
            stats.bump("limit_blocks")
        else:
            with st.status("Reading bill…", expanded=False) as progress:
                log.info("process image=%s: calling extraction API", busy_key[:12])
                outcome, fell_back, exhausted = limits.with_fallback(lambda: _extract_once(data, mime))
                if fell_back:
                    stats.bump("fallbacks")
                if exhausted:  # 429/quota on both models: never cached, never a raw error
                    ss.errors[busy_key] = limits.DAILY_LIMIT
                    progress.update(label="Daily limit reached", state="error")
                else:
                    progress.update(label="Checking numbers…")
                    ss.results[busy_key] = viewer.build_view(outcome)
                    ss.bills_used += 1
                    ss.animate_results = True  # one entrance animation on the next run only
                    if is_sample:
                        limits.store_sample(busy_key, ss.results[busy_key])
                    progress.update(label="Done", state="complete")
    except Exception as error:  # noqa: BLE001 - only sanitized categories reach the UI or log
        failure = error if isinstance(error, ProcessingFailure) else ProcessingFailure(safe_failure(error, "processing"))
        ss.errors[busy_key] = str(failure)
        log.warning("processing_failed %s", json.dumps(failure.diagnostic))
    finally:
        ss.busy = None
    st.rerun()

if key is None:
    if not samples:
        st.info("Upload a bill photo or PDF, then select **Process**.")
    st.stop()

if key in ss.errors:
    st.error(ss.errors[key])
    st.button("Retry", icon=":material/refresh:", on_click=_process, args=(key,))

def _usage_chart(result):
    """Bars in date order; this bill's month in the accent colour, earlier months muted."""
    import altair as alt
    t = ui_styles.tokens(THEME)
    rows = viewer.chart_rows(result)
    order = [r["Label"] for r in rows]
    values = [{"Label": r["Label"], "Units": r["Units"], "Bar": r["Bar"]} for r in rows]
    return (alt.Chart(alt.Data(values=values))
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("Label:O", sort=order, title=None,
                            axis=alt.Axis(labelAngle=0, labelOverlap="greedy", ticks=False, domain=False)),
                    y=alt.Y("Units:Q", title="Units (kWh)", axis=alt.Axis(tickCount=4, domain=False)),
                    color=alt.Color("Bar:N", legend=None,
                                    scale=alt.Scale(domain=["Earlier", "This bill"], range=[t["bar"], t["accent"]])),
                    tooltip=[alt.Tooltip("Label:O", title="Month"), alt.Tooltip("Units:Q", title="Units", format=",")])
            .properties(height=220, padding={"left": 4, "right": 18, "top": 6, "bottom": 4}))


view = ss.results.get(key)
img_col, stats_col = st.columns([2, 3], gap="large")
with img_col, st.container(key="billimg"):
    st.image(data, alt="Your uploaded electricity bill", width="stretch")

with stats_col:
    if view is None:
        if key not in ss.errors:
            st.info("Select **Process** to read this bill.")
        st.stop()
    log.info("image=%s served from session cache (0 API calls)", key[:12])
    if not view["is_bill"]:
        st.warning(viewer.NOT_A_BILL, icon=":material/image_not_supported:")
        st.stop()

    bill, result = view["result"].level1, view["result"]
    with st.container(key="results"):
        with st.container(key="hero"):  # the two numbers people open a bill for: amount, then when it's due
            st.metric("Payable by due date", viewer.money(bill.payable_within_due_date))
            with st.container(key="hero_date"):
                st.metric("Due date", viewer.date(bill.due_date))
        with st.container(key="facts"):
            facts = [("Provider", viewer.show(bill.provider)), ("Bill month", viewer.month(bill.bill_month)),
                     ("Units used", viewer.units(bill.units_consumed)),
                     ("Payable after due date", viewer.money(bill.payable_after_due_date))]
            for row in (facts[:2], facts[2:]):
                for col, (name, value) in zip(st.columns(2), row):
                    col.metric(name, value, border=True)

        st.subheader("Charges and taxes", anchor=False)
        items = viewer.line_items(result)
        if items:
            st.dataframe(items, hide_index=True, width="stretch",
                         height="auto" if len(items) <= 10 else 36 * 10 + 38)
        else:
            st.caption(f"Charges and taxes: {viewer.NOT_SHOWN.lower()}.")

        if viewer.history_rows(result):
            st.subheader("Usage history", anchor=False)
            st.altair_chart(_usage_chart(result), width="stretch")
            st.caption("Units per month in date order; this bill's month is highlighted.")

        st.subheader("Checks", anchor=False)
        st.caption("A passing check finds no inconsistency; it doesn't prove every value is right.")
        with st.container(key="checks", border=True):
            for i, check in enumerate(view["checks"]):
                color, icon = BADGE.get(check["status"], BADGE["Insufficient Data"])
                if i:
                    st.divider()
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    st.badge(check["status"], color=color, icon=icon)
                    st.markdown(f"**{CHECK_LABELS.get(check['name'], check['name'])}**")
                if check["status"] != "Matched" and check.get("reason"):
                    st.caption(check["reason"])  # plain-English why, with the bill's numbers

# ---------- chat with the bill (text only: no image is re-sent) ----------
import chat  # noqa: E402 - only needed once a valid bill is shown

ss.setdefault("chats", {})          # image sha256 -> [{"role", "content", "kind", "is_estimate", "method"}]
ss.setdefault("chat_pending", None)  # (image sha256, question) while an answer is being generated


def _ask(bill_key: str, question: str):
    question = (question or "").strip()[:chat.MAX_QUESTION_CHARS]
    over = limits.questions_left(ss.chats.get(bill_key, []), LIMITS.max_questions_per_bill) == 0
    if question and ss.chat_pending is None and not over:
        ss.chats.setdefault(bill_key, []).append({"role": "user", "content": question})
        ss.chat_pending = (bill_key, question)


def _submit(bill_key: str):
    _ask(bill_key, ss.get("chat_q"))


st.divider()
st.subheader("Ask about your bill", anchor=False)
messages = ss.chats.setdefault(key, [])
answering = ss.chat_pending is not None
chat_block = None
if limits.questions_left(messages, LIMITS.max_questions_per_bill) == 0:
    chat_block = limits.QUESTIONS_LIMIT.format(n=LIMITS.max_questions_per_bill)
elif limits.chats.remaining(LIMITS.daily_chat_cap) == 0:
    chat_block = limits.DAILY_LIMIT
with st.container(horizontal=True, key="chips", gap="small"):
    for suggestion in chat.SUGGESTIONS:
        st.button(suggestion, on_click=_ask, args=(key, suggestion), disabled=answering or chat_block is not None)

for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"], width="stretch")  # plain text (no markdown); full width so RTL can align
        elif message.get("kind") == "failed":
            st.error(message["content"], icon=":material/error:")
        elif message.get("kind") in ("invalid", "limit"):
            st.warning(message["content"], icon=":material/help:")
        else:
            st.markdown(message["content"])
            if message.get("is_estimate"):
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    st.badge("Estimate", color="gray", icon=":material/info:")
                    if message.get("method"):
                        st.caption(message["method"])

if answering and ss.chat_pending[0] == key:
    question = ss.chat_pending[1]

    def _chat_once():
        out = chat.ask(result, messages[:-1], question)
        stats.record_call("chat", out.get("meta"), (out.get("meta", {}).get("failure") or {}).get("category")
                          if out.get("kind") == "failed" else None)
        return out

    if not limits.chats.try_acquire(LIMITS.daily_chat_cap):
        reply = {"kind": "limit", "content": limits.DAILY_LIMIT, "is_estimate": False, "method": ""}
        stats.bump("limit_blocks")
    else:
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            reply, fell_back, exhausted = limits.with_fallback(_chat_once)
        if fell_back:
            stats.bump("fallbacks")
        if exhausted:  # 429/quota on both models
            reply = {"kind": "limit", "content": limits.DAILY_LIMIT, "is_estimate": False, "method": ""}
    messages.append({k: reply[k] for k in ("kind", "content", "is_estimate", "method")} | {"role": "assistant"})
    ss.chat_pending = None
    st.rerun()
elif answering:  # pending question belongs to a bill no longer shown
    ss.chat_pending = None

if chat_block and not answering:
    st.info(chat_block, icon=":material/block:")
st.chat_input("Ask anything about this bill", max_chars=chat.MAX_QUESTION_CHARS,
              disabled=answering or chat_block is not None,
              key="chat_q", on_submit=_submit, args=(key,))
