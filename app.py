"""BillAsaan public page: upload one electricity bill -> Process -> key stats.

Batch/CSV work stays in run.py (CLI). Uploads live only in memory and session_state.
Run: streamlit run app.py
"""
import json
import logging

import streamlit as st

import viewer
from failures import ProcessingFailure, safe_failure

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


st.title("BillAsaan", anchor=False)
st.caption("Bijli ka bill, ab aasaan. Upload your electricity bill to see its key numbers.")

# ---------- choose a bill ----------
samples = viewer.verified_samples()
upload = st.file_uploader("Upload your bill", type=viewer.UPLOAD_TYPES, max_upload_size=8,
                          on_change=_use_upload, key="upload",
                          help="JPG, PNG, PDF (first page) or HEIC, up to 8 MB. Kept in memory for this session only.")
if samples:
    st.button("Try a sample bill", icon=":material/description:", on_click=_use_sample)

data = mime = key = None
if ss.use_sample and samples:
    data, mime = samples[0]["data"], samples[0]["mime"]
elif upload is not None:
    try:
        data, mime = viewer.prepare_upload(upload.name, upload.getvalue())
    except ValueError as error:
        st.error(str(error))
if data is not None:
    key = viewer.image_key(data)

st.button("Process", type="primary", icon=":material/bolt:", on_click=_process, args=(key,),
          disabled=key is None or ss.busy is not None or key in ss.results)

# ---------- one API call per new image ----------
if ss.busy is not None:
    busy_key = ss.busy
    try:
        with st.spinner("Reading your bill…"):
            log.info("process image=%s: calling extraction API", busy_key[:12])
            ss.results[busy_key] = viewer.build_view(viewer.process_image(data, mime))
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

view = ss.results.get(key)
img_col, stats_col = st.columns([2, 3], gap="large")
with img_col:
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
    cards = [("Provider", viewer.show(bill.provider)), ("Bill month", viewer.month(bill.bill_month)),
             ("Units used", viewer.units(bill.units_consumed)),
             ("Payable by due date", viewer.money(bill.payable_within_due_date)),
             ("Due date", viewer.date(bill.due_date)),
             ("Payable after due date", viewer.money(bill.payable_after_due_date))]
    for row in (cards[:3], cards[3:]):
        for col, (name, value) in zip(st.columns(3), row):
            col.metric(name, value, border=True)

    st.subheader("Charges and taxes", anchor=False)
    items = viewer.line_items(result)
    if items:
        st.dataframe(items, hide_index=True, width="stretch")
    else:
        st.caption(f"Charges and taxes: {viewer.NOT_SHOWN.lower()}.")

    history = viewer.history_rows(result)
    if history:
        st.subheader("Usage history", anchor=False)
        st.bar_chart(history, x="Month", y="Units", y_label="Units (kWh)", sort=False,
                     alt="Monthly units consumed, as printed on the bill")

    st.subheader("Checks", anchor=False)
    st.caption("A passing check finds no inconsistency; it doesn't prove every value is right.")
    for check in view["checks"]:
        color, icon = BADGE.get(check["status"], BADGE["Insufficient Data"])
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.badge(check["status"], color=color, icon=icon)
            st.markdown(f"**{CHECK_LABELS.get(check['name'], check['name'])}**")
