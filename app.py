"""BillAsaan public bill explainer. Competition batch workflow remains in run.py."""
import hashlib
import json

import streamlit as st
from pydantic import ValidationError

from schema import ExtractionResult
from viewer import apply_result, process_image, select_bill, validate_image, verified_samples

st.set_page_config(page_title="BillAsaan", page_icon="⚡", layout="wide")
ss = st.session_state
st.title("BillAsaan")
st.caption("Bijli ka bill, ab aasaan. Understand your electricity bill.")
st.caption("JPG or PNG · up to 10 MB. Processing sends the selected image to the existing AI service. "
           "Bill results stay in this session and are not saved to the shared bill cache.")
mode = st.radio("Choose a bill", ["Upload my bill", "Try a sample"], horizontal=True, key="source_mode")
data, mime, identity = None, None, None
if mode == "Upload my bill":
    upload = st.file_uploader("Upload my bill", type=["jpg", "jpeg", "png"], max_upload_size=10)
    if upload is not None:
        try:
            raw = upload.getvalue()
            data, mime = validate_image(upload.name, raw)
            identity = "upload:" + hashlib.sha256(raw).hexdigest()
        except ValueError as error:
            st.error(str(error))
else:
    samples = verified_samples()
    if not samples:
        st.info("No verified anonymized samples are available yet. Upload your own bill to continue.")
    else:
        index = st.selectbox("Anonymized sample", range(len(samples)),
                             format_func=lambda i: samples[i]["label"], index=None)
        if index is not None and isinstance(index, int) and 0 <= index < len(samples):
            sample = samples[index]
            data, mime, identity = sample["data"], sample["mime"], "sample:" + sample["id"]
select_bill(ss, (mode, identity))


def start_processing():
    if identity and ss.original is None and not ss.busy:
        ss.busy = True
        ss.process_error = None


if data:
    with st.expander("Bill preview", expanded=True):
        st.image(data, alt="Preview of your selected bill", width="stretch")
st.button("Process", type="primary", on_click=start_processing,
          disabled=data is None or ss.busy or ss.original is not None)
if ss.busy:
    try:
        with st.spinner("Reading your bill…"):
            apply_result(ss, process_image(data, mime), original=True)
    except Exception:
        ss.process_error = "We could not process this bill. Try again with a clearer image."
    finally:
        ss.busy = False
    st.rerun()
if ss.process_error:
    st.error(ss.process_error)
if ss.corrected is None:
    st.info("Choose an image, then select Process to see its details.")
else:
    with st.expander("Review extracted details"):
        st.caption("Correct fields below using JSON. Missing values use null. Changes apply locally; "
                   "the original extraction stays separate. Include level1 and internal (usage history).")
        with st.expander("Original extraction"):
            st.json(ss.original.model_dump(mode="json"))
        with st.form("review_form"):
            draft = st.text_area("Extracted details and corrections", ss.corrected.model_dump_json(indent=2),
                                 height=350, key=f"review_details_{ss.revision}")
            save = st.form_submit_button("Apply corrections")
        if save:
            try:
                corrected = ExtractionResult.model_validate_json(draft)
                # Pydantic accepts nonfinite floats; bills and calculations must not.
                json.dumps(corrected.model_dump(mode="json"), allow_nan=False)
                apply_result(ss, corrected)
                ss.messages = []
                ss.revision += 1
                st.rerun()
            except (ValidationError, ValueError):
                st.error("Corrections are invalid. Check field names, numeric amounts and date formats. No changes applied.")
        st.caption("Checks and calculations are recomputed after each valid correction.")
    bill = ss.corrected.level1.model_dump(mode="json")
    st.subheader("Bill summary")
    st.dataframe([{"Detail": k.replace("_", " ").capitalize(), "Value": "Not shown" if v is None else str(v)}
                  for k, v in bill.items() if not isinstance(v, list)], hide_index=True, width="stretch")
    st.subheader("Charges and taxes")
    for field in ("charges", "taxes"):
        st.markdown(f"**{field.capitalize()}**")
        if bill[field]:
            st.dataframe(bill[field], hide_index=True, width="stretch")
        else:
            st.caption("No items listed separately.")
    st.subheader("Usage history")
    history = ss.corrected.internal.usage_history
    if history:
        st.dataframe([row.model_dump() for row in history], hide_index=True, width="stretch")
    else:
        st.caption("No usage history was extracted from this bill.")
    st.subheader("Checks")
    st.caption("A passing check finds no inconsistency; it does not prove the extraction is correct.")
    for check in ss.checks:
        with st.container(border=True):
            st.badge(check["status"], color={"Matched": "green", "Mismatch": "red"}.get(check["status"], "gray"))
            st.text(check["name"].replace("_", " ").capitalize())
            st.text(check["detail"])

st.subheader("Ask about this bill")
st.info("Chat integration is pending. Questions cannot be submitted yet; no AI replies are generated.")
for question in ("Why is my bill this amount?", "Which taxes am I paying?", "How has my usage changed?"):
    st.button(question, disabled=True)
for message in ss.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])
st.chat_input("Ask about this bill (integration pending)", disabled=True, key="chat_input")
