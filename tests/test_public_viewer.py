"""Offline public-viewer regression checks. No real bills or API calls."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from schema import ExtractionResult, Level1Bill, InternalFacts, UsageMonth
from viewer import MAX_BYTES, apply_result, select_bill, validate_image, verified_samples


def image_bytes(color="white"):
    out = io.BytesIO()
    Image.new("RGB", (32, 48), color).save(out, "PNG")
    return out.getvalue()


def fixture():
    return ExtractionResult(level1=Level1Bill(provider="KE", bill_month="2026-09", units_consumed=100,
        previous_reading=100, current_reading=200, total_charges=1000, total_taxes=100,
        current_bill=1100, arrears=0, payable_within_due_date=1100, payable_after_due_date=1200),
        internal=InternalFacts(usage_history=[UsageMonth(month="2026-08", units=80)]))


class Upload(io.BytesIO):
    name = "synthetic.png"


class ViewerTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(validate_image("bill.png", image_bytes())[1], "image/png")
        for name, data in [("bill.pdf", image_bytes()), ("bill.png", b""),
                           ("bill.png", b"x" * (MAX_BYTES+1)), ("bill.jpg", image_bytes()),
                           ("bill.png", b"not an image")]:
            with self.subTest(name=name, length=len(data)), self.assertRaises(ValueError):
                validate_image(name, data)

    def test_sample_gate(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(verified_samples(root), [])
            data = image_bytes()
            (root/"sample.png").write_bytes(data)
            entry = dict(file="sample.png", label="Synthetic", verified_anonymized=True,
                         reviewed_by="offline test", sha256=hashlib.sha256(data).hexdigest())
            (root/"manifest.json").write_text(json.dumps([entry]))
            self.assertEqual(len(verified_samples(root)), 1)
            entry["sha256"] = "wrong"
            (root/"manifest.json").write_text(json.dumps([entry]))
            self.assertEqual(verified_samples(root), [])
            entry["verified_anonymized"] = False
            (root/"manifest.json").write_text(json.dumps([entry]))
            self.assertEqual(verified_samples(root), [])

    def test_state_corrections_and_reset(self):
        state = {}
        select_bill(state, "a")
        apply_result(state, fixture(), original=True)
        changed = state["corrected"].model_copy(deep=True)
        changed.level1.units_consumed = 125
        apply_result(state, changed)
        self.assertEqual(state["original"].level1.units_consumed, 100)
        self.assertEqual(state["facts"]["units_consumed"]["value"], 125)
        self.assertTrue(any(c["status"] == "Mismatch" for c in state["checks"]))
        state.update(messages=[{"role": "user", "content": "old"}], review_draft="old", chat_input="old")
        select_bill(state, "b")
        self.assertIsNone(state["original"])
        self.assertEqual(state["messages"], [])
        self.assertNotIn("review_draft", state)
        self.assertNotIn("chat_input", state)
        other = {}
        select_bill(other, "a")
        self.assertIsNone(other["original"])

    def test_app_flow(self):
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("viewer.process_image", return_value=fixture()) as process, \
             patch("google.genai.models.Models.generate_content", side_effect=AssertionError("API forbidden")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            process.assert_not_called()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            process.assert_called_once()
            self.assertTrue(app.button[0].disabled)
            app.run()
            process.assert_called_once()
            self.assertTrue(app.chat_input[0].disabled)
            self.assertEqual(app.session_state["messages"], [])
            labels = [h.value for h in app.subheader]
            for label in ["Bill summary", "Charges and taxes", "Usage history", "Checks"]:
                self.assertIn(label, labels)
            draft = fixture().model_dump()
            draft["level1"]["units_consumed"] = 125
            app.text_area[0].set_value(json.dumps(draft))
            next(b for b in app.button if b.label == "Apply corrections").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["corrected"].level1.units_consumed, 125)
            self.assertEqual(app.session_state["original"].level1.units_consumed, 100)
            app.text_area[0].set_value('{"bad": 1}')
            next(b for b in app.button if b.label == "Apply corrections").click().run()
            self.assertTrue(app.error)
            self.assertEqual(app.session_state["corrected"].level1.units_consumed, 125)
            app.session_state["messages"] = [{"role": "user", "content": "old question"}]
            app.radio[0].set_value("Try a sample").run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["original"])
            self.assertEqual(app.session_state["messages"], [])
            self.assertTrue(any("No verified anonymized" in item.value for item in app.info))
            process.assert_called_once()

    def test_invalid_and_failure(self):
        with patch("streamlit.file_uploader", return_value=Upload(b"invalid")), \
             patch("viewer.process_image") as process:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertTrue(app.button[0].disabled)
            process.assert_not_called()
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("viewer.process_image", side_effect=RuntimeError("private contents")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertNotIn("private contents", app.error[0].value)
            self.assertFalse(app.session_state["busy"])
            self.assertFalse(app.button[0].disabled)

    def test_sample_selection_and_new_upload_reset(self):
        sample = {"label": "Synthetic sample", "data": image_bytes(), "mime": "image/png", "id": "synthetic"}
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())) as uploader, \
             patch("viewer.verified_samples", return_value=[sample]), \
             patch("viewer.process_image", return_value=fixture()) as process:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            app.button[0].click().run()
            app.session_state["messages"] = [{"role": "user", "content": "old"}]
            uploader.return_value = Upload(image_bytes("black"))
            app.run()
            self.assertIsNone(app.session_state["original"])
            self.assertEqual(app.session_state["messages"], [])
            process.assert_called_once()
            app.radio[0].set_value("Try a sample").run()
            app.selectbox[0].select(0).run()
            self.assertFalse(app.button[0].disabled)
            self.assertIsNone(app.session_state["original"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIsNotNone(app.session_state["original"])
            self.assertEqual(process.call_count, 2)

    def test_competition_csv_pipeline_with_mock_processing(self):
        import csv
        from run import run_pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bills, templates, output = root / "bills", root / "templates", root / "output"
            bills.mkdir()
            templates.mkdir()
            (bills / "synthetic.png").write_bytes(image_bytes())
            (templates / "level1.csv").write_text("bill_id,json\nsynthetic,\n", encoding="utf-8")
            (templates / "level2.csv").write_text(
                "bill_id,question_id,question,answer\nsynthetic,Q1,What are the units?,\n", encoding="utf-8")
            with patch("run.process_with_answers", return_value=(fixture(), {"error": None},
                       {"Q1": "100 units"}, None)), \
                 patch("google.genai.models.Models.generate_content", side_effect=AssertionError("API forbidden")):
                result = run_pipeline(bills, templates, output)
            self.assertEqual((result["level1_rows"], result["level2_rows"]), (1, 1))
            with (output / "level1.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(json.loads(row["json"])["units_consumed"], 100)
            with (output / "level2.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["answer"], "100 units")

    def test_private_extraction_does_not_dump(self):
        from extract import _parse
        with patch("extract._dump") as dump:
            result, error = _parse("upload", fixture().model_dump_json(), "", private=True)
            self.assertIsNone(error)
            self.assertIsNotNone(result)
            dump.assert_not_called()


if __name__ == "__main__":
    unittest.main()


