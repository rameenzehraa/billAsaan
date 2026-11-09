"""Offline public-page regression checks. No real bills, no API calls (all providers mocked)."""
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from failures import ProcessingFailure
from schema import ExtractionResult, InternalFacts, Level1Bill, UsageMonth
from viewer import (MAX_BYTES, NOT_A_BILL, NOT_SHOWN, build_view, is_bill, money, prepare_upload, show,
                    verified_samples)

APP = str(Path(__file__).resolve().parents[1] / "app.py")
OFFLINE_ENV = {"API_KEY": "offline-placeholder", "GEMINI_API_KEY": "", "GENAI_IPV4_ONLY": "0"}


def image_bytes(color="white", fmt="PNG"):
    out = io.BytesIO()
    if fmt == "HEIF":
        import pillow_heif
        pillow_heif.register_heif_opener()
    Image.new("RGB", (32, 48), color).save(out, fmt)
    return out.getvalue()


def fixture(**level1):
    fields = dict(provider="LESCO", bill_month="2026-09", due_date="2026-10-05", units_consumed=100,
                  previous_reading=100, current_reading=200, total_charges=1000, total_taxes=100,
                  current_bill=1100, arrears=0, payable_within_due_date=1100, payable_after_due_date=1200,
                  charges=[{"type": "energy", "amount": 1000}], taxes=[{"type": "gst", "amount": 100}])
    fields.update(level1)
    return ExtractionResult(level1=Level1Bill(**fields),
                            internal=InternalFacts(usage_history=[UsageMonth(month="2026-08", units=80)]))


class Upload(io.BytesIO):
    def __init__(self, data, name="synthetic.png"):
        super().__init__(data)
        self.name = name


def button(app, label):
    return next(b for b in app.button if b.label == label)


def all_text(app):
    parts = [m.value for m in app.metric] + [m.value for m in app.markdown] + [c.value for c in app.caption]
    return " ".join(str(p) for p in parts)


class UploadTests(unittest.TestCase):
    def test_accepted_formats_become_png_in_memory(self):
        pdf = image_bytes(fmt="PDF")
        for name, data in [("bill.png", image_bytes()), ("bill.jpg", image_bytes(fmt="JPEG")),
                           ("bill.JPEG", image_bytes(fmt="JPEG")), ("bill.pdf", pdf),
                           ("bill.heic", image_bytes(fmt="HEIF"))]:
            with self.subTest(name=name):
                out, mime = prepare_upload(name, data)
                self.assertEqual(mime, "image/png")
                self.assertEqual(Image.open(io.BytesIO(out)).format, "PNG")

    def test_rejections(self):
        for name, data in [("bill.gif", image_bytes()), ("bill.png", b""), ("bill.png", b"x" * (MAX_BYTES + 1)),
                           ("bill.jpg", image_bytes()), ("bill.png", b"not an image"), ("bill.pdf", b"%PDF-broken")]:
            with self.subTest(name=name, length=len(data)), self.assertRaises(ValueError):
                prepare_upload(name, data)

    def test_upload_never_touches_disk(self):
        with patch("builtins.open", side_effect=AssertionError("disk write")), \
             patch("pathlib.Path.write_bytes", side_effect=AssertionError("disk write")):
            prepare_upload("bill.pdf", image_bytes(fmt="PDF"))

    def test_sample_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(verified_samples(root), [])
            data = image_bytes()
            (root / "sample.png").write_bytes(data)
            entry = dict(file="sample.png", label="Synthetic", verified_anonymized=True,
                         reviewed_by="offline test", sha256=hashlib.sha256(data).hexdigest())
            (root / "manifest.json").write_text(json.dumps([entry]))
            self.assertEqual(len(verified_samples(root)), 1)
            for change in ({"sha256": "wrong"}, {"verified_anonymized": False}):
                (root / "manifest.json").write_text(json.dumps([{**entry, **change}]))
                self.assertEqual(verified_samples(root), [])

    def test_shipped_manifest_is_empty(self):
        self.assertEqual(verified_samples(), [])


class SchemaAndDisplayTests(unittest.TestCase):
    def test_all_discos_and_unknown_provider(self):
        for p in ["IESCO", "LESCO", "KE", "MEPCO", "FESCO", "GEPCO", "PESCO", "HESCO", "SEPCO", "QESCO", "TESCO"]:
            self.assertEqual(Level1Bill(provider=p.lower()).provider, p)
        self.assertEqual(Level1Bill(provider="K-Electric").provider, "KE")
        self.assertEqual(Level1Bill(provider="Some New Utility").provider, "OTHER")
        self.assertIsNone(Level1Bill(provider="").provider)

    def test_null_display_and_bill_detection(self):
        self.assertEqual(show(None), NOT_SHOWN)
        self.assertEqual(show(""), NOT_SHOWN)
        self.assertEqual(money(None), NOT_SHOWN)
        self.assertEqual(money(1100), "Rs 1,100")
        self.assertTrue(is_bill(fixture()))
        self.assertTrue(is_bill(fixture(payable_within_due_date=None, units_consumed=None)))
        self.assertFalse(is_bill(ExtractionResult(level1=Level1Bill())))


class AppTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, OFFLINE_ENV)
        env.start()
        self.addCleanup(env.stop)
        guard = patch("google.genai.models.Models.generate_content", side_effect=AssertionError("API forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def run_app(self, uploader):
        with patch("streamlit.file_uploader", uploader):
            return AppTest.from_file(APP, default_timeout=30).run()

    def test_stats_flow_and_zero_calls_on_rerun(self):
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("viewer.process_image", return_value=fixture()) as process:
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertFalse(app.exception)
            process.assert_not_called()
            self.assertFalse(app.metric)
            button(app, "Process").click().run()
            self.assertFalse(app.exception)
            process.assert_called_once()
            self.assertTrue(button(app, "Process").disabled)
            metrics = {m.label: m.value for m in app.metric}
            self.assertEqual(list(metrics), ["Provider", "Bill month", "Units used", "Payable by due date",
                                             "Due date", "Payable after due date"])
            self.assertEqual(metrics["Payable by due date"], "Rs 1,100")
            self.assertEqual(metrics["Bill month"], "Sep 2026")
            subheaders = [h.value for h in app.subheader]
            self.assertEqual(subheaders, ["Charges and taxes", "Usage history", "Checks"])
            checks = next(iter(app.session_state["results"].values()))["checks"]
            self.assertTrue(checks)
            self.assertTrue({c["status"] for c in checks} <= {"Matched", "Mismatch", "Insufficient Data"})
            for _ in range(3):  # reruns: cached by image hash, zero API calls
                app.run()
            process.assert_called_once()
            self.assertFalse(app.text_area)
            self.assertFalse(app.chat_input)

    def test_nulls_render_as_not_shown(self):
        sparse = fixture(provider=None, due_date=None, payable_after_due_date=None, charges=[], taxes=[])
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("viewer.process_image", return_value=sparse):
            app = AppTest.from_file(APP, default_timeout=30).run()
            button(app, "Process").click().run()
            values = [m.value for m in app.metric]
            self.assertEqual(values.count(NOT_SHOWN), 3)
            self.assertFalse(any(v in ("", "None", None) for v in values))
            self.assertNotIn("None", all_text(app))

    def test_not_a_bill(self):
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes("green"))), \
             patch("viewer.process_image", return_value=ExtractionResult(level1=Level1Bill())):
            app = AppTest.from_file(APP, default_timeout=30).run()
            button(app, "Process").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.warning[0].value, NOT_A_BILL)
            self.assertFalse(app.metric)

    def test_invalid_upload(self):
        with patch("streamlit.file_uploader", return_value=Upload(b"invalid")), \
             patch("viewer.process_image") as process:
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertTrue(button(app, "Process").disabled)
            process.assert_not_called()

    def test_api_failure_shows_message_and_retry(self):
        failure = ProcessingFailure({"category": "network", "exception_type": "ConnectError", "status": None,
                                     "cause_types": ["ConnectError"]})
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("viewer.process_image", side_effect=[failure, fixture()]) as process:
            app = AppTest.from_file(APP, default_timeout=30).run()
            with self.assertLogs("app", level="WARNING") as logs:
                button(app, "Process").click().run()
            self.assertIn("ConnectError", "".join(logs.output))
            self.assertFalse(app.exception)
            self.assertIn("connection", app.error[0].value.lower())
            self.assertNotIn("Traceback", app.error[0].value)
            self.assertFalse(app.metric)
            self.assertIsNone(app.session_state["busy"])
            button(app, "Retry").click().run()
            self.assertEqual(process.call_count, 2)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 6)

    def test_unexpected_error_is_sanitized(self):
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("viewer.process_image", side_effect=RuntimeError("private contents offline-placeholder")):
            app = AppTest.from_file(APP, default_timeout=30).run()
            with self.assertLogs("app", level="WARNING") as logs:
                button(app, "Process").click().run()
            self.assertFalse(app.exception)
            for text in ("".join(logs.output), app.error[0].value):
                self.assertNotIn("private contents", text)
                self.assertNotIn("offline-placeholder", text)
            self.assertTrue(any(b.label == "Retry" for b in app.button))

    def test_connection_failure_through_real_extract(self):
        import httpx
        with patch("streamlit.file_uploader", return_value=Upload(image_bytes())), \
             patch("extract._client") as client, patch("extract.MAX_RETRIES", 0), \
             patch("extract._dump") as dump, patch("cache.put") as cache_write:
            client.return_value.models.generate_content.side_effect = httpx.ConnectError("private details")
            app = AppTest.from_file(APP, default_timeout=30).run()
            button(app, "Process").click().run()
            self.assertFalse(app.exception)
            self.assertIn("connection", app.error[0].value.lower())
            self.assertNotIn("private details", app.error[0].value)
            client.return_value.models.generate_content.assert_called_once()
            dump.assert_not_called()
            cache_write.assert_not_called()

    def test_sample_button_hidden_without_samples(self):
        with patch("streamlit.file_uploader", return_value=None):
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertFalse(any(b.label == "Try a sample bill" for b in app.button))

    def test_sample_button_with_samples(self):
        sample = {"label": "Synthetic", "data": image_bytes(), "mime": "image/png", "id": "s"}
        with patch("streamlit.file_uploader", return_value=None), \
             patch("viewer.verified_samples", return_value=[sample]), \
             patch("viewer.process_image", return_value=fixture()) as process:
            app = AppTest.from_file(APP, default_timeout=30).run()
            button(app, "Try a sample bill").click().run()
            button(app, "Process").click().run()
            self.assertFalse(app.exception)
            process.assert_called_once()
            self.assertEqual(len(app.metric), 6)


class ExtractionFailureTests(unittest.TestCase):
    def test_safe_failure_categories(self):
        import ssl
        import httpx
        from failures import safe_failure
        private = "never reveal this bill or key"

        class ProviderError(Exception):
            def __init__(self, code):
                super().__init__(private)
                self.code = code
        for code, category in [(401, "authentication"), (403, "authentication"),
                               (429, "quota"), (400, "api"), (404, "api"), (503, "api")]:
            with self.subTest(code=code):
                detail = safe_failure(ProviderError(code), "api")
                self.assertEqual(detail["category"], category)
                self.assertNotIn(private, str(detail))
        error = httpx.ConnectError(private)
        error.__cause__ = ssl.SSLEOFError(private)
        self.assertEqual(safe_failure(error, "api")["category"], "network")
        self.assertEqual(safe_failure(TimeoutError(private), "api")["category"], "timeout")

    def test_missing_config_fails_before_provider(self):
        from viewer import process_image
        with patch("config.load_settings", return_value=SimpleNamespace(api_key="")), \
             patch("extract._client") as client:
            with self.assertRaises(ProcessingFailure) as caught:
                process_image(image_bytes(), "image/png")
            self.assertEqual(caught.exception.diagnostic["category"], "configuration")
            client.assert_not_called()

    def test_invalid_model_json_is_validation_failure(self):
        from viewer import process_image
        response = SimpleNamespace(text="invalid private response", usage_metadata=None, candidates=[])
        with patch.dict(os.environ, OFFLINE_ENV), patch("extract._client") as client, \
             patch("extract._dump") as dump:
            client.return_value.models.generate_content.return_value = response
            with self.assertRaises(ProcessingFailure) as caught:
                process_image(image_bytes(), "image/png")
            self.assertEqual(caught.exception.diagnostic["category"], "validation")
            self.assertNotIn("invalid private response", str(caught.exception.diagnostic))
            self.assertEqual(client.return_value.models.generate_content.call_count, 2)  # original + re-ask
            dump.assert_not_called()

    def test_build_view_runs_checks_without_api(self):
        view = build_view(fixture())
        self.assertTrue(view["is_bill"])
        self.assertTrue(view["checks"])

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


if __name__ == "__main__":
    unittest.main()
