"""D4 offline tests: 429 retry policy in extract._call, secrets-first config, startup key error. No API calls."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors
from streamlit.testing.v1 import AppTest

import config
import extract
import limits
from config import DEFAULT_MODEL, FALLBACK_MODEL, load_limits, load_settings
from failures import ProcessingFailure

APP = str(Path(__file__).resolve().parents[1] / "app.py")
ENV = {"API_KEY": "offline-placeholder", "GEMINI_API_KEY": "", "MODEL_NAME": DEFAULT_MODEL, "MODEL_ID": "",
       "GENAI_IPV4_ONLY": "0", "FREE_TIER": "false"}


def api_error(code, quota_id=None, retry_delay=None):
    details = []
    if quota_id:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_requests",
                                        "quotaId": quota_id}]})
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    status = {429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE", 400: "INVALID_ARGUMENT"}[code]
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": "private provider text", "status": status, "details": details}})


DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
OK = SimpleNamespace(text="{}", usage_metadata=None, candidates=[SimpleNamespace(finish_reason=None)])


class RetryPolicyTests(unittest.TestCase):
    def call(self, *outcomes):
        client = MagicMock()
        client.models.generate_content.side_effect = list(outcomes)
        meta = {"attempts": 0, "input_tokens": None, "output_tokens": None}
        with patch("extract.time.sleep") as sleep, patch("extract.MAX_RETRIES", 2):
            resp, err = extract._call(client, DEFAULT_MODEL, ["x"], "test", meta, config=object())
        return resp, err, meta, client.models.generate_content.call_count, [c.args[0] for c in sleep.call_args_list]

    def test_daily_quota_fails_immediately(self):
        resp, err, meta, calls, sleeps = self.call(api_error(429, DAILY, "40s"), OK)
        self.assertIsNone(resp)
        self.assertEqual((calls, sleeps), (1, []))
        self.assertEqual(meta["failure"]["category"], "quota")

    def test_per_minute_one_short_retry_then_success(self):
        resp, err, meta, calls, sleeps = self.call(api_error(429, MINUTE, "3s"), OK)
        self.assertIs(resp, OK)
        self.assertEqual((calls, sleeps), (2, [3.0]))

    def test_per_minute_retry_capped_at_5s_and_only_once(self):
        resp, err, meta, calls, sleeps = self.call(api_error(429, MINUTE, "37s"), api_error(429, MINUTE, "37s"), OK)
        self.assertIsNone(resp)
        self.assertEqual((calls, sleeps), (2, [5.0]))
        self.assertEqual(meta["failure"]["category"], "quota")

    def test_429_without_details_is_treated_as_per_minute(self):
        resp, err, meta, calls, sleeps = self.call(api_error(429), OK)
        self.assertIs(resp, OK)
        self.assertEqual(sleeps, [2.0])
        self.assertLessEqual(max(sleeps), extract.RATE_RETRY_MAX_S)

    def test_other_transient_errors_unchanged(self):
        resp, err, meta, calls, sleeps = self.call(api_error(503), api_error(503), OK)
        self.assertIs(resp, OK)
        self.assertEqual((calls, sleeps), (3, [2, 4]))
        resp, err, meta, calls, sleeps = self.call(api_error(503), api_error(503), api_error(503))
        self.assertIsNone(resp)
        self.assertEqual((calls, sleeps), (3, [2, 4]))

    def test_non_transient_not_retried(self):
        resp, err, meta, calls, sleeps = self.call(api_error(400), OK)
        self.assertIsNone(resp)
        self.assertEqual((calls, sleeps), (1, []))

    def test_error_text_never_in_failure(self):
        _, _, meta, _, _ = self.call(api_error(429, DAILY))
        self.assertNotIn("private provider text", str(meta["failure"]))

    def test_daily_quota_goes_straight_to_fallback_model(self):
        """End to end: daily 429 on the primary -> no same-model retry -> limits fallback on the second model."""
        models = []

        def generate(model, contents, config):
            models.append(model)
            if model == DEFAULT_MODEL:
                raise api_error(429, DAILY)
            return SimpleNamespace(text='{"level1": {"bill_month": "2026-08"}}', usage_metadata=None,
                                   candidates=[SimpleNamespace(finish_reason=None)])
        client = MagicMock()
        client.models.generate_content.side_effect = generate
        from viewer import process_image
        with patch.dict(os.environ, ENV), patch("extract._client", return_value=client), \
             patch("extract.time.sleep") as sleep:
            outcome, fell_back, exhausted = limits.with_fallback(lambda: process_image(b"png", "image/png"))
        self.assertEqual(models, [DEFAULT_MODEL, FALLBACK_MODEL])
        self.assertTrue(fell_back)
        self.assertFalse(exhausted)
        self.assertEqual(outcome.level1.bill_month, "2026-08")
        sleep.assert_not_called()


class SecretsConfigTests(unittest.TestCase):
    def test_secrets_take_precedence_over_env(self):
        secrets = {"API_KEY": "from-secrets", "MODEL_NAME": FALLBACK_MODEL, "FREE_TIER": False,
                   "DAILY_CHAT_CAP": 7, "MAX_BILLS_PER_SESSION": 2}
        with patch.dict(os.environ, ENV | {"DAILY_CHAT_CAP": "400"}), patch("streamlit.secrets", secrets):
            settings, lim = load_settings(), load_limits()
        self.assertEqual(settings.api_key, "from-secrets")
        self.assertEqual(settings.model_name, FALLBACK_MODEL)
        self.assertEqual((lim.free_tier, lim.daily_chat_cap, lim.max_bills_per_session), (False, 7, 2))
        self.assertNotIn("from-secrets", repr(settings))

    def test_env_used_when_secrets_missing_or_unavailable(self):
        broken = MagicMock()
        broken.get.side_effect = FileNotFoundError("no secrets.toml")
        for secrets in ({}, broken):
            with self.subTest(secrets=type(secrets).__name__), patch.dict(os.environ, ENV), \
                 patch("streamlit.secrets", secrets):
                self.assertEqual(load_settings().api_key, "offline-placeholder")

    def test_blank_secret_falls_back_to_env(self):
        with patch.dict(os.environ, ENV), patch("streamlit.secrets", {"API_KEY": "  "}):
            self.assertEqual(load_settings().api_key, "offline-placeholder")

    def test_missing_key_is_one_clear_error_without_values(self):
        with patch.dict(os.environ, ENV | {"API_KEY": ""}), patch("streamlit.secrets", {}):
            with self.assertRaisesRegex(ValueError, "API_KEY is not set") as caught:
                load_limits()
        self.assertIn("Streamlit secrets", str(caught.exception))

    def test_app_startup_error_when_key_missing(self):
        with patch.dict(os.environ, ENV | {"API_KEY": "", "GEMINI_API_KEY": ""}), patch("streamlit.secrets", {}), \
             patch("google.genai.models.Models.generate_content", side_effect=AssertionError("API forbidden")):
            app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("API_KEY is not set", app.error[0].value)
        self.assertFalse(app.button)  # app stops before any input is shown

    def test_secret_value_never_shown_on_bad_setting(self):
        with patch.dict(os.environ, ENV), patch("streamlit.secrets", {"DAILY_CHAT_CAP": "secret-looking-value"}):
            with self.assertRaises(ValueError) as caught:
                load_limits()
        self.assertNotIn("secret-looking-value", str(caught.exception))


class RepoFilesTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_requirements_all_pinned(self):
        lines = [l.split("#")[0].strip() for l in (self.ROOT / "requirements.txt").read_text().splitlines()]
        reqs = [l for l in lines if l]
        self.assertTrue(reqs)
        for line in reqs:
            self.assertRegex(line, r"^[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.\-]*$")

    def test_streamlit_config(self):
        import tomllib
        cfg = tomllib.loads((self.ROOT / ".streamlit" / "config.toml").read_text())
        self.assertEqual(cfg["server"]["maxUploadSize"], 8)
        self.assertIn("primaryColor", cfg["theme"])

    def test_secrets_example_has_placeholders_only(self):
        import tomllib
        example = tomllib.loads((self.ROOT / ".streamlit" / "secrets.toml.example").read_text())
        self.assertEqual(example["API_KEY"], "your-gemini-api-key")
        self.assertEqual(set(example), {"API_KEY", "MODEL_NAME", "FREE_TIER", "MAX_BILLS_PER_SESSION",
                                        "MAX_QUESTIONS_PER_BILL", "DAILY_EXTRACTION_CAP", "DAILY_CHAT_CAP"})


if __name__ == "__main__":
    unittest.main()
