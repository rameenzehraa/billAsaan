"""D3 offline tests: session limits, daily caps, quota fallback, consent gating, telemetry. No real API calls."""
import io
import os
import threading
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors
from PIL import Image
from streamlit.testing.v1 import AppTest

import chat
import limits
from config import DEFAULT_MODEL, FALLBACK_MODEL, load_limits, load_settings
from failures import ProcessingFailure
from schema import ExtractionResult, InternalFacts, Level1Bill
from telemetry import stats

APP = str(Path(__file__).resolve().parents[1] / "app.py")
ENV = {"API_KEY": "offline-placeholder", "GEMINI_API_KEY": "", "GENAI_IPV4_ONLY": "0", "MODEL_NAME": DEFAULT_MODEL,
       "FREE_TIER": "false", "MAX_BILLS_PER_SESSION": "3", "MAX_QUESTIONS_PER_BILL": "15",
       "DAILY_EXTRACTION_CAP": "150", "DAILY_CHAT_CAP": "400"}
QUOTA = {"category": "quota", "exception_type": "ClientError", "status": 429, "cause_types": ["ClientError"]}


def image_bytes(color="white"):
    out = io.BytesIO()
    Image.new("RGB", (32, 48), color).save(out, "PNG")
    return out.getvalue()


def fixture():
    return ExtractionResult(level1=Level1Bill(provider="LESCO", bill_month="2026-08", units_consumed=65,
                                              payable_within_due_date=44912, payable_after_due_date=45263),
                            internal=InternalFacts())


class Upload(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.name = "synthetic.png"


def quota_error():
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})


def button(app, label):
    return next(b for b in app.button if b.label == label)


class Base(unittest.TestCase):
    env = {}

    def setUp(self):
        for p in (patch.dict(os.environ, ENV | self.env),
                  patch("google.genai.models.Models.generate_content", side_effect=AssertionError("API forbidden"))):
            p.start()
            self.addCleanup(p.stop)
        limits.extractions.reset()
        limits.chats.reset()
        limits._sample_results.clear()
        stats.reset()

    def app(self, uploader, **patches):
        started = [patch("streamlit.file_uploader", uploader)] + [patch(k, v) for k, v in patches.items()]
        for p in started:
            p.start()
            self.addCleanup(p.stop)
        return AppTest.from_file(APP, default_timeout=30).run()


# ---------- daily counter ----------

class DailyCounterTests(unittest.TestCase):
    def test_boundary_and_midnight_reset(self):
        day = [date(2026, 11, 9)]
        counter = limits.DailyCounter(clock=lambda: day[0])
        self.assertTrue(counter.try_acquire(2))
        self.assertTrue(counter.try_acquire(2))
        self.assertFalse(counter.try_acquire(2))  # the cap itself is the last allowed call
        self.assertEqual(counter.remaining(2), 0)
        day[0] = date(2026, 11, 10)                # local midnight passed
        self.assertEqual(counter.remaining(2), 2)
        self.assertTrue(counter.try_acquire(2))
        self.assertEqual(counter.count, 1)

    def test_zero_cap_blocks_everything(self):
        self.assertFalse(limits.DailyCounter().try_acquire(0))

    def test_thread_safe(self):
        counter, granted = limits.DailyCounter(), []
        lock = threading.Lock()

        def worker():
            for _ in range(20):
                if counter.try_acquire(100):
                    with lock:
                        granted.append(1)
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(granted), 100)
        self.assertEqual(counter.count, 100)

    def test_karachi_midnight(self):
        self.assertEqual(limits.PKT.utcoffset(None).total_seconds(), 5 * 3600)
        late_utc = datetime(2026, 11, 9, 19, 30, tzinfo=timezone.utc)  # 00:30 on 10 Nov in Karachi
        self.assertEqual(late_utc.astimezone(limits.PKT).date(), date(2026, 11, 10))


class ConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        with patch.dict(os.environ, {k: "" for k in ENV if k != "API_KEY"} | {"API_KEY": "offline-placeholder"}):
            lim = load_limits()
        self.assertEqual((lim.free_tier, lim.max_bills_per_session, lim.max_questions_per_bill,
                          lim.daily_extraction_cap, lim.daily_chat_cap), (True, 3, 15, 150, 400))
        for name, value in [("FREE_TIER", "maybe"), ("DAILY_CHAT_CAP", "-1"), ("MAX_BILLS_PER_SESSION", "two")]:
            with self.subTest(name=name), patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                load_limits()


# ---------- quota fallback (unit) ----------

class FallbackTests(Base):
    def test_429_then_fallback_success(self):
        models = []

        def call():
            models.append(load_settings().model_name)
            if len(models) == 1:
                raise ProcessingFailure(QUOTA)
            return "ok"
        self.assertEqual(limits.with_fallback(call), ("ok", True, False))
        self.assertEqual(models, [DEFAULT_MODEL, FALLBACK_MODEL])
        self.assertEqual(load_settings().model_name, DEFAULT_MODEL)  # override does not leak

    def test_429_on_both(self):
        outcome, fell_back, exhausted = limits.with_fallback(lambda: (_ for _ in ()).throw(ProcessingFailure(QUOTA)))
        self.assertTrue(fell_back and exhausted)
        self.assertIsInstance(outcome, ProcessingFailure)

    def test_other_failures_do_not_fall_back(self):
        net = ProcessingFailure(QUOTA | {"category": "network", "status": None})
        calls = MagicMock(side_effect=net)
        with self.assertRaises(ProcessingFailure):
            limits.with_fallback(calls)
        calls.assert_called_once()

    def test_chat_reply_quota_falls_back(self):
        replies = iter([{"kind": "failed", "meta": {"failure": {"category": "quota"}}}, {"kind": "answer", "meta": {}}])
        outcome, fell_back, exhausted = limits.with_fallback(lambda: next(replies))
        self.assertEqual((outcome["kind"], fell_back, exhausted), ("answer", True, False))


# ---------- app: 429 through the real extract / chat code paths ----------

class QuotaAppTests(Base):
    def test_extraction_429_fallback_success(self):
        seen = []
        ok_json = fixture().model_dump_json()

        def generate(model, contents, config):
            seen.append(model)
            if model == DEFAULT_MODEL:
                raise quota_error()
            return SimpleNamespace(text=ok_json, usage_metadata=None, function_calls=None,
                                   candidates=[SimpleNamespace(finish_reason=None, content=None)])
        client = MagicMock()
        client.models.generate_content.side_effect = generate
        app = self.app(MagicMock(return_value=Upload(image_bytes())), **{
            "extract._client": MagicMock(return_value=client), "extract.MAX_RETRIES": 0})
        button(app, "Process").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(seen, [DEFAULT_MODEL, FALLBACK_MODEL])
        self.assertEqual(len(app.metric), 6)
        self.assertFalse(app.error)
        snap = stats.snapshot()
        self.assertEqual((snap["extractions"], snap["fallbacks"], snap["failures"]), (2, 1, {"quota": 1}))

    def test_extraction_429_on_both_shows_limit_and_caches_nothing(self):
        client = MagicMock()
        client.models.generate_content.side_effect = quota_error()
        with patch("cache.put") as cache_put, patch("extract._dump") as dump:
            app = self.app(MagicMock(return_value=Upload(image_bytes())), **{
                "extract._client": MagicMock(return_value=client), "extract.MAX_RETRIES": 0})
            button(app, "Process").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertEqual([e.value for e in app.error], [limits.DAILY_LIMIT])
        self.assertNotIn("429", app.error[0].value)
        self.assertEqual(app.session_state["results"], {})
        cache_put.assert_not_called()
        dump.assert_not_called()
        self.assertEqual(app.session_state["bills_used"], 0)

    def test_chat_429_on_both_models(self):
        client = MagicMock()
        client.models.generate_content.side_effect = quota_error()
        app = self.app(MagicMock(return_value=Upload(image_bytes())),
                       **{"viewer.process_image": MagicMock(return_value=fixture()),
                          "chat._client": MagicMock(return_value=client), "extract.MAX_RETRIES": 0})
        button(app, "Process").click().run()
        app.chat_input[0].set_value("How much are taxes?").run()
        self.assertFalse(app.exception)
        models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
        self.assertEqual(models, [DEFAULT_MODEL, FALLBACK_MODEL])
        self.assertIn(limits.DAILY_LIMIT, [w.value for w in app.warning])
        self.assertEqual(limits.chats.count, 1)  # one question = one unit, fallback included


# ---------- session limits ----------

class SessionLimitTests(Base):
    env = {"MAX_BILLS_PER_SESSION": "1", "MAX_QUESTIONS_PER_BILL": "1"}

    def test_bills_per_session(self):
        uploader = MagicMock(return_value=Upload(image_bytes()))
        process = MagicMock(return_value=fixture())
        app = self.app(uploader, **{"viewer.process_image": process})
        button(app, "Process").click().run()
        uploader.return_value = Upload(image_bytes("black"))
        app.run()
        self.assertTrue(button(app, "Process").disabled)
        self.assertIn(limits.BILLS_LIMIT.format(n=1), [w.value for w in app.warning])
        process.assert_called_once()
        uploader.return_value = Upload(image_bytes())  # the processed bill is still viewable
        app.run()
        self.assertEqual(len(app.metric), 6)

    def test_questions_per_bill(self):
        reply = {"kind": "answer", "content": "Taxes are shown.", "is_estimate": False, "method": "", "meta": {}}
        ask = MagicMock(return_value=reply)
        app = self.app(MagicMock(return_value=Upload(image_bytes())),
                       **{"viewer.process_image": MagicMock(return_value=fixture()), "chat.ask": ask})
        button(app, "Process").click().run()
        self.assertFalse(app.chat_input[0].disabled)
        app.chat_input[0].set_value("How much are taxes?").run()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(all(button(app, s).disabled for s in chat.SUGGESTIONS))
        self.assertIn(limits.QUESTIONS_LIMIT.format(n=1), [i.value for i in app.info])
        ask.assert_called_once()


class DailyCapAppTests(Base):
    env = {"DAILY_EXTRACTION_CAP": "1", "DAILY_CHAT_CAP": "1"}

    def test_second_upload_hits_daily_extraction_cap(self):
        uploader = MagicMock(return_value=Upload(image_bytes()))
        process = MagicMock(return_value=fixture())
        app = self.app(uploader, **{"viewer.process_image": process})
        button(app, "Process").click().run()
        uploader.return_value = Upload(image_bytes("black"))
        app.run()
        self.assertTrue(button(app, "Process").disabled)
        self.assertIn(limits.DAILY_LIMIT, [w.value for w in app.warning])
        process.assert_called_once()
        # the cap is server-wide: a brand-new session is blocked too
        other = AppTest.from_file(APP, default_timeout=30).run()
        self.assertTrue(button(other, "Process").disabled)

    def test_daily_chat_cap(self):
        reply = {"kind": "answer", "content": "Taxes are shown.", "is_estimate": False, "method": "", "meta": {}}
        ask = MagicMock(return_value=reply)
        app = self.app(MagicMock(return_value=Upload(image_bytes())),
                       **{"viewer.process_image": MagicMock(return_value=fixture()), "chat.ask": ask})
        button(app, "Process").click().run()
        app.chat_input[0].set_value("How much are taxes?").run()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn(limits.DAILY_LIMIT, [i.value for i in app.info])
        ask.assert_called_once()

    def test_cached_sample_works_after_cap(self):
        sample = {"label": "Sample", "data": image_bytes("blue"), "mime": "image/png", "id": "s"}
        process = MagicMock(return_value=fixture())
        first = self.app(MagicMock(return_value=None),
                         **{"viewer.verified_samples": MagicMock(return_value=[sample]), "viewer.process_image": process})
        button(first, "Try a sample bill").click().run()
        button(first, "Process").click().run()  # uses the only daily extraction
        self.assertEqual(limits.extractions.remaining(1), 0)
        second = AppTest.from_file(APP, default_timeout=30).run()  # another visitor, cap exhausted
        button(second, "Try a sample bill").click().run()
        self.assertFalse(button(second, "Process").disabled)
        button(second, "Process").click().run()
        self.assertEqual(len(second.metric), 6)
        process.assert_called_once()
        self.assertEqual(stats.snapshot()["cache_hits"], 1)


# ---------- consent ----------

class ConsentTests(Base):
    def test_free_tier_requires_consent(self):
        uploader = MagicMock(return_value=Upload(image_bytes()))
        process = MagicMock(return_value=fixture())
        with patch.dict(os.environ, {"FREE_TIER": "true"}):
            app = self.app(uploader, **{"viewer.process_image": process})
            self.assertIn("Gemini free tier", app.checkbox[0].label)
            self.assertIn("Cover your name, address and reference numbers", app.checkbox[0].label)
            self.assertTrue(uploader.call_args.kwargs["disabled"])
            self.assertTrue(button(app, "Process").disabled)  # nothing is read before consent
            app.checkbox[0].check().run()
            self.assertFalse(uploader.call_args.kwargs["disabled"])
            button(app, "Process").click().run()
            process.assert_called_once()

    def test_paid_tier_shows_note_only(self):
        uploader = MagicMock(return_value=Upload(image_bytes()))
        app = self.app(uploader)
        self.assertFalse(app.checkbox)
        self.assertIn("Your bill is processed in memory and not stored.", [c.value for c in app.caption])
        self.assertFalse(uploader.call_args.kwargs["disabled"])


# ---------- telemetry ----------

class TelemetryTests(Base):
    def test_counts_and_no_content(self):
        reply = {"kind": "answer", "content": "SECRET ANSWER", "is_estimate": False, "method": "",
                 "meta": {"latency_s": 2.0, "input_tokens": 100, "output_tokens": 10}}
        app = self.app(MagicMock(return_value=Upload(image_bytes())),
                       **{"viewer.process_image": MagicMock(return_value=fixture()),
                          "chat.ask": MagicMock(return_value=reply)})
        button(app, "Process").click().run()
        app.chat_input[0].set_value("SECRET QUESTION").run()
        snap = stats.snapshot()
        self.assertEqual((snap["extractions"], snap["chat_calls"], snap["tokens"]["total"]), (1, 1, 110))
        self.assertEqual(snap["latency_s"]["chat"]["p50"], 2.0)
        self.assertNotIn("SECRET", str(snap))
        sidebar = " ".join(str(getattr(e, "value", "")) for e in app.sidebar)
        self.assertIn("Chat calls **1**", " ".join(m.value for m in app.sidebar.markdown))
        self.assertNotIn("SECRET", sidebar)

    def test_percentiles(self):
        stats.reset()
        for value in range(1, 21):
            stats.record_call("extraction", {"latency_s": value})
        lat = stats.snapshot()["latency_s"]["extraction"]
        self.assertEqual((lat["p50"], lat["p95"], lat["n"]), (10.0, 19.0, 20))


if __name__ == "__main__":
    unittest.main()
