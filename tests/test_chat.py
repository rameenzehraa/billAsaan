"""Offline chat tests: calculator whitelist, tool loop cap, re-ask, failures, UI. The model is always mocked."""
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types
from PIL import Image
from streamlit.testing.v1 import AppTest

import chat
from schema import ExtractionResult, InternalFacts, LatePayment, Level1Bill, UsageMonth

APP = str(Path(__file__).resolve().parents[1] / "app.py")
OFFLINE_ENV = {"API_KEY": "offline-placeholder", "GEMINI_API_KEY": "", "GENAI_IPV4_ONLY": "0",
               "FREE_TIER": "false"}  # D3: consent gating has its own tests (test_limits.py)


def bill():
    """LESCO_0006-shaped values (no identifiers exist in the schema)."""
    return ExtractionResult(
        level1=Level1Bill(provider="LESCO", bill_month="2026-08", due_date="2026-09-09", units_consumed=65,
                          previous_reading=22238, current_reading=22303, total_charges=3777.49, total_taxes=1723.51,
                          current_bill=5501, arrears=39411, payable_within_due_date=44912,
                          payable_after_due_date=45263),
        internal=InternalFacts(usage_history=[UsageMonth(month="2026-07", units=225),
                                              UsageMonth(month="2026-06", units=350)],
                               late_payment_amounts=[LatePayment(label="after due date", amount=45263)]))


def text_resp(payload):
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(text=raw, function_calls=None, usage_metadata=None,
                           candidates=[SimpleNamespace(content=None, finish_reason=None)])


def tool_resp(*expressions):
    calls = [types.FunctionCall(id=f"c{i}", name="calculate", args={"expression": e}) for i, e in enumerate(expressions)]
    content = types.Content(role="model", parts=[types.Part(function_call=c) for c in calls])
    return SimpleNamespace(text=None, function_calls=calls, usage_metadata=None,
                           candidates=[SimpleNamespace(content=content, finish_reason=None)])


def answer(text, estimate=False, method="printed amounts"):
    return {"answer": text, "values_used": [], "is_estimate": estimate, "method": method}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.configs, self.contents = [], []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        self.contents.append(list(contents))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Base(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, OFFLINE_ENV)
        env.start()
        self.addCleanup(env.stop)
        guard = patch("google.genai.models.Models.generate_content", side_effect=AssertionError("API forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def run_chat(self, responses, question="What if I pay late?", history=None):
        client = FakeClient(responses)
        with patch("chat._client", return_value=client), patch("extract.MAX_RETRIES", 0):
            return chat.ask(bill(), history or [], question), client


class CalculatorTests(unittest.TestCase):
    def test_allowed(self):
        for expr, value in [("45263 - 44912", 351), ("round((225 + 350) / 2, 1)", 287.5), ("sum([1, 2, 3])", 6),
                            ("sum(1, 2, 3)", 6), ("min(4, 2)", 2), ("max(4, 2)", 4), ("-(2 + 3) * 4", -20),
                            ("1723.51 / 5501 * 100", 31.330849)]:
            with self.subTest(expr=expr):
                self.assertAlmostEqual(chat.calculate(expr), value, places=5)

    def test_rejected(self):
        for expr in ['__import__("os").system("x")', "payable_within_due_date - 1", "x", "(1).real",
                     "().__class__", "2 ** 64", "abs(-1)", "eval('1')", "lambda: 1", "'a' + 'b'", "[1, 2]",
                     "1 if 1 else 2", "1 < 2", "True + 1", "round(1.5, 99)", "1 / 0", "9" * 301, "", "1e300 * 1e300"]:
            with self.subTest(expr=expr[:40]), self.assertRaises(ValueError):
                chat.calculate(expr)

    def test_no_eval_used(self):
        import ast
        tree = ast.parse(Path(chat.__file__).read_text(encoding="utf-8"))
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertFalse(called & {"eval", "exec", "compile"})


class AskTests(Base):
    def test_late_payment_with_calculator(self):
        reply, client = self.run_chat([
            tool_resp("45263 - 44912"),
            text_resp(answer("If you pay after the due date you owe Rs 45,263 instead of Rs 44,912, "
                             "which is Rs 351 more.", method="printed amounts; difference via calculator"))])
        self.assertEqual(reply["kind"], "answer")
        self.assertIn("Rs 351", reply["content"])
        self.assertEqual(reply["meta"]["tool_calls"], 1)
        self.assertEqual(len(client.configs), 2)
        response_part = client.contents[1][-1].parts[0].function_response
        self.assertEqual(response_part.response, {"result": 351})

    def test_context_has_no_image_and_last_six_turns(self):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
        history.append({"role": "assistant", "content": FAILED_TEXT, "kind": "failed"})
        reply, client = self.run_chat([text_resp(answer("Your bill doesn't show that."))], history=history)
        sent = client.contents[0][0]
        self.assertIsInstance(sent, str)
        context = json.loads(sent)
        self.assertEqual(set(context), {"bill", "history", "slabs", "late_payment", "facts", "chat", "question"})
        self.assertEqual([m["content"] for m in context["chat"]], ["m5", "m6", "m7", "m8", "m9"])
        self.assertIn("payable_within_due_date", context["facts"])
        self.assertFalse(any(isinstance(c, (bytes, types.Part)) for c in client.contents[0]))

    def test_tool_loop_cap(self):
        reply, client = self.run_chat([tool_resp("1 + 1")] * 10)
        self.assertLessEqual(reply["meta"]["tool_calls"], chat.MAX_TOOL_CALLS)
        self.assertEqual(reply["kind"], "invalid")
        modes = [c.tool_config.function_calling_config.mode for c in client.configs]
        self.assertEqual(modes[:3], [types.FunctionCallingConfigMode.AUTO] * 3)
        self.assertTrue(all(m == types.FunctionCallingConfigMode.NONE for m in modes[3:]))
        self.assertLessEqual(len(client.configs), 5)

    def test_parallel_calls_beyond_cap_get_error(self):
        reply, client = self.run_chat([tool_resp("1+1", "2+2", "3+3", "4+4"),
                                       text_resp(answer("Your bill doesn't show that."))])
        payloads = [p.function_response.response for p in client.contents[1][-1].parts]
        self.assertEqual(payloads[:3], [{"result": 2}, {"result": 4}, {"result": 6}])
        self.assertIn("error", payloads[3])
        self.assertEqual(reply["kind"], "answer")

    def test_invalid_output_reasks_once_then_answers(self):
        reply, client = self.run_chat([text_resp("not json"), text_resp(answer("Your bill doesn't show a meter serial number."))])
        self.assertEqual(reply["kind"], "answer")
        self.assertTrue(reply["meta"]["reasked"])
        self.assertEqual(len(client.configs), 2)
        self.assertIn("rejected", client.contents[1][-1])

    def test_invalid_twice_gives_fallback(self):
        reply, client = self.run_chat([text_resp("not json"), text_resp({"answer": ""})])
        self.assertEqual(reply["kind"], "invalid")
        self.assertEqual(reply["content"], chat.INVALID)
        self.assertEqual(len(client.configs), 2)

    def test_untraced_number_triggers_reask(self):
        reply, client = self.run_chat([text_resp(answer("Paying late costs Rs 9,876 more.")),
                                       text_resp(answer("Paying late costs Rs 45,263, not Rs 44,912."))])
        self.assertTrue(reply["meta"]["reasked"])
        self.assertIn("9,876", client.contents[1][-1])
        self.assertEqual(reply["kind"], "answer")

    def test_identifier_never_returned(self):
        reply, _ = self.run_chat([text_resp(answer("Your meter number is 0412345678901.")),
                                  text_resp(answer("Your meter number is 0412345678901."))])
        self.assertEqual(reply["kind"], "invalid")
        self.assertNotIn("0412345678901", reply["content"])

    def test_estimate_flag_passed_through(self):
        reply, _ = self.run_chat([text_resp(answer("About 225 units next month.", True, "last month's units"))])
        self.assertTrue(reply["is_estimate"])
        self.assertEqual(reply["method"], "last month's units")

    def test_api_failure_message(self):
        import httpx
        with self.assertLogs("chat", level="INFO") as logs:
            reply, client = self.run_chat([httpx.ConnectError("private details")])
        self.assertEqual(reply["kind"], "failed")
        self.assertEqual(reply["content"], chat.FAILED)
        self.assertEqual(len(client.configs), 1)  # no re-ask after an API failure
        self.assertNotIn("private", "".join(logs.output))
        self.assertIn("failure=network", "".join(logs.output))

    def test_missing_key_fails_without_client(self):
        with patch.dict(os.environ, {"API_KEY": "", "GEMINI_API_KEY": ""}), patch("chat._client") as client:
            reply = chat.ask(bill(), [], "How much are taxes?")
        self.assertEqual(reply["kind"], "failed")
        client.assert_not_called()

    def test_logs_contain_no_content(self):
        with self.assertLogs("chat", level="INFO") as logs:
            self.run_chat([text_resp(answer("Taxes are Rs 1,723.51."))], question="secret question text")
        out = "".join(logs.output)
        self.assertNotIn("secret question", out)
        self.assertNotIn("1,723.51", out)
        for field in ("latency=", "in=", "out=", "tool_calls="):
            self.assertIn(field, out)


FAILED_TEXT = chat.FAILED


def image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (32, 48), "white").save(out, "PNG")
    return out.getvalue()


class Upload(io.BytesIO):
    name = "synthetic.png"


class ChatUiTests(Base):
    def open_app(self, reply):
        self.patches = [patch("streamlit.file_uploader", return_value=Upload(image_bytes())),
                        patch("viewer.process_image", return_value=bill()),
                        patch("chat.ask", side_effect=reply if isinstance(reply, list) else [reply])]
        mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)
        app = AppTest.from_file(APP, default_timeout=30).run()
        next(b for b in app.button if b.label == "Process").click().run()
        return app, mocks[2]

    def test_suggestion_chip_and_estimate(self):
        app, ask = self.open_app({"kind": "answer", "content": "About 225 units.", "is_estimate": True,
                                  "method": "last month's units", "meta": {}})
        labels = [b.label for b in app.button]
        for s in chat.SUGGESTIONS:
            self.assertIn(s, labels)
        next(b for b in app.button if b.label == "What if I pay late?").click().run()
        self.assertFalse(app.exception)
        ask.assert_called_once()
        self.assertEqual(ask.call_args.args[2], "What if I pay late?")
        self.assertEqual(ask.call_args.args[1], [])  # history excludes the current question
        self.assertEqual(len(app.chat_message), 2)
        self.assertIn("last month's units", [c.value for c in app.caption])
        self.assertFalse(app.chat_input[0].disabled)
        self.assertEqual(app.chat_input[0].max_chars, chat.MAX_QUESTION_CHARS) \
            if hasattr(app.chat_input[0], "max_chars") else None

    def test_typed_question_and_failure_is_distinct(self):
        failed = {"kind": "failed", "content": chat.FAILED, "is_estimate": False, "method": "", "meta": {}}
        not_shown = {"kind": "answer", "content": "Your bill doesn't show a meter serial number.",
                     "is_estimate": False, "method": "", "meta": {}}
        app, ask = self.open_app([failed, not_shown])
        app.chat_input[0].set_value("How much are taxes?").run()
        self.assertEqual([e.value for e in app.error], [chat.FAILED])
        app.chat_input[0].set_value("What's my meter's serial number?").run()
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(len(app.error), 1)  # the "doesn't show" answer is not rendered as an error
        self.assertIn("doesn't show", " ".join(m.value for m in app.markdown))

    def test_history_kept_per_bill(self):
        app, _ = self.open_app({"kind": "answer", "content": "Taxes are Rs 1,723.51.", "is_estimate": False,
                                "method": "", "meta": {}})
        app.chat_input[0].set_value("How much are taxes?").run()
        chats = app.session_state["chats"]
        self.assertEqual(len(chats), 1)
        self.assertEqual([m["role"] for m in next(iter(chats.values()))], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
