"""Chat about one extracted bill: grounded answers, arithmetic only through a whitelisted calculator tool.

ask() never raises. Per question: one generate_content call plus at most MAX_TOOL_CALLS tool round-trips,
and one re-ask if the final output is invalid. No image is sent. Logs carry numbers only, never content.
"""
import ast
import json
import logging
import operator
import re
import time

from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from calc import compute_facts
from config import TEMPERATURE, load_settings
from extract import _call, _client
from failures import safe_failure
from schema import ExtractionResult

log = logging.getLogger("chat")

MAX_QUESTION_CHARS = 300
MAX_TOOL_CALLS = 3
MAX_HISTORY_MESSAGES = 6
MAX_OUTPUT_TOKENS = 4096
FAILED = "Processing failed, try again."
INVALID = "Sorry, I couldn't produce a reliable answer to that. Please try rephrasing your question."
SUGGESTIONS = ("Why is my bill this amount?", "What if I pay late?",
               "How does my usage compare to past months?", "How much are taxes?")

SYSTEM_PROMPT = """You explain ONE Pakistani electricity bill to its owner. The user message is JSON:
bill (printed fields), history (monthly usage table), slabs (tariff rate lines), late_payment (printed late amounts),
facts (values already computed by code, each with its method), chat (recent turns), question.

Rules:
1. Use ONLY this data. Never use outside knowledge about tariffs, prices or the user. If a value is not in the data,
   say "Your bill doesn't show <that value>." Do not guess it.
2. NEVER do arithmetic yourself, not even simple subtraction. Every number in your answer must be either copied
   exactly from the data, or the result of the calculate tool. If facts already holds the exact value you need
   (e.g. late_surcharge, prev12_avg_units), copy it. Otherwise you MUST call calculate before answering, for any
   difference ("how much more/less"), sum, ratio, percentage or average. Stating a difference that is neither a
   fact nor a calculate result is an error; never present an amount itself as a difference.
3. If the question is ambiguous, answer the most likely meaning and say which meaning you used.
4. If the question is not about electricity or this bill (weather, news, general chat), do NOT say "Your bill doesn't
   show"; reply with one polite line that you can only help with this bill, and suggest a bill question
   (e.g. "I can only help with this bill. Try asking why it is this amount or when it is due.").
5. Reply in the user's language: English, Urdu script, or Roman Urdu (match what they wrote).
6. Never repeat names, addresses, account/reference/consumer/meter numbers or other personal identifiers,
   even if asked; say the app doesn't show them.
7. Be brief and plain: 1 to 4 sentences. Amounts as "Rs 1,234". No markdown tables.
8. When the answer involves two amounts (late vs on-time, this month vs past, taxes vs total), state both AND the
   difference, taken from facts (e.g. late_surcharge, late_surcharge_tier_N) or from calculate. If late payment
   has several printed tiers, give each tier with its date label.

Output JSON only: answer (the reply), values_used (each value you relied on: name and value as text),
is_estimate (true only if the answer is a projection or approximation, not a printed or exactly computed value),
method (one short line: where the numbers came from, e.g. "printed amounts; difference via calculator")."""

REASK = ("Your previous reply was rejected: {error}. Answer the same question again following every rule. "
         "Use calculate for any arithmetic and copy numbers exactly. Output the JSON only.")


class UsedValue(BaseModel):
    name: str
    value: str


class ChatAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)
    values_used: list[UsedValue] = Field(default_factory=list)
    is_estimate: bool = False
    method: str = ""


CALCULATOR = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="calculate",
    description="Evaluate arithmetic exactly. Allowed: numbers, + - * / and parentheses, round(x, n), min(...), "
                "max(...), sum(...). No names or variables: substitute the numbers from the data.",
    parameters_json_schema={"type": "object", "properties": {"expression": {
        "type": "string", "description": "e.g. 45263 - 44912 or round((80 + 92 + 75) / 3, 1)"}},
        "required": ["expression"]},
)])


# ---- Calculator: AST whitelist, never eval() ----

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round", "min", "max", "sum"}


def calculate(expression: str) -> float:
    """Numbers, + - * / ( ), round/min/max/sum. Anything else (names, attributes, imports, **) raises ValueError."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 300:
        raise ValueError("expression must be 1-300 characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ValueError("not a valid arithmetic expression") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("division by zero")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords and node.args):
            name = node.func.id
            if name == "sum":  # sum(a, b, c) or sum([a, b, c])
                items = node.args[0].elts if len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)) \
                    else node.args
                return sum(ev(a) for a in items)
            if name == "round":
                if len(node.args) > 2:
                    raise ValueError("round takes 1 or 2 arguments")
                digits = ev(node.args[1]) if len(node.args) == 2 else 0
                if digits != int(digits) or not 0 <= digits <= 6:
                    raise ValueError("round digits must be 0-6")
                return round(ev(node.args[0]), int(digits))
            return (min if name == "min" else max)(ev(a) for a in node.args)
        raise ValueError(f"not allowed: {type(node).__name__}")

    value = ev(tree)
    if abs(value) > 1e12:
        raise ValueError("result out of range")
    return round(value, 6)


# ---- Context and grounding ----

def build_context(result: ExtractionResult, history: list[dict], question: str) -> dict:
    """Everything the model may use. No image, no free-text flags; only extracted values and code facts."""
    internal = result.internal
    facts = compute_facts(result)
    return {
        "bill": result.level1.model_dump(mode="json"),
        "history": [m.model_dump(mode="json") for m in internal.usage_history],
        "slabs": [r.model_dump(mode="json") for r in internal.rates],
        "late_payment": [p.model_dump(mode="json") for p in internal.late_payment_amounts],
        "facts": {k: {"value": v["value"], "method": v["method"]} for k, v in facts.items()},
        "chat": [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY_MESSAGES:]
                 if m.get("kind") != "failed"],
        "question": question,
    }


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return {float(obj)}
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _numbers(v)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.add(float(m.replace(",", "")))
            except ValueError:
                pass
    return out


def untraced_numbers(answer: str, context: dict, tool_results: list[float]) -> list[str]:
    """Numbers in the answer that are neither in the data nor a calculator result (days/years allowed)."""
    allowed = _numbers({k: v for k, v in context.items() if k not in ("chat", "question")}) | set(tool_results)
    allowed |= {abs(a) for a in allowed}
    bad = []
    for raw in _NUM.findall(answer):
        try:
            n = abs(float(raw.replace(",", "")))
        except ValueError:
            continue
        if (n.is_integer() and (n <= 31 or 1900 <= n <= 2100)) or any(
                abs(n - a) <= 0.01 or (abs(a) >= 1 and round(a) == n) or round(a, 1) == n or round(a, 2) == n
                for a in allowed):
            continue
        bad.append(raw)
    return bad


_IDENTIFIER = re.compile(r"\d{9,}|\d{2,}[- ]\d{4,}[- ]\d{3,}")


def _check(answer: ChatAnswer, context: dict, tool_results: list[float]) -> str | None:
    if _IDENTIFIER.search(answer.answer):
        return "the answer contains a long identifier-like number; never repeat identifiers"
    if bad := untraced_numbers(answer.answer, context, tool_results):
        return f"these numbers are not in the data and did not come from calculate: {', '.join(bad[:5])}"
    return None


def _config(tools_allowed: bool) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=ChatAnswer,
        tools=[CALCULATOR],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.AUTO if tools_allowed else types.FunctionCallingConfigMode.NONE)),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


def _run_turn(client, model: str, contents: list, meta: dict, tool_results: list[float]):
    """Model call + tool round-trips (shared cap). Returns (final text | None, error | None)."""
    while True:
        resp, err = _call(client, model, contents, "chat", meta, config=_config(meta["tool_calls"] < MAX_TOOL_CALLS))
        if resp is None:
            return None, err
        calls = list(resp.function_calls or [])
        if not calls:
            return resp.text or "", None
        if meta["tool_calls"] >= MAX_TOOL_CALLS:
            return None, "tool call limit exceeded"
        contents.append(resp.candidates[0].content)  # keeps thought signatures for the next turn
        parts = []
        for fc in calls:
            meta["tool_calls"] += 1
            if meta["tool_calls"] > MAX_TOOL_CALLS:
                payload = {"error": f"limit of {MAX_TOOL_CALLS} calculations reached; answer now"}
            elif fc.name != "calculate":
                payload = {"error": f"unknown tool {fc.name}"}
            else:
                try:
                    value = calculate((fc.args or {}).get("expression", ""))
                    tool_results.append(value)
                    payload = {"result": value}
                except ValueError as e:
                    payload = {"error": str(e)}
            parts.append(types.Part(function_response=types.FunctionResponse(id=fc.id, name=fc.name, response=payload)))
        contents.append(types.Content(role="user", parts=parts))


def ask(result: ExtractionResult, history: list[dict], question: str) -> dict:
    """Returns {"kind": "answer"|"invalid"|"failed", "content", "is_estimate", "method", "meta"}. Never raises."""
    meta = {"attempts": 0, "input_tokens": None, "output_tokens": None, "tool_calls": 0, "reasked": False,
            "latency_s": None, "kind": None}
    start = time.perf_counter()
    reply = {"kind": "failed", "content": FAILED, "is_estimate": False, "method": "", "meta": meta}
    try:
        question = (question or "").strip()[:MAX_QUESTION_CHARS]
        settings = load_settings()
        if not settings.api_key:
            raise ValueError("API key not configured")
        context = build_context(result, history, question)
        client = _client(settings.api_key, settings.force_ipv4)
        contents: list = [json.dumps(context, ensure_ascii=False)]
        tool_results: list[float] = []
        error = None
        for attempt in range(2):  # original + one re-ask
            if attempt:
                meta["reasked"] = True
                contents.append(REASK.format(error=error))
            text, call_error = _run_turn(client, settings.model_name, contents, meta, tool_results)
            if text is None:
                if call_error == "tool call limit exceeded":
                    error = call_error
                    continue
                meta["failure"] = meta.get("failure") or safe_failure(RuntimeError(), "api")
                break  # API failure: no re-ask
            try:
                parsed = ChatAnswer.model_validate_json(text)
                error = _check(parsed, context, tool_results)
            except ValidationError:
                parsed, error = None, "output was not valid JSON for the schema"
            if error is None:
                reply.update(kind="answer", content=parsed.answer.strip(), is_estimate=parsed.is_estimate,
                             method=parsed.method.strip())
                break
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))  # rejected reply, for the re-ask
        else:
            reply.update(kind="invalid", content=INVALID)
    except Exception as e:  # noqa: BLE001 - only the sanitized category is kept
        meta["failure"] = safe_failure(e, "configuration" if isinstance(e, ValueError) else "processing")
    meta["latency_s"] = round(time.perf_counter() - start, 2)
    meta["kind"] = reply["kind"]
    log.info("chat kind=%s latency=%ss in=%s out=%s attempts=%s tool_calls=%s reasked=%s failure=%s",
             reply["kind"], meta["latency_s"], meta["input_tokens"], meta["output_tokens"], meta["attempts"],
             meta["tool_calls"], meta["reasked"], (meta.get("failure") or {}).get("category"))
    return reply
