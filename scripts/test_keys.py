"""Smoke-test each Gemini key in .env against each allowed Gemini model.

Usage: python scripts/test_keys.py
Never prints key values, only their labels.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

KEY_LABELS = ["GEMINI_API_KEY", "GEMINI_API_KEY_BACKUP"]
ALLOWED_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]  # competition_context.md §3
TIMEOUT_MS = 20_000  # http_options timeout is in milliseconds
MAX_OUTPUT_TOKENS = 200


class Ping(BaseModel):
    ok: bool


def list_flash_lite(client: genai.Client) -> list[str]:
    try:
        return sorted(m.name for m in client.models.list() if "flash-lite" in (m.name or ""))
    except Exception as e:  # noqa: BLE001
        return [f"(could not list models: {type(e).__name__})"]


def test_one(client: genai.Client, model: str) -> dict:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Ping,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )
    start = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model, contents='Reply with JSON {"ok": true}.', config=config
        )
        latency = time.perf_counter() - start
        Ping.model_validate_json(resp.text or "")
        u = resp.usage_metadata
        return {
            "status": "OK",
            "latency": latency,
            "total": getattr(u, "total_token_count", None),
            "thoughts": getattr(u, "thoughts_token_count", None),
            "err": "",
        }
    except errors.APIError as e:
        result = {"status": "FAIL", "latency": time.perf_counter() - start,
                  "err": f"HTTP {e.code}: {(e.message or '')[:80]}"}
        if e.code == 404 or "not found" in str(e).lower():
            result["not_found"] = True
        return result
    except ValidationError:
        return {"status": "FAIL", "latency": time.perf_counter() - start,
                "err": "response failed Pydantic validation"}
    except Exception as e:  # noqa: BLE001 (timeouts, network)
        return {"status": "FAIL", "latency": time.perf_counter() - start,
                "err": f"{type(e).__name__}: {str(e)[:80]}"}


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    print(f"{'KEY':<24}{'MODEL':<24}{'STATUS':<8}{'LATENCY':>9}{'TOKENS':>8}{'THINK':>7}  ERROR")
    for label in KEY_LABELS:
        key = os.getenv(label, "").strip()
        if not key:
            print(f"{label:<24}{'-':<24}{'SKIP':<8}{'':>9}{'':>8}{'':>7}  not set in .env")
            continue
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        listed = False
        for model in ALLOWED_MODELS:
            r = test_one(client, model)
            tok = "" if r.get("total") is None else str(r["total"])
            think = "" if r.get("thoughts") is None else str(r["thoughts"])
            print(f"{label:<24}{model:<24}{r['status']:<8}{r['latency']:>8.2f}s{tok:>8}{think:>7}  {r['err']}")
            if r.get("not_found") and not listed:
                listed = True
                print("    available flash-lite models:", ", ".join(list_flash_lite(client)) or "(none)")


if __name__ == "__main__":
    main()
