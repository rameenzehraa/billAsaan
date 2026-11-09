"""Runtime configuration. Reads .env; never prints or logs the API key.

Importing this module never raises. Call load_settings() to validate the model choice.
"""
import contextlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _setting(name: str, default: str | None = None) -> str | None:
    """Streamlit secrets first (Community Cloud), then the environment / .env. Values are returned as text
    and never logged. Outside a Streamlit app, or with no secrets.toml, secrets are simply skipped."""
    try:
        import streamlit as st
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file / not running under Streamlit
        value = None
    if value is None or (isinstance(value, str) and not value.strip()):
        value = os.getenv(name)
    if value is None or not str(value).strip():  # blank means "not set"
        return default
    return str(value)

# Allowed models (competition_context.md §3). Any other model means disqualification.
ALLOWED_MODELS = {
    "claude-haiku-4-5": "anthropic",
    "gemini-3.5-flash-lite": "google",
    "gemini-3.1-flash-lite": "google",
    "gpt-4.1-mini": "openai",
    "gpt-5-mini": "openai",
}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

TEMPERATURE = 0
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 2
PROMPT_VERSION = "v3"  # v2: combined-tax rule; v3: side-box charge lines (e.g. Total FPA)
ANSWER_PROMPT_VERSION = "v7"  # v2: specific answers; v3: code-filled dates, strict trace; v4: month lists, estimate label, placeholder rules; v5: partial-history month count; v6: question-handling rules for new facts; v7: tier wording, pct placeholders
CALC_VERSION = "c2"  # c2: avg comparison, seasons, payments, 3-month trend, cost-per-unit estimates, surcharge tiers, net metering
SCHEMA_VERSION = "v1"
CACHE_DIR = ".cache/"
MAX_IMAGE_LONG_SIDE = 2000

# Filename prefix -> JSON provider value (guide §4.2: KESC files map to KE).
PROVIDER_BY_PREFIX = {"KESC": "KE", "KE": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}


def provider_for(bill_id: str) -> str | None:
    """Provider from the bill_id prefix (KESC_0008 -> KE), or None if unknown."""
    return PROVIDER_BY_PREFIX.get(bill_id.split("_", 1)[0].upper())


def validate_model_name(name: str) -> str:
    if name not in ALLOWED_MODELS:
        raise ValueError(
            f"MODEL_NAME '{name}' is not an allowed model. Use one of: {', '.join(sorted(ALLOWED_MODELS))}"
        )
    return name


@dataclass(frozen=True)
class Settings:
    model_provider: str
    model_name: str
    api_key: str = field(repr=False)
    force_ipv4: bool = False


def load_settings() -> Settings:
    """Read and validate MODEL_PROVIDER / MODEL_NAME / API_KEY from the environment."""
    # MODEL_ID and GEMINI_API_KEY are the older names used by scripts/test_keys.py.
    name = validate_model_name(
        getattr(_override, "model", None) or (_setting("MODEL_NAME") or _setting("MODEL_ID") or DEFAULT_MODEL).strip()
    )
    expected = ALLOWED_MODELS[name]
    provider = (_setting("MODEL_PROVIDER") or expected).strip().lower()
    if provider != expected:
        raise ValueError(f"MODEL_PROVIDER '{provider}' does not match MODEL_NAME '{name}' (expected '{expected}')")
    api_key = (_setting("API_KEY") or _setting("GEMINI_API_KEY") or "").strip()
    ipv4 = _setting("GENAI_IPV4_ONLY", "0").strip().lower()
    if ipv4 not in {"0", "1", "false", "true"}:
        raise ValueError("GENAI_IPV4_ONLY must be 0/1 or false/true")
    return Settings(model_provider=provider, model_name=name, api_key=api_key,
                    force_ipv4=ipv4 in {"1", "true"})


# ---- Per-thread model override (quota fallback). Streamlit runs each session's script in its own thread. ----

_override = threading.local()


@contextlib.contextmanager
def use_model(name: str):
    """Within this block, load_settings() returns `name` (must be allowed) on this thread only."""
    validate_model_name(name)
    previous = getattr(_override, "model", None)
    _override.model = name
    try:
        yield
    finally:
        _override.model = previous


# ---- Public demo limits (D3). Counters themselves live in limits.py / session_state. ----

@dataclass(frozen=True)
class Limits:
    free_tier: bool = True
    max_bills_per_session: int = 3
    max_questions_per_bill: int = 15
    daily_extraction_cap: int = 150
    daily_chat_cap: int = 400


def _env_int(name: str, default: int) -> int:
    raw = (_setting(name) or "").strip()
    if not raw:
        return default
    if not raw.isdigit():
        raise ValueError(f"{name} must be a non-negative whole number")
    return int(raw)


def load_limits() -> Limits:
    """App startup config: FREE_TIER and the caps (secrets, then env). Also requires the model settings and an
    API key, so a misconfigured deployment fails at startup with one message. Raises ValueError (no values shown)."""
    if not load_settings().api_key:
        raise ValueError("API_KEY is not set. Add it to Streamlit secrets (Cloud) or to .env (local).")
    free = (_setting("FREE_TIER") or "true").strip().lower()
    if free not in {"0", "1", "false", "true"}:
        raise ValueError("FREE_TIER must be true/false or 1/0")
    return Limits(free_tier=free in {"1", "true"},
                  max_bills_per_session=_env_int("MAX_BILLS_PER_SESSION", 3),
                  max_questions_per_bill=_env_int("MAX_QUESTIONS_PER_BILL", 15),
                  daily_extraction_cap=_env_int("DAILY_EXTRACTION_CAP", 150),
                  daily_chat_cap=_env_int("DAILY_CHAT_CAP", 400))
