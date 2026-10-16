"""Runtime configuration. Reads .env; never prints or logs the API key.

Importing this module never raises. Call load_settings() to validate the model choice.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

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


def load_settings() -> Settings:
    """Read and validate MODEL_PROVIDER / MODEL_NAME / API_KEY from the environment."""
    # MODEL_ID and GEMINI_API_KEY are the older names used by scripts/test_keys.py.
    name = validate_model_name(
        (os.getenv("MODEL_NAME") or os.getenv("MODEL_ID") or DEFAULT_MODEL).strip()
    )
    expected = ALLOWED_MODELS[name]
    provider = (os.getenv("MODEL_PROVIDER") or expected).strip().lower()
    if provider != expected:
        raise ValueError(f"MODEL_PROVIDER '{provider}' does not match MODEL_NAME '{name}' (expected '{expected}')")
    api_key = (os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    return Settings(model_provider=provider, model_name=name, api_key=api_key)
