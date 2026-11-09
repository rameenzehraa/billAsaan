"""Public demo limits: whole-server daily caps, session limits, quota fallback, sample-result cache.

Process-wide state lives in this module (imported once per server process), so it is shared by all
sessions and RESETS WHEN THE APP RESTARTS. Daily counters also reset at local midnight, Asia/Karachi
(UTC+5, no daylight saving, so a fixed offset is exact and needs no tzdata).
"""
import logging
import threading
from datetime import date, datetime, timedelta, timezone

from config import FALLBACK_MODEL, use_model
from failures import ProcessingFailure

log = logging.getLogger("limits")

PKT = timezone(timedelta(hours=5), "Asia/Karachi")
DAILY_LIMIT = "Daily demo limit reached, try tomorrow."
BILLS_LIMIT = "You've reached this session's limit of {n} bills. You can still view and ask about bills you've processed."
QUESTIONS_LIMIT = "You've reached the limit of {n} questions for this bill."
QUOTA = "quota"


def today() -> date:
    return datetime.now(PKT).date()


class DailyCounter:
    """Thread-safe count of API actions per local day. try_acquire() checks and reserves in one step."""

    def __init__(self, clock=today):
        self._lock = threading.Lock()
        self._clock = clock
        self._day = clock()
        self._count = 0

    def _roll(self) -> None:
        day = self._clock()
        if day != self._day:
            self._day, self._count = day, 0

    def try_acquire(self, cap: int) -> bool:
        with self._lock:
            self._roll()
            if self._count >= cap:
                return False
            self._count += 1
            return True

    def remaining(self, cap: int) -> int:
        with self._lock:
            self._roll()
            return max(cap - self._count, 0)

    @property
    def count(self) -> int:
        with self._lock:
            self._roll()
            return self._count

    def reset(self) -> None:
        with self._lock:
            self._day, self._count = self._clock(), 0


extractions = DailyCounter()
chats = DailyCounter()

# Results for reviewed public samples only (never user uploads): sample sha256 -> build_view() output.
_sample_lock = threading.Lock()
_sample_results: dict[str, dict] = {}


def cached_sample(key: str) -> dict | None:
    with _sample_lock:
        return _sample_results.get(key)


def store_sample(key: str, view: dict) -> None:
    with _sample_lock:
        _sample_results[key] = view


# ---- Session limits (counters live in st.session_state) ----

def bills_left(state, cap: int) -> int:
    return max(cap - state.get("bills_used", 0), 0)


def questions_asked(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def questions_left(messages: list[dict], cap: int) -> int:
    return max(cap - questions_asked(messages), 0)


# ---- Quota fallback ----

def _quota_failure(outcome) -> bool:
    """True for a 429 / quota failure, whether raised (extraction) or returned (chat reply dict)."""
    if isinstance(outcome, ProcessingFailure):
        return outcome.diagnostic.get("category") == QUOTA
    if isinstance(outcome, dict):
        return (outcome.get("meta", {}).get("failure") or {}).get("category") == QUOTA
    return False


def with_fallback(call):
    """Run call() on the configured model; on a quota/429 failure, run it once more on FALLBACK_MODEL.
    Returns (outcome, used_fallback, quota_exhausted). Exceptions other than a quota failure propagate."""
    try:
        outcome = call()
    except ProcessingFailure as failure:
        if not _quota_failure(failure):
            raise
        outcome = failure
    if not _quota_failure(outcome):
        return outcome, False, False
    log.warning("quota failure on primary model; retrying once on fallback model")
    with use_model(FALLBACK_MODEL):
        try:
            outcome = call()
        except ProcessingFailure as failure:
            if not _quota_failure(failure):
                raise
            return failure, True, True
    return outcome, True, _quota_failure(outcome)
