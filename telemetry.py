"""In-memory, process-wide usage stats for the public demo. Numbers and categories only: never bill content,
question text or answers. Resets when the app restarts."""
import math
import threading
from collections import Counter, deque

_MAX_SAMPLES = 1000  # latency samples kept per kind (bounded memory)


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.counts: Counter = Counter()    # extractions, chat_calls, cache_hits, fallbacks, limit_blocks
        self.failures: Counter = Counter()  # failure category -> count (e.g. network, quota, validation)
        self.latency = {"extraction": deque(maxlen=_MAX_SAMPLES), "chat": deque(maxlen=_MAX_SAMPLES)}
        self.tokens = {"input": 0, "output": 0}

    def record_call(self, kind: str, meta: dict | None, failure_category: str | None = None) -> None:
        """kind: 'extraction' or 'chat'. meta: latency_s / input_tokens / output_tokens (any may be missing)."""
        meta = meta or {}
        with self._lock:
            self.counts["extractions" if kind == "extraction" else "chat_calls"] += 1
            if failure_category:
                self.failures[failure_category] += 1
            if isinstance(meta.get("latency_s"), (int, float)):
                self.latency[kind].append(float(meta["latency_s"]))
            self.tokens["input"] += int(meta.get("input_tokens") or 0)
            self.tokens["output"] += int(meta.get("output_tokens") or 0)

    def bump(self, name: str) -> None:
        with self._lock:
            self.counts[name] += 1

    @staticmethod
    def _pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)  # nearest-rank percentile
        return round(ordered[max(math.ceil(p / 100 * len(ordered)) - 1, 0)], 2)

    def snapshot(self) -> dict:
        with self._lock:
            lat = {k: list(v) for k, v in self.latency.items()}
            return {
                "extractions": self.counts["extractions"], "chat_calls": self.counts["chat_calls"],
                "cache_hits": self.counts["cache_hits"], "fallbacks": self.counts["fallbacks"],
                "limit_blocks": self.counts["limit_blocks"], "failures": dict(self.failures),
                "latency_s": {k: {"p50": self._pct(v, 50), "p95": self._pct(v, 95), "n": len(v)} for k, v in lat.items()},
                "tokens": dict(self.tokens, total=self.tokens["input"] + self.tokens["output"]),
            }


stats = Stats()
