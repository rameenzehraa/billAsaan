"""Exact-match cache for successful, validated extractions only. Failures are never cached."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

from config import CACHE_DIR, PROMPT_VERSION, ROOT, SCHEMA_VERSION, TEMPERATURE

EXTRACT_DIR = ROOT / CACHE_DIR / "extract"
RESULTS_DIR = ROOT / CACHE_DIR / "results"  # latest successful full result per bill_id (for Level 2)


def extract_key(image_bytes: bytes, model: str, settings_id: str) -> str:
    h = hashlib.sha256(image_bytes)
    h.update(f"|model={model}|temp={TEMPERATURE}|{settings_id}|prompt={PROMPT_VERSION}|schema={SCHEMA_VERSION}".encode())
    return h.hexdigest()


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def get(key: str) -> dict | None:
    path = EXTRACT_DIR / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def put(key: str, entry: dict) -> None:
    """entry must be a successful, validated result; callers never pass fallbacks."""
    _atomic_write(EXTRACT_DIR / f"{key}.json", entry)


def save_result(bill_id: str, entry: dict) -> None:
    _atomic_write(RESULTS_DIR / f"{bill_id}.json", entry)
