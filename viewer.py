"""Session-only public viewer helpers. Never use the batch disk cache for uploads."""
import hashlib
import io
import json
import warnings
from pathlib import Path

from PIL import Image, ImageOps
from calc import compute_facts
from checks import run_checks
from schema import ExtractionResult
from failures import ProcessingFailure, safe_failure

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
SAMPLE_DIR = Path(__file__).parent / "data" / "public_samples"


def validate_image(name: str, data: bytes) -> tuple[bytes, str]:
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Choose a JPG or PNG image.")
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a non-empty image no larger than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                expected = "PNG" if suffix == ".png" else "JPEG"
                if image.format != expected:
                    raise ValueError("The image contents do not match its file type.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Choose an image with at most 20 million pixels.")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                fixed = ImageOps.exif_transpose(image).convert("RGB")
                fixed.thumbnail((2000, 2000))
                output = io.BytesIO()
                fixed.save(output, format="PNG")  # Strip metadata before preview/provider.
                return output.getvalue(), "image/png"
    except ValueError:
        raise
    except Exception:
        raise ValueError("This image cannot be read. Choose a valid JPG or PNG.") from None


def verified_samples(root: Path = SAMPLE_DIR) -> list[dict]:
    """Fail closed: review attestation AND content hash must match; never scan original data."""
    try:
        entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            return []
    except (OSError, ValueError):
        return []
    safe = []
    for entry in entries:
        try:
            path = (root / entry["file"]).resolve()
            if (entry.get("verified_anonymized") is not True or not entry.get("reviewed_by")
                    or path.parent != root.resolve()):
                continue
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                continue
            preview, mime = validate_image(path.name, data)
            safe.append({"label": str(entry["label"]), "data": preview, "mime": mime,
                         "id": entry["sha256"]})
        except (KeyError, TypeError, OSError, ValueError):
            continue
    return safe


def select_bill(state, identity):
    if state.get("bill_identity") != identity:
        for key in list(state):
            if key.startswith("review_") or key.startswith("chat_"):
                del state[key]
        state.update(bill_identity=identity, original=None, corrected=None, checks=[], facts={},
                     messages=[], busy=False, process_error=None, process_diagnostic=None, revision=0)


def apply_result(state, result, *, original=False):
    validated = ExtractionResult.model_validate(result.model_dump(mode="json"))
    checks = run_checks(validated.level1, validated.internal)
    facts = compute_facts(validated)
    if original:
        state["original"] = validated.model_copy(deep=True)
    state.update(corrected=validated, checks=checks, facts=facts)


def process_image(data: bytes, mime: str) -> ExtractionResult:
    # Import lazily: selection, preview and corrections never initialize a provider.
    from extract import extract_bill
    from config import load_settings
    try:
        settings = load_settings()
        if not settings.api_key:
            raise ValueError("API key not configured")
    except Exception as error:
        raise ProcessingFailure(safe_failure(error, "configuration")) from None
    try:
        result, meta = extract_bill("upload", None, "unused", image=(data, mime, None, None), private=True)
    except Exception as error:
        raise ProcessingFailure(safe_failure(error, "api")) from None
    if meta["error"]:
        diagnostic = meta.get("failure") or safe_failure(RuntimeError(), "processing")
        raise ProcessingFailure(diagnostic) from None
    try:
        return ExtractionResult.model_validate(result.model_dump(mode="json"))
    except Exception as error:
        raise ProcessingFailure(safe_failure(error, "validation")) from None
