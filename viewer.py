"""Session-only public viewer helpers. Uploads stay in memory; never use the batch disk cache."""
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

from checks import run_checks
from failures import ProcessingFailure, safe_failure
from images import UPLOAD_EXTS, ImageLoadError, decode_upload
from schema import ExtractionResult

MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 2000
UPLOAD_TYPES = sorted(e.lstrip(".") for e in UPLOAD_EXTS)
SAMPLE_DIR = Path(__file__).parent / "data" / "public_samples"
NOT_SHOWN = "Not shown on bill"
NOT_A_BILL = "This doesn't look like an electricity bill. Try a clearer photo."


def prepare_upload(name: str, data: bytes) -> tuple[bytes, str]:
    """Validate and normalize an upload to metadata-free PNG bytes, entirely in memory."""
    if Path(name).suffix.lower() not in UPLOAD_EXTS:
        raise ValueError("Choose a JPG, PNG, PDF or HEIC file.")
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a non-empty file no larger than 8 MB.")
    try:
        image = decode_upload(name, data)
    except ImageLoadError:
        raise ValueError("This file can't be read as a bill image. Try a clearer photo or another file.") from None
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    image.save(out, format="PNG")  # re-encoding drops EXIF/GPS metadata before preview and the model
    return out.getvalue(), "image/png"


def image_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            preview, mime = prepare_upload(path.name, data)
            safe.append({"label": str(entry["label"]), "data": preview, "mime": mime, "id": entry["sha256"]})
        except (KeyError, TypeError, OSError, ValueError):
            continue
    return safe


def is_bill(result: ExtractionResult) -> bool:
    """A bill must show at least one of: payable amount, units consumed, bill month."""
    b = result.level1
    return any(v is not None for v in (b.payable_within_due_date, b.units_consumed, b.bill_month))


def build_view(result: ExtractionResult) -> dict:
    """Everything the stats page renders, computed once per image (checks are pure, no API)."""
    validated = ExtractionResult.model_validate(result.model_dump(mode="json"))
    return {"result": validated, "is_bill": is_bill(validated),
            "checks": run_checks(validated.level1, validated.internal)}


# ---- Display formatting: null always renders as NOT_SHOWN, never blank or "None" ----

def show(value) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return NOT_SHOWN
    return str(value)


def money(value) -> str:
    if value is None:
        return NOT_SHOWN
    return f"Rs {value:,.0f}" if float(value).is_integer() else f"Rs {value:,.2f}"


def units(value) -> str:
    return NOT_SHOWN if value is None else f"{value:,.0f} kWh" if float(value).is_integer() else f"{value:,} kWh"


def month(value: str | None) -> str:
    if value is None:
        return NOT_SHOWN
    try:
        return datetime.strptime(value, "%Y-%m").strftime("%b %Y")
    except ValueError:
        return value


def date(value: str | None) -> str:
    if value is None:
        return NOT_SHOWN
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return value


def label(code: str) -> str:
    return {"fpa": "FPA (fuel price adj.)", "gst": "GST", "other_tax": "Other tax"}.get(
        code, code.replace("_", " ").capitalize())


def line_items(result: ExtractionResult) -> list[dict]:
    b = result.level1
    rows = [{"Type": "Charge", "Item": label(c.type), "Amount": money(c.amount)} for c in b.charges]
    rows += [{"Type": "Tax", "Item": label(t.type), "Amount": money(t.amount)} for t in b.taxes]
    return rows


def history_rows(result: ExtractionResult) -> list[dict]:
    return [{"Month": show(m.month), "Units": m.units}
            for m in result.internal.usage_history if m.units is not None]


def process_image(data: bytes, mime: str) -> ExtractionResult:
    # Import lazily: selection and preview never initialize a provider.
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
