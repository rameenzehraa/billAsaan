"""List bill images and load them for the model (EXIF transpose, downscale only if too large)."""
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from config import MAX_IMAGE_LONG_SIDE

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg"}


class ImageLoadError(Exception):
    pass


def list_bills(directory: str | Path) -> list[tuple[str, Path]]:
    """[(bill_id, path)] sorted by bill_id. bill_id is the filename stem."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Bill folder not found: {d}")
    return sorted((p.stem, p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_image(path: str | Path) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """Return (bytes, mime, original_size, sent_size). Original bytes are kept unless
    EXIF rotation or downscaling changed the image."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        with Image.open(io.BytesIO(raw)) as img:
            fmt = img.format or "PNG"
            img.load()
            rotated = img.getexif().get(0x0112, 1) != 1  # EXIF Orientation tag
            fixed = ImageOps.exif_transpose(img)
            original_size = fixed.size
            w, h = original_size
            long_side = max(w, h)
            if long_side > MAX_IMAGE_LONG_SIDE:
                scale = MAX_IMAGE_LONG_SIDE / long_side
                fixed = fixed.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            elif not rotated:
                return raw, MIME_BY_FORMAT.get(fmt, "image/png"), original_size, original_size

            out_fmt = "JPEG" if fmt == "JPEG" else "PNG"
            if out_fmt == "JPEG" and fixed.mode not in ("RGB", "L"):
                fixed = fixed.convert("RGB")
            buf = io.BytesIO()
            fixed.save(buf, format=out_fmt, **({"quality": 92} if out_fmt == "JPEG" else {}))
            return buf.getvalue(), MIME_BY_FORMAT[out_fmt], original_size, fixed.size
    except (OSError, UnidentifiedImageError, ValueError) as e:
        raise ImageLoadError(f"Cannot read image file '{p}': {type(e).__name__}: {e}") from e


# ---- Public uploads: decode in memory only (never written to disk) ----

UPLOAD_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".heic"}
_UPLOAD_FORMATS = {".png": {"PNG"}, ".jpg": {"JPEG"}, ".jpeg": {"JPEG"}, ".heic": {"HEIF", "HEIC"}}
UPLOAD_MAX_PIXELS = 40_000_000


def decode_upload(name: str, data: bytes) -> Image.Image:
    """Return an RGB, EXIF-upright PIL image from upload bytes. PDF: first page only.
    Raises ImageLoadError on anything unreadable or whose contents don't match the extension."""
    suffix = Path(name).suffix.lower()
    if suffix not in UPLOAD_EXTS:
        raise ImageLoadError("unsupported type")
    try:
        if suffix == ".pdf":
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(data)
            try:
                if len(pdf) < 1:
                    raise ImageLoadError("empty PDF")
                page = pdf[0]
                w, h = page.get_size()
                scale = min(3.0, MAX_IMAGE_LONG_SIDE / max(w, h, 1))  # ~2000 px long side
                return page.render(scale=scale).to_pil().convert("RGB")
            finally:
                pdf.close()
        if suffix == ".heic":
            import pillow_heif
            pillow_heif.register_heif_opener()
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in _UPLOAD_FORMATS[suffix]:
                raise ImageLoadError("contents do not match file type")
            if img.width * img.height > UPLOAD_MAX_PIXELS:
                raise ImageLoadError("too many pixels")
            img.load()
            return ImageOps.exif_transpose(img).convert("RGB")
    except ImageLoadError:
        raise
    except Exception as e:  # noqa: BLE001 - any decoder failure is "unreadable"
        raise ImageLoadError(type(e).__name__) from None
