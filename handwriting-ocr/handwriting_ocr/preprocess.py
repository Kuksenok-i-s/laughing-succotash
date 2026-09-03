"""Prepare photos for Qwen-VL: fit, then grayscale + stronger contrast.

Handwriting on phone photos is usually low-contrast and noisy colour. A cheap local pass
before the VL model cuts visual clutter without another model. Pillow is the only image
dependency; if decode fails we fall back to the original bytes so a job still runs.

Every image sent to llama.cpp is also fitted to ``max_edge``. A tall receipt at full
phone resolution (or even 1024px) makes Qwen3-VL's mmproj spike unified memory on a
14GB AGX and the board hard-hangs. Fitting first keeps vision tokens bounded.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

log = logging.getLogger(__name__)

# Mild push after autocontrast — enough for pencil on paper without crushing midtones.
_CONTRAST = 1.6
_INK_CONTRAST = 2.1
_AUTCONTRAST_CUTOFF = 1.0
DEFAULT_MAX_EDGE = 512
OCR_VARIANT_NAMES = ("color", "gray", "ink")


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare_triage_bytes(image_path: Path, *, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, str]:
    """Colour JPEG for pass 1. Keep hues (a cat is not a list) but bound the long edge."""
    try:
        with Image.open(image_path) as image:
            image.load()
            rgb = image.convert("RGB")
            fitted = _fit(rgb, max_edge)
            buffer = io.BytesIO()
            fitted.save(buffer, format="JPEG", quality=85, optimize=True)
            payload = buffer.getvalue()
        log.info(
            "triage %s -> JPEG %s (%d -> %d bytes, max_edge=%d)",
            image_path.name,
            fitted.size,
            image_path.stat().st_size,
            len(payload),
            max_edge,
        )
        return payload, "image/jpeg"
    except Exception as exc:
        log.warning("triage preprocess failed for %s (%s); using original bytes", image_path.name, exc)
        return image_path.read_bytes(), _guess_mime(image_path)


def prepare_image_bytes(image_path: Path, *, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, str]:
    """Return ``(png_bytes, mime)`` ready to base64, or the original file on failure."""
    try:
        with Image.open(image_path) as image:
            image.load()
            gray = ImageOps.grayscale(image)
            gray = _fit(gray, max_edge)
            gray = ImageOps.autocontrast(gray, cutoff=_AUTCONTRAST_CUTOFF)
            gray = ImageEnhance.Contrast(gray).enhance(_CONTRAST)
            buffer = io.BytesIO()
            gray.save(buffer, format="PNG", optimize=True)
            payload = buffer.getvalue()
        log.info(
            "preprocessed %s -> grayscale+contrast PNG %s (%d -> %d bytes, max_edge=%d)",
            image_path.name,
            gray.size,
            image_path.stat().st_size,
            len(payload),
            max_edge,
        )
        return payload, "image/png"
    except Exception as exc:
        log.warning("image preprocess failed for %s (%s); using original bytes", image_path.name, exc)
        return image_path.read_bytes(), _guess_mime(image_path)


def prepare_ink_bytes(image_path: Path, *, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, str]:
    """High-contrast ink: gray, punchier contrast, unsharp. Invert if the page is dark."""
    try:
        with Image.open(image_path) as image:
            image.load()
            gray = ImageOps.grayscale(image)
            gray = _fit(gray, max_edge)
            gray = ImageOps.autocontrast(gray, cutoff=2.0)
            gray = ImageEnhance.Contrast(gray).enhance(_INK_CONTRAST)
            gray = gray.filter(ImageFilter.UnsharpMask(radius=1.6, percent=170, threshold=2))
            if _mean_luma(gray) < 90:
                gray = ImageOps.invert(gray)
            buffer = io.BytesIO()
            gray.save(buffer, format="PNG", optimize=True)
            payload = buffer.getvalue()
        log.info(
            "preprocessed %s -> ink PNG %s (%d -> %d bytes, max_edge=%d)",
            image_path.name,
            gray.size,
            image_path.stat().st_size,
            len(payload),
            max_edge,
        )
        return payload, "image/png"
    except Exception as exc:
        log.warning("ink preprocess failed for %s (%s); using original bytes", image_path.name, exc)
        return image_path.read_bytes(), _guess_mime(image_path)


def prepare_ocr_variants(
    image_path: Path, *, max_edge: int = DEFAULT_MAX_EDGE
) -> list[tuple[str, bytes, str]]:
    """Three hardware-bound views of the same page: colour, gray+contrast, ink."""
    return [
        ("color", *prepare_triage_bytes(image_path, max_edge=max_edge)),
        ("gray", *prepare_image_bytes(image_path, max_edge=max_edge)),
        ("ink", *prepare_ink_bytes(image_path, max_edge=max_edge)),
    ]


def _mean_luma(gray: Image.Image) -> float:
    hist = gray.histogram()
    total = sum(hist) or 1
    return sum(index * count for index, count in enumerate(hist)) / total


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"
