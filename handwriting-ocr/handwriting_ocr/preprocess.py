"""Prepare photos for Qwen3-VL: grayscale + stronger contrast.

Handwriting on phone photos is usually low-contrast and noisy colour. A cheap local pass
before Ollama cuts visual clutter without another model. Pillow is the only image dependency;
if decode fails we fall back to the original bytes so a job still runs.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

log = logging.getLogger(__name__)

# Mild push after autocontrast — enough for pencil on paper without crushing midtones.
_CONTRAST = 1.6
_AUTCONTRAST_CUTOFF = 1.0


def prepare_image_bytes(image_path: Path) -> tuple[bytes, str]:
    """Return ``(png_bytes, mime)`` ready to base64 for Ollama, or the original file on failure."""
    try:
        with Image.open(image_path) as image:
            image.load()
            gray = ImageOps.grayscale(image)
            gray = ImageOps.autocontrast(gray, cutoff=_AUTCONTRAST_CUTOFF)
            gray = ImageEnhance.Contrast(gray).enhance(_CONTRAST)
            buffer = io.BytesIO()
            gray.save(buffer, format="PNG", optimize=True)
            payload = buffer.getvalue()
        log.info(
            "preprocessed %s -> grayscale+contrast PNG (%d -> %d bytes)",
            image_path.name,
            image_path.stat().st_size,
            len(payload),
        )
        return payload, "image/png"
    except Exception as exc:
        log.warning("image preprocess failed for %s (%s); using original bytes", image_path.name, exc)
        return image_path.read_bytes(), _guess_mime(image_path)


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"
