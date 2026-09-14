"""Rectify confidently detected documents before fitting images for the vision model."""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

import cv2
import numpy as np

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


def prepare_document(image: Image.Image) -> Image.Image:
    """Rectify a large, bright, closed page; ambiguous scenes remain untouched.

    Detect on a thumbnail, but warp the full resolution source before the model cap.
    The inset test rejects contours at the photo border (screenshots/full-frame pages).
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    preview = _fit(rgb, 768)
    gray = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(polygon)
        if (
            len(polygon) != 4
            or not cv2.isContourConvex(polygon)
            or not 0.25 * w * h < area < 0.95 * w * h
        ):
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        if (
            np.any(points[:, 0] < 3) or np.any(points[:, 0] > w - 4)
            or np.any(points[:, 1] < 3) or np.any(points[:, 1] > h - 4)
        ):
            continue
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, polygon, 255)
        inside = gray[mask > 0]
        outside = gray[mask == 0]
        if np.median(inside) < 150 or np.median(inside) - np.median(outside) < 25:
            continue
        # Clockwise corners, starting at the top left; reject ambiguous/extreme quads.
        sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
        indices = [np.argmin(sums), np.argmin(diffs), np.argmax(sums), np.argmax(diffs)]
        if len(set(indices)) != 4:
            continue
        quad = points[indices]
        sides = [np.linalg.norm(quad[(i+1)%4] - quad[i]) for i in range(4)]
        if (
            min(sides) < 0.15 * min(w, h)
            or max(sides[0], sides[2]) / min(sides[0], sides[2]) > 2
            or max(sides[1], sides[3]) / min(sides[1], sides[3]) > 2
        ):
            continue
        quad *= np.array([rgb.width/w, rgb.height/h], dtype=np.float32)
        tl, tr, br, bl = quad
        width = math.ceil(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
        height = math.ceil(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
        target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        matrix = cv2.getPerspectiveTransform(quad, target)
        rectified = cv2.warpPerspective(np.asarray(rgb), matrix, (width, height), flags=cv2.INTER_CUBIC)
        log.info("rectified document %s -> %s", rgb.size, (width, height))
        return Image.fromarray(rectified)
    return rgb


def enhance_region(image: Image.Image, *, max_edge: int) -> Image.Image:
    """Give small text more pixels in a retry, without another model pass.

    On light paper only: normalize contrast, trim blank margins with padding and
    enlarge up to 3x within the existing model cap. Preserve dark/flat fragments.
    Thresholding is used only to locate ink, never on the returned image.
    """
    gray = ImageOps.grayscale(image)
    low, high = gray.getextrema()
    if high - low < 12 or np.median(np.asarray(gray)) < 180:
        return _fit(image, max_edge)
    contrasted = ImageOps.autocontrast(gray, cutoff=0)
    ink = contrasted.point(lambda value: 255 if value < 210 else 0)
    box = ink.getbbox()
    if box is None:
        return _fit(image, max_edge)
    padding = 12
    left, top, right, bottom = box
    cropped = contrasted.crop((
        max(0, left - padding), max(0, top - padding),
        min(image.width, right + padding), min(image.height, bottom + padding),
    ))
    # Do not magnify almost empty specks as though they were a text fragment.
    if right - left < 8 or bottom - top < 4:
        return _fit(image, max_edge)
    scale = min(3.0, max_edge / max(cropped.size)) if max_edge > 0 else 1.0
    size = (max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale)))
    return cropped.resize(size, Image.Resampling.LANCZOS)


def prepare_region_bytes(image_path: Path, box: tuple[int, int, int, int], *, max_edge: int) -> tuple[bytes, str]:
    """Crop normalized 0..1000 coordinates from the same rectified source as pass 1."""
    with Image.open(image_path) as image:
        page = prepare_document(image)
        x1, y1, x2, y2 = box
        crop = page.crop((int(x1*page.width/1000), int(y1*page.height/1000),
                          math.ceil(x2*page.width/1000), math.ceil(y2*page.height/1000)))
        crop = enhance_region(crop, max_edge=max_edge)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"


def prepare_triage_bytes(image_path: Path, *, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, str]:
    """Colour JPEG for pass 1. Keep hues (a cat is not a list) but bound the long edge."""
    try:
        with Image.open(image_path) as image:
            image.load()
            rgb = prepare_document(image)
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
            gray = ImageOps.grayscale(prepare_document(image))
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
            gray = ImageOps.grayscale(prepare_document(image))
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
