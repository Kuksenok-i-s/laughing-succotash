from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from handwriting_ocr.preprocess import (
    OCR_VARIANT_NAMES,
    prepare_image_bytes,
    prepare_ink_bytes,
    prepare_ocr_variants,
    prepare_triage_bytes,
)


def test_prepare_image_grayscale_png(tmp_path: Path) -> None:
    src = tmp_path / "note.jpg"
    image = Image.new("RGB", (80, 40), (180, 160, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 70, 30), fill=(40, 40, 80))
    image.save(src, format="JPEG", quality=90)

    payload, mime = prepare_image_bytes(src)

    assert mime == "image/png"
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    gray = Image.open(BytesIO(payload))
    assert gray.mode == "L"
    # Autocontrast + enhance should spread the histogram vs the muddy original.
    assert gray.getextrema()[0] < 40
    assert gray.getextrema()[1] > 200


def test_prepare_image_falls_back_on_garbage(tmp_path: Path) -> None:
    src = tmp_path / "broken.jpg"
    src.write_bytes(b"not-an-image")
    payload, mime = prepare_image_bytes(src)
    assert payload == b"not-an-image"
    assert mime == "image/jpeg"


def test_prepare_image_fits_long_edge(tmp_path: Path) -> None:
    src = tmp_path / "tall.jpg"
    Image.new("RGB", (400, 1200), (200, 180, 160)).save(src, format="JPEG")

    payload, mime = prepare_image_bytes(src, max_edge=512)
    assert mime == "image/png"
    fitted = Image.open(BytesIO(payload))
    assert max(fitted.size) == 512
    assert fitted.size == (170, 512)


def test_prepare_triage_keeps_colour_and_fits(tmp_path: Path) -> None:
    src = tmp_path / "scene.jpg"
    Image.new("RGB", (1024, 768), (30, 180, 40)).save(src, format="JPEG")

    payload, mime = prepare_triage_bytes(src, max_edge=512)
    assert mime == "image/jpeg"
    fitted = Image.open(BytesIO(payload))
    assert fitted.mode == "RGB"
    assert max(fitted.size) == 512
    assert fitted.size == (512, 384)


def test_prepare_ocr_variants_returns_color_gray_ink(tmp_path: Path) -> None:
    src = tmp_path / "page.jpg"
    Image.new("RGB", (80, 40), (40, 40, 40)).save(src, format="JPEG")
    variants = prepare_ocr_variants(src, max_edge=64)
    assert [name for name, _payload, _mime in variants] == list(OCR_VARIANT_NAMES)
    assert variants[0][2] == "image/jpeg"
    assert variants[1][2] == "image/png"
    assert variants[2][2] == "image/png"
    ink = Image.open(BytesIO(variants[2][1]))
    assert ink.mode == "L"


def test_prepare_ink_inverts_a_dark_page(tmp_path: Path) -> None:
    src = tmp_path / "dark.jpg"
    Image.new("RGB", (40, 20), (20, 20, 20)).save(src, format="JPEG")
    payload, mime = prepare_ink_bytes(src, max_edge=40)
    assert mime == "image/png"
    ink = Image.open(BytesIO(payload))
    assert ink.getextrema()[1] > 200
