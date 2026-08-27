from pathlib import Path

from PIL import Image, ImageDraw

from handwriting_ocr.preprocess import prepare_image_bytes


def test_prepare_image_grayscale_png(tmp_path: Path) -> None:
    src = tmp_path / "note.jpg"
    image = Image.new("RGB", (80, 40), (180, 160, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 70, 30), fill=(40, 40, 80))
    image.save(src, format="JPEG", quality=90)

    payload, mime = prepare_image_bytes(src)

    assert mime == "image/png"
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    from io import BytesIO

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
