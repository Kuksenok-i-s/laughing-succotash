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


def test_rectifies_perspective_page_before_fitting(tmp_path):
    from handwriting_ocr.preprocess import prepare_document
    image = Image.new("RGB", (1000, 800), (45, 40, 35))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 100), (820, 170), (760, 710), (140, 650)], fill="white")
    draw.line([(300, 300), (680, 340)], fill="black", width=6)
    page = prepare_document(image)
    assert 590 < page.width < 660
    assert 520 < page.height < 590
    assert page.getpixel((page.width//2, 30))[0] > 240
    assert page.convert("L").getextrema()[0] < 30
    src = tmp_path / "document.png"
    image.save(src)
    payload, _ = prepare_triage_bytes(src, max_edge=512)
    assert max(Image.open(BytesIO(payload)).size) == 512


def test_exif_orientation_is_applied(tmp_path):
    src = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (800, 400), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(src, exif=exif)
    payload, _ = prepare_triage_bytes(src, max_edge=512)
    assert Image.open(BytesIO(payload)).size == (256, 512)


def test_full_frame_and_ambiguous_images_are_not_cropped():
    from handwriting_ocr.preprocess import prepare_document
    for color in ["white", "black", "red"]:
        image = Image.new("RGB", (800, 600), color)
        assert prepare_document(image).size == image.size
    image = Image.new("RGB", (800, 600), "black")
    ImageDraw.Draw(image).rectangle((0, 0, 700, 550), fill="white")
    assert prepare_document(image).size == image.size


def test_region_enhancement_removes_blank_margins_without_losing_ink():
    from handwriting_ocr.preprocess import enhance_region
    image = Image.new("RGB", (900, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 260, 40), fill=(130, 130, 130))
    enhanced = enhance_region(image, max_edge=1024)
    assert enhanced.size == (765, 105)
    assert enhanced.getextrema() == (0, 255)
    assert enhanced.getpixel((0, 0)) == 255


def test_region_enhancement_respects_cap_and_does_not_threshold():
    from handwriting_ocr.preprocess import enhance_region
    image = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 700, 80), fill=(100, 100, 100))
    draw.rectangle((40, 100, 700, 140), fill=(170, 170, 170))
    enhanced = enhance_region(image, max_edge=512)
    assert max(enhanced.size) <= 512
    assert len(enhanced.getcolors(maxcolors=256)) > 2


def test_region_enhancement_leaves_dark_and_flat_fragments_alone():
    from handwriting_ocr.preprocess import enhance_region
    for color in ["white", "black", (230, 230, 230)]:
        image = Image.new("RGB", (100, 50), color)
        assert enhance_region(image, max_edge=512).tobytes() == image.tobytes()
    image = Image.new("RGB", (100, 50), "black")
    ImageDraw.Draw(image).rectangle((20, 20, 80, 30), fill="white")
    assert enhance_region(image, max_edge=512).tobytes() == image.tobytes()
