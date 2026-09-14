from handwriting_ocr.engine import parse_triage


def test_parse_triage_text_json() -> None:
    got = parse_triage('{"kind":"text","raw_text":"Купить молоко"}')
    assert got == {"kind": "text", "raw_text": "Купить молоко"}


def test_parse_triage_other_json() -> None:
    got = parse_triage('{"kind":"other","description":"A red mug on a desk"}')
    assert got == {"kind": "other", "description": "A red mug on a desk"}


def test_parse_triage_fenced_and_messy() -> None:
    got = parse_triage('Sure.\n```json\n{"kind":"other","description":"cat"}\n```\n')
    assert got["kind"] == "other"
    assert got["description"] == "cat"


def test_parse_triage_plain_fallback_is_text() -> None:
    got = parse_triage("строка один\nстрока два")
    assert got["kind"] == "text"
    assert "строка один" in got["raw_text"]
