import base64
import io
import json

import pytest
from PIL import Image

from handwriting_ocr.engine import run_recognition


@pytest.mark.parametrize("pipeline", ["ocr", "triage"])
def test_only_fragment_retried_and_surrounding_text_preserved(tmp_path, pipeline):
    path = tmp_path / "page.png"
    Image.new("RGB", (1600, 1200), "white").save(path)
    calls = []
    draft = "Верх\n[?bbox:100,200,400,300|слово?]\nНиз [?неясно?]"

    def chat(prompt, b64, mime):
        calls.append(Image.open(io.BytesIO(base64.b64decode(b64))).size)
        if len(calls) == 1:
            return json.dumps({"kind": "text", "raw_text": draft}) if pipeline == "triage" else draft
        return "молоко"

    result = run_recognition(model="test", image_path=path, chat=chat, image_max_edge=512, max_passes=3, pipeline=pipeline)
    assert calls == [(512, 384), (480, 120)]
    assert result["passes"] == 2
    assert result["raw_text"] == "Верх\nмолоко\nНиз [?неясно?]"


@pytest.mark.parametrize("marker", ["[?неясно?]", "[?bbox:0,0,1000,1000|неясно?]", "[?bbox:200,0,100,50|неясно?]", "[?bbox:0,0,1001,50|неясно?]"])
def test_invalid_or_missing_localization_does_not_repeat_page(tmp_path, marker):
    path = tmp_path / "page.png"
    Image.new("RGB", (600, 800), "white").save(path)
    calls = []
    def chat(*args):
        calls.append(args)
        return marker
    result = run_recognition(model="test", image_path=path, chat=chat, pipeline="ocr")
    assert len(calls) == result["passes"] == 1
    assert result["raw_text"] == "[?неясно?]"


@pytest.mark.parametrize("reply", ["", "[?неясно?]", RuntimeError("backend unavailable")])
def test_failed_or_uncertain_retry_keeps_first_draft(tmp_path, reply):
    path = tmp_path / "page.png"
    Image.new("RGB", (600, 800), "white").save(path)
    calls = []
    def chat(*args):
        calls.append(args)
        if len(calls) == 1:
            return "Текст [?bbox:10,20,100,200|неясно?] конец"
        if isinstance(reply, Exception):
            raise reply
        return reply
    result = run_recognition(model="test", image_path=path, chat=chat, pipeline="ocr")
    assert result["raw_text"] == "Текст [?неясно?] конец"
    assert result["passes"] == 2


def test_retry_budget_and_disable(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (600, 800), "white").save(path)
    for limit, expected in [(1, 1), (2, 2), (99, 2)]:
        calls = []
        def chat(*args):
            calls.append(args)
            return "[?bbox:10,20,100,200|a?] [?bbox:200,300,400,500|b?]" if len(calls) == 1 else "А"
        result = run_recognition(model="test", image_path=path, chat=chat, pipeline="ocr", max_passes=limit)
        assert len(calls) == result["passes"] == expected
        assert "bbox" not in result["raw_text"]
