from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from handwriting_ocr.engine import LlamaCppEngine, pick_best_draft, strip_repeats
from handwriting_ocr.main import build_engine
from handwriting_ocr.config import from_env


class _FakeLlama:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict | None]] = []
        self.replies: list[str] = [
            '{"kind":"text","raw_text":"купить молоко"}',
            "купить молоко",
            "# Список\n\n- купить молоко",
        ]

    def next_reply(self) -> str:
        if not self.replies:
            return ""
        return self.replies.pop(0)


def _serve(fake: _FakeLlama):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            fake.requests.append((self.command, self.path, None))
            body = json.dumps({"data": [{"id": "qwen3-vl-2b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            fake.requests.append((self.command, self.path, payload))
            reply = fake.next_reply()
            body = json.dumps(
                {"choices": [{"message": {"content": reply}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_llamacpp_engine_sends_openai_image_url(tmp_path: Path) -> None:
    fake = _FakeLlama()
    server, thread = _serve(fake)
    try:
        engine = LlamaCppEngine(
            llama_url=f"http://127.0.0.1:{server.server_address[1]}",
            model="qwen3-vl-2b",
            request_timeout=5.0,
        )
        image = tmp_path / "note.jpg"
        image.write_bytes(b"\xff\xd8\xffpretend jpeg")
        result = engine.recognize(image)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert result["kind"] == "text"
    assert result["passes"] == 3
    assert result["raw_text"] == "купить молоко"
    assert result["markdown"].startswith("# Список")
    posts = [item for item in fake.requests if item[0] == "POST"]
    assert len(posts) == 3
    first = posts[0][2]
    assert first is not None
    content = first["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert first["max_tokens"] == 2048


def test_llamacpp_engine_can_stop_after_triage(tmp_path: Path) -> None:
    fake = _FakeLlama()
    server, thread = _serve(fake)
    try:
        engine = LlamaCppEngine(
            llama_url=f"http://127.0.0.1:{server.server_address[1]}",
            model="qwen3-vl-2b",
            request_timeout=5.0,
            max_passes=1,
        )
        image = tmp_path / "note.jpg"
        image.write_bytes(b"\xff\xd8\xffpretend jpeg")
        result = engine.recognize(image)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert result["kind"] == "text"
    assert result["passes"] == 1
    assert result["raw_text"] == "купить молоко"
    assert result["markdown"] == "купить молоко"
    posts = [item for item in fake.requests if item[0] == "POST"]
    assert len(posts) == 1


def test_strip_repeats_cuts_trailing_zero_lines() -> None:
    body = "ИТОГО =2100.08\nскидка =737.02\n" + "\n".join(["00"] * 20)
    assert strip_repeats(body).endswith("737.02")
    assert strip_repeats(body).count("00") == 1  # only the 2100.08


def test_pick_best_draft_keeps_color_unless_another_is_much_richer() -> None:
    close = {
        "color": "you and you alone are the keeper",
        "gray": "year and you alone are the keeper!",
        "ink": "you",
    }
    assert pick_best_draft(close).startswith("you and you")
    richer = {
        "color": "итог 10",
        "gray": "пакет 6.99 картофель 34.89 мандарины 79.99 джин 429.99 итого 2100.08",
        "ink": "x",
    }
    assert "2100.08" in pick_best_draft(richer)


def test_strip_repeats_keeps_a_normal_letter() -> None:
    letter = "you are the fond object of\nmy affection and my desire.\nwith love, gilbert."
    assert strip_repeats(letter) == letter


def test_llamacpp_ocr_pipeline_is_one_shot(tmp_path: Path) -> None:
    fake = _FakeLlama()
    fake.replies = ["ИТОГО =2100.08\n" + "\n".join(["00"] * 12)]
    server, thread = _serve(fake)
    try:
        engine = LlamaCppEngine(
            llama_url=f"http://127.0.0.1:{server.server_address[1]}",
            model="ovis-ocr2",
            request_timeout=5.0,
            pipeline="ocr",
            max_passes=1,
        )
        image = tmp_path / "note.jpg"
        image.write_bytes(b"\xff\xd8\xffpretend jpeg")
        result = engine.recognize(image)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert result["kind"] == "text"
    assert result["passes"] == 1
    assert result["raw_text"].endswith("2100.08")
    posts = [item for item in fake.requests if item[0] == "POST"]
    assert len(posts) == 1
    first = posts[0][2]
    assert first is not None
    content = first["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert "Extract all readable content" in content[1]["text"]
    assert first["repeat_penalty"] == 1.15
    assert first["max_tokens"] == 2048


def test_llamacpp_ocr_ensemble_merges_three_preprocessed_views(tmp_path: Path) -> None:
    fake = _FakeLlama()
    fake.replies = ["цвет", "серый длиннее текст", "тушь"]
    server, thread = _serve(fake)
    try:
        engine = LlamaCppEngine(
            llama_url=f"http://127.0.0.1:{server.server_address[1]}",
            model="ovis-ocr2",
            request_timeout=5.0,
            pipeline="ocr",
            max_passes=3,
        )
        image = tmp_path / "note.jpg"
        from PIL import Image as PilImage

        PilImage.new("RGB", (40, 20), (200, 180, 160)).save(image, format="JPEG")
        result = engine.recognize(image)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert result["kind"] == "text"
    assert result["passes"] == 3
    assert result["raw_text"] == "серый длиннее текст"
    posts = [item for item in fake.requests if item[0] == "POST"]
    assert len(posts) == 3
    for post in posts:
        assert post[2] is not None
        types = [part["type"] for part in post[2]["messages"][0]["content"]]
        assert types[0] == "image_url"


def test_build_engine_picks_llamacpp() -> None:
    settings = from_env(
        {
            "OCR_TOKEN": "t" * 40,
            "OCR_BACKEND": "llamacpp",
            "OCR_LLAMA_URL": "http://127.0.0.1:8081",
            "OCR_MODEL": "qwen3-vl-2b",
        }
    )
    engine = build_engine(settings)
    assert isinstance(engine, LlamaCppEngine)
    assert engine.model_name == "qwen3-vl-2b"
