"""Remote OCR client against a real HTTP stub in a thread."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from agent_core.ocr.base import OcrError
from agent_core.ocr.remote_service import RemoteOcrService

TOKEN = "t" * 40

DONE_RESULT = {
    "kind": "text",
    "raw_text": "купить молоко [?вечером?]",
    "markdown": "# Список\n\n- купить молоко [?вечером?]",
    "description": "",
    "model": "qwen3-vl:2b",
    "elapsed_seconds": 2.5,
    "passes": 3,
}


class Stub:
    def __init__(
        self,
        statuses: list[dict[str, Any]] | None = None,
        *,
        result: dict[str, Any] | None = None,
        model_loaded: bool = True,
    ) -> None:
        self.statuses = deque(
            statuses
            if statuses is not None
            else [
                {"status": "running", "percent": 20.0, "stage": "recognizing"},
                {"status": "running", "percent": 70.0, "stage": "structuring"},
                {"status": "done", "percent": 100.0, "stage": "completed"},
            ]
        )
        self.result = DONE_RESULT if result is None else result
        self.model_loaded = model_loaded
        self.uploads: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def next_status(self) -> dict[str, Any]:
        if len(self.statuses) > 1:
            return self.statuses.popleft()
        return self.statuses[0]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def stub(self) -> Stub:
        return self.server.stub  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {
                "status": "ok",
                "model": "qwen3-vl",
                "model_loaded": self.stub.model_loaded,
                "ollama_reachable": True,
                "queued": 0,
            })
            return
        if path.endswith("/result"):
            status = self.stub.statuses[-1] if self.stub.statuses else {"status": "done"}
            if status.get("status") != "done":
                self._send(409, status)
                return
            self._send(200, self.stub.result)
            return
        self._send(200, self.stub.next_status())

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        query = parse_qs(parsed.query)
        self.stub.uploads.append(
            {
                "path": parsed.path,
                "bytes": len(body),
                "filename": (query.get("filename") or [""])[0],
                "content_type": (query.get("content_type") or [""])[0],
            }
        )
        self._send(202, {"status": "queued", "percent": 0.0, "stage": "queued"})

    def do_DELETE(self) -> None:  # noqa: N802
        job_id = urlparse(self.path).path.rsplit("/", 1)[-1]
        self.stub.deleted.append(job_id)
        self._send(200, {"deleted": True})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


@pytest.fixture
def stub_server() -> Iterator[tuple[Stub, str]]:
    stub = Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.stub = stub  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_recognize_uploads_polls_and_deletes(stub_server, tmp_path: Path) -> None:
    stub, base = stub_server
    image = tmp_path / "note.jpg"
    image.write_bytes(b"jpeg-bytes")
    client = RemoteOcrService(base_url=base, token=TOKEN, poll_interval=0.01)

    result = await client.recognize(image, content_type="image/jpeg")
    await client.close()

    assert result.raw_text.startswith("купить")
    assert result.markdown.startswith("#")
    assert result.passes == 3
    assert result.kind == "text"
    assert stub.uploads[0]["bytes"] == len(b"jpeg-bytes")
    assert stub.uploads[0]["content_type"] == "image/jpeg"
    assert stub.deleted


@pytest.mark.asyncio
async def test_unreachable_service_is_an_ocr_error(tmp_path: Path) -> None:
    image = tmp_path / "note.jpg"
    image.write_bytes(b"x")
    client = RemoteOcrService(base_url="http://127.0.0.1:1", token=TOKEN, request_timeout=0.2)

    with pytest.raises(OcrError, match="unreachable|failed"):
        await client.warmup()
    await client.close()
