"""The GPU service backend, against a real HTTP server in a thread.

Mocking aiohttp here would test the mock. What broke in the pipeline this replaces was never the
happy path: it was a progress callback arriving on the wrong thread, a dead host that looked like a
slow one, and audio nobody deleted. All three are visible only with sockets involved.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from agent_core.stt.base import SttError
from agent_core.stt.fallback import FallbackSTT
from agent_core.stt.gpu_service import GpuServiceSTT

TOKEN = "t" * 40

DONE_RESULT = {
    "text": "готово",
    "language": "ru",
    "duration": 12.0,
    "segments": [{"start": 0.0, "end": 12.0, "text": "готово"}],
}


class Stub:
    """A scripted transcription service."""

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
                {"status": "running", "percent": 25.0},
                {"status": "running", "percent": 75.0},
                {"status": "done", "percent": 100.0},
            ]
        )
        self.result = DONE_RESULT if result is None else result
        self.model_loaded = model_loaded
        self.uploads: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.tokens: list[str | None] = []

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
            self._send(200, {"status": "ok", "model_loaded": self.stub.model_loaded})
            return
        self.stub.tokens.append(self.headers.get("Authorization"))
        if path.endswith("/result"):
            self._send(200, self.stub.result)
            return
        self._send(200, self.stub.next_status())

    def do_PUT(self) -> None:  # noqa: N802
        query = parse_qs(urlparse(self.path).query)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.stub.uploads.append(
            {
                "job_id": urlparse(self.path).path.rsplit("/", 1)[-1],
                "bytes": body,
                "language": (query.get("language") or [None])[0],
                "beam_size": (query.get("beam_size") or [None])[0],
                "filename": (query.get("filename") or [None])[0],
            }
        )
        self._send(202, {"status": "queued", "percent": 0.0})

    def do_DELETE(self) -> None:  # noqa: N802
        self.stub.deleted.append(urlparse(self.path).path.rsplit("/", 1)[-1])
        self._send(200, {"deleted": True})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return None


@pytest.fixture
def stub() -> Stub:
    return Stub()


@pytest.fixture
def service(stub: Stub) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.stub = stub  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def _backend(base_url: str, **kwargs: Any) -> GpuServiceSTT:
    options: dict[str, Any] = {"poll_interval": 0.01, "request_timeout": 5.0}
    options.update(kwargs)
    return GpuServiceSTT(base_url=base_url, token=TOKEN, **options)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"pretend audio")
    return path


async def test_progress_arrives_where_coroutines_may_be_scheduled(
    service: str, tmp_path: Path
) -> None:
    """The hook's contract, and the exact thing the SSH backend got wrong."""
    stt = _backend(service)
    fractions: list[float] = []
    threads: list[str] = []
    reported: list[float] = []

    async def report(fraction: float) -> None:
        reported.append(fraction)

    def on_progress(fraction: float) -> None:
        threads.append(threading.current_thread().name)
        fractions.append(fraction)
        asyncio.ensure_future(report(fraction))

    result = await stt.transcribe(_audio(tmp_path), on_progress=on_progress)
    await asyncio.sleep(0.05)
    await stt.close()

    assert result.text == "готово"
    assert result.language == "ru"
    assert len(result.segments) == 1
    # Only changes are reported: an unchanged percentage would make the Gateway re-edit a status
    # message with identical text, which Telegram rejects.
    assert fractions == [0.25, 0.75, 1.0]
    assert reported == fractions
    assert set(threads) == {threading.current_thread().name}


async def test_the_upload_carries_the_audio_and_the_settings(
    service: str, stub: Stub, tmp_path: Path
) -> None:
    stt = _backend(service, language="auto", beam_size=3)

    await stt.transcribe(_audio(tmp_path))
    await stt.close()

    assert len(stub.uploads) == 1
    upload = stub.uploads[0]
    assert upload["bytes"] == b"pretend audio"
    # "auto" is passed through instead of a hardcoded language: the SSH script could only be given
    # a fixed one, so every transcript claimed to be Russian.
    assert upload["language"] == "auto"
    assert upload["beam_size"] == "3"
    assert upload["filename"] == "voice.ogg"
    assert stub.tokens and all(token == f"Bearer {TOKEN}" for token in stub.tokens)


async def test_a_collected_job_is_deleted(service: str, stub: Stub, tmp_path: Path) -> None:
    stt = _backend(service)

    await stt.transcribe(_audio(tmp_path))
    await stt.close()

    assert stub.deleted == [stub.uploads[0]["job_id"]]


async def test_a_failed_job_is_deleted_too(service: str, stub: Stub, tmp_path: Path) -> None:
    """Otherwise a failing GPU host quietly fills its disk with abandoned audio."""
    stub.statuses = deque([{"status": "failed", "percent": 0.0, "error": "cuda oom"}])
    stt = _backend(service)

    with pytest.raises(SttError, match="cuda oom"):
        await stt.transcribe(_audio(tmp_path))
    await stt.close()

    assert stub.deleted == [stub.uploads[0]["job_id"]]


async def test_a_stalled_job_gives_up(service: str, stub: Stub, tmp_path: Path) -> None:
    stub.statuses = deque([{"status": "running", "percent": 10.0}])
    stt = _backend(service, stall_timeout=0.05)

    with pytest.raises(SttError, match="stalled"):
        await stt.transcribe(_audio(tmp_path))
    await stt.close()


async def test_an_absent_service_fails_instead_of_hanging(tmp_path: Path) -> None:
    # Port 1 on loopback: nothing listens, and the connection is refused immediately.
    stt = _backend("http://127.0.0.1:1")

    with pytest.raises(SttError, match="unreachable"):
        await stt.warmup()
    await stt.close()


async def test_a_model_still_loading_is_not_a_reason_to_give_up(
    service: str, stub: Stub, tmp_path: Path
) -> None:
    """The service listens before the weights are in memory; refusing then would mean CPU forever."""
    stub.model_loaded = False
    stt = _backend(service)

    await stt.warmup()

    assert stt.ready
    await stt.close()


async def test_a_service_failure_hands_the_job_to_the_cpu(
    service: str, stub: Stub, tmp_path: Path
) -> None:
    from agent_core.stt.base import STT_CPU_FALLBACK, TranscriptionResult, TranscriptSegment

    class CpuStt:
        model_name = "cpu"
        ready = True
        calls: list[Path] = []

        async def warmup(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def transcribe(self, path: Path, *, on_progress=None, on_notice=None):
            self.calls.append(path)
            return TranscriptionResult(
                text="на процессоре",
                language="ru",
                duration=1.0,
                segments=[TranscriptSegment(0.0, 1.0, "на процессоре")],
            )

    stub.statuses = deque([{"status": "failed", "percent": 0.0, "error": "cuda oom"}])
    cpu = CpuStt()
    stt = FallbackSTT(primary=_backend(service), fallback=cpu)
    notices: list[str] = []

    audio = _audio(tmp_path)
    result = await stt.transcribe(audio, on_notice=notices.append)

    assert result.text == "на процессоре"
    assert cpu.calls == [audio]
    assert notices == [STT_CPU_FALLBACK]
