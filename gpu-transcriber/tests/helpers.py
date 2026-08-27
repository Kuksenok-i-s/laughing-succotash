"""Test doubles and a minimal HTTP client.

The engine is faked so CUDA and a three gigabyte model stay out of the suite; everything else is
the real thing, because most of what can go wrong in this service is HTTP.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN = "t" * 40


def wait_for(predicate: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    """Poll a condition the GPU thread is responsible for reaching."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeEngine:
    """Emits scripted segments, and can be held mid-transcription so progress is observable."""

    def __init__(
        self,
        segments: list[tuple[float, float, str]] | None = None,
        *,
        duration: float = 60.0,
        error: Exception | None = None,
        load_error: Exception | None = None,
        pause_after: int | None = None,
    ) -> None:
        self._segments = segments or [(0.0, 30.0, "первая"), (30.0, 60.0, "вторая")]
        self._duration = duration
        self._error = error
        self._load_error = load_error
        self._pause_after = pause_after
        self.resume = threading.Event()
        self.reached_pause = threading.Event()
        self.loads = 0
        self.unloads = 0
        self.calls: list[dict[str, Any]] = []
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return "fake-large-v3"

    def load(self) -> None:
        self.loads += 1
        if self._load_error is not None:
            raise self._load_error
        self._ready = True

    def unload(self) -> None:
        self.unloads += 1
        self._ready = False

    def transcribe(self, audio_path, *, language, beam_size, on_progress=None):
        if not self._ready:
            self.load()
        self.calls.append(
            {
                "audio": Path(audio_path),
                "language": language,
                "beam_size": beam_size,
                "bytes": Path(audio_path).stat().st_size,
            }
        )
        if self._error is not None:
            raise self._error

        collected = []
        for index, (start, end, text) in enumerate(self._segments, start=1):
            collected.append({"start": start, "end": end, "text": text})
            if on_progress is not None:
                on_progress(
                    min(100.0, end / self._duration * 100.0), end, self._duration, index
                )
            if self._pause_after == index:
                self.reached_pause.set()
                self.resume.wait(timeout=10.0)

        return {
            "text": " ".join(item["text"] for item in collected),
            "language": language or "ru",
            "duration": self._duration,
            "segments": collected,
        }


@dataclass
class Response:
    status: int
    payload: dict[str, Any] | None = None
    raw: bytes = b""


@dataclass
class Client:
    host: str
    port: int
    token: str = TOKEN

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        token: str | None = "default",
        headers: dict[str, str] | None = None,
    ) -> Response:
        sent = dict(headers or {})
        if token == "default":
            token = self.token
        if token is not None:
            sent["Authorization"] = f"Bearer {token}"
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            raw = response.read()
            status = response.status
        finally:
            connection.close()
        payload = None
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
        return Response(status=status, payload=payload, raw=raw)

    def put_audio(self, job_id: str, data: bytes = b"pretend audio", **query: str) -> Response:
        suffix = "&".join(f"{key}={value}" for key, value in query.items())
        path = f"/v1/jobs/{job_id}" + (f"?{suffix}" if suffix else "")
        return self.request("PUT", path, body=data)
