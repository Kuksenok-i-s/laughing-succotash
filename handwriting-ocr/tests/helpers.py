"""Test doubles and a minimal HTTP client."""

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
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeEngine:
    """Records each Ollama-shaped call so tests can assert triage vs OCR passes."""

    def __init__(
        self,
        *,
        raw_text: str = "сырой текст [?неясно?]",
        markdown: str = "# Заметка\n\n- задача",
        kind: str = "text",
        description: str = "a desk lamp",
        error: Exception | None = None,
        load_error: Exception | None = None,
        pause_after_pass: int | None = None,
    ) -> None:
        self._raw_text = raw_text
        self._markdown = markdown
        self._kind = kind
        self._description = description
        self._error = error
        self._load_error = load_error
        self._pause_after_pass = pause_after_pass
        self.resume = threading.Event()
        self.reached_pause = threading.Event()
        self.loads = 0
        self.unloads = 0
        self.calls: list[dict[str, Any]] = []
        self._ready = False
        self._ollama_ok = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return "fake-qwen3-vl"

    @property
    def backend_reachable(self) -> bool:
        return self._ollama_ok

    @property
    def ollama_reachable(self) -> bool:
        return self._ollama_ok

    def probe(self) -> bool:
        return self._ollama_ok

    def load(self) -> None:
        self.loads += 1
        if self._load_error is not None:
            raise self._load_error
        self._ready = True

    def unload(self) -> None:
        self.unloads += 1
        self._ready = False

    def recognize(self, image_path, *, on_progress=None):
        if not self._ready:
            self.load()
        data = Path(image_path).read_bytes()
        self.calls.append({"pass": 1, "bytes": len(data), "stage": "triage"})
        if on_progress is not None:
            on_progress(5.0, "recognizing")
        if self._pause_after_pass == 1:
            self.reached_pause.set()
            self.resume.wait(timeout=10.0)
        if self._error is not None:
            raise self._error

        if self._kind == "other":
            if on_progress is not None:
                on_progress(100.0, "completed")
            return {
                "kind": "other",
                "raw_text": "",
                "markdown": "",
                "description": self._description,
                "model": self.model_name,
                "elapsed_seconds": 0.4,
                "passes": 1,
            }

        self.calls.append({"pass": 2, "bytes": len(data), "stage": "refine", "raw_text": self._raw_text})
        if on_progress is not None:
            on_progress(35.0, "recognizing")
        if self._pause_after_pass == 2:
            self.reached_pause.set()
            self.resume.wait(timeout=10.0)

        self.calls.append({"pass": 3, "bytes": len(data), "stage": "structure"})
        if on_progress is not None:
            on_progress(70.0, "structuring")
        if self._pause_after_pass == 3:
            self.reached_pause.set()
            self.resume.wait(timeout=10.0)
        if on_progress is not None:
            on_progress(100.0, "completed")

        return {
            "kind": "text",
            "raw_text": self._raw_text,
            "markdown": self._markdown,
            "description": "",
            "model": self.model_name,
            "elapsed_seconds": 1.0,
            "passes": 3,
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
        if body is not None and "Content-Length" not in sent:
            sent["Content-Length"] = str(len(body))
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

    def put_image(
        self,
        job_id: str,
        data: bytes = b"\xff\xd8\xffpretend jpeg",
        **query: str,
    ) -> Response:
        suffix = "&".join(f"{key}={value}" for key, value in query.items())
        path = f"/v1/jobs/{job_id}" + (f"?{suffix}" if suffix else "")
        return self.request(
            "PUT",
            path,
            body=data,
            headers={"Content-Type": query.get("content_type", "image/jpeg")},
        )
