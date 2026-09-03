"""The HTTP surface: job API plus model load/unload.

    PUT    /v1/jobs/{id}         raw image body, query: filename, content_type
    GET    /v1/jobs/{id}         status and progress
    GET    /v1/jobs/{id}/result  markdown + raw_text, once status is done
    DELETE /v1/jobs/{id}         forget the job and remove its image
    POST   /v1/model/load        load Qwen3-VL into Ollama VRAM
    POST   /v1/model/unload      unload immediately (keep_alive=0)
    GET    /health               no token required
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import Settings
from .engine import Engine
from .jobs import Job, JobStore, is_safe_job_id

log = logging.getLogger(__name__)

_JOB = re.compile(r"\A/v1/jobs/(?P<job_id>[^/]+)\Z")
_JOB_RESULT = re.compile(r"\A/v1/jobs/(?P<job_id>[^/]+)/result\Z")

_IMAGE_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/heic",
        "image/heif",
        "application/octet-stream",
    }
)


class OcrApp:
    def __init__(self, settings: Settings, store: JobStore, engine: Engine) -> None:
        self.settings = settings
        self.store = store
        self.engine = engine

    def authorized(self, header: str | None) -> bool:
        if not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[len("Bearer ") :].strip(), self.settings.token)

    def health(self) -> dict[str, Any]:
        backend_ok = getattr(self.engine, "backend_reachable", False)
        if hasattr(self.engine, "probe"):
            backend_ok = self.engine.probe()
        return {
            "status": "ok",
            "model": self.engine.model_name,
            "model_loaded": self.engine.ready,
            "backend_reachable": bool(backend_ok),
            # Kept for older Core /health readers; same value as backend_reachable.
            "ollama_reachable": bool(backend_ok),
            "queued": self.store.queue_depth(),
        }


class OcrServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: OcrApp) -> None:
        self.app = app
        super().__init__(address, Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError | TimeoutError):
            log.debug("client %s went away: %s", client_address[0], exc)
            return
        log.exception("error while handling a request from %s", client_address[0])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "handwriting-ocr"
    sys_version = ""

    @property
    def app(self) -> OcrApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._reply(HTTPStatus.OK, self.app.health())
            return
        if not self._require_token():
            return

        match = _JOB_RESULT.match(path)
        if match:
            self._get_result(match.group("job_id"))
            return
        match = _JOB.match(path)
        if match:
            self._get_job(match.group("job_id"))
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "no such endpoint")

    def do_PUT(self) -> None:  # noqa: N802
        if not self._require_token():
            return
        match = _JOB.match(urlparse(self.path).path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "no such endpoint")
            return
        self._put_job(match.group("job_id"))

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_token():
            return
        path = urlparse(self.path).path
        if path == "/v1/model/load":
            self._model_load()
            return
        if path == "/v1/model/unload":
            self._model_unload()
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "no such endpoint")

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._require_token():
            return
        match = _JOB.match(urlparse(self.path).path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "no such endpoint")
            return
        job_id = match.group("job_id")
        if not self._valid_id(job_id):
            return
        if self.app.store.delete(job_id):
            self._reply(HTTPStatus.OK, {"deleted": True})
        else:
            self._error(HTTPStatus.NOT_FOUND, "job_not_found", "no such job")

    def _get_job(self, job_id: str) -> None:
        if not self._valid_id(job_id):
            return
        job = self.app.store.get(job_id)
        if job is None:
            self._error(HTTPStatus.NOT_FOUND, "job_not_found", "no such job")
            return
        self._reply(HTTPStatus.OK, job.snapshot())

    def _get_result(self, job_id: str) -> None:
        if not self._valid_id(job_id):
            return
        job = self.app.store.get(job_id)
        if job is None:
            self._error(HTTPStatus.NOT_FOUND, "job_not_found", "no such job")
            return
        if job.status == "failed":
            self._error(HTTPStatus.CONFLICT, "job_failed", job.error or "ocr failed")
            return
        if job.result is None:
            self._reply(HTTPStatus.CONFLICT, job.snapshot())
            return
        self._reply(HTTPStatus.OK, job.result)

    def _model_load(self) -> None:
        self._drain_body()
        try:
            self.app.engine.load()
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "model_load_failed", str(exc))
            return
        self._reply(
            HTTPStatus.OK,
            {"model": self.app.engine.model_name, "model_loaded": self.app.engine.ready},
        )

    def _model_unload(self) -> None:
        self._drain_body()
        try:
            self.app.engine.unload()
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "model_unload_failed", str(exc))
            return
        self._reply(
            HTTPStatus.OK,
            {"model": self.app.engine.model_name, "model_loaded": self.app.engine.ready},
        )

    def handle_expect_100(self) -> bool:
        if not self._require_token():
            return False
        if self._upload_length() is None:
            return False
        self.send_response_only(HTTPStatus.CONTINUE)
        self.end_headers()
        return True

    def _upload_length(self) -> int | None:
        length = self.headers.get("Content-Length")
        if length is None:
            self._error(
                HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required"
            )
            return None
        try:
            remaining = int(length)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "bad_length", "Content-Length is not a number")
            return None
        if remaining <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "empty_body", "no image in the request")
            return None
        if remaining > self.app.settings.max_upload_bytes:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "image_too_large",
                f"{remaining} bytes is over the configured limit",
            )
            return None
        return remaining

    def _put_job(self, job_id: str) -> None:
        if not self._valid_id(job_id):
            return

        query = parse_qs(urlparse(self.path).query)
        filename = _first(query, "filename") or job_id
        content_type = (
            _first(query, "content_type")
            or self.headers.get("Content-Type")
            or "application/octet-stream"
        ).split(";")[0].strip().lower()
        if content_type not in _IMAGE_TYPES and not content_type.startswith("image/"):
            remaining = self._upload_length()
            if remaining is not None:
                self._drain(remaining)
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "invalid_image",
                f"content type {content_type!r} is not an image",
            )
            return

        remaining = self._upload_length()
        if remaining is None:
            return

        known = self.app.store.get(job_id)
        if known is not None:
            self._drain(remaining)
            self._reply(HTTPStatus.OK, known.snapshot())
            return

        target = self.app.store.prepare_spool(job_id)
        try:
            written = self._spool(target, remaining)
        except (OSError, ConnectionError) as exc:
            self.app.store.discard_spool(job_id)
            log.warning("upload for job %s failed: %s", job_id, exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "upload_failed", str(exc))
            return

        if written != remaining:
            self.app.store.discard_spool(job_id)
            self._error(
                HTTPStatus.BAD_REQUEST,
                "upload_incomplete",
                f"expected {remaining} bytes, received {written}",
            )
            return

        job = self.app.store.submit(
            Job(
                job_id=job_id,
                image_path=target,
                filename=filename,
                content_type=content_type,
            )
        )
        self._reply(HTTPStatus.ACCEPTED, job.snapshot())

    def _spool(self, target, remaining: int) -> int:
        chunk_size = self.app.settings.upload_chunk_size
        written = 0
        with target.open("wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(chunk_size, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
        return written

    def _drain(self, remaining: int) -> None:
        chunk_size = self.app.settings.upload_chunk_size
        while remaining > 0:
            chunk = self.rfile.read(min(chunk_size, remaining))
            if not chunk:
                return
            remaining -= len(chunk)

    def _drain_body(self) -> None:
        length = self.headers.get("Content-Length")
        if not length:
            return
        try:
            remaining = int(length)
        except ValueError:
            return
        if remaining > 0:
            self._drain(remaining)

    def _require_token(self) -> bool:
        if self.app.authorized(self.headers.get("Authorization")):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "a valid bearer token is required")
        return False

    def _valid_id(self, job_id: str) -> bool:
        if is_safe_job_id(job_id):
            return True
        self._error(HTTPStatus.BAD_REQUEST, "bad_job_id", "job id is not acceptable")
        return False

    def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, code: str, message: str) -> None:
        self.close_connection = True
        self._reply(status, {"code": code, "message": message})

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("http %s", fmt % args)

    def log_error(self, fmt: str, *args: Any) -> None:
        log.warning("http %s", fmt % args)


def _first(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return values[0] if values else None
