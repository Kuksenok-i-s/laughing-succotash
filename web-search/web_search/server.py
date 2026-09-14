"""The HTTP surface.

    POST   /v1/search   {"queries": [...], "limit": 5, "lang": "ru"} -> results per query
    POST   /v1/fetch    {"url": "..."} -> readable text for one page
    GET    /health      no token required

Every call answers in the same request: unlike the STT and OCR services there is no job registry,
because the work takes a fraction of a second and a poll would cost more than it does.
"""

from __future__ import annotations

import hmac
import json
import logging
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .config import Settings
from .service import SearchService, UnsupportedContent
from .transport import UpstreamError
from .urls import UrlRefused

log = logging.getLogger(__name__)


class SearchApp:
    def __init__(self, settings: Settings, service: SearchService) -> None:
        self.settings = settings
        self.service = service

    def authorized(self, header: str | None) -> bool:
        if not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[len("Bearer ") :].strip(), self.settings.token)

    def health(self) -> dict[str, Any]:
        return self.service.health()


class SearchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: SearchApp) -> None:
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
    server_version = "web-search"
    sys_version = ""

    @property
    def app(self) -> SearchApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._reply(HTTPStatus.OK, self.app.health())
            return
        if not self._require_token():
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "no such endpoint")

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_token():
            return
        path = urlparse(self.path).path
        if path == "/v1/search":
            self._search()
            return
        if path == "/v1/fetch":
            self._fetch()
            return
        self._drain_body()
        self._error(HTTPStatus.NOT_FOUND, "not_found", "no such endpoint")

    def _search(self) -> None:
        body = self._read_json()
        if body is None:
            return

        queries = _queries_of(body)
        if queries is None:
            self._error(
                HTTPStatus.BAD_REQUEST, "bad_request", "queries must be a non-empty list of strings"
            )
            return
        if len(queries) > self.app.settings.max_queries:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "too_many_queries",
                f"{len(queries)} queries is over the configured limit of "
                f"{self.app.settings.max_queries}",
            )
            return

        limit = body.get("limit")
        if limit is not None and not isinstance(limit, int):
            self._error(HTTPStatus.BAD_REQUEST, "bad_request", "limit must be an integer")
            return
        lang = body.get("lang")
        if lang is not None and not isinstance(lang, str):
            self._error(HTTPStatus.BAD_REQUEST, "bad_request", "lang must be a string")
            return

        try:
            payload = self.app.service.search(queries, limit=limit, lang=lang)
        except Exception as exc:
            log.exception("search failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "search_failed", str(exc))
            return
        self._reply(HTTPStatus.OK, payload)

    def _fetch(self) -> None:
        body = self._read_json()
        if body is None:
            return
        if not self.app.settings.fetch_enabled:
            self._error(HTTPStatus.FORBIDDEN, "fetch_disabled", "page fetching is disabled")
            return

        url = body.get("url")
        if not isinstance(url, str) or not url.strip():
            self._error(HTTPStatus.BAD_REQUEST, "bad_request", "url must be a non-empty string")
            return

        try:
            payload = self.app.service.fetch(url.strip())
        except UrlRefused as exc:
            self._error(HTTPStatus.BAD_REQUEST, "url_refused", str(exc))
        except UnsupportedContent as exc:
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_content", str(exc))
        except UpstreamError as exc:
            self._error(HTTPStatus.BAD_GATEWAY, "upstream_error", str(exc))
        except Exception as exc:
            log.exception("fetch failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "fetch_failed", str(exc))
        else:
            self._reply(HTTPStatus.OK, payload)

    # ---- request plumbing ------------------------------------------------

    def _read_json(self) -> dict[str, Any] | None:
        """The decoded request object, or ``None`` after an error has been sent."""
        length = self.headers.get("Content-Length")
        if length is None:
            self._error(
                HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required"
            )
            return None
        try:
            size = int(length)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "bad_length", "Content-Length is not a number")
            return None
        if size <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "empty_body", "no JSON in the request")
            return None
        if size > self.app.settings.max_body_bytes:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "body_too_large",
                f"{size} bytes is over the configured limit",
            )
            return None

        raw = self.rfile.read(size)
        if len(raw) != size:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "body_incomplete",
                f"expected {size} bytes, received {len(raw)}",
            )
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, "bad_json", f"could not parse the body: {exc}")
            return None
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "bad_json", "the body must be a JSON object")
            return None
        return payload

    def _drain_body(self) -> None:
        length = self.headers.get("Content-Length")
        if not length:
            return
        try:
            remaining = int(length)
        except ValueError:
            return
        while remaining > 0:
            chunk = self.rfile.read(min(64 * 1024, remaining))
            if not chunk:
                return
            remaining -= len(chunk)

    def _require_token(self) -> bool:
        if self.app.authorized(self.headers.get("Authorization")):
            return True
        self._drain_body()
        self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "a valid bearer token is required")
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


def _queries_of(body: dict[str, Any]) -> list[str] | None:
    """Accept a batch or a single query; reject anything that is not a list of real strings."""
    raw = body.get("queries")
    if raw is None and isinstance(body.get("query"), str):
        raw = [body["query"]]
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(item, str) for item in raw):
        return None
    queries = [item for item in raw if item.strip()]
    return queries or None
