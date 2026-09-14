"""Test doubles and a minimal HTTP client."""

from __future__ import annotations

import http.client
import json
import threading
from dataclasses import dataclass
from typing import Any

from web_search.backends import Result
from web_search.transport import UpstreamError

TOKEN = "t" * 40


class FakeBackend:
    """Records every query and answers from a script. Never touches the network."""

    def __init__(
        self,
        *,
        results: list[Result] | None = None,
        error: Exception | None = None,
        errors: dict[str, Exception] | None = None,
        reachable: bool = True,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self._results = results if results is not None else [
            Result(title="Пример", url="https://example.com/a", excerpt="выдержка"),
        ]
        self._error = error
        self._errors = errors or {}
        self._reachable = reachable
        # When set, every call waits for the others: satisfied only if they truly overlap.
        self._barrier = barrier
        self.queries: list[tuple[str, int, str]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fake"

    def probe(self) -> bool:
        return self._reachable

    def search(self, query: str, limit: int, *, lang: str) -> list[Result]:
        with self._lock:
            self.queries.append((query, limit, lang))
        if self._barrier is not None:
            self._barrier.wait(timeout=5.0)
        failure = self._errors.get(query, self._error)
        if failure is not None:
            raise failure
        return list(self._results[:limit])

    def asked(self) -> list[str]:
        with self._lock:
            return [query for query, _, _ in self.queries]


def upstream_error(message: str = "HTTP 429 rate limited") -> UpstreamError:
    return UpstreamError(message, status=429)


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

    def post(self, path: str, payload: Any, **kwargs: Any) -> Response:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **kwargs.pop("headers", {})}
        return self.request("POST", path, body=body, headers=headers, **kwargs)
