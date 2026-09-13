"""The search client against a real HTTP stub in a thread.

The client is the Core's only way onto the web, so what is asserted here is mostly what it refuses
to pass on: a failed query must not look like an empty one, a page must not be fetched from a
private address, and a provider field the service grows must not reach the model by accident.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from agent_core.search.base import SearchError
from agent_core.search.remote_service import RemoteSearchProvider

TOKEN = "t" * 40

RESULT = {"title": "Пример", "url": "https://example.com/a", "excerpt": "выдержка"}


class Stub:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.search_status = 200
        self.search_payload: dict[str, Any] | None = None
        self.fetch_status = 200
        self.fetch_payload: dict[str, Any] = {
            "url": "https://example.com/a",
            "title": "Страница",
            "text": "текст страницы",
            "truncated": False,
        }

    def answer_search(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.search_payload is not None:
            return self.search_payload
        queries = body.get("queries") or []
        return {
            "backend": "brave",
            "cached": 0,
            "elapsed_seconds": 0.1,
            "queries": [
                {"query": query, "results": [dict(RESULT)], "cached": False, "error": None}
                for query in queries
            ],
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def stub(self) -> Stub:
        return self.server.stub  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._send(
                200,
                {
                    "status": "ok",
                    "backend": "brave",
                    "backend_reachable": True,
                    "fetch_enabled": True,
                    "cache": {"entries": 0, "hits": 0, "misses": 0},
                },
            )
            return
        self._send(404, {"code": "not_found", "message": "no such endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.stub.requests.append(
            {"path": path, "body": body, "auth": self.headers.get("Authorization")}
        )
        if path == "/v1/search":
            self._send(self.stub.search_status, self.stub.answer_search(body))
            return
        if path == "/v1/fetch":
            self._send(self.stub.fetch_status, self.stub.fetch_payload)
            return
        self._send(404, {"code": "not_found", "message": "no such endpoint"})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
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
    thread = threading.Thread(target=lambda: server.serve_forever(0.02), daemon=True)
    thread.start()
    try:
        yield stub, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def provider(stub_server) -> Iterator[tuple[Stub, RemoteSearchProvider]]:
    stub, base = stub_server
    client = RemoteSearchProvider(base_url=base, token=TOKEN, request_timeout=5.0)
    yield stub, client


@pytest.mark.asyncio
async def test_warmup_reads_the_backend_from_health(provider) -> None:
    stub, client = provider

    await client.warmup()
    ready, backend = client.ready, client.backend
    await client.close()

    assert ready is True
    assert backend == "brave"
    # Closing releases the session, so the provider stops claiming to be usable.
    assert client.ready is False


@pytest.mark.asyncio
async def test_search_returns_the_three_agreed_fields(provider) -> None:
    stub, client = provider

    results = await client.search("курс биткоина", limit=3)
    await client.close()

    assert results == [RESULT]
    request = stub.requests[0]
    assert request["path"] == "/v1/search"
    assert request["body"] == {"queries": ["курс биткоина"], "limit": 3, "lang": "ru"}
    assert request["auth"] == f"Bearer {TOKEN}"


@pytest.mark.asyncio
async def test_a_batch_is_one_request(provider) -> None:
    """The whole reason the provider is a service: three claims, one round trip."""
    stub, client = provider

    answers = await client.search_many(["первый", "второй", "третий"])
    await client.close()

    assert len(stub.requests) == 1
    assert [answer["query"] for answer in answers] == ["первый", "второй", "третий"]


@pytest.mark.asyncio
async def test_blank_queries_never_reach_the_service(provider) -> None:
    stub, client = provider

    assert await client.search_many(["   ", ""]) == []
    await client.close()

    assert stub.requests == []


@pytest.mark.asyncio
async def test_a_failed_query_in_a_batch_keeps_its_error(provider) -> None:
    stub, client = provider
    stub.search_payload = {
        "backend": "brave",
        "cached": 0,
        "queries": [
            {"query": "хороший", "results": [dict(RESULT)], "error": None},
            {"query": "плохой", "results": [], "error": "HTTP 429 rate limited"},
        ],
    }

    good, bad = await client.search_many(["хороший", "плохой"])
    await client.close()

    assert good["results"] == [RESULT]
    assert bad["results"] == []
    assert bad["error"] == "HTTP 429 rate limited"


@pytest.mark.asyncio
async def test_a_single_failed_query_is_an_error_not_an_empty_result(provider) -> None:
    """An empty list reads as 'nothing found', which is a different claim from 'could not look'."""
    stub, client = provider
    stub.search_payload = {
        "backend": "brave",
        "queries": [{"query": "x", "results": [], "error": "HTTP 429 rate limited"}],
    }

    with pytest.raises(SearchError, match="429"):
        await client.search("x")
    await client.close()


@pytest.mark.asyncio
async def test_extra_provider_fields_are_dropped(provider) -> None:
    """A service that grows a field must not start feeding it to the model by accident."""
    stub, client = provider
    stub.search_payload = {
        "queries": [
            {
                "query": "x",
                "results": [
                    {
                        "title": "Заголовок",
                        "url": "https://example.com/a",
                        "excerpt": "выдержка",
                        "profile": {"sponsored": True},
                        "thumbnail": "https://cdn.example.com/x.png",
                    }
                ],
                "error": None,
            }
        ]
    }

    results = await client.search("x")
    await client.close()

    assert results == [RESULT | {"title": "Заголовок"}]


@pytest.mark.asyncio
async def test_a_result_without_a_url_is_dropped(provider) -> None:
    stub, client = provider
    stub.search_payload = {
        "queries": [
            {
                "query": "x",
                "results": [{"title": "нет ссылки", "excerpt": "y"}, dict(RESULT)],
                "error": None,
            }
        ]
    }

    assert await client.search("x") == [RESULT]
    await client.close()


@pytest.mark.asyncio
async def test_a_batch_of_the_wrong_length_is_refused(provider) -> None:
    """Zipping a short answer onto a long request would attribute results to the wrong query."""
    stub, client = provider
    stub.search_payload = {"queries": [{"query": "a", "results": [], "error": None}]}

    with pytest.raises(SearchError, match="malformed"):
        await client.search_many(["a", "b"])
    await client.close()


@pytest.mark.asyncio
async def test_a_service_error_becomes_a_search_error(provider) -> None:
    stub, client = provider
    stub.search_status = 502
    stub.search_payload = {"code": "upstream_error", "message": "brave is down"}

    with pytest.raises(SearchError, match="brave is down"):
        await client.search("x")
    await client.close()


@pytest.mark.asyncio
async def test_a_five_hundred_marks_the_provider_not_ready(provider) -> None:
    stub, client = provider
    await client.warmup()
    stub.search_status = 500
    stub.search_payload = {"code": "search_failed", "message": "boom"}

    with pytest.raises(SearchError):
        await client.search("x")
    await client.close()

    assert client.ready is False


@pytest.mark.asyncio
async def test_fetch_returns_readable_text(provider) -> None:
    stub, client = provider

    page = await client.fetch("https://example.com/a")
    await client.close()

    assert page == {
        "url": "https://example.com/a",
        "title": "Страница",
        "text": "текст страницы",
        "truncated": False,
    }
    assert stub.requests[0]["body"] == {"url": "https://example.com/a"}


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8931/mcp",  # the Core's own MCP server
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ],
)
@pytest.mark.asyncio
async def test_fetch_refuses_a_non_public_url_before_it_leaves_the_core(
    provider, url: str
) -> None:
    stub, client = provider

    with pytest.raises(SearchError):
        await client.fetch(url)
    await client.close()

    assert stub.requests == []


@pytest.mark.asyncio
async def test_an_unreachable_service_is_a_search_error() -> None:
    client = RemoteSearchProvider(
        base_url="http://127.0.0.1:1", token=TOKEN, request_timeout=0.2
    )

    with pytest.raises(SearchError, match="unreachable|failed"):
        await client.warmup()
    await client.close()
