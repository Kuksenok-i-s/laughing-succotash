"""Fixtures: a real HTTP server on an ephemeral port, in front of a fake backend."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

import pytest
from helpers import TOKEN, Client, FakeBackend

from web_search.config import Settings
from web_search.server import SearchApp, SearchServer
from web_search.service import SearchService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        backend="brave",
        brave_api_key="k" * 32,
        default_limit=3,
        max_limit=10,
        max_queries=5,
        cache_ttl_seconds=60.0,
        parallel=4,
    )


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def service(settings: Settings, backend: FakeBackend) -> Iterator[SearchService]:
    created = SearchService(settings, backend)
    try:
        yield created
    finally:
        created.close()


@pytest.fixture
def app(settings: Settings, service: SearchService) -> SearchApp:
    return SearchApp(settings, service)


@pytest.fixture
def serve() -> Iterator[Callable[[SearchApp], Client]]:
    running: list[tuple[SearchServer, threading.Thread]] = []

    def start(app: SearchApp) -> Client:
        server = SearchServer(("127.0.0.1", 0), app)
        # serve_forever polls at 0.5s by default, and shutdown() waits for the current tick.
        # Across a suite of this size that is most of the wall clock.
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
        )
        thread.start()
        running.append((server, thread))
        return Client("127.0.0.1", server.server_address[1])

    yield start

    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


@pytest.fixture
def client(app: SearchApp, serve: Callable[[SearchApp], Client]) -> Client:
    return serve(app)
