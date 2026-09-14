"""Fixtures: a real HTTP server on an ephemeral port, in front of a fake engine."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from helpers import TOKEN, Client, FakeEngine

from gpu_transcriber.config import Settings
from gpu_transcriber.jobs import JobStore
from gpu_transcriber.server import TranscriptionApp, TranscriptionServer


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        work_dir=tmp_path / "work",
        max_upload_mb=1,
        job_ttl_seconds=3600.0,
    )


@pytest.fixture
def store(settings: Settings) -> JobStore:
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    return JobStore(settings.work_dir, ttl_seconds=settings.job_ttl_seconds)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def app(settings: Settings, store: JobStore, engine: FakeEngine) -> TranscriptionApp:
    return TranscriptionApp(settings, store, engine)


@pytest.fixture
def serve() -> Iterator[Callable[[TranscriptionApp], Client]]:
    """Start a server for an app and hand back a client pointed at it."""
    running: list[tuple[TranscriptionServer, threading.Thread]] = []

    def start(app: TranscriptionApp) -> Client:
        server = TranscriptionServer(("127.0.0.1", 0), app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return Client("127.0.0.1", server.server_address[1])

    yield start

    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


@pytest.fixture
def client(app: TranscriptionApp, serve: Callable[[TranscriptionApp], Client]) -> Client:
    return serve(app)
