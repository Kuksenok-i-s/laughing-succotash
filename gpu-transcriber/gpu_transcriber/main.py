"""Service entrypoint: HTTP surface, one GPU worker, one sweeper.

The HTTP server starts before the model is loaded, so ``/health`` can answer honestly while the
weights are still coming off disk. That is the difference between a client that knows to wait and a
client that sees a refused connection and gives up on the GPU entirely.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading

from .config import Settings, from_env
from .engine import WhisperEngine
from .jobs import JobStore
from .logging_setup import configure_logging
from .server import TranscriptionApp, TranscriptionServer
from .worker import TranscriptionWorker

log = logging.getLogger(__name__)


class Service:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = JobStore(settings.work_dir, ttl_seconds=settings.job_ttl_seconds)
        self._engine = WhisperEngine(
            model=settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
        )
        self._app = TranscriptionApp(settings, self._store, self._engine)
        self._worker = TranscriptionWorker(self._store, self._engine)
        self._server: TranscriptionServer | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        self._settings.work_dir.mkdir(parents=True, exist_ok=True)
        # Audio left behind by a previous run is removed on the first sweep, not here: a restart
        # while the Core is still waiting should not throw away a file it is about to ask about.
        self._server = TranscriptionServer(
            (self._settings.host, self._settings.port), self._app
        )
        self._spawn("http", self._server.serve_forever)
        self._spawn("gpu", lambda: self._worker.run(self._stop))
        self._spawn("sweeper", self._sweep_forever)
        log.info(
            "listening on %s:%d (work dir %s)",
            self._settings.host,
            self._server.server_address[1],
            self._settings.work_dir,
        )

    def _spawn(self, name: str, target) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _sweep_forever(self) -> None:
        while not self._stop.wait(self._settings.sweep_interval_seconds):
            try:
                self._store.sweep()
            except Exception:
                log.exception("sweep failed")

    def request_shutdown(self, *_args: object) -> None:
        self._stop.set()

    def run(self) -> None:
        self.start()
        self._stop.wait()
        self.stop()

    def stop(self) -> None:
        log.info("shutting down")
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        for thread in self._threads:
            thread.join(timeout=5.0)
        log.info("stopped")


def main() -> int:
    settings = from_env()
    configure_logging(settings.log_level, settings.log_format)

    problems = settings.validate_runtime()
    if problems:
        for problem in problems:
            log.error("configuration error: %s", problem)
        return 2

    service = Service(settings)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, service.request_shutdown)

    try:
        service.run()
    except Exception:
        log.exception("service crashed")
        service.stop()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
