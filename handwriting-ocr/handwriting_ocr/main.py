"""Service entrypoint: HTTP surface, one OCR worker, one sweeper."""

from __future__ import annotations

import logging
import signal
import sys
import threading

from .config import Settings, from_env
from .engine import Engine, LlamaCppEngine, OllamaEngine
from .jobs import JobStore
from .logging_setup import configure_logging
from .server import OcrApp, OcrServer
from .worker import OcrWorker


def build_engine(settings: Settings) -> Engine:
    if settings.backend == "llamacpp":
        return LlamaCppEngine(
            llama_url=settings.llama_url,
            model=settings.model,
            request_timeout=settings.request_timeout,
            max_tokens=settings.max_tokens,
            image_max_edge=settings.image_max_edge,
            max_passes=settings.max_passes,
            pipeline=settings.pipeline,
        )
    return OllamaEngine(
        ollama_url=settings.ollama_url,
        model=settings.model,
        keep_alive=settings.keep_alive,
        request_timeout=settings.request_timeout,
        image_max_edge=settings.image_max_edge,
        max_passes=settings.max_passes,
        pipeline=settings.pipeline,
    )

log = logging.getLogger(__name__)


class Service:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = JobStore(settings.work_dir, ttl_seconds=settings.job_ttl_seconds)
        self._engine = build_engine(settings)
        self._app = OcrApp(settings, self._store, self._engine)
        self._worker = OcrWorker(
            self._store,
            self._engine,
            idle_unload_seconds=settings.idle_unload_seconds,
        )
        self._server: OcrServer | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        self._settings.work_dir.mkdir(parents=True, exist_ok=True)
        self._server = OcrServer((self._settings.host, self._settings.port), self._app)
        self._spawn("http", self._server.serve_forever)
        self._spawn("ocr", lambda: self._worker.run(self._stop))
        self._spawn("sweeper", self._sweep_forever)
        log.info(
            "listening on %s:%d (work dir %s, backend=%s model=%s)",
            self._settings.host,
            self._server.server_address[1],
            self._settings.work_dir,
            self._settings.backend,
            self._settings.model,
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
