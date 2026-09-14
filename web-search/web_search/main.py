"""Service entrypoint: HTTP surface and one cache sweeper."""

from __future__ import annotations

import logging
import signal
import sys
import threading

from .backends import build_backend
from .config import Settings, from_env
from .logging_setup import configure_logging
from .server import SearchApp, SearchServer
from .service import SearchService

log = logging.getLogger(__name__)


class Service:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._search = SearchService(settings, build_backend(settings))
        self._app = SearchApp(settings, self._search)
        self._server: SearchServer | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        self._server = SearchServer((self._settings.host, self._settings.port), self._app)
        self._spawn("http", self._server.serve_forever)
        self._spawn("sweeper", self._sweep_forever)
        log.info(
            "listening on %s:%d (backend=%s parallel=%d cache_ttl=%.0fs fetch=%s)",
            self._settings.host,
            self._server.server_address[1],
            self._settings.backend,
            self._settings.parallel,
            self._settings.cache_ttl_seconds,
            "on" if self._settings.fetch_enabled else "off",
        )

    def _spawn(self, name: str, target) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _sweep_forever(self) -> None:
        while not self._stop.wait(self._settings.sweep_interval_seconds):
            try:
                self._search.cache.sweep()
            except Exception:
                log.exception("cache sweep failed")

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
        self._search.close()
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
