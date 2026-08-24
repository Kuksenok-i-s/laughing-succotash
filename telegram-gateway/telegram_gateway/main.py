"""Telegram Gateway entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from aiogram import Bot, Dispatcher
from aiohttp import web

from .config import Settings, get_settings
from .delivery.service import SubmissionService
from .logging_setup import configure_logging
from .rpc.server import CoreLink
from .storage.database import Database
from .storage.models import GatewayStore
from .telegram.handlers import build_router
from .telegram.renderer import TelegramRenderer

log = logging.getLogger(__name__)


class Gateway:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = Database(settings.resolved_database_path)
        self._store = GatewayStore(self._db)
        self._bot = Bot(token=settings.telegram_bot_token)
        self._dispatcher = Dispatcher()
        self._core = CoreLink(settings, self._store)
        self._submissions = SubmissionService(
            self._core, self._store, settings, bot=self._bot
        )
        self._runner: web.AppRunner | None = None
        self._polling: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        await self._db.connect()

        renderer = TelegramRenderer(self._bot, self._store, self._settings)
        self._core.register_all(renderer.handlers())
        # Anything queued while the Core was away goes out as soon as it says hello.
        self._core.on_ready = self._on_core_ready

        self._dispatcher.include_router(
            build_router(self._bot, self._store, self._core, self._submissions, self._settings)
        )

        app = web.Application()
        app.router.add_get(self._settings.rpc_path, self._core.handle)
        app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._settings.host, self._settings.port)
        await site.start()
        log.info(
            "rpc endpoint listening on %s:%s%s",
            self._settings.host, self._settings.port, self._settings.rpc_path,
        )

        await self._submissions.start()

        me = await self._bot.get_me()
        log.info("telegram bot @%s ready", me.username)
        self._polling = asyncio.ensure_future(
            self._dispatcher.start_polling(self._bot, handle_signals=False)
        )

    async def _on_core_ready(self) -> None:
        self._submissions.nudge()

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "ok", "core_connected": self._core.connected}
        )

    async def run(self) -> None:
        await self.start()
        await self._shutdown.wait()
        await self.stop()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def stop(self) -> None:
        """Graceful shutdown: stop accepting, flush, then close everything.

        Order matters. Polling stops first so no new Telegram update can arrive mid-teardown; the
        pending queue is already durable, so anything unsent is simply picked up next start.
        """
        log.info("shutting down gateway")

        if self._polling is not None:
            self._polling.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._polling
        with contextlib.suppress(Exception):
            await self._dispatcher.storage.close()

        await self._submissions.stop()

        if self._runner is not None:
            await self._runner.cleanup()

        with contextlib.suppress(Exception):
            await self._bot.session.close()

        await self._db.close()
        log.info("gateway stopped")


async def amain() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    problems = settings.validate_runtime()
    if problems:
        for problem in problems:
            log.error("configuration error: %s", problem)
        return 2

    gateway = Gateway(settings)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, gateway.request_shutdown)

    try:
        await gateway.run()
    except Exception:
        log.exception("gateway crashed")
        await gateway.stop()
        return 1
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
