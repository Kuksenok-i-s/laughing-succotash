"""Agent Core entrypoint.

Composition happens here and only here: every component is constructed with its dependencies
passed in, so each one can be tested against fakes without a Mac mini, a Telegram token or a
Cursor licence.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from .assistant.confirmations import ConfirmationService
from .assistant.service import AssistantService
from .assistant.sessions import SessionManager
from .assistant.transcript import TranscriptAnalyzer
from .audio.storage import UploadManager
from .calendar.local import LocalCalendarProvider
from .config import Settings, get_settings
from .jobs.manager import JobManager
from .logging_setup import configure_logging
from .mcp.server import ContextRegistry, McpServer, ToolRegistry
from .mcp.tools import register_tools
from .rpc.connection import GatewayLink
from .rpc.handlers import CoreHandlers
from .scheduler.service import Scheduler
from .storage.database import Database
from .storage.repositories import Repositories

log = logging.getLogger(__name__)

CAPABILITIES = [
    "assistant",
    "audio",
    "reminders",
    "calendar",
    "tasks",
    "notes",
    "memory",
    "contacts",
    "timers",
]


class Core:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = Database(settings.resolved_database_path)
        self._shutdown = asyncio.Event()

        self._repos: Repositories | None = None
        self._link: GatewayLink | None = None
        self._mcp: McpServer | None = None
        self._backend = None
        self._stt = None
        self._scheduler: Scheduler | None = None
        self._jobs = JobManager()
        self._uploads: UploadManager | None = None
        self._confirmations: ConfirmationService | None = None

    # ---- startup ---------------------------------------------------------

    async def start(self) -> None:
        await self._db.connect()
        repos = Repositories.build(self._db, self._settings.default_timezone)
        self._repos = repos

        # Nothing is executing jobs that were mid-flight when the process died; leaving them
        # 'running' would make /status lie for ever.
        orphans = await repos.jobs.recover_orphans()
        if orphans:
            log.warning("failed %d job(s) interrupted by the previous shutdown", orphans)

        link = GatewayLink(self._settings, repos.events, capabilities=CAPABILITIES)
        self._link = link

        self._confirmations = ConfirmationService(
            repos.pending_actions,
            link,
            timeout_seconds=self._settings.confirmation_timeout_seconds,
        )

        contexts = ContextRegistry()
        registry = ToolRegistry()
        self._mcp = McpServer(
            registry,
            contexts,
            repos.operations,
            self._confirmations,
            host=self._settings.mcp_host,
            port=self._settings.mcp_port,
            token=self._settings.mcp_token,
        )

        self._uploads = UploadManager(
            repos.uploads,
            self._settings.resolved_temp_dir,
            max_bytes=self._settings.max_audio_bytes,
            idle_timeout=self._settings.upload_idle_timeout,
        )

        self._backend = self._build_backend()
        self._stt = self._build_stt()

        sessions = SessionManager(
            repos.conversations,
            self._backend,
            contexts,
            self._mcp,
            default_workspace=self._settings.resolved_assistant_workspace,
        )

        assistant = AssistantService(
            self._settings,
            repos,
            link,
            sessions,
            self._jobs,
            self._backend,
            stt=self._stt,
            uploads=self._uploads,
            analyzer=TranscriptAnalyzer(
                self._backend,
                workspace=self._settings.resolved_assistant_workspace,
                chunk_chars=self._settings.transcript_chunk_chars,
            ),
        )

        self._scheduler = Scheduler(
            repos,
            assistant,
            tick_seconds=self._settings.scheduler_tick_seconds,
            confirmations=self._confirmations,
            uploads=self._uploads,
            default_timezone=self._settings.default_timezone,
        )

        register_tools(
            registry,
            repos,
            calendar_provider=LocalCalendarProvider(repos.calendar),
            scheduler=self._scheduler,
            # No search provider is configured, so web_search and web_fetch are not registered and
            # the assistant has no network reach through MCP at all. See agent_core/search/base.py
            # for the contract an implementation has to satisfy.
            search_provider=None,
        )
        log.info("mcp tools registered: %s", ", ".join(registry.names()))

        handlers = CoreHandlers(
            self._settings, repos, assistant, self._uploads, self._confirmations, self._scheduler
        )
        link.register_all(handlers.as_map())
        link.on_binary = handlers.on_binary

        await self._mcp.start()
        await self._start_backend()
        if self._stt is not None:
            try:
                await self._stt.warmup()
            except Exception as exc:
                log.error("speech-to-text warmup failed: %s", exc)
        await self._scheduler.start()
        await link.start()

        log.info("agent core %s started", self._settings.instance_id)

    def _build_backend(self):
        from .agent.cursor_acp import CursorACPBackend

        if self._settings.agent_backend != "acp":
            raise SystemExit(
                f"unsupported AGENT_BACKEND {self._settings.agent_backend!r}; "
                "only 'acp' is implemented"
            )
        return CursorACPBackend(
            self._settings.cursor_agent_binary,
            # The conversation session is rooted in a sandbox directory: Cursor's built-in write
            # and shell tools do not go through the permission callback (docs/cursor-acp.md), so
            # the session must not start anywhere that matters.
            default_workspace=self._settings.resolved_assistant_workspace,
            model=self._settings.cursor_model,
            startup_timeout=self._settings.agent_startup_timeout,
            prompt_timeout=self._settings.agent_prompt_timeout,
        )

    def _local_stt(self):
        from .stt.faster_whisper import FasterWhisperSTT

        return FasterWhisperSTT(
            model=self._settings.stt_model,
            device=self._settings.stt_device,
            compute_type=self._settings.stt_compute_type,
            language=self._settings.stt_language,
            beam_size=self._settings.stt_beam_size,
            vad_filter=self._settings.stt_vad_filter,
            max_concurrent=self._settings.stt_max_concurrent,
            download_root=self._settings.resolved_data_dir / "models",
        )

    def _build_stt(self):
        if self._settings.stt_backend != "gpu":
            return self._local_stt()

        from .stt.remote_gpu import RemoteGpuWhisperSTT

        gpu = RemoteGpuWhisperSTT(
            config_path=self._settings.resolved_stt_gpu_config,
            data_dir=self._settings.resolved_data_dir,
            language=self._settings.stt_language,
            beam_size=self._settings.stt_beam_size,
            max_concurrent=self._settings.stt_max_concurrent,
        )
        if not self._settings.stt_cpu_fallback:
            return gpu

        from .stt.fallback import FallbackSTT

        return FallbackSTT(primary=gpu, fallback=self._local_stt())

    async def _start_backend(self) -> None:
        """Start Cursor eagerly, but do not die if it is not there.

        A Core that refuses to boot because ``cursor-agent`` is missing would also stop reminders
        firing and stop the user being told what is wrong.
        """
        try:
            await self._backend.start()
        except Exception as exc:
            log.error("cursor agent unavailable at startup: %s", exc)

    # ---- run / shutdown ----------------------------------------------------

    async def run(self) -> None:
        await self.start()
        await self._shutdown.wait()
        await self.stop()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def stop(self) -> None:
        """Ordered teardown: stop taking work, finish what is running, then close resources."""
        log.info("shutting down agent core")

        if self._scheduler is not None:
            await self._scheduler.stop()

        # Give running jobs a chance to deliver their replies while the link is still up.
        await self._jobs.drain(timeout=15.0)

        if self._confirmations is not None:
            self._confirmations.abandon_all()

        if self._backend is not None:
            with contextlib.suppress(Exception):
                await self._backend.close()

        if self._stt is not None:
            with contextlib.suppress(Exception):
                await self._stt.close()

        if self._uploads is not None:
            await self._uploads.shutdown()

        if self._mcp is not None:
            await self._mcp.stop()

        if self._link is not None:
            await self._link.stop()

        await self._db.close()
        log.info("agent core stopped")


async def amain() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    problems = settings.validate_runtime()
    if problems:
        for problem in problems:
            log.error("configuration error: %s", problem)
        return 2

    core = Core(settings)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, core.request_shutdown)

    try:
        await core.run()
    except Exception:
        log.exception("agent core crashed")
        await core.stop()
        return 1
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
