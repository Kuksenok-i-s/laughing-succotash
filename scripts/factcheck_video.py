"""Run the real YouTube factcheck pass on one video, outside Telegram.

Jobs normally arrive from the gateway and there is no other way in, so checking whether the
factcheck actually searches meant either sending a Telegram message by hand or this. It reuses
the Core's own builders (``_build_backend``, ``_build_stt``, ``_build_search``), the real
downloader, the real transcript analyzer, the real MCP server with its scoped-token restriction,
and the real prompt. What it does not reuse is the live database: DATA_DIR points at a scratch
directory so a run cannot touch the assistant's own state, and the search tools never read it.

Run it on the Core host, where the services and the agent binary live:

    DATA_DIR=~/factcheck-run/data ~/agent-core/.venv/bin/python scripts/factcheck_video.py URL

The audio and the transcript are cached under ``~/factcheck-run``, so a second run costs the
agent calls and nothing else. Delete ``transcript.json`` to force a fresh transcription.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from agent_core.agent.base import AgentContext, Provenance
from agent_core.assistant import prompts
from agent_core.assistant.transcript import TranscriptAnalyzer
from agent_core.calendar.local import LocalCalendarProvider
from agent_core.config import Settings
from agent_core.main import Core
from agent_core.mcp.permissions import ToolContext
from agent_core.mcp.server import ContextRegistry, McpServer, ToolRegistry
from agent_core.mcp.tools import register_tools
from agent_core.storage.database import Database
from agent_core.stt.base import TranscriptionResult, TranscriptSegment
from agent_core.storage.repositories import Repositories
from agent_core.youtube import YoutubeDownloader
from agent_core.youtube.download import YoutubeMedia

URL = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/YPO254GApK8"
OUT = Path.home() / "factcheck-run"
# Read by pydantic directly rather than sourced by the shell: the file holds unquoted values with
# commas and Cyrillic spaces, which a shell splits into commands.
ENV_FILE = Path.home() / "agent-core/.env"
TITLE = (
    "Медицинский психолог о жестоких мужчинах и скрытых сигналах агрессора. "
    "Выводы за 55 лет практики"
)


class NoConfirmations:
    """Nothing here may ask the user anything: there is no user attached to this run."""

    async def request(self, **kwargs: object) -> bool:
        print(f"  !! a tool asked for confirmation and was refused: {kwargs.get('tool_name')}")
        return False


def step(label: str) -> float:
    print(f"\n=== {label}", flush=True)
    return time.monotonic()


def done(started: float) -> None:
    print(f"    ({time.monotonic() - started:.0f}s)", flush=True)


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings(_env_file=str(ENV_FILE))
    user_id = sorted(settings.allowed_users)[0]
    print(f"core instance {settings.instance_id}, acting as {user_id}")
    print(f"search enabled={settings.search_enabled} url={settings.search_service_url}")

    core = Core(settings)
    backend = core._build_backend()  # noqa: SLF001
    stt = core._build_stt()  # noqa: SLF001
    search = core._build_search()  # noqa: SLF001
    if search is None:
        print("SEARCH_ENABLED is off; nothing to test")
        return 2
    await search.warmup()
    print(f"search backend: {search.backend}, ready={search.ready}")

    db = Database(settings.resolved_database_path)
    await db.connect()
    repos = Repositories.build(db, settings.default_timezone)

    registry = ToolRegistry()
    contexts = ContextRegistry()
    register_tools(
        registry,
        repos,
        calendar_provider=LocalCalendarProvider(repos.calendar),
        search_provider=search,
    )
    mcp = McpServer(
        registry, contexts, repos.operations, NoConfirmations(),
        host="127.0.0.1", port=0, token=settings.mcp_token,
    )
    await mcp.start()

    # ---- download ---------------------------------------------------------
    started = step(f"downloading audio for {URL}")
    downloader = YoutubeDownloader.from_settings(settings)
    if downloader is None:
        print("no YouTube downloader configured")
        return 2
    cached = [
        path
        for path in sorted((OUT / "audio").rglob("*"))
        if path.suffix.lower() in (".mp3", ".opus", ".m4a", ".webm", ".wav")
    ]
    if cached:
        # A rerun should not spend another minute on the proxy for a file already here.
        print(f"    reusing {cached[0].name}")
        media = YoutubeMedia(
            url=URL, video_id="YPO254GApK8", title=TITLE,
            duration=None, audio_path=cached[0],
        )
    else:
        batch = await downloader.fetch_audio(URL, OUT / "audio", job_id="factcheck-driver")
        media = batch.items[0]
    print(f"    {media.title} ({media.duration or 0:.0f}s) -> {media.audio_path.name}")
    done(started)

    # ---- transcribe -------------------------------------------------------
    started = step("transcribing")
    last = [0.0]

    def on_progress(fraction: float) -> None:
        """Called synchronously by the STT client, so this must not be a coroutine."""
        if fraction - last[0] >= 0.1:
            last[0] = fraction
            print(f"    {fraction * 100:.0f}%", flush=True)

    # Segments and not just text: the analyzer chunks by segment, and a cache that drops them
    # silently reduces the factcheck to the first 8000 characters of the excerpt fallback.
    cached_transcript = OUT / "transcript.json"
    if cached_transcript.exists() and cached_transcript.stat().st_size > 1000:
        # An hour of audio costs minutes of GPU; a rerun of the pass being tested should not.
        print(f"    reusing {cached_transcript.name}")
        raw = json.loads(cached_transcript.read_text(encoding="utf-8"))
        result = TranscriptionResult(
            text=raw["text"],
            language=raw.get("language"),
            duration=raw.get("duration"),
            segments=[TranscriptSegment(**s) for s in raw.get("segments", [])],
        )
    else:
        warmup = getattr(stt, "warmup", None)
        if warmup is not None:
            await warmup()
        result = await stt.transcribe(media.audio_path, on_progress=on_progress)
        cached_transcript.write_text(
            json.dumps(
                {
                    "text": result.text,
                    "language": result.language,
                    "duration": result.duration,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in result.segments
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if not result.segments:
        print("    !! no segments: the analyzer will produce no notes")
    print(f"    {len(result.text)} chars, {len(result.segments)} segments")
    done(started)

    # A tzinfo, not the setting's string: the prompt builders call astimezone on it.
    tz = ZoneInfo(settings.default_timezone)
    context = AgentContext(
        user_id=user_id,
        conversation_id="factcheck-driver",
        job_id="factcheck-driver",
        timezone=tz,
        now=datetime.now(timezone.utc),
        provenance=Provenance.UNTRUSTED_CONTENT,
        owner_name="driver",
    )

    # ---- notes ------------------------------------------------------------
    started = step("analysing the transcript into notes")
    analyzer = TranscriptAnalyzer(
        backend,
        workspace_for=settings.user_workspace,
        chunk_chars=settings.transcript_chunk_chars,
        parallel=settings.transcript_parallel,
    )
    cached_notes = OUT / "notes.txt"
    if cached_notes.exists() and cached_notes.stat().st_size > 100:
        print(f"    reusing {cached_notes.name}")
        notes = cached_notes.read_text(encoding="utf-8")
        excerpt = result.with_timestamps() or result.text
    else:
        analysis = await analyzer.analyze(result, context)
        notes = analysis.notes
        excerpt = analysis.excerpt or result.with_timestamps() or result.text
        cached_notes.write_text(notes, encoding="utf-8")
    print(f"    notes: {len(notes)} chars")
    done(started)

    # ---- factcheck --------------------------------------------------------
    source = dict(
        title=media.title,
        notes=notes,
        context=context,
        excerpt=None if notes else excerpt[:8000],
        duration_seconds=result.duration or media.duration,
    )

    started = step("factcheck pass (this is the one that searches)")
    tool_context = ToolContext(
        user_id=user_id,
        conversation_id="factcheck-driver",
        provenance=Provenance.UNTRUSTED_CONTENT,
        job_id="factcheck-driver",
        timezone=tz,
        now=datetime.now(timezone.utc),
    )
    token = contexts.issue_scoped_token(
        tool_context, tools=frozenset({"web_search", "web_fetch"})
    )
    session_id = await backend.create_session(
        workspace=settings.user_workspace(user_id),
        mcp_servers=[mcp.session_entry(token)],
    )
    cached_factcheck = OUT / "factcheck.md"
    if cached_factcheck.exists() and cached_factcheck.stat().st_size > 100:
        print(f"    reusing {cached_factcheck.name}")
        factcheck = cached_factcheck.read_text(encoding="utf-8")
    else:
        response = await backend.send_message(
            session_id,
            prompts.youtube_factcheck(**source),
            context,
            on_progress=_log_stage,
        )
        factcheck = (response.text or "").strip()
        cached_factcheck.write_text(factcheck, encoding="utf-8")
    # Released before the second pass, which must not search. A token that outlives its pass is a
    # way back in.
    contexts.release(token)
    done(started)

    # ---- summary ----------------------------------------------------------
    started = step("summary pass (no search; this is the file the user would receive)")
    second = await backend.send_message(
        session_id,
        prompts.youtube_summary(**source, factcheck=factcheck or None),
        context,
        on_progress=_log_stage,
    )
    summary = (second.text or "").strip()
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    done(started)

    print("\n" + "=" * 70)
    print(summary)
    print("=" * 70)
    print(f"\nwritten to {OUT}")

    await search.close()
    await mcp.stop()
    await db.close()
    return 0


async def _log_stage(stage: str, detail: str | None) -> None:
    print(f"    [{stage}] {detail or ''}", flush=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
