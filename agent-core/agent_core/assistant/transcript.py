"""Hierarchical analysis of long recordings.

An hour of speech does not fit comfortably in one prompt, and asking for a summary of summaries
throws away exactly what matters — who committed to what, by when. So the transcript is split on
segment boundaries, each chunk is mined for facts, and the facts are concatenated in order for a
final pass.

The chunk passes run in scratch sessions created with no MCP servers at all. That is not a
convention but a hard guarantee: while the agent is reading untrusted recorded speech, it has no
tools to misuse.

Chunks are independent, so ``parallel`` scratch sessions work through them side by side; each
session still takes one prompt at a time, which is all the ACP transport allows. The facts are
reassembled in transcript order regardless of which session finished first.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.base import AgentBackend, AgentContext, AgentError
from ..stt.base import TranscriptionResult, TranscriptSegment
from . import prompts

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Chunk:
    index: int
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Analysis:
    notes: str = ""
    excerpt: str = ""
    chunk_count: int = 0
    failures: list[int] = field(default_factory=list)


def split_transcript(result: TranscriptionResult, chunk_chars: int) -> list[Chunk]:
    """Split on segment boundaries so no sentence is cut in half.

    Whisper's segments are already the natural pause points, and respecting them keeps a decision
    and the sentence that qualifies it in the same chunk.
    """
    segments = result.segments or [TranscriptSegment(0.0, result.duration or 0.0, result.text)]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    size = 0
    start = segments[0].start
    end = segments[0].end

    for segment in segments:
        line = segment.timestamped()
        if buffer and size + len(line) > chunk_chars:
            chunks.append(Chunk(len(chunks) + 1, start, end, "\n".join(buffer)))
            buffer, size = [], 0
            start = segment.start
        buffer.append(line)
        size += len(line) + 1
        end = segment.end

    if buffer:
        chunks.append(Chunk(len(chunks) + 1, start, end, "\n".join(buffer)))
    return chunks


class TranscriptAnalyzer:
    def __init__(
        self,
        backend: AgentBackend,
        *,
        workspace_for: Callable[[str], Path],
        chunk_chars: int = 12000,
        excerpt_chars: int = 4000,
        parallel: int = 3,
    ) -> None:
        self._backend = backend
        self._workspace_for = workspace_for
        self._chunk_chars = chunk_chars
        self._excerpt_chars = excerpt_chars
        self._parallel = max(1, parallel)

    async def analyze(
        self,
        result: TranscriptionResult,
        context: AgentContext,
        *,
        on_progress=None,
    ) -> Analysis:
        chunks = split_transcript(result, self._chunk_chars)

        if len(chunks) <= 1:
            # Short enough to reason about whole: analysing it separately would only add a
            # lossy layer between the agent and the actual words.
            return Analysis(notes="", excerpt=result.with_timestamps(), chunk_count=1)

        workers = min(self._parallel, len(chunks))
        log.info("analysing transcript in %d chunks (%d sessions)", len(chunks), workers)

        analysis = Analysis(chunk_count=len(chunks))
        workspace = self._workspace_for(context.user_id)
        pending = deque(chunks)
        notes: dict[int, str] = {}
        done = 0

        async def run_session() -> None:
            nonlocal done
            try:
                session_id = await self._backend.create_session(workspace=workspace, mcp_servers=[])
            except AgentError as exc:
                # The other sessions pick up this one's share; leftovers become failures below.
                log.warning("could not open a transcript session: %s", exc)
                return
            while pending:
                chunk = pending.popleft()
                try:
                    response = await self._backend.send_message(
                        session_id,
                        prompts.chunk_analysis(chunk.text, chunk.index, len(chunks), context),
                        context,
                    )
                except AgentError as exc:
                    # One failed chunk should not lose the other fifty-nine minutes.
                    log.warning("chunk %d analysis failed: %s", chunk.index, exc)
                    analysis.failures.append(chunk.index)
                else:
                    if response.text:
                        notes[chunk.index] = (
                            f"--- Фрагмент {chunk.index}/{len(chunks)} "
                            f"({_clock(chunk.start)}–{_clock(chunk.end)}) ---\n{response.text}"
                        )
                done += 1
                if on_progress is not None:
                    await on_progress(done / len(chunks))

        await asyncio.gather(*(run_session() for _ in range(workers)))
        analysis.failures.extend(chunk.index for chunk in pending)
        analysis.failures.sort()

        analysis.notes = "\n\n".join(notes[index] for index in sorted(notes))
        if analysis.failures:
            analysis.notes += (
                "\n\n[Не удалось разобрать фрагменты: "
                + ", ".join(str(i) for i in analysis.failures)
                + ". Скажи об этом пользователю.]"
            )
        return analysis


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
