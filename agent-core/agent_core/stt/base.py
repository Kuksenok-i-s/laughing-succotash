"""Transcription contract.

faster-whisper objects never cross this boundary: its ``Segment`` is a lazily-evaluated generator
item tied to an open model handle, and letting that leak would make the rest of the Core depend on
when transcription actually runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Both hooks are invoked on the event loop thread, whatever thread the backend does its work on:
# callers schedule coroutines from them, which is illegal anywhere else. A backend that transcribes
# in a worker thread must marshal its callbacks with ``loop.call_soon_threadsafe``.
ProgressHook = Callable[[float], None]
NoticeHook = Callable[[str], None]

# A backend reports a degraded run through the notice hook so the caller can tell the user why the
# wait got longer. Stable identifier: the wording belongs to the Gateway, not here.
STT_CPU_FALLBACK = "stt_cpu_fallback"


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def timestamped(self) -> str:
        return f"[{_clock(self.start)}] {self.text}"


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def with_timestamps(self) -> str:
        """Segment text with clock prefixes, for long recordings where position matters."""
        return "\n".join(segment.timestamped() for segment in self.segments)


class SttError(RuntimeError):
    """Transcription failed. The job fails; the Core stays up and the temp file is removed."""


class SpeechToText(Protocol):
    async def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: ProgressHook | None = None,
        on_notice: NoticeHook | None = None,
    ) -> TranscriptionResult: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...

    @property
    def ready(self) -> bool: ...


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"
