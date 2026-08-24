"""Transcription contract.

faster-whisper objects never cross this boundary: its ``Segment`` is a lazily-evaluated generator
item tied to an open model handle, and letting that leak would make the rest of the Core depend on
when transcription actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


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
    async def transcribe(self, audio_path: Path) -> TranscriptionResult: ...

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
