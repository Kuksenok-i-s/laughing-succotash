"""Split a long recording into overlapping windows and stitch transcripts."""

from __future__ import annotations

from .base import TranscriptionResult, TranscriptSegment

DEFAULT_CHUNK_SECONDS = 600.0
DEFAULT_OVERLAP_SECONDS = 2.0


def chunk_windows(
    duration: float,
    chunk_seconds: float,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[tuple[float, float]]:
    """Cover ``duration`` with overlapping ``(start, length)`` windows."""
    if duration <= 0:
        return [(0.0, 0.0)]
    if chunk_seconds <= 0 or duration <= chunk_seconds:
        return [(0.0, duration)]
    overlap = min(max(0.0, overlap_seconds), chunk_seconds / 4.0)
    step = chunk_seconds - overlap
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        length = min(chunk_seconds, duration - start)
        windows.append((start, length))
        if start + length >= duration:
            break
        start += step
    return windows


def merge_results(
    parts: list[tuple[float, TranscriptionResult]],
    overlap_seconds: float,
) -> TranscriptionResult:
    """Shift each slice onto the source timeline and drop the overlapped head of later slices."""
    segments: list[TranscriptSegment] = []
    language = None
    duration = 0.0
    for index, (offset, result) in enumerate(parts):
        if language is None:
            language = result.language
        chunk_duration = result.duration or 0.0
        duration = max(duration, offset + chunk_duration)
        skip_before = offset + (overlap_seconds if index else 0.0)
        for segment in result.segments:
            start = segment.start + offset
            end = segment.end + offset
            if index and start < skip_before:
                continue
            text = segment.text.strip()
            if not text:
                continue
            segments.append(TranscriptSegment(start, end, text))
    return TranscriptionResult(
        text=" ".join(item.text for item in segments).strip(),
        language=language,
        duration=duration,
        segments=segments,
    )
