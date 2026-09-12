"""Split a long recording into overlapping windows and stitch the transcripts.

A two-hour file in one faster-whisper pass grows CUDA working set until Xavier OOMs.
Ten-minute slices stay inside the service memory cap; a two-second overlap absorbs words
cut at the join. Windows are (start, length) in seconds, both relative to the source.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .engine import ProgressHook
from .preprocess import prepare_whisper_audio

log = logging.getLogger(__name__)

DEFAULT_CHUNK_SECONDS = 600.0
DEFAULT_OVERLAP_SECONDS = 2.0
# ffmpeg loudnorm is single-threaded and slower than the GPU on a 300 s window; two cutters keep
# the next window ready without starving CTranslate2 of CPU on an 8-core Jetson.
DEFAULT_PREFETCH = 2

Probe = Callable[[Path], float | None]
Extract = Callable[[Path, Path, float, float], None]
Prepare = Callable[[Path, Path], Path]


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


def merge_chunk_results(
    parts: list[tuple[float, dict[str, Any]]],
    overlap_seconds: float,
) -> dict[str, Any]:
    """Shift each slice onto the source timeline and drop the overlapped head of later slices."""
    segments: list[dict[str, Any]] = []
    language = None
    duration = 0.0
    for index, (offset, result) in enumerate(parts):
        if language is None:
            language = result.get("language")
        chunk_duration = float(result.get("duration") or 0.0)
        duration = max(duration, offset + chunk_duration)
        skip_before = offset + (overlap_seconds if index else 0.0)
        for item in result.get("segments") or []:
            start = float(item["start"]) + offset
            end = float(item["end"]) + offset
            if index and start < skip_before:
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            segments.append({"start": start, "end": end, "text": text})
    return {
        "text": " ".join(item["text"] for item in segments).strip(),
        "language": language,
        "duration": duration,
        "segments": segments,
    }


def probe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        value = json.loads(completed.stdout)["format"]["duration"]
        return float(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def extract_chunk(source: Path, dest: Path, start: float, length: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed; cannot split a long recording")
    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(120.0, length + 60.0)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        err = completed.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(f"ffmpeg failed to extract {length:.0f}s at {start:.0f}s: {err}")


def transcribe_audio(
    engine,
    audio_path: Path,
    *,
    language: str | None,
    beam_size: int | None,
    on_progress: ProgressHook | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    probe: Probe = probe_duration,
    extract: Extract = extract_chunk,
    prepare: Prepare = prepare_whisper_audio,
    prepared: bool = False,
    prefetch: int = DEFAULT_PREFETCH,
) -> dict[str, Any]:
    """Transcribe ``audio_path``, splitting when it is longer than ``chunk_seconds``.

    ``prepared`` says the uploader already ran the filter chain (a DualHost slice from the Core),
    so it is not run again here. Long files are cut window by window in a small thread pool while
    the GPU decodes the previous window, so ffmpeg time hides behind decode time instead of
    adding to it.
    """
    duration = probe(audio_path)
    windows = chunk_windows(duration or 0.0, chunk_seconds, overlap_seconds)

    if duration is None or len(windows) <= 1:
        prepared_dir = audio_path.parent / "prepared"
        source = audio_path if prepared else prepare(audio_path, prepared_dir / "input.wav")
        try:
            return engine.transcribe(
                source, language=language, beam_size=beam_size, on_progress=on_progress
            )
        finally:
            if source != audio_path:
                shutil.rmtree(prepared_dir, ignore_errors=True)

    work = audio_path.parent / "chunks"
    work.mkdir(parents=True, exist_ok=True)
    log.info(
        "splitting %s into %d × %.0fs chunks (%.0fs audio, prefetch %d%s)",
        audio_path.name,
        len(windows),
        chunk_seconds,
        duration,
        prefetch,
        ", already prepared" if prepared else "",
    )

    def cut(index: int, start: float, length: float) -> Path:
        raw = work / f"{index:03d}.raw.wav"
        extract(audio_path, raw, start, length)
        if prepared:
            return raw
        clean = prepare(raw, work / f"{index:03d}.wav")
        if clean != raw:
            raw.unlink(missing_ok=True)
        return clean

    parts: list[tuple[float, dict[str, Any]]] = []
    detected = language
    done_segments = 0
    pool = ThreadPoolExecutor(max_workers=max(1, prefetch), thread_name_prefix="chunk-prep")
    try:
        futures = [pool.submit(cut, index, start, length) for index, (start, length) in enumerate(windows)]
        for index, (start, length) in enumerate(windows):
            chunk_path = futures[index].result()

            def progress(
                percent: float,
                position: float,
                _chunk_duration: float | None,
                segments: int,
                *,
                _start: float = start,
            ) -> None:
                if on_progress is None:
                    return
                global_pos = _start + position
                global_pct = min(100.0, global_pos / duration * 100.0) if duration else percent
                on_progress(global_pct, global_pos, duration, done_segments + segments)

            result = engine.transcribe(
                chunk_path,
                language=detected,
                beam_size=beam_size,
                on_progress=progress,
            )
            chunk_path.unlink(missing_ok=True)
            if detected is None:
                detected = result.get("language")
            done_segments += len(result.get("segments") or [])
            parts.append((start, result))
        merged = merge_chunk_results(parts, overlap_seconds)
        merged["duration"] = duration
        return merged
    finally:
        # Windows not yet cut are dropped; one mid-cut finishes so its file is removed with the rest.
        pool.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(work, ignore_errors=True)
