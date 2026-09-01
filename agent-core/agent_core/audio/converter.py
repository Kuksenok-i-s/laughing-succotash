"""ffmpeg/ffprobe helpers.

faster-whisper decodes most containers itself, so audio is normally handed to it untouched. What
we do need before starting an expensive transcription is the duration, so a file over the
configured limit can be rejected in a second rather than after an hour of CPU time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class AudioProbeError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


async def probe_duration(path: Path) -> float | None:
    """Duration in seconds, or ``None`` when ffprobe is unavailable or the file is unreadable."""
    if shutil.which("ffprobe") is None:
        return None

    process = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        log.info("ffprobe failed for %s: %s", path.name, stderr.decode()[:200])
        return None

    try:
        value = json.loads(stdout)["format"]["duration"]
        return float(value)
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


async def to_wav16k(source: Path, target: Path) -> Path:
    """Transcode to 16 kHz mono WAV — whisper's native input format.

    Only used as a fallback when the container cannot be decoded directly, since transcoding a
    long recording costs real time and disk.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioProbeError("ffmpeg is not installed")

    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-y", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise AudioProbeError(f"ffmpeg failed: {stderr.decode()[-300:]}")
    return target
