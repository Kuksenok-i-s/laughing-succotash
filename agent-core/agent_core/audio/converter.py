"""ffmpeg/ffprobe helpers.

Duration is probed before an expensive GPU job so a file over the limit is refused immediately.
Slices sent to Whisper are 16 kHz mono with a high-pass and loudnorm: quiet Telegram notes
starve Silero VAD, and rumble under 80 Hz is not speech.
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


WHISPER_RATE = 16_000
HIGHPASS_HZ = 80
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
WHISPER_FILTER = f"highpass=f={HIGHPASS_HZ},{LOUDNORM}"


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


async def extract_slice(source: Path, target: Path, start: float, length: float) -> Path:
    """Cut ``length`` seconds from ``start`` as 16 kHz mono WAV for Whisper.

    High-pass and loudnorm run here so DualSTT slices are already at speech
    level before they hit the GPU service.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioProbeError("ffmpeg is not installed")
    target.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{length:.3f}",
        "-i",
        str(source),
        "-af",
        WHISPER_FILTER,
        "-ac",
        "1",
        "-ar",
        str(WHISPER_RATE),
        "-c:a",
        "pcm_s16le",
        str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        raise AudioProbeError(f"ffmpeg failed: {stderr.decode()[-300:]}")
    return target


async def to_wav16k(source: Path, target: Path) -> Path:
    """Transcode to 16 kHz mono WAV — whisper's native input format.

    Only used as a fallback when the container cannot be decoded directly, since transcoding a
    long recording costs real time and disk.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioProbeError("ffmpeg is not installed")

    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-y", "-i", str(source),
        "-af", WHISPER_FILTER,
        "-ac", "1", "-ar", str(WHISPER_RATE), "-c:a", "pcm_s16le", str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise AudioProbeError(f"ffmpeg failed: {stderr.decode()[-300:]}")
    return target
