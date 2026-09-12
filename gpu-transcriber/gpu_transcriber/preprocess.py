"""Make audio easier for Whisper and its VAD before a CUDA pass.

faster-whisper will resample internally, but a quiet voice note or a YouTube
download with rumble under 80 Hz makes Silero VAD drop speech and the encoder
see a near-silent spectrogram. One ffmpeg pass is cheap next to large-v3-turbo.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

WHISPER_RATE = 16_000
HIGHPASS_HZ = 85
# Speech, not cinema: -16 LUFS is the podcast/voice-note target. Single-pass
# loudnorm is an approximation; two-pass would cost another decode for little
# gain on already-compressed Telegram and YouTube audio.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
FILTER = f"highpass=f={HIGHPASS_HZ},{LOUDNORM}"


def prepare_whisper_audio(source: Path, dest: Path) -> Path:
    """Return 16 kHz mono WAV with rumble cut and loudness normalized.

    On any failure the original path is returned so a transcription still runs.
    """
    if shutil.which("ffmpeg") is None:
        log.info("ffmpeg missing; handing %s to whisper as-is", source.name)
        return source

    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout = min(7200.0, max(120.0, source.stat().st_size / 8_000 + 60.0))
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-af",
                FILTER,
                "-ac",
                "1",
                "-ar",
                str(WHISPER_RATE),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not prepare %s (%s); using the original", source.name, exc)
        dest.unlink(missing_ok=True)
        return source

    if completed.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        err = completed.stderr.decode("utf-8", errors="replace")[-300:]
        log.warning("ffmpeg prepare failed for %s: %s; using the original", source.name, err)
        dest.unlink(missing_ok=True)
        return source

    log.info("prepared %s for whisper (%s, %d Hz mono)", source.name, FILTER, WHISPER_RATE)
    return dest
