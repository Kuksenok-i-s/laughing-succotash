"""Remote GPU faster-whisper, via the shared ``transcription.gpu_remote`` pipeline.

The transcription package lives next to Core state (``DATA_DIR``), not in this repo: it is the
same SSH/SCP helper the YouTube worker uses. This module only adapts that pipeline to the Core
STT contract. Import of ``GpuTranscriber`` is deferred so constructing the backend does not fail
when the helper is absent — warmup/transcribe surface the error, and ``FallbackSTT`` can catch it.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .base import SpeechToText, SttError, TranscriptionResult, TranscriptSegment

log = logging.getLogger(__name__)

ProgressHook = Callable[[float], None]


def _bootstrap_transcription(data_dir: Path) -> None:
    root = str(data_dir.expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class RemoteGpuWhisperSTT(SpeechToText):
    def __init__(
        self,
        *,
        config_path: Path,
        data_dir: Path,
        language: str = "auto",
        beam_size: int = 5,
        max_concurrent: int = 1,
    ) -> None:
        self._config_path = config_path.expanduser().resolve()
        self._data_dir = data_dir.expanduser().resolve()
        self._language = None if language in ("auto", "", None) else language
        self._beam_size = beam_size
        self._transcriber = None
        self._ready = False
        self._slots = asyncio.Semaphore(max(1, max_concurrent))

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return "gpu/faster-whisper-large-v3"

    def _get_transcriber(self):
        if self._transcriber is None:
            _bootstrap_transcription(self._data_dir)
            from transcription.gpu_remote import GpuTranscriber

            self._transcriber = GpuTranscriber(self._config_path)
        return self._transcriber

    async def warmup(self) -> None:
        try:
            await asyncio.to_thread(self._get_transcriber().ensure_ready)
        except Exception as exc:
            raise SttError(f"GPU whisper setup failed: {exc}") from exc
        self._ready = True
        log.info("remote GPU whisper ready (%s)", self._config_path)

    async def close(self) -> None:
        self._ready = False

    async def transcribe(
        self, audio_path: Path, *, on_progress: ProgressHook | None = None
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise SttError(f"audio file not found: {audio_path.name}")
        if audio_path.stat().st_size == 0:
            raise SttError("audio file is empty")

        if not self._ready:
            await self.warmup()

        async with self._slots:
            started = time.monotonic()
            try:
                payload = await asyncio.to_thread(
                    self._transcribe_blocking, audio_path, on_progress
                )
            except SttError:
                raise
            except Exception as exc:
                raise SttError(f"remote GPU transcription failed: {exc}") from exc

        segments = [
            TranscriptSegment(item["start"], item["end"], item["text"])
            for item in payload.get("segments", [])
        ]
        result = TranscriptionResult(
            text=payload.get("text", "").strip(),
            language=payload.get("language"),
            duration=payload.get("duration"),
            segments=segments,
        )
        log.info(
            "transcribed %s on GPU in %.1fs (%.0fs audio, %d segments, lang=%s)",
            audio_path.name,
            time.monotonic() - started,
            result.duration or 0,
            len(result.segments),
            result.language,
        )
        return result

    def _transcribe_blocking(
        self, audio_path: Path, on_progress: ProgressHook | None
    ) -> dict:
        lang = self._language or "ru"
        return self._get_transcriber().transcribe(
            audio_path,
            namespace="voice",
            language=lang,
            beam_size=self._beam_size,
            on_fraction=on_progress,
            auto_setup=True,
            repair_on_failure=True,
        )
