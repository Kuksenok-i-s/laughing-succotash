"""CPU fallback around a primary STT backend.

Used when ``STT_BACKEND=gpu`` and ``STT_CPU_FALLBACK=true``: if the GPU host is unreachable or a
transcription fails, use the CPU temporarily and retry the GPU after a cooldown.
Serialize requests across both backends to avoid overlapping model loads during recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .base import (
    STT_CPU_FALLBACK,
    NoticeHook,
    ProgressHook,
    SpeechToText,
    TranscriptionResult,
)

log = logging.getLogger(__name__)


class FallbackSTT:
    def __init__(self, *, primary: SpeechToText, fallback: SpeechToText,
                 retry_seconds: float = 60.0) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active: SpeechToText = primary
        self._retry_seconds = max(0.0, retry_seconds)
        self._retry_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return bool(getattr(self._active, "ready", False))

    @property
    def model_name(self) -> str:
        name = getattr(self._active, "model_name", "stt")
        if self._active is self._fallback:
            return f"fallback/{name}"
        return name

    async def warmup(self) -> None:
        try:
            await self._primary.warmup()
            self._active = self._primary
        except Exception as exc:
            log.warning("primary STT unavailable (%s); using CPU fallback", exc)
            await self._fallback.warmup()
            self._active = self._fallback
            self._retry_at = time.monotonic() + self._retry_seconds

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()

    async def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: ProgressHook | None = None,
        on_notice: NoticeHook | None = None,
    ) -> TranscriptionResult:
        async with self._lock:
            return await self._transcribe(audio_path, on_progress=on_progress, on_notice=on_notice)

    async def _transcribe(self, audio_path: Path, *, on_progress=None, on_notice=None):
        recovering = self._active is self._fallback
        if recovering and time.monotonic() >= self._retry_at:
            self._retry_at = time.monotonic() + self._retry_seconds
            try:
                await self._primary.warmup()
            except Exception as exc:
                log.info("primary STT still unavailable: %s", exc)
            else:
                self._active = self._primary
        if self._active is self._fallback:
            # Already degraded from an earlier failure: the caller still has to be told, otherwise
            # only the first recording after the switch explains the slower run.
            if on_notice is not None:
                on_notice(STT_CPU_FALLBACK)
            return await self._transcribe_on_cpu(audio_path, on_progress, on_notice)
        try:
            result = await self._primary.transcribe(
                audio_path, on_progress=on_progress, on_notice=on_notice
            )
        except Exception as exc:
            log.warning("primary STT failed (%s); switching to CPU fallback", exc)
            await self._fallback.warmup()
            self._active = self._fallback
            self._retry_at = time.monotonic() + self._retry_seconds
            if on_notice is not None:
                on_notice(STT_CPU_FALLBACK)
            return await self._transcribe_on_cpu(audio_path, on_progress, on_notice)
        if recovering:
            try:
                await self._fallback.close()
            except Exception:
                log.exception("could not release CPU fallback after GPU recovery")
            log.info("GPU transcription recovered")
        return result

    async def _transcribe_on_cpu(
        self,
        audio_path: Path,
        on_progress: ProgressHook | None,
        on_notice: NoticeHook | None,
    ) -> TranscriptionResult:
        return await self._fallback.transcribe(
            audio_path, on_progress=on_progress, on_notice=on_notice
        )
