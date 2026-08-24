"""Local transcription with faster-whisper.

Accuracy is the priority, not latency: this runs large-v3 on an Intel Mac mini's CPU, where an
hour of audio takes a long while. That is acceptable because transcription is a background job and
the user is told what stage it is at.

Two constraints shape the implementation. The model is loaded once and kept, because loading
large-v3 costs far more than a typical transcription. And inference is CPU-bound C++ that would
block the event loop for minutes, so it runs in a worker thread with a semaphore capping
concurrency (``STT_MAX_CONCURRENT=1`` by default).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from .base import SpeechToText, SttError, TranscriptionResult, TranscriptSegment

log = logging.getLogger(__name__)

ProgressHook = Callable[[float], None]


class FasterWhisperSTT(SpeechToText):
    def __init__(
        self,
        *,
        model: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "auto",
        language: str = "auto",
        beam_size: int = 5,
        vad_filter: bool = True,
        max_concurrent: int = 1,
        download_root: Path | None = None,
        cpu_threads: int = 0,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = _resolve_compute_type(compute_type, device)
        self._language = None if language in ("auto", "", None) else language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._download_root = download_root
        self._cpu_threads = cpu_threads

        self._model = None
        self._load_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max(1, max_concurrent))

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    async def warmup(self) -> None:
        """Load the model ahead of the first request so the first voice message is not slowest."""
        await self._ensure_model()

    async def close(self) -> None:
        self._model = None

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            log.info(
                "loading whisper model %s (device=%s compute=%s)",
                self._model_name, self._device, self._compute_type,
            )
            started = time.monotonic()
            try:
                self._model = await asyncio.to_thread(self._load_model)
            except ImportError as exc:
                raise SttError(
                    "faster-whisper is not installed; install it on the Agent Core machine"
                ) from exc
            except Exception as exc:
                raise SttError(f"could not load whisper model {self._model_name}: {exc}") from exc
            log.info("whisper model ready in %.1fs", time.monotonic() - started)
            return self._model

    def _load_model(self):
        from faster_whisper import WhisperModel

        kwargs = {
            "device": self._device,
            "compute_type": self._compute_type,
        }
        if self._download_root is not None:
            kwargs["download_root"] = str(self._download_root)
        if self._cpu_threads:
            kwargs["cpu_threads"] = self._cpu_threads
        return WhisperModel(self._model_name, **kwargs)

    async def transcribe(
        self, audio_path: Path, *, on_progress: ProgressHook | None = None
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise SttError(f"audio file not found: {audio_path.name}")
        if audio_path.stat().st_size == 0:
            raise SttError("audio file is empty")

        model = await self._ensure_model()

        # One transcription at a time by default: two large-v3 runs on this hardware are slower
        # together than one after the other, and the memory spike risks the whole process.
        async with self._slots:
            started = time.monotonic()
            loop = asyncio.get_running_loop()

            def emit(fraction: float) -> None:
                if on_progress is not None:
                    loop.call_soon_threadsafe(on_progress, fraction)

            try:
                result = await asyncio.to_thread(self._run, model, audio_path, emit)
            except SttError:
                raise
            except Exception as exc:
                raise SttError(f"transcription failed: {type(exc).__name__}: {exc}") from exc

        log.info(
            "transcribed %s in %.1fs (%.0fs audio, %d segments, lang=%s)",
            audio_path.name, time.monotonic() - started, result.duration or 0,
            len(result.segments), result.language,
        )
        return result

    def _run(self, model, audio_path: Path, emit: ProgressHook) -> TranscriptionResult:
        """Blocking transcription. Runs in a worker thread."""
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            # Word timestamps would roughly double the cost and nothing downstream uses them.
            word_timestamps=False,
            condition_on_previous_text=True,
        )

        total = getattr(info, "duration", None) or 0.0
        segments: list[TranscriptSegment] = []
        # faster-whisper only does the work as the generator is consumed, which is precisely what
        # makes incremental progress reporting possible for a long recording.
        for segment in segments_iter:
            text = (segment.text or "").strip()
            if text:
                segments.append(TranscriptSegment(segment.start, segment.end, text))
            if total:
                emit(min(segment.end / total, 1.0))

        return TranscriptionResult(
            text=" ".join(segment.text for segment in segments).strip(),
            language=getattr(info, "language", None),
            duration=total or None,
            segments=segments,
        )


def _resolve_compute_type(requested: str, device: str) -> str:
    """Pick a compute type that this hardware can actually run.

    ``auto`` on an Intel CPU means int8: float16 is unsupported there and CTranslate2 would either
    refuse or fall back silently after loading the whole model.
    """
    if requested and requested != "auto":
        return requested
    return "int8" if device == "cpu" else "float16"
