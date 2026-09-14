"""faster-whisper on the local GPU.

The model is loaded on the worker thread, not at HTTP bind, and dropped after a stretch of idle
time so OCR and Whisper can share one card.

Decoding is batched: Silero VAD cuts the file into speech windows and ``batch_size`` of them go
through the encoder and decoder together. On Xavier the decoder is the bottleneck, so this is
worth 2–4x over one window at a time. Batched mode needs VAD; with it off we fall back to the
sequential path.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
try:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except ImportError:
    pass
log = logging.getLogger(__name__)

# percent, position in the audio, total duration, segments so far.
ProgressHook = Callable[[float, float, float | None, int], None]


class Engine(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None,
        beam_size: int | None,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]: ...


class WhisperEngine:
    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 2,
        vad_filter: bool = True,
        batch_size: int = 8,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._batch_size = max(0, batch_size)
        self._model: Any = None
        self._batched: Any = None

    @property
    def batched(self) -> bool:
        return self._batch_size > 0 and self._vad_filter

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        started = time.monotonic()
        self._model = WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )
        self._batched = BatchedInferencePipeline(self._model) if self.batched else None
        log.info(
            "whisper %s ready in %.1fs (device=%s compute=%s batch=%s)",
            self._model_name,
            time.monotonic() - started,
            self._device,
            self._compute_type,
            self._batch_size if self.batched else "off",
        )

    def unload(self) -> None:
        if self._model is None:
            return
        self._batched = None
        self._model = None
        gc.collect()

        log.info("whisper %s unloaded", self._model_name)

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None,
        beam_size: int | None,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]:
        if self._model is None:
            self.load()

        options: dict[str, Any] = {
            "language": language,
            "beam_size": beam_size or self._beam_size,
            "vad_filter": self._vad_filter,
            # Word timestamps roughly double the cost and nothing downstream uses them.
            "word_timestamps": False,
            # Each window decodes on its own: faster, and a hallucinated phrase cannot seed the
            # next thirty seconds and repeat itself for a minute.
            "condition_on_previous_text": False,
        }
        if self._batched is not None:
            segments_iter, info = self._batched.transcribe(
                str(audio_path), batch_size=self._batch_size, **options
            )
        else:
            segments_iter, info = self._model.transcribe(str(audio_path), **options)

        duration = float(getattr(info, "duration", 0.0) or 0.0) or None
        collected: list[dict[str, Any]] = []

        # faster-whisper yields lazily, so this loop *is* the transcription: reporting per segment
        # is what turns an hour of silence into a moving percentage.
        for segment in segments_iter:
            text = segment.text.strip()
            collected.append({"start": segment.start, "end": segment.end, "text": text})
            if on_progress is not None:
                percent = 100.0 if not duration else min(100.0, segment.end / duration * 100.0)
                on_progress(percent, float(segment.end), duration, len(collected))

        return {
            "text": " ".join(item["text"] for item in collected).strip(),
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": duration,
            "segments": collected,
        }
