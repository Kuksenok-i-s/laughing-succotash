"""faster-whisper on the local GPU, loaded once and kept.

Loading large-v3 costs far more than a typical transcription, which is the reason this service
exists at all: the SSH pipeline it replaces paid that cost on every single job, plus a virtualenv
check, a script upload and a process launch.

The import is deferred so the module can be exercised without CUDA or faster-whisper present.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

# percent, position in the audio, total duration, segments so far.
ProgressHook = Callable[[float, float, float | None, int], None]


class Engine(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    def load(self) -> None: ...

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
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model: Any = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        from faster_whisper import WhisperModel

        started = time.monotonic()
        self._model = WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )
        log.info(
            "whisper %s ready in %.1fs (device=%s compute=%s)",
            self._model_name,
            time.monotonic() - started,
            self._device,
            self._compute_type,
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None,
        beam_size: int | None,
        on_progress: ProgressHook | None = None,
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("model is not loaded yet")

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size or self._beam_size,
            vad_filter=self._vad_filter,
            # Word timestamps roughly double the cost and nothing downstream uses them.
            word_timestamps=False,
            condition_on_previous_text=True,
        )

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
