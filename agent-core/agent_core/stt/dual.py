"""Run 10-minute slices on two GPU hosts when the OCR card is free.

The primary transcriber still splits a whole file by itself; this wrapper is only the parallel
path. Both hosts consume a shared queue; each free host takes the next slice.
The first slices detect their language independently until slice zero supplies it.
A queued or running OCR job on the second GPU host keeps that card off the roster so
handwriting and whisper do not share it.

Slices are cut by a few ffmpeg processes at once and dispatched as each becomes ready, so the
GPUs start on slice zero while the rest of the hour is still being cut. The cut already applies
the Whisper filter chain, and the upload says so (``prepared``) so the GPU host does not run
loudnorm a second time over the same ten minutes.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiohttp

from ..audio.converter import extract_slice, probe_duration
from .base import NoticeHook, ProgressHook, SpeechToText, TranscriptionResult
from .chunks import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    chunk_windows,
    merge_results,
)

log = logging.getLogger(__name__)

Probe = Callable[[Path], Awaitable[float | None]]
Extract = Callable[[Path, Path, float, float], Awaitable[Path]]
Busy = Callable[[], Awaitable[bool]]

# ffmpeg loudnorm is single-threaded; two cutters keep both GPUs fed on an 8-core board without
# taking the CPU the local transcriber needs for its own decode.
DEFAULT_CUT_PARALLEL = 2


class _SplitFailed(Exception):
    """A slice could not be cut; the caller sends the whole file to the primary instead."""


class DualHostSTT:
    def __init__(
        self,
        *,
        primary: SpeechToText,
        secondary: SpeechToText,
        ocr_health_url: str = "",
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
        temp_dir: Path,
        probe: Probe = probe_duration,
        extract: Extract = extract_slice,
        ocr_busy: Busy | None = None,
        cut_parallel: int = DEFAULT_CUT_PARALLEL,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._ocr_health_url = ocr_health_url.rstrip("/")
        self._chunk_seconds = chunk_seconds
        self._overlap_seconds = overlap_seconds
        self._temp_dir = temp_dir
        self._probe = probe
        self._extract = extract
        self._ocr_busy = ocr_busy
        self._cut_parallel = max(1, cut_parallel)

    @property
    def ready(self) -> bool:
        return bool(getattr(self._primary, "ready", False))

    @property
    def model_name(self) -> str:
        return getattr(self._primary, "model_name", "gpu-service")

    async def warmup(self) -> None:
        await self._primary.warmup()
        try:
            await self._secondary.warmup()
        except Exception as exc:
            log.info("secondary GPU transcriber unavailable: %s", exc)

    async def close(self) -> None:
        await self._primary.close()
        try:
            await self._secondary.close()
        except Exception:
            log.debug("secondary GPU transcriber close failed", exc_info=True)

    async def transcribe(
        self,
        audio_path: Path,
        *,
        on_progress: ProgressHook | None = None,
        on_notice: NoticeHook | None = None,
    ) -> TranscriptionResult:
        duration = await self._probe(audio_path)
        windows = chunk_windows(duration or 0.0, self._chunk_seconds, self._overlap_seconds)
        if duration is None or len(windows) <= 1 or not await self._secondary_free():
            return await self._primary.transcribe(
                audio_path, on_progress=on_progress, on_notice=on_notice
            )

        work = self._temp_dir / f"stt-chunks-{audio_path.stem}"
        work.mkdir(parents=True, exist_ok=True)
        log.info(
            "splitting %s into %d chunks across two GPUs (%.0fs audio)",
            audio_path.name,
            len(windows),
            duration,
        )
        cutting = self._cut(audio_path, windows, work)
        try:
            return await self._run_slices(cutting, windows, duration, on_progress)
        except _SplitFailed:
            log.warning("could not split audio; sending the whole file to the primary GPU")
            shutil.rmtree(work, ignore_errors=True)
            return await self._primary.transcribe(
                audio_path, on_progress=on_progress, on_notice=on_notice
            )
        finally:
            for task in cutting:
                task.cancel()
            await asyncio.gather(*cutting, return_exceptions=True)
            shutil.rmtree(work, ignore_errors=True)

    def _cut(
        self,
        audio_path: Path,
        windows: list[tuple[float, float]],
        work: Path,
    ) -> list[asyncio.Task[Path]]:
        """Start cutting every window; at most ``cut_parallel`` ffmpeg processes at a time."""
        gate = asyncio.Semaphore(self._cut_parallel)

        async def cut(index: int, start: float, length: float) -> Path:
            async with gate:
                return await self._extract(audio_path, work / f"{index:03d}.wav", start, length)

        return [
            asyncio.create_task(cut(index, start, length))
            for index, (start, length) in enumerate(windows)
        ]

    async def _run_slices(
        self,
        cutting: list[asyncio.Task[Path]],
        windows: list[tuple[float, float]],
        duration: float,
        on_progress: ProgressHook | None,
    ) -> TranscriptionResult:
        fractions = [0.0] * len(cutting)
        lengths = [length for _, length in windows]
        offsets = [start for start, _ in windows]

        def hook(index: int, fraction: float) -> None:
            fractions[index] = min(max(fraction, 0.0), 1.0)
            if on_progress is None:
                return
            done = sum(frac * length for frac, length in zip(fractions, lengths, strict=True))
            on_progress(min(max(done / duration, 0.0), 1.0))

        async def run(host: SpeechToText, index: int, path: Path, language: str | None):
            result = await host.transcribe(  # type: ignore[call-arg]
                path,
                on_progress=lambda fraction, _index=index: hook(_index, fraction),
                language=language,
                prepared=True,
            )
            hook(index, 1.0)
            return result

        parts: list[tuple[float, TranscriptionResult]] = []
        pending = deque(range(len(cutting)))
        idle = [self._primary, self._secondary]
        active: dict[asyncio.Task, tuple[SpeechToText, int]] = {}
        language: str | None = None
        try:
            while pending or active:
                # A free host takes the lowest-numbered slice whose cut has finished.
                while pending and idle and cutting[pending[0]].done():
                    index = pending.popleft()
                    if cutting[index].cancelled() or cutting[index].exception() is not None:
                        raise _SplitFailed(index)
                    host = idle.pop(0)
                    host_name = "primary" if host is self._primary else "secondary"
                    log.info("dispatching chunk %d/%d to %s GPU", index + 1, len(cutting), host_name)
                    task = asyncio.create_task(
                        run(host, index, cutting[index].result(), language)
                    )
                    active[task] = (host, index)

                waiting: set[asyncio.Task] = set(active)
                if pending and idle:
                    waiting.add(cutting[pending[0]])
                completed, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                for task in completed:
                    if task not in active:
                        continue  # a slice finished cutting; the loop above dispatches it
                    host, index = active.pop(task)
                    try:
                        outcome = task.result()
                    except Exception as exc:
                        if host is self._primary:
                            raise
                        log.warning(
                            "secondary GPU failed on chunk %s (%s); disabling it for this "
                            "recording and requeueing the chunk on primary",
                            index,
                            exc,
                        )
                        pending.appendleft(index)
                        continue
                    parts.append((offsets[index], outcome))
                    if index == 0:
                        language = outcome.language
                    idle.append(host)
        finally:
            # Finish cancellation before transcribe() removes the audio slices.
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

        parts.sort(key=lambda item: item[0])
        merged = merge_results(parts, self._overlap_seconds)
        merged.duration = duration
        return merged

    async def _secondary_free(self) -> bool:
        if self._ocr_busy is not None:
            try:
                if await self._ocr_busy():
                    log.info("second GPU is running OCR; transcribing on the primary only")
                    return False
            except Exception:
                log.info("OCR occupancy check failed; transcribing on the primary only")
                return False
        elif self._ocr_health_url and await self._ocr_job_active():
            log.info("second GPU is running OCR; transcribing on the primary only")
            return False
        try:
            await self._secondary.warmup()
        except Exception as exc:
            log.info("secondary GPU transcriber unavailable: %s", exc)
            return False
        return True

    async def _ocr_job_active(self) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self._ocr_health_url + "/health") as response:
                    data = await response.json(content_type=None)
        except (TimeoutError, aiohttp.ClientError, ValueError):
            return False
        return int(data.get("queued") or 0) > 0 or int(data.get("running") or 0) > 0
