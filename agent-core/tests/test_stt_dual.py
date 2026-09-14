"""Two GPU hosts: split a long file, skip the second card while OCR is busy."""

from __future__ import annotations

import asyncio

import pytest

from pathlib import Path

from agent_core.stt.base import TranscriptionResult, TranscriptSegment
from agent_core.stt.dual import DualHostSTT


class FakeGpu:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls: list[tuple[str, str | None]] = []
        self.prepared: list[bool] = []
        self.warmups = 0
        self.ready = True
        self.model_name = name

    async def warmup(self) -> None:
        self.warmups += 1

    async def close(self) -> None:
        return None

    async def transcribe(
        self,
        path: Path,
        *,
        on_progress=None,
        on_notice=None,
        language=None,
        prepared=False,
    ) -> TranscriptionResult:
        self.calls.append((path.name, language))
        self.prepared.append(prepared)
        await asyncio.sleep(0)
        if on_progress is not None:
            on_progress(1.0)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        label = path.stem
        return TranscriptionResult(
            text=f"{self.name}-{label}",
            language=language or "ru",
            duration=10.0,
            segments=[TranscriptSegment(5.0, 10.0, f"{self.name}-{label}")],
        )


async def _probe(_path: Path) -> float:
    return 1800.0


async def _extract(_source: Path, dest: Path, start: float, length: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x")
    return dest


def _dual(tmp_path: Path, **kwargs) -> DualHostSTT:
    options = {
        "primary": FakeGpu("p"),
        "secondary": FakeGpu("s"),
        "temp_dir": tmp_path,
        "probe": _probe,
        "extract": _extract,
        "ocr_busy": _idle,
        "chunk_seconds": 600.0,
        "overlap_seconds": 2.0,
    }
    options.update(kwargs)
    return DualHostSTT(**options)


async def _idle() -> bool:
    return False


async def _busy() -> bool:
    return True


async def test_short_audio_stays_on_the_primary(tmp_path: Path) -> None:
    primary = FakeGpu("p")
    secondary = FakeGpu("s")
    audio = tmp_path / "short.ogg"
    audio.write_bytes(b"x")

    async def probe(_path: Path) -> float:
        return 120.0

    stt = _dual(tmp_path, primary=primary, secondary=secondary, probe=probe)
    result = await stt.transcribe(audio)

    assert result.text == "p-short"
    assert primary.calls == [("short.ogg", None)]
    assert secondary.calls == []


async def test_ocr_busy_keeps_the_second_card_off(tmp_path: Path) -> None:
    primary = FakeGpu("p")
    secondary = FakeGpu("s")
    audio = tmp_path / "talk.ogg"
    audio.write_bytes(b"x")
    stt = _dual(tmp_path, primary=primary, secondary=secondary, ocr_busy=_busy)

    result = await stt.transcribe(audio)

    assert result.text == "p-talk"
    assert secondary.calls == []
    assert primary.calls == [("talk.ogg", None)]


async def test_a_long_file_uses_both_gpus(tmp_path: Path) -> None:
    primary = FakeGpu("p")
    secondary = FakeGpu("s")
    audio = tmp_path / "talk.ogg"
    audio.write_bytes(b"x")
    stt = _dual(tmp_path, primary=primary, secondary=secondary)

    result = await stt.transcribe(audio)

    assert primary.calls[0] == ("000.wav", None)
    assert secondary.calls[0] == ("001.wav", None)
    names = [name for name, _ in primary.calls + secondary.calls]
    assert sorted(names) == ["000.wav", "001.wav", "002.wav", "003.wav"]
    assert all(lang == "ru" for name, lang in primary.calls + secondary.calls if name >= "002.wav")
    assert result.text.index("000") < result.text.index("001") < result.text.index("002") < result.text.index("003")
    assert result.duration == 1800.0
    # Slices carry the filter chain already; the GPU host must not run loudnorm again.
    assert all(primary.prepared) and all(secondary.prepared)


async def test_slices_are_dispatched_as_they_are_cut(tmp_path: Path) -> None:
    """Slice zero goes to a GPU while later slices are still in ffmpeg."""
    gates = {index: asyncio.Event() for index in range(4)}
    cut_order: list[int] = []
    dispatched = asyncio.Event()

    async def slow_extract(_source: Path, dest: Path, start: float, length: float) -> Path:
        index = int(dest.stem)
        await gates[index].wait()
        cut_order.append(index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    class Gpu(FakeGpu):
        async def transcribe(self, path, **kwargs):
            if path.name == "000.wav":
                dispatched.set()
            return await super().transcribe(path, **kwargs)

    primary, secondary = Gpu("p"), Gpu("s")
    stt = _dual(tmp_path, primary=primary, secondary=secondary, extract=slow_extract, cut_parallel=2)
    task = asyncio.create_task(stt.transcribe(tmp_path / "talk.wav"))
    try:
        gates[0].set()
        await asyncio.wait_for(dispatched.wait(), 1)
        assert cut_order == [0]
        assert not task.done()
        for index in (1, 2, 3):
            gates[index].set()
        result = await asyncio.wait_for(task, 1)
    finally:
        for gate in gates.values():
            gate.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    names = sorted(name for name, _ in primary.calls + secondary.calls)
    assert names == ["000.wav", "001.wav", "002.wav", "003.wav"]
    assert result.text.index("000") < result.text.index("003")


async def test_cutting_runs_at_most_cut_parallel_at_once(tmp_path: Path) -> None:
    in_flight = {"now": 0, "peak": 0}

    async def extract(_source: Path, dest: Path, start: float, length: float) -> Path:
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    async def probe(_path: Path) -> float:
        return 3600.0

    stt = _dual(tmp_path, extract=extract, probe=probe, cut_parallel=2)
    await asyncio.wait_for(stt.transcribe(tmp_path / "talk.wav"), 2)

    assert in_flight["peak"] == 2


async def test_a_secondary_failure_retries_on_primary(tmp_path: Path) -> None:
    primary = FakeGpu("p")
    secondary = FakeGpu("s", fail=True)
    audio = tmp_path / "talk.ogg"
    audio.write_bytes(b"x")
    stt = _dual(tmp_path, primary=primary, secondary=secondary)

    result = await stt.transcribe(audio)

    assert "p-001" in result.text
    assert any(name == "001.wav" for name, _ in primary.calls)


async def test_split_failure_falls_back_to_the_whole_file(tmp_path: Path) -> None:
    primary = FakeGpu("p")
    audio = tmp_path / "talk.ogg"
    audio.write_bytes(b"x")

    async def boom(*_args):
        raise RuntimeError("ffmpeg down")

    stt = _dual(tmp_path, primary=primary, extract=boom)
    result = await stt.transcribe(audio)

    assert result.text == "p-talk"
    assert primary.calls == [("talk.ogg", None)]


@pytest.mark.parametrize("slow_host", ["primary", "secondary"])
async def test_fast_host_takes_more_work_without_waiting(tmp_path, slow_host):
    started = asyncio.Event()
    release = asyncio.Event()
    last_started = asyncio.Event()

    class Slow(FakeGpu):
        async def transcribe(self, path, **kwargs):
            started.set()
            await release.wait()
            return await super().transcribe(path, **kwargs)

    class Fast(FakeGpu):
        async def transcribe(self, path, **kwargs):
            await started.wait()
            if path.name == "003.wav":
                last_started.set()
            return await super().transcribe(path, **kwargs)

    slow, fast = Slow("slow"), Fast("fast")
    stt = _dual(tmp_path, **{slow_host: slow, "secondary" if slow_host == "primary" else "primary": fast})
    task = asyncio.create_task(stt.transcribe(tmp_path / "talk.wav"))
    try:
        await asyncio.wait_for(last_started.wait(), 1)
        assert not task.done()
        release.set()
        result = await asyncio.wait_for(task, 1)
        assert len(slow.calls) == 1
        assert len(fast.calls) == 3
        assert result.text.index("000") < result.text.index("001") < result.text.index("002")
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_secondary_failure_after_primary_is_idle_is_retried(tmp_path):
    primary_finished = asyncio.Event()

    class Primary(FakeGpu):
        async def transcribe(self, path, **kwargs):
            result = await super().transcribe(path, **kwargs)
            if path.name == "003.wav":
                primary_finished.set()
            return result

    class Secondary(FakeGpu):
        async def transcribe(self, path, **kwargs):
            await primary_finished.wait()
            return await super().transcribe(path, **kwargs)

    primary, secondary = Primary("p"), Secondary("s", fail=True)
    result = await asyncio.wait_for(
        _dual(tmp_path, primary=primary, secondary=secondary).transcribe(tmp_path / "talk.wav"), 1
    )
    assert len(secondary.calls) == 1
    assert [name for name, _ in primary.calls] == ["000.wav", "002.wav", "003.wav", "001.wav"]
    assert "p-001" in result.text


@pytest.mark.parametrize("primary_fails", [False, True])
async def test_failure_or_cancellation_stops_inflight_chunks_before_cleanup(tmp_path, primary_fails):
    secondary_started = asyncio.Event()
    secondary_stopped = asyncio.Event()

    class Primary(FakeGpu):
        async def transcribe(self, path, **kwargs):
            await secondary_started.wait()
            if primary_fails:
                raise RuntimeError("primary unavailable")
            await asyncio.Event().wait()

    class Secondary(FakeGpu):
        async def transcribe(self, path, **kwargs):
            secondary_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                assert path.exists()
                secondary_stopped.set()

    stt = _dual(tmp_path, primary=Primary("p"), secondary=Secondary("s"))
    task = asyncio.create_task(stt.transcribe(tmp_path / "talk.wav"))
    await asyncio.wait_for(secondary_started.wait(), 1)
    if primary_fails:
        with pytest.raises(RuntimeError, match="primary unavailable"):
            await asyncio.wait_for(task, 1)
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert secondary_stopped.is_set()
    assert not list(tmp_path.glob("stt-chunks-*"))


async def test_two_chunks_start_together(tmp_path):
    started = {"p": asyncio.Event(), "s": asyncio.Event()}

    class RendezvousGpu(FakeGpu):
        async def transcribe(self, path, **kwargs):
            started[self.name].set()
            await started["s" if self.name == "p" else "p"].wait()
            return await super().transcribe(path, **kwargs)

    async def probe(_path):
        return 900.0

    primary, secondary = RendezvousGpu("p"), RendezvousGpu("s")
    result = await asyncio.wait_for(
        _dual(tmp_path, primary=primary, secondary=secondary, probe=probe).transcribe(tmp_path / "talk.wav"), 1
    )
    assert primary.calls == [("000.wav", None)]
    assert secondary.calls == [("001.wav", None)]
    assert result.duration == 900
