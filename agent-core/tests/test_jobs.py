"""Job scheduling: serial within a conversation, parallel across them, cancellable."""

from __future__ import annotations

import asyncio

from agent_core.jobs.manager import JobManager


async def test_one_conversation_runs_its_jobs_in_order() -> None:
    """Three quick messages from one user must not interleave in a single Cursor session."""
    manager = JobManager()
    events: list[str] = []

    def make(name: str, delay: float):
        async def run() -> None:
            events.append(f"{name}:start")
            await asyncio.sleep(delay)
            events.append(f"{name}:end")

        return run

    await manager.submit("conv", "a", make("a", 0.03))
    await manager.submit("conv", "b", make("b", 0.0))
    await manager.submit("conv", "c", make("c", 0.0))

    assert await manager.wait_idle()
    assert events == [
        "a:start", "a:end", "b:start", "b:end", "c:start", "c:end",
    ]
    await manager.drain()


async def test_different_conversations_run_concurrently() -> None:
    manager = JobManager()
    started = asyncio.Event()
    released = asyncio.Event()

    async def blocking() -> None:
        started.set()
        await released.wait()

    async def quick() -> None:
        # Only reachable while the other conversation is still blocked.
        released.set()

    await manager.submit("conv-a", "slow", blocking)
    await started.wait()
    await manager.submit("conv-b", "fast", quick)

    assert await manager.wait_idle(timeout=2.0)
    await manager.drain()


async def test_cancelling_a_running_job_stops_it() -> None:
    manager = JobManager()
    started = asyncio.Event()
    finished = False

    async def long_job() -> None:
        nonlocal finished
        started.set()
        await asyncio.sleep(5)
        finished = True

    await manager.submit("conv", "job-1", long_job)
    await started.wait()

    assert manager.cancel("job-1") is True
    assert await manager.wait_idle(timeout=2.0)
    assert finished is False
    await manager.drain()


async def test_cancelling_a_queued_job_prevents_it_from_starting() -> None:
    manager = JobManager()
    gate = asyncio.Event()
    ran: list[str] = []

    async def first() -> None:
        await gate.wait()
        ran.append("first")

    async def second() -> None:
        ran.append("second")

    await manager.submit("conv", "job-1", first)
    await manager.submit("conv", "job-2", second)

    assert manager.cancel("job-2") is False  # not started yet, so tombstoned
    gate.set()

    assert await manager.wait_idle(timeout=2.0)
    assert ran == ["first"]
    await manager.drain()


async def test_a_crashing_job_does_not_stop_the_queue() -> None:
    manager = JobManager()
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("cursor died")

    async def after() -> None:
        ran.append("after")

    await manager.submit("conv", "job-1", boom)
    await manager.submit("conv", "job-2", after)

    assert await manager.wait_idle(timeout=2.0)
    assert ran == ["after"]
    await manager.drain()


async def test_drain_cancels_work_that_overruns_the_deadline() -> None:
    manager = JobManager()
    started = asyncio.Event()

    async def forever() -> None:
        started.set()
        await asyncio.sleep(30)

    await manager.submit("conv", "job-1", forever)
    await started.wait()

    await manager.drain(timeout=0.05)
    assert manager.running_count == 0
