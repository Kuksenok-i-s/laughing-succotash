"""Serial execution per conversation, concurrent across conversations.

A user who fires off "text A", "text B" and a voice note within two seconds must not have three
turns interleaved in one Cursor session — the session has a single linear history, and concurrent
prompts would scramble it. So each conversation gets a queue and a worker; separate users are
completely independent.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

Runner = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class _Item:
    job_id: str
    runner: Runner


class JobManager:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[_Item]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._running: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()
        self._stopping = False

    # ---- submission ----------------------------------------------------

    async def submit(self, conversation_id: str, job_id: str, runner: Runner) -> None:
        if self._stopping:
            raise RuntimeError("core is shutting down")
        queue = self._queues.get(conversation_id)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[conversation_id] = queue
            self._workers[conversation_id] = asyncio.ensure_future(
                self._worker(conversation_id, queue)
            )
        await queue.put(_Item(job_id, runner))

    async def _worker(self, conversation_id: str, queue: asyncio.Queue[_Item]) -> None:
        while not self._stopping:
            item = await queue.get()
            try:
                if item.job_id in self._cancelled:
                    self._cancelled.discard(item.job_id)
                    log.info("skipping cancelled job %s", item.job_id)
                    continue

                task = asyncio.ensure_future(item.runner())
                self._running[item.job_id] = task
                try:
                    await task
                except asyncio.CancelledError:
                    # The job was cancelled, not the worker. Keep serving this conversation.
                    log.info("job %s cancelled", item.job_id)
                    if self._stopping:
                        raise
                except Exception:
                    log.exception("job %s crashed", item.job_id)
                finally:
                    self._running.pop(item.job_id, None)
            finally:
                queue.task_done()

    # ---- cancellation ----------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job, or tombstone a queued one. Returns whether anything matched."""
        task = self._running.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        # Not started yet: remember the decision so the worker drops it when it comes up.
        self._cancelled.add(job_id)
        return False

    def cancel_all(self, job_ids: list[str]) -> int:
        return sum(1 for job_id in job_ids if self.cancel(job_id))

    def is_running(self, job_id: str) -> bool:
        task = self._running.get(job_id)
        return task is not None and not task.done()

    @property
    def running_count(self) -> int:
        return sum(1 for task in self._running.values() if not task.done())

    @property
    def queued_count(self) -> int:
        return sum(queue.qsize() for queue in self._queues.values())

    async def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until every queue has drained. Used by shutdown and by tests."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self.queued_count == 0 and self.running_count == 0:
                # Yield once more so a worker between two items settles.
                await asyncio.sleep(0)
                if self.queued_count == 0 and self.running_count == 0:
                    return True
            await asyncio.sleep(0.01)
        return False

    # ---- shutdown -------------------------------------------------------

    async def drain(self, timeout: float = 10.0) -> None:
        """Let in-flight jobs finish, then stop the workers.

        Queued-but-unstarted work is deliberately abandoned: it is still durable on the Gateway,
        which resubmits it with the same request_id after the restart.
        """
        self._stopping = True
        running = [task for task in self._running.values() if not task.done()]
        if running:
            log.info("waiting for %d running job(s)", len(running))
            done, pending = await asyncio.wait(running, timeout=timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        for worker in self._workers.values():
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._workers.clear()
        self._queues.clear()
