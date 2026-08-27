"""The scheduler.

It belongs to the Core and depends on nothing else being awake: not Cursor, not the Gateway, not
the conversation the reminder was created in. A reminder fires because the clock says so, the
notification is committed to the durable outbound log, and delivery happens whenever the Gateway
is next reachable.

Firing is idempotent by construction. The delivery id is derived from the reminder and its fire
count, so a crash between "notification queued" and "reminder advanced" results in the same id on
the next attempt, and the Gateway drops the duplicate.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..mcp.timeutil import TimeParseError, format_local, next_occurrence

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        repos,
        assistant,
        *,
        tick_seconds: float = 5.0,
        confirmations=None,
        uploads=None,
        followup=None,
        journal=None,
        default_timezone: str = "UTC",
    ) -> None:
        self._repos = repos
        self._assistant = assistant
        self._tick = tick_seconds
        self._confirmations = confirmations
        self._uploads = uploads
        self.followup = followup
        self.journal = journal
        self._default_timezone = default_timezone

        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._stopping = asyncio.Event()
        self._sweeps = 0
        self.state = "stopped"

    async def start(self) -> None:
        self._task = asyncio.ensure_future(self._loop())
        self.state = "ready"
        pending = await self._repos.reminders.pending_count()
        log.info("scheduler started with %d pending reminder(s)", pending)

    async def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        self.state = "stopped"
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def wake(self) -> None:
        """Re-evaluate now — called when a reminder is created or changed."""
        self._wake.set()

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._tick)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

            if self._stopping.is_set():
                break

            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failing tick must never kill the scheduler: the next one may well succeed, and
                # a dead scheduler silently stops every reminder in the system.
                log.exception("scheduler tick failed")

    async def tick(self, now: datetime | None = None) -> None:
        moment = now or datetime.now(timezone.utc)
        await self._fire_reminders(moment)
        await self._fire_timers(moment)
        if self.journal is not None:
            await self.journal.tick(moment)

        if self._confirmations is not None:
            await self._confirmations.expire_overdue()

        self._sweeps += 1
        # Housekeeping is cheap but pointless every five seconds.
        if self._sweeps % 60 == 0:
            if self._uploads is not None:
                await self._uploads.sweep_stale()
            await self._repos.events.prune()

    # ---- reminders ------------------------------------------------------

    async def _fire_reminders(self, moment: datetime) -> None:
        for reminder in await self._repos.reminders.due_before(moment):
            chat_id = await self._repos.conversations.chat_for(reminder.user_id)
            if chat_id is None:
                log.warning(
                    "reminder %s has no known chat for %s; leaving it scheduled",
                    reminder.reminder_id, reminder.user_id,
                )
                continue

            user_tz = _zone(reminder.timezone, self._default_timezone)
            if self.followup is not None:
                await self.followup.offer(reminder, chat_id)
            else:
                await self._assistant.notify(
                    reminder.user_id,
                    chat_id,
                    f"⏰ {reminder.text}",
                    # Includes fire_count so each occurrence of a recurring reminder is a distinct
                    # delivery, while a retry of the same occurrence is not.
                    delivery_id=f"reminder:{reminder.reminder_id}:{reminder.fire_count}",
                )

            following = None
            if reminder.rrule:
                try:
                    following = next_occurrence(reminder.rrule, moment, user_tz)
                except TimeParseError:
                    log.warning(
                        "reminder %s has an invalid rrule; not rescheduling",
                        reminder.reminder_id,
                    )

            await self._repos.reminders.mark_fired(reminder.reminder_id, following)
            log.info(
                "fired reminder %s for %s (next: %s)",
                reminder.reminder_id, reminder.user_id,
                format_local(following, user_tz) if following else "none",
            )

    # ---- timers ---------------------------------------------------------

    async def _fire_timers(self, moment: datetime) -> None:
        for timer in await self._repos.timers.due_before(moment):
            chat_id = await self._repos.conversations.chat_for(timer["user_id"])
            if chat_id is None:
                await self._repos.timers.mark_fired(timer["timer_id"])
                continue

            label = timer.get("label")
            minutes = max(1, int(timer["duration_seconds"]) // 60)
            text = f"⏱ Таймер {label} — время вышло." if label else f"⏱ Таймер на {minutes} мин — время вышло."

            await self._assistant.notify(
                timer["user_id"], chat_id, text, delivery_id=f"timer:{timer['timer_id']}"
            )
            await self._repos.timers.mark_fired(timer["timer_id"])

    # ---- introspection ---------------------------------------------------

    async def snapshot(self) -> dict:
        upcoming = await self._repos.reminders.next_trigger()
        return {
            "state": self.state,
            "pending_reminders": await self._repos.reminders.pending_count(),
            "next_trigger": upcoming.isoformat() if upcoming else None,
        }


def _zone(name: str | None, fallback: str):
    for candidate in (name, fallback):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            continue
    return timezone.utc
