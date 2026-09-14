"""Buttoned follow-up after a reminder fires.

The scheduler calls :meth:`FollowupService.offer` and moves on. Button presses arrive later
through ``confirmation.resolve`` and are dispatched here even after a Core restart — there is no
in-memory waiter.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pa_protocol import methods, new_ulid

from ..mcp.timeutil import parse_datetime
from ..storage.repositories import PendingAction, Reminder
from . import policy

log = logging.getLogger(__name__)

TOOL = "reminder.followup"
TTL_SECONDS = 3 * 24 * 3600

_DONE = methods.ConfirmAction(id="done", label="Выполнено", style="primary")
_NOT_DONE = methods.ConfirmAction(id="not_done", label="Не выполнено", style="secondary")
_DROP = methods.ConfirmAction(id="drop", label="Отменить", style="danger")
_SNOOZE = methods.ConfirmAction(id="snooze", label="Напомнить", style="primary")
_RESCHEDULE = methods.ConfirmAction(id="reschedule", label="Переназначить", style="secondary")
_ACCEPT = methods.ConfirmAction(id="accept", label="Согласен", style="primary")
_REFUSE = methods.ConfirmAction(id="reject", label="Нет", style="secondary")


class FollowupService:
    TOOL = TOOL

    def __init__(
        self,
        repos,
        confirmations,
        link,
        *,
        default_timezone: str = "UTC",
        wake=None,
    ) -> None:
        self._repos = repos
        self._confirmations = confirmations
        self._link = link
        self._default_timezone = default_timezone
        self._wake = wake or (lambda: None)

    async def offer(self, reminder: Reminder, chat_id: int) -> None:
        """Queue the first pair of buttons. Safe to call twice for the same occurrence."""
        operation_id = f"followup:{reminder.reminder_id}:{reminder.fire_count}"
        if await self._repos.pending_actions.get_by_operation_id(operation_id) is not None:
            return

        await self._prompt(
            reminder,
            chat_id,
            step="ack",
            occurrence=reminder.fire_count,
            operation_id=operation_id,
            delivery_id=f"reminder:{reminder.reminder_id}:{reminder.fire_count}",
            text=(
                f"⏰ {reminder.text}\n\n"
                "Отметьте, сделано ли это."
            ),
            actions=[_DONE, _NOT_DONE, _DROP],
        )

    async def handle(self, action: PendingAction, choice: str) -> None:
        if choice in {"expired", "cancel"}:
            return

        args = action.arguments
        reminder = await self._repos.reminders.get(args.get("reminder_id", ""), action.user_id)
        if reminder is None or action.chat_id is None:
            log.warning("follow-up %s has no reminder or chat; dropping", action.action_id)
            return

        if choice == "drop":
            await self._drop(reminder, action)
            return

        step = args.get("step")
        try:
            if step == "ack":
                await self._on_ack(reminder, action, choice)
            elif step == "missed":
                await self._on_missed(reminder, action, choice)
            elif step == "propose":
                await self._on_propose(reminder, action, choice)
            else:
                log.warning("unknown follow-up step %r", step)
        except Exception:
            log.exception("follow-up failed for reminder %s", reminder.reminder_id)

    # ---- steps ----------------------------------------------------------

    async def _on_ack(self, reminder: Reminder, action: PendingAction, choice: str) -> None:
        if choice == "done":
            if reminder.rrule:
                await self._say(action, f"Отметил это срабатывание: {reminder.text}")
            else:
                await self._repos.reminders.complete(reminder.reminder_id, action.user_id)
                await self._say(action, f"Готово: {reminder.text}")
            return
        if choice == "not_done":
            await self._ask_missed(reminder, action, later=False)

    async def _on_missed(self, reminder: Reminder, action: PendingAction, choice: str) -> None:
        user_tz = policy.zone(reminder.timezone, self._default_timezone)
        now = datetime.now(timezone.utc)

        if choice == "snooze":
            due = policy.snooze_at(now, user_tz)
            await self._apply_time(reminder, action.user_id, due)
            await self._say(
                action,
                f"Напомню «{reminder.text}» {policy.format_when(due, user_tz)}.",
            )
            return
        if choice == "reschedule":
            proposal = policy.propose(
                reminder.text, now, user_tz, later=bool(action.arguments.get("later")),
            )
            when = policy.format_when(proposal.due_at, user_tz)
            await self._prompt(
                reminder,
                action.chat_id or 0,
                step="propose",
                occurrence=action.arguments["occurrence"],
                operation_id=f"followup:{reminder.reminder_id}:{action.arguments['occurrence']}:propose:{new_ulid()}",
                extra={"proposed_due": proposal.due_at.isoformat(), "importance": proposal.importance},
                text=(
                    f"Предлагаю перенести «{reminder.text}» на {when}.\n"
                    f"Важность: {policy.LABELS[proposal.importance]} ({proposal.reason})."
                ),
                actions=[_ACCEPT, _REFUSE, _DROP],
            )

    async def _on_propose(self, reminder: Reminder, action: PendingAction, choice: str) -> None:
        if choice == "accept":
            user_tz = policy.zone(reminder.timezone, self._default_timezone)
            due = parse_datetime(action.arguments["proposed_due"], user_tz)
            await self._apply_time(reminder, action.user_id, due)
            await self._say(
                action,
                f"Перенёс «{reminder.text}» на {policy.format_when(due, user_tz)}.",
            )
            return
        if choice == "reject":
            await self._ask_missed(reminder, action, later=True)

    async def _drop(self, reminder: Reminder, action: PendingAction) -> None:
        await self._repos.reminders.cancel(reminder.reminder_id, action.user_id)
        await self._say(action, f"Отменил: {reminder.text}")

    async def _ask_missed(self, reminder: Reminder, action: PendingAction, *, later: bool) -> None:
        occurrence = action.arguments["occurrence"]
        await self._prompt(
            reminder,
            action.chat_id or 0,
            step="missed",
            occurrence=occurrence,
            operation_id=f"followup:{reminder.reminder_id}:{occurrence}:missed:{new_ulid()}",
            extra={"later": later},
            text=f"«{reminder.text}» — не выполнено.\n\nНапомнить позже или переназначить?",
            actions=[_SNOOZE, _RESCHEDULE, _DROP],
        )

    # ---- persistence ----------------------------------------------------

    async def _apply_time(self, reminder: Reminder, user_id: str, due: datetime) -> None:
        if reminder.rrule:
            await self._repos.reminders.create(
                user_id=user_id,
                text=reminder.text,
                due_at=due,
                timezone_name=reminder.timezone,
                operation_id=f"followup-extra:{reminder.reminder_id}:{due.isoformat()}",
            )
        else:
            await self._repos.reminders.reschedule(reminder.reminder_id, user_id, due)
        self._wake()

    async def _prompt(
        self,
        reminder: Reminder,
        chat_id: int,
        *,
        step: str,
        occurrence: int,
        operation_id: str,
        text: str,
        actions: list[methods.ConfirmAction],
        delivery_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        arguments = {
            "reminder_id": reminder.reminder_id,
            "occurrence": occurrence,
            "step": step,
            **(extra or {}),
        }
        await self._confirmations.prompt(
            user_id=reminder.user_id,
            chat_id=chat_id,
            tool_name=TOOL,
            arguments=arguments,
            operation_id=operation_id,
            prompt_text=text,
            actions=actions,
            delivery_id=delivery_id,
            ttl_seconds=TTL_SECONDS,
        )

    async def _say(self, action: PendingAction, text: str) -> None:
        delivery_id = new_ulid()
        await self._link.send_event(
            methods.TELEGRAM_SEND,
            methods.dump(
                methods.TelegramSendParams(
                    delivery_id=delivery_id,
                    user_id=action.user_id,
                    chat_id=action.chat_id or 0,
                    text=text,
                    kind="notification",
                )
            ),
            delivery_id=delivery_id,
            user_id=action.user_id,
        )
