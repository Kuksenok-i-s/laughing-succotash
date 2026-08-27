"""Evening diary check-in and the month-end summary.

Collection must keep working when Cursor is down: the scheduler asks questions, buttons and
replies land in SQLite, and a later tick can still close the month. The narrative summary uses
the agent when it is there, and a structured local write when it is not.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pa_protocol import methods, new_ulid

from ..agent.base import AgentContext, AgentError, Provenance
from ..assistant import prompts
from ..storage.repositories import PendingAction
from ..storage.repositories.journal import JournalEntry
from . import questions as q

log = logging.getLogger(__name__)

TOOL = "journal.checkin"
TTL_SECONDS = 16 * 3600

_MONTH_GENITIVE = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
_MONTH_SHORT = (
    "янв", "фев", "мар", "апр", "мая", "июн",
    "июл", "авг", "сен", "окт", "ноя", "дек",
)


class JournalService:
    TOOL = TOOL

    def __init__(
        self,
        repos,
        confirmations,
        link,
        *,
        default_timezone: str = "UTC",
        hour: int = 21,
        minute: int = 0,
        summary_hour: int = 10,
        enabled: bool = True,
        backend=None,
        workspace: Path | None = None,
    ) -> None:
        self._repos = repos
        self._confirmations = confirmations
        self._link = link
        self._hour = hour
        self._minute = minute
        self._summary_hour = summary_hour
        self._enabled = enabled
        self._backend = backend
        self._workspace = workspace
        self._inflight: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def drain(self) -> None:
        """Wait for in-flight monthly summaries. Tests use this; the scheduler does not."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---- scheduler ----------------------------------------------------

    async def tick(self, now: datetime) -> None:
        await self.offer_due(now)
        await self.summarize_due(now)

    async def offer_due(self, now: datetime) -> None:
        if not self._enabled:
            return
        for user_id, chat_id in await self._repos.conversations.users_with_chat():
            user_tz = await self._repos.conversations.timezone_for(user_id)
            local = now.astimezone(user_tz)
            if (local.hour, local.minute) < (self._hour, self._minute):
                continue
            today = local.date().isoformat()
            await self._repos.journal.close_stale(user_id, today)
            existing = await self._repos.journal.get_by_date(user_id, today)
            if existing is not None:
                continue
            await self.begin(user_id, chat_id, now=now, mode="offer")

    async def summarize_due(self, now: datetime) -> None:
        if not self._enabled:
            return
        for user_id, chat_id in await self._repos.conversations.users_with_chat():
            user_tz = await self._repos.conversations.timezone_for(user_id)
            local = now.astimezone(user_tz)
            if local.day == 1 and (local.hour, local.minute) < (self._summary_hour, 0):
                continue
            period, start, end = previous_month(local)
            key = f"{user_id}:{period}"
            if key in self._inflight:
                continue
            filled = await self._repos.journal.list_range(
                user_id, start, end, statuses=("complete",)
            )
            if not filled:
                continue
            claimed = await self._repos.journal.begin_summary(user_id=user_id, period=period)
            if claimed is None:
                continue
            self._inflight.add(key)
            self._spawn(self._produce_summary(claimed, user_id, chat_id, period, start, end, now))

    # ---- user-facing --------------------------------------------------

    async def begin(
        self,
        user_id: str,
        chat_id: int,
        *,
        now: datetime | None = None,
        mode: str = "offer",
    ) -> JournalEntry:
        user_tz = await self._repos.conversations.timezone_for(user_id)
        local = (now or datetime.now(timezone.utc)).astimezone(user_tz)
        today = local.date().isoformat()
        await self._repos.journal.close_stale(user_id, today)
        entry, _ = await self._repos.journal.ensure(user_id=user_id, local_date=today)
        if entry.status == "complete" and mode != "refill":
            return entry
        if entry.status in {"skipped", "complete"} and mode in {"fill", "refill"}:
            updated = await self._repos.journal.reopen(entry.entry_id, user_id, step=q.WORK)
            if updated is not None:
                entry = updated
        if mode in {"fill", "refill"} and entry.step == q.OFFER:
            updated = await self._repos.journal.update(
                entry.entry_id, user_id, step=q.WORK,
            )
            if updated is not None:
                entry = updated
        await self._ask(entry, chat_id)
        return entry

    async def start_or_show(self, user_id: str, chat_id: int) -> None:
        user_tz = await self._repos.conversations.timezone_for(user_id)
        today = datetime.now(timezone.utc).astimezone(user_tz).date().isoformat()
        entry = await self._repos.journal.get_by_date(user_id, today)
        if entry is not None and entry.status == "complete":
            await self._say(user_id, chat_id, format_entry(entry))
            return
        await self.begin(user_id, chat_id, mode="fill")

    async def capture(self, user_id: str, text: str, *, chat_id: int | None) -> bool:
        """Consume a message if the user is on a free-text step. ``True`` means 'do not send to Cursor'."""
        entry = await self._repos.journal.open_for(user_id)
        if entry is None or entry.step not in q.TEXT_STEPS:
            return False
        if chat_id is None:
            chat_id = await self._repos.conversations.chat_for(user_id) or 0
        stripped = (text or "").strip()
        if not stripped:
            await self._ask(entry, chat_id)
            return True
        await self._store_and_advance(entry, chat_id, {entry.step: stripped})
        return True

    async def abandon(self, user_id: str, chat_id: int | None) -> bool:
        entry = await self._repos.journal.open_for(user_id)
        if entry is None:
            return False
        await self._repos.journal.update(
            entry.entry_id, user_id, status="skipped", step=q.DONE,
        )
        if chat_id:
            await self._say(user_id, chat_id, f"Пропустил дневник за {date_label(entry.local_date)}.")
        return True

    async def handle(self, action: PendingAction, choice: str) -> None:
        if choice in {"expired", "cancel"}:
            return
        args = action.arguments
        entry = await self._repos.journal.get(args.get("entry_id", ""), action.user_id)
        if entry is None or action.chat_id is None:
            return
        if args.get("step") != entry.step or entry.status != "open":
            return
        try:
            await self._on_choice(entry, action.chat_id, choice)
        except Exception:
            log.exception("journal check-in failed for %s", entry.entry_id)

    # ---- steps --------------------------------------------------------

    async def _on_choice(self, entry: JournalEntry, chat_id: int, choice: str) -> None:
        if entry.step == q.OFFER:
            if choice == "skip":
                await self._repos.journal.update(
                    entry.entry_id, entry.user_id, status="skipped", step=q.DONE,
                )
                await self._say(
                    entry.user_id, chat_id,
                    f"Пропустил дневник за {date_label(entry.local_date)}.",
                )
                return
            if choice == "fill":
                updated = await self._repos.journal.update(
                    entry.entry_id, entry.user_id, step=q.WORK,
                )
                if updated is not None:
                    await self._ask(updated, chat_id)
            return

        if choice == "skip" and entry.step in q.TEXT_STEPS:
            await self._store_and_advance(entry, chat_id, {entry.step: ""})
            return

        if entry.step in {q.MOOD, q.PROGRESS} and choice.isdigit():
            score = int(choice)
            if score not in range(1, 6):
                return
            field = "mood" if entry.step == q.MOOD else "progress"
            labels = q.MOOD_LABELS if field == "mood" else q.PROGRESS_LABELS
            await self._store_and_advance(
                entry, chat_id,
                {field: score, f"{field}_label": labels[score]},
            )

    async def _store_and_advance(
        self, entry: JournalEntry, chat_id: int, answers: dict[str, Any]
    ) -> None:
        following = q.next_step(entry.step)
        complete = following == q.DONE
        updated = await self._repos.journal.update(
            entry.entry_id, entry.user_id,
            answers=answers, step=following, complete=complete,
        )
        if updated is None:
            return
        if complete:
            await self._say(entry.user_id, chat_id, format_entry(updated))
            return
        await self._ask(updated, chat_id)

    async def _ask(self, entry: JournalEntry, chat_id: int) -> None:
        text = q.prompt_for(entry.step, date_label=date_label(entry.local_date))
        actions = q.actions_for(entry.step)
        if not actions:
            await self._say(entry.user_id, chat_id, text)
            return
        delivery_id = (
            f"journal:{entry.user_id}:{entry.local_date}:offer"
            if entry.step == q.OFFER
            else f"journal:{entry.entry_id}:{entry.step}:{new_ulid()}"
        )
        await self._confirmations.prompt(
            user_id=entry.user_id,
            chat_id=chat_id,
            tool_name=TOOL,
            arguments={"entry_id": entry.entry_id, "step": entry.step},
            operation_id=f"journal:{entry.entry_id}:{entry.step}:{new_ulid()}",
            prompt_text=text,
            actions=actions,
            delivery_id=delivery_id,
            ttl_seconds=TTL_SECONDS,
        )

    # ---- monthly summary ----------------------------------------------

    async def _produce_summary(
        self,
        claimed,
        user_id: str,
        chat_id: int,
        period: str,
        start: str,
        end: str,
        now: datetime,
    ) -> None:
        key = f"{user_id}:{period}"
        try:
            entries = await self._repos.journal.list_range(user_id, start, end)
            complete = [item for item in entries if item.status == "complete"]
            skipped = [item for item in entries if item.status == "skipped"]
            body = await self._write_summary(user_id, period, complete, now)
            stored = await self._repos.journal.finish_summary(
                claimed.summary_id,
                body=body,
                entry_count=len(complete),
                skipped_count=len(skipped),
            )
            text = stored.body if stored is not None else body
            await self._say(
                user_id, chat_id, text,
                delivery_id=f"journal-summary:{user_id}:{period}",
            )
        except Exception:
            log.exception("journal summary failed for %s %s", user_id, period)
            await self._repos.journal.finish_summary(
                claimed.summary_id, body="", entry_count=0, skipped_count=0, status="failed",
            )
        finally:
            self._inflight.discard(key)

    async def _write_summary(
        self, user_id: str, period: str, entries: list[JournalEntry], now: datetime
    ) -> str:
        fallback = format_month_fallback(period, entries)
        if self._backend is None or self._workspace is None:
            return fallback
        user_tz = await self._repos.conversations.timezone_for(user_id)
        context = AgentContext(
            user_id=user_id,
            conversation_id=f"journal:{user_id}:{period}",
            timezone=user_tz,
            now=now,
            provenance=Provenance.UNTRUSTED_CONTENT,
        )
        try:
            session_id = await self._backend.create_session(
                workspace=self._workspace, mcp_servers=[]
            )
            response = await self._backend.send_message(
                session_id,
                prompts.journal_month(period, dump_entries(entries), context),
                context,
            )
        except AgentError as exc:
            log.warning("journal summary agent failed: %s", exc)
            return fallback
        except Exception:
            log.exception("journal summary agent crashed")
            return fallback
        text = (response.text or "").strip()
        return text or fallback

    # ---- transport ----------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _say(
        self, user_id: str, chat_id: int, text: str, *, delivery_id: str | None = None
    ) -> None:
        delivery = delivery_id or new_ulid()
        await self._link.send_event(
            methods.TELEGRAM_SEND,
            methods.dump(
                methods.TelegramSendParams(
                    delivery_id=delivery,
                    user_id=user_id,
                    chat_id=chat_id,
                    text=text,
                    kind="notification",
                )
            ),
            delivery_id=delivery,
            user_id=user_id,
        )


def previous_month(local: datetime) -> tuple[str, str, str]:
    """``(YYYY-MM, start_date, end_date)`` of the month before ``local``."""
    first = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = first - timedelta(days=1)
    start = last.replace(day=1).date().isoformat()
    return last.strftime("%Y-%m"), start, last.date().isoformat()


def date_label(local_date: str) -> str:
    year, month, day = (int(part) for part in local_date.split("-"))
    return f"{day} {_MONTH_SHORT[month - 1]}"


def period_label(period: str) -> str:
    year, month = (int(part) for part in period.split("-"))
    return f"{_MONTH_GENITIVE[month - 1]} {year}"


def format_entry(entry: JournalEntry) -> str:
    answers = entry.answers
    lines = [f"*Дневник {date_label(entry.local_date)}*"]
    if answers.get("work"):
        lines.append(f"*Работа:* {_plain(answers['work'])}")
    if answers.get("personal"):
        lines.append(f"*Личное:* {_plain(answers['personal'])}")
    mood = answers.get("mood")
    progress = answers.get("progress")
    bits = []
    if mood is not None:
        label = answers.get("mood_label")
        bits.append(f"настроение {mood}/5" + (f" ({label})" if label else ""))
    if progress is not None:
        label = answers.get("progress_label")
        bits.append(f"прогресс {progress}/5" + (f" ({label})" if label else ""))
    if bits:
        lines.append(" · ".join(bits))
    if answers.get("tomorrow"):
        lines.append(f"*Завтра:* {_plain(answers['tomorrow'])}")
    if len(lines) == 1:
        lines.append("Пустая запись.")
    return "\n".join(lines)


def dump_entries(entries: list[JournalEntry]) -> str:
    blocks = []
    for item in entries:
        answers = item.answers
        lines = [f"## {item.local_date}"]
        if answers.get("work"):
            lines.append(f"работа: {answers['work']}")
        if answers.get("personal"):
            lines.append(f"личное: {answers['personal']}")
        if answers.get("mood") is not None:
            lines.append(
                f"настроение: {answers['mood']}/5 ({answers.get('mood_label') or '—'})"
            )
        if answers.get("progress") is not None:
            lines.append(
                f"прогресс: {answers['progress']}/5 ({answers.get('progress_label') or '—'})"
            )
        if answers.get("tomorrow"):
            lines.append(f"завтра: {answers['tomorrow']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(записей нет)"


def format_month_fallback(period: str, entries: list[JournalEntry]) -> str:
    moods = [int(item.answers["mood"]) for item in entries if item.answers.get("mood") is not None]
    progresses = [
        int(item.answers["progress"])
        for item in entries if item.answers.get("progress") is not None
    ]
    lines = [
        f"*Итог {period_label(period)}*",
        f"Заполнено дней: {len(entries)}.",
    ]
    if moods:
        lines.append(f"Настроение: среднее {sum(moods) / len(moods):.1f}/5.")
    if progresses:
        lines.append(f"Прогресс: среднее {sum(progresses) / len(progresses):.1f}/5.")
    lines.append("")
    for item in entries:
        snippet = _day_snippet(item)
        if snippet:
            lines.append(snippet)
    lines.append("")
    lines.append("Это сырые записи — модель была недоступна, поэтому без сквозного разбора.")
    return "\n".join(lines).strip()


def _day_snippet(entry: JournalEntry) -> str:
    answers = entry.answers
    parts = [f"*{date_label(entry.local_date)}*"]
    ratings = []
    if answers.get("mood") is not None:
        ratings.append(f"н{answers['mood']}")
    if answers.get("progress") is not None:
        ratings.append(f"п{answers['progress']}")
    if ratings:
        parts[0] += " · " + " ".join(ratings)
    if answers.get("work"):
        parts.append("работа: " + _clip(_plain(answers["work"]), 180))
    if answers.get("personal"):
        parts.append("личное: " + _clip(_plain(answers["personal"]), 180))
    if len(parts) == 1:
        return ""
    return "\n".join(parts)


def _plain(text: str) -> str:
    return (
        str(text).replace("*", "·").replace("_", " ").replace("`", "'").strip()
    )


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
