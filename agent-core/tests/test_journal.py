"""Evening diary check-in: collection without Cursor, then a month-end summary."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from agent_core.assistant.confirmations import ConfirmationService
from agent_core.assistant.service import AssistantService
from agent_core.journal import JournalService, previous_month
from agent_core.journal.questions import PERSONAL
from agent_core.scheduler.service import Scheduler

MOSCOW = ZoneInfo("Europe/Moscow")
EVENING = datetime(2026, 8, 27, 18, 0, tzinfo=timezone.utc)  # 21:00 Moscow
BEFORE = datetime(2026, 8, 27, 17, 0, tzinfo=timezone.utc)  # 20:00 Moscow
MONTH_END = datetime(2026, 9, 1, 7, 0, tzinfo=timezone.utc)  # 10:00 Moscow on the 1st


@pytest.fixture
def journal(repos, gateway, settings):
    confirmations = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    service = JournalService(
        repos, confirmations, gateway,
        default_timezone=settings.default_timezone,
        hour=21, summary_hour=10,
    )
    confirmations.register_handler(JournalService.TOOL, service.handle)
    return service, confirmations


async def _user(repos, chat_id: int = 500) -> None:
    await repos.conversations.ensure_user("tg:1")
    await repos.conversations.remember_chat("tg:1", chat_id)


def _press(gateway) -> str:
    return gateway.confirms()[-1]["action_id"]


async def test_nothing_is_asked_before_evening(journal, repos, gateway) -> None:
    service, _ = journal
    await _user(repos)

    await service.tick(BEFORE)

    assert gateway.confirms() == []
    assert await repos.journal.get_by_date("tg:1", "2026-08-27") is None


async def test_evening_prompt_is_offered_once(journal, repos, gateway) -> None:
    service, _ = journal
    await _user(repos)

    await service.tick(EVENING)
    await service.tick(EVENING)

    assert len(gateway.confirms()) == 1
    prompt = gateway.confirms()[0]
    assert "Дневник за 27 авг" in prompt["text"]
    assert [action["id"] for action in prompt["actions"]] == ["fill", "skip"]
    stored = await repos.journal.get_by_date("tg:1", "2026-08-27")
    assert stored is not None
    assert stored.status == "open"


async def test_no_chat_means_no_prompt(journal, repos, gateway) -> None:
    service, _ = journal
    await repos.conversations.ensure_user("tg:1")

    await service.tick(EVENING)

    assert gateway.confirms() == []


async def test_skip_closes_the_day(journal, repos, gateway) -> None:
    service, confirmations = journal
    await _user(repos)
    await service.tick(EVENING)

    await confirmations.resolve(_press(gateway), "tg:1", "skip")

    stored = await repos.journal.get_by_date("tg:1", "2026-08-27")
    assert stored is not None
    assert stored.status == "skipped"
    assert any("Пропустил" in text for text in gateway.texts())


async def test_a_full_checkin_stores_work_and_personal(journal, repos, gateway) -> None:
    service, confirmations = journal
    await _user(repos)
    await service.begin("tg:1", 500, now=EVENING, mode="fill")

    assert "Работа" in gateway.confirms()[-1]["text"]
    assert await service.capture("tg:1", "закрыл релиз", chat_id=500)

    assert "Личное" in gateway.confirms()[-1]["text"]
    assert await service.capture("tg:1", "прогулка с собакой", chat_id=500)

    await confirmations.resolve(_press(gateway), "tg:1", "4")
    await confirmations.resolve(_press(gateway), "tg:1", "5")
    assert await service.capture("tg:1", "утром спорт", chat_id=500)

    stored = await repos.journal.get_by_date("tg:1", "2026-08-27")
    assert stored is not None
    assert stored.status == "complete"
    assert stored.answers["work"] == "закрыл релиз"
    assert stored.answers["personal"] == "прогулка с собакой"
    assert stored.answers["mood"] == 4
    assert stored.answers["progress"] == 5
    assert stored.answers["tomorrow"] == "утром спорт"
    recap = gateway.texts()[-1]
    assert "закрыл релиз" in recap
    assert "прогулка с собакой" in recap


async def test_typed_messages_are_not_eaten_outside_a_checkin(journal, repos) -> None:
    service, _ = journal
    await _user(repos)

    assert await service.capture("tg:1", "что у меня завтра?", chat_id=500) is False


async def test_stale_button_after_advance_is_ignored(journal, repos, gateway) -> None:
    service, confirmations = journal
    await _user(repos)
    await service.begin("tg:1", 500, now=EVENING, mode="fill")
    stale = _press(gateway)
    await service.capture("tg:1", "сделал фичу", chat_id=500)

    await confirmations.resolve(stale, "tg:1", "skip")

    stored = await repos.journal.get_by_date("tg:1", "2026-08-27")
    assert stored is not None
    assert stored.step == PERSONAL
    assert stored.answers["work"] == "сделал фичу"


async def test_unfinished_day_is_kept_when_the_next_evening_starts(
    journal, repos, gateway
) -> None:
    service, _ = journal
    await _user(repos)
    await service.begin("tg:1", 500, now=EVENING, mode="fill")
    await service.capture("tg:1", "черновик работы", chat_id=500)

    next_evening = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    await service.tick(next_evening)

    yesterday = await repos.journal.get_by_date("tg:1", "2026-08-27")
    assert yesterday is not None
    assert yesterday.status == "complete"
    assert yesterday.answers["work"] == "черновик работы"
    today = await repos.journal.get_by_date("tg:1", "2026-08-28")
    assert today is not None
    assert today.status == "open"


async def test_month_end_summary_uses_the_agent(
    journal, repos, gateway, backend, tmp_path
) -> None:
    service, _ = journal
    service._backend = backend
    service._workspace_for = lambda _uid: tmp_path / "workspace"
    backend.reply = "*Итог августа 2026*\n\n*Работа*\n- релиз вышел."
    await _user(repos)

    for day in ("2026-08-03", "2026-08-10", "2026-08-20"):
        entry, _ = await repos.journal.ensure(user_id="tg:1", local_date=day, step="done")
        await repos.journal.update(
            entry.entry_id, "tg:1",
            answers={"work": f"день {day}", "mood": 4, "progress": 3},
            complete=True,
        )

    await service.tick(MONTH_END)
    await service.drain()

    assert any("Итог августа" in text for text in gateway.texts())
    stored = await repos.journal.get_summary("tg:1", "2026-08")
    assert stored is not None
    assert stored.status == "ready"
    assert stored.entry_count == 3
    assert backend.prompts
    assert "<journal" in backend.prompts[0][1]


async def test_month_end_falls_back_when_the_agent_is_down(
    journal, repos, gateway
) -> None:
    service, _ = journal
    await _user(repos)
    entry, _ = await repos.journal.ensure(user_id="tg:1", local_date="2026-08-05", step="done")
    await repos.journal.update(
        entry.entry_id, "tg:1",
        answers={"work": "починил бота", "personal": "спал"},
        complete=True,
    )

    await service.tick(MONTH_END)
    await service.drain()

    text = gateway.texts()[-1]
    assert "Итог августа" in text
    assert "починил бота" in text
    assert await repos.journal.get_summary("tg:1", "2026-08") is not None


async def test_month_end_does_not_fire_twice(journal, repos, gateway) -> None:
    service, _ = journal
    await _user(repos)
    entry, _ = await repos.journal.ensure(user_id="tg:1", local_date="2026-08-05", step="done")
    await repos.journal.update(
        entry.entry_id, "tg:1", answers={"work": "раз"}, complete=True,
    )

    await service.tick(MONTH_END)
    await service.drain()
    first = list(gateway.texts())

    await service.tick(MONTH_END)
    await service.drain()

    assert gateway.texts() == first


async def test_previous_month_window() -> None:
    period, start, end = previous_month(datetime(2026, 9, 1, 10, 0, tzinfo=MOSCOW))
    assert period == "2026-08"
    assert start == "2026-08-01"
    assert end == "2026-08-31"


async def test_disabled_journal_does_not_prompt(repos, gateway, settings) -> None:
    confirmations = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    service = JournalService(
        repos, confirmations, gateway,
        default_timezone=settings.default_timezone, enabled=False,
    )
    await _user(repos)

    await service.tick(EVENING)

    assert gateway.confirms() == []


async def test_scheduler_drives_the_evening_prompt(
    settings, repos, gateway, backend
) -> None:
    await _user(repos)
    confirmations = ConfirmationService(repos.pending_actions, gateway, timeout_seconds=60)
    journal = JournalService(
        repos, confirmations, gateway, default_timezone=settings.default_timezone,
    )
    confirmations.register_handler(JournalService.TOOL, journal.handle)
    assistant = AssistantService(settings, repos, gateway, None, None, backend)
    scheduler = Scheduler(
        repos, assistant, confirmations=confirmations, journal=journal,
        default_timezone=settings.default_timezone,
    )

    await scheduler.tick(BEFORE)
    assert gateway.confirms() == []

    await scheduler.tick(EVENING)
    assert len(gateway.confirms()) == 1
    assert "Дневник" in gateway.confirms()[0]["text"]
