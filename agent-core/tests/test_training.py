"""Trainer journal: per-athlete SQLite memory, structured logs, CSV export."""

from __future__ import annotations

from pathlib import Path

from agent_core.training.skill import SKILL_MARKDOWN, seed_into


async def test_an_athlete_belongs_only_to_their_trainer(repos) -> None:
    vasya, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="a-1"
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=vasya["athlete_id"],
        local_date="2026-08-30",
        exercises=[{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 100}]}],
        operation_id="log-1",
    )

    assert await repos.training.get_athlete(vasya["athlete_id"], "tg:2") is None
    assert await repos.training.list_athletes("tg:2") == []
    assert await repos.training.list_logs("tg:2") == []
    assert await repos.training.is_enabled("tg:1") is True
    assert await repos.training.is_enabled("tg:2") is False


async def test_each_athlete_has_their_own_program_and_log(repos) -> None:
    vasya, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="a-v"
    )
    masha, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Маша", aliases=["Машка"], operation_id="a-m"
    )

    await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=vasya["athlete_id"],
        title="Сила 3 дня",
        weekly_plan=[{"weekday": "пн", "title": "Ноги", "exercises": ["Присед"]}],
        operation_id="p-v",
    )
    await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=masha["athlete_id"],
        title="Гипертрофия",
        operation_id="p-m",
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=vasya["athlete_id"],
        local_date="2026-08-30",
        exercises=[{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 120}]}],
        operation_id="l-v",
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=masha["athlete_id"],
        local_date="2026-08-30",
        exercises=[{"name": "Жим", "sets": [{"reps": 8, "weight_kg": 40}]}],
        operation_id="l-m",
    )

    vasya_logs = await repos.training.list_logs("tg:1", athlete_id=vasya["athlete_id"])
    masha_logs = await repos.training.list_logs("tg:1", athlete_id=masha["athlete_id"])
    assert vasya_logs[0]["exercises"][0]["name"] == "Присед"
    assert masha_logs[0]["exercises"][0]["name"] == "Жим"
    assert [item["title"] for item in await repos.training.list_programs(
        "tg:1", athlete_id=vasya["athlete_id"]
    )] == ["Сила 3 дня"]
    found = await repos.training.search_athletes("tg:1", "машка")
    assert [item["display_name"] for item in found] == ["Маша"]


async def test_a_new_active_program_archives_the_previous_one(repos) -> None:
    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="я", is_self=True, operation_id="self"
    )
    first, _ = await repos.training.upsert_program(
        user_id="tg:1", athlete_id=athlete["athlete_id"], title="A", operation_id="p1"
    )
    second, _ = await repos.training.upsert_program(
        user_id="tg:1", athlete_id=athlete["athlete_id"], title="B", operation_id="p2"
    )

    assert first["program_id"] != second["program_id"]
    stored = await repos.training.get_program(first["program_id"], "tg:1")
    assert stored is not None and stored["status"] == "archived"
    active = await repos.training.list_programs("tg:1", athlete_id=athlete["athlete_id"])
    assert [item["title"] for item in active] == ["B"]


async def test_the_same_training_operation_twice_creates_one_log(repos) -> None:
    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="a"
    )
    args = dict(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-08-30",
        exercises=[{"name": "Жим", "sets": [{"reps": 5, "weight_kg": 80}]}],
        operation_id="same-log",
    )
    first, dup1 = await repos.training.save_log(**args)
    second, dup2 = await repos.training.save_log(**args)

    assert dup1 is False and dup2 is True
    assert first["log_id"] == second["log_id"]
    assert len(await repos.training.list_logs("tg:1")) == 1


async def test_logging_a_workout_marks_the_planned_session_done(repos) -> None:
    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="a"
    )
    session, _ = await repos.training.upsert_session(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-08-31",
        title="Ноги",
        plan=[{"name": "Присед", "sets": "4x8", "target_weight_kg": 100}],
        operation_id="s1",
    )
    log, _ = await repos.training.save_log(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-08-31",
        exercises=[{"name": "Присед", "sets": [{"reps": 8, "weight_kg": 100}]}],
        operation_id="l1",
    )

    assert log["session_id"] == session["session_id"]
    stored = await repos.training.get_session(session["session_id"], "tg:1")
    assert stored is not None and stored["status"] == "done"
    assert stored["weekday"] == "пн"


async def test_workout_csv_has_one_row_per_set(repos) -> None:
    from agent_core.storage.repositories.training import logs_to_csv

    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="a"
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-08-30",
        title="Жим",
        exercises=[{
            "name": "Жим лёжа",
            "sets": [
                {"reps": 5, "weight_kg": 80, "rpe": 7},
                {"reps": 5, "weight_kg": 85},
            ],
        }],
        operation_id="l1",
    )
    csv_text = logs_to_csv(await repos.training.list_logs("tg:1"))
    assert "Жим лёжа" in csv_text
    assert csv_text.count("\n") == 3  # header + two sets
    assert "80" in csv_text and "85" in csv_text


async def test_adding_another_person_switches_to_trainer_mode(repos) -> None:
    self_row, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="я", is_self=True, operation_id="self"
    )
    assert (await repos.training.get_profile("tg:1"))["mode"] == "self"

    await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="vasya"
    )
    profile = await repos.training.get_profile("tg:1")
    assert profile is not None and profile["mode"] == "trainer"
    assert self_row["is_self"] is True


async def test_set_mode_does_not_drop_existing_athletes(repos) -> None:
    await repos.training.upsert_athlete(user_id="tg:1", display_name="Вася", operation_id="v")
    await repos.training.set_mode("tg:1", "self")
    assert (await repos.training.get_profile("tg:1"))["mode"] == "self"
    assert [item["display_name"] for item in await repos.training.list_athletes("tg:1")] == ["Вася"]


async def test_ambiguous_athlete_search_returns_everyone(repos) -> None:
    await repos.training.upsert_athlete(user_id="tg:1", display_name="Саша Иванов", operation_id="1")
    await repos.training.upsert_athlete(user_id="tg:1", display_name="Саша Петров", operation_id="2")

    found = await repos.training.search_athletes("tg:1", "саша")
    assert len(found) == 2


async def test_programme_length_is_weeks_times_days(repos) -> None:
    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="я", is_self=True, operation_id="self"
    )
    program, _ = await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        title="Сила",
        days_per_week=3,
        weeks=4,
        started_on="2026-08-01",
        operation_id="p-len",
    )

    assert program["total_sessions"] == 12
    assert program["weeks"] == 4
    assert program["progress"]["done"] == 0
    assert program["progress"]["remaining"] == 12
    assert program["progress"]["label"] == "проведено 0 из 12, осталось 12"


async def test_each_log_counts_toward_remaining_workouts(repos) -> None:
    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="я", is_self=True, operation_id="self"
    )
    await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        title="12 тренировок",
        total_sessions=12,
        started_on="2026-08-01",
        operation_id="p-12",
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-08-30",
        exercises=[{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 80}]}],
        operation_id="l-1",
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-08-31",
        exercises=[{"name": "Жим", "sets": [{"reps": 5, "weight_kg": 60}]}],
        operation_id="l-2",
    )

    progress = await repos.training.progress(user_id="tg:1", athlete_id=athlete["athlete_id"])
    assert progress["done"] == 2
    assert progress["total"] == 12
    assert progress["remaining"] == 10
    assert progress["label"] == "проведено 2 из 12, осталось 10"


async def test_trainer_progress_is_per_athlete(repos) -> None:
    vasya, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Вася", operation_id="a-v"
    )
    masha, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="Маша", operation_id="a-m"
    )
    await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=vasya["athlete_id"],
        title="Вася сила",
        total_sessions=8,
        started_on="2026-08-01",
        operation_id="p-v",
    )
    await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=masha["athlete_id"],
        title="Маша гипертрофия",
        total_sessions=10,
        started_on="2026-08-01",
        operation_id="p-m",
    )
    await repos.training.save_log(
        user_id="tg:1",
        athlete_id=vasya["athlete_id"],
        local_date="2026-08-30",
        exercises=[{"name": "Присед", "sets": [{"reps": 5, "weight_kg": 100}]}],
        operation_id="l-v",
    )

    by_name = {
        item["athlete_name"]: item
        for item in await repos.training.progress_all("tg:1")
    }
    assert by_name["Вася"]["done"] == 1
    assert by_name["Вася"]["remaining"] == 7
    assert by_name["Маша"]["done"] == 0
    assert by_name["Маша"]["remaining"] == 10


async def test_without_a_length_remaining_is_planned_sessions(repos) -> None:
    athlete, _ = await repos.training.upsert_athlete(
        user_id="tg:1", display_name="я", is_self=True, operation_id="self"
    )
    program, _ = await repos.training.upsert_program(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        title="Без длины",
        started_on="2026-08-01",
        operation_id="p-open",
    )
    await repos.training.upsert_session(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-09-01",
        title="A",
        operation_id="s-a",
    )
    await repos.training.upsert_session(
        user_id="tg:1",
        athlete_id=athlete["athlete_id"],
        local_date="2026-09-03",
        title="B",
        operation_id="s-b",
    )

    progress = await repos.training.progress(user_id="tg:1", athlete_id=athlete["athlete_id"])
    assert progress["done"] == 0
    assert progress["remaining"] == 2
    assert progress["total"] is None
    assert progress["label"] == "проведено 0, в расписании ещё 2"

    session = await repos.training.get_session(
        (await repos.training.list_sessions("tg:1"))[0]["session_id"], "tg:1"
    )
    assert session is not None and session["program_id"] == program["program_id"]


def test_progress_label_covers_the_common_shapes() -> None:
    from agent_core.storage.repositories.training import format_progress

    assert format_progress({"done": 3, "remaining": 9, "total": 12}) == (
        "проведено 3 из 12, осталось 9"
    )
    assert format_progress({"done": 1, "remaining": 2, "planned": 2}) == (
        "проведено 1, в расписании ещё 2"
    )
    assert format_progress({"done": 4}) == "проведено 4"


def test_the_packaged_skill_matches_the_project_skill() -> None:
    root = Path(__file__).resolve().parents[2]
    project = (root / ".cursor/skills/trainer-journal/SKILL.md").read_text(encoding="utf-8")
    assert project == SKILL_MARKDOWN


def test_the_skill_is_seeded_into_the_user_workspace(tmp_path: Path) -> None:
    path = seed_into(tmp_path)
    assert path.read_text(encoding="utf-8") == SKILL_MARKDOWN
    seed_into(tmp_path)
    assert list(path.parent.glob("SKILL.md")) == [path]
