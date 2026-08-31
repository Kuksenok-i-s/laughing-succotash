"""Trainer journal: athletes, programmes, schedule and structured workout logs.

This is long-term memory for a Telegram user who coaches (themselves or a group). Each athlete
belongs to one trainer ``user_id``; programmes and logs are per athlete, never mixed. Writes are
idempotent on ``operation_id`` like the rest of the assistant objects.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime
from typing import Any

from pa_protocol import new_ulid

from ..database import Database, utcnow

_WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def _json_list(value: Any) -> str:
    return json.dumps(list(value or []), ensure_ascii=False)


def _load_list(value: str | None) -> list[Any]:
    if not value:
        return []
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else []


def _local_date(value: str) -> str:
    text = (value or "").strip()
    datetime.strptime(text, "%Y-%m-%d")
    return text


def weekday_of(local_date: str) -> str:
    return _WEEKDAYS[datetime.strptime(local_date, "%Y-%m-%d").weekday()]


def normalize_exercises(raw: Any) -> list[dict[str, Any]]:
    """Coerce model-produced exercise blobs into a stable list of {name, sets, notes}."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("exercises must be a list")
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append({"name": name, "sets": [], "notes": None})
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("exercise") or "").strip()
        if not name:
            continue
        sets: list[dict[str, Any]] = []
        for entry in item.get("sets") or []:
            if not isinstance(entry, dict):
                continue
            weight = entry.get("weight_kg", entry.get("weight"))
            sets.append(
                {
                    "reps": _number(entry.get("reps")),
                    "weight_kg": _number(weight),
                    "rpe": _number(entry.get("rpe")),
                    "note": _text(entry.get("note") or entry.get("notes")),
                }
            )
        out.append(
            {
                "name": name,
                "sets": sets,
                "notes": _text(item.get("notes") or item.get("note")),
            }
        )
    return out


def normalize_plan(raw: Any) -> list[dict[str, Any]]:
    """Weekly or session plan: same shape as exercises, plus optional target fields."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("plan must be a list")
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append({"name": name, "sets": None, "target_weight_kg": None, "notes": None})
            continue
        if not isinstance(item, dict):
            continue
        weekday = item.get("weekday")
        if weekday and item.get("title") and "name" not in item:
            out.append(
                {
                    "weekday": str(weekday).strip(),
                    "title": str(item.get("title") or "").strip(),
                    "exercises": normalize_exercises(item.get("exercises") or item.get("plan")),
                    "notes": _text(item.get("notes")),
                }
            )
            continue
        name = str(item.get("name") or item.get("exercise") or item.get("title") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "sets": item.get("sets"),
                "target_weight_kg": _number(item.get("target_weight_kg") or item.get("weight_kg")),
                "notes": _text(item.get("notes")),
            }
        )
    return out


def logs_to_csv(logs: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["date", "athlete", "title", "exercise", "set", "reps", "weight_kg", "rpe", "notes"]
    )
    for log in logs:
        athlete = log.get("athlete_name") or ""
        title = log.get("title") or ""
        date = log.get("local_date") or ""
        exercises = log.get("exercises") or []
        if not exercises:
            writer.writerow([date, athlete, title, "", "", "", "", "", log.get("notes") or ""])
            continue
        for exercise in exercises:
            name = exercise.get("name") or ""
            sets = exercise.get("sets") or []
            extra = exercise.get("notes") or log.get("notes") or ""
            if not sets:
                writer.writerow([date, athlete, title, name, "", "", "", "", extra])
                continue
            for index, entry in enumerate(sets, start=1):
                writer.writerow(
                    [
                        date,
                        athlete,
                        title,
                        name,
                        index,
                        _csv(entry.get("reps")),
                        _csv(entry.get("weight_kg")),
                        _csv(entry.get("rpe")),
                        extra if index == 1 else "",
                    ]
                )
    return buf.getvalue()


def schedule_to_csv(sessions: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "weekday", "athlete", "title", "status", "plan", "notes"])
    for item in sessions:
        plan = item.get("plan") or []
        names = []
        for block in plan:
            names.append(block.get("name") or block.get("title") or "")
        writer.writerow(
            [
                item.get("local_date") or "",
                item.get("weekday") or "",
                item.get("athlete_name") or "",
                item.get("title") or "",
                item.get("status") or "",
                "; ".join(part for part in names if part),
                item.get("notes") or "",
            ]
        )
    return buf.getvalue()


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _csv(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_mode(mode: str) -> str:
    value = (mode or "").strip().lower()
    if value not in {"self", "trainer"}:
        raise ValueError("mode must be self or trainer")
    return value


def _public_profile(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "enabled": bool(row["enabled"]),
        "mode": row["mode"] or "self",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _public_athlete(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "athlete_id": row["athlete_id"],
        "display_name": row["display_name"],
        "aliases": _load_list(row["aliases"]),
        "note": row["note"],
        "is_self": bool(row["is_self"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _public_program(row: sqlite3.Row) -> dict[str, Any]:
    total = row["total_sessions"] if "total_sessions" in row.keys() else None
    weeks = row["weeks"] if "weeks" in row.keys() else None
    return {
        "program_id": row["program_id"],
        "athlete_id": row["athlete_id"],
        "title": row["title"],
        "goal": row["goal"],
        "days_per_week": row["days_per_week"],
        "weeks": weeks,
        "total_sessions": total,
        "weekly_plan": _load_list(row["weekly_plan"]),
        "notes": row["notes"],
        "status": row["status"],
        "started_on": row["started_on"],
        "ended_on": row["ended_on"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def resolve_total_sessions(
    *,
    total_sessions: int | None,
    weeks: int | None,
    days_per_week: int | None,
    current: sqlite3.Row | None = None,
) -> tuple[int | None, int | None]:
    """``(total_sessions, weeks)`` after filling total from weeks × days when needed."""
    stored_weeks = weeks
    stored_total = total_sessions
    if current is not None:
        if stored_weeks is None:
            stored_weeks = current["weeks"] if "weeks" in current.keys() else None
        if stored_total is None:
            stored_total = (
                current["total_sessions"] if "total_sessions" in current.keys() else None
            )
        if days_per_week is None:
            days_per_week = current["days_per_week"]
    if stored_total is None and stored_weeks and days_per_week:
        stored_total = int(stored_weeks) * int(days_per_week)
    return stored_total, stored_weeks


def format_progress(progress: dict[str, Any]) -> str:
    done = int(progress.get("done") or 0)
    remaining = progress.get("remaining")
    total = progress.get("total")
    if total is not None:
        return f"проведено {done} из {total}, осталось {remaining}"
    planned = int(progress.get("planned") or 0)
    if planned:
        return f"проведено {done}, в расписании ещё {planned}"
    return f"проведено {done}"


def _public_session(row: sqlite3.Row, *, athlete_name: str | None = None) -> dict[str, Any]:
    local_date = row["local_date"]
    payload = {
        "session_id": row["session_id"],
        "athlete_id": row["athlete_id"],
        "program_id": row["program_id"],
        "local_date": local_date,
        "weekday": weekday_of(local_date),
        "title": row["title"],
        "plan": _load_list(row["plan"]),
        "notes": row["notes"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if athlete_name is not None:
        payload["athlete_name"] = athlete_name
    return payload


def _public_log(row: sqlite3.Row, *, athlete_name: str | None = None) -> dict[str, Any]:
    payload = {
        "log_id": row["log_id"],
        "athlete_id": row["athlete_id"],
        "session_id": row["session_id"],
        "local_date": row["local_date"],
        "title": row["title"],
        "raw_text": row["raw_text"],
        "exercises": _load_list(row["exercises"]),
        "notes": row["notes"],
        "duration_minutes": row["duration_minutes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if athlete_name is not None:
        payload["athlete_name"] = athlete_name
    return payload


class TrainingRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- profile / long-term flag ----------------------------------------

    async def touch_profile(self, user_id: str, *, mode: str | None = None) -> dict[str, Any]:
        chosen = _validate_mode(mode) if mode is not None else None
        now = utcnow().isoformat()

        def run(connection: sqlite3.Connection) -> sqlite3.Row:
            current = connection.execute(
                "SELECT * FROM training_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            if current is None:
                stored = chosen or "self"
                connection.execute(
                    "INSERT INTO training_profiles(user_id, enabled, mode, created_at, updated_at) "
                    "VALUES (?, 1, ?, ?, ?)",
                    (user_id, stored, now, now),
                )
            else:
                stored = chosen if chosen is not None else current["mode"] or "self"
                connection.execute(
                    "UPDATE training_profiles SET enabled = 1, mode = ?, updated_at = ? "
                    "WHERE user_id = ?",
                    (stored, now, user_id),
                )
            return connection.execute(
                "SELECT * FROM training_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()

        return _public_profile(await self._db.transaction(run))

    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM training_profiles WHERE user_id = ?", (user_id,)
        )
        return _public_profile(row) if row else None

    async def set_mode(self, user_id: str, mode: str) -> dict[str, Any]:
        return await self.touch_profile(user_id, mode=mode)

    async def is_enabled(self, user_id: str) -> bool:
        profile = await self.get_profile(user_id)
        return bool(profile and profile["enabled"])

    # ---- athletes --------------------------------------------------------

    async def upsert_athlete(
        self,
        *,
        user_id: str,
        display_name: str,
        aliases: list[str] | None = None,
        note: str | None = None,
        is_self: bool = False,
        athlete_id: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        name = display_name.strip()
        if not name:
            raise ValueError("display_name is required")
        now = utcnow().isoformat()
        new_id = new_ulid()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            if operation_id:
                existing = connection.execute(
                    "SELECT * FROM training_athletes WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    return existing, True
            if athlete_id:
                current = connection.execute(
                    "SELECT * FROM training_athletes WHERE athlete_id = ? AND user_id = ?",
                    (athlete_id, user_id),
                ).fetchone()
                if current is None:
                    raise LookupError("athlete not found")
                aliases_json = (
                    _json_list(aliases) if aliases is not None else current["aliases"]
                )
                note_value = current["note"] if note is None else note
                is_self_value = int(is_self) if is_self else current["is_self"]
                connection.execute(
                    "UPDATE training_athletes SET display_name = ?, aliases = ?, note = ?, "
                    "is_self = ?, updated_at = ? WHERE athlete_id = ? AND user_id = ?",
                    (name, aliases_json, note_value, is_self_value, now, athlete_id, user_id),
                )
                row = connection.execute(
                    "SELECT * FROM training_athletes WHERE athlete_id = ?", (athlete_id,)
                ).fetchone()
                return row, False
            connection.execute(
                "INSERT INTO training_athletes(athlete_id, user_id, display_name, aliases, note, "
                "is_self, status, operation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                (
                    new_id, user_id, name, _json_list(aliases), note,
                    int(is_self), operation_id, now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_athletes WHERE athlete_id = ?", (new_id,)
            ).fetchone()
            return row, False

        row, duplicate = await self._db.transaction(run)
        await self.touch_profile(
            user_id, mode="trainer" if not is_self and not athlete_id else None,
        )
        return _public_athlete(row), duplicate

    async def get_athlete(self, athlete_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM training_athletes WHERE athlete_id = ? AND user_id = ?",
            (athlete_id, user_id),
        )
        return _public_athlete(row) if row else None

    async def list_athletes(
        self, user_id: str, *, status: str | None = "active", limit: int = 100
    ) -> list[dict[str, Any]]:
        if status and status != "all":
            rows = await self._db.fetch_all(
                "SELECT * FROM training_athletes WHERE user_id = ? AND status = ? "
                "ORDER BY is_self DESC, display_name LIMIT ?",
                (user_id, status, limit),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM training_athletes WHERE user_id = ? "
                "ORDER BY is_self DESC, display_name LIMIT ?",
                (user_id, limit),
            )
        return [_public_athlete(row) for row in rows]

    async def search_athletes(
        self, user_id: str, query: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        needle = query.strip().casefold()
        rows = await self._db.fetch_all(
            "SELECT * FROM training_athletes WHERE user_id = ? AND status = 'active' "
            "ORDER BY display_name",
            (user_id,),
        )
        matches = [
            row
            for row in rows
            if not needle
            or needle in row["display_name"].casefold()
            or needle in (row["aliases"] or "").casefold()
        ]
        return [_public_athlete(row) for row in matches[:limit]]

    async def archive_athlete(self, athlete_id: str, user_id: str) -> bool:
        changed = await self._db.execute(
            "UPDATE training_athletes SET status = 'archived', updated_at = ? "
            "WHERE athlete_id = ? AND user_id = ? AND status = 'active'",
            (utcnow().isoformat(), athlete_id, user_id),
        )
        return changed > 0

    # ---- programmes ------------------------------------------------------

    async def upsert_program(
        self,
        *,
        user_id: str,
        athlete_id: str,
        title: str,
        goal: str | None = None,
        days_per_week: int | None = None,
        weeks: int | None = None,
        total_sessions: int | None = None,
        weekly_plan: Any = None,
        notes: str | None = None,
        started_on: str | None = None,
        status: str = "active",
        program_id: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if await self.get_athlete(athlete_id, user_id) is None:
            raise LookupError("athlete not found")
        title_text = title.strip()
        if not title_text:
            raise ValueError("title is required")
        plan = normalize_plan(weekly_plan)
        start = _local_date(started_on) if started_on else None
        if status not in {"active", "completed", "archived"}:
            raise ValueError("status must be active, completed or archived")
        if weeks is not None and weeks < 1:
            raise ValueError("weeks must be >= 1")
        if total_sessions is not None and total_sessions < 1:
            raise ValueError("total_sessions must be >= 1")
        now = utcnow().isoformat()
        new_id = new_ulid()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            if operation_id:
                existing = connection.execute(
                    "SELECT * FROM training_programs WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    return existing, True
            if program_id:
                current = connection.execute(
                    "SELECT * FROM training_programs WHERE program_id = ? AND user_id = ?",
                    (program_id, user_id),
                ).fetchone()
                if current is None:
                    raise LookupError("program not found")
                stored_total, stored_weeks = resolve_total_sessions(
                    total_sessions=total_sessions,
                    weeks=weeks,
                    days_per_week=days_per_week,
                    current=current,
                )
                connection.execute(
                    "UPDATE training_programs SET title = ?, goal = ?, days_per_week = ?, "
                    "weeks = ?, total_sessions = ?, weekly_plan = ?, notes = ?, status = ?, "
                    "started_on = ?, updated_at = ? WHERE program_id = ? AND user_id = ?",
                    (
                        title_text,
                        goal if goal is not None else current["goal"],
                        days_per_week if days_per_week is not None else current["days_per_week"],
                        stored_weeks,
                        stored_total,
                        _json_list(plan) if weekly_plan is not None else current["weekly_plan"],
                        notes if notes is not None else current["notes"],
                        status,
                        start if started_on is not None else current["started_on"],
                        now, program_id, user_id,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM training_programs WHERE program_id = ?", (program_id,)
                ).fetchone()
                return row, False
            stored_total, stored_weeks = resolve_total_sessions(
                total_sessions=total_sessions,
                weeks=weeks,
                days_per_week=days_per_week,
            )
            if status == "active":
                connection.execute(
                    "UPDATE training_programs SET status = 'archived', updated_at = ? "
                    "WHERE user_id = ? AND athlete_id = ? AND status = 'active'",
                    (now, user_id, athlete_id),
                )
            connection.execute(
                "INSERT INTO training_programs(program_id, user_id, athlete_id, title, goal, "
                "days_per_week, weeks, total_sessions, weekly_plan, notes, status, started_on, "
                "operation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id, user_id, athlete_id, title_text, goal, days_per_week,
                    stored_weeks, stored_total, _json_list(plan), notes, status, start,
                    operation_id, now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_programs WHERE program_id = ?", (new_id,)
            ).fetchone()
            return row, False

        row, duplicate = await self._db.transaction(run)
        await self.touch_profile(user_id)
        return await self._program_with_progress(user_id, row), duplicate

    async def get_program(self, program_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT * FROM training_programs WHERE program_id = ? AND user_id = ?",
            (program_id, user_id),
        )
        if row is None:
            return None
        return await self._program_with_progress(user_id, row)

    async def list_programs(
        self, user_id: str, *, athlete_id: str | None = None, status: str | None = "active"
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM training_programs WHERE user_id = ?"
        params: list[Any] = [user_id]
        if athlete_id:
            sql += " AND athlete_id = ?"
            params.append(athlete_id)
        if status and status != "all":
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC"
        rows = await self._db.fetch_all(sql, params)
        return [await self._program_with_progress(user_id, row) for row in rows]

    async def progress(
        self,
        user_id: str,
        *,
        athlete_id: str | None = None,
        program_id: str | None = None,
    ) -> dict[str, Any]:
        """How many workouts are done and how many remain.

        ``done`` is completed logs in the programme window (or all logs if there is no
        programme). ``remaining`` is ``total - done`` when the programme has a length,
        otherwise the number of still-planned sessions.
        """
        program: dict[str, Any] | None = None
        if program_id:
            row = await self._db.fetch_one(
                "SELECT * FROM training_programs WHERE program_id = ? AND user_id = ?",
                (program_id, user_id),
            )
            if row is None:
                raise LookupError("program not found")
            program = _public_program(row)
            athlete_id = program["athlete_id"]
        elif athlete_id:
            row = await self._db.fetch_one(
                "SELECT * FROM training_programs WHERE user_id = ? AND athlete_id = ? "
                "AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
                (user_id, athlete_id),
            )
            if row is not None:
                program = _public_program(row)

        date_from = None
        date_to = None
        total = None
        if program:
            date_from = program.get("started_on") or (program.get("created_at") or "")[:10] or None
            date_to = program.get("ended_on")
            total = program.get("total_sessions")
            athlete_id = program["athlete_id"]

        logs = await self.list_logs(
            user_id,
            athlete_id=athlete_id,
            date_from=date_from,
            date_to=date_to,
            limit=500,
        )
        sessions = await self.list_sessions(
            user_id,
            athlete_id=athlete_id,
            date_from=date_from,
            date_to=date_to,
            status="all",
            limit=500,
        )
        if program:
            sessions = [
                item for item in sessions
                if item.get("program_id") in {program["program_id"], None}
            ]
        done = len(logs)
        planned = sum(1 for item in sessions if item["status"] == "planned")
        skipped = sum(1 for item in sessions if item["status"] == "skipped")
        remaining = max(0, int(total) - done) if total is not None else planned
        payload = {
            "done": done,
            "remaining": remaining,
            "total": total,
            "planned": planned,
            "skipped": skipped,
            "athlete_id": athlete_id,
            "program_id": program["program_id"] if program else None,
            "program_title": program["title"] if program else None,
        }
        payload["label"] = format_progress(payload)
        return payload

    async def progress_all(self, user_id: str) -> list[dict[str, Any]]:
        athletes = await self.list_athletes(user_id)
        out: list[dict[str, Any]] = []
        for athlete in athletes:
            item = await self.progress(user_id, athlete_id=athlete["athlete_id"])
            item["athlete_name"] = athlete["display_name"]
            item["label"] = format_progress(item)
            out.append(item)
        return out

    async def _program_with_progress(
        self, user_id: str, row: sqlite3.Row
    ) -> dict[str, Any]:
        program = _public_program(row)
        program["progress"] = await self.progress(
            user_id, program_id=program["program_id"]
        )
        return program

    # ---- schedule --------------------------------------------------------

    async def upsert_session(
        self,
        *,
        user_id: str,
        athlete_id: str,
        local_date: str,
        title: str,
        plan: Any = None,
        notes: str | None = None,
        status: str = "planned",
        program_id: str | None = None,
        session_id: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if await self.get_athlete(athlete_id, user_id) is None:
            raise LookupError("athlete not found")
        day = _local_date(local_date)
        title_text = title.strip()
        if not title_text:
            raise ValueError("title is required")
        if status not in {"planned", "done", "skipped"}:
            raise ValueError("status must be planned, done or skipped")
        if program_id:
            if await self._db.fetch_one(
                "SELECT 1 FROM training_programs WHERE program_id = ? AND user_id = ?",
                (program_id, user_id),
            ) is None:
                raise LookupError("program not found")
        else:
            active = await self._db.fetch_one(
                "SELECT program_id FROM training_programs WHERE user_id = ? AND athlete_id = ? "
                "AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
                (user_id, athlete_id),
            )
            if active is not None:
                program_id = active["program_id"]
        blocks = normalize_plan(plan)
        now = utcnow().isoformat()
        new_id = new_ulid()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            if operation_id:
                existing = connection.execute(
                    "SELECT * FROM training_sessions WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    return existing, True
            if session_id:
                current = connection.execute(
                    "SELECT * FROM training_sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                ).fetchone()
                if current is None:
                    raise LookupError("session not found")
                connection.execute(
                    "UPDATE training_sessions SET local_date = ?, title = ?, plan = ?, notes = ?, "
                    "status = ?, program_id = ?, updated_at = ? "
                    "WHERE session_id = ? AND user_id = ?",
                    (
                        day, title_text,
                        _json_list(blocks) if plan is not None else current["plan"],
                        notes if notes is not None else current["notes"],
                        status,
                        program_id if program_id is not None else current["program_id"],
                        now, session_id, user_id,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM training_sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                return row, False
            connection.execute(
                "INSERT INTO training_sessions(session_id, user_id, athlete_id, program_id, "
                "local_date, title, plan, notes, status, operation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id, user_id, athlete_id, program_id, day, title_text,
                    _json_list(blocks), notes, status, operation_id, now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_sessions WHERE session_id = ?", (new_id,)
            ).fetchone()
            return row, False

        row, duplicate = await self._db.transaction(run)
        await self.touch_profile(user_id)
        athlete = await self.get_athlete(athlete_id, user_id)
        name = athlete["display_name"] if athlete else None
        return _public_session(row, athlete_name=name), duplicate

    async def get_session(self, session_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT s.*, a.display_name AS athlete_name FROM training_sessions s "
            "JOIN training_athletes a ON a.athlete_id = s.athlete_id "
            "WHERE s.session_id = ? AND s.user_id = ?",
            (session_id, user_id),
        )
        return _public_session(row, athlete_name=row["athlete_name"]) if row else None

    async def list_sessions(
        self,
        user_id: str,
        *,
        athlete_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT s.*, a.display_name AS athlete_name FROM training_sessions s "
            "JOIN training_athletes a ON a.athlete_id = s.athlete_id "
            "WHERE s.user_id = ?"
        )
        params: list[Any] = [user_id]
        if athlete_id:
            sql += " AND s.athlete_id = ?"
            params.append(athlete_id)
        if date_from:
            sql += " AND s.local_date >= ?"
            params.append(_local_date(date_from))
        if date_to:
            sql += " AND s.local_date <= ?"
            params.append(_local_date(date_to))
        if status and status != "all":
            sql += " AND s.status = ?"
            params.append(status)
        sql += " ORDER BY s.local_date, a.display_name LIMIT ?"
        params.append(limit)
        rows = await self._db.fetch_all(sql, params)
        return [_public_session(row, athlete_name=row["athlete_name"]) for row in rows]

    async def mark_session_done(self, session_id: str, user_id: str) -> None:
        await self._db.execute(
            "UPDATE training_sessions SET status = 'done', updated_at = ? "
            "WHERE session_id = ? AND user_id = ? AND status = 'planned'",
            (utcnow().isoformat(), session_id, user_id),
        )

    # ---- logs ------------------------------------------------------------

    async def save_log(
        self,
        *,
        user_id: str,
        athlete_id: str,
        local_date: str,
        exercises: Any,
        title: str | None = None,
        raw_text: str | None = None,
        notes: str | None = None,
        duration_minutes: int | None = None,
        session_id: str | None = None,
        log_id: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if await self.get_athlete(athlete_id, user_id) is None:
            raise LookupError("athlete not found")
        day = _local_date(local_date)
        structured = normalize_exercises(exercises)
        if session_id:
            session = await self.get_session(session_id, user_id)
            if session is None or session["athlete_id"] != athlete_id:
                raise LookupError("session not found")
        elif not log_id:
            planned = await self.list_sessions(
                user_id, athlete_id=athlete_id, date_from=day, date_to=day, status="planned",
            )
            if len(planned) == 1:
                session_id = planned[0]["session_id"]
        now = utcnow().isoformat()
        new_id = new_ulid()

        def run(connection: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            if operation_id:
                existing = connection.execute(
                    "SELECT * FROM training_logs WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    return existing, True
            if log_id:
                current = connection.execute(
                    "SELECT * FROM training_logs WHERE log_id = ? AND user_id = ?",
                    (log_id, user_id),
                ).fetchone()
                if current is None:
                    raise LookupError("log not found")
                connection.execute(
                    "UPDATE training_logs SET local_date = ?, title = ?, raw_text = ?, "
                    "exercises = ?, notes = ?, duration_minutes = ?, session_id = ?, "
                    "updated_at = ? WHERE log_id = ? AND user_id = ?",
                    (
                        day,
                        title if title is not None else current["title"],
                        raw_text if raw_text is not None else current["raw_text"],
                        _json_list(structured),
                        notes if notes is not None else current["notes"],
                        duration_minutes if duration_minutes is not None else current["duration_minutes"],
                        session_id if session_id is not None else current["session_id"],
                        now, log_id, user_id,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM training_logs WHERE log_id = ?", (log_id,)
                ).fetchone()
                return row, False
            connection.execute(
                "INSERT INTO training_logs(log_id, user_id, athlete_id, session_id, local_date, "
                "title, raw_text, exercises, notes, duration_minutes, operation_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id, user_id, athlete_id, session_id, day, title, raw_text,
                    _json_list(structured), notes, duration_minutes, operation_id, now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_logs WHERE log_id = ?", (new_id,)
            ).fetchone()
            return row, False

        row, duplicate = await self._db.transaction(run)
        if session_id and not duplicate:
            await self.mark_session_done(session_id, user_id)
        await self.touch_profile(user_id)
        athlete = await self.get_athlete(athlete_id, user_id)
        name = athlete["display_name"] if athlete else None
        return _public_log(row, athlete_name=name), duplicate

    async def get_log(self, log_id: str, user_id: str) -> dict[str, Any] | None:
        row = await self._db.fetch_one(
            "SELECT l.*, a.display_name AS athlete_name FROM training_logs l "
            "JOIN training_athletes a ON a.athlete_id = l.athlete_id "
            "WHERE l.log_id = ? AND l.user_id = ?",
            (log_id, user_id),
        )
        return _public_log(row, athlete_name=row["athlete_name"]) if row else None

    async def list_logs(
        self,
        user_id: str,
        *,
        athlete_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        query: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT l.*, a.display_name AS athlete_name FROM training_logs l "
            "JOIN training_athletes a ON a.athlete_id = l.athlete_id "
            "WHERE l.user_id = ?"
        )
        params: list[Any] = [user_id]
        if athlete_id:
            sql += " AND l.athlete_id = ?"
            params.append(athlete_id)
        if date_from:
            sql += " AND l.local_date >= ?"
            params.append(_local_date(date_from))
        if date_to:
            sql += " AND l.local_date <= ?"
            params.append(_local_date(date_to))
        sql += " ORDER BY l.local_date DESC, l.created_at DESC"
        rows = await self._db.fetch_all(sql, params)
        needle = query.strip().casefold()
        logs = [_public_log(row, athlete_name=row["athlete_name"]) for row in rows]
        if needle:
            logs = [item for item in logs if _log_matches(item, needle)]
        return logs[:limit]


def _log_matches(log: dict[str, Any], needle: str) -> bool:
    haystacks = [
        log.get("title") or "",
        log.get("raw_text") or "",
        log.get("notes") or "",
        log.get("athlete_name") or "",
        json.dumps(log.get("exercises") or [], ensure_ascii=False),
    ]
    return any(needle in text.casefold() for text in haystacks)
