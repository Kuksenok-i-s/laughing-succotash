"""Trainer-journal MCP tools.

The Telegram user is the coach. Athletes, programmes, the timetable and workout logs live in
SQLite for that user_id and survive session restarts. The model structures free-text reports;
these tools only persist and retrieve.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..storage.repositories.training import format_progress, logs_to_csv, schedule_to_csv
from .permissions import ToolContext
from .server import ToolRegistry


class _Args(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operation_id: str | None = Field(
        default=None,
        description="Optional idempotency key. Reuse it when retrying so the action is not "
        "performed twice.",
    )

_MEMORY = (
    "Пользователь ведёт журнал тренировок (таблицы training_*). "
    "Режим self или trainer — смотри training_profile_get. "
    "Читай training_athlete_list / training_schedule_list / training_log_list / "
    "training_progress, не выдумывай тренировки. После отчёта говори, сколько проведено и сколько осталось."
)

_SELF_NAMES = frozenset({"я", "меня", "себе", "себя", "сам", "сама", "мой", "моя"})


class AthleteList(_Args):
    query: str = Field(default="", description="Name or alias. Empty lists every active athlete.")
    status: str = Field(default="active", description="active, archived or all")


class EmptyProfile(_Args):
    pass


class ProfileSet(_Args):
    mode: str = Field(
        description="self — the Telegram user trains alone. trainer — they coach a group; "
        "every schedule and log must name an athlete.",
    )


class AthleteId(_Args):
    athlete_id: str


class AthleteUpsert(_Args):
    display_name: str = Field(description="How the coach refers to this person.")
    athlete_id: str | None = Field(
        default=None, description="Set to update an existing athlete rather than creating one."
    )
    aliases: list[str] = Field(default_factory=list, description="Nicknames used to find them.")
    note: str | None = Field(
        default=None, description="Goals, injuries, equipment, anything durable about them."
    )
    is_self: bool = Field(
        default=False, description="True when this is the Telegram user training themselves."
    )


class ProgramList(_Args):
    athlete_id: str | None = None
    athlete_name: str | None = None
    status: str = Field(default="active")


class ProgressQuery(_Args):
    athlete_id: str | None = None
    athlete_name: str | None = None
    program_id: str | None = None


class ProgramUpsert(_Args):
    title: str
    athlete_id: str | None = None
    athlete_name: str | None = None
    program_id: str | None = Field(default=None, description="Set to update an existing programme.")
    goal: str | None = None
    days_per_week: int | None = Field(default=None, ge=1, le=14)
    weeks: int | None = Field(
        default=None, ge=1, le=104,
        description="Programme length in weeks. With days_per_week sets total_sessions.",
    )
    total_sessions: int | None = Field(
        default=None, ge=1, le=500,
        description="How many workouts this programme contains. Remaining = total - done.",
    )
    weekly_plan: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Weekly split: [{weekday, title, exercises: [{name, sets, notes}]}].",
    )
    notes: str | None = None
    started_on: str | None = Field(default=None, description="YYYY-MM-DD")
    status: str = Field(default="active")


class ScheduleList(_Args):
    athlete_id: str | None = None
    athlete_name: str | None = None
    date_from: str | None = Field(default=None, description="YYYY-MM-DD inclusive.")
    date_to: str | None = Field(default=None, description="YYYY-MM-DD inclusive.")
    status: str | None = Field(default=None, description="planned, done, skipped or all")
    limit: int = Field(default=60, ge=1, le=200)


class ScheduleUpsert(_Args):
    local_date: str = Field(description="Session date, YYYY-MM-DD, in the user's timezone.")
    title: str = Field(description='e.g. "Ноги / присед" or "Full body A".')
    athlete_id: str | None = None
    athlete_name: str | None = None
    session_id: str | None = Field(default=None, description="Set to update an existing session.")
    program_id: str | None = None
    plan: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Planned exercises: [{name, sets, target_weight_kg, notes}].",
    )
    notes: str | None = None
    status: str = Field(default="planned", description="planned, done or skipped")


class LogList(_Args):
    query: str = Field(default="", description="Empty returns recent logs.")
    athlete_id: str | None = None
    athlete_name: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    limit: int = Field(default=40, ge=1, le=100)


class LogId(_Args):
    log_id: str


class LogSave(_Args):
    local_date: str = Field(description="Workout date, YYYY-MM-DD.")
    exercises: list[dict[str, Any]] = Field(
        description="Structured exercises: [{name, sets: [{reps, weight_kg, rpe, note}], notes}]. "
        "Parse free-text reports into this shape before saving."
    )
    athlete_id: str | None = None
    athlete_name: str | None = None
    log_id: str | None = Field(default=None, description="Set to update an existing log.")
    session_id: str | None = Field(
        default=None, description="Planned session this report completes, if known."
    )
    title: str | None = None
    raw_text: str | None = Field(
        default=None, description="The user's original report, kept for search."
    )
    notes: str | None = Field(default=None, description="RPE, well-being, substitutions.")
    duration_minutes: int | None = Field(default=None, ge=1, le=600)


class TrainingExport(_Args):
    kind: str = Field(
        default="logs",
        description="logs (workout table), schedule (timetable) or both.",
    )
    athlete_id: str | None = None
    athlete_name: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    filename: str | None = Field(default=None, description="CSV basename. Defaults from kind.")
    send: bool = Field(
        default=True,
        description="If true, deliver the CSV to Telegram. Always also return the rows.",
    )


def register_training_tools(registry: ToolRegistry, repos, *, file_delivery=None) -> None:
    async def profile_get(_args: EmptyProfile, ctx: ToolContext) -> dict[str, Any]:
        profile = await repos.training.get_profile(ctx.user_id)
        athletes = await repos.training.list_athletes(ctx.user_id)
        progress = await repos.training.progress_all(ctx.user_id)
        return {
            "profile": profile,
            "mode": (profile or {}).get("mode") or "self",
            "athletes": athletes,
            "progress": progress,
            "count": len(athletes),
        }

    registry.add(
        "training_profile_get",
        "Read the training-journal mode for this Telegram user: self (own workouts) or "
        "trainer (a group, each person with their own programme). Call this before logging "
        "or scheduling if the mode is unknown.",
        EmptyProfile,
        profile_get,
    )

    async def profile_set(args: ProfileSet, ctx: ToolContext) -> dict[str, Any]:
        profile = await repos.training.set_mode(ctx.user_id, args.mode)
        await _remember_trainer(repos, ctx)
        athletes = await repos.training.list_athletes(ctx.user_id)
        return {
            **profile,
            "athletes": athletes,
            "count": len(athletes),
            "guidance": (
                "Trainer mode: every schedule and report needs an athlete name. "
                "Do not mix people in one log."
                if profile["mode"] == "trainer"
                else "Self mode: logs without a name go to the user themselves."
            ),
        }

    registry.add(
        "training_profile_set",
        "Set the training-journal mode. Call with trainer when the user says they are a "
        "coach or start naming several athletes. Call with self when they only train themselves.",
        ProfileSet,
        profile_set,
        idempotent_write=True,
    )

    async def athlete_list(args: AthleteList, ctx: ToolContext) -> dict[str, Any]:
        if args.query.strip():
            items = await repos.training.search_athletes(ctx.user_id, args.query)
        else:
            items = await repos.training.list_athletes(ctx.user_id, status=args.status)
        profile = await repos.training.get_profile(ctx.user_id)
        mode = (profile or {}).get("mode") or "self"
        return {
            "mode": mode,
            "athletes": items,
            "count": len(items),
            "ambiguous": len(items) > 1 and bool(args.query.strip()),
            "guidance": (
                "Several athletes match. Ask the user which one; do not guess."
                if len(items) > 1 and args.query.strip()
                else (
                    "Trainer mode: address athletes by name."
                    if mode == "trainer"
                    else ""
                )
            ),
        }

    registry.add(
        "training_athlete_list",
        "List or search athletes in the trainer journal. Empty query returns everyone this "
        "Telegram user coaches. If more than one matches, ask; do not guess.",
        AthleteList,
        athlete_list,
    )

    async def athlete_get(args: AthleteId, ctx: ToolContext) -> dict[str, Any]:
        athlete = await repos.training.get_athlete(args.athlete_id, ctx.user_id)
        if athlete is None:
            return {"error": "not found"}
        programs = await repos.training.list_programs(ctx.user_id, athlete_id=athlete["athlete_id"])
        progress = await repos.training.progress(ctx.user_id, athlete_id=athlete["athlete_id"])
        return {"athlete": athlete, "programs": programs, "progress": progress}

    registry.add(
        "training_athlete_get",
        "Get one athlete, their programmes, and workout progress (done / remaining).",
        AthleteId,
        athlete_get,
    )

    async def athlete_upsert(args: AthleteUpsert, ctx: ToolContext) -> dict[str, Any]:
        try:
            athlete, duplicate = await repos.training.upsert_athlete(
                user_id=ctx.user_id,
                display_name=args.display_name,
                aliases=args.aliases,
                note=args.note,
                is_self=args.is_self,
                athlete_id=args.athlete_id,
                operation_id=args.operation_id,
            )
        except LookupError:
            return {"error": "not found"}
        profile = await _remember_trainer(repos, ctx)
        return {
            **athlete,
            "mode": profile["mode"],
            "created": not duplicate and not args.athlete_id,
            "duplicate": duplicate,
        }

    registry.add(
        "training_athlete_upsert",
        "Add or update an athlete this Telegram user coaches. Search first so you do not create "
        "a duplicate. Use is_self when the user is logging their own training. Persists in SQLite.",
        AthleteUpsert,
        athlete_upsert,
        idempotent_write=True,
    )

    async def athlete_archive(args: AthleteId, ctx: ToolContext) -> dict[str, Any]:
        ok = await repos.training.archive_athlete(args.athlete_id, ctx.user_id)
        return {"status": "archived" if ok else "not_found_or_already_archived"}

    registry.add(
        "training_athlete_archive",
        "Archive an athlete. Logs and programmes stay; they disappear from the active list.",
        AthleteId,
        athlete_archive,
        idempotent_write=True,
    )

    async def program_list(args: ProgramList, ctx: ToolContext) -> dict[str, Any]:
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, required=False
        )
        if "error" in resolved:
            return resolved
        athlete = resolved.get("athlete")
        athlete_id = athlete["athlete_id"] if athlete else None
        items = await repos.training.list_programs(
            ctx.user_id, athlete_id=athlete_id, status=args.status
        )
        return {"programs": items, "count": len(items)}

    registry.add(
        "training_program_list",
        "List training programmes with progress (done / remaining). Pass an athlete to see only theirs.",
        ProgramList,
        program_list,
    )

    async def training_progress(args: ProgressQuery, ctx: ToolContext) -> dict[str, Any]:
        if args.program_id:
            try:
                item = await repos.training.progress(
                    ctx.user_id, program_id=args.program_id
                )
            except LookupError:
                return {"error": "not found"}
            return {"progress": item, "items": [item]}
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, required=False
        )
        if "error" in resolved:
            return resolved
        athlete = resolved.get("athlete")
        if athlete:
            item = await repos.training.progress(
                ctx.user_id, athlete_id=athlete["athlete_id"]
            )
            item["athlete_name"] = athlete["display_name"]
            item["label"] = format_progress(item)
            return {"progress": item, "items": [item]}
        items = await repos.training.progress_all(ctx.user_id)
        return {"items": items, "count": len(items)}

    registry.add(
        "training_progress",
        "How many workouts are done and how many remain. Pass an athlete or program_id, "
        "or omit both to see everyone. Always tell the user the numbers from progress.label.",
        ProgressQuery,
        training_progress,
    )

    async def program_upsert(args: ProgramUpsert, ctx: ToolContext) -> dict[str, Any]:
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, create=True,
            operation_id=args.operation_id,
        )
        if "error" in resolved:
            return resolved
        athlete = resolved["athlete"]
        try:
            program, duplicate = await repos.training.upsert_program(
                user_id=ctx.user_id,
                athlete_id=athlete["athlete_id"],
                title=args.title,
                goal=args.goal,
                days_per_week=args.days_per_week,
                weeks=args.weeks,
                total_sessions=args.total_sessions,
                weekly_plan=args.weekly_plan,
                notes=args.notes,
                started_on=args.started_on,
                status=args.status,
                program_id=args.program_id,
                operation_id=args.operation_id,
            )
        except LookupError:
            return {"error": "not found"}
        profile = await _remember_trainer(repos, ctx)
        return {
            **program,
            "athlete": athlete,
            "mode": profile["mode"],
            "created": not duplicate and not args.program_id,
            "duplicate": duplicate,
        }

    registry.add(
        "training_program_upsert",
        "Save or replace the current programme for one athlete. Pass total_sessions (or weeks "
        "and days_per_week) so remaining workouts can be counted. A new active programme "
        "archives the previous active one for that person.",
        ProgramUpsert,
        program_upsert,
        idempotent_write=True,
    )

    async def schedule_list(args: ScheduleList, ctx: ToolContext) -> dict[str, Any]:
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, required=False
        )
        if "error" in resolved:
            return resolved
        athlete = resolved.get("athlete")
        items = await repos.training.list_sessions(
            ctx.user_id,
            athlete_id=athlete["athlete_id"] if athlete else None,
            date_from=args.date_from,
            date_to=args.date_to,
            status=args.status,
            limit=args.limit,
        )
        return {"sessions": items, "count": len(items)}

    registry.add(
        "training_schedule_list",
        "List planned and completed training sessions. Filter by athlete and date range.",
        ScheduleList,
        schedule_list,
    )

    async def schedule_upsert(args: ScheduleUpsert, ctx: ToolContext) -> dict[str, Any]:
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, create=True,
            operation_id=args.operation_id,
        )
        if "error" in resolved:
            return resolved
        athlete = resolved["athlete"]
        try:
            session, duplicate = await repos.training.upsert_session(
                user_id=ctx.user_id,
                athlete_id=athlete["athlete_id"],
                local_date=args.local_date,
                title=args.title,
                plan=args.plan,
                notes=args.notes,
                status=args.status,
                program_id=args.program_id,
                session_id=args.session_id,
                operation_id=args.operation_id,
            )
        except LookupError:
            return {"error": "not found"}
        profile = await _remember_trainer(repos, ctx)
        return {
            **session,
            "mode": profile["mode"],
            "created": not duplicate and not args.session_id,
            "duplicate": duplicate,
        }

    registry.add(
        "training_schedule_upsert",
        "Create or update one scheduled workout for one athlete. In trainer mode repeat the "
        "call per person — never put two people on one session. Persists in SQLite.",
        ScheduleUpsert,
        schedule_upsert,
        idempotent_write=True,
    )

    async def log_list(args: LogList, ctx: ToolContext) -> dict[str, Any]:
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, required=False
        )
        if "error" in resolved:
            return resolved
        athlete = resolved.get("athlete")
        items = await repos.training.list_logs(
            ctx.user_id,
            athlete_id=athlete["athlete_id"] if athlete else None,
            date_from=args.date_from,
            date_to=args.date_to,
            query=args.query,
            limit=args.limit,
        )
        return {"logs": items, "count": len(items)}

    registry.add(
        "training_log_list",
        "Search structured workout reports. Empty query returns recent logs. Use this instead "
        "of inventing past weights.",
        LogList,
        log_list,
    )

    async def log_get(args: LogId, ctx: ToolContext) -> dict[str, Any]:
        log = await repos.training.get_log(args.log_id, ctx.user_id)
        return log or {"error": "not found"}

    registry.add("training_log_get", "Get one workout log by id.", LogId, log_get)

    async def log_save(args: LogSave, ctx: ToolContext) -> dict[str, Any]:
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, create=True,
            operation_id=args.operation_id,
        )
        if "error" in resolved:
            return resolved
        athlete = resolved["athlete"]
        try:
            log, duplicate = await repos.training.save_log(
                user_id=ctx.user_id,
                athlete_id=athlete["athlete_id"],
                local_date=args.local_date,
                exercises=args.exercises,
                title=args.title,
                raw_text=args.raw_text,
                notes=args.notes,
                duration_minutes=args.duration_minutes,
                session_id=args.session_id,
                log_id=args.log_id,
                operation_id=args.operation_id,
            )
        except LookupError:
            return {"error": "not found"}
        profile = await _remember_trainer(repos, ctx)
        progress = await repos.training.progress(
            ctx.user_id, athlete_id=athlete["athlete_id"]
        )
        return {
            **log,
            "mode": profile["mode"],
            "progress": progress,
            "created": not duplicate and not args.log_id,
            "duplicate": duplicate,
        }

    registry.add(
        "training_log_save",
        "Save a structured workout report (exercises, sets, weights) for one athlete. "
        "Parse the user's free-text or voice report first. The result includes progress "
        "(done / remaining). Always tell the user how many workouts are done and how many remain.",
        LogSave,
        log_save,
        idempotent_write=True,
    )

    async def training_export(args: TrainingExport, ctx: ToolContext) -> dict[str, Any]:
        kind = (args.kind or "logs").strip().lower()
        if kind not in {"logs", "schedule", "both"}:
            raise ValueError("kind must be logs, schedule or both")
        resolved = await _resolve_athlete(
            repos, ctx, args.athlete_id, args.athlete_name, required=False
        )
        if "error" in resolved:
            return resolved
        athlete = resolved.get("athlete")
        athlete_id = athlete["athlete_id"] if athlete else None
        chunks: list[tuple[str, str]] = []
        if kind in {"logs", "both"}:
            logs = await repos.training.list_logs(
                ctx.user_id,
                athlete_id=athlete_id,
                date_from=args.date_from,
                date_to=args.date_to,
                limit=200,
            )
            chunks.append(("тренировки.csv", logs_to_csv(logs)))
        if kind in {"schedule", "both"}:
            sessions = await repos.training.list_sessions(
                ctx.user_id,
                athlete_id=athlete_id,
                date_from=args.date_from,
                date_to=args.date_to,
                limit=200,
            )
            chunks.append(("расписание.csv", schedule_to_csv(sessions)))
        sent: list[dict[str, Any]] = []
        if args.send:
            if file_delivery is None:
                raise RuntimeError("file delivery is not configured")
            for default_name, csv_body in chunks:
                filename = args.filename or default_name
                if kind == "both" and args.filename:
                    filename = default_name
                sent.append(
                    await file_delivery.send(
                        ctx,
                        filename=filename,
                        content=csv_body,
                        caption="Отчёт из журнала тренера",
                        operation_id=(
                            (args.operation_id or f"training-export:{ctx.user_id}")
                            + f":{default_name}"
                        ),
                    )
                )
        previews = [
            {"filename": name, "csv": body, "rows": body.count("\n")}
            for name, body in chunks
        ]
        return {"files": previews, "sent": sent, "count": len(previews)}

    registry.add(
        "training_export",
        "Build a CSV table of workout logs and/or the timetable and send it to Telegram. "
        "Use this when the user asks for a table or spreadsheet report. Do not paste the "
        "full CSV into the chat.",
        TrainingExport,
        training_export,
        idempotent_write=True,
    )


async def _remember_trainer(repos, ctx: ToolContext) -> dict[str, Any]:
    profile = await repos.training.get_profile(ctx.user_id) or await repos.training.touch_profile(
        ctx.user_id
    )
    await repos.memory.remember(
        user_id=ctx.user_id,
        content=_MEMORY,
        category="training",
        source="explicit" if ctx.trusted else "confirmed",
        operation_id=f"training:enabled:{ctx.user_id}",
    )
    return profile


async def _resolve_athlete(
    repos,
    ctx: ToolContext,
    athlete_id: str | None,
    athlete_name: str | None,
    *,
    required: bool = True,
    create: bool = False,
    operation_id: str | None = None,
) -> dict[str, Any]:
    profile = await repos.training.get_profile(ctx.user_id)
    mode = (profile or {}).get("mode") or "self"
    if athlete_id:
        athlete = await repos.training.get_athlete(athlete_id, ctx.user_id)
        if athlete is None:
            return {"error": "athlete not found", "mode": mode}
        return {"athlete": athlete, "mode": mode}
    name = (athlete_name or "").strip()
    if name and name.casefold() in _SELF_NAMES:
        selves = [
            item for item in await repos.training.list_athletes(ctx.user_id) if item["is_self"]
        ]
        if len(selves) == 1:
            return {"athlete": selves[0], "mode": mode}
        if create:
            athlete, _ = await repos.training.upsert_athlete(
                user_id=ctx.user_id,
                display_name="я",
                is_self=True,
                operation_id=(f"{operation_id}:self" if operation_id else None),
            )
            return {"athlete": athlete, "mode": mode}
        return {"error": "athlete not found", "mode": mode}
    if name:
        matches = await repos.training.search_athletes(ctx.user_id, name)
        exact = [item for item in matches if item["display_name"].casefold() == name.casefold()]
        if len(exact) == 1:
            return {"athlete": exact[0], "mode": mode}
        if len(matches) == 1:
            return {"athlete": matches[0], "mode": mode}
        if len(matches) > 1:
            return {
                "error": "ambiguous",
                "mode": mode,
                "athletes": matches,
                "guidance": "Several athletes match. Ask the user which one; do not guess.",
            }
        if create:
            athlete, _ = await repos.training.upsert_athlete(
                user_id=ctx.user_id,
                display_name=name,
                operation_id=(f"{operation_id}:athlete" if operation_id else None),
            )
            profile = await repos.training.get_profile(ctx.user_id)
            return {"athlete": athlete, "mode": (profile or {}).get("mode") or "trainer"}
        return {"error": "athlete not found", "mode": mode, "guidance": "Create the athlete first."}
    active = await repos.training.list_athletes(ctx.user_id)
    if not required and not create:
        return {"athlete": None, "mode": mode}
    if mode == "trainer":
        return {
            "error": "athlete_required",
            "mode": "trainer",
            "athletes": active,
            "guidance": "Trainer mode: ask which athlete this is for. Do not guess.",
        }
    selves = [item for item in active if item["is_self"]]
    if len(selves) == 1:
        return {"athlete": selves[0], "mode": mode}
    if len(active) == 1:
        return {"athlete": active[0], "mode": mode}
    if not active and create:
        athlete, _ = await repos.training.upsert_athlete(
            user_id=ctx.user_id,
            display_name="я",
            is_self=True,
            operation_id=(f"{operation_id}:self" if operation_id else None),
        )
        return {"athlete": athlete, "mode": "self"}
    return {
        "error": "athlete_required",
        "mode": mode,
        "athletes": active,
        "guidance": "Ask which athlete this is for; do not guess.",
    }
