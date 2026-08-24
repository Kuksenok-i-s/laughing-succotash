"""The Core's orchestration layer: turns inbound requests into jobs, and jobs into replies.

Everything a user sends becomes a job with an identifier, and every job reports its progress and
its outcome to the Gateway. Nothing here talks to Telegram directly — it emits ``telegram.*``
intents, which the Gateway renders.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pa_protocol import methods

from ..agent.base import AgentContext, AgentError, AgentUnavailable, Provenance
from ..audio.converter import probe_duration
from ..mcp.permissions import ToolContext
from ..mcp.timeutil import format_local
from ..stt.base import SttError, TranscriptionResult
from ..storage.repositories import Job, Upload
from . import prompts

log = logging.getLogger(__name__)

_MAX_REPLY_CHARS = 24000


class AssistantService:
    def __init__(
        self,
        settings,
        repos,
        link,
        sessions,
        job_manager,
        backend,
        *,
        stt=None,
        uploads=None,
        analyzer=None,
    ) -> None:
        self._settings = settings
        self._repos = repos
        self._link = link
        self._sessions = sessions
        self._jobs = job_manager
        self._backend = backend
        self._stt = stt
        self._uploads = uploads
        self._analyzer = analyzer

    # ---- intake ---------------------------------------------------------

    async def submit(self, params: methods.AssistantSubmitParams) -> methods.AcceptedResult:
        """Accept a text message or command. Returns immediately; the work happens in a job."""
        conversation = await self._repos.conversations.get_or_create_conversation(params.user_id)

        job, duplicate = await self._repos.jobs.create_or_get(
            request_id=params.request_id,
            user_id=params.user_id,
            kind=params.kind,
            chat_id=params.chat_id,
            message_id=params.message_id,
            payload={
                "text": params.text,
                "command": params.command,
                "conversation_id": conversation.conversation_id,
            },
        )
        if duplicate:
            # The Gateway retried after a lost response. The original job is already running or
            # finished; handing back its id is the whole point of request_id being unique.
            log.info("duplicate submit for request %s -> job %s", params.request_id, job.job_id)
            return methods.AcceptedResult(job_id=job.job_id, dedup=True)

        if params.kind == "command":
            runner = self._wrap(job, lambda: self._run_command(job, params.command or ""))
            # Control commands run on their own lane. /cancel queued behind the job it is meant to
            # cancel would only take effect once that job finished, which is worse than useless;
            # the listings never touch the Cursor session, so serialising them buys nothing.
            lane = (
                f"control:{params.user_id}"
                if _is_control(params.command)
                else conversation.conversation_id
            )
        else:
            runner = self._wrap(job, lambda: self._run_text(job, params.text or ""))
            lane = conversation.conversation_id

        await self._jobs.submit(lane, job.job_id, runner)
        return methods.AcceptedResult(job_id=job.job_id)

    async def start_audio_job(self, upload: Upload) -> str:
        """Begin processing a committed audio upload.

        Audio has no ``assistant.submit``: the upload's own ``request_id`` identifies the request,
        so a re-uploaded file after a reconnect maps to the same job.
        """
        conversation = await self._repos.conversations.get_or_create_conversation(upload.user_id)

        job, duplicate = await self._repos.jobs.create_or_get(
            request_id=upload.request_id,
            user_id=upload.user_id,
            kind="audio",
            chat_id=upload.chat_id,
            message_id=upload.message_id,
            payload={
                "upload_id": upload.upload_id,
                "purpose": upload.purpose,
                "conversation_id": conversation.conversation_id,
            },
        )
        if duplicate and job.is_terminal:
            log.info("audio for request %s already processed", upload.request_id)
            return job.job_id

        await self._jobs.submit(
            conversation.conversation_id,
            job.job_id,
            self._wrap(job, lambda: self._run_audio(job, upload)),
        )
        return job.job_id

    def _wrap(self, job: Job, body):
        """Common job lifecycle: running → terminal, with cancellation and failure reporting."""

        async def runner() -> None:
            await self._repos.jobs.mark_running(job.job_id)
            started = time.monotonic()
            try:
                await body()
            except asyncio.CancelledError:
                await self._repos.jobs.finish(job.job_id, "cancelled")
                await self._report_failure(job, "cancelled", "job cancelled")
                raise
            except AgentUnavailable as exc:
                log.warning("job %s: agent unavailable: %s", job.job_id, exc)
                await self._fail(job, "agent_unavailable", str(exc))
            except AgentError as exc:
                log.warning("job %s: agent failed: %s", job.job_id, exc)
                await self._fail(job, "agent_failed", str(exc))
            except SttError as exc:
                log.warning("job %s: transcription failed: %s", job.job_id, exc)
                await self._fail(job, "stt_failed", str(exc))
            except Exception as exc:
                log.exception("job %s failed", job.job_id)
                await self._fail(job, "internal_error", f"{type(exc).__name__}: {exc}")
            else:
                await self._repos.jobs.finish(job.job_id, "completed")
                await self._link.send_event(
                    methods.JOB_COMPLETED,
                    methods.dump(
                        methods.JobCompletedParams(
                            job_id=job.job_id, user_id=job.user_id, chat_id=job.chat_id
                        )
                    ),
                    delivery_id=f"{job.job_id}:done",
                    user_id=job.user_id,
                )
                log.info(
                    "job %s completed in %.1fs (kind=%s user=%s)",
                    job.job_id, time.monotonic() - started, job.kind, job.user_id,
                )

        return runner

    # ---- text -----------------------------------------------------------

    async def _run_text(self, job: Job, text: str) -> None:
        if not text.strip():
            return
        response = await self._converse(
            job, text, provenance=Provenance.DIRECT_COMMAND, wrap=prompts.direct_turn
        )
        await self._reply(job, response)

    async def _converse(
        self,
        job: Job,
        message: str,
        *,
        provenance: Provenance,
        wrap,
    ) -> str:
        conversation_id = job.payload.get("conversation_id") or (
            await self._repos.conversations.get_or_create_conversation(job.user_id)
        ).conversation_id

        session, is_new = await self._sessions.ensure_session(conversation_id)
        user_tz = await self._repos.conversations.timezone_for(job.user_id)
        now = datetime.now(timezone.utc)

        agent_context = AgentContext(
            user_id=job.user_id,
            conversation_id=conversation_id,
            job_id=job.job_id,
            timezone=user_tz,
            now=now,
            provenance=provenance,
        )
        # The MCP server reads this to decide whether a write may run unattended. It is set from
        # the Core's own knowledge of where the turn came from, never from anything the model says.
        tool_context = ToolContext(
            user_id=job.user_id,
            conversation_id=conversation_id,
            provenance=provenance,
            job_id=job.job_id,
            chat_id=job.chat_id,
            timezone=user_tz,
            now=now,
        )

        prompt = (
            prompts.first_turn(message, agent_context)
            if is_new
            else wrap(message, agent_context)
        )

        self._sessions.begin_turn(conversation_id, tool_context)
        await self._progress(job, "agent")
        try:
            response = await self._backend.send_message(
                session.external_id,
                prompt,
                agent_context,
                on_progress=lambda stage, detail: self._progress(job, stage, detail=detail),
            )
        except asyncio.CancelledError:
            await self._backend.cancel(session.external_id)
            raise
        finally:
            self._sessions.end_turn(conversation_id)
            await self._repos.conversations.touch_conversation(conversation_id)

        if response.cancelled:
            return ""
        return response.text

    # ---- audio ------------------------------------------------------------

    async def _run_audio(self, job: Job, upload: Upload) -> None:
        if self._stt is None:
            raise SttError("speech-to-text is not configured on this Core")

        transcription = await self._transcribe(job, upload)

        if upload.purpose == "transcribe_only":
            # Explicitly requested raw transcription: no conversation turn, no tools, no analysis.
            await self._reply(job, transcription.text or "Речь не распознана.")
            return

        if len(transcription.text) <= self._settings.long_transcript_chars:
            # Short enough to be a spoken instruction rather than a recording of other people.
            response = await self._converse(
                job,
                transcription.text,
                provenance=Provenance.DIRECT_COMMAND,
                wrap=prompts.voice_turn,
            )
            await self._reply(job, response)
            return

        await self._analyze_recording(job, transcription)

    async def _transcribe(self, job: Job, upload: Upload) -> TranscriptionResult:
        await self._progress(job, "transcribing")

        duration = await probe_duration(upload.temp_path) or upload.duration_seconds
        if duration and duration > self._settings.max_audio_duration_seconds:
            await self._uploads.release(upload)
            raise SttError(
                f"recording is {int(duration // 60)} minutes, over the configured limit"
            )

        started = time.monotonic()

        async def report(fraction: float) -> None:
            await self._progress(job, "transcribing", progress=fraction)

        def on_progress(fraction: float) -> None:
            # Called from the whisper worker thread via call_soon_threadsafe.
            asyncio.ensure_future(report(fraction))

        try:
            result = await self._stt.transcribe(upload.temp_path, on_progress=on_progress)
        finally:
            # The recording is deleted whether transcription succeeded or not: it is the most
            # sensitive artefact in the system and the transcript is what we actually need.
            if self._uploads is not None:
                await self._uploads.release(upload)

        await self._repos.transcriptions.record(
            user_id=job.user_id,
            job_id=job.job_id,
            filename=upload.filename,
            language=result.language,
            duration=result.duration or duration,
            segment_count=len(result.segments),
            char_count=len(result.text),
            model=getattr(self._stt, "model_name", "unknown"),
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

        if result.empty:
            raise SttError("no speech detected in the recording")
        return result

    async def _analyze_recording(self, job: Job, transcription: TranscriptionResult) -> None:
        """Long recording: hierarchical extraction, then one conversational turn over the notes."""
        await self._progress(job, "summarizing")

        conversation_id = job.payload.get("conversation_id")
        agent_context = AgentContext(
            user_id=job.user_id,
            conversation_id=conversation_id or "",
            job_id=job.job_id,
            timezone=await self._repos.conversations.timezone_for(job.user_id),
            now=datetime.now(timezone.utc),
            provenance=Provenance.UNTRUSTED_CONTENT,
        )

        analysis = await self._analyzer.analyze(
            transcription,
            agent_context,
            on_progress=lambda fraction: self._progress(
                job, "summarizing", progress=fraction
            ),
        )

        message = prompts.transcript_turn(
            analysis.notes,
            agent_context,
            duration_seconds=transcription.duration,
            excerpt=analysis.excerpt or None,
        )
        # Provenance stays UNTRUSTED_CONTENT for the final turn too: a proposal that came out of
        # the recording still needs the user's explicit yes before anything is created.
        response = await self._converse(
            job, message, provenance=Provenance.UNTRUSTED_CONTENT, wrap=_verbatim
        )
        await self._reply(job, response)

    # ---- commands ------------------------------------------------------------

    async def _run_command(self, job: Job, command: str) -> None:
        name = command.split()[0].lstrip("/").lower() if command else ""

        if name == "cancel":
            await self._reply(job, await self._cancel_active(job))
        elif name == "reminders":
            await self._reply(job, await self._render_reminders(job.user_id))
        elif name == "tasks":
            await self._reply(job, await self._render_tasks(job.user_id))
        else:
            log.info("unhandled command %s from %s", command, job.user_id)
            await self._reply(job, "Не знаю такую команду.")

    async def _cancel_active(self, job: Job) -> str:
        active = [
            other
            for other in await self._repos.jobs.active_for_user(job.user_id)
            if other.job_id != job.job_id
        ]
        if not active:
            return "Нечего отменять."

        cancelled = self._jobs.cancel_all([other.job_id for other in active])
        for other in active:
            if not self._jobs.is_running(other.job_id):
                await self._repos.jobs.finish(other.job_id, "cancelled")
        return "Отменил." if cancelled or active else "Нечего отменять."

    async def _render_reminders(self, user_id: str) -> str:
        user_tz = await self._repos.conversations.timezone_for(user_id)
        reminders = await self._repos.reminders.list(user_id, status="scheduled", limit=30)
        if not reminders:
            return "Напоминаний нет."
        lines = ["*Напоминания*"]
        for reminder in reminders:
            when = format_local(reminder.due_at, user_tz)
            repeat = " (повтор)" if reminder.rrule else ""
            lines.append(f"• {when}{repeat} — {reminder.text}")
        return "\n".join(lines)

    async def _render_tasks(self, user_id: str) -> str:
        user_tz = await self._repos.conversations.timezone_for(user_id)
        tasks = await self._repos.tasks.list(user_id, status="open", limit=30)
        if not tasks:
            return "Открытых задач нет."
        lines = ["*Задачи*"]
        for task in tasks:
            due = f" — до {format_local(task['due_at'], user_tz)}" if task["due_at"] else ""
            owner = f" [{task['owner']}]" if task.get("owner") else ""
            lines.append(f"• {task['title']}{owner}{due}")
        return "\n".join(lines)

    # ---- session control -------------------------------------------------------

    async def reset_session(self, user_id: str) -> str:
        return await self._sessions.reset(user_id)

    async def cancel_job(self, job_id: str) -> bool:
        job = await self._repos.jobs.get(job_id)
        if job is None or job.is_terminal:
            return False
        cancelled = self._jobs.cancel(job_id)
        if not cancelled:
            # Queued but not started: it will be skipped, so it is already effectively cancelled.
            await self._repos.jobs.finish(job_id, "cancelled")
        return True

    async def status(self) -> dict:
        counts = await self._repos.jobs.counts()
        return {
            "core": {
                "instance_id": self._settings.instance_id,
                "uptime_seconds": None,
            },
            "cursor": {"state": self._backend.state, "backend": self._backend.name},
            "stt": {
                "state": "ready" if (self._stt and self._stt.ready) else "idle",
                "model": getattr(self._stt, "model_name", "-"),
            },
            "jobs": {
                "queued": counts["queued"] + self._jobs.queued_count,
                "running": max(counts["running"], self._jobs.running_count),
            },
        }

    # ---- outbound helpers ---------------------------------------------------------

    async def _reply(self, job: Job, text: str) -> None:
        body = (text or "").strip()
        if not body:
            log.info("job %s produced no text", job.job_id)
            body = "Готово."
        if len(body) > _MAX_REPLY_CHARS:
            body = body[:_MAX_REPLY_CHARS] + "\n\n[…ответ обрезан]"

        await self._link.send_event(
            methods.TELEGRAM_SEND,
            methods.dump(
                methods.TelegramSendParams(
                    delivery_id=f"{job.job_id}:reply",
                    user_id=job.user_id,
                    chat_id=job.chat_id or 0,
                    text=body,
                    reply_to_message_id=job.message_id,
                )
            ),
            # Deterministic delivery_id: replaying this job can never produce two messages.
            delivery_id=f"{job.job_id}:reply",
            user_id=job.user_id,
        )

    async def notify(self, user_id: str, chat_id: int, text: str, *, delivery_id: str) -> None:
        """Core-initiated message — reminders, timers and other unsolicited notifications."""
        await self._link.send_event(
            methods.TELEGRAM_SEND,
            methods.dump(
                methods.TelegramSendParams(
                    delivery_id=delivery_id,
                    user_id=user_id,
                    chat_id=chat_id,
                    text=text,
                    kind="notification",
                )
            ),
            delivery_id=delivery_id,
            user_id=user_id,
        )

    async def _progress(
        self, job: Job, stage: str, *, progress: float | None = None, detail: str | None = None
    ) -> None:
        # Progress is advisory and high-frequency, so it is a notification: if the link is down,
        # dropping it is the correct behaviour.
        await self._repos.jobs.set_stage(job.job_id, stage)
        await self._link.notify(
            methods.JOB_PROGRESS,
            methods.dump(
                methods.JobProgressParams(
                    job_id=job.job_id,
                    user_id=job.user_id,
                    chat_id=job.chat_id,
                    stage=stage,
                    progress=progress,
                    detail=detail,
                )
            ),
        )

    async def _fail(self, job: Job, code: str, detail: str) -> None:
        await self._repos.jobs.finish(job.job_id, "failed", error_code=code, error_detail=detail)
        await self._report_failure(job, code, detail)

    async def _report_failure(self, job: Job, code: str, detail: str) -> None:
        await self._link.send_event(
            methods.JOB_FAILED,
            methods.dump(
                methods.JobFailedParams(
                    job_id=job.job_id,
                    user_id=job.user_id,
                    chat_id=job.chat_id,
                    error=methods.JobErrorInfo(code=code, message=detail[:200]),
                    retryable=code in ("agent_unavailable", "not_ready"),
                )
            ),
            delivery_id=f"{job.job_id}:failed",
            user_id=job.user_id,
        )


_CONTROL_COMMANDS = frozenset({"cancel", "reminders", "tasks", "status"})


def _is_control(command: str | None) -> bool:
    """Commands answered from the Core's own state, without going near the agent session."""
    if not command:
        return False
    return command.split()[0].lstrip("/").lower() in _CONTROL_COMMANDS


def _verbatim(message: str, _context: AgentContext) -> str:
    """The transcript prompt already carries its own context and guard rails."""
    return message


def workspace_for(settings, project: str | None) -> Path:
    """Resolve a project name to an allowlisted path.

    Anything not named in ``assistant.toml`` is refused: Cursor may only open directories the user
    has explicitly opted in.
    """
    if project is None:
        return settings.resolved_assistant_workspace
    configured = settings.projects.get(project)
    if configured is None:
        raise PermissionError(f"project {project!r} is not in the allowlist")
    return configured.path
