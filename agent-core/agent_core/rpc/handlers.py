"""Gateway → Core RPC surface.

Two rules apply to everything here.

The Gateway is authenticated as a service, not as a person: a ``user_id`` in a payload is a claim,
and every method re-checks it against the Core's own allowlist. A compromised Gateway can talk to
the Core, but it cannot act as a user the Core does not know.

Handlers return quickly. Whisper and Cursor take minutes; holding an RPC request open for that
long would tie the outcome of the work to the survival of one TCP connection.
"""

from __future__ import annotations

import logging
from typing import Any

from pa_protocol import AudioFrame, RpcError, errors, methods
from pydantic import ValidationError

from ..audio.storage import AudioUploadError

log = logging.getLogger(__name__)


class CoreHandlers:
    def __init__(self, settings, repos, assistant, uploads, confirmations, scheduler) -> None:
        self._settings = settings
        self._repos = repos
        self._assistant = assistant
        self._uploads = uploads
        self._confirmations = confirmations
        self._scheduler = scheduler

    def as_map(self) -> dict[str, Any]:
        return {
            methods.ASSISTANT_SUBMIT: self.assistant_submit,
            methods.AUDIO_BEGIN: self.audio_begin,
            methods.AUDIO_COMMIT: self.audio_commit,
            methods.AUDIO_ABORT: self.audio_abort,
            methods.JOB_CANCEL: self.job_cancel,
            methods.CONFIRMATION_RESOLVE: self.confirmation_resolve,
            methods.SESSION_RESET: self.session_reset,
            methods.STATUS_GET: self.status_get,
        }

    # ---- authorization ---------------------------------------------------

    def _authorize(self, user_id: str) -> None:
        if user_id not in self._settings.allowed_users:
            log.warning("rejecting request for unknown user %s", user_id)
            raise RpcError(errors.UNAUTHORIZED_USER, "unauthorized_user")

    # ---- assistant --------------------------------------------------------

    async def assistant_submit(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.AssistantSubmitParams, raw)
        self._authorize(params.user_id)

        await self._repos.conversations.ensure_user(params.user_id)
        # Remembered so a reminder created now can still be delivered in three hours' time.
        await self._repos.conversations.remember_chat(params.user_id, params.chat_id)

        result = await self._assistant.submit(params)
        return methods.dump(result)

    # ---- audio -------------------------------------------------------------

    async def audio_begin(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.AudioBeginParams, raw)
        self._authorize(params.user_id)

        await self._repos.conversations.ensure_user(params.user_id)
        await self._repos.conversations.remember_chat(params.user_id, params.chat_id)

        if (
            params.duration_seconds
            and params.duration_seconds > self._settings.max_audio_duration_seconds
        ):
            raise RpcError(errors.AUDIO_TOO_LONG, "audio_too_long")

        try:
            upload, resume_offset = await self._uploads.begin(
                request_id=params.request_id,
                user_id=params.user_id,
                chat_id=params.chat_id,
                message_id=params.message_id,
                filename=params.filename,
                content_type=params.content_type,
                size=params.size,
                duration_seconds=params.duration_seconds,
                purpose=params.purpose,
            )
        except AudioUploadError as exc:
            if exc.code == "too_large":
                raise RpcError(errors.AUDIO_TOO_LARGE, "audio_too_large") from exc
            raise RpcError(errors.INTERNAL_ERROR, "internal_error") from exc

        return methods.dump(
            methods.AudioBeginResult(
                upload_id=upload.upload_id,
                chunk_size=256 * 1024,
                resume_offset=resume_offset,
            )
        )

    async def audio_commit(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.AudioCommitParams, raw)
        try:
            upload = await self._uploads.commit(
                params.upload_id, sha256=params.sha256, total_size=params.total_size
            )
        except AudioUploadError as exc:
            code = {
                "unknown_upload": errors.UNKNOWN_UPLOAD,
                "size_mismatch": errors.UPLOAD_INCOMPLETE,
                "checksum_mismatch": errors.UPLOAD_INCOMPLETE,
            }.get(exc.code, errors.INTERNAL_ERROR)
            raise RpcError(code, errors.NAMES.get(code, "upload_failed")) from exc

        self._authorize(upload.user_id)
        job_id = await self._assistant.start_audio_job(upload)
        return methods.dump(methods.AcceptedResult(job_id=job_id))

    async def audio_abort(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.AudioAbortParams, raw)
        await self._uploads.abort(params.upload_id, params.reason)
        return {}

    async def on_binary(self, frame: AudioFrame) -> None:
        await self._uploads.handle_frame(frame)

    # ---- jobs ---------------------------------------------------------------

    async def job_cancel(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.JobCancelParams, raw)
        cancelled = await self._assistant.cancel_job(params.job_id)
        return methods.dump(methods.JobCancelResult(cancelled=cancelled))

    # ---- confirmations ---------------------------------------------------------

    async def confirmation_resolve(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.ConfirmationResolveParams, raw)
        self._authorize(params.user_id)
        status = await self._confirmations.resolve(
            params.action_id, params.user_id, params.choice
        )
        return methods.dump(methods.ConfirmationResolveResult(status=status))

    # ---- sessions -----------------------------------------------------------------

    async def session_reset(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _parse(methods.SessionResetParams, raw)
        self._authorize(params.user_id)
        conversation_id = await self._assistant.reset_session(params.user_id)
        return methods.dump(methods.SessionResetResult(conversation_id=conversation_id))

    async def status_get(self, _raw: dict[str, Any]) -> dict[str, Any]:
        status = await self._assistant.status()
        status["scheduler"] = await self._scheduler.snapshot()
        return status


def _parse(model, raw: dict[str, Any]):
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise RpcError(
            errors.INVALID_PARAMS, "invalid_params", {"detail": exc.errors(include_url=False)[:3]}
        ) from exc
