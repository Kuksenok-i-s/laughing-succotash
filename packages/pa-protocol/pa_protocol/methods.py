"""Method names and validated payload models.

Both peers import these, so a rename cannot drift between the two applications. The models are the
executable form of ``docs/protocol.md``; ``docs/schemas/`` is generated from them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Payload(BaseModel):
    # Forbid unknown fields so a typo in a method call fails loudly at the boundary rather than
    # being silently ignored and producing a subtly wrong result.
    model_config = ConfigDict(extra="forbid")


# --- Gateway-facing method names (Core calls these) ----------------------

TELEGRAM_SEND = "telegram.send"
TELEGRAM_SEND_DOCUMENT = "telegram.send_document"
TELEGRAM_EDIT = "telegram.edit"
TELEGRAM_DELETE = "telegram.delete"
TELEGRAM_ACTION = "telegram.action"
TELEGRAM_CONFIRM = "telegram.confirm"
JOB_PROGRESS = "job.progress"
JOB_COMPLETED = "job.completed"
JOB_FAILED = "job.failed"

# --- Core-facing method names (Gateway calls these) ----------------------

CORE_HELLO = "core.hello"
ASSISTANT_SUBMIT = "assistant.submit"
AUDIO_BEGIN = "audio.begin"
AUDIO_COMMIT = "audio.commit"
AUDIO_ABORT = "audio.abort"
IMAGE_BEGIN = "image.begin"
IMAGE_COMMIT = "image.commit"
IMAGE_ABORT = "image.abort"
JOB_CANCEL = "job.cancel"
CONFIRMATION_RESOLVE = "confirmation.resolve"
SESSION_RESET = "session.reset"
STATUS_GET = "status.get"

Stage = Literal[
    "queued",
    "downloading",
    "transcribing",
    # Same work as `transcribing`, but the GPU host was unusable and the Core fell back to local
    # CPU whisper. A separate stage because the user needs to know why the wait grew.
    "transcribing_cpu",
    "recognizing",
    "recognizing_album",
    "structuring",
    "summarizing",
    "agent",
    "executing_tool",
    "waiting_confirmation",
    "completed",
]

SubmitKind = Literal["text", "command", "transcribe_request"]
AudioPurpose = Literal["assistant", "transcribe_only"]
ImagePurpose = Literal["ocr"]


# --- handshake -----------------------------------------------------------


class CoreHelloParams(_Payload):
    instance_id: str
    protocol_version: int
    last_received_seq: int = 0
    capabilities: list[str] = Field(default_factory=list)


class CoreHelloResult(_Payload):
    gateway_version: str
    protocol_version: int
    last_received_seq: int = 0
    server_time: datetime


# --- Gateway -> Core -----------------------------------------------------


class ReplyContext(_Payload):
    message_id: int
    has_audio: bool = False


ActorKind = Literal["user", "hidden_user", "chat", "channel"]


class TelegramActor(_Payload):
    """A person or chat Telegram attributed content to."""

    kind: ActorKind = "user"
    name: str | None = None
    username: str | None = None
    telegram_user_id: str | None = None
    chat_id: int | None = None
    chat_title: str | None = None


class MessageSource(_Payload):
    """Who originally produced the content, as Telegram reported it.

    ``user_id`` on the parent payload is always the allowlisted person talking to the bot.
    ``author`` is that same person when the message is original, and the original writer when
    the update is a forward.
    """

    forwarded: bool = False
    author: TelegramActor
    date: datetime | None = None
    signature: str | None = None


class AssistantSubmitParams(_Payload):
    request_id: str
    user_id: str
    chat_id: int
    message_id: int
    kind: SubmitKind = "text"
    text: str | None = None
    command: str | None = None
    reply_to: ReplyContext | None = None
    upload_id: str | None = None
    client_time: datetime | None = None
    sender: TelegramActor | None = None
    source: MessageSource | None = None


class AcceptedResult(_Payload):
    job_id: str
    status: Literal["accepted"] = "accepted"
    dedup: bool = False


class AudioBeginParams(_Payload):
    request_id: str
    user_id: str
    chat_id: int
    message_id: int
    filename: str
    content_type: str
    size: int
    duration_seconds: float | None = None
    purpose: AudioPurpose = "assistant"
    sender: TelegramActor | None = None
    source: MessageSource | None = None


class AudioBeginResult(_Payload):
    upload_id: str
    chunk_size: int
    resume_offset: int = 0


class AudioCommitParams(_Payload):
    upload_id: str
    sha256: str
    total_size: int


class AudioAbortParams(_Payload):
    upload_id: str
    reason: str | None = None


class ImageBeginParams(_Payload):
    request_id: str
    user_id: str
    chat_id: int
    message_id: int
    filename: str
    content_type: str
    size: int
    purpose: ImagePurpose = "ocr"
    caption: str | None = None
    # Telegram album: shared id, 0-based index, total parts. All null for a single photo.
    album_id: str | None = None
    part_index: int | None = None
    part_count: int | None = None
    sender: TelegramActor | None = None
    source: MessageSource | None = None


class ImageBeginResult(_Payload):
    upload_id: str
    chunk_size: int
    resume_offset: int = 0


class ImageCommitParams(_Payload):
    upload_id: str
    sha256: str
    total_size: int


class ImageAbortParams(_Payload):
    upload_id: str
    reason: str | None = None


class JobCancelParams(_Payload):
    job_id: str


class JobCancelResult(_Payload):
    cancelled: bool


class ConfirmationResolveParams(_Payload):
    action_id: str
    user_id: str
    choice: str
    resolved_at: datetime | None = None


class ConfirmationResolveResult(_Payload):
    status: Literal["applied", "already_resolved", "expired", "unknown"]


class SessionResetParams(_Payload):
    user_id: str
    request_id: str


class SessionResetResult(_Payload):
    conversation_id: str


class StatusGetParams(_Payload):
    pass


# --- Core -> Gateway -----------------------------------------------------


class TelegramSendParams(_Payload):
    delivery_id: str
    user_id: str
    chat_id: int
    text: str
    parse_mode: Literal["markdown", "plain"] = "markdown"
    reply_to_message_id: int | None = None
    silent: bool = False
    kind: Literal["reply", "status", "notification"] = "reply"


class TelegramSendResult(_Payload):
    message_id: int | None = None
    dedup: bool = False


class TelegramSendDocumentParams(_Payload):
    """A downloadable file. ``content`` is UTF-8 text; the Gateway encodes it for Telegram."""

    delivery_id: str
    user_id: str
    chat_id: int
    filename: str
    content: str
    mime_type: str = "text/markdown"
    caption: str | None = None
    parse_mode: Literal["markdown", "plain"] = "markdown"
    reply_to_message_id: int | None = None
    silent: bool = False


class TelegramSendDocumentResult(_Payload):
    message_id: int | None = None
    dedup: bool = False


class TelegramEditParams(_Payload):
    delivery_id: str
    chat_id: int
    message_id: int
    text: str
    parse_mode: Literal["markdown", "plain"] = "markdown"


class TelegramEditResult(_Payload):
    edited: bool


class TelegramDeleteParams(_Payload):
    delivery_id: str
    chat_id: int
    message_id: int


class TelegramDeleteResult(_Payload):
    deleted: bool


class TelegramActionParams(_Payload):
    chat_id: int
    action: Literal["typing", "record_voice", "upload_document"] = "typing"


class ConfirmAction(_Payload):
    id: str
    label: str
    style: Literal["primary", "secondary", "danger"] = "secondary"


class TelegramConfirmParams(_Payload):
    delivery_id: str
    action_id: str
    user_id: str
    chat_id: int
    text: str
    actions: list[ConfirmAction]
    expires_at: datetime | None = None


class TelegramConfirmResult(_Payload):
    message_id: int | None = None
    dedup: bool = False


class JobProgressParams(_Payload):
    job_id: str
    user_id: str
    chat_id: int | None = None
    stage: Stage
    progress: float | None = None
    detail: str | None = None


class JobCompletedParams(_Payload):
    job_id: str
    user_id: str
    chat_id: int | None = None
    result_kind: str = "text"
    summary_sent: bool = True


class JobErrorInfo(_Payload):
    code: str
    message: str


class JobFailedParams(_Payload):
    job_id: str
    user_id: str
    chat_id: int | None = None
    error: JobErrorInfo
    retryable: bool = False


def dump(model: BaseModel) -> dict[str, Any]:
    """Serialise a payload model for the wire, dropping unset optionals."""
    return model.model_dump(mode="json", exclude_none=True)
