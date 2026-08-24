"""The agent abstraction the rest of the Core depends on.

Nothing outside this package imports ACP types. Swapping ``CursorACPBackend`` for
``CursorCLIBackend`` changes transport only — sessions, permissions and job semantics are
unaffected (ADR 3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class Provenance(str, Enum):
    """Where a turn's instruction came from.

    This is what separates "the user told me to create a meeting" from "somebody said the words
    inside a recording". It is assigned by the Core when the turn is built and is never derived
    from model output (ADR 7).
    """

    DIRECT_COMMAND = "direct_command"
    UNTRUSTED_CONTENT = "untrusted_content"


@dataclass(slots=True)
class AgentContext:
    """Everything the agent needs that is not the message itself."""

    user_id: str
    conversation_id: str
    job_id: str | None = None
    timezone: tzinfo | None = None
    now: datetime | None = None
    provenance: Provenance = Provenance.DIRECT_COMMAND
    workspace: Path | None = None
    mode: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolInvocation:
    tool_call_id: str
    title: str
    kind: str
    status: str


@dataclass(slots=True)
class AgentResponse:
    text: str
    stop_reason: str = "end_turn"
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    session_id: str | None = None

    @property
    def cancelled(self) -> bool:
        return self.stop_reason == "cancelled"


class AgentError(RuntimeError):
    """The backend could not complete the turn."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class AgentUnavailable(AgentError):
    """The backend is not running and could not be started."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


# Called as the turn progresses so the Core can drive job.progress without the backend knowing
# anything about the Gateway.
ProgressCallback = Callable[[str, str | None], Awaitable[None]]


class AgentBackend(Protocol):
    async def start(self) -> None: ...

    async def create_session(
        self,
        *,
        workspace: Path,
        context: AgentContext | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str: ...

    async def send_message(
        self,
        session_id: str,
        message: str,
        context: AgentContext | None = None,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResponse: ...

    async def cancel(self, session_id: str) -> None: ...

    async def set_mode(self, session_id: str, mode: str) -> None: ...

    async def close(self) -> None: ...

    @property
    def state(self) -> str: ...

    @property
    def name(self) -> str: ...
