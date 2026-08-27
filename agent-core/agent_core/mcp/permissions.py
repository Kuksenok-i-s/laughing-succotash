"""The authoritative permission engine (ADR 7).

This runs inside the MCP server, where the tool name is exact and the arguments have already been
validated by Pydantic. The ACP permission callback is a coarse first layer only — it identifies a
tool by a display title and renders arguments as a Markdown code fence, which is not a sound basis
for a security decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from enum import Enum
from typing import Any

from ..agent.base import Provenance


class Tier(str, Enum):
    READ = "read"
    SAFE_WRITE = "safe_write"
    DANGEROUS = "dangerous"


class Decision(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(slots=True)
class ToolContext:
    """Per-turn context handed to a tool call.

    ``provenance`` is the load-bearing field: it records whether this turn came from the user
    speaking directly or from analysing untrusted content, and it is assigned by the Core, never
    inferred from model output.
    """

    user_id: str
    conversation_id: str
    provenance: Provenance = Provenance.DIRECT_COMMAND
    job_id: str | None = None
    chat_id: int | None = None
    timezone: tzinfo | None = None
    now: datetime | None = None

    @property
    def trusted(self) -> bool:
        return self.provenance is Provenance.DIRECT_COMMAND


# Every tool must appear here. An unregistered tool is treated as DANGEROUS by `tier_of`, so
# forgetting to classify a new capability fails closed.
TIERS: dict[str, Tier] = {
    # --- read ---
    "calendar_list": Tier.READ,
    "calendar_get": Tier.READ,
    "calendar_find_free_slots": Tier.READ,
    "task_list": Tier.READ,
    "task_get": Tier.READ,
    "note_search": Tier.READ,
    "note_get": Tier.READ,
    "memory_search": Tier.READ,
    "journal_search": Tier.READ,
    "journal_month": Tier.READ,
    "contact_search": Tier.READ,
    "contact_get": Tier.READ,
    "reminder_list": Tier.READ,
    "reminder_get": Tier.READ,
    "timer_list": Tier.READ,
    "system_status": Tier.READ,
    "system_uptime": Tier.READ,
    "system_cpu": Tier.READ,
    "system_memory": Tier.READ,
    "system_disk": Tier.READ,
    "web_search": Tier.READ,
    "web_fetch": Tier.READ,
    # --- safe write ---
    "reminder_create": Tier.SAFE_WRITE,
    "reminder_update": Tier.SAFE_WRITE,
    "task_create": Tier.SAFE_WRITE,
    "task_update": Tier.SAFE_WRITE,
    "task_complete": Tier.SAFE_WRITE,
    "note_create": Tier.SAFE_WRITE,
    "note_update": Tier.SAFE_WRITE,
    "calendar_create": Tier.SAFE_WRITE,
    "calendar_update": Tier.SAFE_WRITE,
    "timer_create": Tier.SAFE_WRITE,
    "timer_cancel": Tier.SAFE_WRITE,
    "reminder_cancel": Tier.SAFE_WRITE,
    "memory_remember": Tier.SAFE_WRITE,
    "contact_create": Tier.SAFE_WRITE,
    "contact_update": Tier.SAFE_WRITE,
    # --- dangerous ---
    "calendar_delete": Tier.DANGEROUS,
    "task_delete": Tier.DANGEROUS,
    "note_delete": Tier.DANGEROUS,
    "memory_forget": Tier.DANGEROUS,
}


def tier_of(tool_name: str) -> Tier:
    return TIERS.get(tool_name, Tier.DANGEROUS)


def decide(tool_name: str, context: ToolContext) -> Decision:
    """Classify one tool call.

    - READ runs automatically.
    - SAFE_WRITE runs automatically only when the turn came from a direct user instruction. The
      same call derived from a transcript becomes a proposal needing confirmation, which is what
      stops a recording of somebody saying "поставь встречу на пятницу" from booking a meeting.
    - DANGEROUS always asks.
    """
    tier = tier_of(tool_name)
    if tier is Tier.READ:
        return Decision.ALLOW
    if tier is Tier.SAFE_WRITE:
        return Decision.ALLOW if context.trusted else Decision.CONFIRM
    return Decision.CONFIRM


def describe_action(tool_name: str, arguments: dict[str, Any]) -> str:
    """Human-readable Russian text for a confirmation prompt.

    The user must be able to see exactly what they are approving, so this renders the concrete
    arguments rather than a generic "выполнить действие?".
    """
    match tool_name:
        case "calendar_create":
            when = _pretty_range(arguments.get("starts_at"), arguments.get("ends_at"))
            return f"Создать встречу «{arguments.get('title', '')}» {when}?"
        case "calendar_update":
            return f"Изменить встречу «{arguments.get('title') or arguments.get('event_id')}»?"
        case "calendar_delete":
            return f"Удалить встречу {arguments.get('event_id')}?"
        case "reminder_create":
            return (
                f"Создать напоминание «{arguments.get('text', '')}» "
                f"на {_pretty(arguments.get('due_at'))}?"
            )
        case "reminder_cancel":
            return f"Отменить напоминание {arguments.get('reminder_id')}?"
        case "task_create":
            due = arguments.get("due_at")
            suffix = f" (срок {_pretty(due)})" if due else ""
            return f"Создать задачу «{arguments.get('title', '')}»{suffix}?"
        case "task_delete":
            return f"Удалить задачу {arguments.get('task_id')}?"
        case "note_create":
            preview = (arguments.get("body") or "")[:120]
            return f"Сохранить заметку: «{preview}»?"
        case "note_delete":
            return f"Удалить заметку {arguments.get('note_id')}?"
        case "memory_remember":
            return f"Запомнить надолго: «{(arguments.get('content') or '')[:150]}»?"
        case "memory_forget":
            return f"Забыть запись {arguments.get('memory_id')}?"
        case "contact_create":
            return f"Добавить контакт «{arguments.get('display_name', '')}»?"
        case "contact_update":
            return (
                f"Изменить контакт "
                f"«{arguments.get('display_name') or arguments.get('contact_id')}»?"
            )
        case "timer_create":
            return f"Поставить таймер на {arguments.get('duration_seconds', 0) // 60} мин?"
        case _:
            return f"Выполнить действие {tool_name}?"


def _pretty(value: Any) -> str:
    if not value:
        return "—"
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    return parsed.strftime("%d.%m %H:%M")


def _pretty_range(start: Any, end: Any) -> str:
    if not start:
        return ""
    if not end:
        return f"на {_pretty(start)}"
    return f"с {_pretty(start)} до {_pretty(end)}"
