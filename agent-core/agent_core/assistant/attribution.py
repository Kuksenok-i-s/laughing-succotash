"""Parse Telegram attribution stored on a job into the shape the prompts use."""

from __future__ import annotations

from typing import Any

from pa_protocol import methods
from pydantic import ValidationError

from ..agent.base import MessageAttribution


def from_payload(payload: dict[str, Any] | None, *, owner_id: str) -> MessageAttribution | None:
    if not payload:
        return None
    raw = payload.get("source")
    if not raw:
        return None
    try:
        source = methods.MessageSource.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return None
    author = source.author
    author_id = author.telegram_user_id
    is_owner = (not source.forwarded) or (author_id is not None and author_id == owner_id)
    return MessageAttribution(
        forwarded=source.forwarded,
        is_owner=is_owner,
        author_kind=author.kind,
        author_name=author.name,
        author_username=author.username,
        author_telegram_user_id=author_id,
        author_chat_title=author.chat_title,
    )


def dump_fields(
    sender: methods.TelegramActor | None,
    source: methods.MessageSource | None,
) -> dict[str, Any]:
    return {
        "sender": methods.dump(sender) if sender is not None else None,
        "source": methods.dump(source) if source is not None else None,
    }


def dump_from_upload(attribution: dict[str, Any] | None) -> dict[str, Any]:
    if not attribution:
        return {"sender": None, "source": None}
    return {"sender": attribution.get("sender"), "source": attribution.get("source")}
