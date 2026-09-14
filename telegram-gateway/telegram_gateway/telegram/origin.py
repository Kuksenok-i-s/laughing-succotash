"""Who a Telegram update is from, and who originally wrote the content.

The bot only talks to an allowlisted user, so ``from_user`` is almost always that person.
A forward still has that user as the sender; the original author lives on ``forward_origin``
(Bot API 7+) or the legacy ``forward_*`` fields. The Core needs both: sender identity for
the user record, original author so a forwarded "поставь встречу" is not treated as an
instruction.
"""

from __future__ import annotations

from datetime import datetime

from aiogram.types import Chat, Message, User
from pa_protocol import methods

_HIDDEN = "hidden_user"
_CHANNEL = "channel"
_CHAT = "chat"
_USER = "user"


def attribution_of(message: Message) -> tuple[methods.TelegramActor, methods.MessageSource]:
    """Return (who sent this update, who originally wrote it)."""
    sender = actor_from_user(message.from_user) or methods.TelegramActor(kind=_USER)
    origin = _from_forward_origin(message)
    if origin is None:
        origin = _from_legacy_forward(message)
    if origin is not None:
        author, date, signature = origin
        return sender, methods.MessageSource(
            forwarded=True, author=author, date=date, signature=signature
        )
    if message.sender_chat is not None:
        return sender, methods.MessageSource(
            forwarded=False, author=actor_from_chat(message.sender_chat)
        )
    return sender, methods.MessageSource(forwarded=False, author=sender)


def actor_from_user(user: User | None) -> methods.TelegramActor | None:
    if user is None:
        return None
    return methods.TelegramActor(
        kind=_USER,
        name=_person_name(user.first_name, user.last_name, user.username, user.id),
        username=user.username,
        telegram_user_id=f"tg:{user.id}",
    )


def actor_from_chat(chat: Chat, *, kind: str | None = None) -> methods.TelegramActor:
    resolved = kind or (_CHANNEL if chat.type == "channel" else _CHAT)
    title = (chat.title or "").strip() or None
    name = title or _person_name(chat.first_name, chat.last_name, chat.username, chat.id)
    return methods.TelegramActor(
        kind=resolved,  # type: ignore[arg-type]
        name=name,
        username=chat.username,
        chat_id=chat.id,
        chat_title=title or name,
    )


def _from_forward_origin(
    message: Message,
) -> tuple[methods.TelegramActor, datetime | None, str | None] | None:
    origin = message.forward_origin
    if origin is None:
        return None
    date = getattr(origin, "date", None)
    origin_type = getattr(origin, "type", None)
    if origin_type == _USER:
        author = actor_from_user(getattr(origin, "sender_user", None))
        if author is None:
            return None
        return author, date, None
    if origin_type == _HIDDEN:
        name = (getattr(origin, "sender_user_name", None) or "").strip() or None
        return (
            methods.TelegramActor(kind=_HIDDEN, name=name),
            date,
            None,
        )
    if origin_type == _CHAT:
        chat = getattr(origin, "sender_chat", None)
        if chat is None:
            return None
        signature = getattr(origin, "author_signature", None)
        return actor_from_chat(chat), date, signature
    if origin_type == _CHANNEL:
        chat = getattr(origin, "chat", None)
        if chat is None:
            return None
        signature = getattr(origin, "author_signature", None)
        return actor_from_chat(chat, kind=_CHANNEL), date, signature
    return None


def _from_legacy_forward(
    message: Message,
) -> tuple[methods.TelegramActor, datetime | None, str | None] | None:
    date = message.forward_date
    signature = message.forward_signature
    if message.forward_from is not None:
        author = actor_from_user(message.forward_from)
        if author is None:
            return None
        return author, date, signature
    if message.forward_from_chat is not None:
        kind = _CHANNEL if message.forward_from_chat.type == "channel" else _CHAT
        return actor_from_chat(message.forward_from_chat, kind=kind), date, signature
    name = (message.forward_sender_name or "").strip() or None
    if name is None:
        return None
    return methods.TelegramActor(kind=_HIDDEN, name=name), date, signature


def _person_name(
    first_name: str | None, last_name: str | None, username: str | None, fallback_id: int
) -> str:
    parts = [part for part in (first_name, last_name) if part]
    if parts:
        return " ".join(parts)
    if username:
        return username
    return str(fallback_id)
