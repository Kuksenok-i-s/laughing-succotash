"""Telegram attribution: who sent the update vs who originally wrote it."""

from __future__ import annotations

from datetime import datetime, timezone

from aiogram.types import Chat, Message, MessageOriginChannel, MessageOriginHiddenUser, MessageOriginUser, User

from telegram_gateway.telegram.origin import attribution_of

CHAT = Chat(id=777, type="private")
OWNER = User(id=1, is_bot=False, first_name="Илья", username="ilya")


def _message(**fields) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=CHAT,
        from_user=OWNER,
        **fields,
    )


def test_an_ordinary_message_is_from_the_user() -> None:
    sender, source = attribution_of(_message(text="привет"))

    assert sender.name == "Илья"
    assert sender.username == "ilya"
    assert sender.telegram_user_id == "tg:1"
    assert source.forwarded is False
    assert source.author.telegram_user_id == "tg:1"
    assert source.author.name == "Илья"


def test_a_forward_from_a_user_keeps_the_original_author() -> None:
    origin = MessageOriginUser(
        type="user",
        date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        sender_user=User(
            id=42, is_bot=False, first_name="Маша", last_name="Иванова", username="masha"
        ),
    )
    sender, source = attribution_of(_message(text="поставь встречу", forward_origin=origin))

    assert sender.telegram_user_id == "tg:1"
    assert source.forwarded is True
    assert source.author.kind == "user"
    assert source.author.name == "Маша Иванова"
    assert source.author.username == "masha"
    assert source.author.telegram_user_id == "tg:42"


def test_a_hidden_forward_keeps_the_display_name() -> None:
    origin = MessageOriginHiddenUser(
        type="hidden_user",
        date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        sender_user_name="Секрет",
    )
    _sender, source = attribution_of(_message(text="привет", forward_origin=origin))

    assert source.forwarded is True
    assert source.author.kind == "hidden_user"
    assert source.author.name == "Секрет"
    assert source.author.telegram_user_id is None


def test_a_channel_forward_keeps_the_channel_title() -> None:
    origin = MessageOriginChannel(
        type="channel",
        date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        chat=Chat(id=-100123, type="channel", title="Новости", username="news"),
        message_id=9,
        author_signature="Редактор",
    )
    _sender, source = attribution_of(_message(text="заголовок", forward_origin=origin))

    assert source.forwarded is True
    assert source.author.kind == "channel"
    assert source.author.name == "Новости"
    assert source.author.username == "news"
    assert source.author.chat_id == -100123
    assert source.signature == "Редактор"


def test_legacy_forward_from_still_works() -> None:
    _sender, source = attribution_of(
        _message(
            text="старый API",
            forward_from=User(id=7, is_bot=False, first_name="Пётр"),
            forward_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
    )

    assert source.forwarded is True
    assert source.author.name == "Пётр"
    assert source.author.telegram_user_id == "tg:7"
