"""The renderer turns Core intent into Bot API calls — and must never do it twice."""

from __future__ import annotations

import pytest
from aiogram.exceptions import TelegramForbiddenError
from pa_protocol import RpcError, errors, methods, new_ulid

from telegram_gateway.telegram.renderer import TelegramRenderer


@pytest.fixture
def renderer(bot, store, settings) -> TelegramRenderer:
    return TelegramRenderer(bot, store, settings)


def send_params(**overrides) -> dict:
    payload = {
        "delivery_id": new_ulid(),
        "user_id": "tg:1",
        "chat_id": 500,
        "text": "Готово.",
    }
    payload.update(overrides)
    return payload


async def test_a_send_reaches_telegram(renderer, bot) -> None:
    result = await renderer.send(send_params(text="Завтра созвон в 15:00"))

    assert bot.texts() == ["Завтра созвон в 15:00"]
    assert result["message_id"] == bot.sent[0].message_id
    assert result["dedup"] is False


async def test_a_replayed_delivery_does_not_send_twice(renderer, bot) -> None:
    """At-least-once delivery means the Core will resend after a reconnect."""
    params = send_params(text="Напоминание")

    first = await renderer.send(params)
    second = await renderer.send(params)

    assert bot.texts() == ["Напоминание"]
    assert second["dedup"] is True
    assert second["message_id"] == first["message_id"]


async def test_a_long_reply_is_delivered_in_parts(renderer, bot, settings) -> None:
    text = "\n\n".join(f"Пункт {i}: " + "детали " * 50 for i in range(60))

    await renderer.send(send_params(text=text))

    assert len(bot.sent) > 1
    # Only the first part quotes the user's message; the rest would be noise.
    assert all(message.chat_id == 500 for message in bot.sent)


async def test_a_failed_send_is_released_so_the_core_can_retry(renderer, bot, store) -> None:
    params = send_params()
    bot.fail_next = RuntimeError("telegram is down")

    with pytest.raises(RpcError) as excinfo:
        await renderer.send(params)
    assert excinfo.value.code == errors.TELEGRAM_SEND_FAILED

    # The claim was released, so the retry actually sends rather than being swallowed as a dupe.
    result = await renderer.send(params)
    assert result["dedup"] is False
    assert len(bot.sent) == 1


async def test_a_blocked_bot_is_a_permanent_failure(renderer, bot) -> None:
    """The Core must stop retrying: this will never succeed."""
    bot.fail_next = TelegramForbiddenError(method=None, message="bot was blocked by the user")

    with pytest.raises(RpcError) as excinfo:
        await renderer.send(send_params())

    assert excinfo.value.code == errors.TELEGRAM_BLOCKED


async def test_a_confirmation_renders_buttons_bound_to_opaque_tokens(
    renderer, bot, store
) -> None:
    action_id = new_ulid()
    result = await renderer.confirm(
        {
            "delivery_id": new_ulid(),
            "action_id": action_id,
            "user_id": "tg:1",
            "chat_id": 500,
            "text": "Создать встречу завтра с 15:00 до 16:00?",
            "actions": [
                {"id": "approve", "label": "Создать"},
                {"id": "reject", "label": "Отмена"},
            ],
        }
    )

    assert result["message_id"] is not None
    keyboard = bot.sent[0].reply_markup.inline_keyboard[0]
    assert [button.text for button in keyboard] == ["Создать", "Отмена"]

    # Callback data carries a random token, never the action id: it is attacker-visible and
    # capped at 64 bytes.
    for button in keyboard:
        assert action_id not in button.callback_data
        record = await store.resolve_confirmation_token(button.callback_data[2:])
        assert record["action_id"] == action_id


async def test_youtube_mode_buttons_keep_their_choice_ids(renderer, bot, store) -> None:
    action_id = new_ulid()
    await renderer.confirm(
        {
            "delivery_id": new_ulid(),
            "action_id": action_id,
            "user_id": "tg:1",
            "chat_id": 500,
            "text": "Ссылка на YouTube. Что сделать?",
            "actions": [
                {"id": "transcribe", "label": "Конспект"},
                {"id": "download", "label": "Скачать видео"},
                {"id": "reject", "label": "Отмена"},
            ],
        }
    )

    keyboard = bot.sent[0].reply_markup.inline_keyboard[0]
    assert [button.text for button in keyboard] == ["Конспект", "Скачать видео", "Отмена"]
    choices = []
    for button in keyboard:
        record = await store.resolve_confirmation_token(button.callback_data[2:])
        choices.append(record["choice"])
    assert choices == ["transcribe", "download", "reject"]


async def test_a_replayed_confirmation_does_not_show_a_second_keyboard(renderer, bot) -> None:
    params = {
        "delivery_id": new_ulid(),
        "action_id": new_ulid(),
        "user_id": "tg:1",
        "chat_id": 500,
        "text": "Удалить?",
        "actions": [{"id": "approve", "label": "Да"}],
    }

    await renderer.confirm(params)
    second = await renderer.confirm(params)

    assert len(bot.sent) == 1
    assert second["dedup"] is True


async def test_progress_updates_edit_a_single_status_message(renderer, bot) -> None:
    job_id = new_ulid()

    await renderer.job_progress(
        {"job_id": job_id, "user_id": "tg:1", "chat_id": 500, "stage": "transcribing"}
    )
    await renderer.job_progress(
        {"job_id": job_id, "user_id": "tg:1", "chat_id": 500, "stage": "summarizing"}
    )

    assert bot.texts() == ["Расшифровываю запись…"]
    assert bot.edits[-1][2] == "Разбираю…"


async def test_advancing_progress_edits_the_same_stage(renderer, bot) -> None:
    """Without this an hour-long transcription shows one unchanging line for the whole hour."""
    job_id = new_ulid()
    for fraction in (None, 0.23, 0.61):
        await renderer.job_progress(
            {
                "job_id": job_id,
                "user_id": "tg:1",
                "chat_id": 500,
                "stage": "transcribing",
                "progress": fraction,
                "detail": "Касперская",
            }
        )

    assert bot.texts() == ["Расшифровываю запись… (Касперская)"]
    assert [edit[2] for edit in bot.edits] == [
        "Расшифровываю запись… (23% · Касперская)",
        "Расшифровываю запись… (61% · Касперская)",
    ]


async def test_a_repeated_identical_stage_is_not_re_edited(renderer, bot) -> None:
    """Telegram rejects a no-op edit; sending it anyway just burns rate limit."""
    job_id = new_ulid()
    for _ in range(3):
        await renderer.job_progress(
            {"job_id": job_id, "user_id": "tg:1", "chat_id": 500, "stage": "agent"}
        )

    assert len(bot.sent) == 1
    assert bot.edits == []


async def test_the_status_message_is_removed_once_the_reply_is_sent(renderer, bot) -> None:
    job_id = new_ulid()
    await renderer.job_progress(
        {"job_id": job_id, "user_id": "tg:1", "chat_id": 500, "stage": "agent"}
    )

    await renderer.job_completed({"job_id": job_id, "user_id": "tg:1", "chat_id": 500})

    assert bot.deleted == [(500, bot.sent[0].message_id)]


async def test_a_failure_replaces_the_status_message_with_a_reason(renderer, bot) -> None:
    job_id = new_ulid()
    await renderer.job_progress(
        {"job_id": job_id, "user_id": "tg:1", "chat_id": 500, "stage": "transcribing"}
    )

    await renderer.job_failed(
        {
            "job_id": job_id,
            "user_id": "tg:1",
            "chat_id": 500,
            "error": {"code": "stt_failed", "message": "no speech"},
        }
    )

    assert bot.edits[-1][2] == "Не удалось распознать запись."


async def test_a_failure_without_a_status_message_still_reaches_the_user(renderer, bot) -> None:
    await renderer.job_failed(
        {
            "job_id": new_ulid(),
            "user_id": "tg:1",
            "chat_id": 500,
            "error": {"code": "agent_unavailable", "message": "down"},
        }
    )

    assert bot.texts() == ["Cursor Agent сейчас недоступен."]


def document_params(**overrides) -> dict:
    payload = {
        "delivery_id": new_ulid(),
        "user_id": "tg:1",
        "chat_id": 500,
        "filename": "Касперская — конспект.md",
        "content": "# Касперская\n\n## Основные тезисы\n1. Тезис.\n",
        "caption": "Конспект",
    }
    payload.update(overrides)
    return payload


async def test_a_document_reaches_telegram_with_its_filename(renderer, bot) -> None:
    result = await renderer.send_document(document_params())

    assert len(bot.documents) == 1
    sent = bot.documents[0]
    assert sent.filename == "Касперская — конспект.md"
    assert sent.caption == "Конспект"
    assert "Основные тезисы" in sent.content.decode("utf-8")
    assert result["dedup"] is False
    assert result["message_id"] == sent.message_id


async def test_a_replayed_document_is_not_sent_twice(renderer, bot) -> None:
    params = document_params()
    await renderer.send_document(params)
    second = await renderer.send_document(params)
    assert len(bot.documents) == 1
    assert second["dedup"] is True
