"""Persistent reply keyboard: tap a command instead of typing a slash.

Telegram sends the button label as a normal text message, so handlers must map labels to the
same actions as ``/new``, ``/status`` and the rest. The keyboard is optional — ``Скрыть кнопки``
removes it, ``/keyboard`` puts it back.
"""

from __future__ import annotations

from aiogram.types import (
    BotCommand,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

HIDE = "Скрыть кнопки"

# Label -> command name (no slash). Keep labels short: they occupy the composer on a phone.
BUTTON_COMMANDS: dict[str, str] = {
    "Новый разговор": "new",
    "Отмена": "cancel",
    "Напоминания": "reminders",
    "Задачи": "tasks",
    "Дневник": "journal",
    "Статус": "status",
    "Справка": "help",
}

BOT_COMMANDS = [
    BotCommand(command="new", description="Начать новый разговор"),
    BotCommand(command="cancel", description="Отменить текущую задачу"),
    BotCommand(command="status", description="Состояние системы"),
    BotCommand(command="reminders", description="Список напоминаний"),
    BotCommand(command="tasks", description="Список задач"),
    BotCommand(command="journal", description="Дневник за сегодня"),
    BotCommand(command="transcribe", description="Только расшифровка (ответом на аудио)"),
    BotCommand(command="keyboard", description="Показать кнопки"),
    BotCommand(command="help", description="Справка"),
]


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Новый разговор"), KeyboardButton(text="Отмена")],
            [KeyboardButton(text="Напоминания"), KeyboardButton(text="Задачи")],
            [KeyboardButton(text="Дневник"), KeyboardButton(text="Статус")],
            [KeyboardButton(text="Справка")],
            [KeyboardButton(text=HIDE)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Напишите или нажмите кнопку",
    )


def hide_keyboard() -> ReplyKeyboardRemove:
    return ReplyKeyboardRemove()
