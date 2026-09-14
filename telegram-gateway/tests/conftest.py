"""Gateway fixtures, including a fake Bot API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from telegram_gateway.config import Settings
from telegram_gateway.storage.database import Database
from telegram_gateway.storage.models import GatewayStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="123:fake",
        core_token="c" * 40,
        allowed_users=["tg:1"],
        data_dir=tmp_path / "gw",
        status_edit_min_interval=0.0,
        delivery_max_attempts=3,
    )


@pytest.fixture
async def db(settings: Settings) -> Database:
    database = Database(settings.resolved_database_path)
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def store(db: Database) -> GatewayStore:
    return GatewayStore(db)


@dataclass
class SentMessage:
    chat_id: int
    text: str
    message_id: int
    parse_mode: str | None = None
    reply_markup: object | None = None


class FakeBot:
    """Just enough Bot API for the renderer, with scripted failures."""

    def __init__(self) -> None:
        self.sent: list[SentMessage] = []
        self.documents: list[SentDocument] = []
        self.edits: list[tuple[int, int, str]] = []
        self.deleted: list[tuple[int, int]] = []
        self.actions: list[tuple[int, str]] = []
        self._next_id = 100
        # Exception to raise on the next send, to model Telegram refusing.
        self.fail_next: Exception | None = None

    async def send_message(
        self, chat_id: int, text: str, parse_mode=None, reply_markup=None, **kwargs
    ) -> SentMessage:
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        self._next_id += 1
        message = SentMessage(chat_id, text, self._next_id, parse_mode, reply_markup)
        self.sent.append(message)
        return message

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, **kwargs) -> None:
        self.edits.append((chat_id, message_id, text))

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        self.deleted.append((chat_id, message_id))

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))

    async def send_document(
        self, chat_id: int, document, caption=None, parse_mode=None, **kwargs
    ) -> SentMessage:
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        filename = getattr(document, "filename", "file")
        data = getattr(document, "data", None)
        if data is None:
            data = bytes(document) if not hasattr(document, "read") else document.read()
        self._next_id += 1
        record = SentDocument(chat_id, filename, data, caption, self._next_id)
        self.documents.append(record)
        message = SentMessage(chat_id, caption or "", self._next_id, parse_mode, None)
        self.sent.append(message)
        return message

    def texts(self) -> list[str]:
        return [message.text for message in self.sent]


@dataclass
class SentDocument:
    chat_id: int
    filename: str
    content: bytes
    caption: str | None
    message_id: int


@pytest.fixture
def bot() -> FakeBot:
    return FakeBot()
