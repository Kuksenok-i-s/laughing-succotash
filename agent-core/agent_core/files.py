"""User-sandbox files the assistant may create and hand to Telegram.

Chat sessions run in Cursor ``plan`` mode, so ACP Write is blocked. These helpers are what the
MCP ``file_*`` tools call: write inside ``DATA_DIR/user_{tg_id}`` and deliver the result as
``telegram.send_document``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pa_protocol import methods, new_ulid

if TYPE_CHECKING:
    from .mcp.permissions import ToolContext

log = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 400_000
MAX_READ_CHARS = 80_000
MAX_STEM = 80

_CONTROL = re.compile(r"[\x00-\x1f]")
_UNSAFE = re.compile(r'[\\/:*?"<>|]')
_SPACES = re.compile(r"\s+")

# UTF-8 documents Telegram can usefully download. Binary formats (xlsx, pdf, png) are out of
# scope: the protocol carries text, and the agent cannot assemble those anyway.
ALLOWED_SUFFIXES = frozenset(
    {
        ".md",
        ".txt",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".html",
        ".htm",
        ".ics",
        ".vcf",
        ".svg",
        ".toml",
        ".ini",
        ".log",
        ".sql",
        ".diff",
        ".patch",
        ".rst",
        ".org",
        ".css",
    }
)

_MIME = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".html": "text/html",
    ".htm": "text/html",
    ".ics": "text/calendar",
    ".vcf": "text/vcard",
    ".svg": "image/svg+xml",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".log": "text/plain",
    ".sql": "application/sql",
    ".diff": "text/x-diff",
    ".patch": "text/x-diff",
    ".rst": "text/x-rst",
    ".org": "text/org",
    ".css": "text/css",
}


def sanitize_filename(raw: str) -> str:
    """Return a basename safe to write under the user workspace and to show in Telegram.

    Path components, control characters and Windows-forbidden punctuation are stripped. A missing
    extension becomes ``.md``. Anything outside ``ALLOWED_SUFFIXES`` is rejected so the agent
    cannot pretend it produced a PDF.
    """
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = _CONTROL.sub("", name)
    name = _UNSAFE.sub(" ", name)
    name = _SPACES.sub(" ", name).strip(" .")
    if not name or name in {".", ".."}:
        raise ValueError("filename is empty or invalid")

    path = Path(name)
    suffix = path.suffix.lower()
    stem = path.stem.strip(" .") or "file"
    if len(stem) > MAX_STEM:
        stem = stem[:MAX_STEM].rstrip(" .") or "file"
    if not suffix:
        suffix = ".md"
    elif suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ValueError(
            f"unsupported file type {suffix}; send a UTF-8 document ({allowed})"
        )
    return f"{stem}{suffix}"


def mime_for(filename: str) -> str:
    return _MIME.get(Path(filename).suffix.lower(), "text/plain")


_FILE_REQUEST = re.compile(
    r"(?i)("
    r"\bфайл(?:ом|а|е|ы)?\b|"
    r"маркдаун|markdown|"
    r"\.(?:md|csv|txt|json|ics|ya?ml)\b|"
    r"\bcsv\b|"
    r"выгрузк|экспорт|"
    r"сохрани\s+(?:это\s+)?(?:в\s+)?файл|"
    r"отч[её]т\s+в\b"
    r")"
)


def looks_like_file_request(text: str) -> bool:
    """True when the user asked for a downloadable file, not a chat reply."""
    return bool(_FILE_REQUEST.search(text or ""))


def suggested_filename(user_text: str) -> str:
    lower = (user_text or "").lower()
    if re.search(r"\bcsv\b|\.csv\b|таблиц", lower):
        return "данные.csv"
    if re.search(r"\bjson\b|\.json\b", lower):
        return "данные.json"
    return "отчёт.md"


class FileDelivery:
    """Write a text file into the per-user sandbox and send it as a Telegram document."""

    def __init__(
        self,
        link,
        workspace_for: Callable[[str], Path],
        *,
        max_content_chars: int = MAX_CONTENT_CHARS,
    ) -> None:
        self._link = link
        self._workspace_for = workspace_for
        self._max_content_chars = max_content_chars
        self._sent_jobs: set[str] = set()

    def sent_for(self, job_id: str | None) -> bool:
        return bool(job_id) and job_id in self._sent_jobs

    def list_files(self, user_id: str) -> dict[str, Any]:
        root = self._workspace_for(user_id)
        items: list[dict[str, Any]] = []
        if root.is_dir():
            for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not path.is_file() or path.name.startswith("."):
                    continue
                items.append(
                    {
                        "filename": path.name,
                        "bytes": path.stat().st_size,
                    }
                )
        return {"files": items, "count": len(items)}

    def read_file(self, user_id: str, filename: str) -> dict[str, Any]:
        path = self._resolve(user_id, filename)
        if path is None or not path.is_file():
            return {"error": "not found"}
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"error": "file is not valid UTF-8"}
        truncated = len(text) > MAX_READ_CHARS
        if truncated:
            text = text[:MAX_READ_CHARS]
        return {
            "filename": path.name,
            "content": text,
            "truncated": truncated,
            "bytes": path.stat().st_size,
        }

    async def send(
        self,
        ctx: ToolContext,
        *,
        filename: str,
        content: str | None,
        caption: str | None,
        operation_id: str | None,
    ) -> dict[str, Any]:
        name = sanitize_filename(filename)
        root = self._workspace_for(ctx.user_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / name

        if content is None:
            if not path.is_file():
                raise ValueError(f"file {name!r} does not exist; pass content to create it")
            try:
                body = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("existing file is not valid UTF-8") from exc
        else:
            body = self._validate_content(content)
            written = body if body.endswith("\n") else body + "\n"
            path.write_text(written, encoding="utf-8")
            body = written

        delivery_id = operation_id or new_ulid()
        if ctx.job_id:
            delivery_id = f"{ctx.job_id}:file:{delivery_id}"

        chat_id = ctx.chat_id or 0
        if chat_id:
            await self._link.notify(
                methods.TELEGRAM_ACTION,
                methods.dump(
                    methods.TelegramActionParams(chat_id=chat_id, action="upload_document")
                ),
            )

        await self._link.send_event(
            methods.TELEGRAM_SEND_DOCUMENT,
            methods.dump(
                methods.TelegramSendDocumentParams(
                    delivery_id=delivery_id,
                    user_id=ctx.user_id,
                    chat_id=chat_id,
                    filename=name,
                    content=body,
                    mime_type=mime_for(name),
                    caption=(caption or "").strip() or None,
                    reply_to_message_id=ctx.message_id,
                )
            ),
            delivery_id=delivery_id,
            user_id=ctx.user_id,
        )
        if ctx.job_id:
            self._sent_jobs.add(ctx.job_id)
        log.info("sent file %s (%d bytes) for %s", name, path.stat().st_size, ctx.user_id)
        return {
            "filename": name,
            "bytes": path.stat().st_size,
            "mime_type": mime_for(name),
            "sent": True,
        }

    def _validate_content(self, content: str) -> str:
        if "\x00" in content:
            raise ValueError("file content must not contain NUL bytes")
        if not content.strip():
            raise ValueError("file content is empty")
        if len(content) > self._max_content_chars:
            raise ValueError(
                f"file is too large ({len(content)} chars; max {self._max_content_chars})"
            )
        return content

    def _resolve(self, user_id: str, filename: str) -> Path | None:
        try:
            name = sanitize_filename(filename)
        except ValueError:
            return None
        root = self._workspace_for(user_id)
        path = (root / name).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return None
        return path
