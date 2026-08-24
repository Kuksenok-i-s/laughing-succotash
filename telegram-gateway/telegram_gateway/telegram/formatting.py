"""Rendering helpers.

The Gateway owns everything Telegram-specific: message splitting, escaping and error wording. The
Core sends intent and plain text.
"""

from __future__ import annotations

import re

TELEGRAM_LIMIT = 4096

_MARKDOWN_V2_SPECIALS = r"_*[]()~`>#+-=|{}.!"

# Human-facing text for protocol error codes. The Core sends stable identifiers; the user reads
# this. Keeping the mapping here means rewording never requires a Core deploy.
ERROR_TEXT = {
    "not_ready": "Ядро сейчас недоступно. Запрос сохранён и будет обработан после переподключения.",
    "unauthorized_user": "У вас нет доступа к этому ассистенту.",
    "rate_limited": "Слишком много запросов. Попробуйте чуть позже.",
    "audio_too_large": "Файл слишком большой.",
    "audio_too_long": "Запись слишком длинная.",
    "upload_incomplete": "Загрузка прервалась. Попробуйте отправить файл ещё раз.",
    "job_not_found": "Задача не найдена — возможно, она уже завершилась.",
    "agent_unavailable": "Cursor Agent сейчас недоступен.",
    "agent_failed": "Cursor Agent не смог обработать запрос.",
    "stt_unavailable": "Распознавание речи сейчас недоступно.",
    "stt_failed": "Не удалось распознать запись.",
    "youtube_download_failed": "Не удалось скачать видео с YouTube.",
    "interrupted": "Обработка прервалась из-за перезапуска ядра.",
}

STAGE_TEXT = {
    "queued": "В очереди…",
    "downloading": "Скачиваю…",
    "transcribing": "Расшифровываю запись…",
    "transcribing_cpu": "GPU недоступен — расшифровываю на CPU, это дольше…",
    "summarizing": "Расшифровка готова. Анализирую…",
    "agent": "Думаю…",
    "executing_tool": "Выполняю действие…",
    "waiting_confirmation": "Жду подтверждения…",
    "completed": "Готово.",
}


def describe_error(code: str, fallback: str = "Что-то пошло не так.") -> str:
    return ERROR_TEXT.get(code, fallback)


def describe_stage(stage: str, detail: str | None = None, progress: float | None = None) -> str:
    """Status line for one job.

    The share done goes first and before the detail: on an hour-long recording the stage text alone
    never changes, and a status message that never changes reads as a dead pipeline.
    """
    text = STAGE_TEXT.get(stage, "Обрабатываю…")
    parts = []
    if progress is not None:
        parts.append(f"{min(max(round(progress * 100), 0), 100)}%")
    if detail:
        parts.append(detail)
    return f"{text} ({' · '.join(parts)})" if parts else text


def escape_markdown_v2(text: str) -> str:
    return re.sub(f"([{re.escape(_MARKDOWN_V2_SPECIALS)}])", r"\\\1", text)


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split text into Telegram-sized parts.

    Prefers paragraph boundaries, then line boundaries, then a hard cut. A fenced code block that
    would be split across parts is reopened in the next part so neither half renders as broken
    Markdown.
    """
    if len(text) <= limit:
        return [text] if text else [""]

    parts: list[str] = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]
        cut = _find_cut(window)
        chunk = remaining[:cut].rstrip()
        parts.append(chunk)
        remaining = remaining[cut:].lstrip("\n")

    if remaining:
        parts.append(remaining)

    return _rebalance_code_fences(parts)


def _find_cut(window: str) -> int:
    for separator in ("\n\n", "\n", ". ", " "):
        index = window.rfind(separator)
        # Refuse a boundary in the first quarter: splitting a 4096-character message at
        # character 40 produces a stream of tiny fragments.
        if index > len(window) // 4:
            return index + len(separator)
    return len(window)


def _rebalance_code_fences(parts: list[str]) -> list[str]:
    balanced: list[str] = []
    carry_language: str | None = None

    for part in parts:
        prefix = f"```{carry_language or ''}\n" if carry_language is not None else ""
        body = prefix + part
        fences = re.findall(r"^```(\w*)", body, flags=re.MULTILINE)
        if len(fences) % 2 == 1:
            carry_language = fences[-1]
            body += "\n```"
        else:
            carry_language = None
        balanced.append(body)

    return balanced


def confirmation_expired_notice() -> str:
    return "Срок подтверждения истёк."
