"""Readable markdown documents for a YouTube transcription.

Telegram shows the filename on the download chip, so the stem is the video title — never a ULID.
"""

from __future__ import annotations

import re

from pathlib import Path

from ..stt.base import TranscriptionResult

_CONTROL = re.compile(r"[\x00-\x1f]")
_FILENAME_UNSAFE = re.compile(r'[\\/:*?"<>|]')
_TG_MARKDOWN = re.compile(r"[*_`\[\\]")
_SPACES = re.compile(r"\s+")


def readable_title(title: str, *, fallback: str = "YouTube") -> str:
    """Human-readable title for chat and document headings. Keeps punctuation such as ':'."""
    cleaned = _SPACES.sub(" ", _CONTROL.sub(" ", title)).strip(" .")
    return cleaned or fallback


def telegram_title(title: str) -> str:
    """Same title, stripped of Telegram Markdown metacharacters so italics wrapping is safe."""
    cleaned = _SPACES.sub(" ", _TG_MARKDOWN.sub("", readable_title(title))).strip(" .")
    return cleaned or "YouTube"


def readable_stem(title: str, *, max_stem: int = 80) -> str:
    """Directory-safe title stem: colons become ' -', path punctuation is stripped."""
    stem = readable_title(title).replace(":", " -")
    stem = _SPACES.sub(" ", _FILENAME_UNSAFE.sub(" ", stem)).strip(" .")
    if len(stem) > max_stem:
        stem = stem[:max_stem].rstrip(" .")
    return stem or "YouTube"


def readable_filename(title: str, suffix: str, *, max_stem: int = 80) -> str:
    return f"{readable_stem(title, max_stem=max_stem)} — {suffix}.md"


def unique_dir(root: Path, stem: str) -> Path:
    candidate = root / stem
    index = 2
    while candidate.exists():
        candidate = root / f"{stem} ({index})"
        index += 1
    return candidate


def format_duration(seconds: float | None) -> str | None:
    if not seconds:
        return None
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    if hours and minutes:
        return f"{hours} ч {minutes} мин"
    if hours:
        return f"{hours} ч"
    return f"{max(minutes, 1)} мин"


def transcript_markdown(
    *,
    title: str,
    url: str,
    transcription: TranscriptionResult,
) -> str:
    body = transcription.with_timestamps() or transcription.text
    lines = [f"# {readable_title(title)}", "", f"Источник: {url}"]
    duration = format_duration(transcription.duration)
    if duration:
        lines.append(f"Длительность: {duration}")
    if transcription.language:
        lines.append(f"Язык: {transcription.language}")
    lines.extend(["", "## Транскрипт", "", body.strip(), ""])
    return "\n".join(lines)


def summary_markdown(
    *,
    title: str,
    url: str,
    duration: float | None,
    body: str,
) -> str:
    heading = f"# {readable_title(title)}"
    text = (body or "").strip()
    meta = [heading, "", f"Источник: {url}"]
    formatted = format_duration(duration)
    if formatted:
        meta.append(f"Длительность: {formatted}")
    meta.append("")
    if text.startswith("#") and "Источник:" in text[:800]:
        return text if text.endswith("\n") else text + "\n"
    return "\n".join(meta) + text + "\n"


def save_library_text(directory: Path, filename: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    body = content if content.endswith("\n") else content + "\n"
    path.write_text(body, encoding="utf-8")
    return path
