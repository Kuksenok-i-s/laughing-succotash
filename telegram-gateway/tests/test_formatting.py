"""Message rendering. A long assistant reply must arrive complete and readable."""

from __future__ import annotations

from telegram_gateway.telegram.formatting import (
    describe_error,
    describe_stage,
    escape_markdown_v2,
    split_message,
)


def test_a_short_message_is_not_split() -> None:
    assert split_message("привет") == ["привет"]


def test_a_long_message_is_split_within_the_limit() -> None:
    text = "\n\n".join(f"Абзац номер {i}. " + "текст " * 40 for i in range(40))
    parts = split_message(text, limit=1000)

    assert len(parts) > 1
    assert all(len(part) <= 1000 for part in parts)


def test_splitting_loses_no_words() -> None:
    text = "\n\n".join(f"Пункт {i}: важные данные." for i in range(200))
    parts = split_message(text, limit=500)

    rejoined = " ".join(parts)
    for i in range(200):
        assert f"Пункт {i}:" in rejoined


def test_splitting_prefers_paragraph_boundaries() -> None:
    text = ("А" * 400) + "\n\n" + ("Б" * 400)
    parts = split_message(text, limit=500)

    assert parts[0] == "А" * 400
    assert parts[1] == "Б" * 400


def test_a_code_block_split_across_parts_is_reopened() -> None:
    """Otherwise the first half renders as an unterminated fence and the second as plain text."""
    code = "\n".join(f"line_{i} = {i}" for i in range(120))
    text = f"Вот код:\n\n```python\n{code}\n```"

    parts = split_message(text, limit=600)

    assert len(parts) > 1
    for part in parts:
        assert part.count("```") % 2 == 0
    assert parts[1].startswith("```python")


def test_a_line_with_no_spaces_is_cut_hard_rather_than_dropped() -> None:
    text = "x" * 2000
    parts = split_message(text, limit=500)

    assert "".join(parts) == text


def test_markdown_v2_specials_are_escaped() -> None:
    assert escape_markdown_v2("a_b*c[d]") == r"a\_b\*c\[d\]"
    assert escape_markdown_v2("2026-08-24.") == r"2026\-08\-24\."


def test_error_codes_become_human_text() -> None:
    assert "недоступно" in describe_error("not_ready")
    assert describe_error("something_new") == "Что-то пошло не так."


def test_stage_text_describes_the_current_work() -> None:
    assert describe_stage("transcribing") == "Расшифровываю запись…"
    assert describe_stage("summarizing").startswith("Расшифровка готова")
    assert describe_stage("executing_tool", "calendar_create").endswith("(calendar_create)")
    assert describe_stage("unknown_stage") == "Обрабатываю…"
