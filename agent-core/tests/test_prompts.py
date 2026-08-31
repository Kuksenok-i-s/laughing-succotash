"""Prompt builders for YouTube конспект: two-pass factcheck then the downloadable file."""

from agent_core.assistant import prompts

from .conftest import context_for


def test_youtube_factcheck_asks_for_search_and_forbids_the_file() -> None:
    text = prompts.youtube_factcheck(
        "Лекция",
        "ТЕМЫ: угрозы\nЦИФРЫ: 40%",
        context_for(),
        duration_seconds=600,
    )
    assert "Это первый ход" in text
    assert "Не пиши конспект" in text
    assert "Не больше 8 запросов" in text
    assert "https://полный-url" in text
    assert "<transcript_analysis>" in text
    assert "40%" in text


def test_youtube_summary_embeds_factcheck_and_requires_links() -> None:
    text = prompts.youtube_summary(
        "Лекция",
        "ТЕМЫ: угрозы",
        context_for(),
        factcheck="1. «40%»\n   статус: подтверждено\n   источники:\n   - ENISA — https://example.com/enisa",
    )
    assert "Это второй ход" in text
    assert "<factcheck>" in text
    assert "https://example.com/enisa" in text
    assert "## Фактчек" in text
    assert "[название](https://полный-url)" in text
    assert "Поиск больше не вызывай" in text
    assert "Тезисы — речь спикера" in text


def test_youtube_summary_omits_factcheck_section_when_the_first_pass_failed() -> None:
    text = prompts.youtube_summary("Лекция", "", context_for(), excerpt="просто текст")
    assert "Фактчека нет" in text
    assert "<factcheck>" not in text
    assert "<transcript_excerpt>" in text
    assert "просто текст" in text
