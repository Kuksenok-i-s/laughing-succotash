from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_core.agent.base import AgentError
from agent_core.stt.base import TranscriptionResult, TranscriptSegment
from agent_core.youtube.documents import transcript_markdown, timestamp_link
from agent_core.youtube.topics import parse_topics, transcript_topics


def test_topics_and_links_preserve_every_segment():
    result = TranscriptionResult(text="first second", segments=[
        TranscriptSegment(0, 5, "first"), TranscriptSegment(3661.8, 3665, "second")])
    body = transcript_markdown(title="Video", url="https://youtu.be/jNQXAC9IVRw?t=99",
                               transcription=result, topics={0: "Вступление", 1: "Следующая тема"})
    assert "## Темы" in body
    assert "### [1:01:01](https://www.youtube.com/watch?v=jNQXAC9IVRw&t=3661s) — Следующая тема" in body
    assert body.count(" first") == 1
    assert body.count(" second") == 1
    assert "t=99&" not in body


def test_no_segments_does_not_invent_timestamps():
    body = transcript_markdown(title="Video", url="https://youtu.be/jNQXAC9IVRw",
                               transcription=TranscriptionResult(text="Full text"))
    assert "Full text" in body
    assert "&t=" not in body


@pytest.mark.parametrize("url", ["https://youtu.be/jNQXAC9IVRw?t=100",
                                  "https://www.youtube.com/shorts/jNQXAC9IVRw",
                                  "https://www.youtube.com/watch?v=jNQXAC9IVRw&list=PLabc&t=50"])
def test_links_use_original_video_and_segment_time(url):
    assert timestamp_link(url, 65.9) == "[1:05](https://www.youtube.com/watch?v=jNQXAC9IVRw&t=65s)"


def test_parse_topics_rejects_invented_boundaries():
    assert parse_topics('```json\n[{"segment": 3, "title": "Тема"}, '
                        '{"segment": 99, "title": "Нет"}, '
                        '{"segment": true, "title": "Нет"}]\n```', 2, 5) == {3: "Тема"}
    assert parse_topics('not json', 0, 5) == {}


async def test_all_chunks_are_processed_and_failures_preserve_text(tmp_path):
    backend = SimpleNamespace(create_session=AsyncMock(return_value="session"),
                              send_message=AsyncMock(side_effect=[
                                  SimpleNamespace(text='[{"segment":0,"title":"Начало"}]'),
                                  AgentError("unavailable"),
                                  SimpleNamespace(text='[{"segment":2,"title":"Конец"}]')]))
    result = TranscriptionResult(text="", segments=[
        TranscriptSegment(i * 10, i * 10 + 5, f"sentence {i}") for i in range(3)])
    topics = await transcript_topics(backend, tmp_path, None, result, chunk_chars=15)
    assert topics == {0: "Начало", 1: "Фрагмент без тематической разметки", 2: "Конец"}
    assert backend.send_message.await_count == 3
    assert all(call.kwargs["mcp_servers"] == [] for call in backend.create_session.call_args_list)
    body = transcript_markdown(title="Video", url="https://youtu.be/jNQXAC9IVRw",
                               transcription=result, topics=topics)
    for i in range(3):
        assert body.count(f"sentence {i}") == 1
