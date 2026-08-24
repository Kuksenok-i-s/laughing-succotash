"""Chunking and hierarchical analysis of long transcripts."""

from __future__ import annotations

from agent_core.agent.base import AgentError, AgentResponse, Provenance
from agent_core.assistant.transcript import TranscriptAnalyzer, split_transcript
from agent_core.stt.base import TranscriptionResult, TranscriptSegment

from .conftest import FakeBackend, context_for


def transcription(segment_count: int, words: int = 20) -> TranscriptionResult:
    segments = [
        TranscriptSegment(i * 15.0, (i + 1) * 15.0, " ".join(["слово"] * words))
        for i in range(segment_count)
    ]
    return TranscriptionResult(
        text=" ".join(s.text for s in segments),
        language="ru",
        duration=segment_count * 15.0,
        segments=segments,
    )


def test_a_short_transcript_stays_in_one_chunk() -> None:
    chunks = split_transcript(transcription(2), chunk_chars=10000)
    assert len(chunks) == 1


def test_splitting_never_cuts_a_segment_in_half() -> None:
    result = transcription(30)
    chunks = split_transcript(result, chunk_chars=500)

    assert len(chunks) > 1
    rejoined = "\n".join(chunk.text for chunk in chunks)
    for segment in result.segments:
        assert segment.text in rejoined


def test_chunks_stay_in_order_and_carry_timestamps() -> None:
    chunks = split_transcript(transcription(30), chunk_chars=500)

    assert [chunk.index for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].start < chunks[-1].start
    assert "[0:00]" in chunks[0].text


def test_a_transcript_with_no_segments_still_produces_a_chunk() -> None:
    """Some backends return text without timings; the pipeline must not fall over."""
    result = TranscriptionResult(text="просто текст", duration=5.0, segments=[])
    chunks = split_transcript(result, chunk_chars=100)

    assert len(chunks) == 1
    assert "просто текст" in chunks[0].text


async def test_a_single_chunk_is_passed_through_verbatim(tmp_path) -> None:
    """No point paying for an extraction pass over text that already fits."""
    backend = FakeBackend()
    analyzer = TranscriptAnalyzer(backend, workspace=tmp_path, chunk_chars=10000)

    analysis = await analyzer.analyze(transcription(2), context_for())

    assert backend.prompts == []
    assert analysis.notes == ""
    assert "слово" in analysis.excerpt


async def test_each_chunk_is_analysed_separately_and_merged_in_order(tmp_path) -> None:
    seen: list[str] = []

    def respond(message, _context):
        seen.append(message)
        index = message.split("Фрагмент ")[1].split(" ")[0]
        return AgentResponse(text=f"РЕШЕНИЯ: решение {index}")

    backend = FakeBackend(on_prompt=respond)
    analyzer = TranscriptAnalyzer(backend, workspace=tmp_path, chunk_chars=500)

    analysis = await analyzer.analyze(
        transcription(30), context_for(provenance=Provenance.UNTRUSTED_CONTENT)
    )

    assert analysis.chunk_count > 1
    assert analysis.notes.index("решение 1") < analysis.notes.index("решение 2")
    assert all("transcript of a recording" in prompt for prompt in seen)


async def test_one_failed_chunk_does_not_lose_the_rest(tmp_path) -> None:
    calls = {"n": 0}

    def flaky(_message, _context):
        calls["n"] += 1
        if calls["n"] == 2:
            raise AgentError("cursor hiccup")
        return AgentResponse(text="ЗАДАЧИ: что-то")

    backend = FakeBackend(on_prompt=flaky)
    analyzer = TranscriptAnalyzer(backend, workspace=tmp_path, chunk_chars=500)

    analysis = await analyzer.analyze(transcription(30), context_for())

    assert analysis.failures == [2]
    assert "ЗАДАЧИ" in analysis.notes
    # The gap is surfaced rather than hidden, so the agent can tell the user.
    assert "Не удалось разобрать фрагменты: 2" in analysis.notes


async def test_progress_is_reported_per_chunk(tmp_path) -> None:
    fractions: list[float] = []

    async def record(fraction: float) -> None:
        fractions.append(fraction)

    analyzer = TranscriptAnalyzer(FakeBackend(), workspace=tmp_path, chunk_chars=500)
    await analyzer.analyze(transcription(30), context_for(), on_progress=record)

    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
