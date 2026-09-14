"""Long-recording windows and transcript stitching."""

from agent_core.stt.base import TranscriptionResult, TranscriptSegment
from agent_core.stt.chunks import chunk_windows, merge_results


def test_a_short_file_is_a_single_window() -> None:
    assert chunk_windows(500, 600, 2) == [(0.0, 500.0)]


def test_a_long_file_overlaps_the_joins() -> None:
    windows = chunk_windows(1300, 600, 2)

    assert windows[0] == (0.0, 600.0)
    assert windows[1] == (598.0, 600.0)
    assert windows[2] == (1196.0, 104.0)


def test_later_chunks_drop_the_overlapped_head() -> None:
    first = TranscriptionResult(
        text="начало край",
        language="ru",
        duration=600.0,
        segments=[
            TranscriptSegment(0.0, 10.0, "начало"),
            TranscriptSegment(590.0, 600.0, "край"),
        ],
    )
    second = TranscriptionResult(
        text="повтор дальше",
        language="ru",
        duration=102.0,
        segments=[
            TranscriptSegment(0.0, 3.0, "повтор"),
            TranscriptSegment(5.0, 20.0, "дальше"),
        ],
    )

    merged = merge_results([(0.0, first), (598.0, second)], overlap_seconds=2.0)

    assert [item.text for item in merged.segments] == ["начало", "край", "дальше"]
    assert merged.text == "начало край дальше"
    assert merged.duration == 700.0
