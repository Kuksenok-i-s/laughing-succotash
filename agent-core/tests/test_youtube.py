from __future__ import annotations

from agent_core.youtube.download import YoutubeDownloader, YoutubeError
from agent_core.youtube.documents import (
    readable_filename,
    readable_stem,
    readable_title,
    transcript_markdown,
    unique_dir,
)
from agent_core.youtube.urls import extract_youtube_url
from agent_core.stt.base import TranscriptionResult, TranscriptSegment


def test_watch_shorts_and_share_urls_canonicalize() -> None:
    assert extract_youtube_url("смотри https://youtu.be/jNQXAC9IVRw сейчас") == (
        "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    )
    assert extract_youtube_url(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw&t=12s"
    ) == "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    assert extract_youtube_url("https://youtube.com/shorts/jNQXAC9IVRw") == (
        "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    )
    assert extract_youtube_url("просто текст без ссылки") is None


def test_playlist_and_channel_urls_are_not_collapsed_to_one_video() -> None:
    from agent_core.youtube.urls import extract_youtube_link, youtube_mode_hint

    playlist = extract_youtube_link(
        "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
    )
    assert playlist is not None
    assert playlist.kind == "playlist"
    assert "playlist?list=" in playlist.url

    channel = extract_youtube_link("смотри https://www.youtube.com/@veritasium/videos")
    assert channel is not None
    assert channel.kind == "channel"

    assert youtube_mode_hint("скачай это") == "download"
    assert youtube_mode_hint("сделай конспект") == "transcribe"
    assert youtube_mode_hint("просто ссылка") is None


def test_filenames_use_the_readable_title() -> None:
    title = 'Касперская: "кибервойна" / лекция?'
    assert readable_title(title) == 'Касперская: "кибервойна" / лекция?'
    name = readable_filename(title, "конспект")
    assert name == "Касперская - кибервойна лекция — конспект.md"
    assert "/" not in name
    assert ":" not in name
    assert readable_title("") == "YouTube"
    assert readable_stem('Касперская: "кибервойна" / лекция?') == (
        "Касперская - кибервойна лекция"
    )


def test_unique_dir_does_not_clobber_an_existing_folder(tmp_path) -> None:
    first = unique_dir(tmp_path, "Me at the zoo")
    first.mkdir()
    second = unique_dir(tmp_path, "Me at the zoo")
    assert first == tmp_path / "Me at the zoo"
    assert second == tmp_path / "Me at the zoo (2)"


def test_transcript_markdown_includes_clocks_and_source() -> None:
    result = TranscriptionResult(
        text="привет зоопарк",
        language="en",
        duration=19.0,
        segments=[TranscriptSegment(0.0, 5.0, "hello zoo")],
    )
    body = transcript_markdown(
        title="Me at the zoo", url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        transcription=result,
    )
    assert body.startswith("# Me at the zoo")
    assert "Источник: https://www.youtube.com/watch?v=jNQXAC9IVRw" in body
    assert "[0:00] hello zoo" in body


def test_downloader_refuses_tmp_on_the_proxy() -> None:
    try:
        YoutubeDownloader(remote="root@host", remote_dir="/tmp/ytdl", ssh_key="~/.ssh/id")
    except YoutubeError as exc:
        assert "/tmp" in str(exc)
    else:
        raise AssertionError("expected YoutubeError")
