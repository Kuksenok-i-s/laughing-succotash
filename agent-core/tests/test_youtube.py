from __future__ import annotations

import json
from pathlib import Path

from agent_core.youtube.download import YoutubeDownloader, YoutubeError
from agent_core.youtube.documents import (
    readable_filename,
    readable_media_filename,
    readable_stem,
    readable_title,
    transcript_markdown,
    unique_dir,
    unique_file,
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
    assert readable_media_filename(title, ".mp4") == (
        "Касперская - кибервойна лекция.mp4"
    )
    assert readable_media_filename(title, ".mp4", index=1) == (
        "01 - Касперская - кибервойна лекция.mp4"
    )


def test_unique_dir_does_not_clobber_an_existing_folder(tmp_path) -> None:
    first = unique_dir(tmp_path, "Me at the zoo")
    first.mkdir()
    second = unique_dir(tmp_path, "Me at the zoo")
    assert first == tmp_path / "Me at the zoo"
    assert second == tmp_path / "Me at the zoo (2)"


def test_unique_file_skips_the_path_being_renamed(tmp_path: Path) -> None:
    current = tmp_path / "Me at the zoo [jNQXAC9IVRw].mp4"
    current.write_bytes(b"x")
    taken = tmp_path / "Me at the zoo.mp4"
    taken.write_bytes(b"y")
    assert unique_file(tmp_path, "Me at the zoo.mp4", ignore=current) == (
        tmp_path / "Me at the zoo (2).mp4"
    )
    assert unique_file(tmp_path, current.name, ignore=current) == current


def test_downloaded_videos_are_renamed_to_the_readable_title(tmp_path: Path) -> None:
    dest = tmp_path / "job"
    dest.mkdir()
    ugly = dest / "Касперская： кибервойна [jNQXAC9IVRw].mp4"
    ugly.write_bytes(b"video")
    (dest / "Касперская： кибервойна [jNQXAC9IVRw].info.json").write_text(
        '{"id": "jNQXAC9IVRw", "title": "Касперская: \\"кибервойна\\" / лекция?"}',
        encoding="utf-8",
    )
    downloader = YoutubeDownloader(
        remote="root@host", remote_dir="/root/ytdl", ssh_key=Path("~/.ssh/id")
    )
    library = downloader._parse_video_dir(dest, "video")
    assert [path.name for path in library.files] == [
        "Касперская - кибервойна лекция.mp4"
    ]
    assert library.files[0].is_file()
    assert not list(dest.glob("*.info.json"))
    assert "[" not in library.files[0].name


def test_playlist_videos_are_numbered_without_youtube_ids(tmp_path: Path) -> None:
    dest = tmp_path / "job"
    dest.mkdir()
    items = [
        ("Intro [aaaaaaaaaaa].mp4", "Intro", 1),
        ("Deep dive [bbbbbbbbbbb].mp4", "Deep dive: sockets / TCP", 2),
    ]
    for filename, title, index in items:
        (dest / filename).write_bytes(b"video")
        stem = filename[: -len(".mp4")]
        (dest / f"{stem}.info.json").write_text(
            json.dumps(
                {
                    "id": "x",
                    "title": title,
                    "playlist_title": "Курс по сетям",
                    "playlist_index": index,
                }
            ),
            encoding="utf-8",
        )
    downloader = YoutubeDownloader(
        remote="root@host", remote_dir="/root/ytdl", ssh_key=Path("~/.ssh/id")
    )
    library = downloader._parse_video_dir(dest, "playlist")
    assert library.title == "Курс по сетям"
    assert [path.name for path in library.files] == [
        "01 - Intro.mp4",
        "02 - Deep dive - sockets TCP.mp4",
    ]


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


def test_downloader_refuses_tmp_cookies() -> None:
    try:
        YoutubeDownloader(
            remote="root@host",
            remote_dir="/root/ytdl",
            ssh_key="~/.ssh/id",
            cookies="/tmp/cookies.txt",
        )
    except YoutubeError as exc:
        assert "/tmp" in str(exc)
    else:
        raise AssertionError("expected YoutubeError")


def test_ytdlp_commands_pass_cookies_when_configured() -> None:
    downloader = YoutubeDownloader(
        remote="root@host",
        remote_dir="/root/ytdl",
        ssh_key=Path("~/.ssh/id"),
        cookies="/var/lib/telegram-gateway/youtube/cookies.txt",
    )
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    audio = downloader._audio_remote_cmd(url, "job1", "video")
    video = downloader._video_remote_cmd(url, "job1", "video")
    flag = "--cookies /var/lib/telegram-gateway/youtube/cookies.txt"
    assert flag in audio
    assert flag in video


def test_ytdlp_commands_omit_cookies_when_unset() -> None:
    downloader = YoutubeDownloader(
        remote="root@host", remote_dir="/root/ytdl", ssh_key=Path("~/.ssh/id")
    )
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    assert "--cookies" not in downloader._audio_remote_cmd(url, "job1", "video")
    assert "--cookies" not in downloader._video_remote_cmd(url, "job1", "video")


def test_from_settings_reads_cookies(tmp_path: Path) -> None:
    from agent_core.config import Settings

    key = tmp_path / "id"
    key.write_text("x")
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[paths]\n"
        f'ssh_key = "{key}"\n'
        f'cookies = "{cookies}"\n'
        "[download]\n"
        'remote = "root@host"\n'
        'cookies = "/var/lib/telegram-gateway/youtube/cookies.txt"\n'
    )
    settings = Settings(
        instance_id="test-core",
        gateway_url="ws://localhost/rpc",
        core_token="x" * 40,
        mcp_token="y" * 40,
        allowed_users=["tg:1"],
        data_dir=tmp_path / "data",
        youtube_config=cfg,
    )
    downloader = YoutubeDownloader.from_settings(settings)
    assert downloader is not None
    assert downloader._cookies == "/var/lib/telegram-gateway/youtube/cookies.txt"
    assert downloader._local_cookies == cookies
