"""YouTube URL detection.

A message that contains a YouTube link is a library job, not chat. The link is classified so a
playlist or channel is not silently reduced to one video, and a watch URL stays a single item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

YoutubeKind = Literal["video", "playlist", "channel"]
YoutubeMode = Literal["transcribe", "download"]

_VIDEO_ID = r"(?P<id>[\w-]{11})"
_PLAYLIST_ID = r"(?P<list>[\w-]{10,})"

_CHANNEL_PATTERNS = (
    re.compile(
        r"https?://(?:www\.)?youtube\.com/(?:channel|c|user)/[^\s/?]+",
        re.IGNORECASE,
    ),
    re.compile(r"https?://(?:www\.)?youtube\.com/@[^\s/?]+", re.IGNORECASE),
)
_PLAYLIST_PATTERN = re.compile(
    rf"https?://(?:www\.)?youtube\.com/playlist\?[^\s]*?list={_PLAYLIST_ID}",
    re.IGNORECASE,
)
_VIDEO_PATTERNS = (
    re.compile(rf"https?://(?:www\.)?youtu\.be/{_VIDEO_ID}", re.IGNORECASE),
    re.compile(rf"https?://(?:www\.)?youtube\.com/watch\?[^\s]*?v={_VIDEO_ID}", re.IGNORECASE),
    re.compile(rf"https?://(?:www\.)?youtube\.com/shorts/{_VIDEO_ID}", re.IGNORECASE),
    re.compile(rf"https?://(?:www\.)?youtube\.com/live/{_VIDEO_ID}", re.IGNORECASE),
    re.compile(rf"https?://(?:www\.)?youtube\.com/embed/{_VIDEO_ID}", re.IGNORECASE),
    re.compile(rf"https?://(?:m\.)?youtube\.com/watch\?[^\s]*?v={_VIDEO_ID}", re.IGNORECASE),
)

_TRANSCRIBE_HINT = re.compile(
    r"конспект|транскрипт|расшифр|саммари|\btranscrib|\bsummary\b",
    re.IGNORECASE,
)
_DOWNLOAD_HINT = re.compile(
    r"скач|выкач|download|\bmp4\b|\b720p\b|\b1080p\b|видеофайл",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class YoutubeLink:
    kind: YoutubeKind
    url: str
    video_id: str | None = None


def extract_youtube_link(text: str) -> YoutubeLink | None:
    for pattern in _CHANNEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return YoutubeLink(kind="channel", url=match.group(0).rstrip(").,]"))
    match = _PLAYLIST_PATTERN.search(text)
    if match:
        return YoutubeLink(
            kind="playlist",
            url=f"https://www.youtube.com/playlist?list={match.group('list')}",
        )
    for pattern in _VIDEO_PATTERNS:
        match = pattern.search(text)
        if match:
            video_id = match.group("id")
            return YoutubeLink(
                kind="video",
                url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
            )
    return None


def extract_video_id(text: str) -> str | None:
    link = extract_youtube_link(text)
    return None if link is None else link.video_id


def extract_youtube_url(text: str) -> str | None:
    link = extract_youtube_link(text)
    return None if link is None else link.url


def youtube_mode_hint(text: str) -> YoutubeMode | None:
    """Explicit words in the same message pick a mode so we do not have to ask."""
    wants_transcript = bool(_TRANSCRIBE_HINT.search(text))
    wants_video = bool(_DOWNLOAD_HINT.search(text))
    if wants_transcript and not wants_video:
        return "transcribe"
    if wants_video and not wants_transcript:
        return "download"
    return None
