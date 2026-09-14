"""YouTube ingestion: URL detection, proxy download, readable transcript documents."""

from .download import (
    YoutubeAudioBatch,
    YoutubeDownloader,
    YoutubeError,
    YoutubeLibrary,
    YoutubeMedia,
)
from .urls import YoutubeLink, extract_youtube_link, extract_youtube_url, youtube_mode_hint

__all__ = [
    "YoutubeAudioBatch",
    "YoutubeDownloader",
    "YoutubeError",
    "YoutubeLibrary",
    "YoutubeLink",
    "YoutubeMedia",
    "extract_youtube_link",
    "extract_youtube_url",
    "youtube_mode_hint",
]
