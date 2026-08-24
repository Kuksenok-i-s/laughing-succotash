"""Download YouTube media via the proxy VPS.

YouTube is blocked from the Mac mini and the GPU host, so yt-dlp runs on the gateway proxy over
SSH. The remote working directory is never ``/tmp`` — that host treats ``/tmp`` as off-limits.

Two jobs share this helper:

* audio-only, for transcription
* real video files (one watch URL, a playlist, or a channel), archived onto the Core disk
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .urls import YoutubeKind

log = logging.getLogger(__name__)

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
_AUDIO_SUFFIXES = {".mp3", ".m4a", ".opus", ".ogg", ".wav", ".webm"}


class YoutubeError(RuntimeError):
    """Download or probe failed. The job fails; the Core stays up."""


@dataclass(slots=True)
class YoutubeMedia:
    url: str
    video_id: str
    title: str
    duration: float | None
    audio_path: Path
    index: int | None = None


@dataclass(slots=True)
class YoutubeAudioBatch:
    dest: Path
    items: list[YoutubeMedia] = field(default_factory=list)
    title: str = "YouTube"
    kind: YoutubeKind = "video"


@dataclass(slots=True)
class YoutubeLibrary:
    dest: Path
    files: list[Path] = field(default_factory=list)
    title: str = "YouTube"
    kind: YoutubeKind = "video"


def _refuse_tmp(remote_dir: str) -> None:
    normalized = remote_dir.rstrip("/") or remote_dir
    if normalized == "/tmp" or normalized.startswith("/tmp/"):
        raise YoutubeError("refusing to use /tmp on the proxy host")


class YoutubeDownloader:
    def __init__(
        self,
        *,
        remote: str,
        remote_dir: str,
        ssh_key: Path,
        audio_format: str = "mp3",
        video_format: str = "bv*[height<=1080]+ba/b[height<=1080]/b",
        transcripts_dir: Path | None = None,
        videos_dir: Path | None = None,
        max_videos: int = 40,
    ) -> None:
        _refuse_tmp(remote_dir)
        self._remote = remote
        self._remote_dir = remote_dir.rstrip("/")
        self._ssh_key = ssh_key.expanduser()
        self._audio_format = audio_format
        self._video_format = video_format
        self.transcripts_dir = transcripts_dir
        self.videos_dir = videos_dir
        self.max_videos = max(1, max_videos)

    @classmethod
    def from_settings(cls, settings) -> YoutubeDownloader | None:
        path = settings.resolved_youtube_config
        if not path.exists():
            log.info("no youtube config at %s; YouTube URLs will be refused", path)
            return None
        with path.open("rb") as handle:
            cfg = tomllib.load(handle)
        download = cfg.get("download") or {}
        remote = download.get("remote")
        if not remote:
            return None
        paths = cfg.get("paths") or {}
        key = Path(paths.get("ssh_key") or "")
        if not key:
            log.warning("youtube config has download.remote but no paths.ssh_key")
            return None
        base = Path(paths.get("base") or (settings.resolved_data_dir / "youtube"))
        return cls(
            remote=str(remote),
            remote_dir=str(download.get("remote_dir") or "/root/ytdl"),
            ssh_key=key,
            transcripts_dir=Path(paths.get("transcripts") or (base / "transcripts")),
            videos_dir=Path(paths.get("videos") or (base / "videos")),
            max_videos=int(download.get("max_videos") or 40),
            video_format=str(
                download.get("format") or "bv*[height<=1080]+ba/b[height<=1080]/b"
            ),
        )

    def _ssh(self) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self._ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            self._remote,
        ]

    def _pull_dir(self, remote_job: str, dest_dir: Path) -> None:
        """Copy the remote job directory onto the Core.

        OpenSSH 9+ ``scp`` speaks SFTP and rejects ``host:dir/.`` as ``not a regular file``.
        A tar stream over the existing SSH session does not hit that.
        """
        quoted_job = shlex.quote(remote_job)
        archive = subprocess.run(
            [*self._ssh(), f"tar -C {quoted_job} -cf - ."],
            capture_output=True,
        )
        if archive.returncode != 0:
            err = (archive.stderr or archive.stdout or b"tar failed").decode("utf-8", "replace")
            raise YoutubeError(err.strip()[:500])
        extract = subprocess.run(
            ["tar", "-C", str(dest_dir), "-xf", "-"],
            input=archive.stdout,
            capture_output=True,
        )
        if extract.returncode != 0:
            err = (extract.stderr or b"tar extract failed").decode("utf-8", "replace")
            raise YoutubeError(err.strip()[:500])

    def _run_remote(
        self,
        remote_cmd: str,
        dest_dir: Path,
        job_id: str,
        *,
        ok_codes: tuple[int, ...] = (0,),
    ) -> None:
        _refuse_tmp(self._remote_dir)
        remote_job = f"{self._remote_dir}/{job_id}"
        quoted_job = shlex.quote(remote_job)
        dest_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [*self._ssh(), remote_cmd],
            text=True,
            capture_output=True,
        )
        if proc.returncode not in ok_codes:
            err = (proc.stderr or proc.stdout or "yt-dlp failed").strip()
            raise YoutubeError(err[:500])
        try:
            self._pull_dir(remote_job, dest_dir)
        finally:
            subprocess.run(
                [*self._ssh(), f"rm -rf {quoted_job}"],
                text=True,
                capture_output=True,
            )

    async def fetch(self, url: str, dest_dir: Path, *, job_id: str) -> YoutubeMedia:
        batch = await self.fetch_audio(url, dest_dir, job_id=job_id, kind="video")
        return batch.items[0]

    async def fetch_audio(
        self,
        url: str,
        dest_dir: Path,
        *,
        job_id: str,
        kind: YoutubeKind = "video",
    ) -> YoutubeAudioBatch:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(self._fetch_audio_batch, url, dest_dir, job_id, kind)

    def _fetch_audio_batch(
        self, url: str, dest_dir: Path, job_id: str, kind: YoutubeKind
    ) -> YoutubeAudioBatch:
        remote_job = f"{self._remote_dir}/{job_id}"
        quoted_job = shlex.quote(remote_job)
        quoted_url = shlex.quote(url)
        fmt = shlex.quote(self._audio_format)
        playlist_flag = "--no-playlist" if kind == "video" else "--yes-playlist"
        max_flag = "" if kind == "video" else f"--max-downloads {self.max_videos} "
        ignore_flag = "" if kind == "video" else "--ignore-errors "
        quoted_out = shlex.quote(f"{remote_job}/%(title)s [%(id)s].%(ext)s")
        # Audio only: the Core transcribes; video would only fill the proxy disk.
        # The yt-dlp template must be quoted: bash otherwise treats %(id)s as a subshell.
        remote_cmd = (
            f"mkdir -p {quoted_job} && "
            f"yt-dlp {playlist_flag} {max_flag}{ignore_flag}"
            f"-f ba/bestaudio/best -x --audio-format {fmt} "
            f"--write-info-json --no-progress "
            f"-o {quoted_out} {quoted_url}"
        )
        ok_codes = (0,) if kind == "video" else (0, 101)
        self._run_remote(remote_cmd, dest_dir, job_id, ok_codes=ok_codes)
        batch = self._parse_audio_dir(dest_dir, url, kind)
        log.info(
            "downloaded %d youtube audio file(s) (%s) -> %s",
            len(batch.items),
            batch.title,
            dest_dir,
        )
        return batch

    def _parse_audio_dir(
        self, dest_dir: Path, fallback_url: str, kind: YoutubeKind
    ) -> YoutubeAudioBatch:
        items: list[YoutubeMedia] = []
        collection = dest_dir.name
        for info_path in sorted(dest_dir.rglob("*.info.json")):
            info = json.loads(info_path.read_text(encoding="utf-8"))
            stem = info_path.name[: -len(".info.json")]
            audio = next(
                (
                    path
                    for path in info_path.parent.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in _AUDIO_SUFFIXES
                    and path.stem == stem
                ),
                None,
            )
            if audio is None:
                continue
            title = str(info.get("title") or stem)
            video_id = str(info.get("id") or stem)
            raw_duration = info.get("duration")
            duration = float(raw_duration) if raw_duration is not None else None
            raw_index = info.get("playlist_index") or info.get("playlist_autonumber")
            index = int(raw_index) if raw_index is not None else None
            webpage = str(info.get("webpage_url") or info.get("original_url") or fallback_url)
            if kind == "video":
                collection = title
            else:
                collection = str(
                    info.get("playlist_title")
                    or info.get("uploader")
                    or info.get("channel")
                    or collection
                )
            items.append(
                YoutubeMedia(
                    url=webpage,
                    video_id=video_id,
                    title=title,
                    duration=duration,
                    audio_path=audio,
                    index=index,
                )
            )
        items.sort(key=lambda item: (item.index is None, item.index or 0, item.title))
        if not items:
            raise YoutubeError("yt-dlp produced no audio file")
        return YoutubeAudioBatch(dest=dest_dir, items=items, title=collection, kind=kind)

    async def fetch_video(
        self,
        url: str,
        dest_dir: Path,
        *,
        job_id: str,
        kind: YoutubeKind = "video",
    ) -> YoutubeLibrary:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(self._fetch_video, url, dest_dir, job_id, kind)

    def _fetch_video(
        self, url: str, dest_dir: Path, job_id: str, kind: YoutubeKind
    ) -> YoutubeLibrary:
        remote_job = f"{self._remote_dir}/{job_id}"
        quoted_job = shlex.quote(remote_job)
        quoted_url = shlex.quote(url)
        playlist_flag = "--no-playlist" if kind == "video" else "--yes-playlist"
        max_flag = "" if kind == "video" else f"--max-downloads {self.max_videos} "
        quoted_fmt = shlex.quote(self._video_format)
        quoted_out = shlex.quote(f"{remote_job}/%(title)s [%(id)s].%(ext)s")
        remote_cmd = (
            f"mkdir -p {quoted_job} && "
            f"yt-dlp {playlist_flag} {max_flag}"
            f"-f {quoted_fmt} "
            f"--merge-output-format mp4 --write-info-json --no-progress --ignore-errors "
            f"-o {quoted_out} {quoted_url}"
        )
        # yt-dlp exits 101 when --max-downloads is reached; that is the cap working, not a failure.
        self._run_remote(remote_cmd, dest_dir, job_id, ok_codes=(0, 101))

        files = sorted(
            path
            for path in dest_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
        )
        if not files:
            raise YoutubeError("yt-dlp produced no video file")

        title = dest_dir.name
        info_path = next(dest_dir.rglob("*.info.json"), None)
        if info_path is not None:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            title = str(
                info.get("playlist_title")
                or info.get("uploader")
                or info.get("channel")
                or info.get("title")
                or title
            )

        log.info("downloaded %d youtube video(s) (%s) -> %s", len(files), title, dest_dir)
        return YoutubeLibrary(dest=dest_dir, files=files, title=title, kind=kind)
