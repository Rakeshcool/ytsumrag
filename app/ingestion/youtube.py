"""YouTube URL handling — extract video_id, metadata, and coordinates ingestion."""

from __future__ import annotations

import logging
import re
from typing import Any

import yt_dlp

from app.models.schemas import VideoMetadata

logger = logging.getLogger(__name__)

_YT_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)"
    r"([a-zA-Z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    """Extract a YouTube video ID from a URL."""
    match = _YT_URL_PATTERN.search(url)
    return match.group(1) if match else None


def fetch_metadata(url: str) -> VideoMetadata:
    """Use yt-dlp to extract video metadata without downloading."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info: dict[str, Any] = ydl.extract_info(url, download=False)

    return VideoMetadata(
        video_id=video_id,
        url=url,
        title=info.get("title"),
        duration=info.get("duration"),
        language=info.get("language"),
        thumbnail_url=info.get("thumbnail"),
    )
