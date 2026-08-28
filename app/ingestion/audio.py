"""Audio download from YouTube for Whisper transcription."""

from __future__ import annotations

import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)

_AUDIO_DIR = Path("data/audio")


def download_audio(video_id: str, url: str) -> Path:
    """Download audio-only from YouTube and return the file path."""
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    output_path = _AUDIO_DIR / f"{video_id}.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    logger.info("Downloading audio for %s…", video_id)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_file = _AUDIO_DIR / f"{video_id}.mp3"
    if not audio_file.exists():
        # Try other extensions
        for ext in ["wav", "opus", "webm", "m4a"]:
            candidate = _AUDIO_DIR / f"{video_id}.{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Audio download succeeded but no output file found for {video_id}"
        )

    return audio_file
