"""YouTube caption/subtitle extraction."""

from __future__ import annotations

import logging
import re
from html import unescape

import yt_dlp

from app.models.schemas import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


def _parse_timed_text(xml_text: str) -> list[TranscriptSegment]:
    """Parse YouTube's timed text XML into TranscriptSegments."""
    segments: list[TranscriptSegment] = []

    # Match <p> or <text> elements with start and dur attributes
    # YouTube captions use <p t="start_ms" d="dur_ms"> or <text start="s" dur="s">
    p_pattern = re.compile(
        r'<p\s[^>]*?t="(\d+)"[^>]*?d="(\d+)"[^>]*?>(.*?)</p>',
        re.DOTALL,
    )
    text_pattern = re.compile(
        r'<text\s[^>]*?start="([\d.]+)"[^>]*?dur="([\d.]+)"[^>]*?>(.*?)</text>',
        re.DOTALL,
    )

    matches = p_pattern.findall(xml_text) or text_pattern.findall(xml_text)

    for start_raw, dur_raw, inner in matches:
        # Clean inner HTML
        clean = re.sub(r"<[^>]+>", "", inner)
        clean = unescape(clean).strip()
        if not clean:
            continue

        # Handle ms vs s
        if p_pattern.search(xml_text):
            start = int(start_raw) / 1000.0
            end = (int(start_raw) + int(dur_raw)) / 1000.0
        else:
            start = float(start_raw)
            end = float(start_raw) + float(dur_raw)

        segments.append(TranscriptSegment(start=start, end=end, text=clean))

    return segments


def fetch_captions(video_id: str, url: str) -> Transcript | None:
    """Attempt to fetch captions for a YouTube video.

    Tries: manual subs → auto-generated subs → None.
    """
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subformat": "json3",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        logger.warning("yt-dlp failed to extract info for %s: %s", video_id, exc)
        return None

    if info is None:
        return None

    title = info.get("title")
    subs: dict = info.get("subtitles") or {}
    auto_subs: dict = info.get("automatic_captions") or {}

    # Try manual subs first, then auto
    chosen_lang = None
    sub_data = None
    for lang in ["en", "en-US", "en-GB"]:
        if lang in subs:
            chosen_lang = lang
            sub_data = subs[lang]
            break
    if sub_data is None:
        for lang in ["en", "en-US", "en-GB"]:
            if lang in auto_subs:
                chosen_lang = lang
                sub_data = auto_subs[lang]
                break

    if sub_data is None:
        logger.info("No captions found for %s.", video_id)
        return None

    # Find a json3 or srv3 format
    caption_url = None
    for fmt in sub_data:
        if fmt.get("ext") == "json3":
            caption_url = fmt["url"]
            break
    if caption_url is None and sub_data:
        caption_url = sub_data[0].get("url")

    if not caption_url:
        return None

    # Download and parse
    import httpx

    resp = httpx.get(caption_url, timeout=30.0)
    resp.raise_for_status()
    content = resp.text

    # Parse json3 format
    import json

    try:
        data = json.loads(content)
        segments: list[TranscriptSegment] = []
        for event in data.get("events", []):
            if "segs" not in event:
                continue
            start_ms = event.get("tStartMs", 0)
            dur_ms = event.get("dDurationMs", 0)
            text_parts = [seg.get("utf8", "") for seg in event["segs"]]
            text = "".join(text_parts).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=start_ms / 1000.0,
                    end=(start_ms + dur_ms) / 1000.0,
                    text=text,
                )
            )

        if segments:
            return Transcript(
                video_id=video_id,
                title=title,
                language=chosen_lang,
                segments=segments,
            )
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: parse as XML
    segments = _parse_timed_text(content)
    if segments:
        return Transcript(
            video_id=video_id,
            title=title,
            language=chosen_lang,
            segments=segments,
        )

    return None
