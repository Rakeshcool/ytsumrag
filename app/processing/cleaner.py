"""Transcript cleaning — normalize text while preserving timestamps."""

from __future__ import annotations

import re

from app.models.schemas import Transcript, TranscriptSegment


def clean_transcript(transcript: Transcript) -> Transcript:
    """Clean transcript text without discarding any timestamps."""
    cleaned_segments: list[TranscriptSegment] = []

    for seg in transcript.segments:
        text = seg.text

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove music/sound effect markers like [Music], [Applause]
        pattern = r"\[.*?(?:Music|Applause|Laughter|Sound|noise).*?\]"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove empty artifacts like "()" or "( )"
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\[\s*\]", "", text)

        # Collapse multiple spaces again after removals
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            cleaned_segments.append(
                TranscriptSegment(start=seg.start, end=seg.end, text=text)
            )

    return Transcript(
        video_id=transcript.video_id,
        title=transcript.title,
        language=transcript.language,
        segments=cleaned_segments,
    )
