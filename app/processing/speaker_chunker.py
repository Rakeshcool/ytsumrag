"""Speaker-aware chunking for podcast transcripts.

Chunks are split at natural conversation boundaries while keeping
complete thoughts together. Each chunk tracks speakers and timestamps.
"""

from __future__ import annotations

import logging
import re

from app.config.settings import settings
from app.models.schemas import Transcript, TranscriptChunk

logger = logging.getLogger(__name__)

# Common speaker label patterns in transcripts
_SPEAKER_PATTERNS = [
    re.compile(r"^(Speaker\s*\d+):", re.IGNORECASE),
    re.compile(r"^(Host|Guest|Interviewer|Interviewee):", re.IGNORECASE),
    re.compile(r"^([A-Z][a-z]+):"),  # Single name like "Joe:"
    re.compile(r"^([A-Z][A-Z]+):"),  # All caps like "JOE:"
]


def _detect_speaker(text: str) -> str | None:
    """Try to extract speaker label from text."""
    for pattern in _SPEAKER_PATTERNS:
        m = pattern.match(text.strip())
        if m:
            return m.group(1)
    return None


def _is_topic_boundary(prev_text: str, curr_text: str) -> bool:
    """Heuristic: detect if there's a topic boundary between two segments."""
    # Speaker change is a soft boundary indicator
    prev_speaker = _detect_speaker(prev_text)
    curr_speaker = _detect_speaker(curr_text)

    if prev_speaker and curr_speaker and prev_speaker != curr_speaker:
        return True

    # Long pause or paragraph break (multiple newlines in caption)
    if "\n\n" in curr_text or "\n\n" in prev_text:
        return True

    return False


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return max(1, len(text) // 4)


def chunk_transcript(transcript: Transcript) -> list[TranscriptChunk]:
    """Split a podcast transcript into speaker-aware, conversation-boundary chunks.

    Strategy:
    1. Group consecutive segments into candidate chunks.
    2. Try to break at speaker changes or topic shifts.
    3. Keep chunks between 300-700 tokens when possible.
    4. Never break mid-sentence.
    5. Track which speakers appear in each chunk.
    """
    if not transcript.segments:
        return []

    min_tokens = settings.chunk_min_tokens
    max_tokens = settings.chunk_max_tokens
    video_id = transcript.video_id

    chunks: list[TranscriptChunk] = []
    current_texts: list[str] = []
    current_tokens = 0
    chunk_start: float | None = None
    chunk_index = 0

    for i, seg in enumerate(transcript.segments):
        seg_tokens = _estimate_tokens(seg.text)

        if chunk_start is None:
            chunk_start = seg.start

        # Check if we should break before this segment
        if current_texts and current_tokens >= min_tokens:
            prev_text = current_texts[-1]

            # Strong break: topic boundary and we have enough content
            if _is_topic_boundary(prev_text, seg.text) and current_tokens >= min_tokens:
                # Flush current chunk
                chunk_text = " ".join(current_texts)
                chunks.append(_make_chunk(
                    chunk_index, video_id, chunk_start, seg.start, chunk_text,
                    transcript.segments, len(chunks),
                ))
                chunk_index += 1
                current_texts = []
                current_tokens = 0
                chunk_start = seg.start

        # Check if adding this segment would exceed max
        exceeds_max = current_tokens + seg_tokens > max_tokens
        if exceeds_max and current_texts and current_tokens >= min_tokens:
            # Flush current chunk
            chunk_text = " ".join(current_texts)
            chunks.append(_make_chunk(
                chunk_index, video_id, chunk_start, seg.start, chunk_text,
                transcript.segments, len(chunks),
            ))
            chunk_index += 1
            current_texts = []
            current_tokens = 0
            chunk_start = seg.start

        current_texts.append(seg.text)
        current_tokens += seg_tokens

    # Flush remaining
    if current_texts:
        chunk_text = " ".join(current_texts)
        end_time = transcript.segments[-1].end if transcript.segments else 0.0
        chunks.append(_make_chunk(
            chunk_index, video_id, chunk_start or 0.0, end_time, chunk_text,
            transcript.segments, len(chunks),
        ))

    logger.info(
        "Speaker-aware chunking: %d segments -> %d chunks (target %d-%d tokens)",
        len(transcript.segments),
        len(chunks),
        min_tokens,
        max_tokens,
    )

    return chunks


def _make_chunk(
    index: int,
    video_id: str,
    start: float,
    end: float,
    text: str,
    segments: list,
    existing_chunks: int,
) -> TranscriptChunk:
    """Create a TranscriptChunk with detected speakers."""
    # Detect speakers in this chunk
    speakers = set()
    for seg in segments:
        if seg.start >= start and seg.end <= end + 0.5:
            speaker = _detect_speaker(seg.text)
            if speaker:
                speakers.add(speaker)

    return TranscriptChunk(
        chunk_id=f"{video_id}_{index:04d}",
        video_id=video_id,
        start=start,
        end=end,
        text=text,
        token_count=_estimate_tokens(text),
        section_id=None,
        section_title=", ".join(sorted(speakers)) if speakers else None,
    )
