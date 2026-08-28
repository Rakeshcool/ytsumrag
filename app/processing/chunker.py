"""Sentence-aware transcript chunking with overlap."""

from __future__ import annotations

import re

from app.config.settings import settings
from app.models.schemas import Transcript, TranscriptChunk

# Rough token estimate: ~1 token per 4 characters (conservative for English)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the delimiter attached."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_transcript(transcript: Transcript) -> list[TranscriptChunk]:
    """Split a transcript into overlapping, sentence-aware chunks."""
    max_tokens = settings.chunk_max_tokens
    overlap_tokens = settings.chunk_overlap_tokens
    video_id = transcript.video_id

    chunks: list[TranscriptChunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_start: float | None = None
    chunk_index = 0

    for seg in transcript.segments:
        sentences = _split_sentences(seg.text)

        for sentence in sentences:
            sent_tokens = _estimate_tokens(sentence)

            if chunk_start is None:
                chunk_start = seg.start

            # Would adding this sentence exceed the limit?
            if current_tokens + sent_tokens > max_tokens and current_sentences:
                # Flush current chunk
                chunk_text = " ".join(current_sentences)
                chunk_end = seg.start  # approximate end
                chunk_id = f"chunk_{chunk_index:04d}"
                chunks.append(
                    TranscriptChunk(
                        chunk_id=chunk_id,
                        video_id=video_id,
                        start=chunk_start,
                        end=chunk_end,
                        text=chunk_text,
                        token_count=current_tokens,
                        section_id=None,
                        section_title=None,
                    )
                )
                chunk_index += 1

                # Build overlap: take sentences from the end that fit
                overlap_sentences: list[str] = []
                overlap_tok = 0
                for s in reversed(current_sentences):
                    st = _estimate_tokens(s)
                    if overlap_tok + st > overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tok += st

                current_sentences = overlap_sentences
                current_tokens = overlap_tok
                chunk_start = seg.start

            current_sentences.append(sentence)
            current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_end = transcript.segments[-1].end if transcript.segments else 0.0
        chunk_id = f"chunk_{chunk_index:04d}"
        chunks.append(
            TranscriptChunk(
                chunk_id=chunk_id,
                video_id=video_id,
                start=chunk_start or 0.0,
                end=chunk_end,
                text=chunk_text,
                token_count=current_tokens,
                section_id=None,
                section_title=None,
            )
        )

    return chunks
