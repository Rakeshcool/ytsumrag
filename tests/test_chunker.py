"""Unit tests for transcript chunking."""

from __future__ import annotations

import os

# Set env vars before importing app modules
os.environ.setdefault("CHUNK_MAX_TOKENS", "50")
os.environ.setdefault("CHUNK_OVERLAP_TOKENS", "10")

from app.models.schemas import Transcript, TranscriptSegment
from app.processing.chunker import chunk_transcript


def _make_transcript(segments_data: list[tuple[float, float, str]]) -> Transcript:
    segments = [
        TranscriptSegment(start=s, end=e, text=t) for s, e, t in segments_data
    ]
    return Transcript(video_id="test123", title="Test", language="en", segments=segments)


def test_empty_transcript():
    transcript = _make_transcript([])
    chunks = chunk_transcript(transcript)
    assert chunks == []


def test_single_segment_single_chunk():
    transcript = _make_transcript([(0.0, 5.0, "Hello world.")])
    chunks = chunk_transcript(transcript)
    assert len(chunks) == 1
    assert chunks[0].video_id == "test123"
    assert chunks[0].start == 0.0
    assert chunks[0].end == 5.0
    assert "Hello world." in chunks[0].text


def test_multiple_segments_produce_chunks():
    # Create enough text to exceed the small chunk limit
    sentences = [f"This is sentence number {i}. " * 10 for i in range(20)]
    segments = [(i * 1.0, (i + 1) * 1.0, s) for i, s in enumerate(sentences)]
    transcript = _make_transcript(segments)
    chunks = chunk_transcript(transcript)
    assert len(chunks) > 1
    # All chunks should have the same video_id
    for c in chunks:
        assert c.video_id == "test123"
        assert c.chunk_id.startswith("chunk_")


def test_chunk_preserves_timestamps():
    segments = [
        (0.0, 2.0, "First sentence."),
        (2.0, 4.0, "Second sentence."),
        (4.0, 6.0, "Third sentence."),
    ]
    transcript = _make_transcript(segments)
    chunks = chunk_transcript(transcript)
    assert len(chunks) >= 1
    assert chunks[0].start >= 0.0
