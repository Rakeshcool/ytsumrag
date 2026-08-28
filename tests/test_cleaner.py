"""Unit tests for transcript cleaning."""

from __future__ import annotations

from app.models.schemas import Transcript, TranscriptSegment
from app.processing.cleaner import clean_transcript


def _make_transcript(segments_data: list[tuple[float, float, str]]) -> Transcript:
    segments = [
        TranscriptSegment(start=s, end=e, text=t) for s, e, t in segments_data
    ]
    return Transcript(video_id="test123", title="Test", language="en", segments=segments)


def test_removes_music_markers():
    transcript = _make_transcript([
        (0.0, 2.0, "[Music] Hello world"),
        (2.0, 4.0, "This is great [Applause]"),
    ])
    cleaned = clean_transcript(transcript)
    assert len(cleaned.segments) == 2
    assert "[Music]" not in cleaned.segments[0].text
    assert "[Applause]" not in cleaned.segments[1].text


def test_normalizes_whitespace():
    transcript = _make_transcript([
        (0.0, 2.0, "  Hello   world  "),
    ])
    cleaned = clean_transcript(transcript)
    assert cleaned.segments[0].text == "Hello world"


def test_removes_empty_parens():
    transcript = _make_transcript([
        (0.0, 2.0, "Hello () world"),
        (2.0, 4.0, "Another ( ) test"),
    ])
    cleaned = clean_transcript(transcript)
    assert "()" not in cleaned.segments[0].text
    assert "( )" not in cleaned.segments[1].text


def test_preserves_timestamps():
    transcript = _make_transcript([
        (1.5, 3.5, "Important text."),
    ])
    cleaned = clean_transcript(transcript)
    assert cleaned.segments[0].start == 1.5
    assert cleaned.segments[0].end == 3.5
