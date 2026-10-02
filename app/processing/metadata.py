"""Chunk metadata enrichment — generates summaries and keywords for each chunk."""

from __future__ import annotations

import logging
import re

from app.models.schemas import TranscriptChunk

logger = logging.getLogger(__name__)

# Common filler words to exclude from keywords
_FILLER_WORDS = {
    "that", "this", "with", "from", "have", "been", "were", "they",
    "their", "what", "when", "which", "there", "about", "would",
    "could", "should", "will", "just", "like", "know", "think",
    "really", "very", "some", "more", "into", "than", "them",
    "then", "also", "well", "back", "even", "your", "only",
}


def _extract_keywords(text: str, top_k: int = 5) -> list[str]:
    """Extract keywords from text using frequency analysis."""
    # Extract words 4+ characters, lowercase
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    # Count frequency, exclude fillers
    word_freq: dict[str, int] = {}
    for w in words:
        if w not in _FILLER_WORDS:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Top keywords by frequency
    keywords = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:top_k]
    return keywords


def _extract_summary_heuristic(text: str) -> str:
    """Extract first meaningful sentence as summary."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            return s[:150]
    return text[:150]


async def enrich_chunk_metadata(chunk: TranscriptChunk) -> dict:
    """Generate summary and keywords for a chunk.

    Uses heuristic extraction only (fast, reliable).
    """
    keywords = _extract_keywords(chunk.text)
    summary = _extract_summary_heuristic(chunk.text)

    return {
        "summary": summary,
        "keywords": keywords,
    }


async def enrich_chunks(chunks: list[TranscriptChunk]) -> list[dict]:
    """Enrich a list of chunks with metadata.

    Returns:
        List of dicts with 'summary' and 'keywords' for each chunk.
    """
    logger.info("Enriching %d chunks with metadata...", len(chunks))

    results = []
    for chunk in chunks:
        metadata = await enrich_chunk_metadata(chunk)
        results.append(metadata)

    logger.info("Metadata enrichment complete.")
    return results
