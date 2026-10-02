"""BM25 index for keyword-based search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.models.schemas import TranscriptChunk

logger = logging.getLogger(__name__)

_INDEX_DIR = Path("data/bm25")


class BM25Index:
    """In-memory BM25 index for transcript chunks."""

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._chunks: list[TranscriptChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def build(self, chunks: list[TranscriptChunk]) -> None:
        """Build the BM25 index from transcript chunks."""
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d chunks", len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[TranscriptChunk, float]]:
        """Search the index and return chunks with BM25 scores."""
        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunks[idx], float(scores[idx])))

        return results

    def save(self, video_id: str) -> None:
        """Save index metadata to disk (chunks + tokenized corpus)."""
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.model_dump(mode="json") for c in self._chunks],
            "tokenized_corpus": self._tokenized_corpus,
        }
        path = _INDEX_DIR / f"{video_id}_bm25.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("BM25 index saved for %s", video_id)

    def load(self, video_id: str) -> bool:
        """Load index from disk. Returns True if successful."""
        path = _INDEX_DIR / f"{video_id}_bm25.json"
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._chunks = [TranscriptChunk(**c) for c in data["chunks"]]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._index = BM25Okapi(self._tokenized_corpus)
            logger.info("BM25 index loaded for %s (%d chunks)", video_id, len(self._chunks))
            return True
        except Exception as exc:
            logger.error("Failed to load BM25 index for %s: %s", video_id, exc)
            return False


# Module-level cache: video_id -> BM25Index
_bm25_cache: dict[str, BM25Index] = {}


def get_bm25_index(video_id: str) -> BM25Index:
    """Get or load the BM25 index for a video."""
    if video_id in _bm25_cache:
        return _bm25_cache[video_id]

    index = BM25Index()
    index.load(video_id)
    _bm25_cache[video_id] = index
    return index


def build_bm25_index(video_id: str, chunks: list[TranscriptChunk]) -> BM25Index:
    """Build and cache a new BM25 index for a video."""
    index = BM25Index()
    index.build(chunks)
    index.save(video_id)
    _bm25_cache[video_id] = index
    return index
