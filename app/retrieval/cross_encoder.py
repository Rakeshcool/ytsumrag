"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import logging

from app.models.schemas import TranscriptChunk

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder model
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker model...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder reranker loaded.")
    return _model


class CrossEncoderReranker:
    """Rerank search results using a cross-encoder model."""

    def __init__(self) -> None:
        pass

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[TranscriptChunk, float]],
        top_k: int = 8,
    ) -> list[tuple[TranscriptChunk, float]]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: The user's question.
            candidates: List of (chunk, original_score) tuples.
            top_k: Number of results to return.

        Returns:
            Reranked list of (chunk, cross_encoder_score) tuples.
        """
        if not candidates:
            return []

        model = _get_model()

        # Build query-document pairs
        pairs = [(query, chunk.text) for chunk, _ in candidates]

        # Score with cross-encoder
        scores = model.predict(pairs)

        # Combine with original scores and sort
        reranked = []
        for i, (chunk, orig_score) in enumerate(candidates):
            # Combine cross-encoder score with original score (weighted)
            combined_score = float(scores[i]) * 0.7 + orig_score * 0.3
            reranked.append((chunk, combined_score))

        # Sort by combined score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]


# Module-level singleton
reranker = CrossEncoderReranker()
