"""RAG retrieval pipeline — hybrid search + reranking + relevance gate."""

from __future__ import annotations

import logging

from app.config.settings import settings
from app.models.schemas import TranscriptChunk
from app.retrieval.cross_encoder import reranker
from app.retrieval.hybrid_retriever import hybrid_retrieve

logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    video_id: str,
    top_k: int | None = None,
    neighbor_window: int = 1,
) -> list[tuple[TranscriptChunk, float, dict]]:
    """Retrieve relevant chunks using hybrid search.

    Returns:
        List of (chunk, score, metadata) tuples.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    return await hybrid_retrieve(query, video_id, top_k=top_k, neighbor_window=neighbor_window)


async def retrieve_and_rerank(
    query: str,
    video_id: str,
    top_k: int | None = None,
    final_k: int | None = None,
    neighbor_window: int = 1,
) -> list[tuple[TranscriptChunk, float, dict]]:
    """Retrieve + rerank pipeline with neighbor expansion.

    Returns:
        List of (chunk, score, metadata) tuples sorted by reranked relevance.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k
    if final_k is None:
        final_k = settings.rag_final_k

    # 1. Hybrid retrieval with neighbors
    candidates = await hybrid_retrieve(
        query, video_id, top_k=top_k, neighbor_window=neighbor_window,
    )

    if not candidates:
        return []

    # 2. Cross-encoder reranking
    # Convert to (chunk, score) for reranker
    reranker_input = [(chunk, score) for chunk, score, _ in candidates]
    reranked = await reranker.rerank(query, reranker_input, top_k=final_k)

    # 3. Merge metadata back
    metadata_lookup = {chunk.chunk_id: meta for chunk, _, meta in candidates}
    results = []
    for chunk, score in reranked:
        meta = metadata_lookup.get(chunk.chunk_id, {})
        results.append((chunk, score, meta))

    return results


def check_relevance(
    results: list[tuple[TranscriptChunk, float, dict]],
    threshold: float = -10.0,
) -> bool:
    """Check if retrieved results are relevant enough to answer the question.

    Cross-encoder scores are logits (can be negative).
    Only blocks if scores are extremely low (truly irrelevant).
    """
    if not results:
        return False

    top_score = results[0][1]
    if top_score < threshold:
        logger.warning(
            "Very low relevance (top=%.3f). Refusing to answer.",
            top_score,
        )
        return False

    return True


def format_evidence(
    results: list[tuple[TranscriptChunk, float, dict]],
) -> str:
    """Format retrieved chunks as evidence for the LLM.

    Includes timestamps, speakers, summaries, and raw text.
    """
    evidence_parts = []

    for i, (chunk, score, meta) in enumerate(results):
        # Format timestamp
        m1, s1 = divmod(int(chunk.start), 60)
        h1, m1 = divmod(m1, 60)
        m2, s2 = divmod(int(chunk.end), 60)
        h2, m2 = divmod(m2, 60)
        timestamp = f"{h1:02d}:{m1:02d}:{s1:02d}-{h2:02d}:{m2:02d}:{s2:02d}"

        # Get speakers
        speakers = meta.get("speakers", "") or chunk.section_title or ""

        # Get summary and keywords
        summary = meta.get("summary", "")
        keywords = meta.get("keywords", [])

        # Build evidence block
        block = f"[Evidence {i + 1}] [{timestamp}]"
        if speakers:
            block += f" [{speakers}]"
        if summary:
            block += f"\nSummary: {summary}"
        if keywords:
            block += f"\nKeywords: {', '.join(keywords)}"
        block += f"\n{chunk.text}"

        evidence_parts.append(block)

    return "\n\n".join(evidence_parts)


def select_final_context(
    results: list[tuple[TranscriptChunk, float, dict]],
    max_tokens: int = 5000,
) -> list[tuple[TranscriptChunk, float, dict]]:
    """Select final context chunks, respecting token budget.

    Returns chunks fitting within ~max_tokens (4 chars per token).
    """
    max_chars = max_tokens * 4
    selected = []
    total_chars = 0

    for chunk, score, meta in results:
        if total_chars + len(chunk.text) > max_chars:
            break
        selected.append((chunk, score, meta))
        total_chars += len(chunk.text)

    return selected
