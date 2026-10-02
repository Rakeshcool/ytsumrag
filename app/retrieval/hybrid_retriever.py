"""Hybrid retriever combining vector search + BM25 with RRF fusion and neighbor expansion."""

from __future__ import annotations

import logging

from app.models.schemas import TranscriptChunk
from app.retrieval.bm25_index import get_bm25_index
from app.vectorstore.qdrant import vector_store

logger = logging.getLogger(__name__)

# Embedding client for query vectorization
_embedding_client = None


def _get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        from app.embeddings.qwen import embedding_client
        _embedding_client = embedding_client
    return _embedding_client


def _reciprocal_rank_fusion(
    results_list: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}

    for results in results_list:
        for rank, (item_id, _) in enumerate(results):
            if item_id not in scores:
                scores[item_id] = 0.0
            scores[item_id] += 1.0 / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def _parse_qdrant_hit(hit: dict) -> tuple[str, float, dict]:
    """Parse a Qdrant search hit into (chunk_id, score, payload)."""
    chunk_id = hit.get("chunk_id", "")
    score = hit.get("score", 0.0)
    return chunk_id, score, hit


def _payload_to_chunk(payload: dict) -> TranscriptChunk:
    """Convert Qdrant payload back to TranscriptChunk."""
    return TranscriptChunk(
        chunk_id=payload.get("chunk_id", ""),
        video_id=payload.get("video_id", ""),
        start=payload.get("start", 0.0),
        end=payload.get("end", 0.0),
        text=payload.get("text", ""),
        token_count=payload.get("token_count", 0),
        section_id=None,
        section_title=payload.get("speakers", ""),
    )


async def hybrid_retrieve(
    query: str,
    video_id: str,
    top_k: int = 15,
    neighbor_window: int = 1,
) -> list[tuple[TranscriptChunk, float, dict]]:
    """Retrieve relevant chunks using hybrid search with neighbor expansion.

    Args:
        query: The user's question.
        video_id: Video to search within.
        top_k: Number of initial results from each search.
        neighbor_window: Number of neighboring chunks to include (±N).

    Returns:
        List of (chunk, score, metadata) tuples.
    """
    # 1. Vector search
    vector_results = []
    vector_payloads: dict[str, dict] = {}
    try:
        client = _get_embedding_client()
        query_embedding = await client.embed_texts([query])
        hits = vector_store.search(query_embedding[0], video_id, top_k=top_k)
        for hit in hits:
            chunk_id, score, payload = _parse_qdrant_hit(hit)
            vector_results.append((chunk_id, score))
            vector_payloads[chunk_id] = payload
    except Exception as exc:
        logger.warning("Vector search failed: %s", exc)

    # 2. BM25 search
    bm25_results = []
    try:
        bm25_index = get_bm25_index(video_id)
        if bm25_index._index is not None:
            bm25_hits = bm25_index.search(query, top_k=top_k)
            for chunk, score in bm25_hits:
                bm25_results.append((chunk.chunk_id, score))
                # Also store the chunk data for BM25 results
                vector_payloads[chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "video_id": chunk.video_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "speakers": chunk.section_title or "",
                    "token_count": chunk.token_count,
                    "summary": "",
                    "keywords": [],
                }
    except Exception as exc:
        logger.warning("BM25 search failed: %s", exc)

    if not vector_results and not bm25_results:
        return []

    # 3. RRF fusion
    rrf_merged = _reciprocal_rank_fusion([vector_results, bm25_results])

    # 4. Collect all chunk IDs (including neighbors)
    all_chunk_ids = set()
    chunk_id_list = [cid for cid, _ in rrf_merged]

    for chunk_id in chunk_id_list:
        all_chunk_ids.add(chunk_id)

        # Add neighbors
        if neighbor_window > 0:
            # Find index in the original list
            try:
                idx = chunk_id_list.index(chunk_id)
                for offset in range(-neighbor_window, neighbor_window + 1):
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(chunk_id_list):
                        all_chunk_ids.add(chunk_id_list[neighbor_idx])
            except ValueError:
                pass

    # 5. Fetch all chunks from Qdrant (including neighbors)
    all_chunks_data = {}
    try:
        fetched = vector_store.search_by_chunk_ids(
            list(all_chunk_ids), video_id,
        )
        for item in fetched:
            cid = item.get("chunk_id", "")
            all_chunks_data[cid] = item
    except Exception as exc:
        logger.warning("Failed to fetch neighbor chunks: %s", exc)

    # 6. Build final results
    results = []
    for chunk_id, rrf_score in rrf_merged:
        payload = all_chunks_data.get(chunk_id) or vector_payloads.get(chunk_id)
        if payload:
            chunk = _payload_to_chunk(payload)
            results.append((chunk, rrf_score, payload))

    logger.info(
        "Hybrid retrieval: %d vector + %d bm25 -> %d fused -> %d with neighbors",
        len(vector_results),
        len(bm25_results),
        len(rrf_merged),
        len(results),
    )

    return results
