"""RAG retriever — embeds a query and retrieves relevant chunks from Qdrant."""

from __future__ import annotations

import logging

from app.config.settings import settings
from app.embeddings.qwen import embedding_client
from app.models.schemas import ChatSource
from app.vectorstore.qdrant import vector_store

logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    video_id: str,
    top_k: int | None = None,
) -> list[ChatSource]:
    """Retrieve the most relevant chunks for a query within a video."""
    k = top_k or settings.retrieval_top_k

    query_vector = await embedding_client.embed_query(query)
    hits = vector_store.search(query_vector, video_id, top_k=k)

    sources: list[ChatSource] = []
    for hit in hits:
        sources.append(
            ChatSource(
                text=hit.get("text", ""),
                video_id=hit.get("video_id", video_id),
                start=hit.get("start", 0.0),
                end=hit.get("end", 0.0),
                section_title=hit.get("section_title"),
            )
        )

    return sources


def select_final_context(
    sources: list[ChatSource],
    final_k: int | None = None,
) -> list[ChatSource]:
    """Select the top final_k sources from the retrieval results."""
    k = final_k or settings.rag_final_k
    return sources[:k]
