"""Reranker interface — no-op implementation for V1."""

from __future__ import annotations

from app.models.schemas import ChatSource


class Reranker:
    """Optional reranking interface. Currently a no-op."""

    async def rerank(
        self,
        query: str,
        chunks: list[ChatSource],
    ) -> list[ChatSource]:
        """Return chunks as-is. Override with a real reranker later."""
        return chunks


reranker = Reranker()
