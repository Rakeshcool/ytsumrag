"""Embedding client — sends requests to the local embedding llama-server."""

from __future__ import annotations

import logging

import httpx

from app.config.settings import settings
from app.model_servers.manager import model_server_manager

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Thin async client for the Qwen3 embedding server."""

    def __init__(self) -> None:
        self._base_url = settings.embedding_base_url
        self._dimension = settings.embedding_dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vectors."""
        await model_server_manager.ensure_embedding_server()

        payload = {
            "model": "embedding",
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/embeddings",
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()["data"]
        # Sort by index to preserve order
        data.sort(key=lambda x: x["index"])
        vectors = [item["embedding"] for item in data]

        # Validate dimensions
        for i, vec in enumerate(vectors):
            if len(vec) != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {len(vec)} for text index {i}"
                )

        return vectors

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]


embedding_client = EmbeddingClient()
