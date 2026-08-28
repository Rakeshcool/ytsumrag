"""Qdrant vectorstore client — manages the youtube_chunks collection."""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config.settings import settings
from app.models.schemas import TranscriptChunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Thin wrapper around the Qdrant client for this project's collection."""

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._collection = settings.qdrant_collection
        self._dimension = settings.embedding_dimension

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self._collection not in collections:
            self.client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self._collection)

    def upsert_chunks(
        self,
        chunks: list[TranscriptChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunk vectors + metadata. Returns number of upserted points."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.video_id}:{chunk.chunk_id}"))
            payload = {
                "chunk_id": chunk.chunk_id,
                "video_id": chunk.video_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
            }
            points.append(
                PointStruct(id=point_id, vector=embedding, payload=payload)
            )

        # Upsert in batches of 100
        batch_size = 100
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self._collection, points=batch)
            total += len(batch)

        logger.info("Upserted %d chunks into Qdrant.", total)
        return total

    def search(
        self,
        query_vector: list[float],
        video_id: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Search for similar chunks, filtered by video_id."""
        k = top_k or settings.retrieval_top_k

        query_filter = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )

        results = self.client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )

        hits = []
        for point in results.points:
            payload = point.payload
            payload["score"] = point.score  # type: ignore[assignment]
            hits.append(payload)

        return hits

    def delete_video(self, video_id: str) -> None:
        """Delete all chunks for a given video."""
        self.client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
        )
        logger.info("Deleted all chunks for video %s from Qdrant.", video_id)

    def count_video_chunks(self, video_id: str) -> int:
        """Count how many chunks exist for a video."""
        results = self.client.count(
            collection_name=self._collection,
            count_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
        )
        return results.count

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


vector_store = QdrantVectorStore()
