"""Video ingestion API endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, IngestRequest, IngestResponse
from app.services.pipeline import answer_question, ingest_video

router = APIRouter(prefix="/videos", tags=["videos"])
logger = logging.getLogger(__name__)


def _load_record(video_id: str) -> dict | None:
    """Load a video record from cache."""
    cache_path = Path("data/cache") / f"{video_id}_record.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _list_cached_videos() -> list[dict]:
    """List all cached video records."""
    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        return []
    videos = []
    for f in sorted(cache_dir.glob("*_record.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            videos.append({
                "video_id": data["metadata"]["video_id"],
                "title": data["metadata"].get("title", "Unknown"),
                "status": data.get("status", "unknown"),
                "thumbnail_url": data["metadata"].get("thumbnail_url"),
                "duration": data["metadata"].get("duration"),
                "ingested_at": data["metadata"].get("ingested_at"),
            })
        except Exception:
            pass
    return videos


@router.get("")
async def list_videos() -> list[dict]:
    """List all ingested videos."""
    return _list_cached_videos()


@router.post("", response_model=IngestResponse)
async def ingest_youtube_video(req: IngestRequest) -> IngestResponse:
    """Ingest a YouTube video: download, transcribe, chunk, embed, store."""
    try:
        record = await ingest_video(req.url)
        return IngestResponse(
            video_id=record.metadata.video_id,
            status=record.status,
            message="Video ingested successfully.",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.get("/{video_id}")
async def get_video(video_id: str) -> dict:
    """Get video metadata and status."""
    record = _load_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    return record


@router.get("/{video_id}/transcript")
async def get_transcript(video_id: str) -> dict:
    """Get the transcript for a video."""
    record = _load_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    if record.get("transcript") is None:
        raise HTTPException(status_code=404, detail="Transcript not yet available")
    return record["transcript"]


@router.post("/{video_id}/search")
async def search_video(video_id: str, req: ChatRequest) -> list[dict]:
    """Search chunks in a video using hybrid retrieval."""
    record = _load_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        from app.retrieval.retriever import retrieve_and_rerank
        results = await retrieve_and_rerank(req.question, video_id, top_k=10, final_k=10)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "score": round(score, 3),
                "summary": meta.get("summary", ""),
                "keywords": meta.get("keywords", []),
                "speakers": meta.get("speakers", ""),
            }
            for chunk, score, meta in results
        ]
    except Exception as exc:
        logger.exception("Search failed for %s", video_id)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.post("/{video_id}/chat")
async def chat_about_video(video_id: str, req: ChatRequest) -> dict:
    """Ask a question about a specific video using hybrid retrieval."""
    record = _load_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        answer = await answer_question(video_id, req.question)
        return {"answer": answer}
    except Exception as exc:
        logger.exception("Chat failed for video %s", video_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")
