"""Video ingestion API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import IngestRequest, IngestResponse
from app.services.pipeline import ingest_video

router = APIRouter(prefix="/videos", tags=["videos"])
logger = logging.getLogger(__name__)

# Simple in-memory store for V1. Replace with a database later.
_video_records: dict[str, dict] = {}


@router.post("", response_model=IngestResponse)
async def ingest_youtube_video(req: IngestRequest) -> IngestResponse:
    """Ingest a YouTube video: download, transcribe, embed, and summarize."""
    try:
        record = await ingest_video(req.url)
        _video_records[record.metadata.video_id] = record.model_dump()
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
    record = _video_records.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    return record


@router.get("/{video_id}/transcript")
async def get_transcript(video_id: str) -> dict:
    """Get the transcript for a video."""
    record = _video_records.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    if record.get("transcript") is None:
        raise HTTPException(status_code=404, detail="Transcript not yet available")
    return record["transcript"]


@router.get("/{video_id}/summary")
async def get_summary(video_id: str) -> dict:
    """Get the summary for a video."""
    record = _video_records.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    if record.get("summary") is None:
        raise HTTPException(status_code=404, detail="Summary not yet available")
    return {
        "video_id": video_id,
        "summary": record["summary"],
        "sections": record.get("sections", []),
    }


@router.get("/{video_id}/sections")
async def get_sections(video_id: str) -> list[dict]:
    """Get the section summaries for a video."""
    record = _video_records.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    sections = record.get("sections")
    if sections is None:
        raise HTTPException(status_code=404, detail="Sections not yet available")
    return sections
