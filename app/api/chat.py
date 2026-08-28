"""Chat API endpoint for RAG Q&A about videos."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.pipeline import answer_question

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/videos/{video_id}/chat", response_model=ChatResponse)
async def chat_about_video(video_id: str, req: ChatRequest) -> ChatResponse:
    """Ask a question about a specific video using RAG."""
    try:
        answer = await answer_question(video_id, req.question)
        return ChatResponse(answer=answer, sources=None)
    except Exception as exc:
        logger.exception("Chat failed for video %s", video_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")
