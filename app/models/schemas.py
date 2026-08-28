"""Pydantic models / schemas used across the application."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ── Transcript ──────────────────────────────────────────────────────


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    video_id: str
    title: str | None = None
    language: str | None = None
    segments: list[TranscriptSegment]


# ── Chunks ──────────────────────────────────────────────────────────


class TranscriptChunk(BaseModel):
    chunk_id: str
    video_id: str
    start: float
    end: float
    text: str
    token_count: int
    section_id: str | None = None
    section_title: str | None = None


# ── Video metadata ──────────────────────────────────────────────────


class VideoMetadata(BaseModel):
    video_id: str
    url: str
    title: str | None = None
    duration: float | None = None
    language: str | None = None
    thumbnail_url: str | None = None
    uploaded_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class VideoStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Summaries ───────────────────────────────────────────────────────


class SectionSummary(BaseModel):
    section_id: str
    title: str
    summary: str
    start: float
    end: float


class VideoSummary(BaseModel):
    video_id: str
    sections: list[SectionSummary]
    final_summary: str


class VideoRecord(BaseModel):
    metadata: VideoMetadata
    status: VideoStatus = VideoStatus.PENDING
    transcript: Transcript | None = None
    summary: str | None = None
    sections: list[SectionSummary] | None = None
    error: str | None = None


# ── API request / response ──────────────────────────────────────────


class IngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    video_id: str
    status: VideoStatus
    message: str


class ChatRequest(BaseModel):
    question: str


class ChatSource(BaseModel):
    text: str
    video_id: str
    start: float
    end: float
    section_title: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] | None = None


# ── Health ──────────────────────────────────────────────────────────


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    FAILED = "failed"


class HealthComponent(BaseModel):
    status: ServiceStatus
    detail: str | None = None


class HealthResponse(BaseModel):
    app: ServiceStatus
    qdrant: HealthComponent
    embedding_server: HealthComponent
    generation_server: HealthComponent
