"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """All application settings, populated from .env and environment."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # ── llama.cpp ────────────────────────────────────────────────────
    llama_server_path: str = ""

    # ── Generation server ───────────────────────────────────────────
    generation_host: str = "0.0.0.0"
    generation_port: int = 3000
    generation_base_url: str = "http://127.0.0.1:3000"
    generation_model: str = ""

    # ── Embedding server ────────────────────────────────────────────
    embedding_host: str = "127.0.0.1"
    embedding_port: int = 8081
    embedding_base_url: str = "http://127.0.0.1:8081"
    embedding_model: str = ""
    embedding_dimension: int = 1024

    # ── Qdrant ──────────────────────────────────────────────────────
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "youtube_chunks"

    # ── Whisper ─────────────────────────────────────────────────────
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # ── Chunking ────────────────────────────────────────────────────
    chunk_max_tokens: int = 700
    chunk_overlap_tokens: int = 100

    # ── Retrieval ───────────────────────────────────────────────────
    retrieval_top_k: int = 20
    rag_final_k: int = 8

    # ── Generation tokens ──────────────────────────────────────────
    summary_chunk_max_tokens: int = 512
    summary_section_max_tokens: int = 768
    summary_final_max_tokens: int = 4096
    rag_answer_max_tokens: int = 2048
    topic_detection_max_tokens: int = 30

    # ── Model server lifecycle ──────────────────────────────────────
    model_server_startup_timeout: int = 120
    model_server_health_interval: float = 0.5
    model_server_shutdown_timeout: int = 15


settings = Settings()
