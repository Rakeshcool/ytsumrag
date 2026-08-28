"""Chunk-level summarization."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import settings
from app.llm.llama_client import llama_client
from app.models.schemas import TranscriptChunk

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "chunk_summary.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


async def summarize_chunk(chunk: TranscriptChunk) -> str:
    """Generate a summary for a single transcript chunk."""
    prompt_template = _load_prompt()

    prompt = prompt_template.format(
        section_title=chunk.section_title or "Unknown",
        start=chunk.start,
        end=chunk.end,
        text=chunk.text,
    )

    try:
        summary = await llama_client.generate(
            prompt,
            system_prompt="You are a precise video transcript summarizer.",
            temperature=0.2,
            max_tokens=settings.summary_chunk_max_tokens,
        )
        return summary.strip()
    except Exception as exc:
        logger.error("Failed to summarize chunk %s: %s", chunk.chunk_id, exc)
        raise
