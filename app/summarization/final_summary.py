"""Final summary generation from section summaries."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import settings
from app.llm.llama_client import llama_client
from app.models.schemas import SectionSummary

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "final_summary.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


async def generate_final_summary(section_summaries: list[SectionSummary]) -> str:
    """Combine section summaries into a comprehensive final summary."""
    if not section_summaries:
        return "No content to summarize."

    sections_text = "\n\n".join(
        f"## {s.title} ({s.start:.1f}s – {s.end:.1f}s)\n{s.summary}"
        for s in section_summaries
    )

    prompt_template = _load_prompt()
    prompt = prompt_template.format(section_summaries=sections_text)

    try:
        final = await llama_client.generate(
            prompt,
            system_prompt="You are a precise video transcript summarizer.",
            temperature=0.3,
            max_tokens=settings.summary_final_max_tokens,
        )
        return final.strip()
    except Exception as exc:
        logger.error("Failed to generate final summary: %s", exc)
        raise
