"""Hierarchical summarization — chunk → section → final."""

from __future__ import annotations

import logging
from collections import defaultdict

from app.config.settings import settings
from app.llm.llama_client import llama_client
from app.models.schemas import SectionSummary, TranscriptChunk
from app.summarization.chunk_summary import summarize_chunk

logger = logging.getLogger(__name__)


async def _summarize_section(
    section_title: str,
    section_chunks: list[TranscriptChunk],
    chunk_summaries: list[str],
    start: float,
    end: float,
) -> SectionSummary:
    """Summarize all chunks in a section into a single section summary."""
    combined = "\n\n".join(
        f"[{i + 1}] {summary}" for i, summary in enumerate(chunk_summaries)
    )

    prompt = (
        f'Below are summaries of chunks from a section titled "{section_title}"\n'
        f'of a YouTube video (timestamps {start:.1f}s – {end:.1f}s).\n\n'
        f"Section chunk summaries:\n{combined}\n\n"
        f"Provide a coherent, concise summary of this entire section (3-6 sentences).\n"
        f"Focus on the key points, arguments, and conclusions discussed."
    )

    try:
        summary = await llama_client.generate(
            prompt,
            system_prompt="You are a precise video transcript summarizer.",
            temperature=0.2,
            max_tokens=settings.summary_section_max_tokens,
        )
        return SectionSummary(
            section_id=section_chunks[0].section_id or "unknown",
            title=section_title,
            summary=summary.strip(),
            start=start,
            end=end,
        )
    except Exception as exc:
        logger.error("Failed to summarize section '%s': %s", section_title, exc)
        raise


async def summarize_sections(chunks: list[TranscriptChunk]) -> list[SectionSummary]:
    """Hierarchically summarize transcript chunks into section summaries.

    1. Summarize each chunk.
    2. Group chunks by section.
    3. Summarize each section from its chunk summaries.
    """
    if not chunks:
        return []

    # Step 1: Summarize individual chunks
    logger.info("Summarizing %d chunks…", len(chunks))
    chunk_summaries: list[str] = []
    for chunk in chunks:
        s = await summarize_chunk(chunk)
        chunk_summaries.append(s)

    # Step 2: Group by section
    sections: dict[str, list[tuple[TranscriptChunk, str]]] = defaultdict(list)
    for chunk, summary in zip(chunks, chunk_summaries):
        key = chunk.section_id or "unknown"
        sections[key].append((chunk, summary))

    # Step 3: Summarize each section
    logger.info("Summarizing %d sections…", len(sections))
    section_summaries: list[SectionSummary] = []
    for section_id, items in sections.items():
        section_chunks = [c for c, _ in items]
        section_chunk_sums = [s for _, s in items]
        title = section_chunks[0].section_title or "Unknown"
        start = section_chunks[0].start
        end = section_chunks[-1].end

        section_sum = await _summarize_section(
            title, section_chunks, section_chunk_sums, start, end
        )
        section_summaries.append(section_sum)

    return section_summaries
