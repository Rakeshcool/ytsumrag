"""Topic / section detection for transcript chunks."""

from __future__ import annotations

import logging

from app.config.settings import settings
from app.llm.llama_client import llama_client
from app.models.schemas import TranscriptChunk

logger = logging.getLogger(__name__)

_TOPIC_PROMPT = (
    "Analyze the following transcript chunk and identify the main topic or section title.\n"
    "Return ONLY the topic/section title as a short phrase (3-8 words).\n"
    "Do not add any explanation.\n\n"
    "Transcript chunk:\n"
    "{text}"
)


async def detect_topics(chunks: list[TranscriptChunk]) -> list[TranscriptChunk]:
    """Assign section_id and section_title to each chunk using the generation LLM.

    Consecutive chunks with the same topic are grouped under the same section.
    """
    if not chunks:
        return chunks

    # Process in batches to avoid excessive LLM calls
    section_counter = 0
    previous_title: str | None = None

    for i, chunk in enumerate(chunks):
        # Sample: use first 500 chars to reduce token usage
        sample = chunk.text[:500]
        prompt = _TOPIC_PROMPT.format(text=sample)

        try:
            title = await llama_client.generate(
                prompt, temperature=0.1, max_tokens=settings.topic_detection_max_tokens
            )
            title = title.strip().strip('"').strip("'")

            # If same as previous, reuse section
            if title.lower() == (previous_title or "").lower():
                chunk.section_id = f"section_{section_counter:03d}"
                chunk.section_title = title
            else:
                section_counter += 1
                chunk.section_id = f"section_{section_counter:03d}"
                chunk.section_title = title
                previous_title = title

        except Exception as exc:
            logger.warning("Topic detection failed for chunk %s: %s", chunk.chunk_id, exc)
            section_counter += 1
            chunk.section_id = f"section_{section_counter:03d}"
            chunk.section_title = f"Section {section_counter}"
            previous_title = chunk.section_title

    return chunks
