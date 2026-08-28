"""End-to-end video ingestion pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.embeddings.qwen import embedding_client
from app.ingestion.audio import download_audio
from app.ingestion.captions import fetch_captions
from app.ingestion.youtube import fetch_metadata
from app.llm.llama_client import llama_client
from app.model_servers.manager import model_server_manager
from app.models.schemas import (
    Transcript,
    VideoRecord,
    VideoStatus,
)
from app.processing.chunker import chunk_transcript
from app.processing.cleaner import clean_transcript
from app.processing.topics import detect_topics
from app.summarization.final_summary import generate_final_summary
from app.summarization.hierarchical import summarize_sections
from app.transcription.whisper import transcribe_audio
from app.vectorstore.qdrant import vector_store

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")


def _cache_path(video_id: str, name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{video_id}_{name}.json"


def _load_cache(video_id: str, name: str) -> dict | None:
    p = _cache_path(video_id, name)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(video_id: str, name: str, data: dict) -> None:
    p = _cache_path(video_id, name)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def ingest_video(url: str) -> VideoRecord:
    """Full ingestion pipeline for a YouTube video URL."""
    # 1. Extract video ID and metadata
    metadata = fetch_metadata(url)
    video_id = metadata.video_id
    logger.info("Starting ingestion for video %s — %s", video_id, metadata.title)

    record = VideoRecord(metadata=metadata, status=VideoStatus.TRANSCRIBING)

    # 2. Try to load cached transcript
    transcript: Transcript | None = None
    cached = _load_cache(video_id, "transcript")
    if cached:
        transcript = Transcript(**cached)
        logger.info("Loaded cached transcript for %s.", video_id)
    else:
        # Try captions first
        transcript = fetch_captions(video_id, url)

        if transcript is None:
            # Download audio and transcribe
            logger.info("No captions; downloading audio…")
            audio_path = download_audio(video_id, url)
            transcript = transcribe_audio(audio_path, metadata.title, video_id)
            # Cache transcript
            _save_cache(video_id, "transcript", transcript.model_dump(mode="json"))

    record.transcript = transcript
    record.status = VideoStatus.PROCESSING

    # 3. Clean transcript
    cleaned = clean_transcript(transcript)

    # 4. Chunk transcript
    chunks = chunk_transcript(cleaned)

    # 5. Topic detection (uses generation server)
    await model_server_manager.ensure_generation_server()
    try:
        chunks = await detect_topics(chunks)
    finally:
        await model_server_manager.release_generation_server()

    record.status = VideoStatus.EMBEDDING

    # 6. Generate embeddings and store in Qdrant
    vector_store.ensure_collection()
    await model_server_manager.ensure_embedding_server()
    try:
        texts = [c.text for c in chunks]
        embeddings = await embedding_client.embed_texts(texts)
        vector_store.upsert_chunks(chunks, embeddings)
    finally:
        await model_server_manager.release_embedding_server()

    record.status = VideoStatus.SUMMARIZING

    # 7. Hierarchical summarization
    await model_server_manager.ensure_generation_server()
    try:
        section_summaries = await summarize_sections(chunks)
        final_summary = await generate_final_summary(section_summaries)
    finally:
        await model_server_manager.release_generation_server()

    record.summary = final_summary
    record.sections = [
        {
            "section_id": s.section_id,
            "title": s.title,
            "summary": s.summary,
            "start": s.start,
            "end": s.end,
        }
        for s in section_summaries
    ]
    record.status = VideoStatus.COMPLETED

    # Cache the full record
    _save_cache(video_id, "record", record.model_dump(mode="json"))

    logger.info("Ingestion complete for %s.", video_id)
    return record


async def answer_question(video_id: str, question: str) -> str:
    """Answer a question about a video using RAG."""
    from app.config.settings import settings
    from app.retrieval.reranker import reranker
    from app.retrieval.retriever import retrieve, select_final_context

    # 1. Retrieve relevant chunks
    sources = await retrieve(question, video_id)

    # 2. Rerank (no-op for V1)
    sources = await reranker.rerank(question, sources)

    # 3. Select final context
    context_sources = select_final_context(sources)

    if not context_sources:
        return "No relevant information found for this video."

    # 4. Build context
    context_parts = []
    for src in context_sources:
        timestamp = f"[{src.start:.1f}s – {src.end:.1f}s]"
        context_parts.append(f"{timestamp} {src.text}")
    context = "\n\n".join(context_parts)

    # 5. Generate answer
    prompt_path = Path(__file__).parent.parent / "prompts" / "rag_answer.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(context=context, question=question)

    answer = await llama_client.generate(
        prompt,
        system_prompt="You are a helpful assistant answering questions about YouTube videos.",
        temperature=0.3,
        max_tokens=settings.rag_answer_max_tokens,
    )

    return answer
