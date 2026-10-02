"""End-to-end video ingestion and RAG pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config.settings import settings
from app.embeddings.qwen import embedding_client
from app.ingestion.audio import download_audio
from app.ingestion.captions import fetch_captions
from app.ingestion.youtube import fetch_metadata
from app.model_servers.manager import model_server_manager
from app.models.schemas import (
    Transcript,
    VideoRecord,
    VideoStatus,
)
from app.processing.cleaner import clean_transcript
from app.processing.metadata import enrich_chunks
from app.processing.speaker_chunker import chunk_transcript
from app.retrieval.bm25_index import build_bm25_index
from app.transcription.whisper import transcribe_audio
from app.vectorstore.qdrant import vector_store

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")
_AUDIO_DIR = Path("data/audio")


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


def _cleanup_audio(video_id: str) -> None:
    """Remove audio files for a video after transcription."""
    if not _AUDIO_DIR.exists():
        return
    for ext in ("mp3", "m4a", "wav", "ogg", "webm", "opus"):
        for f in _AUDIO_DIR.glob(f"{video_id}*.{ext}"):
            try:
                f.unlink()
                logger.info("Cleaned up audio file: %s", f.name)
            except OSError as e:
                logger.warning("Failed to delete audio file %s: %s", f.name, e)


async def ingest_video(
    url: str,
    progress_callback=None,
    force_whisper: bool = False,
    no_cleanup: bool = False,
) -> VideoRecord:
    """Full ingestion pipeline for a YouTube video URL.

    Pipeline: transcript → clean → speaker-aware chunk → metadata enrich → embed → store

    Args:
        url: YouTube video URL.
        progress_callback: Optional async callable(phase, message, step, total).
        force_whisper: Skip captions, always use Whisper transcription.
        no_cleanup: Keep audio files after transcription.

    Returns:
        Completed VideoRecord.
    """
    total_steps = 6
    step = 0

    async def _progress(phase: str, message: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            await progress_callback(phase, message, step, total_steps)
        logger.info("[%s] %s", phase, message)

    # 1. Extract video ID and metadata
    await _progress("ingest", "Fetching video metadata...")
    metadata = fetch_metadata(url)
    video_id = metadata.video_id
    logger.info("Starting ingestion for video %s — %s", video_id, metadata.title)

    record = VideoRecord(metadata=metadata, status=VideoStatus.TRANSCRIBING)

    # 2. Try to load cached transcript
    transcript: Transcript | None = None
    cached = _load_cache(video_id, "transcript")
    if cached and not force_whisper:
        transcript = Transcript(**cached)
        await _progress("ingest", "Loaded cached transcript")
    else:
        # Try captions first
        if not force_whisper:
            await _progress("ingest", "Fetching captions...")
            transcript = fetch_captions(video_id, url)

        if transcript is None:
            # Download audio and transcribe
            await _progress("ingest", "Downloading audio...")
            audio_path = download_audio(video_id, url)
            await _progress("ingest", "Transcribing with Whisper...")
            transcript = transcribe_audio(audio_path, metadata.title, video_id)
            _save_cache(video_id, "transcript", transcript.model_dump(mode="json"))

            # Auto-cleanup audio
            if not no_cleanup and settings.auto_cleanup_audio:
                _cleanup_audio(video_id)

    record.transcript = transcript
    record.status = VideoStatus.PROCESSING

    # 3. Clean transcript
    await _progress("process", "Cleaning transcript...")
    cleaned = clean_transcript(transcript)

    # 4. Speaker-aware chunking
    await _progress("process", "Speaker-aware chunking...")
    chunks = chunk_transcript(cleaned)

    # 5. Enrich chunks with metadata (summary + keywords)
    await _progress("llm", "Generating chunk metadata...")
    await model_server_manager.ensure_generation_server()
    try:
        metadata_list = await enrich_chunks(chunks)
    finally:
        await model_server_manager.release_generation_server()

    record.status = VideoStatus.EMBEDDING

    # 6. Generate embeddings and store in Qdrant
    await _progress("embed", "Generating embeddings...")
    vector_store.ensure_collection()
    await model_server_manager.ensure_embedding_server()
    try:
        texts = [c.text for c in chunks]
        embeddings = await embedding_client.embed_texts(texts)
        vector_store.upsert_chunks(chunks, embeddings, metadata_list)
    finally:
        await model_server_manager.release_embedding_server()

    # 7. Build BM25 index
    await _progress("index", "Building BM25 index...")
    build_bm25_index(video_id, chunks)

    record.status = VideoStatus.COMPLETED

    # Cache the full record
    _save_cache(video_id, "record", record.model_dump(mode="json"))

    await _progress("done", "Ingestion complete!")
    logger.info(
        "Ingestion complete for %s (%d chunks with metadata)",
        video_id,
        len(chunks),
    )
    return record


async def answer_question(video_id: str, question: str) -> str:
    """Answer a question about a video using hybrid retrieval + relevance gate.

    Pipeline: retrieve → rerank → relevance gate → LLM with evidence
    """
    from app.llm.llama_client import llama_client
    from app.retrieval.retriever import (
        check_relevance,
        format_evidence,
        retrieve_and_rerank,
        select_final_context,
    )

    # 1. Hybrid retrieval + cross-encoder reranking
    reranked = await retrieve_and_rerank(question, video_id)

    if not reranked:
        return "The transcript does not provide enough information to answer this."

    # 2. Select final context within token budget
    context_results = select_final_context(reranked)

    if not context_results:
        return "The transcript does not provide enough information to answer this."

    # 3. Relevance gate — prevent hallucination
    if not check_relevance(context_results):
        return (
            "The retrieved transcript sections do not appear to contain "
            "relevant information to answer this question. "
            "The transcript does not provide enough information to answer this."
        )

    # 4. Format evidence for the LLM
    evidence = format_evidence(context_results)

    # 5. Generate answer with strict evidence-based prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "rag_answer.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(evidence=evidence, question=question)

    system = (
        "You are a precise assistant answering questions about podcast transcripts. "
        "Use ONLY the provided evidence."
    )
    answer = await llama_client.generate(
        prompt,
        system_prompt=system,
        temperature=0.1,
        max_tokens=settings.rag_answer_max_tokens,
    )

    return answer


async def generate_summary(video_id: str) -> str:
    """Generate a structured summary of a video using all its chunks."""
    from app.llm.llama_client import llama_client
    from app.retrieval.retriever import format_evidence, retrieve_and_rerank, select_final_context

    # Retrieve top chunks (use broad query to get diverse content)
    reranked = await retrieve_and_rerank(
        "main topics key points summary",
        video_id,
        top_k=20,
        final_k=15,
    )

    if not reranked:
        return "No transcript content available to summarize."

    # Use larger context for summary
    context_results = select_final_context(reranked, max_tokens=8000)

    if not context_results:
        return "No transcript content available to summarize."

    # Format evidence
    evidence = format_evidence(context_results)

    # Load summary prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "summary.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(evidence=evidence)

    system = (
        "You are a precise podcast summarizer. "
        "Generate structured summaries from transcript evidence."
    )

    summary = await llama_client.generate(
        prompt,
        system_prompt=system,
        temperature=0.2,
        max_tokens=settings.summary_final_max_tokens,
    )

    return summary
