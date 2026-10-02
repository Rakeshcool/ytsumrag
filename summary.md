# YouTube Local RAG Summarizer — Project Summary

## Overview

A fully local YouTube video ingestion and RAG (Retrieval-Augmented Generation) question-answering system. No external APIs — everything runs on your machine using local LLMs.

---

## What Was Built

### Phase 1: Foundation
- FastAPI server with REST API endpoints
- Pydantic settings loaded from `.env`
- Qdrant vector database integration
- llama-server process lifecycle manager
- YouTube caption fetching + Whisper transcription fallback
- Basic pipeline: transcript → clean → chunk → embed → store

### Phase 2: Model Server Management
- Automatic startup/shutdown of llama-server processes
- Health polling with timeout
- Ownership tracking (started_by_application)
- Reference counting for concurrent usage
- Pre-existing server detection
- Orphan process killing
- Graceful shutdown with fallback to kill
- Sequential model usage for 8GB VRAM

### Phase 3: CLI & QoL Features
- Rich CLI with colorized output (tables, panels, progress bars)
- Real-time progress tracking during ingestion
- Auto-cleanup of audio files after transcription
- Export to markdown files
- Multiple CLI commands: ingest, ask, summary, search, transcript, list, health, server

### Phase 4: Hybrid Retrieval Architecture
- Speaker-aware chunking (breaks at conversation boundaries)
- Multiple representations per chunk (text, summary, keywords, timestamps, speakers)
- BM25 keyword search index
- Hybrid retrieval combining vector search + BM25
- Reciprocal Rank Fusion (RRF) for merging results
- Neighbor expansion (±1 chunk for context continuity)
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Relevance gate (prevents hallucination)
- Evidence-based answers with timestamps
- On-demand structured summaries

---

## Architecture

```
INGESTION
  YouTube URL → Transcript → Clean → Speaker-Aware Chunk → Metadata
      → Embed (Qwen3) → Store (Qdrant + BM25)

RAG Q&A
  User Question
      │
      ├──► Vector Search (Qdrant, top 15)
      ├──► BM25 Search (rank_bm25, top 15)
      │
      ▼
  Reciprocal Rank Fusion
      │
      ▼
  Add Neighbors (±1 chunk)
      │
      ▼
  Cross-Encoder Reranker → Top 5-8
      │
      ▼
  Relevance Gate ──weak──► "Not enough info"
      │
     strong
      │
      ▼
  LLM (strict evidence-based prompt)
      │
      ▼
  Answer + timestamps

SUMMARY
  Retrieve top chunks → LLM → Structured markdown
  (Overview, Takeaways, Topics, Concepts, Quotes, Conclusion)
```

---

## Models Used

| Model | Port | Purpose |
|-------|------|---------|
| LFM2.5-2.6B-Q8_0 | 3000 | RAG answers, summaries |
| Qwen3-Embedding-0.6B-Q8_0 | 8081 | Vector embeddings |
| all-MiniLM-L6-v2 | — | Semantic chunking (sentence-transformers) |
| ms-marco-MiniLM-L-6-v2 | — | Cross-encoder reranking |
| faster-whisper large-v3 | — | Audio transcription (fallback) |

---

## CLI Commands

```bash
# Ingestion
uv run python scripts/cli.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python scripts/cli.py ingest --force-whisper "URL"
uv run python scripts/cli.py ingest --no-cleanup "URL"

# RAG Q&A
uv run python scripts/cli.py ask VIDEO_ID "What is this video about?"

# Summary
uv run python scripts/cli.py summary VIDEO_ID

# Search (test retrieval)
uv run python scripts/cli.py search VIDEO_ID "query" --top-k 5

# View Data
uv run python scripts/cli.py transcript VIDEO_ID
uv run python scripts/cli.py list

# System
uv run python scripts/cli.py health
uv run python scripts/cli.py server start|stop|status
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /videos | Ingest a YouTube video |
| GET | /videos/{video_id} | Get video metadata |
| GET | /videos/{video_id}/transcript | Get transcript |
| POST | /videos/{video_id}/chat | Ask a question (RAG) |
| GET | /health | Health check |

---

## Key Features

### Speaker-Aware Chunking
- Breaks at conversation boundaries (speaker changes, topic shifts)
- Tracks which speakers appear in each chunk
- Keeps chunks between 300-700 tokens
- Preserves complete thoughts

### Multiple Representations
Each chunk stores:
- Raw transcript text
- Summary (heuristic extraction)
- Keywords (frequency analysis)
- Timestamps (start/end)
- Speaker labels

### Hybrid Retrieval
- Vector search (Qdrant) — semantic similarity
- BM25 search (rank_bm25) — keyword matching
- Reciprocal Rank Fusion — merges results from both
- Neighbor expansion — adds ±1 neighboring chunks
- Cross-encoder reranking — precise relevance scoring

### Relevance Gate
- Checks if retrieved evidence is strong enough
- Refuses to answer when evidence is weak
- Prevents hallucination by not generating when unsure

### Evidence-Based Answers
- LLM must cite specific evidence chunks
- Includes timestamps for each reference
- Cannot use outside knowledge
- Strict prompt rules prevent guessing

### Structured Summaries
On-demand generation with:
- Overview (2-4 paragraphs)
- Key Takeaways (5-10 bullets)
- Main Topics (with explanations)
- Important Concepts
- Key Quotes (with timestamps)
- Conclusion

---

## Configuration

Key settings in `.env`:

```bash
# Paths
LLAMA_SERVER_PATH=path/to/llama-server.exe
GENERATION_MODEL=path/to/LFM2.5-2.6B-Q8_0.gguf
EMBEDDING_MODEL=path/to/Qwen3-Embedding-0.6B-Q8_0.gguf

# Chunking
CHUNK_MIN_TOKENS=300
CHUNK_MAX_TOKENS=700

# Retrieval
RETRIEVAL_TOP_K=20
RAG_FINAL_K=8

# Generation
RAG_ANSWER_MAX_TOKENS=2048
SUMMARY_FINAL_MAX_TOKENS=4096

# Cleanup
AUTO_CLEANUP_AUDIO=true
```

---

## Project Structure

```
app/
├── main.py                     # FastAPI application
├── api/
│   ├── videos.py               # Video ingestion endpoints
│   └── chat.py                 # RAG Q&A endpoint
├── config/
│   └── settings.py             # Pydantic settings from .env
├── model_servers/
│   └── manager.py              # llama-server lifecycle manager
├── ingestion/
│   ├── youtube.py              # YouTube metadata extraction
│   ├── captions.py             # Caption/subtitle download
│   └── audio.py                # Audio download for Whisper
├── transcription/
│   └── whisper.py              # faster-whisper transcription
├── processing/
│   ├── cleaner.py              # Transcript text cleaning
│   ├── speaker_chunker.py      # Speaker-aware chunking
│   └── metadata.py             # Heuristic metadata extraction
├── embeddings/
│   └── qwen.py                 # Embedding client
├── vectorstore/
│   └── qdrant.py               # Qdrant vector store client
├── retrieval/
│   ├── retriever.py            # Retrieval pipeline + relevance gate
│   ├── hybrid_retriever.py     # Vector + BM25 with RRF fusion
│   ├── cross_encoder.py        # Cross-encoder reranking
│   └── bm25_index.py           # BM25 keyword index
├── llm/
│   └── llama_client.py         # Generation LLM client
├── models/
│   └── schemas.py              # Pydantic models/schemas
├── prompts/
│   ├── rag_answer.txt          # Evidence-based RAG prompt
│   └── summary.txt             # Summary generation prompt
├── cli/
│   └── helpers.py              # Rich formatting helpers
└── services/
    └── pipeline.py             # End-to-end pipeline orchestration
scripts/
├── cli.py                      # Command-line interface
├── test_pipeline.py            # Full pipeline test
├── test_model_servers.py       # Model server test
├── test_qdrant.py              # Qdrant connection test
├── test_embedding.py           # Embedding test
└── test_llm.py                 # LLM generation test
tests/
├── test_chunker.py             # Speaker-aware chunker tests
├── test_cleaner.py             # Cleaner tests
└── test_model_servers.py       # Server lifecycle tests (39 tests)
```

---

## Tests

```bash
# Run all tests (47 total)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model_servers.py -v

# Lint
uv run ruff check app/ scripts/ tests/
```

### Test Coverage

| Test File | Tests | What's Tested |
|-----------|-------|---------------|
| test_chunker.py | 4 | Speaker-aware chunking |
| test_cleaner.py | 4 | Text cleaning |
| test_model_servers.py | 39 | Server lifecycle, health, kill, adopt, release |

**Total: 47 tests, all passing**

---

## Dependencies

```bash
# Core
fastapi
uvicorn
httpx
pydantic
pydantic-settings

# Vector store
qdrant-client

# Embeddings & Reranking
sentence-transformers

# Search
rank-bm25

# Transcription
faster-whisper
yt-dlp

# CLI
rich

# Dev
pytest
pytest-asyncio
ruff
```

---

## How It Works

### Ingestion Flow
1. Fetch YouTube metadata (title, duration, etc.)
2. Try captions first, fallback to Whisper transcription
3. Clean transcript (remove markers, normalize whitespace)
4. Speaker-aware chunking (300-700 tokens per chunk)
5. Extract metadata (summary + keywords via heuristics)
6. Generate embeddings (Qwen3-0.6B)
7. Store in Qdrant + build BM25 index

### RAG Q&A Flow
1. User asks a question
2. Hybrid retrieval: vector search + BM25 search
3. Reciprocal Rank Fusion merges results
4. Add neighboring chunks (±1)
5. Cross-encoder reranking
6. Relevance gate checks evidence quality
7. LLM generates evidence-based answer with timestamps

### Summary Flow
1. Retrieve top diverse chunks (broad query)
2. Select context within token budget (8000 tokens)
3. Format evidence with timestamps
4. LLM generates structured markdown summary

---

## Known Limitations

- LFM2.5-2.6B sometimes returns empty content (metadata uses heuristics only)
- Progress bar output is noisy on Windows (cosmetic)
- No persistent video store (cached to JSON files)
- No web UI (CLI + API only)
- Single-session — no concurrent user support

---

## Future Improvements

- Add web UI for browsing videos and asking questions
- Add persistent video database (SQLite/PostgreSQL)
- Add conversation history for follow-up questions
- Add batch ingestion from playlists
- Add support for multiple languages
- Add audio file caching with configurable cleanup
- Add API authentication
- Add WebSocket for real-time progress updates

---

## Commands to Run

```bash
# Setup
podman machine start
podman compose up -d
uv sync

# Ingest
uv run python scripts/cli.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Ask
uv run python scripts/cli.py ask VIDEO_ID "question"

# Summary
uv run python scripts/cli.py summary VIDEO_ID

# Search
uv run python scripts/cli.py search VIDEO_ID "query"

# Tests
uv run pytest tests/ -v
```

---

## Stats

- **Files created:** 30+ (after cleanup)
- **Lines of code:** 2500+
- **Tests:** 47 (all passing)
- **Dependencies:** 15+
- **Models used:** 5
- **CLI commands:** 9
- **API endpoints:** 5
- **Dead code removed:** 12 files
