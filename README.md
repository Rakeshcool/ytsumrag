# YouTube Local RAG Summarizer

A fully local YouTube video ingestion and RAG question-answering system. No external APIs — everything runs on your machine.

> **Quick start?** See [QUICKSTART.md](QUICKSTART.md)

## Features

- **YouTube Ingestion**: Auto-fetches captions or downloads audio for Whisper transcription
- **Speaker-Aware Chunking**: Chunks at conversation boundaries, tracks speakers, preserves complete thoughts
- **Multiple Representations**: Each chunk stores raw text, summary, keywords, timestamps, and speakers
- **Hybrid Retrieval**: Combines vector search (Qdrant) + BM25 keyword search with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Uses ms-marco-MiniLM-L-6-v2 for precise relevance scoring
- **Neighbor Expansion**: Retrieves neighboring chunks for context continuity
- **Relevance Gate**: Refuses to answer when evidence is weak (prevents hallucination)
- **Evidence-Based Answers**: LLM must cite evidence with timestamps, cannot use outside knowledge
- **On-Demand Summaries**: Generate structured summaries with topics, takeaways, and key quotes
- **Embeddings**: Qwen3-Embedding-0.6B via local `llama-server`
- **Vector Store**: Qdrant for similarity search and retrieval
- **Model Server Management**: Automatic lifecycle with orphan detection and reference counting
- **Rich CLI**: Colorized tables, panels, progress bars, and markdown rendering
- **Auto-cleanup**: Audio files automatically deleted after Whisper transcription

## Architecture

```
INGESTION
  YouTube URL → Transcript → Clean → Speaker-Aware Chunk → Metadata
      → Embed (Qwen3) → Store (Qdrant + BM25)

QUERY (RAG Q&A)
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

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **NVIDIA GPU with CUDA** — 8GB VRAM minimum
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — `llama-server` executable
- **[Podman](https://podman.io/)** — For running Qdrant (or Docker)
- **FFmpeg** — For audio processing

### Required Model Files

| Model | Purpose |
|-------|---------|
| `LFM2.5-2.6B-Q8_0.gguf` | Text generation (RAG answers, summaries) |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | Vector embeddings |

## Setup

### 1. Start Qdrant

```bash
podman machine start
podman compose up -d
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment

Edit `.env` with your paths:

```bash
LLAMA_SERVER_PATH=path/to/llama-server.exe
GENERATION_MODEL=path/to/LFM2.5-2.6B-Q8_0.gguf
EMBEDDING_MODEL=path/to/Qwen3-Embedding-0.6B-Q8_0.gguf
```

### 4. Run the Server

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/videos` | Ingest a YouTube video |
| `GET` | `/videos/{video_id}` | Get video metadata and status |
| `GET` | `/videos/{video_id}/transcript` | Get the full transcript |
| `POST` | `/videos/{video_id}/chat` | Ask a question (RAG) |
| `GET` | `/health` | Health check |

## CLI Commands

```bash
# Ingest a video
uv run python scripts/cli.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"

# Ask a question (hybrid retrieval + reranking)
uv run python scripts/cli.py ask VIDEO_ID "What is this video about?"

# Generate a structured summary
uv run python scripts/cli.py summary VIDEO_ID

# Search chunks (test retrieval)
uv run python scripts/cli.py search VIDEO_ID "machine learning" --top-k 5

# View transcript
uv run python scripts/cli.py transcript VIDEO_ID

# List all videos
uv run python scripts/cli.py list

# Health check
uv run python scripts/cli.py health

# Manage model servers
uv run python scripts/cli.py server start|stop|status
```

### Summary Output Format

The `summary` command generates structured markdown:

```markdown
# Overview
2-4 paragraphs summarizing the main topic

# Key Takeaways
- 5-10 bullet points

# Main Topics
## Topic 1: Name
Explanation...

# Important Concepts
- Concept: explanation

# Key Quotes
- "Quote" — [timestamp]

# Conclusion
Short synthesis
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_PATH` | — | Path to `llama-server` executable |
| `GENERATION_PORT` | `3000` | Generation server port |
| `GENERATION_MODEL` | — | Path to generation model |
| `EMBEDDING_PORT` | `8081` | Embedding server port |
| `EMBEDDING_MODEL` | — | Path to embedding model |
| `EMBEDDING_DIMENSION` | `1024` | Embedding vector dimension |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant connection URL |
| `CHUNK_MIN_TOKENS` | `300` | Min tokens per chunk |
| `CHUNK_MAX_TOKENS` | `700` | Max tokens per chunk |
| `RETRIEVAL_TOP_K` | `20` | Initial retrieval count |
| `RAG_FINAL_K` | `8` | Final context count |
| `RAG_ANSWER_MAX_TOKENS` | `2048` | Max tokens for RAG answers |
| `SUMMARY_FINAL_MAX_TOKENS` | `4096` | Max tokens for summaries |
| `AUTO_CLEANUP_AUDIO` | `true` | Delete audio after transcription |

## Model Server Lifecycle

| Server | Port | Purpose |
|--------|------|---------|
| Generation | 3000 | RAG answers, summaries |
| Embedding | 8081 | Vector embeddings |

- **On-demand startup**: Started only when needed
- **Pre-existing detection**: Adopts servers already running on the port
- **Orphan detection**: Kills unhealthy processes before starting new ones
- **Reference counting**: Won't stop while operations depend on it
- **Sequential usage**: Embedding stops before generation starts (8GB VRAM safe)

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

## Troubleshooting

### Qdrant Connection Failed
```bash
podman compose down && podman compose up -d
```

### Model Server Won't Start
- Verify `LLAMA_SERVER_PATH` points to the executable
- Check CUDA: `nvidia-smi`
- Check logs: `data/logs/`

### VRAM Out of Memory
- System uses sequential model usage (embedding stops before generation)
- Reduce `--n-gpu-layers` if needed
- Use smaller Whisper model (`medium` instead of `large-v3`)

### Multiple llama-server Processes
- Kill orphaned processes: `taskkill /F /IM llama-server.exe`
- The manager now detects and adopts pre-existing servers

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check app/ scripts/ tests/

# Add dependencies
uv add package-name
```

## License

This project is for personal/local use.
