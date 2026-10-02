# Quick Start — YouTube Local RAG Summarizer

Get up and running in 3 steps.

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Podman (or Docker)
- NVIDIA GPU with CUDA (8GB VRAM minimum)
- llama-server.exe and model files

## 1. Start Qdrant

```bash
podman machine start
podman compose up -d
```

## 2. Install & Run

```bash
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 3. Use It

### Ingest a video
```bash
uv run python scripts/cli.py ingest "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Ask a question
```bash
uv run python scripts/cli.py ask VIDEO_ID "What is this video about?"
```

### Generate a summary
```bash
uv run python scripts/cli.py summary VIDEO_ID
```

### Search chunks
```bash
uv run python scripts/cli.py search VIDEO_ID "machine learning"
```

### Or use the API
```bash
# Ingest
curl -X POST http://localhost:8000/videos \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'

# Ask
curl -X POST http://localhost:8000/videos/VIDEO_ID/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this video about?"}'
```

## That's It

The app automatically:
- Fetches YouTube captions (or downloads audio + Whisper transcription)
- Chunks at conversation boundaries with speaker tracking
- Generates embeddings + BM25 index
- Retrieves via hybrid search (vector + BM25)
- Reranks with cross-encoder
- Answers with evidence citations and timestamps
- Generates structured summaries on demand

Model servers start on-demand and stop when not needed.
