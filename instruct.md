# YouTube Local RAG Summarizer — Implementation Instructions

## 1. Project Objective

Build a fully local YouTube video ingestion, summarization, and RAG question-answering system.

The system must:

1. Accept a YouTube URL.
2. Obtain YouTube captions when available.
3. Fall back to downloading audio when captions are unavailable.
4. Use `faster-whisper` for transcription.
5. Produce a timestamped transcript.
6. Clean and intelligently chunk the transcript.
7. Generate embeddings using `Qwen3-Embedding-0.6B-GGUF`.
8. Run the embedding model through `llama-server`.
9. Store embeddings and metadata in Qdrant.
10. Generate complete video summaries using hierarchical summarization.
11. Allow users to ask questions about videos using RAG.
12. Use the local generation model through `llama-server`.
13. Start model servers automatically when required.
14. Stop model servers when they are no longer needed, where safe to do so.
15. Return timestamps for retrieved information whenever possible.
16. Keep the architecture modular so models and inference configurations can be changed later.

Do NOT implement vision/video-frame processing in V1.

---

# 2. Target Hardware

The target machine is:

```text
GPU:
NVIDIA GeForce RTX 5060

Architecture:
Blackwell

VRAM:
8 GB

System RAM:
32 GB
```

The system is intended for local AI inference.

Hardware-aware decisions should consider the 8 GB VRAM constraint.

Do not assume that multiple large models can always remain loaded simultaneously.

---

# 3. Model Server Architecture

The application owns and manages both llama-server processes.

There are two independent llama-server instances:

```text
Generation Server
    ↓
localhost:3000

Embedding Server
    ↓
localhost:8081
```

The Python application is responsible for:

```text
starting
monitoring
health checking
reusing
and safely stopping
```

these processes.

The application must NOT assume that either server is already running.

---

# 4. Generation Model

Generation uses:

```text
LFM2.5-2.6B-Q8_0.gguf
```

Model path:

```text
D:\AI\build\llama_official\models\LFM2.5-2.6B-Q8_0.gguf
```

The generation server must be started with the user's specified configuration:

```text
llama-server.exe
--model D:\AI\build\llama_official\models\LFM2.5-2.6B-Q8_0.gguf
--port 3000
--host 0.0.0.0
--flash-attn on
--cache-type-k q4_0
--cache-type-v q4_0
-c 128000
--n-gpu-layers all
--load-mode mlock
--threads 7
--ubatch-size 512
--batch-size 2048
--parallel 1
--no-warmup
--reasoning-format deepseek
--reasoning-preserve
```

These parameters should NOT be scattered throughout the code.

Store them in configuration.

The application should construct the command from configuration.

---

# 5. Embedding Model

Embedding uses:

```text
Qwen3-Embedding-0.6B-Q8_0.gguf
```

The model is expected to be available as:

```text
Qwen3-Embedding-0.6B-Q8_0.gguf
```

The embedding server must use:

```text
--embedding
--pooling last
-ub 8192
-ngl 99
--host 127.0.0.1
--port 8081
```

Conceptually the configuration is:

```text
llama-server.exe
-m Qwen3-Embedding-0.6B-Q8_0.gguf
--embedding
--pooling last
-ub 8192
-ngl 99
--host 127.0.0.1
--port 8081
```

The model path must be configurable.

Do NOT hardcode the model path into application logic.

---

# 6. Model Server Lifecycle

Create a dedicated model-server manager.

Recommended module:

```text
app/
└── model_servers/
    ├── __init__.py
    ├── manager.py
    ├── generation.py
    └── embedding.py
```

Do NOT use raw `subprocess.Popen()` calls throughout the application.

All process management should go through the server manager.

---

# 7. Server Manager Responsibilities

The server manager must provide functionality similar to:

```python
class ModelServerManager:

    async def ensure_generation_server(self):
        ...

    async def ensure_embedding_server(self):
        ...

    async def stop_generation_server(self):
        ...

    async def stop_embedding_server(self):
        ...

    async def stop_all(self):
        ...

    async def generation_is_healthy(self):
        ...

    async def embedding_is_healthy(self):
        ...
```

The manager should:

1. Check whether the required server is already running.
2. If healthy, reuse it.
3. If not running, start it.
4. Wait for the HTTP endpoint to become ready.
5. Only return once the server is actually usable.
6. Detect startup failures.
7. Capture stdout/stderr into application logs.
8. Track the process ID.
9. Shut down processes cleanly.
10. Kill the process only if graceful shutdown fails.

---

# 8. Do Not Start Servers Unnecessarily

The application should start model servers based on actual requirements.

For example:

### Embedding required

During ingestion:

```text
YouTube
   ↓
Transcript
   ↓
Chunks
   ↓
Embedding required
   ↓
ensure_embedding_server()
   ↓
localhost:8081
```

### Generation required

For summarization:

```text
Transcript
   ↓
Summary required
   ↓
ensure_generation_server()
   ↓
localhost:3000
```

For chat:

```text
User question
   ↓
Retrieval
   ↓
Generation required
   ↓
ensure_generation_server()
   ↓
localhost:3000
```

---

# 9. Server Reuse

Do not start a second instance if the required server is already running.

Before starting:

```text
Check:
localhost:3000
```

or:

```text
localhost:8081
```

If the existing server is healthy, reuse it.

The manager should distinguish between:

```text
server started by this application
```

and:

```text
server already running before application startup
```

This is important for safe shutdown.

---

# 10. Ownership Tracking

Track ownership explicitly.

Example:

```python
ServerState(
    process=None,
    started_by_application=False,
    healthy=True,
)
```

If the application discovers an already-running server:

```text
started_by_application = False
```

If the application launches it:

```text
started_by_application = True
```

The application MUST NOT automatically terminate a server that it did not start.

Therefore:

```text
Existing server
      ↓
Application uses it
      ↓
Application shutdown
      ↓
Server remains running
```

But:

```text
Application starts server
      ↓
Application uses server
      ↓
Application shutdown
      ↓
Application may stop server
```

---

# 11. Server Startup Readiness

Starting a subprocess is NOT equivalent to the server being ready.

After launching llama-server:

```text
Popen()
   ↓
wait
   ↓
poll health endpoint
   ↓
server ready
```

Implement a configurable startup timeout.

Example:

```env
MODEL_SERVER_STARTUP_TIMEOUT=120
MODEL_SERVER_HEALTH_INTERVAL=0.5
```

Do not use an arbitrary fixed sleep such as:

```python
await asyncio.sleep(10)
```

as the readiness mechanism.

---

# 12. Server Failure Handling

If a server exits during startup:

```text
process.poll() != None
```

capture:

```text
return code
stdout
stderr
```

and raise a useful application-level error.

Example:

```text
Failed to start embedding llama-server.

Process exited with code 1.

Check:
- model path
- llama-server executable
- CUDA availability
- VRAM
- llama.cpp parameters
```

Do not hide the original error.

---

# 13. Configuration

Use environment variables.

Example:

```env
APP_HOST=127.0.0.1
APP_PORT=8000

# llama.cpp executable
LLAMA_SERVER_PATH=D:\AI\build\llama_official\llama-server.exe

# Generation server
GENERATION_HOST=0.0.0.0
GENERATION_PORT=3000
GENERATION_MODEL=D:\AI\build\llama_official\models\LFM2.5-2.6B-Q8_0.gguf

# Embedding server
EMBEDDING_HOST=127.0.0.1
EMBEDDING_PORT=8081
EMBEDDING_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf

# Qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=youtube_chunks

# Whisper
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Embedding
EMBEDDING_DIMENSION=1024

# Chunking
CHUNK_MAX_TOKENS=700
CHUNK_OVERLAP_TOKENS=100

# Retrieval
RETRIEVAL_TOP_K=20
RAG_FINAL_K=8

# Server lifecycle
MODEL_SERVER_STARTUP_TIMEOUT=120
MODEL_SERVER_HEALTH_INTERVAL=0.5
MODEL_SERVER_SHUTDOWN_TIMEOUT=15
```

All paths must be configurable.

---

# 14. Windows Compatibility

The target environment is Windows.

The model-server manager must work correctly on Windows.

Use:

```python
subprocess.Popen
```

through an isolated process-management layer.

Do not depend on Unix-only commands.

Do not use:

```text
kill
pkill
bash
systemctl
```

for process management.

Use Windows-compatible Python process APIs.

---

# 15. llama-server Executable

The llama-server executable path is configurable:

```env
LLAMA_SERVER_PATH=D:\AI\build\llama_official\llama-server.exe
```

Before starting a server:

1. Verify the executable exists.
2. Verify the model path exists.
3. Produce a clear error if either is missing.

Do not automatically download llama.cpp or model files.

---

# 16. Generation Client

The generation client communicates with:

```text
http://127.0.0.1:3000
```

even though the generation server binds to:

```text
0.0.0.0
```

Use:

```text
/v1/chat/completions
```

The generation client should call:

```python
await model_server_manager.ensure_generation_server()
```

before making generation requests.

---

# 17. Embedding Client

The embedding client communicates with:

```text
http://127.0.0.1:8081
```

Use:

```text
/v1/embeddings
```

Before embedding:

```python
await model_server_manager.ensure_embedding_server()
```

Then perform the request.

---

# 18. Server Lifecycle During Ingestion

For a typical ingestion request:

```text
POST /videos
        │
        ▼
YouTube ingestion
        │
        ▼
Transcript
        │
        ▼
Chunking
        │
        ▼
ensure embedding server
        │
        ▼
Embedding
        │
        ▼
Qdrant
        │
        ▼
ensure generation server
        │
        ▼
Hierarchical summarization
```

The application may stop the embedding server after ingestion if:

```text
it was started by the application
```

and:

```text
no other task is using it
```

Do not stop it while another operation is actively using it.

---

# 19. Server Reference Counting / Usage Tracking

Implement lightweight usage tracking.

Conceptually:

```text
Embedding users:
    ingestion job A
    ingestion job B

Generation users:
    summary job A
    chat request B
```

Do not stop a server while active operations depend on it.

A simple async lock/reference counter is sufficient for V1.

Do not introduce a distributed service.

---

# 20. Avoid Concurrent Startup Races

If two requests simultaneously require the same server:

```text
Request A → ensure embedding
Request B → ensure embedding
```

they must NOT start two servers.

Use an async lock:

```python
asyncio.Lock()
```

around server startup.

Desired behavior:

```text
Request A
   ↓
acquires startup lock
   ↓
starts server
   ↓
waits for readiness
   ↓
releases lock

Request B
   ↓
waits
   ↓
sees healthy server
   ↓
reuses it
```

---

# 21. GPU / VRAM Considerations

The target GPU is:

```text
RTX 5060
8 GB VRAM
Blackwell
```

The system must account for limited VRAM.

The application should not attempt to automatically modify llama.cpp GPU parameters.

The configured generation server uses:

```text
--n-gpu-layers all
```

The configured embedding server uses:

```text
-ngl 99
```

These values are part of the user's chosen configuration.

Do not override them dynamically.

If simultaneous model loading causes a CUDA/VRAM failure, report the llama-server error clearly.

---

# 22. Potential Sequential Model Usage

Because the GPU has 8 GB VRAM, the application should support sequential model usage.

For example:

```text
Embedding phase
      ↓
Embedding server active
      ↓
Finish embedding
      ↓
Stop embedding server
      ↓
Generation phase
      ↓
Generation server active
```

This should be possible without changing the application architecture.

However, do not blindly stop one server whenever another is requested.

Use lifecycle ownership and active-operation tracking.

---

# 23. Project Structure

Use:

```text
youtube-rag/
│
├── app/
│   ├── main.py
│
│   ├── api/
│   │   ├── videos.py
│   │   ├── summary.py
│   │   └── chat.py
│
│   ├── config/
│   │   └── settings.py
│
│   ├── model_servers/
│   │   ├── manager.py
│   │   ├── generation.py
│   │   └── embedding.py
│
│   ├── ingestion/
│   │   ├── youtube.py
│   │   ├── captions.py
│   │   └── audio.py
│
│   ├── transcription/
│   │   └── whisper.py
│
│   ├── processing/
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── topics.py
│
│   ├── embeddings/
│   │   └── qwen.py
│
│   ├── vectorstore/
│   │   └── qdrant.py
│
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── reranker.py
│
│   ├── summarization/
│   │   ├── chunk_summary.py
│   │   ├── hierarchical.py
│   │   └── final_summary.py
│
│   ├── llm/
│   │   └── llama_client.py
│
│   ├── models/
│   │   └── schemas.py
│
│   ├── prompts/
│   │   ├── chunk_summary.txt
│   │   ├── section_summary.txt
│   │   ├── final_summary.txt
│   │   ├── rag_answer.txt
│   │   └── topic_detection.txt
│
│   └── services/
│       └── pipeline.py
│
├── data/
│   ├── audio/
│   ├── transcripts/
│   ├── summaries/
│   └── cache/
│
├── scripts/
│   ├── test_embedding.py
│   ├── test_qdrant.py
│   ├── test_llm.py
│   └── test_model_servers.py
│
├── tests/
│
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

# 24. Python Environment — uv

Use `uv` exclusively for Python environment and dependency management.

Initialize:

```text
uv init
```

Install:

```text
uv add fastapi uvicorn httpx pydantic pydantic-settings python-dotenv yt-dlp qdrant-client faster-whisper
```

Development dependencies:

```text
uv add --dev pytest pytest-asyncio ruff
```

Run:

```text
uv run pytest
```

and:

```text
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Do not use pip directly.

---

# 25. YouTube Ingestion

Preferred order:

```text
1. Existing manual captions
2. Automatic captions
3. Audio download
4. faster-whisper
```

Do not download the full video.

Use `yt-dlp`.

Normalize both YouTube captions and Whisper output into the same timestamped transcript schema.

---

# 26. Transcript Schema

```python
class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    video_id: str
    title: str | None
    language: str | None
    segments: list[TranscriptSegment]
```

Never discard timestamps.

---

# 27. Transcript Chunking

Use sentence-aware chunking.

Initial configuration:

```text
500–700 tokens
~100 token overlap
```

Each chunk must preserve:

```text
chunk_id
video_id
start
end
text
token_count
section_id
section_title
```

---

# 28. Qwen Embeddings

Embedding model:

```text
Qwen3-Embedding-0.6B-GGUF
```

Expected vector:

```text
1024 dimensions
```

Use:

```text
POST /v1/embeddings
```

Batch requests whenever possible.

Validate vector dimensionality.

---

# 29. Qdrant

Use:

```text
http://127.0.0.1:6333
```

Collection:

```text
youtube_chunks
```

Configuration:

```text
dimension = 1024
distance = cosine
```

Store:

```text
video_id
chunk_id
start
end
text
section_id
section_title
```

as payload.

---

# 30. Retrieval

Pipeline:

```text
Question
   ↓
Query instruction
   ↓
Qwen embedding
   ↓
Qdrant
   ↓
Top 20
   ↓
Context selection
   ↓
Top 8
   ↓
Generation LLM
```

Always filter retrieval by `video_id`.

---

# 31. Reranking

Reranking is optional for V1.

Create an interface:

```python
class Reranker:

    async def rerank(
        self,
        query,
        chunks
    ):
        ...
```

Initially use a no-op implementation.

---

# 32. Full Video Summarization

Do not use top-K RAG to generate the complete video summary.

Use hierarchical summarization:

```text
Transcript
    ↓
Chunks
    ↓
Chunk summaries
    ↓
Section summaries
    ↓
Final summary
```

All generation operations must use the user-configured generation server.

---

# 33. Generation Prompt

Use a grounded prompt for RAG:

```text
You are answering a question about a YouTube video.

Use ONLY the supplied transcript context.

If the context does not contain enough information
to answer the question, say that the transcript does
not contain enough information.

Do not invent facts.

Include relevant timestamps when possible.

CONTEXT:

{context}

QUESTION:

{question}
```

---

# 34. API

Implement:

```text
POST /videos
GET  /videos/{video_id}
GET  /videos/{video_id}/transcript
GET  /videos/{video_id}/summary
GET  /videos/{video_id}/sections
POST /videos/{video_id}/chat
GET  /health
```

---

# 35. Health Checks

Health checks should verify:

```text
FastAPI
Qdrant
Embedding server
Generation server
```

But a model server being stopped is NOT necessarily an application failure.

For example:

```text
Embedding: stopped
Generation: stopped
Qdrant: healthy
API: healthy
```

can be a valid idle state.

The health endpoint should distinguish:

```text
healthy
running
stopped
starting
failed
```

for model servers.

---

# 36. Application Shutdown

On FastAPI shutdown:

1. Stop accepting new work.
2. Wait for active model operations to finish.
3. Stop model servers that were started by the application.
4. Do NOT stop servers that were already running before application startup.
5. Close HTTP clients.
6. Close other resources.

Use graceful termination first.

Only force-kill after the configured shutdown timeout.

---

# 37. Model Server Logs

Capture model-server stdout/stderr.

Store logs through the application logger.

Do not allow a child process pipe to block because its output is not being consumed.

Model-server logs should make debugging CUDA/VRAM/startup failures straightforward.

---

# 38. Error Handling

Handle:

```text
Missing llama-server executable
Missing model file
Server startup failure
Server crash
Server health timeout
Embedding endpoint failure
Generation endpoint failure
Qdrant failure
Whisper failure
yt-dlp failure
Invalid YouTube URL
Empty transcript
Malformed LLM response
CUDA failure
VRAM failure
```

Return useful errors.

Do not expose raw internal stack traces through the API.

---

# 39. Testing Model Server Management

Create:

```text
scripts/test_model_servers.py
```

Test:

```text
1. Start embedding server.
2. Wait for readiness.
3. Verify /v1/embeddings.
4. Reuse existing embedding server.
5. Start generation server.
6. Verify /v1/chat/completions.
7. Verify duplicate startup does not occur.
8. Verify shutdown behavior.
```

For automated tests, allow server commands to be mocked.

Do not require actual model startup for every unit test.

---

# 40. Concurrency

Use async locks for server startup.

Never allow:

```text
two requests
    ↓
two instances of llama-server
```

for the same port.

Use one managed instance per service.

---

# 41. Performance

Priorities:

1. Correctness
2. Retrieval quality
3. VRAM stability
4. GPU utilization
5. Embedding throughput
6. Generation throughput
7. RAM efficiency
8. Disk I/O

Batch embedding requests.

Reuse the Whisper model.

Cache expensive processing.

Do not unnecessarily keep both model servers loaded.

---

# 42. Caching

Cache:

```text
metadata
raw transcript
clean transcript
chunks
embeddings
chunk summaries
section summaries
final summary
```

Use YouTube video ID as the stable identifier.

Do not rerun Whisper if a valid transcript exists.

Do not regenerate embeddings when the chunks and embedding configuration have not changed.

---

# 43. No Vision in V1

Do NOT implement:

```text
video frame extraction
vision models
multimodal embeddings
audio embeddings
video embeddings
```

V1 is:

```text
Audio/captions
      ↓
Transcript
      ↓
Chunks
      ↓
Embeddings
      ↓
Qdrant
      ↓
RAG
      ↓
Local LLM
```

---

# 44. Future Multimodal Architecture

Keep interfaces extensible for:

```text
YouTube
   │
   ├── Audio → Whisper → Transcript
   │
   └── Video → Vision → Visual information
                         │
                         ▼
                  Multimodal embeddings
                         │
                         ▼
                       Qdrant
                         │
                         ▼
                    Local LLM
```

Do not implement this in V1.

---

# 45. Strict Responsibility Separation

```text
yt-dlp
    ↓
YouTube acquisition

faster-whisper
    ↓
transcription

Qwen3-Embedding-0.6B
    ↓
embedding generation

Qdrant
    ↓
vector storage + retrieval

Generation llama-server
    ↓
summarization + reasoning + answers

ModelServerManager
    ↓
process lifecycle management
```

The `ModelServerManager` is the ONLY component allowed to start, stop, or monitor llama-server processes.

---

# 46. Final Runtime Architecture

```text
                         FastAPI :8000
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
          Video Pipeline                Chat API
                │                           │
                ▼                           ▼
        Transcript / RAG              Retrieval
                │                           │
                ▼                           │
       ┌────────────────┐                  │
       │ Model Manager  │◄─────────────────┘
       └───────┬────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
Embedding Server    Generation Server
     :8081               :3000
       │                │
       ▼                ▼
Qwen3-Embedding       LFM2.5-2.6B
     0.6B                Q8_0
       │                │
       └───────┬────────┘
               │
               ▼
             Qdrant
              :6333
```

---

# 47. Agent Rules

You are implementing this project as a coding agent.

Follow these rules strictly:

1. Use `uv` exclusively for Python dependency management.
2. Use `uv run` for project commands.
3. Never use pip directly.
4. Generation llama-server is managed by the application.
5. Embedding llama-server is managed by the application.
6. Do not start llama-server from random modules.
7. All llama-server lifecycle operations must go through `ModelServerManager`.
8. Do not start duplicate servers.
9. Reuse healthy existing servers.
10. Track whether a server was started by the application.
11. Never terminate a server that the application did not start.
12. Preserve timestamps throughout the pipeline.
13. Batch embedding requests.
14. Cache expensive processing.
15. Keep embedding and generation independent.
16. Do not implement vision in V1.
17. Do not introduce Redis, Celery, Kafka, or Kubernetes.
18. Do not introduce LangChain or LlamaIndex unless explicitly requested.
19. Do not silently swallow exceptions.
20. Keep model paths and llama.cpp arguments configurable.
21. Do not automatically download model files.
22. Do not dynamically alter the user's llama.cpp parameters.
23. Account for the RTX 5060's 8 GB VRAM.
24. Use system RAM efficiently because the machine has 32 GB RAM.
25. Prevent concurrent startup races.
26. Wait for actual HTTP readiness after launching llama-server.
27. Capture llama-server stdout/stderr.
28. Gracefully shut down application-owned servers.
29. Run tests after meaningful changes.
30. Do not declare the project complete until both hierarchical summarization and RAG Q&A work.

---

# 48. Definition of Done

The user runs only the Python application.

The application manages:

```text
Embedding llama-server
localhost:8081

Generation llama-server
localhost:3000
```

and communicates with:

```text
Qdrant
localhost:6333
```

The user submits:

```text
POST /videos
```

with:

```json
{
    "url": "YOUTUBE_URL"
}
```

The application then automatically:

```text
YouTube
   ↓
Captions / Audio
   ↓
faster-whisper
   ↓
Transcript
   ↓
Chunking
   ↓
Start embedding llama-server if required
   ↓
Qwen3 embeddings
   ↓
Qdrant
   ↓
Stop embedding server when safe
   ↓
Start generation llama-server if required
   ↓
Hierarchical summarization
   ↓
Persist summary
```

For a user question:

```text
Question
   ↓
Start embedding server if required
   ↓
Embed query
   ↓
Qdrant retrieval
   ↓
Stop embedding server when safe
   ↓
Start generation server if required
   ↓
Grounded answer
   ↓
Timestamps
```

The complete system must operate locally without an external LLM API.
