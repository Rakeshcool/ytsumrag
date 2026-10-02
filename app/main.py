"""FastAPI application — YouTube Local RAG Summarizer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.videos import router as videos_router
from app.config.settings import settings
from app.model_servers.manager import model_server_manager
from app.models.schemas import HealthComponent, HealthResponse, ServiceStatus
from app.vectorstore.qdrant import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info("Starting YouTube RAG Summarizer…")

    # Ensure Qdrant collection exists
    try:
        vector_store.ensure_collection()
        logger.info("Qdrant collection ready.")
    except Exception as exc:
        logger.warning("Could not connect to Qdrant at startup: %s", exc)

    # Detect any pre-existing model servers
    gen_healthy = await model_server_manager.generation_is_healthy()
    emb_healthy = await model_server_manager.embedding_is_healthy()

    if gen_healthy:
        from app.model_servers.manager import ServerState
        model_server_manager.generation_server.state = ServerState.RUNNING
        logger.info(
            "Detected existing generation server on port %d.",
            settings.generation_port,
        )
    if emb_healthy:
        from app.model_servers.manager import ServerState
        model_server_manager.embedding_server.state = ServerState.RUNNING
        logger.info(
            "Detected existing embedding server on port %d.",
            settings.embedding_port,
        )

    yield

    # Shutdown
    logger.info("Shutting down…")
    await model_server_manager.stop_all()
    vector_store.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="YouTube Local RAG Summarizer",
    description=(
        "Ingest YouTube videos, generate transcripts, embed,"
        " summarize, and answer questions — all locally."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(videos_router)

# Serve web UI
app.mount("/app", StaticFiles(directory="app/static", html=True), name="static")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of the app and all dependencies."""
    # Qdrant
    try:
        vector_store.client.get_collections()
        qdrant_status = HealthComponent(status=ServiceStatus.HEALTHY)
    except Exception as exc:
        qdrant_status = HealthComponent(status=ServiceStatus.FAILED, detail=str(exc))

    # Generation server
    gen_server = model_server_manager.generation_server
    gen_state = gen_server.state.value
    if gen_state == "running":
        gen_healthy = await model_server_manager.generation_is_healthy()
        gen_status = ServiceStatus.HEALTHY if gen_healthy else ServiceStatus.FAILED
    elif gen_state == "starting":
        gen_status = ServiceStatus.STARTING
    else:
        gen_status = ServiceStatus.STOPPED

    # Embedding server
    emb_server = model_server_manager.embedding_server
    emb_state = emb_server.state.value
    if emb_state == "running":
        emb_healthy = await model_server_manager.embedding_is_healthy()
        emb_status = ServiceStatus.HEALTHY if emb_healthy else ServiceStatus.FAILED
    elif emb_state == "starting":
        emb_status = ServiceStatus.STARTING
    else:
        emb_status = ServiceStatus.STOPPED

    return HealthResponse(
        app=ServiceStatus.HEALTHY,
        qdrant=qdrant_status,
        embedding_server=HealthComponent(status=emb_status),
        generation_server=HealthComponent(status=gen_status),
    )
