"""Central manager for all llama-server process lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class ServerType(str, Enum):
    GENERATION = "generation"
    EMBEDDING = "embedding"


class ServerState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class ManagedServer:
    server_type: ServerType
    state: ServerState = ServerState.STOPPED
    process: subprocess.Popen | None = None  # type: ignore[type-arg]
    started_by_application: bool = False
    health_url: str = ""
    reference_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if not self.health_url:
            if self.server_type == ServerType.GENERATION:
                self.health_url = f"{settings.generation_base_url}/health"
            else:
                self.health_url = f"{settings.embedding_base_url}/health"


class ModelServerManager:
    """Manages lifecycle of llama-server processes.

    Responsibilities:
    - Start / stop / health-check both server types.
    - Prevent duplicate starts via async locks.
    - Track ownership (started by app vs pre-existing).
    - Reference-count active users so servers are not stopped prematurely.
    """

    def __init__(self) -> None:
        self._servers: dict[ServerType, ManagedServer] = {
            ServerType.GENERATION: ManagedServer(server_type=ServerType.GENERATION),
            ServerType.EMBEDDING: ManagedServer(server_type=ServerType.EMBEDDING),
        }
        self._locks: dict[ServerType, asyncio.Lock] = {
            ServerType.GENERATION: asyncio.Lock(),
            ServerType.EMBEDDING: asyncio.Lock(),
        }

    # ── Convenience helpers ─────────────────────────────────────────

    @property
    def generation_server(self) -> ManagedServer:
        return self._servers[ServerType.GENERATION]

    @property
    def embedding_server(self) -> ManagedServer:
        return self._servers[ServerType.EMBEDDING]

    # ── Health checking ─────────────────────────────────────────────

    async def _check_health(self, server: ManagedServer) -> bool:
        """Poll the /health endpoint and return True if reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(server.health_url)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def _wait_for_ready(self, server: ManagedServer) -> bool:
        """Poll health endpoint until timeout."""
        timeout = settings.model_server_startup_timeout
        interval = settings.model_server_health_interval
        elapsed = 0.0

        while elapsed < timeout:
            # Check if process already exited
            if server.process and server.process.poll() is not None:
                exit_code = server.process.returncode
                stdout = ""
                stderr = ""
                if server.process.stdout:
                    try:
                        stdout = server.process.stdout.read().decode(errors="replace")
                    except Exception:
                        pass
                if server.process.stderr:
                    try:
                        stderr = server.process.stderr.read().decode(errors="replace")
                    except Exception:
                        pass
                logger.error(
                    "llama-server (%s) exited with code %d during startup.\nstdout: %s\nstderr: %s",
                    server.server_type.value,
                    exit_code,
                    stdout[-2000:] if stdout else "",
                    stderr[-2000:] if stderr else "",
                )
                server.state = ServerState.FAILED
                return False

            if await self._check_health(server):
                server.state = ServerState.RUNNING
                logger.info("llama-server (%s) is healthy.", server.server_type.value)
                return True

            await asyncio.sleep(interval)
            elapsed += interval

        logger.error(
            "llama-server (%s) failed health check within %ds timeout.",
            server.server_type.value,
            timeout,
        )
        server.state = ServerState.FAILED
        return False

    # ── Build command ───────────────────────────────────────────────

    def _build_generation_command(self) -> list[str]:
        cmd = [
            settings.llama_server_path,
            "--model", settings.generation_model,
            "--port", str(settings.generation_port),
            "--host", settings.generation_host,
            "--flash-attn", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
            "-c", "128000",
            "--n-gpu-layers", "all",
            "--load-mode", "mlock",
            "--threads", "7",
            "--ubatch-size", "512",
            "--batch-size", "2048",
            "--parallel", "1",
            "--no-warmup",
            "--reasoning-format", "deepseek",
            "--reasoning-preserve",
        ]
        return cmd

    def _build_embedding_command(self) -> list[str]:
        cmd = [
            settings.llama_server_path,
            "--model", settings.embedding_model,
            "--port", str(settings.embedding_port),
            "--host", settings.embedding_host,
            "--embedding",
            "--pooling", "last",
            "-ub", "8192",
            "-ngl", "99",
        ]
        return cmd

    # ── Start ───────────────────────────────────────────────────────

    async def _start_server(self, server_type: ServerType) -> None:
        """Start a llama-server subprocess (must hold the lock)."""
        server = self._servers[server_type]

        # Validate executable and model
        if not os.path.isfile(settings.llama_server_path):
            raise FileNotFoundError(
                f"llama-server executable not found: {settings.llama_server_path}"
            )

        if server_type == ServerType.GENERATION:
            model_path = settings.generation_model
        else:
            model_path = settings.embedding_model

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cmd = (
            self._build_generation_command()
            if server_type == ServerType.GENERATION
            else self._build_embedding_command()
        )

        logger.info("Starting llama-server (%s): %s", server_type.value, " ".join(cmd))

        # Create data dir for logs
        log_dir = Path("data") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        stdout_file = open(log_dir / f"{server_type.value}_stdout.log", "w")  # noqa: SIM115
        stderr_file = open(log_dir / f"{server_type.value}_stderr.log", "w")  # noqa: SIM115

        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creation_flags if sys.platform == "win32" else 0,
        )

        server.process = process
        server.started_by_application = True
        server.state = ServerState.STARTING

        # Wait for readiness
        ready = await self._wait_for_ready(server)
        if not ready:
            raise RuntimeError(
                f"Failed to start {server_type.value} llama-server. "
                "Check data/logs/ for stdout/stderr output."
            )

    # ── Public API ──────────────────────────────────────────────────

    async def ensure_generation_server(self) -> None:
        """Ensure the generation server is running and healthy."""
        async with self._locks[ServerType.GENERATION]:
            server = self.generation_server

            # Already running
            if server.state == ServerState.RUNNING and await self._check_health(server):
                server.reference_count += 1
                return

            # Pre-existing server we detected
            if server.state == ServerState.RUNNING:
                if await self._check_health(server):
                    server.reference_count += 1
                    return
                server.state = ServerState.STOPPED

            await self._start_server(ServerType.GENERATION)
            server.reference_count = 1

    async def ensure_embedding_server(self) -> None:
        """Ensure the embedding server is running and healthy."""
        async with self._locks[ServerType.EMBEDDING]:
            server = self.embedding_server

            if server.state == ServerState.RUNNING and await self._check_health(server):
                server.reference_count += 1
                return

            if server.state == ServerState.RUNNING:
                if await self._check_health(server):
                    server.reference_count += 1
                    return
                server.state = ServerState.STOPPED

            await self._start_server(ServerType.EMBEDDING)
            server.reference_count = 1

    async def release_generation_server(self) -> None:
        """Decrement reference count; stop server if owned and unused."""
        server = self.generation_server
        server.reference_count = max(0, server.reference_count - 1)

        if server.reference_count == 0 and server.started_by_application:
            await self.stop_generation_server()

    async def release_embedding_server(self) -> None:
        """Decrement reference count; stop server if owned and unused."""
        server = self.embedding_server
        server.reference_count = max(0, server.reference_count - 1)

        if server.reference_count == 0 and server.started_by_application:
            await self.stop_embedding_server()

    async def stop_generation_server(self) -> None:
        await self._stop_server(ServerType.GENERATION)

    async def stop_embedding_server(self) -> None:
        await self._stop_server(ServerType.EMBEDDING)

    async def _stop_server(self, server_type: ServerType) -> None:
        server = self._servers[server_type]

        if server.process is None:
            server.state = ServerState.STOPPED
            return

        # Only stop if we started it
        if not server.started_by_application:
            logger.info(
                "Skipping stop of %s — not started by this application.",
                server_type.value,
            )
            server.process = None
            server.state = ServerState.STOPPED
            return

        logger.info("Stopping llama-server (%s)…", server_type.value)

        try:
            server.process.terminate()
            try:
                server.process.wait(timeout=settings.model_server_shutdown_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Graceful shutdown timed out for %s; killing.",
                    server_type.value,
                )
                server.process.kill()
                server.process.wait(timeout=5)
        except Exception as exc:
            logger.error("Error stopping %s: %s", server_type.value, exc)

        server.process = None
        server.state = ServerState.STOPPED
        server.started_by_application = False
        logger.info("llama-server (%s) stopped.", server_type.value)

    async def stop_all(self) -> None:
        """Stop all application-owned servers."""
        await self.stop_generation_server()
        await self.stop_embedding_server()

    async def generation_is_healthy(self) -> bool:
        return await self._check_health(self.generation_server)

    async def embedding_is_healthy(self) -> bool:
        return await self._check_health(self.embedding_server)

    def get_server_status(self, server_type: ServerType) -> ServerState:
        return self._servers[server_type].state


# Module-level singleton
model_server_manager = ModelServerManager()
