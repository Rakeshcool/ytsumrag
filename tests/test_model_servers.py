"""Comprehensive tests for model server manager."""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.model_servers.manager import ManagedServer, ModelServerManager, ServerState, ServerType

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def manager() -> ModelServerManager:
    """Create a fresh manager for each test."""
    ModelServerManager._orphans_cleaned = False
    with patch.object(ModelServerManager, "kill_all_orphans"):
        yield ModelServerManager()


@pytest.fixture
def mock_process() -> MagicMock:
    """Create a mock subprocess."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 12345
    proc.poll.return_value = None  # Still running
    proc.returncode = None
    proc.terminate.return_value = None
    proc.kill.return_value = None
    proc.wait.return_value = 0
    return proc


# ── ManagedServer Tests ───────────────────────────────────────────────


class TestManagedServer:
    def test_generation_server_health_url(self):
        server = ManagedServer(server_type=ServerType.GENERATION)
        assert "3000" in server.health_url

    def test_embedding_server_health_url(self):
        server = ManagedServer(server_type=ServerType.EMBEDDING)
        assert "8081" in server.health_url

    def test_is_process_alive_no_process(self):
        server = ManagedServer(server_type=ServerType.GENERATION)
        assert server.is_process_alive() is False

    def test_is_process_alive_with_process(self, mock_process):
        server = ManagedServer(server_type=ServerType.GENERATION)
        server.process = mock_process
        assert server.is_process_alive() is True

    def test_is_process_alive_dead_process(self, mock_process):
        mock_process.poll.return_value = 1  # Exited
        server = ManagedServer(server_type=ServerType.GENERATION)
        server.process = mock_process
        assert server.is_process_alive() is False

    def test_initial_state(self):
        server = ManagedServer(server_type=ServerType.GENERATION)
        assert server.state == ServerState.STOPPED
        assert server.started_by_application is False
        assert server.reference_count == 0


# ── ServerManager Init Tests ──────────────────────────────────────────


class TestManagerInit:
    def test_has_both_servers(self, manager):
        assert ServerType.GENERATION in manager._servers
        assert ServerType.EMBEDDING in manager._servers

    def test_has_both_locks(self, manager):
        assert ServerType.GENERATION in manager._locks
        assert ServerType.EMBEDDING in manager._locks

    def test_generation_server_property(self, manager):
        assert manager.generation_server.server_type == ServerType.GENERATION

    def test_embedding_server_property(self, manager):
        assert manager.embedding_server.server_type == ServerType.EMBEDDING


# ── Health Check Tests ────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_check_health_success(self, manager):
        server = manager.generation_server
        with patch("app.model_servers.manager.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=AsyncMock(get=AsyncMock(return_value=mock_resp))
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await manager._check_health(server)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self, manager):
        server = manager.generation_server
        with patch("app.model_servers.manager.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=AsyncMock(
                    get=AsyncMock(side_effect=ConnectionError("refused"))
                )
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await manager._check_health(server)
            assert result is False


# ── Kill Process Tests ────────────────────────────────────────────────


class TestKillProcess:
    def test_kill_process_none(self, manager):
        # Should not raise
        manager._kill_process(None)

    def test_kill_process_already_dead(self, manager, mock_process):
        mock_process.poll.return_value = 1  # Already exited
        manager._kill_process(mock_process)
        mock_process.terminate.assert_not_called()

    def test_kill_process_alive(self, manager, mock_process):
        manager._kill_process(mock_process)
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_kill_process_timeout_kills(self, manager, mock_process):
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=5),
            None,
        ]
        manager._kill_process(mock_process)
        mock_process.kill.assert_called_once()

    def test_kill_existing_resets_state(self, manager, mock_process):
        server = manager.generation_server
        server.process = mock_process
        server.started_by_application = True
        server.state = ServerState.RUNNING
        server.reference_count = 5

        manager._kill_existing(ServerType.GENERATION)

        assert server.process is None
        assert server.state == ServerState.STOPPED
        assert server.started_by_application is False
        assert server.reference_count == 0


# ── Command Building Tests ────────────────────────────────────────────


class TestCommandBuilding:
    def test_generation_command_has_model(self, manager):
        cmd = manager._build_generation_command()
        assert "--model" in cmd
        assert "--port" in cmd
        assert "--flash-attn" in cmd

    def test_embedding_command_has_embedding_flag(self, manager):
        cmd = manager._build_embedding_command()
        assert "--embedding" in cmd
        assert "--pooling" in cmd

    def test_no_reasoning_format(self, manager):
        cmd = manager._build_generation_command()
        assert "--reasoning-format" not in cmd


# ── Ensure Server Tests ───────────────────────────────────────────────


class TestEnsureServer:
    @pytest.mark.asyncio
    async def test_ensure_reuses_healthy_process(self, manager, mock_process):
        server = manager.generation_server
        server.process = mock_process
        server.state = ServerState.RUNNING
        server.reference_count = 0

        with patch.object(manager, "_check_health", new_callable=AsyncMock, return_value=True):
            await manager.ensure_generation_server()

        assert server.reference_count == 1
        mock_process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_kills_unhealthy_process(self, manager, mock_process):
        server = manager.generation_server
        server.process = mock_process
        server.state = ServerState.RUNNING
        server.started_by_application = True

        no_health = AsyncMock(return_value=False)
        no_adopt = AsyncMock(return_value=False)
        with patch.object(manager, "_check_health", side_effect=no_health):
            with patch.object(manager, "_adopt_pre_existing", side_effect=no_adopt):
                with patch.object(manager, "_start_server", new_callable=AsyncMock):
                    await manager.ensure_generation_server()

        # Process should have been killed
        mock_process.terminate.assert_called()

    @pytest.mark.asyncio
    async def test_ensure_adopts_pre_existing(self, manager):
        server = manager.generation_server
        assert server.process is None

        with patch.object(manager, "_check_health", new_callable=AsyncMock, return_value=True):
            await manager.ensure_generation_server()

        # Should adopt, not start new
        assert server.started_by_application is False
        assert server.state == ServerState.RUNNING
        assert server.reference_count == 1

    @pytest.mark.asyncio
    async def test_ensure_starts_new_if_nothing_exists(self, manager):
        server = manager.generation_server

        no_health = AsyncMock(return_value=False)
        no_adopt = AsyncMock(return_value=False)
        with patch.object(manager, "_check_health", side_effect=no_health):
            with patch.object(manager, "_adopt_pre_existing", side_effect=no_adopt):
                with patch.object(manager, "_start_server", new_callable=AsyncMock):
                    await manager.ensure_generation_server()

        assert server.reference_count == 1


# ── Release Server Tests ──────────────────────────────────────────────


class TestReleaseServer:
    @pytest.mark.asyncio
    async def test_release_decrements_count(self, manager):
        server = manager.generation_server
        server.reference_count = 3
        server.started_by_application = False

        await manager.release_generation_server()
        assert server.reference_count == 2

    @pytest.mark.asyncio
    async def test_release_stops_owned_server(self, manager):
        server = manager.generation_server
        server.reference_count = 1
        server.started_by_application = True
        server.process = None  # No process to stop

        await manager.release_generation_server()
        assert server.reference_count == 0

    @pytest.mark.asyncio
    async def test_release_does_not_stop_pre_existing(self, manager):
        server = manager.generation_server
        server.reference_count = 1
        server.started_by_application = False

        await manager.release_generation_server()
        # Should not have tried to stop
        assert server.reference_count == 0

    @pytest.mark.asyncio
    async def test_release_never_goes_negative(self, manager):
        server = manager.generation_server
        server.reference_count = 0
        server.started_by_application = False

        await manager.release_generation_server()
        assert server.reference_count == 0


# ── Stop Server Tests ─────────────────────────────────────────────────


class TestStopServer:
    @pytest.mark.asyncio
    async def test_stop_no_process(self, manager):
        server = manager.generation_server
        server.process = None
        server.state = ServerState.RUNNING

        await manager.stop_generation_server()
        assert server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_pre_existing_not_killed(self, manager, mock_process):
        server = manager.generation_server
        server.process = mock_process
        server.started_by_application = False

        await manager.stop_generation_server()
        mock_process.terminate.assert_not_called()
        assert server.process is None

    @pytest.mark.asyncio
    async def test_stop_owned_terminates(self, manager, mock_process):
        server = manager.generation_server
        server.process = mock_process
        server.started_by_application = True
        server.state = ServerState.RUNNING

        await manager.stop_generation_server()
        mock_process.terminate.assert_called_once()
        assert server.process is None
        assert server.state == ServerState.STOPPED
        assert server.started_by_application is False

    @pytest.mark.asyncio
    async def test_stop_all_calls_both(self, manager):
        with patch.object(manager, "stop_generation_server", new_callable=AsyncMock) as mock_gen:
            with patch.object(manager, "stop_embedding_server", new_callable=AsyncMock) as mock_emb:
                await manager.stop_all()
                mock_gen.assert_called_once()
                mock_emb.assert_called_once()


# ── Adopt Pre-existing Tests ──────────────────────────────────────────


class TestAdoptPreExisting:
    @pytest.mark.asyncio
    async def test_adopt_when_healthy(self, manager):
        server = manager.generation_server

        with patch.object(manager, "_check_health", new_callable=AsyncMock, return_value=True):
            result = await manager._adopt_pre_existing(ServerType.GENERATION)

        assert result is True
        assert server.state == ServerState.RUNNING
        assert server.started_by_application is False

    @pytest.mark.asyncio
    async def test_adopt_skips_if_process_alive(self, manager, mock_process):
        server = manager.generation_server
        server.process = mock_process

        result = await manager._adopt_pre_existing(ServerType.GENERATION)
        assert result is False

    @pytest.mark.asyncio
    async def test_adopt_skips_if_not_healthy(self, manager):
        with patch.object(manager, "_check_health", new_callable=AsyncMock, return_value=False):
            result = await manager._adopt_pre_existing(ServerType.GENERATION)

        assert result is False


# ── Kill All Orphans Tests ───────────────────────────────────────────


class TestKillAllOrphans:
    def test_kill_all_kills_llama_servers(self):
        """Test: only llama-server processes are killed, others are spared."""
        mock_proc1 = MagicMock()
        mock_proc1.info = {"pid": 100, "name": "llama-server.exe"}
        mock_proc2 = MagicMock()
        mock_proc2.info = {"pid": 200, "name": "python.exe"}
        mock_proc3 = MagicMock()
        mock_proc3.info = {"pid": 300, "name": "llama-server.exe"}

        procs = [mock_proc1, mock_proc2, mock_proc3]
        with patch("app.model_servers.manager.psutil.process_iter", return_value=procs):
            ModelServerManager.kill_all_orphans()

        mock_proc1.kill.assert_called_once()
        mock_proc2.kill.assert_not_called()  # Not llama-server
        mock_proc3.kill.assert_called_once()

    def test_kill_all_handles_access_denied(self):
        import psutil as _psutil

        mock_proc = MagicMock()
        mock_proc.info = {"pid": 100, "name": "llama-server.exe"}
        mock_proc.kill.side_effect = _psutil.AccessDenied(pid=100)

        with patch("app.model_servers.manager.psutil.process_iter", return_value=[mock_proc]):
            # Should not raise
            ModelServerManager.kill_all_orphans()

    def test_kill_all_handles_no_such_process(self):
        import psutil as _psutil

        mock_proc = MagicMock()
        mock_proc.info = {"pid": 100, "name": "llama-server.exe"}
        mock_proc.kill.side_effect = _psutil.NoSuchProcess(pid=100)

        with patch("app.model_servers.manager.psutil.process_iter", return_value=[mock_proc]):
            # Should not raise
            ModelServerManager.kill_all_orphans()

    def test_kill_all_empty_process_list(self):
        with patch("app.model_servers.manager.psutil.process_iter", return_value=[]):
            # Should not raise
            ModelServerManager.kill_all_orphans()


# ── Integration-style Tests ───────────────────────────────────────────


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_full_cycle_adopt_release(self, manager):
        """Test: adopt pre-existing → use → release (should NOT stop it)."""
        server = manager.generation_server

        # Adopt
        with patch.object(manager, "_check_health", new_callable=AsyncMock, return_value=True):
            await manager.ensure_generation_server()

        assert server.reference_count == 1
        assert server.started_by_application is False

        # Release
        await manager.release_generation_server()
        assert server.reference_count == 0
        # Should still be running (not owned)
        assert server.process is None  # No process tracked

    @pytest.mark.asyncio
    async def test_full_cycle_start_release(self, manager):
        """Test: start owned → use → release (should stop it)."""
        server = manager.generation_server

        async def fake_start(server_type):
            server.started_by_application = True
            server.state = ServerState.RUNNING

        # Start
        no_health = AsyncMock(return_value=False)
        no_adopt = AsyncMock(return_value=False)
        with patch.object(manager, "_check_health", side_effect=no_health):
            with patch.object(manager, "_adopt_pre_existing", side_effect=no_adopt):
                with patch.object(manager, "_start_server", side_effect=fake_start):
                    await manager.ensure_generation_server()

        assert server.reference_count == 1
        assert server.started_by_application is True

        # Release
        await manager.release_generation_server()
        assert server.reference_count == 0

    @pytest.mark.asyncio
    async def test_multiple_references(self, manager):
        """Test: multiple ensure calls increment count."""
        server = manager.generation_server

        with patch.object(manager, "_check_health", new_callable=AsyncMock, return_value=True):
            await manager.ensure_generation_server()
            await manager.ensure_generation_server()
            await manager.ensure_generation_server()

        assert server.reference_count == 3

        await manager.release_generation_server()
        assert server.reference_count == 2

        await manager.release_generation_server()
        assert server.reference_count == 1

        await manager.release_generation_server()
        assert server.reference_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, manager):
        """Test: concurrent ensure calls are serialized by lock."""
        server = manager.generation_server
        call_count = 0

        async def slow_check(server):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return True

        with patch.object(manager, "_check_health", side_effect=slow_check):
            # Launch 3 concurrent calls
            await asyncio.gather(
                manager.ensure_generation_server(),
                manager.ensure_generation_server(),
                manager.ensure_generation_server(),
            )

        # All 3 should have incremented the count
        assert server.reference_count == 3
