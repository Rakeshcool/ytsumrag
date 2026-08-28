"""Test script for model server management.

Usage:
    uv run python scripts/test_model_servers.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def main() -> None:
    from app.model_servers.manager import model_server_manager

    print("=== Model Server Management Test ===\n")

    # Test 1: Check initial health
    print("1. Checking initial generation server health...")
    gen_healthy = await model_server_manager.generation_is_healthy()
    print(f"   Generation server healthy: {gen_healthy}\n")

    print("2. Checking initial embedding server health...")
    emb_healthy = await model_server_manager.embedding_is_healthy()
    print(f"   Embedding server healthy: {emb_healthy}\n")

    # Test 2: Try to ensure embedding server
    print("3. Ensuring embedding server...")
    try:
        await model_server_manager.ensure_embedding_server()
        print("   Embedding server is running!\n")
    except FileNotFoundError as exc:
        print(f"   Expected error (missing executable/model): {exc}\n")
    except RuntimeError as exc:
        print(f"   Startup failed: {exc}\n")

    # Test 3: Try to ensure generation server
    print("4. Ensuring generation server...")
    try:
        await model_server_manager.ensure_generation_server()
        print("   Generation server is running!\n")
    except FileNotFoundError as exc:
        print(f"   Expected error (missing executable/model): {exc}\n")
    except RuntimeError as exc:
        print(f"   Startup failed: {exc}\n")

    # Test 4: Stop all
    print("5. Stopping all servers...")
    await model_server_manager.stop_all()
    print("   Done.\n")

    print("=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
