"""Test script for LLM generation.

Usage:
    uv run python scripts/test_llm.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


async def main() -> None:
    from app.llm.llama_client import llama_client
    from app.model_servers.manager import model_server_manager

    print("=== LLM Generation Test ===\n")

    try:
        await model_server_manager.ensure_generation_server()
        print("[OK] Generation server is running\n")

        print("Sending test prompt...")
        response = await llama_client.generate(
            "What is 2 + 2? Answer in one sentence.",
            system_prompt="You are a helpful assistant.",
            temperature=0.1,
            max_tokens=100,
        )
        print(f"[OK] Response: {response}\n")

    except Exception as exc:
        print(f"[FAIL] Failed: {exc}")
        print("\nMake sure llama-server is running on port 3000")
    finally:
        await model_server_manager.stop_all()

    print("=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
