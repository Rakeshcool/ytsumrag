"""Test script for embedding generation.

Usage:
    uv run python scripts/test_embedding.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


async def main() -> None:
    from app.embeddings.qwen import embedding_client
    from app.model_servers.manager import model_server_manager

    print("=== Embedding Test ===\n")

    try:
        await model_server_manager.ensure_embedding_server()
        print("[OK] Embedding server is running\n")

        print("Embedding test texts...")
        texts = [
            "This is a test sentence about machine learning.",
            "Python is a popular programming language.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        vectors = await embedding_client.embed_texts(texts)

        for i, (text, vec) in enumerate(zip(texts, vectors)):
            print(f"  Text {i}: dim={len(vec)}, first_5={vec[:5]}")

        print(f"\n[OK] Embedded {len(vectors)} texts, dimension={len(vectors[0])}")

    except Exception as exc:
        print(f"[FAIL] Failed: {exc}")
        print("\nMake sure llama-server is running on port 8081 with --embedding")
    finally:
        await model_server_manager.release_embedding_server()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
