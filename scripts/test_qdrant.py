"""Test script for Qdrant connection.

Usage:
    uv run python scripts/test_qdrant.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    from app.vectorstore.qdrant import vector_store

    print("=== Qdrant Connection Test ===\n")

    try:
        vector_store.ensure_collection()
        print("[OK] Connected to Qdrant")
        print(f"[OK] Collection '{vector_store._collection}' ready")
    except Exception as exc:
        print(f"[FAIL] Failed to connect: {exc}")
        print("\nMake sure Qdrant is running:")
        print("  podman machine start")
        print("  podman compose up -d")

    vector_store.close()
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
