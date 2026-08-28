"""End-to-end test of the full ingestion pipeline.

Usage:
    uv run python scripts/test_pipeline.py <YOUTUBE_URL>
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

YOUTUBE_URL = "https://www.youtube.com/watch?v=QRZ_tQ1wqV8"


async def main() -> None:
    from app.model_servers.manager import model_server_manager
    from app.services.pipeline import ingest_video, answer_question
    from app.vectorstore.qdrant import vector_store

    url = sys.argv[1] if len(sys.argv) > 1 else YOUTUBE_URL

    print("=" * 60)
    print("  FULL PIPELINE END-TO-END TEST")
    print("=" * 60)
    print(f"\n  URL: {url}\n")

    # Ensure Qdrant collection
    vector_store.ensure_collection()
    print("[OK] Qdrant collection ready\n")

    # ── Ingestion ───────────────────────────────────────────────
    print("-" * 60)
    print("  PHASE 1: INGESTION")
    print("-" * 60)

    try:
        record = await ingest_video(url)
        print(f"\n[OK] Ingestion complete!")
        print(f"  Video ID : {record.metadata.video_id}")
        print(f"  Title    : {record.metadata.title}")
        print(f"  Status   : {record.status.value}")
        print(f"  Segments : {len(record.transcript.segments) if record.transcript else 0}")
        print(f"  Summary  : {len(record.summary or '')} chars")
        print(f"  Sections : {len(record.sections or [])}")

        if record.summary:
            print(f"\n--- Summary (first 500 chars) ---")
            print(record.summary[:500])
            print("...\n")

    except Exception as exc:
        print(f"\n[FAIL] Ingestion failed: {exc}")
        import traceback
        traceback.print_exc()
        await model_server_manager.stop_all()
        vector_store.close()
        return

    # ── RAG Q&A ────────────────────────────────────────────────
    print("-" * 60)
    print("  PHASE 2: RAG Q&A")
    print("-" * 60)

    video_id = record.metadata.video_id
    questions = [
        "What is this video about?",
        "What are the main points discussed?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        try:
            answer = await answer_question(video_id, q)
            print(f"  A: {answer[:300]}")
            if len(answer) > 300:
                print("  ...")
        except Exception as exc:
            print(f"  [FAIL] {exc}")

    # ── Cleanup ─────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  CLEANUP")
    print("-" * 60)

    await model_server_manager.stop_all()
    vector_store.close()
    print("[OK] Servers stopped, connections closed\n")

    print("=" * 60)
    print("  PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
