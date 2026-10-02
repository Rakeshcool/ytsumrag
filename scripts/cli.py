"""Rich CLI for the YouTube Local RAG Summarizer.

Usage:
    uv run python scripts/cli.py ingest <youtube_url>
    uv run python scripts/cli.py ask <video_id> <question>
    uv run python scripts/cli.py summary <video_id>
    uv run python scripts/cli.py search <video_id> <query>
    uv run python scripts/cli.py transcript <video_id>
    uv run python scripts/cli.py list
    uv run python scripts/cli.py health
    uv run python scripts/cli.py server start|stop|status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli.helpers import (
    console,
    create_progress,
    format_answer,
    print_error,
    print_header,
    print_health,
    print_info,
    print_success,
    print_transcript,
    print_video_table,
)


def _get_video_store() -> dict:
    """Load all cached video records."""
    cache_dir = Path("data/cache")
    videos = {}
    if cache_dir.exists():
        for f in cache_dir.glob("*_record.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                vid = data.get("metadata", {}).get("video_id")
                if vid:
                    videos[vid] = data
            except Exception:
                continue
    return videos


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a YouTube video with progress tracking."""
    from app.services.pipeline import ingest_video
    from app.vectorstore.qdrant import vector_store

    vector_store.ensure_collection()

    print_header("Ingesting Video", args.url)

    async def _run():
        state = {"progress": None, "task": None}

        async def _cb(phase: str, message: str, step: int, total: int) -> None:
            state["progress"].update(
                state["task"], completed=step,
                description=f"[bold]{phase.upper()}[/] {message}",
            )

        p = create_progress()
        with p:
            t = p.add_task("Starting...", total=5)
            state["progress"] = p
            state["task"] = t
            try:
                record = await ingest_video(
                    args.url,
                    progress_callback=_cb,
                    force_whisper=args.force_whisper,
                    no_cleanup=args.no_cleanup,
                )
                p.update(t, completed=5, description="[bold green]Done![/]")

                console.print()
                vid = record.metadata.video_id
                segs = len(record.transcript.segments)
                print_success(f"Video ID : [bold]{vid}[/]")
                print_success(f"Title    : {record.metadata.title}")
                print_success(f"Status   : {record.status.value}")
                print_success(f"Segments : {segs}")
                console.print()
                print_info(
                    f"Ask: [bold]uv run python"
                    f" scripts/cli.py ask {vid} \"question\"[/]"
                )

            except Exception as exc:
                p.update(t, description="[bold red]Failed![/]")
                console.print()
                print_error(f"{exc}")
                if args.verbose:
                    console.print_exception()

    asyncio.run(_run())


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question about a video."""
    from app.services.pipeline import answer_question

    print_header("RAG Q&A", f"Video: [yellow]{args.video_id}[/]")

    print_info("Thinking...")

    async def _run():
        nonlocal answer
        answer = await answer_question(args.video_id, args.question)

    answer = ""
    asyncio.run(_run())

    console.print(format_answer(answer, args.question))


def cmd_summary(args: argparse.Namespace) -> None:
    """Generate a structured summary of a video."""
    from app.cli.helpers import print_summary
    from app.services.pipeline import generate_summary

    print_header("Generating Summary", f"Video: [yellow]{args.video_id}[/]")

    print_info("Generating summary...")

    async def _run():
        nonlocal summary
        summary = await generate_summary(args.video_id)

    summary = ""
    asyncio.run(_run())

    # Get title from cache
    videos = _get_video_store()
    data = videos.get(args.video_id)
    title = data.get("metadata", {}).get("title", "Video") if data else "Video"

    print_summary(summary, title=title)


def cmd_search(args: argparse.Namespace) -> None:
    """Search for relevant chunks using hybrid retrieval."""
    from app.retrieval.retriever import retrieve_and_rerank

    print_header("Hybrid Search", f"Video: [yellow]{args.video_id}[/]")

    async def _run():
        results = await retrieve_and_rerank(
            args.query, args.video_id, top_k=10, final_k=args.top_k,
        )
        return results

    results = asyncio.run(_run())

    if not results:
        print_error("No results found.")
        return

    console.print(f"\n[bold]Top {len(results)} results:[/]\n")
    for i, (chunk, score, meta) in enumerate(results):
        m1, s1 = divmod(int(chunk.start), 60)
        m2, s2 = divmod(int(chunk.end), 60)
        h1, m1 = divmod(m1, 60)
        h2, m2 = divmod(m2, 60)
        speakers = meta.get("speakers", "") or ""
        summary = meta.get("summary", "")
        keywords = meta.get("keywords", [])

        console.print(
            f"  [bold cyan]{i + 1}.[/] [yellow]{h1:02d}:{m1:02d}:{s1:02d}-"
            f"{h2:02d}:{m2:02d}:{s2:02d}[/]"
            f" [dim](score: {score:.3f})[/]"
        )
        if speakers:
            console.print(f"     [bold]Speakers:[/] {speakers}")
        if summary:
            console.print(f"     [dim]Summary: {summary}[/]")
        if keywords:
            console.print(f"     [dim]Keywords: {', '.join(keywords)}[/]")
        console.print(f"     {chunk.text[:150]}{'...' if len(chunk.text) > 150 else ''}")
        console.print()


def cmd_transcript(args: argparse.Namespace) -> None:
    """Show the transcript for a video."""
    videos = _get_video_store()
    data = videos.get(args.video_id)

    if not data:
        print_error(f"Video [bold]{args.video_id}[/] not found. Ingest it first.")
        return

    transcript = data.get("transcript")
    if not transcript:
        print_error(f"Transcript not available yet for [bold]{args.video_id}[/].")
        return

    title = data.get("metadata", {}).get("title", "Unknown")
    segments = transcript.get("segments", [])

    print_transcript(segments, title=f"{title} ({len(segments)} segments)")


def cmd_list(args: argparse.Namespace) -> None:
    """List all ingested videos."""
    videos = _get_video_store()
    print_video_table(videos)


def cmd_health(args: argparse.Namespace) -> None:
    """Check health of all services."""
    import httpx

    services = []

    # Check Qdrant
    try:
        resp = httpx.get("http://127.0.0.1:6333/collections", timeout=5.0)
        services.append({"name": "Qdrant (6333)", "healthy": resp.status_code == 200})
    except Exception:
        services.append({"name": "Qdrant (6333)", "healthy": False, "detail": "not reachable"})

    # Check generation server
    try:
        resp = httpx.get("http://127.0.0.1:3000/health", timeout=5.0)
        services.append({"name": "Generation (3000)", "healthy": resp.status_code == 200})
    except Exception:
        services.append({"name": "Generation (3000)", "healthy": False, "detail": "not running"})

    # Check embedding server
    try:
        resp = httpx.get("http://127.0.0.1:8081/health", timeout=5.0)
        services.append({"name": "Embedding (8081)", "healthy": resp.status_code == 200})
    except Exception:
        services.append({"name": "Embedding (8081)", "healthy": False, "detail": "not running"})

    # Cached videos
    videos = _get_video_store()
    services.append({
        "name": "Cached Videos",
        "healthy": True,
        "detail": f"{len(videos)} videos",
    })

    print_health(services)


def cmd_server(args: argparse.Namespace) -> None:
    """Start/stop/status model servers."""
    from app.model_servers.manager import model_server_manager

    async def _run():
        if args.action == "start":
            print_info("Starting servers...")
            try:
                await model_server_manager.ensure_embedding_server()
                print_success("Embedding server started")
            except Exception as e:
                print_error(f"Embedding server failed: {e}")
            try:
                await model_server_manager.ensure_generation_server()
                print_success("Generation server started")
            except Exception as e:
                print_error(f"Generation server failed: {e}")

        elif args.action == "stop":
            print_info("Stopping servers...")
            await model_server_manager.stop_all()
            print_success("All servers stopped")

        elif args.action == "status":
            gen = model_server_manager.generation_server
            emb = model_server_manager.embedding_server

            services = [
                {
                    "name": f"Generation ({gen.server_type.value})",
                    "healthy": gen.state.value == "running",
                    "detail": f"owned: {gen.started_by_application}",
                },
                {
                    "name": f"Embedding ({emb.server_type.value})",
                    "healthy": emb.state.value == "running",
                    "detail": f"owned: {emb.started_by_application}",
                },
            ]
            print_health(services)

    asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[bold blue]YouTube Local RAG Summarizer CLI[/]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingest "https://www.youtube.com/watch?v=VIDEO_ID"
  %(prog)s ask jNQXAC9IVRw "What is this video about?"
  %(prog)s summary jNQXAC9IVRw
  %(prog)s search jNQXAC9IVRw "machine learning"
  %(prog)s transcript jNQXAC9IVRw
  %(prog)s list
  %(prog)s health
  %(prog)s server start|stop|status
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a YouTube video")
    p_ingest.add_argument("url", help="YouTube video URL")
    p_ingest.add_argument(
        "--force-whisper", action="store_true",
        help="Force Whisper transcription (skip captions)",
    )
    p_ingest.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep audio files after transcription",
    )
    p_ingest.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full traceback on error",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question about a video")
    p_ask.add_argument("video_id", help="Video ID")
    p_ask.add_argument("question", help="Your question")
    p_ask.set_defaults(func=cmd_ask)

    # summary
    p_summary = subparsers.add_parser("summary", help="Generate a structured summary")
    p_summary.add_argument("video_id", help="Video ID")
    p_summary.set_defaults(func=cmd_summary)

    # search
    p_search = subparsers.add_parser("search", help="Search chunks using hybrid retrieval")
    p_search.add_argument("video_id", help="Video ID")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results (default: 5)",
    )
    p_search.set_defaults(func=cmd_search)

    # transcript
    p_transcript = subparsers.add_parser("transcript", help="Show video transcript")
    p_transcript.add_argument("video_id", help="Video ID")
    p_transcript.set_defaults(func=cmd_transcript)

    # list
    p_list = subparsers.add_parser("list", help="List all ingested videos")
    p_list.set_defaults(func=cmd_list)

    # health
    p_health = subparsers.add_parser("health", help="Check service health")
    p_health.set_defaults(func=cmd_health)

    # server
    p_server = subparsers.add_parser("server", help="Manage model servers")
    p_server.add_argument("action", choices=["start", "stop", "status"], help="Action")
    p_server.set_defaults(func=cmd_server)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
