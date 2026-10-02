"""Rich formatting helpers for the CLI."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console(
    file=sys.stdout,
    force_terminal=True,
    color_system="auto",
)


def print_header(title: str, subtitle: str = "") -> None:
    """Print a styled header panel."""
    text = f"[bold white]{title}[/]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/]"
    console.print(Panel(text, border_style="blue", padding=(0, 2)))


def print_success(message: str) -> None:
    """Print a success message with checkmark."""
    console.print(f"  [bold green]OK[/] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"  [bold red]ERR[/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"  [bold cyan]>>[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"  [bold yellow]!![/] {message}")


def print_summary(markdown_text: str, title: str = "Summary") -> None:
    """Render a markdown summary in a panel."""
    md = Markdown(markdown_text)
    console.print(Panel(md, title=f"[bold]{title}[/]", border_style="green", padding=(1, 2)))


def print_transcript(segments: list[dict], title: str = "Transcript") -> None:
    """Print transcript segments as a styled table."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Time", style="yellow", no_wrap=True, width=16)
    table.add_column("Text", ratio=1)

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        m1, s1 = divmod(int(start), 60)
        m2, s2 = divmod(int(end), 60)
        table.add_row(
            f"{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}",
            seg["text"],
        )

    console.print(table)


def print_sections(sections: list[dict], title: str = "Sections") -> None:
    """Print section summaries as styled cards."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Topic", style="bold")
    table.add_column("Time", style="yellow", no_wrap=True, width=14)
    table.add_column("Summary", ratio=2)

    for s in sections:
        sec_id = s.get("section_id", "?")
        sec_title = s.get("title", "Unknown")
        start = s.get("start", 0)
        end = s.get("end", 0)
        summary = s.get("summary", "")
        m1, s1 = divmod(int(start), 60)
        m2, s2 = divmod(int(end), 60)

        table.add_row(
            str(sec_id),
            sec_title,
            f"{m1:02d}:{s1:02d}–{m2:02d}:{s2:02d}",
            summary[:120] + ("..." if len(summary) > 120 else ""),
        )

    console.print(table)


def print_video_table(videos: dict) -> None:
    """Print a table of ingested videos."""
    if not videos:
        print_info("No videos ingested yet. Use: [bold]cli.py ingest <url>[/]")
        return

    table = Table(
        title=f"Ingested Videos ({len(videos)})",
        show_header=True,
        header_style="bold blue",
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Video ID", style="yellow", no_wrap=True)
    table.add_column("Title", ratio=2)
    table.add_column("Status", justify="center")
    table.add_column("Summary", justify="center")

    for vid, data in videos.items():
        title = data.get("metadata", {}).get("title", "Unknown")
        status = data.get("status", "unknown")
        has_summary = "[green]Y[/]" if data.get("summary") else "[dim]-[/]"
        status_style = "green" if status == "completed" else "yellow"

        table.add_row(
            vid,
            title[:60] + ("..." if len(title) > 60 else ""),
            f"[{status_style}]{status}[/]",
            has_summary,
        )

    console.print(table)


def print_health(services: list[dict]) -> None:
    """Print a health status table."""
    table = Table(
        title="Service Health",
        show_header=True,
        header_style="bold blue",
        border_style="dim",
    )
    table.add_column("Service", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for svc in services:
        name = svc["name"]
        healthy = svc["healthy"]
        detail = svc.get("detail", "")
        status = "[green]UP[/]" if healthy else "[red]DOWN[/]"
        table.add_row(name, status, detail)

    console.print(table)


def create_progress() -> Progress:
    """Create a rich progress bar for pipeline operations."""
    return Progress(
        TextColumn("[bold blue]{task.description}[/]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def format_answer(answer: str, question: str) -> Panel:
    """Format a RAG answer in a styled panel."""
    md = Markdown(answer)
    return Panel(
        md,
        title=f"[bold]Q: {question}[/]",
        border_style="cyan",
        padding=(1, 2),
    )
