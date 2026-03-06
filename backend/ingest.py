"""
CLI script to ingest documents into pgvector.
Run before starting the API server.

Usage:
    python ingest.py --docs-dir ./docs/
    python ingest.py --docs-dir ./docs/ --force   # re-ingest all
    python ingest.py --stats                       # show current stats
"""

import sys as _sys
from pathlib import Path as _Path
_BACKEND_DIR = _Path(__file__).resolve().parent
while not (_BACKEND_DIR / "main.py").exists() and _BACKEND_DIR != _BACKEND_DIR.parent:
    _BACKEND_DIR = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in _sys.path:
    _sys.path.insert(0, str(_BACKEND_DIR))


import argparse
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv("env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector for RAG")
    parser.add_argument("--docs-dir", type=str, help="Directory containing .md and .pdf files")
    parser.add_argument("--force", action="store_true", help="Re-ingest even unchanged files")
    parser.add_argument("--stats", action="store_true", help="Show ingestion stats and exit")
    args = parser.parse_args()

    from rag.rag_tool import init_db, ingest_directory, get_doc_stats

    # Init DB
    console.print("[bold cyan]Initializing database...[/bold cyan]")
    init_db()
    console.print("[green]✓ Database ready[/green]")

    if args.stats:
        console.print("\n[bold]Document Store Stats:[/bold]")
        console.print(get_doc_stats())
        return

    if not args.docs_dir:
        console.print("[red]Error: --docs-dir is required[/red]")
        return

    console.print(f"\n[bold cyan]Ingesting documents from: {args.docs_dir}[/bold cyan]")
    results = ingest_directory(args.docs_dir, force=args.force)

    # Display results table
    table = Table(title="Ingestion Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Chunks", justify="right")

    total_chunks = 0
    for r in results:
        status_color = {"ingested": "green", "skipped": "yellow", "error": "red", "empty": "dim"}.get(r["status"], "white")
        table.add_row(
            r["file"],
            f"[{status_color}]{r['status']}[/{status_color}]",
            r.get("doc_type", "-"),
            str(r.get("chunks", 0)),
        )
        total_chunks += r.get("chunks", 0)

    console.print(table)
    console.print(f"\n[bold green]Total: {len(results)} files, {total_chunks} chunks ingested[/bold green]")


if __name__ == "__main__":
    main()
