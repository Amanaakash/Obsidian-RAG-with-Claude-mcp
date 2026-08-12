"""Command-line entry point for indexing and searching notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from obsidian_rag.embeddings import DEFAULT_EMBEDDING_MODEL, OllamaEmbedder
from obsidian_rag.indexer import NoteSearcher, VaultIndexer
from obsidian_rag.store import ChromaStore


DEFAULT_DATABASE_PATH = Path(".rag_data/chroma")
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="Directory where Chroma stores its local database",
    )
    parser.add_argument(
        "--collection",
        default="obsidian_notes",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="URL of the local Ollama server",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obsidian-rag",
        description="Index and semantically search a local Obsidian vault.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="Synchronize changed Markdown notes into Chroma"
    )
    _add_common_arguments(index_parser)
    index_parser.add_argument("--vault", type=Path, required=True, help="Obsidian vault path")
    index_parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Maximum approximate characters per chunk",
    )

    search_parser = subparsers.add_parser(
        "search", help="Search indexed notes using natural language"
    )
    _add_common_arguments(search_parser)
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    return parser


def _build_dependencies(args: argparse.Namespace) -> tuple[ChromaStore, OllamaEmbedder]:
    store = ChromaStore(database_path=args.db, collection_name=args.collection)
    embedder = OllamaEmbedder(model=args.model, host=args.ollama_host)
    return store, embedder


def _run_index(args: argparse.Namespace) -> int:
    store, embedder = _build_dependencies(args)
    summary = VaultIndexer(
        vault_path=args.vault,
        store=store,
        embedder=embedder,
        max_chunk_characters=args.max_chars,
    ).run()

    print("Indexing complete")
    print(f"  Markdown files found : {summary.scanned_files}")
    print(f"  New or changed files : {summary.indexed_files}")
    print(f"  Unchanged files      : {summary.skipped_files}")
    print(f"  Empty notes ignored  : {summary.empty_files}")
    print(f"  Deleted files removed: {summary.deleted_files}")
    print(f"  Chunks in Chroma     : {summary.stored_chunks}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    store, embedder = _build_dependencies(args)
    results = NoteSearcher(store=store, embedder=embedder).search(
        query=args.query,
        top_k=args.top_k,
    )

    if not results:
        print("No indexed notes found. Run the index command first.")
        return 0

    print(f'Search results for: "{args.query}"')
    for rank, result in enumerate(results, start=1):
        print()
        print(f"[{rank}] {result.source}  |  {result.heading}")
        print(f"    distance: {result.distance:.4f}")
        print("-" * 72)
        print(result.document)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        exit_code = _run_index(args) if args.command == "index" else _run_search(args)
    except Exception as error:  # Friendly boundary for a learning-oriented CLI.
        print(f"Error: {error}", file=sys.stderr)
        print(
            "Check that the vault path is correct and that Ollama is running with "
            f"the '{args.model}' model installed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    raise SystemExit(exit_code)
