"""Coordinate scanning, parsing, chunking, embedding and persistence."""

from __future__ import annotations

from pathlib import Path

from obsidian_rag.chunker import chunk_sections
from obsidian_rag.embeddings import Embedder
from obsidian_rag.models import IndexSummary, SearchResult
from obsidian_rag.parser import parse_note
from obsidian_rag.scanner import scan_vault
from obsidian_rag.store import ChromaStore


class VaultIndexer:
    """Incrementally synchronize an Obsidian vault into Chroma."""

    def __init__(
        self,
        vault_path: Path,
        store: ChromaStore,
        embedder: Embedder,
        max_chunk_characters: int = 800,
    ) -> None:
        self.vault_path = vault_path
        self.store = store
        self.embedder = embedder
        self.max_chunk_characters = max_chunk_characters

    def run(self) -> IndexSummary:
        vault_files = scan_vault(self.vault_path)
        indexed_hashes = self.store.indexed_file_hashes()
        current_sources = {vault_file.relative_path for vault_file in vault_files}

        indexed_files = 0
        skipped_files = 0
        empty_files = 0

        for vault_file in vault_files:
            if indexed_hashes.get(vault_file.relative_path) == vault_file.content_hash:
                skipped_files += 1
                continue

            sections = parse_note(vault_file)
            chunks = chunk_sections(
                sections,
                file_hash=vault_file.content_hash,
                max_characters=self.max_chunk_characters,
            )
            if not chunks:
                # Empty notes have no vector record in which to store a hash.
                # If the note used to contain text, remove its previous chunks;
                # otherwise report it separately rather than as reindexed.
                if vault_file.relative_path in indexed_hashes:
                    self.store.delete_file(vault_file.relative_path)
                    indexed_files += 1
                else:
                    empty_files += 1
                continue

            embeddings = self.embedder.embed_documents(
                [chunk.document for chunk in chunks]
            )
            self.store.replace_file(vault_file.relative_path, chunks, embeddings)
            indexed_files += 1

        deleted_sources = set(indexed_hashes) - current_sources
        for source in sorted(deleted_sources):
            self.store.delete_file(source)

        return IndexSummary(
            scanned_files=len(vault_files),
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            empty_files=empty_files,
            deleted_files=len(deleted_sources),
            stored_chunks=self.store.count(),
        )


class NoteSearcher:
    """Embed a natural-language question and retrieve matching chunks."""

    def __init__(self, store: ChromaStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, top_k=top_k)
