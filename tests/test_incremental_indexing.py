from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from obsidian_rag.indexer import NoteSearcher, VaultIndexer
from obsidian_rag.store import ChromaStore


class LocalTestEmbedder:
    """A tiny deterministic embedder used so tests do not require Ollama."""

    dimensions = 256

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._vector(document) for document in documents]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


def _write_initial_vault(vault: Path) -> None:
    vault.mkdir()
    (vault / "Search.md").write_text(
        "# Hybrid Search\n\nHybrid search combines semantic vector retrieval with exact keyword retrieval.",
        encoding="utf-8",
    )
    (vault / "Cooking.md").write_text(
        "# Bread\n\nBread dough needs flour, water, salt, yeast, and time to rise.",
        encoding="utf-8",
    )
    (vault / "Empty.md").write_text("", encoding="utf-8")


def test_incremental_indexing_search_updates_deletions_and_no_duplicates(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    database = tmp_path / "chroma"
    _write_initial_vault(vault)

    store = ChromaStore(database)
    embedder = LocalTestEmbedder()
    indexer = VaultIndexer(vault, store, embedder)

    first = indexer.run()
    first_chunk_count = store.count()
    assert first.scanned_files == 3
    assert first.indexed_files == 2
    assert first.skipped_files == 0
    assert first.empty_files == 1

    second = indexer.run()
    assert second.indexed_files == 0
    assert second.skipped_files == 2
    assert second.empty_files == 1
    assert store.count() == first_chunk_count

    results = NoteSearcher(store, embedder).search(
        "semantic vector and keyword retrieval", top_k=1
    )
    assert results[0].source == "Search.md"
    assert results[0].heading == "Hybrid Search"

    (vault / "Search.md").write_text(
        "# Hybrid Search\n\nHybrid retrieval joins dense semantic search and BM25 keyword search.",
        encoding="utf-8",
    )
    third = indexer.run()
    assert third.indexed_files == 1
    assert third.skipped_files == 1
    assert third.empty_files == 1
    assert store.count() == first_chunk_count

    (vault / "Cooking.md").unlink()
    fourth = indexer.run()
    assert fourth.deleted_files == 1
    assert "Cooking.md" not in store.indexed_file_hashes()
    assert store.count() < first_chunk_count
