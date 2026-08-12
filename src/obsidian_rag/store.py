"""A thin persistence layer around ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from obsidian_rag.models import NoteChunk, SearchResult


class ChromaStore:
    """Persist note chunks and vectors in a local Chroma collection."""

    def __init__(
        self,
        database_path: Path,
        collection_name: str = "obsidian_notes",
    ) -> None:
        database_path = database_path.expanduser().resolve()
        database_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(database_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def indexed_file_hashes(self) -> dict[str, str]:
        """Return the last indexed hash for every source file."""

        records = self.collection.get(include=["metadatas"])
        hashes: dict[str, str] = {}
        for metadata in records.get("metadatas") or []:
            if not metadata:
                continue
            source = metadata.get("source")
            file_hash = metadata.get("file_hash")
            if isinstance(source, str) and isinstance(file_hash, str):
                hashes[source] = file_hash
        return hashes

    def replace_file(
        self,
        source: str,
        chunks: list[NoteChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Atomically from the indexer's perspective, replace one note's chunks.

        Embeddings are calculated before this method is called. Therefore an
        Ollama failure cannot delete the previously indexed copy of the note.
        """

        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")

        self.collection.delete(where={"source": source})
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.document for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=embeddings,
        )

    def delete_file(self, source: str) -> None:
        """Remove every stored chunk belonging to a deleted note."""

        self.collection.delete(where={"source": source})

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the nearest stored chunks for a query embedding."""

        available = self.collection.count()
        if available == 0:
            return []

        response: dict[str, Any] = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            results.append(
                SearchResult(
                    document=document or "",
                    source=str((metadata or {}).get("source", "Unknown source")),
                    heading=str((metadata or {}).get("heading_path", "Unknown section")),
                    distance=float(distance),
                    chunk_id=chunk_id,
                )
            )
        return results

    def count(self) -> int:
        return self.collection.count()
