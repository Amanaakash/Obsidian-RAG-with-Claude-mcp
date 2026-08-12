"""Environment-only configuration for the local Obsidian MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from obsidian_rag.embeddings import DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True)
class MCPSettings:
    """Resolved settings shared by Claude Desktop and Claude Code."""

    vault_path: Path
    database_path: Path
    model: str = DEFAULT_EMBEDDING_MODEL
    collection: str = "obsidian_notes"
    ollama_host: str = "http://localhost:11434"
    max_chunk_characters: int = 800

    @property
    def lock_path(self) -> Path:
        return self.database_path.with_name(f"{self.database_path.name}.refresh.lock")

    @property
    def status_path(self) -> Path:
        return self.database_path.with_name(f"{self.database_path.name}.status.json")

    @classmethod
    def from_environment(cls) -> "MCPSettings":
        vault = os.environ.get("OBSIDIAN_RAG_VAULT", "").strip()
        database = os.environ.get("OBSIDIAN_RAG_DB", "").strip()
        if not vault:
            raise ValueError("OBSIDIAN_RAG_VAULT is required")
        if not database:
            raise ValueError("OBSIDIAN_RAG_DB is required")

        vault_path = Path(vault).expanduser().resolve()
        if not vault_path.is_dir():
            raise ValueError("OBSIDIAN_RAG_VAULT must point to an existing directory")

        database_path = Path(database).expanduser().resolve()
        model = os.environ.get("OBSIDIAN_RAG_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        collection = os.environ.get("OBSIDIAN_RAG_COLLECTION", "obsidian_notes").strip()
        ollama_host = os.environ.get(
            "OBSIDIAN_RAG_OLLAMA_HOST", "http://localhost:11434"
        ).strip()

        if not model:
            raise ValueError("OBSIDIAN_RAG_MODEL cannot be empty")
        if not collection:
            raise ValueError("OBSIDIAN_RAG_COLLECTION cannot be empty")

        return cls(
            vault_path=vault_path,
            database_path=database_path,
            model=model,
            collection=collection,
            ollama_host=ollama_host,
        )
