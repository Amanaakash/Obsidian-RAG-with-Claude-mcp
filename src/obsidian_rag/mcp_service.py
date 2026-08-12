"""Read-only application service exposed through Model Context Protocol."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock, Timeout

from obsidian_rag.embeddings import Embedder, OllamaEmbedder
from obsidian_rag.indexer import NoteSearcher, VaultIndexer
from obsidian_rag.mcp_settings import MCPSettings
from obsidian_rag.models import VaultFile
from obsidian_rag.parser import parse_note
from obsidian_rag.scanner import file_sha256
from obsidian_rag.store import ChromaStore


EmbedderFactory = Callable[[], Embedder]
StoreFactory = Callable[[], ChromaStore]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ObsidianMCPService:
    """Own safe vault reads, semantic search, and background index refresh."""

    def __init__(
        self,
        settings: MCPSettings,
        embedder_factory: EmbedderFactory | None = None,
        store_factory: StoreFactory | None = None,
    ) -> None:
        self.settings = settings
        self._embedder_factory = embedder_factory or (
            lambda: OllamaEmbedder(
                model=settings.model,
                host=settings.ollama_host,
            )
        )
        self._store_factory = store_factory or (
            lambda: ChromaStore(
                database_path=settings.database_path,
                collection_name=settings.collection,
            )
        )
        self._state_lock = threading.Lock()
        self._refresh_thread: threading.Thread | None = None
        self._state = self._load_status()

    def _default_status(self) -> dict[str, Any]:
        return {
            "refresh_state": "not_started",
            "last_successful_refresh": None,
            "last_refresh_attempt": None,
            "last_summary": None,
            "error": None,
        }

    def _load_status(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.settings.status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return {**self._default_status(), **loaded}
        except (OSError, json.JSONDecodeError):
            pass
        return self._default_status()

    def _set_state(self, *, persist: bool = False, **changes: Any) -> None:
        with self._state_lock:
            self._state = {**self._state, **changes}
            snapshot = dict(self._state)
        if persist:
            self._persist_status(snapshot)

    def _persist_status(self, status: dict[str, Any]) -> None:
        path = self.settings.status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(status, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _sanitized_error(error: Exception) -> dict[str, str]:
        message = str(error).casefold()
        if "ollama" in message or "connection" in message:
            return {
                "code": "ollama_unavailable",
                "message": "The local Ollama embedding service is unavailable.",
            }
        if "context length" in message or "embedding" in message:
            return {
                "code": "embedding_failed",
                "message": "A note could not be embedded by the configured model.",
            }
        return {
            "code": "refresh_failed",
            "message": "The background vault refresh did not complete.",
        }

    def _new_lock(self) -> FileLock:
        self.settings.lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.settings.lock_path))

    def _lock_is_busy(self) -> bool:
        lock = self._new_lock()
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return True
        else:
            lock.release()
            return False

    def start_background_refresh(self) -> bool:
        """Start one daemon refresh worker for this MCP process."""

        with self._state_lock:
            if self._refresh_thread and self._refresh_thread.is_alive():
                return False
            self._refresh_thread = threading.Thread(
                target=self._refresh_worker,
                name="obsidian-rag-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
        return True

    def wait_for_refresh(self, timeout: float | None = None) -> bool:
        """Test/support helper; normal MCP startup never blocks on refresh."""

        thread = self._refresh_thread
        if not thread:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _refresh_worker(self) -> None:
        attempted_at = _utc_now()
        lock = self._new_lock()
        try:
            lock.acquire(timeout=0)
        except Timeout:
            self._set_state(
                refresh_state="external_refresh_in_progress",
                last_refresh_attempt=attempted_at,
                error=None,
            )
            print(
                "Obsidian RAG refresh skipped because another client is refreshing.",
                file=sys.stderr,
            )
            return

        try:
            self._set_state(
                refresh_state="refreshing",
                last_refresh_attempt=attempted_at,
                error=None,
            )
            summary = VaultIndexer(
                vault_path=self.settings.vault_path,
                store=self._store_factory(),
                embedder=self._embedder_factory(),
                max_chunk_characters=self.settings.max_chunk_characters,
            ).run()
            completed_at = _utc_now()
            self._set_state(
                persist=True,
                refresh_state="ready",
                last_successful_refresh=completed_at,
                last_refresh_attempt=attempted_at,
                last_summary=asdict(summary),
                error=None,
            )
            print(
                f"Obsidian RAG refresh complete: {summary.stored_chunks} chunks.",
                file=sys.stderr,
            )
        except Exception as error:  # The MCP server must survive refresh failures.
            print(f"Obsidian RAG refresh error: {error}", file=sys.stderr)
            self._set_state(
                persist=True,
                refresh_state="failed",
                last_refresh_attempt=attempted_at,
                error=self._sanitized_error(error),
            )
        finally:
            lock.release()

    def _index_exists(self) -> bool:
        return (
            self.settings.database_path.is_dir()
            and (self.settings.database_path / "chroma.sqlite3").is_file()
        )

    def search_notes(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        lock = self._new_lock()
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return {
                "status": "index_refreshing",
                "query": query,
                "message": "The local index is refreshing. Try this search again shortly.",
                "results": [],
            }

        try:
            if not self._index_exists():
                return {
                    "status": "no_index",
                    "query": query,
                    "message": (
                        "No usable local index exists yet. Keep Ollama running and "
                        "check vault_status for background refresh progress."
                    ),
                    "results": [],
                }

            store = self._store_factory()
            if store.count() == 0:
                return {
                    "status": "no_index",
                    "query": query,
                    "message": (
                        "The local Chroma database contains no searchable chunks yet. "
                        "Keep Ollama running and check vault_status for refresh progress."
                    ),
                    "results": [],
                }

            try:
                search_results = NoteSearcher(
                    store=store,
                    embedder=self._embedder_factory(),
                ).search(query=query, top_k=top_k)
            except Exception as error:
                print(f"Obsidian RAG search error: {error}", file=sys.stderr)
                sanitized = self._sanitized_error(error)
                return {
                    "status": "search_failed",
                    "query": query,
                    "message": sanitized["message"],
                    "error_code": sanitized["code"],
                    "results": [],
                }
            results = [
                {
                    "rank": rank,
                    "text": result.document,
                    "source": result.source,
                    "heading": result.heading,
                    "distance": result.distance,
                    "chunk_id": result.chunk_id,
                    "citation": f"{result.source}#{result.heading}",
                }
                for rank, result in enumerate(search_results, start=1)
            ]
            return {
                "status": "ok",
                "query": query,
                "result_count": len(results),
                "results": results,
            }
        finally:
            lock.release()

    def _resolve_note(self, source: str) -> tuple[Path, str]:
        source = source.strip()
        if not source:
            raise ValueError("source cannot be empty")

        untrusted = Path(source)
        if untrusted.is_absolute() or untrusted.drive or ".." in untrusted.parts:
            raise ValueError("source must be a relative path inside the configured vault")
        if any(part.startswith(".") for part in untrusted.parts):
            raise ValueError("hidden vault paths cannot be read")
        if untrusted.suffix.casefold() != ".md":
            raise ValueError("only Markdown (.md) notes can be read")

        try:
            vault_root = self.settings.vault_path.resolve(strict=True)
            candidate = (vault_root / untrusted).resolve(strict=True)
        except OSError as error:
            raise ValueError("source note was not found or cannot be read") from error
        try:
            relative = candidate.relative_to(vault_root)
        except ValueError as error:
            raise ValueError("source resolves outside the configured vault") from error
        if not candidate.is_file():
            raise ValueError("source does not identify a Markdown file")
        if any(part.startswith(".") for part in relative.parts):
            raise ValueError("hidden vault paths cannot be read")
        return candidate, relative.as_posix()

    def read_note(
        self,
        source: str,
        heading: str | None = None,
        max_chars: int = 12_000,
    ) -> dict[str, Any]:
        if not 1 <= max_chars <= 20_000:
            raise ValueError("max_chars must be between 1 and 20000")

        note_path, canonical_source = self._resolve_note(source)
        selected_heading: str | None = None
        if heading and heading.strip():
            requested = heading.strip().casefold()
            vault_file = VaultFile(
                absolute_path=note_path,
                relative_path=canonical_source,
                content_hash=file_sha256(note_path),
            )
            try:
                sections = parse_note(vault_file)
            except (OSError, UnicodeError) as error:
                raise ValueError("source note could not be parsed") from error
            path_matches = [
                section
                for section in sections
                if " > ".join(section.heading_path).casefold() == requested
            ]
            matches = path_matches or [
                section for section in sections if section.heading.casefold() == requested
            ]
            if not matches:
                raise ValueError("the requested heading was not found in this note")
            selected_heading = " > ".join(matches[0].heading_path)
            content = "\n\n".join(section.body for section in matches)
        else:
            try:
                content = note_path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as error:
                raise ValueError("source note could not be read as UTF-8 Markdown") from error

        truncated = len(content) > max_chars
        returned_content = content[:max_chars]
        citation = (
            f"{canonical_source}#{selected_heading}"
            if selected_heading
            else canonical_source
        )
        response: dict[str, Any] = {
            "status": "ok",
            "source": canonical_source,
            "heading": selected_heading,
            "citation": citation,
            "content": returned_content,
            "characters_returned": len(returned_content),
            "total_characters": len(content),
            "truncated": truncated,
        }
        if truncated:
            response["message"] = (
                "The note was truncated. Request a specific heading for focused context."
            )
        return response

    def vault_status(self) -> dict[str, Any]:
        lock_busy = self._lock_is_busy()
        with self._state_lock:
            state = dict(self._state)

        if state.get("refresh_state") == "external_refresh_in_progress" and not lock_busy:
            shared = self._load_status()
            with self._state_lock:
                self._state = shared
            state = shared

        indexed_chunks: int | None = None
        if not lock_busy and self._index_exists():
            lock = self._new_lock()
            try:
                lock.acquire(timeout=0)
                indexed_chunks = self._store_factory().count()
            except (Timeout, Exception):
                indexed_chunks = None
            finally:
                if lock.is_locked:
                    lock.release()
        elif isinstance(state.get("last_summary"), dict):
            indexed_chunks = state["last_summary"].get("stored_chunks")

        refresh_state = state.get("refresh_state", "not_started")
        if lock_busy and refresh_state != "refreshing":
            refresh_state = "external_refresh_in_progress"

        index_available = indexed_chunks is not None and indexed_chunks > 0
        return {
            "read_only": True,
            "index_available": index_available,
            "indexed_chunks": indexed_chunks,
            "embedding_model": self.settings.model,
            "collection": self.settings.collection,
            "refresh_state": refresh_state,
            "last_successful_refresh": state.get("last_successful_refresh"),
            "last_refresh_attempt": state.get("last_refresh_attempt"),
            "last_summary": state.get("last_summary"),
            "error": state.get("error"),
        }
