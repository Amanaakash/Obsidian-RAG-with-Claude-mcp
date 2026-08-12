from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
from filelock import FileLock

from obsidian_rag.mcp_service import ObsidianMCPService
from obsidian_rag.mcp_settings import MCPSettings


class LocalTestEmbedder:
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


class FailingEmbedder:
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        raise ConnectionError("private machine path and Ollama details")

    def embed_query(self, query: str) -> list[float]:
        raise ConnectionError("private machine path and Ollama details")


def make_settings(tmp_path: Path) -> MCPSettings:
    vault = tmp_path / "vault"
    vault.mkdir()
    return MCPSettings(
        vault_path=vault,
        database_path=tmp_path / "index",
        model="test-embedding-model",
    )


def make_service(
    settings: MCPSettings,
    embedder: LocalTestEmbedder | FailingEmbedder | None = None,
) -> ObsidianMCPService:
    selected = embedder or LocalTestEmbedder()
    return ObsidianMCPService(settings, embedder_factory=lambda: selected)


def test_refresh_search_citations_status_and_incremental_changes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = settings.vault_path / "ML" / "RAG.md"
    note.parent.mkdir()
    note.write_text(
        "# Retrieval\n\nRAG retrieves external context.\n\n"
        "## Evaluation\n\nRecall at k measures whether relevant notes were retrieved.",
        encoding="utf-8",
    )
    service = make_service(settings)

    service._refresh_worker()
    status = service.vault_status()
    results = service.search_notes("relevant retrieval recall", top_k=1)

    assert status["read_only"] is True
    assert status["refresh_state"] == "ready"
    assert status["indexed_chunks"] == 2
    assert results["status"] == "ok"
    assert results["results"][0]["source"] == "ML/RAG.md"
    assert results["results"][0]["citation"].startswith("ML/RAG.md#")
    json.dumps(results)

    service._refresh_worker()
    assert service.vault_status()["last_summary"]["indexed_files"] == 0

    (settings.vault_path / "New.md").write_text(
        "# New concept\n\nA newly added note.", encoding="utf-8"
    )
    service._refresh_worker()
    assert service.vault_status()["last_summary"]["indexed_files"] == 1

    (settings.vault_path / "New.md").unlink()
    service._refresh_worker()
    assert service.vault_status()["last_summary"]["deleted_files"] == 1


def test_search_validation_no_index_and_refresh_lock(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = make_service(settings)

    assert service.search_notes("question")["status"] == "no_index"
    # A failed/empty refresh may create Chroma's SQLite shell without vectors;
    # it must still be reported as no usable index.
    service._store_factory()
    assert service.search_notes("question")["status"] == "no_index"
    assert service.vault_status()["index_available"] is False
    with pytest.raises(ValueError, match="empty"):
        service.search_notes("   ")
    with pytest.raises(ValueError, match="between 1 and 10"):
        service.search_notes("question", top_k=11)

    with FileLock(str(settings.lock_path)):
        assert service.search_notes("question")["status"] == "index_refreshing"
        competing = make_service(settings)
        competing._refresh_worker()
        assert competing.vault_status()["refresh_state"] == "external_refresh_in_progress"


def test_failed_refresh_preserves_previous_index_and_sanitizes_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = settings.vault_path / "Stable.md"
    note.write_text("# Stable\n\nOriginal searchable knowledge.", encoding="utf-8")
    good_service = make_service(settings)
    good_service._refresh_worker()
    original_count = good_service.vault_status()["indexed_chunks"]

    note.write_text("# Stable\n\nChanged knowledge.", encoding="utf-8")
    failing_service = make_service(settings, FailingEmbedder())
    failing_service._refresh_worker()
    failed_status = failing_service.vault_status()

    assert failed_status["refresh_state"] == "failed"
    assert failed_status["error"] == {
        "code": "ollama_unavailable",
        "message": "The local Ollama embedding service is unavailable.",
    }
    assert "private" not in json.dumps(failed_status)
    assert good_service.vault_status()["indexed_chunks"] == original_count


def test_read_note_full_heading_truncation_and_path_safety(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = settings.vault_path / "Folder" / "Note.md"
    note.parent.mkdir()
    note.write_text(
        "# Topic\n\nIntro text.\n\n## Details\n\n" + ("Useful details. " * 100),
        encoding="utf-8",
    )
    service = make_service(settings)

    full = service.read_note("Folder/Note.md", max_chars=40)
    section = service.read_note("Folder/Note.md", heading="Topic > Details")

    assert full["truncated"] is True
    assert full["characters_returned"] == 40
    assert "specific heading" in full["message"]
    assert section["heading"] == "Topic > Details"
    assert section["citation"] == "Folder/Note.md#Topic > Details"
    assert "Useful details" in section["content"]

    hidden = settings.vault_path / ".secret"
    hidden.mkdir()
    (hidden / "Hidden.md").write_text("secret", encoding="utf-8")
    (settings.vault_path / "plain.txt").write_text("not markdown", encoding="utf-8")

    invalid_sources = [
        "../outside.md",
        str(note.resolve()),
        ".secret/Hidden.md",
        "plain.txt",
    ]
    for invalid in invalid_sources:
        with pytest.raises(ValueError):
            service.read_note(invalid)

    with pytest.raises(ValueError) as missing_error:
        service.read_note("Folder/does-not-exist.md")
    assert str(settings.vault_path) not in str(missing_error.value)

    with pytest.raises(ValueError, match="heading was not found"):
        service.read_note("Folder/Note.md", heading="Missing")
    with pytest.raises(ValueError, match="between 1 and 20000"):
        service.read_note("Folder/Note.md", max_chars=20_001)


def test_search_failure_returns_sanitized_structured_response(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    (settings.vault_path / "Note.md").write_text(
        "# Searchable\n\nExisting content.", encoding="utf-8"
    )
    make_service(settings)._refresh_worker()

    response = make_service(settings, FailingEmbedder()).search_notes("content")

    assert response["status"] == "search_failed"
    assert response["error_code"] == "ollama_unavailable"
    assert "private" not in json.dumps(response)


def test_read_note_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = settings.vault_path / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is not available in this Windows environment")

    with pytest.raises(ValueError, match="outside"):
        make_service(settings).read_note("linked.md")
