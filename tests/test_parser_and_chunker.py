from pathlib import Path

from obsidian_rag.chunker import chunk_sections
from obsidian_rag.models import VaultFile
from obsidian_rag.parser import parse_note
from obsidian_rag.scanner import file_sha256


def test_parser_preserves_heading_hierarchy_tags_and_links(tmp_path: Path) -> None:
    note_path = tmp_path / "RAG.md"
    note_path.write_text(
        """---
title: My RAG Notes
tags: [rag, learning]
---
# Retrieval
Intro text with #search and [[Embeddings|vector embeddings]].

## Hybrid Search
Hybrid search combines semantic and keyword retrieval.
""",
        encoding="utf-8",
    )
    vault_file = VaultFile(note_path, "RAG.md", file_sha256(note_path))

    sections = parse_note(vault_file)

    assert len(sections) == 2
    assert sections[0].title == "My RAG Notes"
    assert sections[1].heading == "Hybrid Search"
    assert sections[1].heading_path == ("Retrieval", "Hybrid Search")
    assert sections[0].tags == ("learning", "rag", "search")
    assert sections[0].outgoing_links == ("Embeddings",)


def test_chunk_ids_are_stable_and_metadata_contains_source(tmp_path: Path) -> None:
    note_path = tmp_path / "Notes.md"
    note_path.write_text("# Topic\n\nA short explanation.", encoding="utf-8")
    content_hash = file_sha256(note_path)
    sections = parse_note(VaultFile(note_path, "Notes.md", content_hash))

    first_run = chunk_sections(sections, file_hash=content_hash)
    second_run = chunk_sections(sections, file_hash=content_hash)

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert first_run[0].metadata()["source"] == "Notes.md"
    assert first_run[0].metadata()["heading_path"] == "Topic"


def test_long_sections_are_split_into_multiple_chunks(tmp_path: Path) -> None:
    note_path = tmp_path / "Long.md"
    note_path.write_text("# Long section\n\n" + ("retrieval context " * 100), encoding="utf-8")
    content_hash = file_sha256(note_path)
    sections = parse_note(VaultFile(note_path, "Long.md", content_hash))

    chunks = chunk_sections(sections, file_hash=content_hash, max_characters=200)

    assert len(chunks) > 1
    assert all(chunk.heading == "Long section" for chunk in chunks)


def test_repeated_headings_receive_different_chunk_ids(tmp_path: Path) -> None:
    note_path = tmp_path / "Repeated.md"
    note_path.write_text(
        "# Weekly review\n\nFirst review.\n\n# Weekly review\n\nSecond review.",
        encoding="utf-8",
    )
    content_hash = file_sha256(note_path)
    sections = parse_note(VaultFile(note_path, "Repeated.md", content_hash))

    chunks = chunk_sections(sections, file_hash=content_hash)

    assert len(chunks) == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_malformed_frontmatter_does_not_prevent_parsing(tmp_path: Path) -> None:
    note_path = tmp_path / "Malformed.md"
    note_path.write_text(
        "---\ntags: [rag\nthis is not valid yaml\n---\n# Valid body\n\nUseful note content.",
        encoding="utf-8",
    )
    content_hash = file_sha256(note_path)

    sections = parse_note(VaultFile(note_path, "Malformed.md", content_hash))

    assert len(sections) == 1
    assert sections[0].heading == "Valid body"
    assert sections[0].body == "Useful note content."
