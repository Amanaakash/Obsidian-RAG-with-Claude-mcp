"""Small data objects shared by the indexing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VaultFile:
    """One Markdown file discovered inside the vault."""

    absolute_path: Path
    relative_path: str
    content_hash: str


@dataclass(frozen=True)
class NoteSection:
    """A heading and the text that belongs below it."""

    source: str
    title: str
    heading: str
    heading_path: tuple[str, ...]
    body: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    outgoing_links: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NoteChunk:
    """The smallest searchable unit stored in Chroma."""

    chunk_id: str
    document: str
    source: str
    title: str
    heading: str
    heading_path: str
    chunk_index: int
    tags: str
    outgoing_links: str
    file_hash: str

    def metadata(self) -> dict[str, str | int]:
        """Return metadata in the scalar-only format Chroma accepts."""

        return {
            "source": self.source,
            "title": self.title,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk_index,
            "tags": self.tags,
            "outgoing_links": self.outgoing_links,
            "file_hash": self.file_hash,
        }


@dataclass(frozen=True)
class SearchResult:
    """A search hit returned to the command-line interface."""

    document: str
    source: str
    heading: str
    distance: float
    chunk_id: str


@dataclass(frozen=True)
class IndexSummary:
    """Counts describing one incremental indexing run."""

    scanned_files: int
    indexed_files: int
    skipped_files: int
    empty_files: int
    deleted_files: int
    stored_chunks: int
