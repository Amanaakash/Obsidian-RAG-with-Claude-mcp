"""Turn heading-based note sections into moderately sized search chunks."""

from __future__ import annotations

import hashlib
import json
import re

from obsidian_rag.models import NoteChunk, NoteSection


PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n")


def _split_long_text(text: str, max_characters: int) -> list[str]:
    """Prefer paragraph boundaries, with a simple hard split as fallback."""

    paragraphs = [part.strip() for part in PARAGRAPH_BREAK_PATTERN.split(text) if part.strip()]
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_characters:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                paragraph[start : start + max_characters].strip()
                for start in range(0, len(paragraph), max_characters)
            )
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_characters:
            current = candidate
        else:
            pieces.append(current)
            current = paragraph

    if current:
        pieces.append(current)

    return pieces or [text.strip()]


def chunk_sections(
    sections: list[NoteSection],
    file_hash: str,
    max_characters: int = 1_800,
) -> list[NoteChunk]:
    """Create stable, metadata-rich chunks from parsed note sections."""

    chunks: list[NoteChunk] = []
    for section_index, section in enumerate(sections):
        for section_chunk_index, body_piece in enumerate(
            _split_long_text(section.body, max_characters=max_characters)
        ):
            identity = "|".join(
                [
                    section.source,
                    " > ".join(section.heading_path),
                    str(section_index),
                    str(section_chunk_index),
                ]
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            document = (
                f"Note: {section.title}\n"
                f"Section: {' > '.join(section.heading_path)}\n\n"
                f"{body_piece}"
            )
            chunks.append(
                NoteChunk(
                    chunk_id=chunk_id,
                    document=document,
                    source=section.source,
                    title=section.title,
                    heading=section.heading,
                    heading_path=" > ".join(section.heading_path),
                    chunk_index=section_chunk_index,
                    tags=json.dumps(section.tags),
                    outgoing_links=json.dumps(section.outgoing_links),
                    file_hash=file_hash,
                )
            )

    return chunks
