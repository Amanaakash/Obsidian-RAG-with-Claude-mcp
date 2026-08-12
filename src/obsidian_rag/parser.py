"""Parse Obsidian Markdown into heading-based sections."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from obsidian_rag.models import NoteSection, VaultFile


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
INLINE_TAG_PATTERN = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]+)")


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Separate optional YAML front matter from the Markdown body."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            raw_frontmatter = "\n".join(lines[1:index])
            try:
                parsed = yaml.safe_load(raw_frontmatter) or {}
            except yaml.YAMLError:
                # A malformed property block should not make the entire vault
                # unindexable. Ignore only its metadata and keep the note body.
                return {}, "\n".join(lines[index + 1 :])
            metadata = parsed if isinstance(parsed, dict) else {}
            return metadata, "\n".join(lines[index + 1 :])

    return {}, text


def _normalise_tags(value: Any) -> set[str]:
    """Convert common Obsidian tag property formats into plain strings."""

    if isinstance(value, str):
        candidates = re.split(r"[,\s]+", value)
    elif isinstance(value, list):
        candidates = [str(item) for item in value]
    else:
        candidates = []

    return {candidate.strip().lstrip("#") for candidate in candidates if candidate.strip()}


def _clean_heading(raw_heading: str) -> str:
    """Remove optional trailing Markdown heading markers."""

    return raw_heading.rstrip("#").strip()


def parse_note(vault_file: VaultFile) -> list[NoteSection]:
    """Read one note and split it into sections while preserving hierarchy."""

    text = vault_file.absolute_path.read_text(encoding="utf-8-sig")
    frontmatter, body = _split_frontmatter(text)
    lines = body.splitlines()

    first_h1 = next(
        (
            _clean_heading(match.group(2))
            for line in lines
            if (match := HEADING_PATTERN.match(line)) and len(match.group(1)) == 1
        ),
        None,
    )
    title = str(frontmatter.get("title") or first_h1 or Path(vault_file.relative_path).stem)

    tags = _normalise_tags(frontmatter.get("tags"))
    non_heading_body = "\n".join(line for line in lines if not HEADING_PATTERN.match(line))
    tags.update(INLINE_TAG_PATTERN.findall(non_heading_body))
    outgoing_links = set(WIKILINK_PATTERN.findall(body))

    sections: list[NoteSection] = []
    heading_stack: list[str] = []
    current_heading = "Introduction"
    current_heading_path: tuple[str, ...] = ("Introduction",)
    current_lines: list[str] = []

    def flush_section() -> None:
        section_body = "\n".join(current_lines).strip()
        if not section_body:
            return
        sections.append(
            NoteSection(
                source=vault_file.relative_path,
                title=title,
                heading=current_heading,
                heading_path=current_heading_path,
                body=section_body,
                tags=tuple(sorted(tags)),
                outgoing_links=tuple(sorted(outgoing_links)),
            )
        )

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if not match:
            current_lines.append(line)
            continue

        flush_section()
        current_lines = []

        level = len(match.group(1))
        current_heading = _clean_heading(match.group(2))
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(current_heading)
        current_heading_path = tuple(heading_stack)

    flush_section()
    return sections
