"""Discover Markdown files and calculate stable content hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from obsidian_rag.models import VaultFile


DEFAULT_IGNORED_DIRECTORIES = frozenset({".git", ".obsidian", ".trash"})


def file_sha256(path: Path) -> str:
    """Hash file bytes so unchanged notes can be skipped."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_vault(
    vault_path: Path,
    ignored_directories: frozenset[str] = DEFAULT_IGNORED_DIRECTORIES,
) -> list[VaultFile]:
    """Recursively return every visible Markdown file in a vault."""

    vault_path = vault_path.expanduser().resolve()
    if not vault_path.is_dir():
        raise ValueError(f"Vault directory does not exist: {vault_path}")

    discovered: list[VaultFile] = []
    for path in vault_path.rglob("*.md"):
        relative_parts = path.relative_to(vault_path).parts
        if any(part in ignored_directories or part.startswith(".") for part in relative_parts[:-1]):
            continue

        discovered.append(
            VaultFile(
                absolute_path=path,
                relative_path=path.relative_to(vault_path).as_posix(),
                content_hash=file_sha256(path),
            )
        )

    return sorted(discovered, key=lambda item: item.relative_path.casefold())
