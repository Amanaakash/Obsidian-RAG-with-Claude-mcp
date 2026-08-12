from pathlib import Path

from obsidian_rag.scanner import scan_vault


def test_scanner_recurses_and_ignores_internal_and_hidden_directories(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    (vault / "Nested").mkdir(parents=True)
    (vault / "Nested" / "Visible.md").write_text("visible", encoding="utf-8")

    for hidden_name in (".obsidian", ".git", ".trash", ".private"):
        hidden_directory = vault / hidden_name
        hidden_directory.mkdir()
        (hidden_directory / "Hidden.md").write_text("hidden", encoding="utf-8")

    discovered = scan_vault(vault)

    assert [item.relative_path for item in discovered] == ["Nested/Visible.md"]
