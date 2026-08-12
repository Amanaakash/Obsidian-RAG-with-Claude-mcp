# Contributing

Thank you for helping improve Obsidian RAG Assistant.

## Development setup

Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/), then run:

```shell
git clone https://github.com/Amanaakash/Obsidian-RAG-with-Claude-mcp.git
cd Obsidian-RAG-with-Claude-mcp
uv sync --locked --all-groups
uv run pytest -q
```

The automated suite uses fake embedders and does not require Ollama, Claude, or a real Obsidian vault.

For an end-to-end local test, install Ollama, pull `all-minilm:22m`, and use only the included `sample_vault` or notes you are permitted to process.

## Making changes

- Keep the MCP interface read-only unless a separate proposal explicitly addresses write safety.
- Preserve source paths and heading metadata in retrieval results.
- Add or update tests for behavior changes.
- Keep platform-specific paths out of source, tests, and documentation examples.
- Do not commit real vault content, generated Chroma data, credentials, or Claude configuration.
- Keep stdout free of diagnostics in the stdio MCP server; write diagnostics to stderr.

## Pull requests

1. Create a focused branch from `main`.
2. Make the smallest coherent change.
3. Run `uv run pytest -q`, `uv build`, and `uv run obsidian-rag --help`.
4. Update README or CHANGELOG when public behavior changes.
5. Open a pull request describing the problem, solution, tests, and privacy implications.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
