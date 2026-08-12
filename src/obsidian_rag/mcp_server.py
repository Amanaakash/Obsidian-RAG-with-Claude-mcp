"""MCP v2 stdio server connecting Claude to the local Obsidian RAG index."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp.server import MCPServer

from obsidian_rag.mcp_service import ObsidianMCPService
from obsidian_rag.mcp_settings import MCPSettings


SERVER_INSTRUCTIONS = """
This is a read-only connection to the user's local Obsidian knowledge base.
For questions about the user's notes, call search_notes before answering. Use
read_note only when a complete note or one specific heading is needed. Cite
note-derived claims as [path#heading]. You may supplement the notes with your
own knowledge, but put unsupported additions under a clearly labeled
"General knowledge" section. Never imply that general knowledge came from the
vault. No tool can edit, create, or delete notes.
""".strip()


def create_mcp_server(service: ObsidianMCPService) -> MCPServer:
    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[None]:
        service.start_background_refresh()
        try:
            yield
        finally:
            # Avoid terminating the process while Chroma is persisting an
            # incremental refresh. Forced termination can leave HNSW files in
            # an unreadable state.
            await asyncio.to_thread(service.wait_for_refresh, 300)

    server = MCPServer(
        name="obsidian-rag",
        title="Obsidian Knowledge Assistant",
        description="Read-only semantic search and note reading for a local Obsidian vault.",
        instructions=SERVER_INSTRUCTIONS,
        version="0.2.0",
        lifespan=lifespan,
    )

    @server.tool(structured_output=True)
    def search_notes(query: str, top_k: int = 5) -> dict[str, Any]:
        """Semantically search indexed Obsidian notes and return cited passages."""

        return service.search_notes(query=query, top_k=top_k)

    @server.tool(structured_output=True)
    def read_note(
        source: str,
        heading: str | None = None,
        max_chars: int = 12_000,
    ) -> dict[str, Any]:
        """Read one Markdown note or an exact heading using a safe relative path."""

        return service.read_note(
            source=source,
            heading=heading,
            max_chars=max_chars,
        )

    @server.tool(structured_output=True)
    def vault_status() -> dict[str, Any]:
        """Check local index availability, chunk count, and refresh state."""

        return service.vault_status()

    @server.prompt()
    def ask_obsidian(question: str) -> str:
        """Answer a question using the Obsidian vault with explicit citations."""

        return f"""Use the Obsidian Knowledge Assistant tools to answer this question:

{question}

Instructions:
1. Call search_notes before drafting the answer.
2. Call read_note only if a retrieved passage needs more surrounding context.
3. Cite every vault-derived claim as [path#heading].
4. If you add information not supported by the notes, put it under a separate
   heading named "General knowledge" and do not cite it as vault content.
5. If retrieval is insufficient, say what evidence is missing.
"""

    return server


def main() -> None:
    try:
        settings = MCPSettings.from_environment()
        service = ObsidianMCPService(settings)
        create_mcp_server(service).run(transport="stdio")
    except Exception as error:
        print(f"Obsidian RAG MCP failed to start: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
