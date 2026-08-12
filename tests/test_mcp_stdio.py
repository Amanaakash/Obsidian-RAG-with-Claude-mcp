from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_mcp_stdio_lists_and_invokes_tools_and_prompt(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note.md").write_text(
        "# MCP\n\nThe server exposes read-only tools.", encoding="utf-8"
    )
    database = tmp_path / "missing-index"

    async def exercise_server() -> None:
        environment = {
            **os.environ,
            "OBSIDIAN_RAG_VAULT": str(vault),
            "OBSIDIAN_RAG_DB": str(database),
            "OBSIDIAN_RAG_MODEL": "all-minilm:22m",
            "OBSIDIAN_RAG_COLLECTION": "obsidian_notes",
            "OBSIDIAN_RAG_OLLAMA_HOST": "http://127.0.0.1:1",
        }
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "obsidian_rag.mcp_server"],
            env=environment,
            cwd=Path.cwd(),
        )

        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                prompts = await session.list_prompts()
                tool_names = {tool.name for tool in tools.tools}
                prompt_names = {prompt.name for prompt in prompts.prompts}

                assert tool_names == {"search_notes", "read_note", "vault_status"}
                assert "ask_obsidian" in prompt_names

                search = await session.call_tool(
                    "search_notes", {"query": "MCP tools", "top_k": 3}
                )
                read = await session.call_tool(
                    "read_note", {"source": "Note.md", "heading": "MCP"}
                )
                status = await session.call_tool("vault_status", {})
                prompt = await session.get_prompt(
                    "ask_obsidian", {"question": "What tools are available?"}
                )

                assert search.is_error is not True
                assert read.is_error is not True
                assert status.is_error is not True
                assert "Note.md#MCP" in json.dumps(read.structured_content)
                assert prompt.messages

    asyncio.run(exercise_server())
