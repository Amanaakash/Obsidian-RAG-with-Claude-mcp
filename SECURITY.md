# Security Policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability or include private Obsidian notes, local paths, credentials, Claude configuration, or retrieved note contents in a report.

Use GitHub's private vulnerability reporting feature:

1. Open the repository's **Security** tab.
2. Select **Advisories**.
3. Choose **Report a vulnerability**.

Include the affected version, impact, reproduction steps using synthetic data, and any suggested mitigation. The maintainer will acknowledge a valid report as soon as practical and coordinate disclosure after a fix is available.

## Supported versions

Security fixes are applied to the latest tagged release and the `main` branch.

## Security model

- The MCP tools are intentionally read-only.
- Vault paths are configured locally through environment variables.
- Chroma and Ollama run locally by default.
- Content returned by an MCP tool is provided to the connected Claude client.
- Generated vector indexes may contain recoverable note text and must be treated as private data.
