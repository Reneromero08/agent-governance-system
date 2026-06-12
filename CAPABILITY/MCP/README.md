<!-- CONTENT_HASH: c084fa70fe92bd7c6a309f083be69ecad337e9dffc70bfad5c4c25c26587e514 -->

# MCP Integration README

This directory contains the AGS MCP (Model Context Protocol) server, enabling external AI clients to interact with the Agent Governance System.

The server speaks JSON-RPC 2.0 over stdio. It supports both Content-Length framed messages (VS Code style clients) and newline-delimited JSON (simple clients); the mode is auto-detected from the first request.

---

## Quick Start

MCP clients launch the server themselves over stdio - there is no daemon to
manage. Point your client at the canonical entrypoint:

```
LAW/CONTRACTS/ags_mcp_entrypoint.py
```

### Connect Claude Desktop

1. Find your Claude Desktop config file:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add this entry (or copy from `CAPABILITY/MCP/claude_desktop_config.json`,
   adjusting the repository path for your machine):

**Windows:**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/LAW/CONTRACTS/ags_mcp_entrypoint.py"],
      "cwd": "D:/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

**WSL (python3):**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python3",
      "args": ["/mnt/d/CCC 2.0/AI/agent-governance-system/LAW/CONTRACTS/ags_mcp_entrypoint.py"],
      "cwd": "/mnt/d/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

3. Restart Claude Desktop (or the MCP client).

4. You should now see AGS tools available in Claude.

### Foreground / manual run

```cmd
CAPABILITY\MCP\start_simple.cmd
```

Runs the server in the current window over stdio. Press Ctrl+C to stop.

`server_wrapper.py` is an optional launcher that starts the server as a
background process (PID file in `LAW/CONTRACTS/_runs/mcp_logs/server.pid`)
for setups that want a single shared instance.

## Test the Server

Run the built-in test:

```bash
python LAW/CONTRACTS/ags_mcp_entrypoint.py --test
```

Run the governance smoke test (exits non-zero on failure):

```bash
python CAPABILITY/MCP/verify_governance.py
```

## Available Tools

| Tool | Description |
|------|-------------|
| `context_search` | Search ADRs, preferences, and other context records in LAW/CONTEXT |
| `context_review` | Check for overdue or upcoming ADR reviews |
| `canon_read` | Read canon files (CONTRACT, INVARIANTS, etc.) |
| `skill_run` | Execute an AGS skill with JSON input (governed: preflight + admission + critic) |
| `codebook_lookup` | Look up codebook entries by ID, with optional semantic query |
| `skill_discovery` | Find skills matching a natural-language intent |
| `find_related` | Find artifacts related to a given artifact ID |
| `cassette_network_query` | Semantic search across the cassette network (NAVIGATION/CORTEX) |
| `semantic_stats` | Embedding and cassette network statistics |
| `memory` | Unified memory operations: save, query, recall, neighbors, stats |
| `session_info` | Current MCP session metadata and audit log entries |

Notes:

- `skill_run` is the only tool that executes code. It is wrapped by the
  governance gate (preflight, admission control, critic) and fails closed.
- Skill execution has no timeout by default. Set the `AGS_SKILL_TIMEOUT`
  environment variable (seconds) to bound it.

## Available Resources

| URI | Description |
|-----|-------------|
| `ags://canon/contract` | The CONTRACT.md (supreme authority) |
| `ags://canon/invariants` | The INVARIANTS.md (locked decisions) |
| `ags://canon/genesis` | The Genesis Prompt for session bootstrap |
| `ags://canon/versioning` | The VERSIONING.md |
| `ags://canon/arbitration` | The ARBITRATION.md |
| `ags://canon/deprecation` | The DEPRECATION.md |
| `ags://canon/migration` | The MIGRATION.md |
| `ags://context/decisions` | Live list of all ADRs |
| `ags://context/preferences` | Live list of STYLE records |
| `ags://context/rejected` | Live list of rejected proposals |
| `ags://context/open` | Live list of open questions |
| `ags://cortex/index` | Full cortex index in JSON |
| `ags://maps/entrypoints` | The ENTRYPOINTS.md |
| `ags://agents` | The AGENTS.md |

## Available Prompts

| Name | Description |
|------|-------------|
| `genesis` | Bootstrap prompt for AGS session initialization |
| `skill_template` | Template for creating a new Skill |
| `conflict_resolution` | Guide for resolving conflicts in Canon (Arbitration) |
| `deprecation_workflow` | Checklist for deprecating tokens or features |

## PowerShell Bridge (standalone)

`powershell_bridge.ps1` is a local HTTP bridge for controlled command
execution (used by the `powershell-bridge` skill). It is NOT exposed as an
MCP tool.

Security notes - read before starting it:

1. Set a fresh token in `CAPABILITY/MCP/powershell_bridge_config.json`
   (never reuse a token that has been committed to git).
2. Populate `allowed_prefixes` with the commands you intend to allow.
   The bridge refuses to start with an empty allowlist unless
   `allow_all_commands` is explicitly set to `true`.
3. Keep `listen_host` on `127.0.0.1` unless you need WSL2 access.

WSL note: WSL2 cannot reach Windows services bound to 127.0.0.1. Set
`listen_host` to `0.0.0.0` and `connect_host` to the Windows host IP
(from WSL: `grep nameserver /etc/resolv.conf`) in
`CAPABILITY/MCP/powershell_bridge_config.json`.

## Example Usage (in Claude)

"Read the CONTRACT rules"
-> Claude uses `canon_read` tool with `{"file": "CONTRACT"}`

"What do we know about the packer?"
-> Claude uses `cassette_network_query` tool with `{"query": "packer"}`

"Are any ADR reviews overdue?"
-> Claude uses `context_review` tool

"Run the pack-validate skill on the latest pack"
-> Claude uses `skill_run` tool with `{"skill": "pack-validate", "input": {...}}`
