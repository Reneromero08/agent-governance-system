# MCP Integration README

This directory contains the AGS MCP (Model Context Protocol) server, enabling external AI clients to interact with the Agent Governance System.

## Quick Start

Preferred entrypoint: `CONTRACTS/_runs/ags_mcp_entrypoint.py` (redirects audit logs to `CONTRACTS/_runs/mcp_logs/`).

### Connect Claude Desktop

1. Find your Claude Desktop config file:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add one of these entries (or update your existing):

**Windows (native Python):**

```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/CONTRACTS/_runs/ags_mcp_entrypoint.py"],
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
      "args": ["/mnt/d/CCC 2.0/AI/agent-governance-system/CONTRACTS/_runs/ags_mcp_entrypoint.py"],
      "cwd": "/mnt/d/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

3. Restart Claude Desktop (or the MCP client).

4. You should now see AGS tools available in Claude.

## Test the Server

Run the built-in test:
```bash
python CONTRACTS/_runs/ags_mcp_entrypoint.py --test
```

Or use the verification skills:
- `mcp-smoke` (CLI smoke test)
- `mcp-extension-verify` (extension-agnostic checklist + smoke test)

## Available Tools

| Tool | Description |
|------|-------------|
| `cortex_query` | Search the cortex index for files and entities |
| `context_search` | Search ADRs, preferences, and other context records |
| `context_review` | Check for overdue or upcoming ADR reviews |
| `canon_read` | Read canon files (CONTRACT, INVARIANTS, etc.) |
| `skill_run` | Execute an AGS skill with JSON input |
| `pack_validate` | Validate a memory pack for completeness |
| `critic_run` | Run governance checks via `TOOLS/critic.py` |
| `adr_create` | Create a new ADR with the standard template |
| `commit_ceremony` | Execute commit ceremony checks |
| `codebook_lookup` | Look up codebook entries by ID |
| `research_cache` | Manage the persistent research cache |

## Available Resources

| URI | Description |
|-----|-------------|
| `ags://canon/contract` | The CONTRACT.md (supreme authority) |
| `ags://canon/invariants` | The INVARIANTS.md (locked decisions) |
| `ags://canon/genesis` | The Genesis Prompt for session bootstrap |
| `ags://canon/versioning` | The VERSIONING.md |
| `ags://context/decisions` | Live list of all ADRs |
| `ags://context/preferences` | Live list of STYLE records |
| `ags://cortex/index` | Full cortex index in JSON |
| `ags://maps/entrypoints` | The ENTRYPOINTS.md |
| `ags://agents` | The AGENTS.md |

## Available Prompts

| Name | Description |
|------|-------------|
| `genesis` | Bootstrap prompt for AGS session initialization |
| `commit_ceremony` | Checklist for the commit ceremony |
| `adr_template` | Template for creating new ADRs |

## Example Usage (in Claude)

"Read the CONTRACT rules"
→ Claude uses `canon_read` tool with `{"file": "CONTRACT"}`

"What files mention 'packer'?"
→ Claude uses `cortex_query` tool with `{"query": "packer"}`

"Are any ADR reviews overdue?"
→ Claude uses `context_review` tool

"Run the pack-validate skill on the latest pack"
→ Claude uses `skill_run` tool with `{"skill": "pack-validate", "input": {...}}`
