# MCP Integration README

This directory contains the AGS MCP (Model Context Protocol) server, enabling external AI clients to interact with the Agent Governance System.

## Quick Start

### Connect Claude Desktop

1. Find your Claude Desktop config file:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add this entry (or merge with existing):

```json
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/MCP/server.py"],
      "cwd": "D:/CCC 2.0/AI/agent-governance-system"
    }
  }
}
```

3. Restart Claude Desktop.

4. You should now see AGS tools available in Claude.

## Test the Server

Run the built-in test:
```bash
python MCP/server.py --test
```

## Available Tools

| Tool | Description |
|------|-------------|
| `cortex_query` | Search the cortex index for files and entities |
| `context_search` | Search ADRs, preferences, and other context records |
| `context_review` | Check for overdue or upcoming ADR reviews |
| `canon_read` | Read canon files (CONTRACT, INVARIANTS, etc.) |
| `skill_run` | Execute an AGS skill with JSON input |
| `pack_validate` | Validate a memory pack for completeness |

## Available Resources

| URI | Description |
|-----|-------------|
| `ags://canon/contract` | The CONTRACT.md (supreme authority) |
| `ags://canon/invariants` | The INVARIANTS.md (locked decisions) |
| `ags://canon/genesis` | The Genesis Prompt for session bootstrap |
| `ags://context/decisions` | Live list of all ADRs |
| `ags://context/preferences` | Live list of STYLE records |
| `ags://cortex/index` | Full cortex index in JSON |

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
