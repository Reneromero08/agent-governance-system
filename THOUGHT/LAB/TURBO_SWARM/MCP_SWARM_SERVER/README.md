# Swarm MCP Server

Experimental MCP server for swarm coordination, extracted from the main AGS MCP server.

## Overview

This is a minimal standalone MCP server providing only swarm coordination tools:

- **Message Board**: Inter-agent communication via persistent message boards
- **Agent Inbox**: Task claiming and finalization for distributed work

## Tools

| Tool | Description |
|------|-------------|
| `message_board_list` | List messages from a named board |
| `message_board_write` | Post, pin, unpin, delete, or purge messages (governed) |
| `agent_inbox_list` | List tasks by status (pending/active/completed/failed) |
| `agent_inbox_claim` | Claim a pending task for processing |
| `agent_inbox_finalize` | Mark a task as completed or failed with result |

## Usage

### Stdio Mode (Default)

```bash
python server.py
```

The server communicates via JSON-RPC 2.0 with Content-Length framing (MCP standard).

### Test Mode

```bash
python server.py --test
```

Runs a quick smoke test of all tools.

## Data Locations

- **Message Boards**: `LAW/CONTRACTS/_runs/message_board/{board}.jsonl`
- **Agent Inbox**: `INBOX/agents/Local Models/{STATUS}_TASKS/*.json`
- **Board Roles**: `CAPABILITY/MCP/board_roles.json`

## Governance

The `message_board_write` tool uses the `@governed_tool` decorator which attempts to run preflight checks via `ags.py`. In experimental mode, missing governance tools are gracefully handled.

## Integration with TURBO_SWARM

This server is designed to work with the swarm orchestrators in this LAB:

1. Orchestrator posts tasks to the inbox
2. Worker agents claim tasks via `agent_inbox_claim`
3. Workers report results via `agent_inbox_finalize`
4. Message board provides coordination and status updates

## MCP Client Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "swarm": {
      "command": "python",
      "args": ["THOUGHT/LAB/TURBO_SWARM/MCP_SWARM_SERVER/server.py"]
    }
  }
}
```
