**required_canon_version:** >=3.0.0

# Skill: mcp-toolkit

**Version:** 1.0.0

**Status:** Active

## Purpose

Unified toolkit for MCP (Model Context Protocol) operations including server building, access validation, extension verification, message board management, pre-commit checks, and smoke testing.

## Trigger

Use when performing any MCP-related operation:
- Building MCP servers
- Validating agent access patterns
- Verifying MCP extensions in IDEs
- Managing message board operations
- Running pre-commit MCP checks
- Performing MCP smoke tests
- Wrapping MCP server execution

## Operations

| Operation | Description |
|-----------|-------------|
| `build` | Build MCP servers (placeholder) |
| `validate_access` | Validate agent uses MCP tools vs manual operations |
| `verify_extension` | Verify AGS MCP server in IDE extensions |
| `message_board` | Message board operations (placeholder) |
| `precommit` | Pre-commit MCP health checks |
| `smoke` | MCP server smoke testing |
| `adapt` | MCP adapter task wrapper |

## Inputs

`input.json` fields:

**Common:**
- `operation` (string, required): One of the operations listed above

**For `validate_access`:**
- `agent_action` (string): Description of agent action
- `agent_code_snippet` (string): Code the agent wrote
- `files_accessed` (array): Files accessed manually
- `databases_queried` (array): Databases queried directly

**For `verify_extension`:**
- `client` (string): `vscode`, `claude`, or `generic`
- `entrypoint_substring` (string): Path to MCP entrypoint
- `args` (array): Arguments for entrypoint

**For `precommit`:**
- `entrypoint` (string): Path to MCP entrypoint
- `auto_entrypoint` (string): Path to auto entrypoint
- `args` (array): Args for entrypoint
- `require_running` (bool): Require running server PID
- `require_autostart` (bool): Require Windows autostart
- `dry_run` (bool): Skip execution

**For `smoke`:**
- `entrypoint_substring` (string): Path to MCP entrypoint
- `args` (array): Arguments for entrypoint
- `bridge_smoke` (object): Bridge smoke test config

**For `adapt`:**
- `task` (object): Task information with `id` field

**For `build`:**
- `task` (object): Task information

**For `message_board`:**
- (placeholder - not yet implemented)

## Outputs

Output varies by operation. See individual skill documentation for details.

## Constraints

- Uses CORTEX query for file discovery (no direct filesystem scan)
- Must not modify canon or context
- Writes only under allowed roots via GuardedWriter
- On non-Windows, running/autostart checks use PowerShell bridge

## Fixtures

- `fixtures/basic/` - Basic operation tests

**required_canon_version:** >=3.0.0
