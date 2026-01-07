# MCP Toolkit

Unified toolkit for MCP (Model Context Protocol) operations. Consolidates 7 formerly separate skills into a single multi-operation skill.

## Operations

| Operation | Description | Source Skill |
|-----------|-------------|--------------|
| `build` | Build MCP servers | `mcp-builder` |
| `validate_access` | Validate MCP tool usage | `mcp-access-validator` |
| `verify_extension` | Verify MCP in IDE extensions | `mcp-extension-verify` |
| `message_board` | Message board ops | `mcp-message-board` |
| `precommit` | Pre-commit MCP checks | `mcp-precommit-check` |
| `smoke` | MCP smoke testing | `mcp-smoke` |
| `adapt` | MCP adapter wrapper | `mcp-adapter` |

## Usage

### CLI

```bash
python run.py input.json output.json
```

### Input JSON Format

All operations require an `operation` field:

```json
{
  "operation": "validate_access|verify_extension|precommit|smoke|build|adapt|message_board",
  ...operation-specific fields...
}
```

### Examples

**Validate MCP access:**
```json
{
  "operation": "validate_access",
  "agent_action": "I need to read the CONTRACT.md file",
  "agent_code_snippet": "open('LAW/CANON/CONTRACT.md').read()"
}
```

**Verify extension:**
```json
{
  "operation": "verify_extension",
  "client": "vscode",
  "args": ["--test"]
}
```

**Pre-commit check (dry run):**
```json
{
  "operation": "precommit",
  "dry_run": true
}
```

**Smoke test:**
```json
{
  "operation": "smoke",
  "args": ["--test"]
}
```

## Permissions

- Uses CORTEX query for file discovery (no direct filesystem scan)
- Writes only to allowed roots via GuardedWriter
- PowerShell bridge for cross-platform Windows checks

## Migration from Legacy Skills

This toolkit replaces:
- `mcp/mcp-builder` → Use `operation: "build"`
- `mcp/mcp-access-validator` → Use `operation: "validate_access"`
- `mcp/mcp-extension-verify` → Use `operation: "verify_extension"`
- `mcp/mcp-message-board` → Use `operation: "message_board"`
- `mcp/mcp-precommit-check` → Use `operation: "precommit"`
- `mcp/mcp-smoke` → Use `operation: "smoke"`
- `mcp/mcp-adapter` → Use `operation: "adapt"`

Legacy skills are deprecated and will be removed in a future version.
