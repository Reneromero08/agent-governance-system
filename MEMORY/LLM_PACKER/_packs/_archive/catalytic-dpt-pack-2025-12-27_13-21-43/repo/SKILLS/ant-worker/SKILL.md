---
name: ant-worker
version: "0.1.0"
description: Executes mechanical file operations, code adaptation, and validation tasks. Use when you need to copy/move files with hash verification, adapt code imports, or run validations. Reports to Governor, logs to immutable ledger.
compatibility: Python 3.8+, MCP server running
---

# Ant Worker

Distributed task executor for CATALYTIC-DPT swarm.

## Usage

```bash
python scripts/run.py input.json output.json
```

## Task Types

| Type | Operations |
|------|------------|
| `file_operation` | copy, move, delete, read |
| `code_adapt` | find/replace, update imports |
| `validate` | run fixtures, check syntax |
| `research` | analyze file structure |

## Input Schema

```json
{
  "task_id": "copy-files-001",
  "task_type": "file_operation",
  "operation": "copy",
  "files": [
    {"source": "path/to/src", "destination": "path/to/dest"}
  ],
  "verify_integrity": true
}
```

## Governance

- All operations hash-verified (SHA-256)
- Logged to `CONTRACTS/_runs/<task_id>/`
- On error: STOP and escalate to Governor
