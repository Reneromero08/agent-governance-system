---
name: ant-worker
description: "Distributed task executor for CATALYTIC-DPT swarm."
---
<!-- CONTENT_HASH: e9b3918a7aa216870420a0d5852f11f7e78c5647138300d6ae772490d6e79ffc -->

**required_canon_version:** >=3.0.0


# Skill: ant-worker
**Version:** 0.1.0
**Status:** Active


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

**required_canon_version:** >=3.0.0

