# Ant Worker Skill

**Purpose**: Distributed task executor for CATALYTIC-DPT
**Role**: Executes mechanical tasks under Governor supervision
**Model**: Configurable (default: Kilo Code CLI with Grok Fast)

> [!IMPORTANT]
> Ants follow strict templates. They cannot deviate from protocol.
> Schema-valid output or HARD FAIL.

---

## Architecture

```
Governor (The Conductor)
    ├─ Distributes subtask to Ant-1
    ├─ Distributes subtask to Ant-2
    └─ Monitors all workers

Ant Worker 1: File operations (copy, move, analyze)
Ant Worker 2: Code adaptation (update imports, refactor)

All workers:
- Report to Governor
- Write to MCP-controlled ledger
- Share terminal output (bidirectional)
- Escalate to higher authority on uncertainty
```

---

## Usage

```bash
python run.py input.json output.json
```

## Input Schema

```json
{
  "task_id": "ant-copy-files-1",
  "task_type": "file_operation",
  "operation": "copy",
  "files": [...],
  "verify_integrity": true,
  "timeout_seconds": 60,
  "on_error": "STOP_AND_REPORT"
}
```

## Task Types

- `file_operation`: Copy, move, delete, read files
- `code_adapt`: Update imports, refactor code
- `validate`: Test files, run fixtures
- `research`: Analyze and understand code

---

## Governance Rules

1. **Template-Bound**: Ants ONLY fill in template placeholders
2. **MCP-Mediated**: All file operations via MCP (no direct writes)
3. **Hash Verification**: Every operation verified with SHA-256
4. **Immutable Ledger**: Every action logged to CONTRACTS/_runs/
5. **Escalation**: If uncertain, STOP and escalate

---

## Integration with Governor

```
Governor: "Ant-1, execute task spec"
    ↓
Ant reads template JSON
    ↓
Ant fills placeholders (cannot modify structure)
    ↓
Ant calls MCP: file_sync(source, dest, verify_hash=True)
    ↓
MCP logs operation to ledger
    ↓
Ant reports to Governor: "✓ Success, hash verified"
```

---

## Model Configuration

Edit `~/.kilocode/cli/config.json` to swap models:

```json
{
  "kilocodeModel": "x-ai/grok-code-fast-1"  // Swap to local model later
}
```

**Swappable to**:
- Local 1B/3B/7B models (strict instruction following)
- Any model that respects template constraints

---

## Why Small Models Work Here

✅ **Constrained**: Ants only fill templates, not reason
✅ **Validated**: Schema-valid or rejected
✅ **Deterministic**: Same input → same output
✅ **Auditable**: Every action logged to ledger
✅ **Cheap**: Minimal token cost for mechanical work

---

## Error Handling

```json
{
  "status": "error",
  "task_id": "ant-copy-files-1",
  "error": "Source file not found",
  "action_taken": "No files modified (atomic)",
  "escalate_to": "Governor",
  "recommendation": "Check source path"
}
```

---

## Related Skills

- **governor/**: Dispatches strict JobSpecs to workers
- **file-analyzer/**: Analyzes file structure before operations

---

**Status**: Ready
**Workers**: 2 (configurable)
**Integration**: MCP, CLI, templates
