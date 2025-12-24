# Governor Skill

**Purpose**: Dispatch strict JobSpecs to Ant Workers
**Role**: The Conductor - analyzes, breaks down, and distributes tasks
**CLI**: Configurable (default: Gemini CLI)

---

## Architecture

```
Higher Authority (Claude/User)
    ↓
Governor (analyzes + distributes)
    ├─ Breaks task into subtasks
    ├─ Creates ant-task.json for each Ant
    ├─ Dispatches to Ant Workers
    └─ Aggregates results
        ↓
Ant Workers (execute templates)
```

---

## Usage

```bash
python run.py input.json output.json
```

## Input Schema

```json
{
  "goal": "Import swarm files to CATALYTIC-DPT",
  "task_id": "governor-import-swarm",
  "break_into_subtasks": true,
  "max_workers": 2,
  "timeout_seconds": 120
}
```

---

## Governor Responsibilities

1. **Analyze**: Understand the high-level goal
2. **Decompose**: Break into ant-sized subtasks
3. **Template**: Create strict ant-task.json for each worker
4. **Dispatch**: Send to Ant Workers via MCP/CLI
5. **Monitor**: Track worker progress
6. **Aggregate**: Collect and merge results
7. **Report**: Send summary to higher authority

---

## Governance Rules

1. **Single Authority**: Governor is the only one that commits
2. **Strict Templates**: Ants receive templates, not freeform prompts
3. **Deterministic**: Same input → same subtask breakdown
4. **Logged**: Every dispatch logged to ledger
5. **Escalation**: Unknown situations → escalate, don't guess

---

## Example Workflow

**Input from Claude:**
```json
{
  "goal": "Copy swarm-governor files to CATALYTIC-DPT",
  "task_id": "import-swarm-20251224"
}
```

**Governor creates ant tasks:**
```json
[
  {
    "ant_id": "ant-1",
    "task_id": "ant-copy-run-py",
    "task_type": "file_operation",
    "operation": "copy",
    "files": [{"source": "...", "destination": "..."}]
  },
  {
    "ant_id": "ant-2",
    "task_id": "ant-copy-validate-py",
    "task_type": "file_operation",
    "operation": "copy",
    "files": [{"source": "...", "destination": "..."}]
  }
]
```

**Governor aggregates results:**
```json
{
  "status": "success",
  "task_id": "import-swarm-20251224",
  "workers_used": 2,
  "files_copied": 3,
  "ledger": "CONTRACTS/_runs/import-swarm-20251224/"
}
```

---

## CLI Configuration

The Governor can use different CLIs. Default is Gemini:

```bash
gemini --experimental-acp
```

Swappable to other frontier models or local models when needed.

---

## Related Skills

- **ant-worker/**: Executes templates dispatched by Governor
- **file-analyzer/**: Provides file analysis before dispatch

---

**Status**: Ready
**CLI**: Gemini (configurable)
**Workers**: Dispatches to 2 Ant Workers
