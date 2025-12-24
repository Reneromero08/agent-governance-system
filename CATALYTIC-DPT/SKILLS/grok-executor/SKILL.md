# Grok Executor Skill

**Purpose**: Distributed task executor for CATALYTIC-DPT
**Model**: Grok 1.5 Fast (free, fast, reliable)
**Role**: Executes mechanical tasks under Conductor supervision
**Integration**: Via Kilo Code, MCP, and Antigravity Bridge

---

## Architecture

```
Google Conductor (Manager)
    ├─ Distributes subtask to Grok-1
    ├─ Distributes subtask to Grok-2
    ├─ Distributes subtask to Grok-3
    └─ Monitors all workers

Grok Worker 1: File operations (copy, move, analyze)
Grok Worker 2: Code adaptation (update imports, refactor)
Grok Worker 3: Testing & validation

All workers:
- Report to Conductor
- Write to MCP-controlled ledger
- Share terminal output (bidirectional)
- Can escalate to Claude for decisions
```

---

## Usage

```bash
python run.py input.json output.json
```

## Input Schema

```json
{
  "task_id": "grok-copy-swarm-governor-1",
  "task_type": "file_operation",
  "operation": "copy",
  "source": "D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py",
  "destination": "D:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py",
  "verify_integrity": true,
  "mcp_terminal_id": "grok_worker_1"
}
```

## Task Types

- `file_operation`: Copy, move, delete, read files
- `code_adapt`: Update imports, refactor code
- `validate`: Test files, run fixtures
- `research`: Analyze and understand code

---

## Output Schema

```json
{
  "status": "success",
  "task_id": "grok-copy-swarm-governor-1",
  "task_type": "file_operation",
  "operation": "copy",
  "result": {
    "source": "D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py",
    "destination": "D:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py",
    "source_hash": "abc123def456...",
    "dest_hash": "abc123def456...",
    "hash_verified": true,
    "size_bytes": 12345
  },
  "mcp_ledger": "CONTRACTS/_runs/grok-copy-swarm-governor-1/",
  "timestamp": "2025-12-24T14:30:22Z"
}
```

---

## Task Examples

### Task 1: Copy Files from AGI Swarm

```json
{
  "task_id": "grok-import-swarm-1",
  "task_type": "file_operation",
  "operation": "copy",
  "files": [
    {
      "source": "D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py",
      "destination": "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py"
    },
    {
      "source": "D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/validate.py",
      "destination": "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/validate.py"
    }
  ],
  "verify_integrity": true
}
```

### Task 2: Adapt Code (Cline → Gemini)

```json
{
  "task_id": "grok-adapt-swarm-gemini",
  "task_type": "code_adapt",
  "file": "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py",
  "adaptations": [
    {
      "find": "cline '{safe_prompt}' {model_arg}",
      "replace": "gemini -o json '{safe_prompt}'",
      "reason": "Use Gemini CLI instead of Cline"
    },
    {
      "find": "from concurrent.futures import ThreadPoolExecutor",
      "replace": "from concurrent.futures import ThreadPoolExecutor  # Conductor manages this",
      "reason": "Conductor handles parallelism, not swarm directly"
    }
  ]
}
```

### Task 3: Validate Integration

```json
{
  "task_id": "grok-validate-swarm-adapted",
  "task_type": "validate",
  "file": "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py",
  "checks": [
    "imports are valid (no missing modules)",
    "run() function signature is correct",
    "output matches expected schema",
    "no references to 'cline' (all 'gemini')"
  ],
  "fixtures": "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/fixtures/"
}
```

---

## Governance Rules

1. **MCP-Mediated Execution**: All file operations via MCP (no direct writes)
2. **Hash Verification**: Every copy verified with SHA-256
3. **Immutable Ledger**: Every action logged to CONTRACTS/_runs/
4. **Terminal Sharing**: All output visible to Conductor, Claude, and You
5. **Escalation Path**: If uncertain, ask Claude via MCP

---

## Integration with Conductor

When Conductor distributes a task to Grok:

```
Conductor: "Grok-1, copy swarm-governor/run.py to CATALYTIC-DPT"
    ↓
Grok reads task spec
    ↓
Grok calls MCP: file_sync(source, dest, verify_hash=True)
    ↓
MCP logs operation, computes hashes, copies file
    ↓
Grok reads result from MCP ledger
    ↓
Grok reports to Conductor: "✓ Success, hash verified"
    ↓
Conductor aggregates all worker results
    ↓
Conductor reports to Claude: "All files imported successfully"
```

---

## Why Grok?

✅ **Fast**: Grok 1.5 Fast is optimized for speed
✅ **Free**: No token cost to you (unlike Claude/Gemini)
✅ **Reliable**: Consistent output for mechanical tasks
✅ **Local Control**: Runs in your environment via Kilo Code
✅ **Parallelizable**: Can run multiple Grok workers simultaneously
✅ **MCP-Compatible**: Integrates cleanly with terminal/skill/file operations

---

## Integration Points

1. **Conductor**: Receives task assignments from Google Conductor
2. **MCP**: All file/skill operations via MCP server
3. **Terminal**: Output shared bidirectionally via MCP
4. **Ledger**: Results immutably logged to CONTRACTS/_runs/
5. **Claude**: Can escalate questions or request help

---

## Error Handling

If Grok encounters error:

```json
{
  "status": "error",
  "task_id": "grok-copy-swarm-governor-1",
  "error": "Source file not found",
  "path": "D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py",
  "action_taken": "No files modified (MCP prevented partial state)",
  "escalate_to": "Claude",
  "recommendation": "Verify AGI repo path is correct"
}
```

MCP ensures **atomic operations** - either all succeeds or nothing changes.

---

## Success Example

```
$ python CATALYTIC-DPT/SKILLS/grok-executor/run.py task_import_swarm.json output.json

[MCP] Registering terminal: grok_worker_1
[Grok] Executing task: grok-import-swarm-1
[MCP] file_sync: AGI/swarm-governor/run.py → CATALYTIC-DPT/swarm-governor-adapted/run.py
[MCP] Hash verification: PASS (abc123 == abc123)
[MCP] Logging to: CONTRACTS/_runs/grok-import-swarm-1/
[Grok] Task status: SUCCESS
[Grok] Output written to: output.json

Result: 3 files copied, 3 hashes verified, 0 errors
```

---

## Related Skills

- **gemini-executor**: Analyzes tasks before Conductor distributes
- **swarm-governor-adapted**: Orchestrates Grok workers in parallel
- **conductor-task-builder**: Creates task specs for distribution

---

**Status**: Ready for implementation
**Model**: Grok 1.5 Fast (via Kilo Code)
**Integration**: MCP, Antigravity Bridge, skills
