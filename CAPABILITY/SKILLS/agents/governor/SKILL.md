---
name: governor
version: "0.1.0"
description: Analyzes high-level goals, breaks them into subtasks, and dispatches to ant-workers. Use when you need task decomposition and worker coordination. Aggregates results and reports to higher authority.
compatibility: Python 3.8+, Gemini CLI optional
---

# Governor

The Conductor - analyzes, decomposes, and dispatches tasks to Ant Workers.

## Usage

```bash
python scripts/run.py input.json output.json
```

## Input Schema

```json
{
  "gemini_prompt": "Analyze D:/path/to/files and summarize",
  "task_id": "analyze-001",
  "command_type": "analyze",
  "timeout_seconds": 60
}
```

## Command Types

| Type | Purpose |
|------|---------|
| `analyze` | Read/analyze files, return findings |
| `execute` | Run a command, return results |
| `research` | Deep research on topic |
| `report` | Generate comprehensive report |

## Workflow

1. Receive directive from Claude
2. Analyze and decompose into subtasks
3. Dispatch to Ant Workers via MCP
4. Monitor progress
5. Aggregate results
6. Report to higher authority
