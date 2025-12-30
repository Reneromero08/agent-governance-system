# Skill: governor
**Version:** 0.1.0
**Status:** Active
**Required_canon_version:** ">=3.0.0 <4.0.0"

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
