# Agent Activity Monitor

**Skill:** `agent-activity`
**Version:** 0.1.0
**Status:** Active
**Required_canon_version:** >=2.11.15

**Description:** Monitors active agents and their current tasks by analyzing the MCP audit log.
**Usage:** `skill_run(skill="agent-activity", input={})`

## Input Schema
```json
{
  "limit": 10,           // Number of recent sessions to show
  "active_within": 600   // Only show agents active within last N seconds (default: 600)
}
```

## Output Schema
```json
{
  "active_agents": [
    {
      "session_id": "uuid...",
      "last_seen": "iso-timestamp",
      "tool": "tool_name",
      "file": "file_path_if_any",
      "status": "success/error"
    }
  ]
}
```
