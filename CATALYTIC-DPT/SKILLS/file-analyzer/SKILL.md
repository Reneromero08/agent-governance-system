---
name: file-analyzer
description: Analyzes repository structure to identify critical files for integration. Use when you need to understand a codebase before operations.
compatibility: Python 3.8+
---

# File Analyzer

Analyzes repo structure and identifies critical files.

## Usage

```bash
python scripts/run.py input.json output.json
```

## Input Schema

```json
{
  "repo_path": "D:/path/to/repo",
  "task_type": "analyze_swarm",
  "focus_areas": ["SKILLS/", "MCP/"],
  "output_format": "json"
}
```

## Task Types

| Type | Purpose |
|------|---------|
| `analyze_swarm` | Understand swarm architecture |
| `find_config` | Locate configuration files |
| `identify_dependencies` | Find skill dependencies |
| `list_critical_files` | List essential files |
