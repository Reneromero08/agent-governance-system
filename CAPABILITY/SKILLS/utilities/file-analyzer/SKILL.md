---
name: file-analyzer
description: "Analyzes repo structure and identifies critical files."
---
<!-- CONTENT_HASH: 373d9bb6438898b8652edb9871c3572fc594135267e9af172fcbd5796ac4af67 -->

**required_canon_version:** >=3.0.0


# Skill: file-analyzer
**Version:** 0.1.0
**Status:** Active

**canon_version:** "3.0.0"

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

**required_canon_version:** >=3.0.0

