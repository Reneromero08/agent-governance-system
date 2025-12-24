# Gemini File Analyzer Skill

**Purpose**: Use Gemini CLI to analyze AGI repo and identify critical files for swarm integration

**Status**: Phase 1 CATLAB helper

---

## Usage

```bash
python run.py input.json output.json
```

## Input Schema

```json
{
  "repo_path": "D:/CCC 2.0/AI/AGI",
  "task_type": "analyze_swarm",
  "focus_areas": [
    "SKILLS/swarm-governor",
    "SKILLS/launch-terminal",
    "EXTENSIONS/antigravity-bridge"
  ],
  "output_format": "json"
}
```

## Task Types

- `analyze_swarm`: Understand swarm-governor architecture
- `find_gemini_config`: Locate Gemini CLI configuration
- `identify_dependencies`: Find all skill dependencies
- `list_critical_files`: Which files are essential to understand the system

## Output Schema

```json
{
  "status": "success",
  "analysis": {
    "task_type": "analyze_swarm",
    "findings": [
      {
        "file": "SKILLS/swarm-governor/run.py",
        "importance": "critical",
        "description": "Core swarm orchestration logic"
      }
    ],
    "recommendations": [
      "Port swarm-governor to CATALYTIC-DPT",
      "Update to use Gemini CLI instead of Cline"
    ]
  }
}
```

---

## How It Works

1. User provides `task_type` (analyze_swarm, find_gemini_config, etc.)
2. Skill calls Gemini CLI with task description
3. Gemini analyzes AGI repo
4. Output lists critical files with descriptions
5. Claude uses output to understand integration points

---

## Governance

- Uses Gemini CLI (not Cline)
- Reads-only from AGI repo
- Output guides CATALYTIC-DPT integration
- Part of Phase 1 autonomous task execution
