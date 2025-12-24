# Gemini Executor Skill

**Purpose**: Enable Claude to delegate tasks to Gemini CLI, which runs in YOUR VSCode terminal (not Claude's)

**Status**: Phase 1 CATLAB integration

---

## Architecture

```
Claude (87% budget)
     ↓
gemini-executor skill (routes to Gemini)
     ↓
Gemini CLI (gemini --prompt)
     ↓
Antigravity Bridge (port 4000)
     ↓
YOUR VSCode Terminal
     ↓
Executes commands in YOUR workspace
```

---

## Usage

```bash
python run.py input.json output.json
```

## Input Schema

```json
{
  "gemini_prompt": "Analyze D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/ and list all .py files",
  "task_id": "analyze-swarm-1",
  "timeout_seconds": 60,
  "output_format": "json",
  "command_type": "analyze"
}
```

## Task Types

- `analyze`: Read and analyze files in your AGI repo
- `execute`: Run a command in your VSCode terminal
- `research`: Deep research on a topic using Gemini
- `report`: Generate a report from repo data

## Output Schema

```json
{
  "status": "success",
  "task_id": "analyze-swarm-1",
  "command": "gemini --prompt 'Analyze D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/'",
  "gemini_response": "...",
  "executed_in": "YOUR_VSCode_Terminal",
  "timestamp": "2025-12-24T14:30:22Z"
}
```

---

## Key Capability: YOUR Terminal, Not Mine

Claude cannot directly control your VSCode terminal. But:

1. Claude sends task to Gemini CLI via `gemini-executor`
2. Gemini CLI runs in your environment (YOUR VSCode terminal)
3. Gemini reads/analyzes/executes in YOUR workspace
4. Results come back to Claude

This preserves:
- **Your control**: Gemini runs YOUR terminal, not mine
- **Token efficiency**: Claude delegates, Gemini executes
- **File access**: Gemini can read YOUR AGI repo, not via Claude

---

## Example: Swarm File Discovery

**Claude wants to understand swarm-governor:**

```json
{
  "gemini_prompt": "In D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/, list all Python files and their purposes",
  "task_id": "discover-swarm-files",
  "command_type": "analyze"
}
```

**Gemini (in your terminal):**
```bash
$ gemini "In D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/, list all Python files and their purposes"

Response: [Lists run.py, validate.py, etc. with descriptions]
```

**Result back to Claude:**
```json
{
  "status": "success",
  "gemini_response": "run.py - Main swarm orchestrator\nvalidate.py - Task output validator\n..."
}
```

---

## Governance

- Gemini runs in YOUR environment (not Claude's sandbox)
- You approve/review Gemini responses before Claude acts
- All tasks logged for audit
- Integrates with swarm-governor for batch analysis

---

## Integration with Swarm

Can be used inside swarm tasks:

```json
{
  "tasks": [
    {
      "id": "gemini-analyze-phase0",
      "prompt": "Use gemini-executor to analyze which files from AGI repo we need for catalytic swarm integration",
      "model": "gemini"
    },
    {
      "id": "copy-swarm-governor",
      "prompt": "Based on gemini's analysis, copy necessary swarm files to CATALYTIC-DPT",
      "model": "neural-chat"
    }
  ],
  "num_workers": 2
}
```

---

## Why This Matters

**Before**: Claude can only read via file tools (limited context)
**After**: Gemini can analyze entire AGI repo in YOUR terminal and report back

This unblocks swarm integration without burning Claude's token budget.
