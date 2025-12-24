# Recursive Swarm Architecture

> **CANONICAL DOCUMENTATION** - Do not modify without President approval

## Core Principle

**Claude is President. Claude calls the Governor. Governor appears in VSCode terminal. Governor calls Workers the same way. It's recursive.**

## Key Files

| Component | Location |
|-----------|----------|
| **Antigravity Bridge Extension** | `D:\CCC 2.0\AI\AGI\EXTENSIONS\antigravity-bridge\` |
| **Launch Terminal Skill** | `CATALYTIC-DPT\SKILLS\launch-terminal\` |
| **Extension Docs** | `CATALYTIC-DPT\SKILLS\launch-terminal\ANTIGRAVITY_BRIDGE.md` |
| **MCP Server** | `CATALYTIC-DPT\MCP\server.py` |

```
┌─────────────────────────────────────────────────────────────┐
│                       President (Claude)                    │
│                  - Lives in Antigravity chat                │
│                  - Calls launch-terminal skill              │
│                  - Monitors via MCP ledger                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST to port 4000
                       │ (Antigravity Bridge)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   Governor (Gemini CLI)                     │
│                  - Spawns IN VSCode terminal                │
│                  - User can see + interject                 │
│                  - Calls launch-terminal for Ants           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST to port 4000 (recursive)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   Ant Workers (Kilo Code)                   │
│                  - Spawns IN VSCode terminals               │
│                  - User can see + interject                 │
│                  - Reports back via MCP ledger              │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Antigravity Bridge Extension
- **Location**: `EXTENSIONS/antigravity-bridge/`
- **Port**: 4000 (localhost)
- **Endpoint**: `POST /terminal`
- **Payload**: `{"name": "Terminal Title", "cwd": "/path", "initialCommand": "gemini --prompt '...'"}`
- **Result**: Creates terminal **inside VSCode panel**, not external window

### 2. Launch Terminal Skill  
- **Location**: `SKILLS/launch-terminal/run.py`
- **Usage**: `python run.py input.json output.json`
- **Calls**: `http://127.0.0.1:4000/terminal`

### 3. Recursive Pattern
```
Claude (President):
  → Calls launch-terminal skill
  → Creates "Governor" terminal in VSCode
  → Gemini CLI starts with system prompt
  
Governor (Gemini):
  → Receives directive via MCP (get_directives)
  → Breaks into subtasks
  → Calls launch-terminal skill (same pattern!)
  → Creates "Ant-1", "Ant-2" terminals in VSCode
  → Monitors via MCP (get_results)
  
Ants:
  → Poll for tasks via MCP (get_pending_tasks)
  → Execute templates
  → Report via MCP (report_result)
```

---

## Communication Flow

All agents communicate via **MCP ledger files**, not direct subprocess calls:

```
President → send_directive() → directives.jsonl → Governor reads
Governor → dispatch_task() → task_queue.jsonl → Ants read  
Ants → report_result() → task_results.jsonl → Governor reads
Governor → escalate() → escalations.jsonl → President reads
```

---

## User Control

Because all terminals spawn inside VSCode:
1. **User can see** all agent terminals in the panel
2. **User can interject** by typing in any terminal
3. **User can kill** any agent by closing the terminal
4. **President (Claude)** monitors via MCP ledger, not subprocess

---

## PROHIBITED

> **INV-014**: External window spawning is PROHIBITED
> 
> ❌ `Start-Process wt` (Windows Terminal)
> ❌ `subprocess.Popen` with external windows
> ❌ Any process Claude cannot see
>
> ✅ Only use Antigravity Bridge on port 4000

---

## Quick Start

**From Claude (President)**:
```python
import requests
requests.post("http://127.0.0.1:4000/terminal", json={
    "name": "Governor",
    "cwd": "d:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT",
    "initialCommand": "gemini --prompt 'You are the Governor...'"
})
```

**From Governor (Gemini)**:
```python
# Same pattern - it's recursive
import requests
requests.post("http://127.0.0.1:4000/terminal", json={
    "name": "Ant-1",
    "cwd": "...",
    "initialCommand": "npx @kilocode/cli '...'"
})
```
