# Swarm Architecture

> **Document Hash**: `SHA256:SWARM_ARCH_V1`
> **Canonical Location**: `CATALYTIC-DPT/LAB/ARCHITECTURE/SWARM_ARCHITECTURE.md`
> **Supersedes**: `ORCHESTRATION.md`, `RECURSIVE_SWARM.md`

---

## 1. Core Principle

**Big Brains is President. President calls the Governor. Governor appears in VSCode terminal. Governor calls Workers recursively. MCP is the single source of truth.**

---

## 2. System Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                       GOD (User)                            │
│                  - The Source of Intent                     │
│                  - Final Authority                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  PRESIDENT (Brains)                         │
│                 - Lives in Agent Chat                │
│                 - High-level decisions, governance, tokens  │
│                 - Delegates to Governor                     │
│                 - Monitors via MCP ledger                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST to port 4000
                       │ (Antigravity Bridge)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   GOVERNOR (BIG CLI)                     │
│                 - Spawns IN VSCode terminal                 │
│                 - Analyzes tasks from President             │
│                 - Distributes subtasks to Ants              │
│                 - Calls launch-terminal (recursive pattern) │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST to port 4000 (recursive)
          ┌────────────┼────────────┐
          ↓            ↓            ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   ANT    │ │   ANT    │ │   ANT    │
    │ Worker 1 │ │ Worker 2 │ │ Worker N │
    │ (Local)  │ │ (Local)  │ │ (Local)  │
    └──────────┘ └──────────┘ └──────────┘
          │            │            │
          └────────────┼────────────┘
                       │
                       ↓ MCP Ledger Files
         ┌──────────────────────────────┐
         │  directives.jsonl            │
         │  task_queue.jsonl            │
         │  task_results.jsonl          │
         │  escalations.jsonl           │
         └──────────────────────────────┘
```

---

## 3. Agent Roles

### GOD (The User)
Provides intent and final judgment. The human in the loop.

### PRESIDENT (Orchestrator)
- **Model**: Brains (Main Agent)
- **Config**: `swarm_config.json → roles.president`
- **Responsibilities**:
  - Receives directives from God
  - Formulates strategy
  - Delegates execution blocks to Governor
  - Monitors Governor via MCP terminal tools
  - **Does NOT micromanage execution**

### GOVERNOR (Manager)
- **Model**: BIG CLI
- **Config**: `swarm_config.json → roles.governor`
- **In**: VSCode terminal (user visible)
- **Responsibilities**:
  - Receives directives from President (`get_directives`)
  - Breaks into mechanical steps
  - Dispatches tasks to Ants (`dispatch_task`)
  - Monitors results (`get_results`)
  - Reports back to President

### ANT WORKERS (Executors)
- **Model**: Local Model
- **Config**: `swarm_config.json → roles.ant_worker`
- **In**: VSCode terminals (user visible)
- **Responsibilities**:
  - Poll for tasks (`get_pending_tasks`)
  - Execute strict templates
  - Report pass/fail (`report_result`)
  - Stateless execution units

---

## 4. Communication Flow

All agents communicate via **MCP ledger files**, not direct subprocess calls:

```
President → send_directive() → directives.jsonl → Governor reads
Governor → dispatch_task() → task_queue.jsonl → Ants read  
Ants → report_result() → task_results.jsonl → Governor reads
Governor → escalate() → escalations.jsonl → President reads
```

---

## 5. MCP as Single Source of Truth

**Problem**: Multiple agents editing files → drift and conflicts
**Solution**: MCP server mediates ALL state changes

```
Agent 1 (President)   Agent 2 (Governor)    Agent 3 (Ants)
     │                    │                      │
     └────────────────────┼──────────────────────┘
                          │
                          ↓
                   ┌─────────────┐
                   │ MCP Server  │
                   │ - File lock │
                   │ - State     │
                   │ - Ledger    │
                   │ - Terminal  │
                   └─────────────┘
```

**Rules**:
1. No agent directly modifies files (ideally via MCP tools)
2. All changes via MCP tools
3. MCP logs every change
4. Conflicts resolved by MCP (last-write-wins or merge)
5. Terminal access shared (President sees Governor's output)

---

## 6. Key Implementation Files

| Component | Location |
|-----------|----------|
| **Antigravity Bridge** | `D:\CCC 2.0\AI\AGI\EXTENSIONS\antigravity-bridge\` | (For Antigravity only)
| **Launch Terminal Skill** | `CATALYTIC-DPT/SKILLS/launch-terminal/` |  (Needs Vscode version)
| **MCP Server** | `CATALYTIC-DPT/MCP/server.py` |
| **Swarm Config** | `CATALYTIC-DPT/swarm_config.json` |

### Antigravity Bridge
- **Port**: 4000 (localhost)
- **Endpoint**: `POST /terminal`
- **Payload**: `{"name": "Terminal Title", "cwd": "/path", "initialCommand": "..."}`
- **Result**: Creates terminal **inside VSCode panel**

---

## 7. Drift Prevention Strategies

### Strategy 1: Canonical Skill Definition
```
CATALYTIC-DPT/SKILLS/<skill-name>/
├── SKILL.md              ← Canonical definition
├── run.py                ← Implementation
└── fixtures/             ← Test cases
```
**Rule**: If skill changes, all agents reload from repo.

### Strategy 2: Immutable Ledger
```
CONTRACTS/_runs/<execution_id>/
├── RUN_INFO.json         ← What was requested
├── PRE_MANIFEST.json     ← State before
├── POST_MANIFEST.json    ← State after
├── RESTORE_DIFF.json     ← Changes (should be empty)
└── STATUS.json           ← Final status
```

### Strategy 3: Hash Verification
Every file operation includes hash verification. If hashes don't match → HARD FAIL.

---

## 8. User Control

Because all terminals spawn inside VSCode:
1. **User can see** all agent terminals in the panel
2. **User can interject** by typing in any terminal
3. **User can kill** any agent by closing the terminal
4. **President (Brains)** monitors via MCP ledger, not subprocess

---

## 9. PROHIBITED

> **INV-012**: Visible Execution (External window spawning is PROHIBITED)
> 
> ❌ `Start-Process wt` (Windows Terminal)
> ❌ `subprocess.Popen` with external windows
> ❌ Any process Brains cannot see
>
> ✅ Only use Antigravity Bridge on port 4000 or VSCode terminal. **The Bridge is Invariant Infrastructure (Always On).**

---

## 10. Quick Start

**From Brains (President)**:
```python
import requests
requests.post("http://127.0.0.1:4000/terminal", json={
    "name": "Governor",
    "cwd": "d:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT",
    "initialCommand": "gemini --prompt 'You are the Governor...'"
})
```

**From Governor (Gemini)** - same pattern, it's recursive:
```python
import requests
requests.post("http://127.0.0.1:4000/terminal", json={
    "name": "Ant-1",
    "cwd": "...",
    "initialCommand": "npx @kilocode/cli '...'"
})
```

---

## 11. Why This Works

✅ **Governor analyzes** (good at understanding complexity)
✅ **Ants execute** (fast, cheap, reliable)
✅ **President orchestrates** (big brain makes governance decisions)
✅ **MCP mediates** (single source of truth, zero drift)
✅ **Terminals shared** (transparency, bidirectional monitoring)
✅ **Skills are canonical** (repo is truth)
✅ **Ledger is immutable** (every action logged)
✅ **Token efficient** (big brain delegates, small brains execute)

---

**Status**: Architecture defined, swarm operational
**Config**: See `swarm_config.json` for model assignments
