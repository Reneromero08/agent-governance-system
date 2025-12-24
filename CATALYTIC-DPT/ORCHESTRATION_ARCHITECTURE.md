# Multi-Agent Orchestration Architecture

**Date**: 2025-12-24
**Vision**: Model-Agnostic Swarm Hierarchy
**Governance**: Single source of truth via MCP, zero drift, bidirectional terminal monitoring

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       GOD (User)                            │
│                  - The Source of Intent                     │
│                  - Final Authority                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  PRESIDENT (Main Chat)                      │
│                 - Orchestrator (e.g., Claude)               │
│                 - High-level decision making                │
│                 - Governance logic & Token strategy         │
│                 - Delegates to Governor                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ MCP Protocol
                       │ (shared state, zero drift)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   GOVERNOR (CLI Agent)                      │
│                 - Manager (e.g., Gemini)                    │
│                 - Resides in terminal loop                  │
│                 - Analyzes tasks from President             │
│                 - Distributes subtasks to Ants              │
│                 - Monitors progress                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ↓            ↓            ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   ANT    │ │   ANT    │ │   ANT    │
    │ Worker 1 │ │ Worker 2 │ │ Worker 3 │
    │          │ │          │ │          │
    │ - Coding │ │- File Ops│ │- Testing │
    │ (Models: │ │ (Models: │ │ (Models: │
    │  Grok,   │ │  Haiku,  │ │  Llama)  │
    │  Small)  │ │  Small)  │ │          │
    └──────────┘ └──────────┘ └──────────┘
          │            │            │
          └────────────┼────────────┘
                       │
                       ↓ MCP Tools/Skills
        ┌──────────────────────────────┐
        │  Terminal Sharing (President ←→ Governor)
        │  File Sync (VSCode Bridge)
        │  Task Ledger (CONTRACTS/_runs)
        │  Skill Execution (CATALYTIC-DPT)
        └──────────────────────────────┘
```

---

## Core Principle: MCP as Single Source of Truth

**Problem**: Multiple agents editing files → drift and conflicts
**Solution**: MCP server mediates ALL state changes

### MCP Protocol in This System

```
Agent 1 (President)   Agent 2 (Governor)    Agent 3 (Ants)
     │                    │                      │
     └────────────────────┼──────────────────────┘
                          │
                          ↓
                   ┌─────────────┐
                   │ MCP Server  │
                   │ (Mediator)  │
                   │             │
                   │ - File lock│
                   │ - State    │
                   │ - Ledger   │
                   │ - Terminal │
                   └─────────────┘
```

**Rules**:
1. No agent directly modifies files (ideally via MCP tools)
2. All changes via MCP tools
3. MCP logs every change
4. Conflicts resolved by MCP (last-write-wins or merge)
5. Terminal access shared (President sees Governor's output)

---

## Agent Roles (Hierarchy)

> **Configuration**: See [`swarm_config.json`](file:///d:/CCC%202.0/AI/agent-governance-system/CATALYTIC-DPT/swarm_config.json) for current model assignments.

### 1. GOD (The User)
**Role**: Provides the intent and final judgment. The "human in the loop" who oversees the swarm.

### 2. PRESIDENT (Orchestrator)
**Implementation**: Defined in `swarm_config.json → roles.president`
**Role**:
- Receives high-level directives from God.
- Formulates strategy.
- Delegates execution blocks to the Governor.
- Monitors the Governor via MCP terminal tools.
- **Does NOT**: Microsystem management.

### 3. GOVERNOR (Manager)
**Implementation**: Defined in `swarm_config.json → roles.governor`
**Role**:
- Resides in the terminal.
- Receives directives from the President.
- Breaks directives into mechanical execution steps.
- Dispatches tasks to Ant Workers.
- Aggregates results and reports back to the President.

### 4. ANT WORKERS (Executors)
**Implementation**: Defined in `swarm_config.json → roles.ant_worker`
**Role**:
- Stateless execution units.
- Receive strict templates (inputs/outputs).
- Execute: file operations, code changes, tests.
- Report pass/fail signals to Governor.

---

## Workflow: Bring Swarm Files

### Step 1: You Ask Claude

```
You: "Gemini, bring the swarm-governor files to CATALYTIC-DPT"
```

### Step 2: Claude Creates Task for Conductor

```json
{
  "task_id": "import-swarm-governor-20251224",
  "goal": "Port swarm-governor from AGI to CATALYTIC-DPT",
  "subtasks": [
    "Analyze swarm-governor architecture",
    "Identify essential vs AGI-specific files",
    "Copy essential files",
    "Adapt for CATALYTIC-DPT",
    "Validate integrity"
  ],
  "authority": "Claude",
  "ledger_path": "CONTRACTS/_runs/import-swarm-governor-20251224/"
}
```

### Step 3: Google Conductor (Gemini) Manages

```
Conductor receives task
  ├─ Analyzes swarm-governor
  ├─ Creates 3 Grok subtasks
  │  ├─ Grok-1: "Copy run.py, validate.py, __init__.py"
  │  ├─ Grok-2: "Adapt to use Gemini CLI instead of Cline"
  │  └─ Grok-3: "Test integration with CATALYTIC-DPT skills"
  ├─ Monitors progress
  └─ Waits for completion
```

### Step 4: Grok Workers Execute

**Grok Worker 1**:
```bash
cp D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py \
   D:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/swarm-governor-adapted/
```

**Grok Worker 2**:
```bash
# Adapt: Change "cline" calls to "gemini" calls
# Update: swarm-governor to use Gemini CLI
```

**Grok Worker 3**:
```bash
# Test: Verify adapted swarm-governor works with CATALYTIC-DPT
```

### Step 5: Results Back to Claude

```json
{
  "status": "success",
  "task_id": "import-swarm-governor-20251224",
  "files_copied": [
    "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py",
    "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/validate.py",
    "..."
  ],
  "ledger": "CONTRACTS/_runs/import-swarm-governor-20251224/",
  "ready_for_integration": true
}
```

---

## MCP Terminal Server (God Mode)

**Feature**: You monitor Claude's terminals + Claude monitors yours = shared visibility

### Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  YOUR VSCode     │         │  Claude's MCP    │
│  Terminal        │◄───────►│  Terminal        │
│                  │   MCP   │  (logs/output)   │
└──────────────────┘         └──────────────────┘

Both can:
- View terminal output
- See commands executed
- Monitor Gemini/Grok progress
- Intervene if needed
```

### MCP Tool: `terminal_share`

```python
# MCP Tool: Share terminal output bidirectionally

@mcp.tool()
def share_terminal_output(
    terminal_id: str,    # "claude_main" or "user_vscode"
    command: str = None, # Command executed
    output: str = None,  # Output to share
    timestamp: str = None
) -> dict:
    """Share terminal activity across agents."""

    return {
        "terminal_id": terminal_id,
        "visible_to": ["Claude", "You", "Gemini (read-only)"],
        "command": command,
        "output": output,
        "ledger": f"CONTRACTS/_runs/terminal_{timestamp}.log"
    }
```

---

## MCP Tool: `skill_execution`

Prevent drift via single skill definition:

```python
@mcp.tool()
def execute_skill(
    skill_name: str,           # "swarm-governor-adapted"
    task_spec: dict,           # JobSpec JSON
    executor: str = None       # "Grok-1", "Gemini", etc.
) -> dict:
    """Execute skill via MCP (single source of truth)."""

    # 1. Load skill definition from repo (canonical)
    skill_def = load_skill(f"CATALYTIC-DPT/SKILLS/{skill_name}/SKILL.md")

    # 2. Validate task against skill contract
    if not validate_against_schema(task_spec, skill_def.schema):
        raise ValidationError("Task doesn't match skill contract")

    # 3. Execute (log every step)
    result = {
        "skill": skill_name,
        "executor": executor,
        "status": "success",
        "ledger": f"CONTRACTS/_runs/skill_{uuid.uuid4()}/",
        "inputs": task_spec,
        "outputs": { "files_modified": [...], "status": "..." }
    }

    # 4. All changes via MCP (no direct file writes)
    mcp.log_skill_execution(result)

    return result
```

---

## MCP Tool: `file_sync`

Synchronize files across agents:

```python
@mcp.tool()
def file_sync(
    source: str,         # "/path/to/source"
    destination: str,    # "/path/to/dest"
    executor: str,       # "Grok-1"
    verify_hash: str = None
) -> dict:
    """Copy file via MCP (tracks every change)."""

    result = {
        "source": source,
        "destination": destination,
        "executor": executor,
        "source_hash": compute_hash(source),
        "dest_hash": None,
        "status": "pending"
    }

    # 1. Copy file
    copy_file(source, destination)

    # 2. Verify hash
    result["dest_hash"] = compute_hash(destination)
    result["hash_match"] = (result["source_hash"] == result["dest_hash"])

    if not result["hash_match"]:
        raise IntegrityError("Hash mismatch after copy")

    # 3. Log to ledger (immutable)
    mcp.log_file_operation(result)

    # 4. Notify all agents (Claude, Gemini, Grok)
    mcp.notify_agents(f"File synced: {destination}")

    result["status"] = "success"
    return result
```

---

## Preventing Drift: Single Source of Truth

### Strategy 1: Canonical Skill Definition

```
CATALYTIC-DPT/SKILLS/swarm-governor-adapted/
├── SKILL.md              ← Canonical definition
├── run.py                ← Implementation
├── VERSION.json          ← Version hash
└── schema.json           ← Input/output contract
```

**Rule**: If skill changes, VERSION.json increments. All agents check version before executing.

### Strategy 2: Immutable Ledger

```
CONTRACTS/_runs/<execution_id>/
├── RUN_INFO.json         ← What was requested
├── TASK_SPEC.json        ← Task definition (snapshot)
├── INPUTS.json           ← Inputs
├── OUTPUTS.json          ← Outputs
├── FILES_MODIFIED.json   ← Every file touch
├── TERMINA_LOGS/         ← All commands run
└── RESTORATION_PROOF.json ← Pre/post hashes
```

### Strategy 3: Hash Verification

Every file operation includes hash:
```json
{
  "file": "CATALYTIC-DPT/SKILLS/swarm-governor-adapted/run.py",
  "operation": "copied",
  "source": "D:/CCC 2.0/AI/AGI/SKILLS/swarm-governor/run.py",
  "source_hash_before": "abc123...",
  "dest_hash_after": "abc123...",
  "match": true
}
```

If hashes don't match → HARD FAIL (immutable ledger prevents further changes)

---

## Terminal Monitoring: You ↔ Claude

### What You See

```
[YOUR VSCode Terminal]
$ gemini --experimental-acp
> Claude: "Import swarm-governor to CATALYTIC-DPT"
> Conductor analyzing...
> Grok-1: Copying files...
> Grok-2: Adapting for Gemini...
> Grok-3: Testing...
> Status: ✓ Complete
```

### What Claude Sees (via MCP)

```
[Claude's MCP Terminal View]
terminal_id: user_vscode
commands_executed: [
  "gemini --experimental-acp",
  "python CATALYTIC-DPT/SKILLS/...",
  "..."
]
output: "[Terminal activity log]"
agents_involved: ["Gemini", "Grok-1", "Grok-2", "Grok-3"]
```

### Both Can Intervene

```
You:     [Edit a terminal command] "Stop, Grok-2, use Kilo Code instead"
Claude:  [Via MCP] "Conductor, pause workers"
         [Grok workers pause]
You:     [Resume]
         "Continue with Kilo Code"
Claude:  [Via MCP] "Resume"
         [Grok workers continue]
```

---

## Implementation Roadmap

### Phase 1: MCP Infrastructure (Today)

- [ ] Create MCP server for terminal sharing
- [ ] Create MCP tools: `skill_execution`, `file_sync`, `terminal_share`
- [ ] Implement hash verification
- [ ] Create immutable ledger system

### Phase 2: Conductor Integration (1 day)

- [ ] Create Google Conductor task format
- [ ] Build Conductor → Grok task distribution
- [ ] Implement progress monitoring

### Phase 3: Worker Adaptation (1 day)

- [ ] Adapt Grok to use Kilo Code interface
- [ ] Create Grok task executor
- [ ] Implement result reporting

### Phase 4: Swarm Import (Today/Tomorrow)

- [ ] Use Conductor to orchestrate swarm-governor import
- [ ] Adapt swarm-governor for CATALYTIC-DPT
- [ ] Test with Phase 0 schema validation

### Phase 5: Full Integration (Next)

- [ ] Swarm + Catalytic + Gemini = autonomous research
- [ ] Phase 0 parallel schema validation
- [ ] Phase 1 parallel CATLAB testing

---

## Key Files to Create

```
CATALYTIC-DPT/
├── MCP/
│   ├── server.py              # MCP server (terminal sharing, skill execution)
│   ├── tools.py               # MCP tools (file_sync, skill_execution, etc.)
│   └── schemas.py             # MCP tool schemas
│
├── SKILLS/
│   ├── swarm-governor-adapted/
│   │   ├── SKILL.md
│   │   ├── run.py
│   │   └── VERSION.json
│   │
│   ├── grok-executor/         # New: Executes Grok tasks
│   │   ├── SKILL.md
│   │   ├── run.py
│   │   └── fixtures/
│   │
│   └── conductor-task-builder/# New: Creates task specs for Conductor
│       ├── SKILL.md
│       ├── run.py
│       └── templates/
│
└── ORCHESTRATION_ARCHITECTURE.md
```

---

## Why This Works

✅ **Gemini analyzes** (good at understanding complexity)
✅ **Grok executes** (fast, free, reliable)
✅ **Claude orchestrates** (big brain makes governance decisions)
✅ **MCP mediates** (single source of truth, zero drift)
✅ **Terminals shared** (transparency, bidirectional monitoring)
✅ **Skills are canonical** (VERSION.json prevents drift)
✅ **Ledger is immutable** (every action logged forever)
✅ **Token efficient** (big brain delegates, small brains execute)

---

**Status**: Architecture complete, ready to implement
**Next**: Build MCP server and test terminal sharing
