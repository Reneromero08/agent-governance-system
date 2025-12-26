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
    │ Worker 1 │ │ Worker 2 │ │ Worker N │
    │ (Exec)   │ │ (Exec)   │ │ (Exec)   │
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

```
Agent 1 (President)   Agent 2 (Governor)    Agent 3 (Ants)
     │                    │                      │
     └────────────────────┼──────────────────────┘
                          │
                          ↓
                   ┌─────────────┐
                   │ MCP Server  │
                   │ (Mediator)  │
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

## Agent Roles (Hierarchy)

> **Configuration**: See [`swarm_config.json`](./swarm_config.json) for current model assignments.

### 1. GOD (The User)
**Role**: Provides the intent and final judgment. The "human in the loop" who oversees the swarm.

### 2. PRESIDENT (Orchestrator)
**Implementation**: Defined in `swarm_config.json → roles.president`
**Role**:
- Receives high-level directives from God.
- Formulates strategy.
- Delegates execution blocks to the Governor.
- Monitors the Governor via MCP terminal tools.
- **Does NOT**: Micromanage execution.

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

## Preventing Drift: Single Source of Truth

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

## Why This Works

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
