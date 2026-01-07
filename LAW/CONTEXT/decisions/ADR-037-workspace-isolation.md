---
id: "ADR-037"
title: "Workspace Isolation for Parallel Agent Work"
status: "Accepted"
date: "2026-01-07"
confidence: "High"
impact: "High"
tags: ["agents", "git", "worktrees", "parallel-execution", "governance"]
---

<!-- CONTENT_HASH: pending -->

# ADR-037: Workspace Isolation for Parallel Agent Work

## Context

As the AGS system scales to support multiple concurrent agents working on different tasks, we encountered several critical problems:

1. **Workspace conflicts**: Multiple agents modifying the same files simultaneously causes merge conflicts and lost work
2. **Dirty state interference**: One agent's uncommitted changes can fail another agent's governance checks (preflight, tests)
3. **Branch collision**: Agents working on `main` branch directly can create conflicting commit histories
4. **Evidence destruction**: Failed tasks had their worktrees cleaned up, losing debugging evidence
5. **Manual cleanup burden**: Stale worktrees and branches accumulated without automated cleanup

The previous approach was ad-hoc: agents were expected to "use worktrees" but without standardized tooling, naming conventions, or lifecycle management. This led to inconsistent behavior and leftover artifacts.

## Decision

Implement a **workspace-isolation skill** that provides mechanical, standardized git worktree/branch management for all parallel agent work.

### Core Mechanics

1. **Standard Naming Convention**
   - Branch: `task/<task_id>` (e.g., `task/2.4.1C.5`)
   - Worktree: `../wt-<task_id>` (e.g., `../wt-2.4.1C.5`)

2. **Lifecycle Commands**
   - `create <task_id>` — Create worktree + branch
   - `status [task_id]` — Show worktree status
   - `merge <task_id>` — Merge to main (only after validation)
   - `cleanup <task_id>` — Remove worktree + delete branch
   - `cleanup-stale` — Find and remove merged worktrees

3. **Hard Invariants**
   - Never work in detached HEAD
   - Never merge until validation passes
   - Never auto-delete on failure (preserve evidence)
   - Always cleanup after successful merge
   - One task = one worktree

4. **Implementation**
   - Python-based (cross-platform, replaces Windows-only PowerShell)
   - JSON output for machine parsing
   - Exit codes: 0 = success, 1 = expected error, 2 = unexpected error
   - Full skill structure: SKILL.md, run.py, validate.py, tests

## Alternatives Considered

### A. Manual worktree management
- **Rejected**: Inconsistent naming, forgotten cleanup, no status tracking
- Agents would create worktrees with arbitrary names, making cleanup impossible

### B. Branch-only isolation (no worktrees)
- **Rejected**: Doesn't solve workspace conflicts
- Two agents on different branches still share the same working directory

### C. Separate repository clones
- **Rejected**: Too heavyweight, doesn't scale
- Each agent would need a full clone, wasting disk space and setup time

### D. Container-based isolation
- **Rejected**: Overkill for this use case
- Adds Docker dependency and complexity for a git-native problem

## Rationale

Git worktrees are the right primitive because they:
1. **Are git-native**: No external dependencies, works everywhere git works
2. **Share object store**: Multiple worktrees share `.git/objects`, saving disk space
3. **Support parallel checkouts**: Each worktree has independent working directory
4. **Enable branch isolation**: Each worktree is on its own branch

The skill-based approach provides:
1. **Standardization**: All agents use the same naming and lifecycle
2. **Automation**: Cleanup of stale worktrees is mechanical
3. **Governance integration**: Merge only after validation passes
4. **Evidence preservation**: Failed tasks keep their worktrees for debugging

## Consequences

### Positive
- Multiple agents can work simultaneously without conflicts
- Clear ownership: task ID → branch → worktree
- Automated cleanup prevents artifact accumulation
- Evidence preserved when tasks fail
- Cross-platform (Python replaces PowerShell)

### Negative
- Agents must learn new skill commands
- Worktrees consume disk space until cleaned up
- Merge conflicts still possible if tasks overlap

### Neutral
- Requires explicit cleanup step after merge
- Stale worktree cleanup should be run periodically

## Enforcement

1. **AGENTS.md Section 1C** updated with workspace-isolation skill usage
2. **Skill location**: `CAPABILITY/SKILLS/agents/workspace-isolation/`
3. **Test coverage**: 10 passing tests in `scripts/test_workspace_isolation.py`
4. **Validation**: `validate.py` verifies skill is functional

### Canon Rules
- Agents performing parallel work MUST use the workspace-isolation skill
- Agents MUST cleanup worktrees after successful merge
- Agents MUST NOT delete worktrees on task failure (preserve evidence)

### Fixtures
- Basic naming convention tests
- CLI help and argument parsing tests
- Status command JSON output tests
- Error handling tests
- Cleanup-stale dry-run tests

## Review Triggers

- If git worktree behavior changes significantly
- If agents need more complex isolation (e.g., nested worktrees)
- If cleanup automation proves insufficient
- If disk space becomes a concern with many worktrees
