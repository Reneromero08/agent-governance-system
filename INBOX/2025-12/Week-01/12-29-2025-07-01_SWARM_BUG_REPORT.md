---
uuid: 00000000-0000-0000-0000-000000000000
title: SWARM_BUG_REPORT
section: research
bucket: 2025-12/Week-01
author: System
priority: Medium
created: 2025-12-29 07:01
modified: 2026-01-06 13:09
status: Active
summary: Legacy research document migrated to canon format
tags:
- research
- legacy
hashtags: []
---
<!-- CONTENT_HASH: d7036872db38068b4b234994cfa567d6e24277e12b85b1671ee20fe0b1e968e5 -->

# Swarm Orchestrator Bug Report & Debug Plan

## Executive Summary

The CATALYTIC-DPT swarm orchestrator has a well-designed governance architecture. After code review, some critical bugs have been fixed but several remain. This document tracks all identified issues and their current status.

---

## Bug Status Summary

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | CRITICAL | MCP Import Path Bug | **FIXED** |
| 2 | CRITICAL | Hard-Coded Project Path | **FIXED** |
| 3 | CRITICAL | Race Condition in Task Acknowledgment | OPEN |
| 4 | HIGH | No JSON Error Recovery in MCP Server | OPEN |
| 5 | MEDIUM | No Subprocess Cleanup on Shutdown | OPEN |
| 6 | CRITICAL | agent_loop.py Wrong Import Path | **NEW** |
| 7 | CRITICAL | poll_tasks.py Wrong Import Path | **NEW** |
| 8 | HIGH | ant-worker Directory Missing | **NEW** |
| 9 | MEDIUM | No JSON Error Recovery in poll_and_execute.py | OPEN |

---

## Fixed Issues

### 1. **FIXED: MCP Import Path Bug**

- **Location**: [poll_and_execute.py:23](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py#L23)
- **Original Issue**: Import path was `CATALYTIC_ROOT / "MCP"`
- **Current Code**: `sys.path.insert(0, str(CATALYTIC_ROOT / "LAB" / "MCP"))` - **CORRECT**
- **Status**: Fixed in codebase

### 2. **FIXED: Hard-Coded Project Path**

- **Location**: [launch_swarm.ps1:22](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/launch_swarm.ps1#L22)
- **Original Issue**: Hard-coded path `d:\CCC 2.0\AI\agent-governance-system`
- **Current Code**: `$ProjectRoot = (Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.FullName` - **CORRECT**
- **Status**: Fixed in codebase

---

## Open Issues

### 3. **CRITICAL: Race Condition in Task Acknowledgment** - OPEN

- **Location**: [LAB/MCP/server.py:1194-1261](CATALYTIC-DPT/LAB/MCP/server.py#L1194-L1261)
- **Issue**: `get_pending_tasks()` and `acknowledge_task()` are separate operations
  - Line 1194-1207: `get_pending_tasks()` reads file, returns pending tasks
  - Line 1242-1262: `acknowledge_task()` reads file again, modifies, writes back
  - **Race window**: Multiple ants can read same task before any writes acknowledgment
- **Impact**: Duplicate task execution with multiple ants
- **Fix Required**: Add atomic `acquire_next_task()` with file locking

### 4. **HIGH: No JSON Error Recovery in MCP Server** - OPEN

- **Locations** (all use bare `json.loads()` without try/except):
  - [server.py:1202](CATALYTIC-DPT/LAB/MCP/server.py#L1202): `task = json.loads(line)` in `get_pending_tasks()`
  - [server.py:1235](CATALYTIC-DPT/LAB/MCP/server.py#L1235): `r = json.loads(line)` in `get_results()`
  - [server.py:1251](CATALYTIC-DPT/LAB/MCP/server.py#L1251): `t = json.loads(line)` in `acknowledge_task()`
  - [server.py:1329](CATALYTIC-DPT/LAB/MCP/server.py#L1329): `esc = json.loads(line)` in `get_escalations()`
  - [server.py:1352](CATALYTIC-DPT/LAB/MCP/server.py#L1352): `esc = json.loads(line)` in `resolve_escalation()`
  - [server.py:1417](CATALYTIC-DPT/LAB/MCP/server.py#L1417): `d = json.loads(line)` in `get_directives()`
  - [server.py:1431](CATALYTIC-DPT/LAB/MCP/server.py#L1431): `d = json.loads(line)` in `acknowledge_directive()`
  - [server.py:591](CATALYTIC-DPT/LAB/MCP/server.py#L591): `entry = json.loads(line)` in `get_ledger()`
- **Impact**: One corrupted JSONL line crashes entire read operation
- **Fix Required**: Wrap all `json.loads()` in try/except, skip bad lines

### 5. **MEDIUM: No Subprocess Cleanup on Shutdown** - OPEN

- **Location**: [poll_and_execute.py:130-136](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py#L130-L136)
- **Issue**: Uses `subprocess.run()` which blocks - no handle to terminate on Ctrl+C
- **Impact**: If interrupted during task execution, subprocess runs to completion or orphaned
- **Fix Required**: Use `subprocess.Popen()` + track handle for cleanup

### 6. **CRITICAL: agent_loop.py Wrong Import Path** - NEW

- **Location**: [agent_loop.py:12](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/agent_loop.py#L12)
- **Issue**: `sys.path.insert(0, str(Path(__file__).parent / "MCP"))` - looks for MCP in scripts folder!
- **Correct Path**: Should be `Path(__file__).parent.parent.parent / "LAB" / "MCP"`
- **Impact**: Import fails, agent_loop.py is completely broken

### 7. **CRITICAL: poll_tasks.py Wrong Import Path** - NEW

- **Location**: [poll_tasks.py:22](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_tasks.py#L22)
- **Issue**: `sys.path.insert(0, str(CATALYTIC_ROOT / "MCP"))` - missing "LAB"
- **Correct Path**: `CATALYTIC_ROOT / "LAB" / "MCP"`
- **Impact**: Import fails, poll_tasks.py is completely broken

### 8. **HIGH: ant-worker Directory Missing** - NEW

- **Expected Location**: `CATALYTIC-DPT/SKILLS/ant-worker/`
- **Issue**: Directory does not exist but is referenced by:
  - [poll_and_execute.py:100](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py#L100): `ant_worker = SKILLS_DIR / "ant-worker" / "run.py"`
  - [poll_tasks.py:47](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_tasks.py#L47): `ant_worker = SKILLS_DIR / "ant-worker" / "run.py"`
- **Impact**: Task execution fails with FileNotFoundError
- **Fix Required**: Create ant-worker skill or update references

### 9. **MEDIUM: No JSON Error Recovery in poll_and_execute.py** - PARTIAL

- **Location**: [poll_and_execute.py:38-43, 104-109](CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py)
- **Status**: Has try/except for JSONDecodeError but only catches errors from MCP server calls
- **Issue**: If MCP server returns malformed JSON in the response dict, it's not caught
- **Impact**: Lower priority since MCP server should return valid dicts

---

## Non-Issues (Already Fixed or Invalid)

- **TypeScript `str(err)` bug**: NOT a bug - helper function exists at [extension.ts:86-88](AGI/EXTENSIONS/antigravity-bridge/src/extension.ts#L86-L88)
- **Cursor compatibility**: VSCode extensions work in Cursor (VSCode fork) - no changes needed

---

## Implementation Priority

### Priority 1: Critical Blockers (Must fix before swarm can run)

1. **Fix agent_loop.py import path** (line 12)
2. **Fix poll_tasks.py import path** (line 22)
3. **Create or stub ant-worker skill**

### Priority 2: Correctness (Must fix before multi-ant operation)

4. **Add atomic task acquisition** to prevent race condition

### Priority 3: Robustness (Should fix for production use)

5. **Add JSON error recovery** to all MCP server read functions
6. **Add subprocess cleanup** on shutdown

---

## Detailed Fixes

### Fix agent_loop.py Import Path

```python
# Line 12 - BEFORE:
sys.path.insert(0, str(Path(__file__).parent / "MCP"))

# Line 12 - AFTER:
CATALYTIC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CATALYTIC_ROOT / "LAB" / "MCP"))
```

### Fix poll_tasks.py Import Path

```python
# Line 22 - BEFORE:
sys.path.insert(0, str(CATALYTIC_ROOT / "MCP"))

# Line 22 - AFTER:
sys.path.insert(0, str(CATALYTIC_ROOT / "LAB" / "MCP"))
```

### Add Atomic Task Acquisition

See original implementation plan - add `acquire_next_task()` method with file locking.

### Add JSON Error Recovery

Wrap all `json.loads(line)` calls in try/except:

```python
for line in f:
    try:
        entry = json.loads(line)
        # ... process entry ...
    except json.JSONDecodeError as e:
        print(f"[MCP] Skipping malformed entry: {e}")
        continue
```

---

## Testing Strategy

### Smoke Tests
1. `python agent_loop.py --role Governor` - should not crash on import
2. `python poll_tasks.py --agent Ant-1` - should not crash on import
3. `python poll_and_execute.py --role Governor` - should start polling

### Integration Tests
1. Launch Governor + 2 Ants simultaneously
2. Send 10 tasks rapidly
3. Verify no duplicate execution (check task_results.jsonl for duplicates)
4. Corrupt a ledger line, verify recovery

---

## Files Summary

| File | Status | Issues |
|------|--------|--------|
| `poll_and_execute.py` | Mostly Fixed | Subprocess cleanup needed |
| `launch_swarm.ps1` | Fixed | - |
| `agent_loop.py` | **BROKEN** | Wrong import path |
| `poll_tasks.py` | **BROKEN** | Wrong import path |
| `LAB/MCP/server.py` | Functional | Race condition, no JSON recovery |
| `SKILLS/ant-worker/` | **MISSING** | Directory doesn't exist |

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Import path fixes | Low | Simple string change |
| Race condition fix | Medium | File locking requires careful try/finally |
| JSON recovery | Low | Standard try/except pattern |
| Missing ant-worker | High | Requires design decision on what it should do |

---

## Changelog

- **2024-12-27**: Verified code state, marked fixed bugs, discovered 3 new bugs
- **2024-12-26**: Original bug report created from swarm debugging session

---

*Last updated: December 2024*