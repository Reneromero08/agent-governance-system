---
version: "1.0.0"
status: "Active"
required_canon_version: ">=3.0.0"
---

# Skill: workspace-isolation

## Purpose

Enable parallel agent work without conflicts using git worktrees and branches.
Each agent works in an isolated workspace on its own branch, preventing:
- File conflicts between agents
- Dirty workspace interference
- Accidental commits to main

## When to Use

**ALWAYS** use workspace isolation when:
1. Multiple agents may work simultaneously
2. The main workspace has uncommitted changes
3. The task may fail tests (keeps failures isolated)
4. Running long-running or experimental work

**SKIP** workspace isolation only when:
- Quick single-file edits with clean workspace
- Read-only operations (research, analysis)
- User explicitly requests working in main

## Hard Invariants

1. **Never work in detached HEAD** - Always be on a branch
2. **Never merge until validation passes** - Tests must be green
3. **Never auto-delete on failure** - Preserve evidence for debugging
4. **Always cleanup after merge** - Remove worktree + branch when done
5. **One task = one worktree** - Don't share worktrees between tasks

## Standard Naming

| Item | Pattern | Example |
|------|---------|---------|
| Branch | `task/<task_id>` | `task/2.4.1C.5` |
| Worktree | `../wt-<task_id>` | `../wt-2.4.1C.5` |

## Agent Workflow

### Step 1: Check Current State

```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py status
```

This shows:
- Current branch
- Whether workspace is dirty
- Existing worktrees

### Step 2: Create Isolated Worktree

```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py create <task_id>
```

Example:
```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py create 2.4.1C.5
```

Output:
```json
{
  "task_id": "2.4.1C.5",
  "branch": "task/2.4.1C.5",
  "path": "D:/CCC 2.0/AI/wt-2.4.1C.5",
  "created_at": "2026-01-07T05:00:00Z",
  "base_branch": "main"
}
```

### Step 3: Work in Isolated Worktree

```bash
cd "../wt-<task_id>"
# Do your work
# Make commits
# Run tests
```

### Step 4: Validate Before Merge

Run all required validation:
```bash
python -m pytest CAPABILITY/TESTBENCH -x
python CAPABILITY/PRIMITIVES/preflight.py
```

**DO NOT PROCEED TO MERGE IF VALIDATION FAILS!**

### Step 5: Merge to Main (Only After Validation Passes)

Return to main workspace first:
```bash
cd "D:/CCC 2.0/AI/agent-governance-system"
```

Then merge:
```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py merge <task_id>
```

Example:
```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py merge 2.4.1C.5
```

### Step 6: Cleanup (Required!)

After successful merge, clean up the worktree and branch:

```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py cleanup <task_id>
```

Example:
```bash
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py cleanup 2.4.1C.5
```

## Commands Reference

### create

Create isolated worktree for a task.

```bash
python workspace_isolation.py create <task_id> [--base <branch>]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `task_id` | Unique task identifier | Required |
| `--base` | Base branch for new branch | `main` |

### status

Show worktree status.

```bash
python workspace_isolation.py status [task_id]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `task_id` | Optional filter by task | All worktrees |

### merge

Merge task branch into main.

```bash
python workspace_isolation.py merge <task_id> [--delete-branch]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `task_id` | Task to merge | Required |
| `--delete-branch` | Delete branch after merge | `false` |

**Prerequisites:**
- Must be on `main` branch
- Main workspace must be clean
- Task branch must exist

### cleanup

Remove worktree and delete branch.

```bash
python workspace_isolation.py cleanup <task_id> [--force] [--keep-branch]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `task_id` | Task to clean up | Required |
| `--force` | Force removal with uncommitted changes | `false` |
| `--keep-branch` | Keep branch after removing worktree | `false` |

### cleanup-stale

Find and remove stale worktrees (already merged to main).

```bash
python workspace_isolation.py cleanup-stale [--apply]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--apply` | Actually remove stale worktrees | `false` (dry-run) |

## Failure Handling

### If Tests Fail

1. **DO NOT MERGE** - Keep the worktree for inspection
2. Fix issues in the worktree
3. Commit fixes
4. Re-run validation
5. Only merge when validation passes

### If Merge Conflicts

1. Resolve conflicts manually in the worktree
2. Commit the resolution
3. Re-run validation
4. Retry merge

### If Something Goes Wrong

Force cleanup (preserves nothing):
```bash
python workspace_isolation.py cleanup <task_id> --force
```

## Automation for Orchestrators

### Full Automated Pipeline

```python
import subprocess
import sys

def run_isolated_task(task_id: str, task_commands: list[str]) -> bool:
    """Run a task in an isolated worktree with full lifecycle."""
    wt_script = "CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py"

    # 1. Create worktree
    result = subprocess.run([sys.executable, wt_script, "create", task_id], capture_output=True)
    if result.returncode != 0:
        print(f"Failed to create worktree: {result.stderr}")
        return False

    # Parse worktree path from output
    import json
    wt_info = json.loads(result.stdout)
    wt_path = wt_info["path"]

    # 2. Run task commands in worktree
    success = True
    for cmd in task_commands:
        proc = subprocess.run(cmd, shell=True, cwd=wt_path)
        if proc.returncode != 0:
            success = False
            break

    if not success:
        print(f"Task failed. Worktree preserved at {wt_path}")
        return False

    # 3. Merge to main (from main workspace)
    result = subprocess.run([sys.executable, wt_script, "merge", task_id])
    if result.returncode != 0:
        print(f"Merge failed. Manual intervention required.")
        return False

    # 4. Cleanup
    subprocess.run([sys.executable, wt_script, "cleanup", task_id])

    return True
```

### Periodic Cleanup (Cron/Scheduled Task)

```bash
# Find and remove all stale worktrees
python CAPABILITY/SKILLS/agents/workspace-isolation/scripts/workspace_isolation.py cleanup-stale --apply
```

## Integration with AGS Pipeline

When using `ags run`, the pipeline can automatically:
1. Create worktree before execution
2. Run all steps in the isolated worktree
3. Merge on success, preserve on failure
4. Cleanup after successful merge

To enable, set in your plan:
```json
{
  "workspace_isolation": {
    "enabled": true,
    "task_id": "pipeline-001",
    "cleanup_on_success": true
  }
}
```

## Governance

- All worktree operations logged to stdout
- JSON output for machine parsing
- Exit codes: 0 = success, 1 = expected error, 2 = unexpected error
- Branch naming enforced: `task/<task_id>` pattern only
