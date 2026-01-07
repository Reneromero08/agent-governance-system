# Workspace Isolation Skill

Git worktree/branch management for parallel agent work.

## Quick Start

```bash
# Check status
python run.py status

# Create isolated worktree for a task
python run.py create 2.4.1C.5

# Work in the worktree
cd "../wt-2.4.1C.5"
# ... do work ...
# ... run tests ...

# Merge back to main (after validation passes)
cd "D:/CCC 2.0/AI/agent-governance-system"
python CAPABILITY/SKILLS/agents/workspace-isolation/run.py merge 2.4.1C.5

# Cleanup
python CAPABILITY/SKILLS/agents/workspace-isolation/run.py cleanup 2.4.1C.5
```

## Commands

| Command | Description |
|---------|-------------|
| `create <task_id>` | Create isolated worktree for a task |
| `status [task_id]` | Show worktree status |
| `merge <task_id>` | Merge task branch into main |
| `cleanup <task_id>` | Remove worktree and delete branch |
| `cleanup-stale` | Find and remove stale worktrees |

## Documentation

See [SKILL.md](SKILL.md) for full documentation including:
- When to use workspace isolation
- Hard invariants
- Agent workflow
- Error handling
- Automation examples

## Tests

```bash
python -m pytest scripts/test_workspace_isolation.py -v
```

## Validation

```bash
python validate.py
```
