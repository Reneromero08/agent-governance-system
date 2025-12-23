# STYLE-001: Chunked Commit Ceremony

**Category:** Governance
**Status:** Active
**Scope:** Repository-wide
**Enforcement:** Strict

---

## Context
By default, the agent was committing and pushing every small fix (hotfixes v1.0.1 and v1.0.2). The user prefers to control the "chunking" of work and wants to explicitly approve commits and pushes.

## Rule

### The Core Constraint
**Every single `git commit`, `git push`, and release publication requires explicit, per-instance user approval.**

This is not negotiable. This is not bypassable. This is the highest-priority governance rule for agent behavior.

### Prohibited Interpretations
The following phrases do **NOT** authorize commits or pushes:
- "proceed"
- "there you go"
- "finish it"
- "let's move on to the next task"
- "continue"
- "yes" (unless in direct response to a commit ceremony prompt)

These authorize **implementation** only. They are **never** implicit commit approvals.

### Explicit approvals
An explicit "commit" directive counts as approval to commit once checks have passed and staged files are listed; no additional confirmation prompt is required.

Explicit composite directives that include "commit", "push", and "release" (for example,
"commit, push, and release") count as approval for each action listed in that request.
This does not authorize additional commits beyond the current task.

### Ceremony confirmations
When checks have passed and staged files have been listed, short confirmations such as
"go on" count as approval for the listed actions.

### The Anti-Chaining Rule
**One commit approval = one commit.** If the user approves a commit, the agent may execute that single commit. The agent may NOT:
- Infer that subsequent tasks should also be committed.
- Chain multiple commits under a single approval.
- Commit new work completed after the approval was given.

If the agent completes Task B after receiving approval for Task A, the agent MUST stop and request a new approval for Task B.

### The Ceremony Phase
Before any Git command, the agent MUST:
1. **Failsafe Rule**: Run `TOOLS/critic.py` and `CONTRACTS/runner.py`. Confirm they pass.
2. **Stop** all execution.
3. **List** every file in the staging area.
4. **Ask**: "Ready for the Chunked Commit Ceremony? Shall I commit and push these [N] files?"
5. **Wait** for explicit user approval (e.g., "yes, commit" or "commit and push"), unless the user already issued an explicit "commit" directive or composite approval for the listed actions.

### No Composite Commands
Do not chain `git commit` and `git push` unless the user explicitly says "commit and push."
Explicit "commit, push, and release" is also allowed. Default to separate checkpoints.

## Governance Violation
Any commit or push executed without following this ceremony is a **critical governance violation**.

## Status
**Active**
Added: 2025-12-21
Strengthened: 2025-12-21 (added anti-chaining rule and prohibited interpretations)
