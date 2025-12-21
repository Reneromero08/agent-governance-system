# STYLE-001: Chunked Commit Ceremony

## Context
By default, the agent was committing and pushing every small fix (hotfixes v1.0.1 and v1.0.2). The user prefers to control the "chunking" of work and wants to explicitly approve commits and pushes.

## Rule
- **No Auto-Git**: The agent MUST NOT run `git commit` or `git push` without explicit user permission for that specific set of changes.
- **The "Blanket Approval" Ban**: Generic approvals (e.g., "there you go", "proceed", "finish it") apply ONLY to the implementation level. They NEVER authorize a `git commit` or `git push`.
- **The Ceremony Phase**: Before any Git command, the agent MUST:
  1. **The Failsafe Rule**: Run all mandatory verification tools (`TOOLS/critic.py` and `CONTRACTS/runner.py`) and confirm they pass.
  2. Stop all execution.
  3. List every file in the staging area.
  4. Explicitly ask: **"Ready for the Chunked Commit Ceremony? Shall I commit and push these [N] files?"**
- **No Composite Commands**: Do not chain `git commit` and `git push` unless the user explicitly asks to "Commit and Push." Default to separate checkpoints.

## Governance Violation
Moving forward without the "Explicit Prompt" check is a high-priority governance violation.

## Status
**Active**
Added: 2025-12-21
