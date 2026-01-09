<!-- CONTENT_HASH: placeholder -->

**required_canon_version:** >=3.0.0


# Skill: workspace-isolation

**Version:** 1.0.0

**Status:** Active


## Purpose

Enforces disciplined git workflow for agents to prevent commit bloat and ensure human approval before permanent repository changes.

## Trigger

This skill applies to ALL agent sessions that involve repository modifications. Agents MUST follow these rules whenever performing git operations.

## CRITICAL: Human Review Gate

**The user MUST review and approve ALL work BEFORE any commit or merge.**

Agents do NOT decide when work is ready. The human does. Present your work, then STOP and WAIT.

---

## Git Workflow Rules

### Phase 1: Work Completion (NO GIT OPERATIONS)

1. **Complete ALL work first** - Do not commit anything until the entire task is finished
2. **Stage changes for review** - Use `git add` to stage, but DO NOT commit
3. **STOP AND PRESENT WORK** - Show the user:
   - Full `git diff --staged` output
   - Summary of what was changed and why
   - Proposed commit message
4. **WAIT for explicit approval** - Do NOT proceed until user says "commit" or equivalent
   - "Looks good" is NOT approval to commit
   - "Yes" to a question about the code is NOT approval to commit
   - Only explicit commit approval (e.g., "commit it", "yes commit", "go ahead and commit") authorizes the commit

### Phase 2: Single Commit (AFTER APPROVAL, BEFORE MERGE)

1. **One commit only** - After explicit approval, create exactly ONE commit with all changes
2. **Descriptive message** - Use a clear, comprehensive commit message covering all changes
3. **No intermediate commits** - Never create multiple commits for incremental work
4. **STOP AGAIN** - After committing, STOP and ask about merge. Do NOT auto-merge.

### Phase 3: Merge (SEPARATE APPROVAL REQUIRED)

1. **STOP AND ASK** - Present merge plan to user
2. **Request explicit merge approval** - "Ready to merge to [branch]?" then WAIT
3. **Fast-forward when possible** - Prefer clean history
4. **Rebase if needed** - Keep history linear when appropriate
5. **Delete feature branch after merge** - Clean up merged branches immediately:
   - `git branch -d feature-branch` (safe delete, only works if fully merged)
   - Never leave stale branches cluttering the repo

### Phase 4: Post-Merge Cleanup (CONDITIONAL)

1. **Amend is SAFE only when:**
   - The commit has NOT been pushed to remote
   - OR the branch is a personal feature branch with no collaborators
   - AND the user explicitly approves the amend

2. **Amend is UNSAFE when:**
   - Commit has been pushed to shared remote
   - Others may have pulled the branch
   - Working on main/master/develop branches

3. **For post-merge cleanup** (changelog updates, version bumps):
   - If amend is safe: `git commit --amend` to add to previous commit
   - If amend is unsafe: Create a new "chore:" commit

## Prohibited Actions

Agents MUST NOT:
- `git commit` before all work is complete
- `git commit` without user approval
- `git merge` without user approval
- `git push` without user approval
- `git commit --amend` on pushed commits without explicit user consent
- Create multiple commits for a single logical change
- Commit, merge, then commit again (the Gemini anti-pattern)

## Approval Checkpoints

| Action | Requires Approval |
|--------|-------------------|
| `git add` | No |
| `git commit` | **YES** |
| `git merge` | **YES** |
| `git push` | **YES** |
| `git commit --amend` | **YES** (with safety check) |
| `git rebase` | **YES** |
| `git branch -d` | No (auto after merge) |

## Example Workflow

```
1. Agent receives task: "Add feature X and update changelog"
2. Agent implements feature X (no commits)
3. Agent updates changelog (no commits)
4. Agent stages all changes: git add -A
5. Agent: "All work complete. Here's what I've done:

   [git diff --staged output]

   Summary:
   - Added feature X in src/feature.ts (new endpoint for user preferences)
   - Updated CHANGELOG.md (added entry for v1.2.0)

   Proposed commit message: 'feat: add user preferences endpoint'

   Please review the changes above. Let me know when you'd like me to commit."

   >>> AGENT STOPS HERE AND WAITS <<<

6. User reviews the diff, maybe asks questions, requests changes
7. Agent makes any requested changes, re-stages, presents again
8. User: "Commit it" (explicit approval)
9. Agent: git commit -m "feat: add user preferences endpoint"
10. Agent: "Committed. Ready to merge to main?"

    >>> AGENT STOPS HERE AND WAITS <<<

11. User: "Yes, merge"
12. Agent: git checkout main && git merge feature-branch && git branch -d feature-branch
13. Agent: "Merged and deleted feature-branch."
14. If post-merge cleanup needed:
    Agent: "Need to update version number. Safe to amend? (not pushed yet)"

    >>> AGENT STOPS HERE AND WAITS <<<
```

## Constraints

- Must present `git diff --staged` or `git status` before requesting commit approval
- Must verify push status before offering amend option
- Must track approval state throughout session
- Must escalate to user on any git operation uncertainty

## Outputs

Agent should output at each checkpoint:
- Summary of staged changes
- Proposed commit message
- Safety status for amend operations
- Clear approval request

**required_canon_version:** >=3.0.0
