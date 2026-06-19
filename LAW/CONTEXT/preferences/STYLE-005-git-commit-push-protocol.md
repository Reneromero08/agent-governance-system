<!-- CONTENT_HASH: d0f7bfad8fdf6487184c04f63a3d5e0266d09437b20c719b1087cb8776a5de37 -->

# STYLE-005: Git Commit and Push Protocol

**Authority:** CONTEXT/preferences  
**Version:** 1.2.0  
**Status:** Active  
**Category:** Governance  
**Scope:** Repository  
**Enforcement:** Strict

## Purpose

Prevent uncontrolled staging, committing, and pushing while keeping the mandatory push boundary complete for the code actually being published.

## Hard Rules

### 1. Staging

Never use `git add .` or `git add -A`. Stage explicit paths and preserve architectural commit chunks.

### 2. Branches

Work on feature branches. Do not push directly to `main`; integrate through reviewed history or an explicitly authorized local cherry-pick/merge workflow.

### 3. Commit verification

Before a commit, run:

```bash
python CAPABILITY/TOOLS/governance/critic.py
```

The pre-commit hook also enforces canon governance and INBOX policy.

### 4. Push verification

Before every content-bearing push, run from a clean non-lab tree:

```bash
python CAPABILITY/TOOLS/utilities/ci_local_gate.py --full
```

`--full` always runs critic, all contract fixtures, and deterministic core pytest. It then restores every conditional suite whose owning source, test, dependency, or configuration path changed. The groups are write firewall, symbol resolution, MCP/capability contracts, skill discovery, cassette network, and embeddings.

Planner, dependency, test-configuration, local-gate, and CI-workflow changes select all groups. Semantic changes may select several dependent groups. Untracked files count as dirty; `THOUGHT/` is the explicit lab exemption.

For releases, nightly verification, or deliberate whole-repository validation:

```bash
python CAPABILITY/TOOLS/utilities/ci_local_gate.py --exhaustive
```

No-op and deletion-only pushes do not require a receipt.

### 5. Verification receipts

A successful full gate writes `LAW/CONTRACTS/_runs/ALLOW_PUSH.token` as schema-validated JSON bound to the exact commit, resolved tested base SHA, plan hash, selected suites, and selected risk groups.

The pre-push hook validates Git's actual ref updates. A receipt cannot authorize another branch, a changed commit, or a remote ref that advanced after testing. New refs require the tested base to be an ancestor. Annotated tags resolve to their target commit. Multiple distinct tips or remote bases must be pushed separately.

The receipt is retained after pre-push so a network retry of the same commit against the same base does not rerun verification.

### 6. Change-set calculation

The planner unions merge-base and direct tree diffs. Explicit or environment-provided base refs must resolve. If no base exists, all tracked paths in `HEAD` are treated as changed.

### 7. Approval boundary

One approval authorizes one commit. Commit and push remain separate actions unless the human explicitly grants composite approval.

## Canonical flow

1. Complete the architectural change on a feature branch.
2. Run focused tests while developing.
3. Run critic.
4. Stage explicit paths and obtain commit approval.
5. Commit once.
6. Confirm the non-lab tree is clean.
7. Run `ci_local_gate.py --full`.
8. Review selected risk groups and matched paths.
9. Obtain push approval.
10. Push the verified commit tip.

## Enforcement

- `.githooks/pre-commit` enforces the fast commit boundary.
- `.githooks/pre-push` validates actual ref updates against the receipt.
- `.github/workflows/contracts.yml` runs the same frozen plan with per-suite evidence.
- `push_test_plan.py` is the single source of truth for core, conditional, and exhaustive selection.
- Focused planner tests prove every ignored suite has exactly one owner.
