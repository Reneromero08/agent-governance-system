<!-- CONTENT_HASH: 242ba51ef7ef56d3d0799898f9a1de784f8c6a525d3abdac364a78581fe6d7da -->

# STYLE-005: Git Commit and Push Protocol

**Authority:** CONTEXT/preferences  
**Version:** 1.2.0  
**Status:** Active  
**Category:** Governance  
**Scope:** Repository  
**Enforcement:** Strict

## Purpose

Prevent uncontrolled staging, committing, and pushing while keeping the mandatory push boundary proportionate to the code actually being published.

## Hard Rules

### 1. Staging

Never use `git add .` or `git add -A`. Stage explicit paths and preserve architectural commit chunks.

### 2. Branches

Work on feature branches. Do not push directly to `main`; integrate through reviewed branch history or an explicitly authorized local cherry-pick/merge workflow.

### 3. Commit verification

Before a commit, run the fast governance boundary:

```bash
python CAPABILITY/TOOLS/governance/critic.py
```

The pre-commit hook also runs canon governance, INBOX policy, and critic checks.

### 4. Push verification

Before every content-bearing push, run:

```bash
python CAPABILITY/TOOLS/utilities/ci_local_gate.py --full
```

Run the full gate only from a clean non-lab working tree. The gate checks this before expensive work and again afterward. Untracked files count as dirty. `THOUGHT/` remains the explicit lab exemption.

`--full` means the complete mandatory push plan for the actual unpushed change set:

- critic and all contract fixtures always run;
- deterministic core pytest always runs;
- each excluded infrastructure suite is restored when its owning source, test, dependency, or configuration path changed;
- semantic changes may select several dependent groups;
- planner, test configuration, dependency, local-gate, and CI-workflow changes select every conditional group.

The conditional groups are embeddings, skill discovery, cassette network, MCP/capability contracts, write firewall, and symbol resolution. An ignored test path without an owning risk group is a gate defect and must fail focused planner tests.

For releases, nightly verification, or deliberate whole-repository validation, run:

```bash
python CAPABILITY/TOOLS/utilities/ci_local_gate.py --exhaustive
```

`--exhaustive` implies `--full` and runs the entire TESTBENCH without normal push exclusions.

No-op pushes and deletion-only pushes introduce no commit content and do not require a verification receipt.

### 5. Verification receipts

A successful full gate writes `LAW/CONTRACTS/_runs/ALLOW_PUSH.token` as a schema-validated JSON verification receipt tied to the exact commit, test-plan hash, selected suites, and selected risk groups.

The receipt is not deleted by `pre-push`. Git hooks run before network transmission and cannot know whether the remote accepted the push. Retaining the receipt allows a network retry without rerunning unchanged verification.

The hook validates the commit tips in Git's actual pre-push ref stream, not whichever branch is currently checked out. A receipt cannot authorize a different branch. Annotated tags are resolved to their target commit. Pushes containing multiple distinct commit tips must be split and verified separately.

### 6. Change-set calculation

The planner combines merge-base and direct tree diffs so divergent or rollback pushes include both locally introduced paths and remotely removed paths. Explicit or environment-provided base refs must resolve or the gate fails. If no base exists, every tracked path in `HEAD` is treated as changed.

### 7. Approval boundary

One approval authorizes one commit. Commit and push remain separate actions unless the human explicitly grants composite approval.

## Canonical flow

1. Make the complete architectural change on a feature branch.
2. Run relevant focused tests while developing.
3. Run critic before commit.
4. Stage explicit paths and obtain commit approval.
5. Commit once.
6. Confirm the non-lab tree is clean.
7. Run `ci_local_gate.py --full` before a content-bearing push.
8. Review selected risk groups and their matched paths in the gate output.
9. Obtain push approval.
10. Push the verified commit tip.

## Enforcement

- `.githooks/pre-commit` enforces the fast commit boundary.
- `.githooks/pre-push` parses actual ref updates, bypasses no-op/deletion-only operations, and requires a valid receipt for the commit being introduced.
- `.github/workflows/contracts.yml` uses the same canonical pytest planner as the local full gate.
- `CAPABILITY/TOOLS/utilities/push_test_plan.py` is the single source of truth for core, conditional risk groups, and exhaustive pytest selection.
- `CAPABILITY/TESTBENCH/01_core/test_push_test_plan.py` proves that every ignored suite has a conditional owner.
