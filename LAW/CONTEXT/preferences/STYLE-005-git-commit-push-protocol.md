<!-- CONTENT_HASH: 242ba51ef7ef56d3d0799898f9a1de784f8c6a525d3abdac364a78581fe6d7da -->

# STYLE-005: Git Commit and Push Protocol

**Authority:** CONTEXT/preferences  
**Version:** 1.1.0  
**Status:** Active  
**Category:** Governance  
**Scope:** Repository  
**Enforcement:** Strict

## Purpose

Prevent uncontrolled staging, committing, and pushing while keeping the mandatory push boundary proportionate to the code being published.

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

Before every push, run:

```bash
python CAPABILITY/TOOLS/utilities/ci_local_gate.py --full
```

`--full` means the complete mandatory push plan for the actual change set:

- critic and all contract fixtures always run;
- deterministic core pytest always runs;
- real embedding integration tests run when embedding, semantic-index, canon-index, ADR-index, model-registry, dependency, or embedding-test paths changed;
- other expensive infrastructure suites are reserved for explicit exhaustive verification.

For releases, nightly verification, test-infrastructure changes, or deliberate whole-repository validation, run:

```bash
python CAPABILITY/TOOLS/utilities/ci_local_gate.py --exhaustive
```

`--exhaustive` implies `--full` and runs the entire TESTBENCH without normal push exclusions.

### 5. Verification receipts

A successful full gate writes `LAW/CONTRACTS/_runs/ALLOW_PUSH.token` as a JSON verification receipt tied to the exact `HEAD` and test-plan hash.

The receipt is not deleted by `pre-push`. Git hooks run before network transmission and cannot know whether the remote accepted the push. Retaining the receipt allows a network retry without rerunning unchanged verification. Any new commit changes `HEAD` and invalidates the receipt automatically.

### 6. Approval boundary

One approval authorizes one commit. Commit and push remain separate actions unless the human explicitly grants composite approval.

## Canonical flow

1. Make the complete architectural change.
2. Run relevant focused tests while developing.
3. Run critic before commit.
4. Stage explicit paths and obtain commit approval.
5. Commit once.
6. Run `ci_local_gate.py --full` before push.
7. Obtain push approval.
8. Push the verified `HEAD`.

## Enforcement

- `.githooks/pre-commit` enforces the fast commit boundary.
- `.githooks/pre-push` requires a valid HEAD-bound receipt.
- `.github/workflows/contracts.yml` uses the same canonical pytest planner as the local full gate.
- `CAPABILITY/TOOLS/utilities/push_test_plan.py` is the single source of truth for core, conditional embedding, and exhaustive pytest selection.
