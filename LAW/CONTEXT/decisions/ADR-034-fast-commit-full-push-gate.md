---
id: "ADR-034"
title: "Fast Commit, Risk-Complete Push Gate"
status: "Accepted"
date: "2026-01-03"
updated: "2026-06-18"
confidence: "High"
impact: "High"
tags: ["governance", "workflow", "ci", "performance"]
---

<!-- CONTENT_HASH: c85985b6fd756fe0d115c26c6e7d65ae067b0a6287452a58a10cce5e8506f7d1 -->
# ADR-034: Fast Commit, Risk-Complete Push Gate

## Context

Running the full repository test universe during frequent commits created excessive friction. A later implementation also conflated the mandatory push boundary with every expensive integration test, causing routine pushes to spend substantial time loading real embedding models even when semantic code had not changed.

The local gate and GitHub Actions additionally maintained separate pytest exclusion lists, allowing verification drift. The first path-aware implementation fixed that duplication but still left several excluded suites with no conditional path back into the mandatory gate. It also validated receipts against the checked-out `HEAD` rather than the refs Git was actually pushing, and branch deletion could trigger a full test run despite introducing no commit content.

## Decision

Adopt three verification depths:

- **Fast commit:** pre-commit governance checks and critic only.
- **Full push:** critic, all contract fixtures, deterministic core pytest, plus every conditional suite whose owning paths intersect the complete unpushed change set.
- **Exhaustive:** every TESTBENCH test for releases, nightly runs, explicit deep verification, and debugging.

Every content-bearing push requires `ci_local_gate.py --full`. Full means complete mandatory verification for that push, not unconditional execution of every test category. No-op and deletion-only pushes introduce no commit content and therefore bypass the receipt requirement.

`CAPABILITY/TOOLS/utilities/push_test_plan.py` is the canonical test-selection implementation used by both the local gate and GitHub Actions. Conditional test exclusions are derived from one risk-group registry. Every ignored test path must be owned by exactly one or more registered groups with explicit source/configuration triggers; an orphaned ignore is a test failure.

The current conditional groups are:

- real embeddings;
- skill discovery and skill indexing;
- cassette-network semantics and determinism;
- MCP/capability routing and pipeline contracts;
- write-firewall and guarded-write enforcement;
- stacked symbol resolution.

Test infrastructure, dependency, planner, local-gate, and CI-workflow changes select all conditional groups. Semantic changes may select multiple dependent groups rather than only the most obvious direct suite.

## Change-set determination

The planner combines merge-base and direct tree diffs. This preserves ordinary branch behavior while also including remote-only removals during divergence, rollback, or force-update scenarios. An explicitly supplied or environment-supplied base that does not resolve is a hard failure. When no usable base exists, every tracked path in `HEAD` is treated as changed instead of under-selecting tests.

## Clean-state boundary

A full gate requires a clean non-lab working tree before expensive work begins and verifies cleanliness again afterward. Untracked files are included. `THOUGHT/` remains the explicit lab exemption.

The preflight check prevents wasted full-suite runs that can only fail at the end and makes restoration of known test-generated indexes safe: there can be no pre-existing user edits to those files when cleanup runs.

## Verification receipt

A successful full or exhaustive gate records a JSON receipt containing the exact `HEAD`, mode, selected suites, selected risk groups, base reference, plan hash, and timestamp. The receipt schema is validated before authorization.

The pre-push hook retains this receipt because it executes before network transmission and cannot observe remote success. Network retries for the same commit therefore reuse valid evidence. A changed commit invalidates the receipt automatically.

Authorization is evaluated against the commit tips in Git's actual pre-push ref stream, including annotated-tag dereferencing. A receipt for the checked-out branch cannot authorize a different branch. A single invocation containing multiple distinct commit tips must be split into separately verified pushes.

## Rationale

The push boundary remains strict while expensive verification follows dependency risk. Centralized ownership eliminates local/CI drift and prevents permanent test exclusions. Risk reasons and timing output expose future bottlenecks. Ref-bound receipts preserve correctness without punishing transport failures, branch cleanup, or unchanged retries.

## Consequences

- Routine unrelated pushes do not run conditional infrastructure suites.
- Changes to an excluded subsystem always restore its owning tests to the plan.
- Cross-subsystem semantic changes may run several groups.
- Full-gate planner changes deliberately run every conditional group.
- Dirty trees fail before contracts or pytest consume time.
- Branch deletion and no-op pushes complete without a full gate.
- Pushing a branch other than the verified commit fails closed.
- Exhaustive verification remains available and explicit.
- Local and remote pytest selection share one implementation.
- Gate output identifies changed paths, selected groups, selection reasons, plan hash, and elapsed time.
- A failed network push does not force unchanged verification to run again.

## Enforcement

- `.githooks/pre-commit` remains fast.
- `.githooks/pre-push` parses the actual ref updates and requires a valid schema-checked receipt for the single commit tip being introduced.
- `ci_local_gate.py --full` is mandatory before content-bearing pushes and requires a clean tree before and after verification.
- `ci_local_gate.py --exhaustive` runs the whole TESTBENCH.
- `.github/workflows/contracts.yml` invokes the canonical push test planner.
- `CAPABILITY/TESTBENCH/01_core/test_push_test_plan.py` enforces complete risk-group ownership and fail-closed base behavior.
- `CAPABILITY/TESTBENCH/01_core/test_pre_push_guard.py` enforces ref and receipt authorization behavior.

## Review triggers

- Remote CI catches failures that the local selected plan should have caught.
- A conditional suite becomes a material source of false negatives or unnecessary latency.
- Test taxonomy changes enough to require a new risk group.
- Any ignored test path lacks an owning conditional group.
- Full-push latency regresses without a corresponding increase in relevant coverage.
