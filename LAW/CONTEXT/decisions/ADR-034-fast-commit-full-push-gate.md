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

<!-- CONTENT_HASH: 50843d65841dd80dc0d749e3a3944b649bc7498b9ff5ab11e13425f44706b559 -->
# ADR-034: Fast Commit, Risk-Complete Push Gate

## Context

Running the entire repository test universe during frequent commits created excessive friction. The first optimized push gate reduced embedding work, but several suites were excluded from core pytest without an explicit conditional path back into mandatory verification. Local and remote selection could drift, receipts were tied only to the checked-out `HEAD`, and dirty trees could consume the whole gate before failing.

## Decision

Adopt three verification depths:

- **Fast commit:** pre-commit governance and critic checks.
- **Full push:** critic, all contract fixtures, deterministic core pytest, plus every conditional suite whose owning paths intersect the complete unpushed change set.
- **Exhaustive:** the entire TESTBENCH for releases, nightly validation, and deliberate deep verification.

Every content-bearing push requires `ci_local_gate.py --full`. No-op and deletion-only pushes introduce no commit content and bypass the receipt requirement.

`CAPABILITY/TOOLS/utilities/push_test_plan.py` is the canonical planner used locally and in GitHub Actions. Conditional exclusions are derived from one risk-group registry. Every ignored test path must have exactly one owner, with explicit source, test, dependency, and configuration triggers. Planner and test-infrastructure changes select all groups.

The mandatory conditional groups are:

- write-firewall enforcement;
- stacked symbol resolution;
- MCP/capability routing and pipeline contracts;
- skill discovery and indexing;
- cassette-network semantics and determinism;
- real embedding integration.

Suites that mutate fixed shared fixtures may disable xdist while the rest of the plan remains parallel.

## Change-set determination

The planner combines merge-base and direct tree diffs. This includes locally introduced paths and remote-only removals during divergence, rollback, or force-update scenarios. Explicit or environment-supplied bases must resolve or planning fails closed. If no base exists, every tracked path in `HEAD` is treated as changed.

## Clean-state boundary

A full gate requires a clean non-lab working tree before expensive checks and verifies cleanliness again afterward. Untracked files count as dirty. `THOUGHT/` remains the explicit lab exemption. The gate also verifies that `HEAD` did not move while checks were running.

This preflight makes restoration of known test-generated indexes safe: there can be no pre-existing user edits to those paths when cleanup runs.

## Verification receipt

A successful full or exhaustive gate records a schema-validated JSON receipt containing the exact commit, resolved tested base SHA, mode, selected suites, selected risk groups, plan hash, and timestamp.

The pre-push hook validates Git's actual ref stream rather than the currently checked-out branch. Existing remote refs must still be at the tested base; new refs require that base to be an ancestor. Annotated tags are dereferenced to commits. A push containing multiple distinct commit tips or remote bases must be split.

Receipts survive transport failures because pre-push cannot observe whether the remote accepted the update. Any commit or remote-base change invalidates the evidence.

## CI execution

GitHub Actions freezes one canonical plan, runs each selected suite as a separately named step, retains per-suite logs, and aggregates all selected outcomes before failing. Semantic dependencies are installed only when a semantic suite is selected.

## Consequences

- Routine unrelated pushes remain fast without permanently excluding infrastructure tests.
- Every excluded suite has a mechanically enforced path back into mandatory coverage.
- Cross-subsystem changes may select several groups.
- Dirty trees and moving commits fail before a receipt can be minted.
- Network retries reuse valid evidence; remote advances do not.
- CI identifies the exact failing suite and preserves its timing/log evidence.
- Exhaustive verification remains explicit.

## Enforcement

- `.githooks/pre-commit` remains fast.
- `.githooks/pre-push` authorizes actual pushed refs against a commit-and-base-bound receipt.
- `ci_local_gate.py --full` is mandatory before content-bearing pushes.
- `ci_local_gate.py --exhaustive` runs the whole TESTBENCH.
- `.github/workflows/contracts.yml` executes the frozen canonical plan.
- Focused core tests enforce risk ownership, path portability, receipt schema, clean-state behavior, and suite isolation.

## Review triggers

- Remote CI catches a failure the selected local plan should have caught.
- An ignored suite lacks exactly one conditional owner.
- A risk group becomes materially slow or noisy.
- Test taxonomy or shared-fixture behavior changes.
- Full-push latency regresses without increased relevant coverage.
