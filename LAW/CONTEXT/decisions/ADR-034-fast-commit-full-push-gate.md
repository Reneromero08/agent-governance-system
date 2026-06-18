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

The local gate and GitHub Actions additionally maintained separate pytest exclusion lists, allowing verification drift.

## Decision

Adopt three verification depths:

- **Fast commit:** pre-commit governance checks and critic only.
- **Full push:** critic, all contract fixtures, deterministic core pytest, plus conditional expensive suites selected from the complete unpushed change set.
- **Exhaustive:** every TESTBENCH test for releases, nightly runs, explicit deep verification, and debugging.

Every push still requires `ci_local_gate.py --full`. Full means complete mandatory verification for that push, not unconditional execution of every test category.

Real embedding integration tests are included when changes touch embedding implementation, semantic indexes, canon or ADR indexes, model registry, relevant dependencies or configuration, or the embedding tests themselves. Unrelated documentation, governance, utility, and lab changes do not load the real embedding model.

`CAPABILITY/TOOLS/utilities/push_test_plan.py` is the canonical test-selection implementation used by both the local gate and GitHub Actions.

## Verification receipt

A successful full or exhaustive gate records a JSON receipt containing the exact `HEAD`, mode, selected suites, base reference, plan hash, and timestamp.

The pre-push hook retains this receipt because it executes before network transmission and cannot observe remote success. Network retries for the same `HEAD` therefore reuse valid evidence. A changed `HEAD` invalidates the receipt automatically.

## Rationale

The push boundary remains strict while expensive verification follows dependency risk. Centralized planning eliminates local and CI drift, timing output exposes future bottlenecks, and HEAD-bound receipts preserve correctness without punishing transport failures.

## Consequences

- Routine pushes no longer run real embedding tests without an embedding-sensitive change.
- Embedding changes still receive real-model integration coverage before publication.
- Exhaustive verification remains available and explicit.
- Local and remote pytest selection share one implementation.
- Gate output identifies suite selection, plan hash, and elapsed time.
- A failed network push does not force unchanged verification to run again.

## Enforcement

- `.githooks/pre-commit` remains fast.
- `.githooks/pre-push` requires a valid receipt for current `HEAD`.
- `ci_local_gate.py --full` is mandatory before push.
- `ci_local_gate.py --exhaustive` runs the whole TESTBENCH.
- `.github/workflows/contracts.yml` invokes the canonical push test planner.

## Review triggers

- Remote CI catches failures that the local selected plan should have caught.
- A conditional suite becomes a material source of false negatives or unnecessary latency.
- Test taxonomy changes enough to require a new risk group.
- Full-push latency regresses without a corresponding increase in relevant coverage.
