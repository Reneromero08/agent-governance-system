---
id: "ADR-034"
title: "Fast Commit, Full Push Gate"
status: "Accepted"
date: "2026-01-03"
confidence: "High"
impact: "Medium"
tags: ["governance", "workflow", "ci"]
---

<!-- CONTENT_HASH: 3babf6b153336dc41515a80bce4580ff0203c4a665a76fefb7434f505c0c90a8 -->

# ADR-034: Fast Commit, Full Push Gate

## Context

Local development was slowed by running the full governance + test suite during frequent commits. This created friction and made routine iteration feel blocked, even when changes were not ready to be pushed.

## Decision

Adopt a two-tier workflow:

- **Fast commits:** allow frequent commits after fast local checks (pre-commit + critic) without requiring the full long-running suite.
- **Full push gate:** require the full CI-aligned suite to pass **before push** by minting a one-time token tied to `HEAD`.

## Alternatives considered

- **Run full suite before every commit:** safest, but too much drag for iterative work.
- **Rely on CI only:** fastest locally, but increases failed pushes/PRs and slows feedback.
- **Run only `runner` on push:** faster than full, but misses `pytest CAPABILITY/TESTBENCH/` coverage.

## Rationale

The push boundary is the point where changes become shared and should meet the strongest guarantees. Keeping commit-time checks fast enables small, frequent commits without sacrificing CI alignment, because the push gate blocks unverified history from being published.

## Consequences

- Faster iteration and more frequent commits.
- Developers may have local commits that would fail full CI, but they cannot be pushed until the full gate passes.
- Push becomes the canonical “full confidence” moment.

## Enforcement

- `.githooks/pre-commit` remains fast.
- `.githooks/pre-push` requires a `CI_OK` token for the current `HEAD` (minted by `python CAPABILITY/TOOLS/utilities/ci_local_gate.py --full`) or runs the full gate when a legacy/manual token is present.
- `AGENTS.md` documents the split: fast commit, full push.

## Review triggers

- CI failure rate increases despite local gating.
- The full gate becomes fast enough to consider moving portions back to commit-time.
