---
id: "ADR-020"
title: "Admission Control Gate"
status: "Accepted"
date: "2025-12-27"
confidence: "High"
impact: "High"
tags: ["governance", "admission-control", "determinism", "policy"]
---

<!-- CONTENT_HASH: 8a8c9bcd3b5686256c46e946f7a4a4b3e8e67b11bb006413d8d53c8dd93b7c41 -->

# ADR-020: Admission Control Gate

## Problem

Preflight ensures the repository is fresh/clean, but it does not validate whether an execution request is *allowed*:
- A clean repo can still receive an illegal intent (e.g., writes in read-only mode)
- Artifact-only runs can attempt to write outside allowed artifact roots
- Repo-write execution can be requested without explicit opt-in

Without a mechanical gate, governed execution can proceed with an unsafe intent even when the repo state is valid.

## Decision

Introduce a mandatory admission control gate immediately after preflight:
- Command: `ags admit --intent <intent.json>`
- Output: JSON to stdout (deterministic field order)
- Exit codes:
  - `0` = ALLOW
  - `2` = BLOCK (policy violation)
  - `3` = ERROR (cannot evaluate)

Intent schema (minimal):
- `mode`: `read-only` | `artifact-only` | `repo-write`
- `paths.read`: list of repo-relative paths
- `paths.write`: list of repo-relative paths
- optional `allow_repo_write` (default false)
- optional limits (`max_writes`, `max_total_bytes`)

Default policy:
- Reject invalid mode → `MODE_INVALID`
- Reject invalid paths (absolute, traversal, empty) → `PATH_INVALID`
- `read-only` + any writes → `WRITE_NOT_ALLOWED`
- `artifact-only` writes must be under the allowed artifacts root (`CONTRACTS/_runs`) → `WRITE_OUTSIDE_ARTIFACTS`
- `repo-write` requires `allow_repo_write == true` → `REPO_WRITE_FLAG_REQUIRED`

Governance wiring:
- Governed execution MUST run preflight first, admission second, then proceed to other enforcement (critic/tool).
- If admission exit code is not `0`, execution must stop (fail-closed).

## Rationale

- **Separation of concerns**: Preflight validates repository state; admission validates intent legality.
- **Determinism**: A fully mechanical gate prevents hidden or implicit behavior changes.
- **Explicit consent**: Repo-write requires an explicit flag in the intent.

## Consequences

- Clients must provide a valid intent for governed execution.
- Illegal intents are rejected early with stable reason codes.

## Related

- ADR-019: Preflight Freshness Gate
- INV-005: Determinism
- INV-006: Output Roots