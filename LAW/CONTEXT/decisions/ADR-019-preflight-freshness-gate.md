---
id: "ADR-019"
title: "Preflight Freshness Gate"
status: "Accepted"
date: "2025-12-27"
confidence: "High"
impact: "High"
tags: ["governance", "preflight", "cortex", "determinism"]
---

<!-- CONTENT_HASH: c518319f5689dcd18e156cff14e06a81810aae5905d7a5b36c252faf2a610272 -->

# ADR-019: Preflight Freshness Gate

## Problem

Governed execution can run against a stale or ambiguous repository state:
- Tracked files may be dirty (unknown state)
- Cortex may be missing, or out of date relative to Canon
- Agents may execute without a mechanical “freshness” gate, causing drift and invalid assumptions

This makes runs non-reproducible and undermines determinism and governance enforcement.

## Decision

Introduce a mandatory mechanical preflight check:
- Command: `ags preflight`
- Output: JSON to stdout (deterministic field order)
- Exit codes:
  - `0` = SAFE
  - `2` = BLOCKED (policy violation)
  - `3` = ERROR (cannot evaluate)

Default blocking rules:
- Dirty tracked files → `DIRTY_TRACKED`
- Cortex missing → `CORTEX_MISSING`
- Canon changed since cortex build → `CANON_CHANGED_SINCE_CORTEX`

Warnings only (reported but not blocking by default):
- Untracked files present → `UNTRACKED_PRESENT`
- Handoff missing → `HANDOFF_MISSING`

Governance wiring:
- Governed MCP tool execution MUST run preflight before critic checks.
- If preflight exit code is not `0`, execution must stop (fail-closed).

## Rationale

- **Determinism**: A single mechanical gate prevents “unknown repo state” execution.
- **Cortex trust**: Enforces that Canon changes require cortex regeneration to maintain a coherent index.
- **Fail-closed governance**: Prevents silent bypass of safety conditions.

## Consequences

- Canon edits require rebuilding cortex before governed execution will proceed.
- Dirty working trees will block governed execution unless explicitly overridden via preflight flags.
- Repo state becomes explicitly visible as a JSON contract suitable for automated auditing.

## Related

- ADR-014: Cortex Section Index and CLI
- ADR-015: Logging Output Roots
- INV-005: Determinism
- INV-006: Output Roots