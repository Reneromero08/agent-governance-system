# ADR-002: Store packs under MEMORY/LLM_PACKER

**Status:** Accepted

**Date:** 2025-12-20

**Review date:** (optional)

**Confidence:** Medium

**Impact:** Low

**Tags:** [artifacts, memory]

## Context

Packs are generated artifacts and should live near the subsystem that owns them. The LLM packer lives under `MEMORY/LLM_PACKER/`, but the initial implementation wrote packs under `MEMORY/_packs/`.

## Decision

- The canonical packs are stored in **`MEMORY/LLM_PACKER/_packs/`**.
- Within `_packs/`, non-pack artifacts (fixtures, baselines, archives) are stored under `_packs/_system/`.
- Delta baseline state is stored under `MEMORY/LLM_PACKER/_packs/_system/_state/`.

## Alternatives considered

- Keep packs under `MEMORY/_packs/`.
- Move packs to a top-level artifact directory.

## Rationale

Colocating packs with the LLM packer keeps the repository modular and reduces ambiguity about what owns the format and lifecycle of packs.

## Consequences

- Docs, fixtures, and canon output roots must reference the new location.
- Existing ignored pack artifacts under the old path are treated as legacy and disposable.

## Enforcement

- `python CONTRACTS/runner.py` includes smoke fixtures that assert pack outputs are created under `MEMORY/LLM_PACKER/_packs/` (fixtures should write to `_packs/_system/`).

## Review triggers

- Packs become shared infrastructure across multiple subsystems.

## Amendment (2025-12-25)

This ADR is amended to clarify the convention inside the existing invariant output root (`MEMORY/LLM_PACKER/_packs/`): keep user packs and fixture packs in separate subfolders to reduce navigation clutter without changing output-root invariants.
