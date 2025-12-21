# ADR-002: Store packs under MEMORY/LLM_PACKER

**Status:** Accepted

**Date:** 2025-12-20

**Review date:** (optional)

**Confidence:** Medium

**Impact:** Low

**Tags:** [artifacts, memory]

## Context

Packs are generated artifacts and should live near the subsystem that owns them. The LLM packer lives under `MEMORY/LLM-PACKER/`, but the initial implementation wrote packs under `MEMORY/_packs/`.

## Decision

- The canonical packs are stored in **`MEMORY/LLM_PACKER/_packs/`**.
- Delta baseline state is stored under `MEMORY/LLM-PACKER/_packs/_state/`.

## Alternatives considered

- Keep packs under `MEMORY/_packs/`.
- Move packs to a top-level artifact directory.

## Rationale

Colocating packs with the LLM packer keeps the repository modular and reduces ambiguity about what owns the format and lifecycle of packs.

## Consequences

- Docs, fixtures, and canon output roots must reference the new location.
- Existing ignored pack artifacts under the old path are treated as legacy and disposable.

## Enforcement

- `python CONTRACTS/runner.py` includes a smoke fixture that asserts pack outputs are created under `MEMORY/LLM-PACKER/_packs/`.

## Review triggers

- Packs become shared infrastructure across multiple subsystems.
