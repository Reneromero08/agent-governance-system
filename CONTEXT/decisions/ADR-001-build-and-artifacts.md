# ADR-001: BUILD is user output, artifacts are subsystem-owned

**Status:** Accepted

**Date:** 2025-12-20

**Review date:** (optional)

**Confidence:** Medium

**Impact:** Medium

**Tags:** [architecture, artifacts]

## Context

The initial scaffold treated `BUILD/` as the output root for system-generated artifacts (fixtures, cortex index, packs).

The intended meaning of `BUILD/` in this template is "dist for users": it is where template users may place their own build outputs. System tooling should not write internal artifacts there.

## Decision

- Reserve `BUILD/` for user build outputs.
- Write system-generated artifacts near their subsystem:
  - `CONTRACTS/_runs/` for fixture runner outputs
  - `CORTEX/_generated/` for the cortex index
  - `MEMORY/LLM-PACKER-1.1/_packs/` for memory and LLM handoff packs
- Include only an inventory of `BUILD/` (file tree) in packs, not its contents.

## Alternatives considered

- Keep a single centralized artifact root in `BUILD/`.
- Introduce a new top-level artifact directory.

## Rationale

Subsystem-owned artifact roots keep outputs colocated with the code that generates them, improve navigation, and prevent `BUILD/` from becoming a mixed-purpose dumping ground.

## Consequences

- Scripts and docs must be updated to match the new locations.
- Existing generated artifacts under `BUILD/` are treated as legacy and disposable.

## Enforcement

- Update `CANON/CONTRACT.md` output roots rule.
- Update `AGENTS.md` mutation and output rules.
- Update `.gitignore` to ignore new artifact roots while keeping their directories tracked.
- Update fixtures so `python CONTRACTS/runner.py` proves pack generation and output location behavior.

## Review triggers

- A need arises for a centralized artifact cache.
- The template introduces a build system that requires a different output convention.
