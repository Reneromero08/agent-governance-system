# ADR-014: Cortex section index and CLI primitives

**Status:** Accepted

**Date:** 2025-12-23

**Review date:** (optional)

**Confidence:** High

**Impact:** Medium

**Tags:** [cortex, navigation, tooling]

## Context

Agents need deterministic, low-context navigation primitives that:
- reference content without raw path discovery;
- support stable citations;
- work in clean checkouts and are reproducible.

The existing Cortex provides a file-level index (SQLite + JSON snapshot), but lacks a section-level index and a small CLI for reading sections by stable identifiers.

## Decision

Add a generated section-level index and a minimal CLI tool:

- Generate `CORTEX/_generated/SECTION_INDEX.json` during the Cortex build.
  - Each record includes: `section_id`, `path`, `heading`, `start_line`, `end_line`, `hash`.
  - Line numbers are 1-based and inclusive.
  - `hash` is `sha256` over the normalized section slice (newlines normalized to `\\n`).

- Provide `TOOLS/cortex.py` commands:
  - `read <section_id>` prints the exact section slice.
  - `resolve <section_id>` prints deterministic JSON metadata for provenance/citations.
  - `search "<query>"` returns `section_id` + `heading` only (no raw paths).

This is an implementation primitive; it does not add enforcement or CI gating by itself.

If `CORTEX_RUN_ID` is set, Cortex CLI commands append JSONL provenance events to `CONTRACTS/_runs/<CORTEX_RUN_ID>/events.jsonl`.

## Alternatives considered

- Store section data inside the SQLite DB only (harder to use from simple tools and fixtures).
- Use file paths directly as citations (violates the preference for governed navigation).
- Embed the section index into `cortex.json` (mixes generated artifacts with legacy snapshot needs).

## Rationale

Keeping `SECTION_INDEX.json` as a generated artifact under `CORTEX/_generated/` preserves determinism and avoids committing large derived artifacts while still enabling stable IDs and citations.

The CLI commands are intentionally small so agents can use them without loading many files into context.

## Consequences

- Cortex build time increases slightly due to markdown section parsing.
- Tools can reference content via `section_id:start_line-end_line` with a stable hash.
- Fixtures can validate parsing edge cases (e.g., fenced code blocks) without coupling to CI enforcement.

## Enforcement

- None in this ADR. Enforcement/CI gates (if desired) should be handled as a separate change.

## Review triggers

- If section indexing needs to expand beyond headings (tables, code symbols).
- If `section_id` stability needs a migration strategy (e.g., heading rename semantics).
