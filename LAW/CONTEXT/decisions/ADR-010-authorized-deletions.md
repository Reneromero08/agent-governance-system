# ADR-010: Authorized deletions with confirmation

**Status:** Accepted

**Date:** 2025-12-21

**Review date:** (optional)

**Confidence:** High

**Impact:** Medium

**Tags:** [governance, deletion]

## Context

The prior rules prohibited deleting authored content. Users need the ability to
explicitly delete files (for example, redundant ADRs) without unnecessary friction,
while preserving safeguards against accidental loss.

## Decision

Allow deletion of authored content only when the user explicitly instructs the
deletion and confirms it when asked. CANON rules remain governed by INV-010, so
superseded canon rules must be archived instead of deleted.

## Alternatives considered

- Keep the blanket deletion prohibition.
- Allow deletions without confirmation (too risky).

## Rationale

Explicit instruction plus confirmation preserves safety while honoring user intent.
Separating CANON archiving requirements avoids violating invariants.

## Consequences

- Agents can delete non-canon authored content when explicitly instructed and confirmed.
- CANON rules still require archiving, not deletion.

## Enforcement

- Update `CANON/CONTRACT.md` and `AGENTS.md` to allow deletions with confirmation.
- Add a governance fixture documenting the deletion approval requirement.

## Review triggers

- If deletion confirmations lead to unintended loss.
