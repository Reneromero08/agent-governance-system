# ADR-008: Commit ceremony approvals and confirmations

**Status:** Accepted

**Date:** 2025-12-21

**Review date:** (optional)

**Confidence:** High

**Impact:** Medium

**Tags:** [governance, commit, release]

## Context

Users sometimes provide a single explicit directive such as "commit, push, and release".
Others respond with short confirmations like "go on" after the ceremony prompt. The
prior ceremony treated both as invalid, creating unnecessary friction even when the
user intent was clear and checks were complete.

## Decision

Treat explicit composite directives that include the verbs "commit", "push", and
"release" (for example, "commit, push, and release") as explicit approval for each
action listed in that request.

When the agent has completed the ceremony steps (checks run, staged files listed)
and the only remaining actions are the explicitly requested git/release operations,
short confirmations such as "go on" are treated as approval for those listed actions.

This does not weaken the anti-chaining rule: approval applies only to the current task.

## Alternatives considered

- Keep the existing ceremony prompt even for explicit composite directives.
- Allow short confirmations in all contexts (too permissive).

## Rationale

Composite directives and ceremony confirmations are unambiguous in context and
preserve user intent while removing redundant prompts. Restricting these to explicit
action lists and the ceremony context prevents implicit approvals.

## Consequences

- Slightly shorter commit ceremony in cases where the user already provided explicit
  composite approval.
- Reduced friction at the final approval step for short confirmations.
- The agent must still run checks and list staged files before acting.

## Enforcement

- Update `CANON/CONTRACT.md` commit ceremony rule.
- Update `AGENTS.md` and `CONTEXT/preferences/STYLE-001-commit-ceremony.md`.
- Add a governance fixture documenting explicit approvals and confirmations.

## Review triggers

- Release tooling changes (e.g., new publish commands).
- If confirmations cause confusion or accidental pushes.
