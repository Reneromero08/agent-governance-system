# ADR-008: Composite approval for commit, push, and release

**Status:** Accepted

**Date:** 2025-12-21

**Review date:** (optional)

**Confidence:** High

**Impact:** Medium

**Tags:** [governance, commit, release]

## Context

Users sometimes provide a single explicit directive such as "commit, push, and release".
The prior ceremony required the agent to re-prompt even when the directive was already
clear, which created unnecessary friction.

## Decision

Treat explicit composite directives that include the verbs "commit", "push", and
"release" (for example, "commit, push, and release") as explicit approval for each
action listed in that request.

This does not weaken the anti-chaining rule: approval applies only to the current task.

## Alternatives considered

- Keep the existing ceremony prompt even for explicit composite directives.
- Allow vague confirmations ("go ahead") to authorize commits.

## Rationale

Composite directives are unambiguous and preserve user intent while removing redundant
prompts. Restricting this to explicit action lists prevents implicit approvals.

## Consequences

- Slightly shorter commit ceremony in cases where the user already provided explicit
  composite approval.
- The agent must still run checks and list staged files before acting.

## Enforcement

- Update `CANON/CONTRACT.md` commit ceremony rule.
- Update `AGENTS.md` and `CONTEXT/preferences/STYLE-001-commit-ceremony.md`.
- Add a governance fixture documenting the explicit composite approval phrase.

## Review triggers

- Release tooling changes (e.g., new publish commands).
- If composite directives cause confusion or accidental pushes.
