# ADR-011: Master Override Directive

**Status:** Accepted
**Date:** 2025-12-23
**Confidence:** High
**Impact:** Critical
**Tags:** [governance, override, interface]

## Context

During active system design, the Sovereign may need to explicitly bypass repository governance rules (e.g., navigation constraints) without repeatedly negotiating workarounds.

`CANON/AGREEMENT.md` states that direct human commands are the absolute state of truth **when delivered through an authorized Governance Interface**, but the system previously lacked a formal interface for that authorization.

## Decision

Introduce a single Sovereign directive token:

- `MASTER_OVERRIDE`

When a user prompt contains `MASTER_OVERRIDE`, the agent is authorized to bypass **any** AGS repository rule for **that prompt only**, including invariants and procedural gates.

## Requirements

1. **Per-prompt scope**
   - Authorization applies only to the prompt that includes `MASTER_OVERRIDE`.
   - It does not persist across subsequent prompts.

2. **Mandatory audit logging**
   - Every use of `MASTER_OVERRIDE` must be logged under:
     - `CONTRACTS/_runs/override_logs/master_override.jsonl`
   - Logging must not require edits outside allowed artifact roots.

3. **Log access is gated**
   - Override logs must not be read, quoted, or summarized unless the user prompt also includes `MASTER_OVERRIDE`.
   - Agents should avoid loading override logs into context unless explicitly asked.

## Alternatives considered

- Multiple scoped override tokens (per-invariant, per-module).
- Time-window based overrides.
- Environment-variable based overrides.

## Rationale

This establishes a clear, explicit, low-friction mechanism that aligns operational behavior with Sovereign authority while retaining auditability.

## Consequences

- Agents can perform otherwise-forbidden actions when explicitly authorized.
- Audit logs become a privileged artifact that should not be routinely loaded.

## Enforcement

- Update CANON and agent guidance to define the interface and required behavior.
- Provide a dedicated `master-override` skill for logging and gated log access.
- Add fixtures ensuring the skill and governance docs remain consistent.

## Review triggers

- Any incident caused by misuse of `MASTER_OVERRIDE`.
- Expansion of AGS to remote/multi-tenant MCP transports.
