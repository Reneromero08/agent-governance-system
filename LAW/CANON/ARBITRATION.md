# Canon Arbitration

This document defines how to resolve conflicts when canon rules contradict each other. It is part of the CANON layer and has authority immediately below `CONTRACT.md`.

## When This Applies

A **canon conflict** exists when:
- Two rules in CANON require mutually exclusive actions.
- A rule in CANON contradicts an invariant in INVARIANTS.md.
- Applying one rule would violate another rule.

This does NOT apply to:
- Ambiguity (unclear rules) — ask for clarification instead.
- Implementation difficulty — implementation must follow canon, not the reverse.
- Preferences conflicting with canon — canon wins, preferences are suggestions.

## Resolution Order

When a conflict is detected, resolve using this priority (highest first):

1. **CONTRACT.md** — The supreme authority. If a rule elsewhere conflicts with CONTRACT, CONTRACT wins.
2. **INVARIANTS.md** — Locked decisions. Invariants outrank other canon files.
3. **VERSIONING.md** — Version constraints and compatibility rules.
4. **Specificity** — A more specific rule outranks a more general rule.
5. **Recency** — If two rules have equal specificity, the more recently added rule wins (check git history or ADR dates).

If priority is still unclear after applying these criteria, **stop and escalate**.

## Escalation Protocol

If an agent cannot resolve a conflict using the above order:

1. **Stop all work** related to the conflicting rules.
2. **Document the conflict** by creating a record in `CONTEXT/open/OPEN-xxx-conflict-*.md` that describes:
   - The two (or more) conflicting rules
   - The specific action that triggers the conflict
   - Why the resolution order above doesn't resolve it
3. **Notify the user** with the conflict summary.
4. **Wait for human ruling** before proceeding.

Do not guess. Do not pick arbitrarily. Do not proceed without resolution.

## Recording Resolutions

Once a conflict is resolved (by human ruling or clear priority):

1. **Update the relevant canon file(s)** to eliminate the conflict (if possible).
2. **Create an ADR** under `CONTEXT/decisions/` documenting:
   - The conflict that existed
   - How it was resolved
   - The rationale for the resolution
3. **Add a "Conflicts" section** to the ADR referencing the superseded interpretation.

## Prevention

To prevent future conflicts:

- **Before adding a new rule**, search existing canon for related rules.
- **Use explicit scope** in rules (e.g., "This applies only to X" or "Except when Y").
- **Reference other rules** when a rule depends on or modifies another.
- **Use the ADR process** for significant rule additions — it forces consideration of existing rules.

## Examples

### Example 1: Specificity Resolution

- Rule A (CONTRACT.md): "All generated files must be written to designated output roots."
- Rule B (SKILLS/foo/SKILL.md): "This skill writes to `BUILD/temp/` for intermediate processing."

**Resolution:** Rule A is in CONTRACT (higher authority). Rule B must be updated to use a designated output root, or the skill must request an exception via ADR.

### Example 2: Recency Resolution

- Rule A (added 2025-01-01): "Pack files must use `.md` extension."
- Rule B (added 2025-06-15): "Pack files must use `.pack.md` extension for clarity."

**Resolution:** Both rules have equal authority (same file, same specificity). Rule B is more recent, so `.pack.md` wins. Rule A should be updated or removed to eliminate the conflict.

### Example 3: Escalation Required

- Rule A: "Agents must not modify CANON without explicit instruction."
- Rule B: "Agents must update CANON when behavior changes."

**Resolution:** These rules conflict when an agent changes behavior. The resolution depends on context (was the behavior change instructed? does it require a canon update?). This requires human ruling per the escalation protocol.

## Status

**Active**
Added: 2025-12-21
