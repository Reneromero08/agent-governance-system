# ADR-017: Skill Formalization and Validation

**Status:** Accepted
**Date:** 2025-12-23
**Confidence:** High
**Impact:** Medium
**Tags:** [skills, governance, validation]

## Problem

Skill structure is inconsistent across the codebase:
- Some skills have `validate.py`, others don't
- No unified contract for what "complete skill" means
- Skill validation is not enforced in CI

This inconsistency makes it harder for agents to understand what a skill should contain and makes CI validation incomplete.

## Decision

All skills must follow a strict contract with four required components:

1. **SKILL.md** - Manifest with metadata and documentation (required, enforced by schema)
2. **run.py** - Implementation script that executes the skill logic (required)
3. **validate.py** - Output validator that compares actual vs expected (required)
4. **fixtures/** - Test cases with input.json and expected.json (required, enforced by CI)

Each skill's `validate.py` must:
- Accept two arguments: `<actual.json>` and `<expected.json>`
- Compare outputs using JSON equality (or domain-specific logic if documented)
- Return exit code 0 on pass, 1 on failure
- Print "Validation passed" on success and detailed diffs on failure

## Rationale

- **Completeness**: Skills are reusable units; incomplete structure leads to runtime failures.
- **Consistency**: Uniform contract makes it easier for agents to understand and extend skills.
- **Testability**: Every skill must be fixture-verified before CI accepts it.
- **Auditability**: Validation output is logged to `CONTRACTS/_runs/skill_logs/` for debugging.

## Consequences

- All existing skills now have complete structure (12/12 ✓).
- Agents cannot add skills without fixtures and validation.
- Runner and critic can enforce this uniformly.
- CI will fail if any skill is missing required files.

## Enforcement

- `CONTRACTS/runner.py` executes all skill fixtures and validates outputs.
- Critic can be extended to check for missing validate.py (future enhancement).
- SKILL.md schema validation already enforced.

## Related

- ADR-015: Logging output roots (affects skill log locations)
- ADR-016: Context edit authority (affects skill documentation)
