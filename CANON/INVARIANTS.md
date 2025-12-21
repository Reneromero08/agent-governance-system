# Invariants

This file lists decisions that are considered invariant.  Changing an invariant requires an exceptional process (including a major version bump) because it may break compatibility with existing content.

## List of invariants

- **[INV-001] Repository structure** - The top-level directory layout (`CANON`, `CONTEXT`, `MAPS`, `SKILLS`, `CONTRACTS`, `MEMORY`, `CORTEX`, `TOOLS`) is stable. New directories may be added, but existing ones cannot be removed or renamed without a major version bump and migration plan.
- **[INV-002] Token grammar** - The set of tokens defined in `CANON/GLOSSARY.md` constitutes a stable interface. Tokens may be added but not removed or changed without deprecation.
- **[INV-003] No raw path access** - Skills may not navigate the filesystem directly. They must query the cortex (`CORTEX/query.py`) to find files.
- **[INV-004] Fixtures gate merges** - No code or rule change may be accepted if any fixture fails. Fixtures define the legal behavior.
- **[INV-005] Determinism** - Given the same inputs and canon, the system must produce the same outputs. Timestamps, random values, and external state must be injected explicitly or omitted.
- **[INV-006] Output roots** - System-generated artifacts must be written only to `CONTRACTS/_runs/`, `CORTEX/_generated/`, or `MEMORY/LLM_PACKER/_packs/`. `BUILD/` is reserved for user outputs.
- **[INV-007] Change ceremony** - Any behavior change must add/update fixtures, update the changelog, and occur in the same commit. Partial changes are not valid.
- **[INV-008] Cortex builder exception** - Cortex builders (`CORTEX/*.build.py`) may scan the filesystem directly. All other skills and agents must query via `CORTEX/query.py`.

## Changing invariants

To modify an invariant:

1. File an ADR under `CONTEXT/decisions/` explaining why the change is necessary and the risks.
2. Propose a migration strategy for affected files and skills.
3. Update the version in `CANON/VERSIONING.md` (major version bump).
4. Provide fixtures that demonstrate compatibility with both the old and new behavior during the migration period.
