<!-- CONTENT_HASH: 4db81512279856a3fcd372c64f7d0bff9670387b7e364c7c5027da5b807e4f00 -->

# Invariants

This file lists decisions that are considered invariant.  Changing an invariant requires an exceptional process (including a major version bump) because it may break compatibility with existing content.

## List of invariants

- **[INV-001] Repository structure** - The top-level directory layout (`LAW`, `CAPABILITY`, `NAVIGATION`, `MEMORY`, `THOUGHT`, `INBOX`) is stable. New directories may be added, but existing ones cannot be removed or renamed without a major version bump and migration plan.
- **[INV-002] Token grammar** - The set of tokens defined in `LAW/CANON/GLOSSARY.md` constitutes a stable interface. Tokens may be added but not removed or changed without deprecation.
- **[INV-003] No raw path access** - Skills may not navigate the filesystem directly. They must query the cortex (`NAVIGATION/CORTEX/semantic/vector_indexer.py` or equivalent API) to find files.
- **[INV-004] Fixtures gate merges** - No code or rule change may be accepted if any fixture fails. Fixtures define the legal behavior.
- **[INV-005] Determinism** - Given the same inputs and canon, the system must produce the same outputs. Timestamps, random values, and external state must be injected explicitly or omitted.
- **[INV-006] Output roots** - System-generated artifacts must be written only to `LAW/CONTRACTS/_runs/`, `NAVIGATION/CORTEX/_generated/`, or `MEMORY/LLM_PACKER/_packs/`. `BUILD/` is reserved for user outputs.
- **[INV-007] Change ceremony** - Any behavior change must add/update fixtures, update the changelog, and occur in the same commit. Partial changes are not valid.
- **[INV-008] Cortex builder exception** - Cortex builders (`NAVIGATION/CORTEX/semantic/*.py`) may scan the filesystem directly. All other skills and agents must query via semantic search or index lookups.
- **[INV-009] Canon readability** - Each file in `LAW/CANON/` must remain readable and focused:
  - Maximum 300 lines per file (excluding examples and templates).
  - Maximum 15 rules per file.
  - If a file exceeds these limits, it must be split via ADR.
- **[INV-010] Canon archiving** - Rules that are superseded or no longer applicable must be:
  - Moved to `LAW/CANON/archive/` (not deleted).
  - Referenced in the superseding rule or ADR.
  - Preserved in git history for audit.
- **[INV-011] Schema Compliance** - All Law-Like files (ADRs, Skills, Style Preferences) must be valid against their respective JSON Schemas in `LAW/SCHEMAS/governance/`.
- **[INV-012] Visible Execution** - Agents must not spawn hidden or external terminal windows (e.g., `start wt`, `xterm`). All interactive or long-running execution must occur via the Antigravity Bridge (invariant infrastructure) or within the current process. The Bridge is considered "Always On".
- **[INV-013] Declared Truth** - Every system-generated artifact MUST be declared in a hash manifest (`OUTPUT_HASHES.json`). If it is not hashed, it is not truth.
- **[INV-014] Disposable Space** - Files under `_tmp/` directories (Catalytic domains) are strictly for scratch-work. They must never be used as a source of truth for verification.
- **[INV-015] Narrative Independence** - Verification Success (`STATUS: success`) is bound only to artifact integrity, not to execution logs, reasoning traces, or chat history.

## Changing invariants

To modify an invariant:

1. File an ADR under `LAW/CONTEXT/decisions/` explaining why the change is necessary and the risks.
2. Propose a migration strategy for affected files and skills.
3. Update the version in `LAW/CANON/VERSIONING.md` (major version bump).
4. Provide fixtures that demonstrate compatibility with both the old and new behavior during the migration period.
