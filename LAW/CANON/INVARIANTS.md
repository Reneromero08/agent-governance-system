<!-- CONTENT_HASH: 74dcf864da760c316aa031d5419d62d7a4ba9e7978ac77aced5bea9c9091fcfd -->

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

## Recovery: Invariant Violation Detection and Remediation

### Where receipts live

Invariant violations are detected and reported in the following locations:

- **LAW/CONTRACTS/_runs/audit_logs/** - Root audit and invariant check results
  - `root_audit.jsonl` - Output from `CAPABILITY/AUDIT/root_audit.py`
  - `canon_audit.jsonl` - Canon compliance and invariant validation
- **LAW/CONTRACTS/_runs/_tmp/** - Temporary receipts for skill and task execution
  - `prompts/*/receipt.json` - Prompt execution receipts (inputs, outputs, hashes)
  - `skills/*/receipt.json` - Skill execution receipts
- **LAW/CONTEXT/decisions/** - Architecture Decision Records for invariant changes
  - ADRs document the rationale for invariant modifications or supersessions

### How to re-run verification

To verify invariant compliance and detect violations:

```bash
# Verify all fixtures pass (invariant INV-004)
python LAW/CONTRACTS/runner.py

# Run root audit (verifies INV-006 compliance)
python CAPABILITY/AUDIT/root_audit.py --verbose

# Run critic to check canon compliance (INV-009, INV-011)
python CAPABILITY/TOOLS/governance/critic.py

# Check canon file line counts and rule counts (INV-009)
python -c "
from pathlib import Path
for f in Path('LAW/CANON').glob('*.md'):
    lines = len(f.read_text().splitlines())
    print(f'{f.name}: {lines} lines')
"
```

### What to delete vs never delete

**Safe to delete (temporary, not source of truth):**
- `LAW/CONTRACTS/_runs/_tmp/` - All subdirectories are disposable scratch space
- `_tmp/` directories anywhere in the repo - Never use as verification source
- Build artifacts in `BUILD/` - User outputs, disposable at any time
- Temporary CAS objects that are unrooted - GC will delete these safely

**Never delete (protected, require ceremony):**
- Files under `LAW/CANON/` - Superseded rules must be moved to `LAW/CANON/archive/`, not deleted
- Rooted CAS objects - Only GC can delete these, and only if unrooted
- `RUN_ROOTS.json` and `GC_PINS.json` - Modify only with explicit ceremony; never delete
- Git history - Preserved for audit; use git bisect for recovery
- ADR records - Append-first; existing records cannot be edited without ceremony

**Recovery procedures:**
- If canon file is accidentally deleted: Restore from git history (`git checkout HEAD~ -- LAW/CANON/file.md`)
- If rooted CAS object is lost: Recover from backup or re-run operation that created it
- If receipt is missing: Re-run skill/task to regenerate; deterministic output will match original receipt
- If invariant violation detected: Check `LAW/CONTEXT/decisions/` for recent ADRs that may explain the change
