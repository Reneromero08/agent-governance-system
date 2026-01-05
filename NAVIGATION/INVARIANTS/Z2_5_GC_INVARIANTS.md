<!-- CONTENT_HASH: 30d4f79adb0d71cd91e80fe8fcd6723eefb05760b2b1142056bb1b7c06f99265 -->

# Z.2.5 GC Invariants (Unreferenced Blobs Cleanup Policy)

Status: CANONICAL (Z.2.5)
Last updated: 2026-01-02
Scope: CAS only, policy-driven garbage collection. No background behavior.

## POLICY LOCK (Choice B)

**CRITICAL**: The following policy is LOCKED and MUST be enforced:

- **Empty roots => fail-closed; no deletions**
  - If root enumeration yields ZERO roots, GC MUST FAIL-CLOSED and perform ZERO deletions
  - This is the default behavior (allow_empty_roots=False)

- **Full sweep requires explicit override flag**
  - A full sweep is ONLY permitted when an explicit override switch is provided
  - The override is: `allow_empty_roots=True`
  - When allow_empty_roots=True and roots==0, GC may delete everything (subject to lock + other validations)

## Purpose
Define a mechanically verifiable, conservative GC strategy for CAS blobs that are not reachable from declared roots.

## Non-negotiable invariants

### I. Identity and immutability
1. CAS identity is SHA-256 over raw bytes only.
2. CAS blobs are immutable and write-once.
3. GC must never rewrite blob contents. GC may only delete blobs.

### II. Root-driven reachability
4. A blob is reachable if it is referenced (directly or transitively) from at least one declared root.
5. A blob is garbage only if it is not reachable from any declared root.
6. GC must not infer liveness from timestamps, file mtimes, access patterns, or size.
7. Roots must be enumerable and auditable.

### III. Allowed root sources (explicit)
GC roots may only come from sources that are deterministic to enumerate:
- Run records: hashes of TASK_SPEC, STATUS, and OUTPUT_HASHES artifacts.
- Pinned roots: explicit pin list file (example path `CAPABILITY/RUNS/GC_PINS.json` or similar).
- Declared artifact refs: if a canonical registry exists that stores artifact refs, it may be included only if it is deterministic and audited.

Forbidden root sources:
- "Anything in the repo"
- "Recently used"
- "Files modified in last N days"
- "Anything in storage directory"

### IV. Traversal rules
8. Mark phase must build a reachable set by traversing from roots.
9. Traversal must be bounded and deterministic:
   - stable ordering of roots
   - stable ordering of visited nodes
10. Traversal must validate hash format strictly (`sha256:<64hex>` or bare `<64hex>` if internal).
11. Invalid refs cause fail-closed abort (no deletions).

### V. Two-phase execution model
12. GC is a two-phase operation: Mark then Sweep.
13. Mark phase must be read-only.
14. Sweep phase may delete only blobs not in the reachable set.
15. GC must support a dry-run mode that performs Mark and produces the same deletion set without deleting.

### VI. Fail-closed and atomicity
16. If enumeration of roots fails, GC must abort without deleting anything.
17. If traversal fails (invalid ref, corruption detected, unreadable root source), GC must abort without deleting anything.
18. Sweep must be atomic per blob:
   - each delete is either fully applied or not applied
   - partial failures must be recorded
19. If GC cannot guarantee correctness, it must do nothing and emit an error report.

### VII. Auditing and receipts
20. Every GC run must emit a deterministic report (receipt) including:
   - inputs: root source identifiers, pin file hash (if used)
   - counts: roots, reachable, candidates, deleted, skipped
   - lists: deleted hashes (deterministic order), skipped hashes with reasons
   - mode: dry-run or apply
   - CAS snapshot identifier (see below)
21. Receipts must be canonical JSON (or canonical line-delimited JSON) with stable key ordering.

CAS snapshot identifier (recommended):
- A deterministic listing hash of current CAS objects (for example: hash of sorted blob hashes).
- Used only for auditing and reproducibility, not for liveness inference.

### VIII. Concurrency and safety
22. GC must run only when explicitly invoked.
23. GC must not run concurrently with writes unless protected by a strict lock (recommended: global CAS GC lock).
24. If a lock cannot be acquired, GC must abort.

### IX. Explicit non-features (forbidden in Z.2.5)
- Automatic scheduling, background threads
- LRU, LFU, time-based eviction
- Reference counting
- Implicit provenance graphs
- Deleting roots or pin sources
- Partial "best effort" cleanup

## Minimal API surface (recommended)
One entry point:
- `gc_collect(dry_run: bool = True, allow_empty_roots: bool = False) -> GCReport`

**Parameters**:
- `dry_run`: If True, do not delete anything (report only). Default: True
- `allow_empty_roots`: If True, allow full sweep when roots==0. Default: False
  - This is the explicit override for Policy B
  - If False and roots==0, fail-closed: delete nothing and report error

GCReport should include:
- `reachable_hashes`
- `delete_candidates`
- `deleted_hashes`
- `skipped_hashes` with reasons
- `root_sources`
- `mode`
- `cas_snapshot_hash`
- `errors` (empty on success)

## Acceptance conditions
Z.2.5 is complete only when:
- Tests prove fail-closed behavior on any root or traversal failure.
- Dry-run produces identical deletion set as apply mode.
- Receipts are deterministic across identical inputs.

---

## Recovery: GC Invariant Violations

### Where receipts live

GC receipts are typically transient or emitted to stdout, but should be captured in:
- **LAW/CONTRACTS/_runs/audit_logs/** - If integrated into pipeline
- **Runtime Standard Output** - For ad-hoc manual runs

### How to re-run verification

To verify GC safety without modification (Dry Run):

```bash
# Run GC in dry-run mode (default)
# This will report what WOULD be deleted
python CAPABILITY/GC/gc.py --dry-run
```

To verify root completeness (Prerequisite):

```bash
# Verify roots are readable
python CAPABILITY/AUDIT/root_audit.py
```

### What to delete vs never delete

**Safe to delete (via GC only):**
- **Unreachable blobs**: Blobs not referenced by `RUN_ROOTS.json` or `GC_PINS.json`
  - **Do NOT delete manually**. Use `gc.py` (without dry-run) to safely remove.
  - Manual deletion risks race conditions or incomplete cleanup.

**Never delete (protected by invariants):**
- **RUN_ROOTS.json**: The primary root anchored list.
- **GC_PINS.json**: Identify manual pins.
- **Any blob in `storage/` manually**: Brittle and dangerous. Always use the tool.

**Recovery procedures:**
- **Roots file corrupted**: Restore from git history.
- **GC Lock stuck**: If a previous run crashed, delete the lock file `CAPABILITY/GC/gc.lock` ONLY after verifying no python processes are running.
- **Accidental deletion**: If a blob was deleted but is now needed, re-generate it from the source (CAS is content-addressed; re-adding identical data restores the object).
