# Z.2.5 GC Implementation Notes

**Status**: CANONICAL  
**Last updated**: 2026-01-02  
**Scope**: Z.2.5 only

---

## Overview

This document summarizes key implementation decisions for Z.2.5 GC (Garbage Collection for CAS).

---

## Policy B: Empty Roots Behavior

**CRITICAL DECISION**: Policy B (Choice B) is LOCKED.

### The Rule

- **If root enumeration yields ZERO roots**:
  - Default behavior (allow_empty_roots=False): **FAIL-CLOSED, no deletions**
  - Override behavior (allow_empty_roots=True): **Full sweep allowed**

### Rationale

Policy B is conservative and safe:
- Prevents accidental deletion of all CAS storage due to misconfiguration
- Requires explicit intent (allow_empty_roots=True) to perform full sweep
- Aligns with fail-closed philosophy: when in doubt, do nothing

### Implementation

The `allow_empty_roots` parameter is the explicit override switch:

```python
gc_collect(dry_run=False, allow_empty_roots=False)  # Default: fail-closed on empty roots
gc_collect(dry_run=False, allow_empty_roots=True)   # Override: allow full sweep
```

When `allow_empty_roots=False` and roots==0:
- GC returns immediately with error: "POLICY_LOCK: Empty roots detected and allow_empty_roots=False"
- No deletions are performed
- Report contains the error in the `errors` field

When `allow_empty_roots=True` and roots==0:
- GC proceeds with full sweep
- All unreferenced blobs are deleted
- No error is reported

---

## Root Sources

GC enumerates roots from exactly two sources:

### A) Run Record Roots

- **File**: `CAPABILITY/RUNS/RUN_ROOTS.json`
- **Format**: JSON array of CAS hashes (64-char lowercase hex)
- **Missing file**: Not an error (treated as empty list)
- **Malformed file**: Fail-closed with error

### B) Pin File Roots

- **File**: `CAPABILITY/RUNS/GC_PINS.json`
- **Format**: JSON array of CAS hashes (64-char lowercase hex)
- **Missing file**: Not an error (treated as empty list)
- **Malformed file**: Fail-closed with error

### Deduplication

Roots from both sources are deduplicated before traversal.

---

## Two-Phase Model

### Phase 1: MARK (read-only)

1. Enumerate roots from all sources
2. Validate all root hashes strictly
3. If enumeration fails => fail-closed (return error, no deletions)
4. If roots==0 and allow_empty_roots==False => fail-closed (return error, no deletions)
5. Traverse references to build reachable set
6. If traversal fails => fail-closed (return error, no deletions)

**Current traversal**: For Z.2.5, traversal is trivial (roots are the reachable set). Future phases may implement deep traversal if CAS objects contain references to other CAS objects.

### Phase 2: SWEEP

1. Enumerate all CAS blobs deterministically (sorted)
2. Compute candidates = all_blobs - reachable_set
3. If dry_run==True => return report only (no deletions)
4. If dry_run==False:
   - Acquire global GC lock (fail-closed if lock unavailable)
   - Delete candidates in stable sorted order
   - Record outcome for each hash (deleted or skipped with reason)
   - Release lock

---

## Determinism

All operations are deterministic:

- **Root enumeration**: Stable order (JSON array order)
- **Deduplication**: Set operations are deterministic
- **CAS blob enumeration**: Sorted by hash
- **Candidate list**: Sorted by hash
- **Deletion order**: Sorted by hash
- **Report fields**: Stable structure and ordering
- **CAS snapshot hash**: Hash of sorted blob list

Same inputs => same outputs (including report content).

---

## Fail-Closed Semantics

GC fails-closed (performs zero deletions) in these cases:

1. Root enumeration fails (malformed JSON, invalid hash format)
2. Zero roots and allow_empty_roots==False (Policy B)
3. Reference traversal fails
4. GC lock cannot be acquired
5. CAS enumeration fails

In all fail-closed cases:
- No deletions are performed
- Error is recorded in report `errors` field
- Report is still returned (with zero deletions)

---

## Locking

GC uses a global threading lock (`_gc_lock`) to ensure single-instance execution.

- Lock is acquired at the start of SWEEP phase (non-blocking)
- If lock cannot be acquired => fail-closed with error
- Lock is released after SWEEP completes (success or failure)

**Rationale**: Prevents concurrent GC runs from interfering with each other or with concurrent CAS writes.

---

## Report Structure

The GC report is a dict with the following fields (all required):

```python
{
    'mode': 'dry_run' | 'apply',
    'allow_empty_roots': bool,
    'root_sources': list[str],           # Human-readable source identifiers
    'roots_count': int,
    'reachable_hashes_count': int,
    'candidate_hashes_count': int,
    'deleted_hashes': list[str],         # Sorted
    'skipped_hashes': list[dict],        # [{'hash': str, 'reason': str}, ...], sorted
    'errors': list[str],                 # Empty on success
    'cas_snapshot_hash': str             # Deterministic hash of CAS state
}
```

Report is deterministic and auditable.

---

## Test Isolation

All tests run in isolated temp storage:
- Isolated CAS root (tmp_path / "CAS" / "storage")
- Isolated RUNS directory (tmp_path / "RUNS")
- Tests never touch real repo CAS storage or RUNS directory

This ensures:
- Tests are safe to run (no accidental deletions)
- Tests are deterministic (no interference from real state)
- Tests are reproducible

---

## Scope Constraints

Z.2.5 implements ONLY the following:

✅ **Implemented**:
- Two-phase mark-and-sweep GC
- Policy B (empty roots fail-closed with override)
- Root enumeration from RUN_ROOTS.json and GC_PINS.json
- Deterministic behavior
- Fail-closed semantics
- Dry-run mode
- Comprehensive reporting
- Single-instance locking

❌ **NOT implemented** (require future roadmap items):
- GC scheduling or automation
- Background threads or async GC
- LRU/LFU eviction
- Time-based eviction (TTL)
- Reference counting
- Provenance graphs
- Complex pin semantics
- Deep reference traversal (future)

---

## Exit Criteria

Z.2.5 is complete when:

1. ✅ All GC tests pass (GC-01 through GC-18)
2. ✅ GC-12 (Policy B) passes both scenarios
3. ✅ All existing repo tests pass (no regressions)
4. ✅ Documentation is complete (invariants + test matrix + this file)
5. ✅ Diff is limited to:
   - CAPABILITY/GC/*
   - CAPABILITY/TESTBENCH/gc/*
   - NAVIGATION/INVARIANTS/Z2_5_GC_INVARIANTS.md
   - NAVIGATION/INVARIANTS/Z2_5_GC_TEST_MATRIX.md
   - CAPABILITY/GC/IMPLEMENTATION.md

---

## Future Work (Out of Scope for Z.2.5)

The following are explicitly deferred to future roadmap items:

- **Z.2.6+**: Deep reference traversal (if CAS objects contain refs to other CAS objects)
- **Z.2.7+**: GC scheduling and automation
- **Z.2.8+**: Advanced lifecycle policies (LRU, TTL, etc.)
- **Z.2.9+**: Provenance graphs and attestation

---
