# CAPABILITY/AUDIT Implementation

**Component**: Z.2.6 ROOT AUDIT
**Version**: Z.2.6.0
**Status**: Active

---

## Overview

The ROOT AUDIT tool provides deterministic, fail-closed verification of root completeness and GC safety. It is designed as a pre-packer integration gate to verify that all required artifacts are reachable from declared roots before production GC operations.

---

## Public API

### `root_audit(*, output_hashes_record: str | None = None, dry_run: bool = True) -> dict`

**Purpose**: Audit root completeness and GC safety.

**Parameters**:
- `output_hashes_record`: Optional CAS hash of OUTPUT_HASHES record
  - `None` → Mode A (general safety audit)
  - `<hash>` → Mode B (run completeness check)
- `dry_run`: Always `True` for audit (no deletions; kept for interface symmetry)

**Returns**: Deterministic receipt dict (see Z2_6_ROOT_AUDIT_INVARIANTS.md)

**Raises**: None (fail-closed via receipt verdict, not exceptions)

---

## Implementation Strategy

### Code Reuse from Z.2.5 GC

To ensure equivalence and minimize divergence, the audit implementation reuses core GC logic:

1. **Root enumeration**: Same validation and file parsing as GC
2. **Reachability traversal**: Identical semantics (currently trivial; roots = reachable set)
3. **CAS blob enumeration**: Same deterministic directory walking
4. **CAS snapshot hash**: Same canonical encoding (sorted blob list)

**Key difference**: Audit uses fail-closed error reporting (errors list in receipt) rather than raising exceptions, to enable complete diagnostics in a single run.

### Fail-Closed Design

Every potential error condition is captured and reported:
- Invalid root format → error list, verdict FAIL
- Empty roots → explicit error, verdict FAIL (no override)
- Missing OUTPUT_HASHES record → error list, verdict FAIL
- Unreachable required outputs → reported in `required_unreachable`, verdict FAIL

The audit never silently succeeds when conditions are unsafe.

### Determinism Guarantees

All outputs are deterministic:
- Root sources: file hashes computed for auditability
- Roots: deduplicated via set union, sorted for display
- Reachable set: traversal order is stable (sorted roots input)
- Error list: sorted before returning
- required_missing / required_unreachable: sorted
- CAS snapshot hash: computed from sorted blob list

Tests verify byte-for-byte receipt equality for identical inputs.

---

## Phases

The audit executes in five phases:

### 1. Root Enumeration
- Read RUN_ROOTS.json and GC_PINS.json
- Validate each hash (64 lowercase hex)
- Compute file content hashes for auditability
- Deduplicate roots (set union)
- Fail-closed if any errors or if roots_count == 0

### 2. Reachability Computation
- Call `_traverse_references(roots, cas_root)`
- Currently: trivial (reachable = roots)
- Future: when GC implements deep traversal, this MUST match
- Fail-closed if traversal raises exception

### 3. CAS Snapshot
- Enumerate all CAS blobs deterministically
- Compute snapshot hash (for comparison and audit trail)
- Fail-closed if enumeration fails

### 4. Mode B - Required Outputs Verification (if enabled)
- Load OUTPUT_HASHES record from CAS
- Validate each required hash
- Check existence in CAS
- Check reachability from roots
- Report missing and unreachable in separate lists

### 5. Verdict Computation
- Mode A: PASS if errors == [] and roots_count > 0
- Mode B: Mode A + required_missing == [] + required_unreachable == []
- All error conditions captured in sorted error list

---

## Testing

See `CAPABILITY/TESTBENCH/audit/test_root_audit.py` for comprehensive test coverage.

**Required tests**:
- A-01: Determinism (identical inputs → identical receipt bytes)
- A-02: Empty roots fail-closed
- A-03: Invalid root format fail-closed
- A-04: Reachable count matches fixture
- B-01: Valid OUTPUT_HASHES, all rooted → PASS
- B-02: Unrooted ref → FAIL (required_unreachable)
- B-03: Invalid ref format → FAIL
- B-04: Missing OUTPUT_HASHES record → FAIL
- B-05: Corrupted blob → FAIL

All tests use isolated temp storage (never touch real CAS).

---

## Future Work

### Deep Traversal Integration

When Z.2.5 GC implements deep traversal (loading OUTPUT_HASHES records and marking referenced blobs as reachable):
1. Extract `_traverse_references()` to a shared helper module
2. Import and reuse in both GC and audit
3. Add tests verifying GC/audit equivalence

### Blob Integrity Checks

Optional: verify each blob's hash matches its content (re-hash and compare).
- Add `corrupted_blobs` field to receipt
- Report in errors list
- Fail-closed on any corruption

### Receipt Persistence

If repo adopts receipt convention:
- Write canonical JSON to `CAPABILITY/AUDIT/receipts/<content-hash>.json`
- Use content hash (not timestamp) for deterministic filenames

---

## Dependencies

- `CAPABILITY/CAS/cas.py`: `_CAS_ROOT`, `_get_object_path`
- `CAPABILITY/RUNS/records.py`: `load_output_hashes()`, exception types
- `CAPABILITY/RUNS/RUN_ROOTS.json`: Root source (optional)
- `CAPABILITY/RUNS/GC_PINS.json`: Root source (optional)

---

## Invariants

See [Z2_6_ROOT_AUDIT_INVARIANTS.md](../../NAVIGATION/INVARIANTS/Z2_6_ROOT_AUDIT_INVARIANTS.md) for complete invariant specification.

---

## Changelog

- **Z.2.6.0** (2026-01-02): Initial implementation
  - Mode A: General root safety audit
  - Mode B: Run completeness check
  - Deterministic receipts
  - Fail-closed validation
  - Full test coverage
