# Z.2.6: ROOT AUDIT INVARIANTS

**Status**: Active
**Version**: Z.2.6.0
**Component**: CAPABILITY/AUDIT
**Dependencies**: Z.2.5 (GC), CAPABILITY/CAS, CAPABILITY/RUNS

---

## Purpose

The ROOT AUDIT tool provides a deterministic, fail-closed verification gate that proves:
1. Root completeness: all required artifacts are reachable from declared roots
2. GC safety: the reachable set is correctly computed using the same semantics as GC
3. Audit trail: deterministic receipts suitable for CI gates and human review

This tool is designed as a **pre-packer integration gate** to verify root integrity before production GC operations.

---

## Inputs

### Root Sources (Explicit and Auditable)

The audit enumerates roots from exactly two sources, using the same logic as Z.2.5 GC:

1. **RUN_ROOTS** (`CAPABILITY/RUNS/RUN_ROOTS.json`)
   - JSON array of CAS hash strings
   - Each hash: exactly 64 lowercase hex characters (SHA-256)
   - Missing file is NOT an error (treated as empty list)
   - Malformed JSON or invalid hash format => fail-closed

2. **GC_PINS** (`CAPABILITY/RUNS/GC_PINS.json`)
   - Same format as RUN_ROOTS
   - Optional operator-managed pins for retention
   - Missing file is NOT an error (treated as empty list)
   - Malformed JSON or invalid hash format => fail-closed

### Root Enumeration Rules

- Roots from both sources are combined and deduplicated (set union)
- Each root must be validated: 64 lowercase hex characters
- Any invalid format => fail-closed with error list
- Empty roots => FAIL with explicit error (no override for audit; fail-closed policy)

### Required Outputs Source (Mode B Only)

When verifying run completeness (Mode B), the audit requires:

- **OUTPUT_HASHES record hash** (CAS hash of stored OUTPUT_HASHES list)
  - Format: 64 lowercase hex SHA-256 hash
  - References a CAS object containing canonical JSON array of output hashes
  - Must exist in CAS and decode successfully
  - Each referenced artifact hash must be valid format and reachable

---

## Reachability Semantics

### Mark Phase Equivalence

The reachable set MUST be computed using identical semantics to Z.2.5 GC mark phase:

- **Current implementation** (Z.2.5): Trivial traversal (roots = reachable set)
- **Future deep traversal**: When GC implements reference traversal, audit MUST use the same logic
- **Determinism**: Traversal order must be stable (sorted roots, deterministic queue/stack)
- **Deduplication**: Reachable set is deduplicated (no double-counting)

### Implementation Strategy

To maintain equivalence:
- Prefer importing and reusing GC's `_traverse_references()` if exposed
- If not exposed, implement a shared helper with identical behavior (minimum change)
- Tests must verify equivalence between audit and GC reachability

### Stable Ordering

For determinism and reproducibility:
- Roots are sorted before traversal
- Traversal uses deterministic data structures (sorted lists, not arbitrary sets)
- Output lists in receipt are sorted (e.g., required_missing, required_unreachable)

---

## Audit Modes

### Mode A: General Root Safety Audit

**Enabled when**: `output_hashes_record = None`

**Verifies**:
- Roots enumerate successfully from declared sources
- Root format is valid
- Reachable set is computable
- Receipt is deterministic and complete

**Pass criteria**:
- No errors encountered
- Roots enumerated (count >= 0, but empty roots => FAIL for audit)
- Reachable set computed successfully

**Fail conditions**:
- Root enumeration error (invalid JSON, invalid hash format)
- Empty roots (fail-closed for audit; no override allowed)
- Traversal error (future: if deep traversal fails)

### Mode B: Run Completeness Check

**Enabled when**: `output_hashes_record = <CAS hash>`

**Verifies**: All conditions from Mode A, PLUS:
- OUTPUT_HASHES record exists in CAS
- Record decodes successfully (canonical JSON array)
- Each listed artifact ref is valid format (64 hex chars)
- Each artifact is reachable from declared roots
- Each referenced blob exists in CAS (not corrupted)

**Pass criteria**:
- Mode A pass criteria met
- OUTPUT_HASHES record loaded successfully
- `required_missing = []` (all artifacts exist)
- `required_unreachable = []` (all artifacts reachable)

**Fail conditions**:
- Any Mode A fail condition, OR
- OUTPUT_HASHES record missing from CAS
- OUTPUT_HASHES record decode failure (invalid JSON, invalid format)
- Any artifact ref has invalid format
- Any artifact is missing from CAS
- Any artifact is unreachable from roots
- Any blob corruption detected (future: when integrity checks enabled)

---

## Validation Rules (Fail-Closed)

The audit operates under strict fail-closed semantics:

1. **Hash/ref format validation**
   - Any hash not matching `^[a-f0-9]{64}$` => FAIL
   - Error reported in `errors` list with specific hash and reason

2. **Missing objects**
   - Required object (e.g., OUTPUT_HASHES record) missing => FAIL
   - Missing artifact when required => reported in `required_missing`

3. **Decode failures**
   - JSON parse error => FAIL
   - Canonicalization failure => FAIL
   - Type mismatch (expected array, got object) => FAIL

4. **Empty roots**
   - Zero roots => FAIL with explicit error
   - No override allowed (unlike GC's `allow_empty_roots`)
   - Rationale: Audit is a safety gate; empty roots indicate missing packer integration

5. **Non-deterministic ordering**
   - Considered a BUG (not runtime error)
   - Tests MUST catch any non-determinism
   - Receipt fields must have stable ordering

---

## Receipt Format (Deterministic)

The audit returns a dict with the following structure:

```python
{
    # Audit metadata
    'mode': 'audit',  # Always 'audit' for this tool

    # Root sources (auditable trail)
    'root_sources': [
        {
            'name': 'RUN_ROOTS',
            'path': 'CAPABILITY/RUNS/RUN_ROOTS.json',
            'exists': True,
            'content_hash': '<SHA-256 of file contents>' | None
        },
        {
            'name': 'GC_PINS',
            'path': 'CAPABILITY/RUNS/GC_PINS.json',
            'exists': False,
            'content_hash': None
        }
    ],

    # Root statistics
    'roots_count': 5,
    'reachable_hashes_count': 5,  # Current: same as roots_count

    # Required outputs check (Mode B)
    'required_check': {
        'enabled': True,  # True if output_hashes_record provided
        'output_hashes_record': '<hash>' | None
    },
    'required_total': 3,  # 0 if Mode A
    'required_missing': [],  # Sorted list of hashes not found in CAS
    'required_unreachable': [],  # Sorted list of hashes not reachable

    # Errors (fail-closed conditions)
    'errors': [],  # Sorted list of error messages

    # CAS snapshot (for determinism verification)
    'cas_snapshot_hash': '<SHA-256 of sorted blob list>',

    # Verdict (computed field)
    'verdict': 'PASS' | 'FAIL'
}
```

### Field Definitions

- **mode**: Always `"audit"` for this tool
- **root_sources**: List of source metadata (name, path, exists, content_hash)
  - `content_hash`: SHA-256 of file contents if file exists, else None
- **roots_count**: Total number of deduplicated roots
- **reachable_hashes_count**: Number of hashes in reachable set
- **required_check.enabled**: True if Mode B (output_hashes_record provided)
- **required_check.output_hashes_record**: CAS hash of OUTPUT_HASHES record, or None
- **required_total**: Number of artifacts in OUTPUT_HASHES list (0 for Mode A)
- **required_missing**: Artifacts that don't exist in CAS (sorted)
- **required_unreachable**: Artifacts not reachable from roots (sorted)
- **errors**: All fail-closed errors encountered (sorted for stability)
- **cas_snapshot_hash**: Deterministic hash of CAS state (same method as GC)
- **verdict**: `"PASS"` or `"FAIL"`

### Verdict Logic

**PASS** criteria:
- Mode A: `errors == []` AND `roots_count > 0`
- Mode B: Mode A criteria AND `required_missing == []` AND `required_unreachable == []`

**FAIL** criteria:
- Any errors reported, OR
- Empty roots (`roots_count == 0`), OR
- Mode B: any required_missing or required_unreachable

---

## Determinism Requirements

The audit MUST produce identical receipts for identical inputs:

1. **Same storage + same roots => identical receipt bytes**
   - When canonically encoded (sorted keys, no whitespace)
   - Tests MUST verify byte-for-byte equality

2. **Stable field ordering**
   - All list fields sorted (errors, required_missing, required_unreachable)
   - root_sources in declaration order (RUN_ROOTS, then GC_PINS)

3. **No wall-clock dependency**
   - No timestamps, random IDs, or time-of-day values
   - Reproducible across runs and environments

4. **Platform-independent**
   - Windows/Linux/macOS path handling
   - Case-insensitive filesystem safety (avoid case-only differences)

---

## Public API

### `root_audit(*, output_hashes_record: str | None = None, dry_run: bool = True) -> dict`

**Parameters**:
- `output_hashes_record`: Optional CAS hash of OUTPUT_HASHES record
  - `None` => Mode A (general safety audit)
  - `<hash>` => Mode B (run completeness check)
- `dry_run`: Always True for audit (no deletions ever)
  - Kept for interface symmetry with GC
  - Future: may be removed if confusing

**Returns**: Receipt dict (see Receipt Format above)

**Raises**: None (fail-closed via receipt, not exceptions)

**Side effects**: None (read-only operation)

---

## Error Handling

All errors are reported in the receipt's `errors` list. The audit does NOT raise exceptions for operational failures (fail-closed via verdict).

Common error scenarios:

1. **Root enumeration errors**:
   - `"RUN_ROOTS: Invalid JSON: <details>"`
   - `"RUN_ROOTS: Invalid hash format: <hash>"`
   - `"GC_PINS: File read error: <details>"`

2. **Empty roots**:
   - `"POLICY_LOCK: Empty roots detected. Audit requires at least one root."`

3. **Mode B errors**:
   - `"OUTPUT_HASHES record missing from CAS: <hash>"`
   - `"OUTPUT_HASHES decode error: <details>"`
   - `"Invalid artifact hash in OUTPUT_HASHES: <hash>"`

4. **CAS errors**:
   - `"CAS enumeration failed: <details>"`
   - `"Blob integrity check failed: <hash>"`

All errors are sorted for determinism.

---

## Integration Points

### Pre-Packer Gate (Future)

Before packer registers roots:
1. Run audit in Mode A to verify storage health
2. After run completion, run audit in Mode B with OUTPUT_HASHES record
3. Only proceed with root registration if audit PASS

### CI/CD Gates

Use audit receipt for automated verification:
- Check `verdict == "PASS"`
- Log receipt for audit trail
- Fail build if verdict == "FAIL"

### Human Review

Receipt provides clear summary:
- Root sources and counts
- Required artifact status
- Clear error messages
- Deterministic snapshot hash for comparison

---

## Test Requirements

Tests MUST cover all invariants and fail-closed conditions:

### Mode A Tests

- **A-01**: Determinism: same inputs => identical receipt bytes
- **A-02**: Empty roots => FAIL with explicit error
- **A-03**: Invalid root format => FAIL with stable error list
- **A-04**: Reachable count matches known fixture

### Mode B Tests

- **B-01**: Valid OUTPUT_HASHES, all rooted => PASS
- **B-02**: OUTPUT_HASHES includes unrooted ref => FAIL (required_unreachable)
- **B-03**: Invalid ref format in OUTPUT_HASHES => FAIL
- **B-04**: OUTPUT_HASHES record missing => FAIL
- **B-05**: Corrupted blob => FAIL (or explicit policy)

All tests MUST use isolated temp storage (never touch real CAS).

---

## Future Extensions

### Deep Traversal

When GC implements reference traversal (OUTPUT_HASHES => artifact blobs):
- Audit MUST use identical traversal logic
- Update `_traverse_references()` to be shared helper
- Tests verify equivalence between audit and GC

### Integrity Verification

Optional blob integrity checks:
- Re-hash each blob and verify against filename
- Report corrupted blobs in errors
- Add `corrupted_blobs` field to receipt

### Receipt Persistence

If repo adopts receipt convention:
- Write canonical JSON to `CAPABILITY/AUDIT/receipts/<timestamp>.json`
- Ensure deterministic filename (content hash, not wall-clock)

---

## Revision History

  - Fail-closed validation
  - Test coverage: A-01 through B-05

---

## Recovery: Root Audit Invariant Violations

### Where receipts live

Audit receipts are generated on-demand but may be captured in:
- **LAW/CONTRACTS/_runs/audit_logs/** - If run as part of a pipeline or CI gate
- **Runtime Standard Output** - Direct CLI usage logs
- **Packer integration logs** - When running `pack_create`

### How to re-run verification

To verify root integrity manually:

```bash
# Mode A: General Root Safety Check (Default)
python CAPABILITY/AUDIT/root_audit.py --verbose

# Mode B: Run Completeness Check (Specific Run)
# Requires a known OUTPUT_HASHES CAS reference
python CAPABILITY/AUDIT/root_audit.py --output-hashes-record SHA256_HASH_HERE
```

### What to delete vs never delete

**Safe to delete:**
- **Audit logs/receipts**: These are informational snapshots.
- **Corrupted RUN_ROOTS.json**: If absolutely necessary, but *prefer git restore*.

**Never delete (protected by invariants):**
- **Existing roots in RUN_ROOTS.json**: Unless you are certain the referenced data is garbage.
- **Artifacts referenced by roots**: Deleting these causes audit FAILURE.

**Recovery procedures:**
- **Audit Fails (Missing Object)**:
  - If the object is truly lost, remove the reference from `RUN_ROOTS.json` (making it clean but data lost) OR restore the object from backup.
- **Audit Fails (Invalid JSON)**:
  - Run `git checkout HEAD -- CAPABILITY/RUNS/RUN_ROOTS.json` to restore the last known good state.
- **Audit Fails (Unreachable Artifact)**:
  - Check if the artifact was manually deleted or if the hash in `OUTPUT_HASHES` is incorrect.
  - Re-run the producing task to regenerate valid artifacts.
