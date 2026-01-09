"""
Z.2.6 – ROOT AUDIT Tests

Tests for root audit functionality covering:
- A-01: Determinism (same inputs => identical receipt)
- A-02: Empty roots fail-closed
- A-03: Invalid root format fail-closed
- A-04: Reachable count matches fixture
- B-01: Valid OUTPUT_HASHES, all rooted => PASS
- B-02: Unrooted ref => FAIL (required_unreachable)
- B-03: Invalid ref format => FAIL
- B-04: Missing OUTPUT_HASHES record => FAIL
- B-05: Corrupted blob => FAIL

All tests run in isolated temp storage and are deterministic.
"""

import json
import hashlib
from pathlib import Path
import pytest

from CAPABILITY.AUDIT.root_audit import root_audit


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """
    Create an isolated environment for audit testing.

    Returns a dict with:
    - cas_root: Path to isolated CAS storage
    - runs_dir: Path to isolated RUNS directory
    """
    cas_root = tmp_path / "CAS" / "storage"
    runs_dir = tmp_path / "RUNS"

    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Patch the global CAS root to use isolated storage
    from CAPABILITY.CAS import cas
    monkeypatch.setattr(cas, '_CAS_ROOT', cas_root)

    return {
        'cas_root': cas_root,
        'runs_dir': runs_dir,
        'tmp_path': tmp_path
    }


def _put_blob_isolated(data: bytes, cas_root: Path) -> str:
    """Helper to put a blob in isolated CAS storage"""
    hash_str = hashlib.sha256(data).hexdigest()

    # Create path: first char / next 2 chars / full hash
    prefix1 = hash_str[0]
    prefix2 = hash_str[1:3]
    obj_path = cas_root / prefix1 / prefix2 / hash_str

    obj_path.parent.mkdir(parents=True, exist_ok=True)

    if not obj_path.exists():
        obj_path.write_bytes(data)

    return hash_str


def _write_run_roots(runs_dir: Path, roots: list):
    """Helper to write RUN_ROOTS.json"""
    root_file = runs_dir / "RUN_ROOTS.json"
    with open(root_file, 'w', encoding='utf-8') as f:
        json.dump(roots, f)


def _write_gc_pins(runs_dir: Path, pins: list):
    """Helper to write GC_PINS.json"""
    pin_file = runs_dir / "GC_PINS.json"
    with open(pin_file, 'w', encoding='utf-8') as f:
        json.dump(pins, f)


def _canonical_json(obj: dict) -> bytes:
    """Canonical JSON encoding for comparison"""
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')


# ============================================================================
# A-01: Determinism - same inputs => identical receipt
# ============================================================================

def test_a01_determinism_identical_inputs(isolated_env, monkeypatch):
    """A-01: Same storage + same roots => identical receipt bytes"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']

    # Create fixture: 3 blobs, 2 rooted
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)

    _write_run_roots(runs_dir, [h1, h2])

    # Run audit twice
    receipt1 = root_audit(cas_root=cas_root, runs_dir=runs_dir)
    receipt2 = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify identical receipts (byte-for-byte)
    assert _canonical_json(receipt1) == _canonical_json(receipt2)

    # Verify both pass
    assert receipt1['verdict'] == 'PASS'
    assert receipt2['verdict'] == 'PASS'


def test_a01_determinism_sorted_errors(isolated_env, monkeypatch):
    """A-01: Errors are sorted for determinism"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']

    # Create malformed RUN_ROOTS with multiple invalid entries
    root_file = runs_dir / "RUN_ROOTS.json"
    with open(root_file, 'w', encoding='utf-8') as f:
        json.dump(["zzzinvalid", "aaainvalid", "mmminvalid"], f)

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify errors are sorted
    assert receipt['verdict'] == 'FAIL'
    errors = receipt['errors']
    assert errors == sorted(errors)
    assert len(errors) == 3


# ============================================================================
# A-02: Empty roots fail-closed
# ============================================================================

def test_a02_empty_roots_fail_closed(isolated_env, monkeypatch):
    """A-02: Empty roots must fail-closed with explicit error"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create some blobs but no roots
    _put_blob_isolated(b"blob1", cas_root)
    _put_blob_isolated(b"blob2", cas_root)

    # No root files (or empty root files)
    _write_run_roots(runs_dir, [])

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify fail-closed
    assert receipt['verdict'] == 'FAIL'
    assert receipt['roots_count'] == 0
    assert receipt['reachable_hashes_count'] == 0
    assert any('POLICY_LOCK' in e and 'Empty roots' in e for e in receipt['errors'])


def test_a02_empty_roots_no_override(isolated_env, monkeypatch):
    """A-02: Audit has no override for empty roots (unlike GC)"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # No roots
    _write_run_roots(runs_dir, [])

    # Audit always fails with empty roots (no allow_empty_roots param)
    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    assert receipt['verdict'] == 'FAIL'
    assert receipt['roots_count'] == 0


# ============================================================================
# A-03: Invalid root format fail-closed
# ============================================================================

def test_a03_invalid_hash_format_fail_closed(isolated_env, monkeypatch):
    """A-03: Invalid root entry format => FAIL with error list stable"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Valid hash for reference
    h1 = _put_blob_isolated(b"blob1", cas_root)

    # Mix of valid and invalid hashes
    _write_run_roots(runs_dir, [
        h1,
        "tooshort",
        "not-hex-chars-but-correct-length-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "UPPERCASE_NOT_ALLOWED_1234567890abcdef1234567890abcdef12345678"
    ])

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify fail-closed
    assert receipt['verdict'] == 'FAIL'
    assert len(receipt['errors']) >= 3  # At least 3 invalid hashes

    # Verify errors mention format
    assert any('Invalid hash format' in e for e in receipt['errors'])

    # Verify errors are sorted
    assert receipt['errors'] == sorted(receipt['errors'])


def test_a03_malformed_json_fail_closed(isolated_env, monkeypatch):
    """A-03: Malformed RUN_ROOTS.json => FAIL"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Write malformed JSON
    root_file = runs_dir / "RUN_ROOTS.json"
    root_file.write_text("{invalid json")

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify fail-closed
    assert receipt['verdict'] == 'FAIL'
    assert any('Invalid JSON' in e for e in receipt['errors'])


def test_a03_root_not_list_fail_closed(isolated_env, monkeypatch):
    """A-03: RUN_ROOTS must be a list"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Write non-list JSON
    root_file = runs_dir / "RUN_ROOTS.json"
    root_file.write_text('{"not": "a list"}')

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify fail-closed
    assert receipt['verdict'] == 'FAIL'
    assert any('Must be a list' in e for e in receipt['errors'])


# ============================================================================
# A-04: Reachable count matches known fixture
# ============================================================================

def test_a04_reachable_count_matches_fixture(isolated_env, monkeypatch):
    """A-04: Reachable count matches known constructed fixture graph"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create known fixture: 5 blobs, 3 rooted (from RUN_ROOTS and GC_PINS)
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)
    h4 = _put_blob_isolated(b"blob4", cas_root)
    h5 = _put_blob_isolated(b"blob5", cas_root)

    _write_run_roots(runs_dir, [h1, h2])
    _write_gc_pins(runs_dir, [h3, h2])  # h2 duplicated (should dedupe)

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify counts
    assert receipt['verdict'] == 'PASS'
    assert receipt['roots_count'] == 3  # h1, h2, h3 (deduplicated)
    assert receipt['reachable_hashes_count'] == 3  # Same as roots (trivial traversal)

    # Verify root sources metadata
    assert len(receipt['root_sources']) == 2
    assert receipt['root_sources'][0]['name'] == 'RUN_ROOTS'
    assert receipt['root_sources'][0]['exists'] is True
    assert receipt['root_sources'][1]['name'] == 'GC_PINS'
    assert receipt['root_sources'][1]['exists'] is True


def test_a04_missing_root_files_not_error(isolated_env, monkeypatch):
    """A-04: Missing root files are not an error (treated as empty)"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Don't write any root files
    # But this will trigger empty roots policy => FAIL

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Empty roots fail, but not because files are missing
    assert receipt['verdict'] == 'FAIL'
    assert receipt['roots_count'] == 0
    assert receipt['root_sources'][0]['exists'] is False
    assert receipt['root_sources'][1]['exists'] is False

    # The only error should be the policy lock, not file-not-found
    assert len(receipt['errors']) == 1
    assert 'POLICY_LOCK' in receipt['errors'][0]


# ============================================================================
# B-01: Valid OUTPUT_HASHES, all rooted => PASS
# ============================================================================

def test_b01_valid_output_hashes_all_rooted_pass(isolated_env, monkeypatch):
    """B-01: Given valid OUTPUT_HASHES record where all refs are rooted => PASS"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create output blobs
    out1 = _put_blob_isolated(b"output1", cas_root)
    out2 = _put_blob_isolated(b"output2", cas_root)
    out3 = _put_blob_isolated(b"output3", cas_root)

    # Create OUTPUT_HASHES record using isolated helper
    output_hashes_json = json.dumps([out1, out2, out3], sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(output_hashes_json, cas_root)

    # Root all outputs
    _write_run_roots(runs_dir, [out1, out2, out3])

    # Run audit in Mode B
    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify PASS
    assert receipt['verdict'] == 'PASS'
    assert receipt['required_check']['enabled'] is True
    assert receipt['required_check']['output_hashes_record'] == output_hashes_record
    assert receipt['required_total'] == 3
    assert receipt['required_missing'] == []
    assert receipt['required_unreachable'] == []
    assert receipt['errors'] == []


def test_b01_output_hashes_record_also_rooted(isolated_env, monkeypatch):
    """B-01: OUTPUT_HASHES record itself should also be rooted"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create outputs
    out1 = _put_blob_isolated(b"output1", cas_root)
    out2 = _put_blob_isolated(b"output2", cas_root)

    # Create OUTPUT_HASHES record using isolated helper
    output_hashes_json = json.dumps([out1, out2], sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(output_hashes_json, cas_root)

    # Root outputs AND the record itself
    _write_run_roots(runs_dir, [output_hashes_record, out1, out2])

    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify PASS
    assert receipt['verdict'] == 'PASS'
    assert receipt['roots_count'] == 3  # record + out1 + out2


# ============================================================================
# B-02: OUTPUT_HASHES includes unrooted ref => FAIL (required_unreachable)
# ============================================================================

def test_b02_unrooted_ref_fail_unreachable(isolated_env, monkeypatch):
    """B-02: OUTPUT_HASHES record exists but includes a ref not rooted => FAIL"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create outputs
    out1 = _put_blob_isolated(b"output1", cas_root)
    out2 = _put_blob_isolated(b"output2", cas_root)
    out3 = _put_blob_isolated(b"output3", cas_root)

    # Create OUTPUT_HASHES record with all three using isolated helper
    output_hashes_json = json.dumps([out1, out2, out3], sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(output_hashes_json, cas_root)

    # Only root out1 and out2 (out3 is unreachable)
    _write_run_roots(runs_dir, [out1, out2])

    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify FAIL with unreachable
    assert receipt['verdict'] == 'FAIL'
    assert receipt['required_total'] == 3
    assert receipt['required_missing'] == []  # All exist in CAS
    assert receipt['required_unreachable'] == [out3]  # Not rooted


def test_b02_unreachable_sorted(isolated_env, monkeypatch):
    """B-02: required_unreachable is sorted for determinism"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create many outputs
    outs = [_put_blob_isolated(f"output{i}".encode(), cas_root) for i in range(10)]

    # Create OUTPUT_HASHES record using isolated helper
    output_hashes_json = json.dumps(outs, sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(output_hashes_json, cas_root)

    # Only root half of them (alternating)
    _write_run_roots(runs_dir, [outs[i] for i in range(0, 10, 2)])

    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify unreachable is sorted
    assert receipt['verdict'] == 'FAIL'
    unreachable = receipt['required_unreachable']
    assert unreachable == sorted(unreachable)
    assert len(unreachable) == 5


# ============================================================================
# B-03: Invalid ref format in OUTPUT_HASHES => FAIL
# ============================================================================

def test_b03_invalid_output_hashes_record_format(isolated_env, monkeypatch):
    """B-03: OUTPUT_HASHES record includes invalid ref format => FAIL"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create a dummy root
    h1 = _put_blob_isolated(b"root1", cas_root)
    _write_run_roots(runs_dir, [h1])

    # Use invalid hash format for output_hashes_record parameter
    invalid_hash = "not-a-valid-hash"

    receipt = root_audit(output_hashes_record=invalid_hash, cas_root=cas_root, runs_dir=runs_dir)

    # Verify FAIL
    assert receipt['verdict'] == 'FAIL'
    assert any('Invalid hash format' in e for e in receipt['errors'])


def test_b03_invalid_ref_in_output_hashes_list(isolated_env, monkeypatch):
    """B-03: Invalid hash inside OUTPUT_HASHES list => FAIL"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Manually create a malformed OUTPUT_HASHES record
    malformed_list = ["validhash" * 8, "invalid"]  # First is 64 chars, second is not

    # We need to bypass put_output_hashes validation and create the blob directly
    # This simulates corrupted/malformed data
    malformed_json = json.dumps(malformed_list, sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(malformed_json, cas_root)

    # Create a valid root
    h1 = _put_blob_isolated(b"root1", cas_root)
    _write_run_roots(runs_dir, [h1])

    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify FAIL
    assert receipt['verdict'] == 'FAIL'
    assert any('invalid format' in e.lower() for e in receipt['errors'])


# ============================================================================
# B-04: OUTPUT_HASHES record hash missing from CAS => FAIL
# ============================================================================

def test_b04_output_hashes_record_missing(isolated_env, monkeypatch):
    """B-04: OUTPUT_HASHES record hash missing from CAS => FAIL"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create a valid root
    h1 = _put_blob_isolated(b"root1", cas_root)
    _write_run_roots(runs_dir, [h1])

    # Use a valid hash format that doesn't exist in CAS
    nonexistent_hash = "0" * 64

    receipt = root_audit(output_hashes_record=nonexistent_hash, cas_root=cas_root, runs_dir=runs_dir)

    # Verify FAIL
    assert receipt['verdict'] == 'FAIL'
    # Check for error about missing OUTPUT_HASHES record (may say "not found" or similar)
    assert len(receipt['errors']) > 0
    assert any('OUTPUT_HASHES' in e for e in receipt['errors'])


# ============================================================================
# B-05: Missing or corrupted referenced blob => FAIL
# ============================================================================

def test_b05_required_output_missing_from_cas(isolated_env, monkeypatch):
    """B-05: OUTPUT_HASHES references blob that doesn't exist in CAS => FAIL"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create one real output
    out1 = _put_blob_isolated(b"output1", cas_root)

    # Create a fake hash that doesn't exist
    fake_out2 = "f" * 64

    # Create OUTPUT_HASHES record referencing both
    # Bypass put_output_hashes which would validate
    malformed_list = [out1, fake_out2]
    malformed_json = json.dumps(malformed_list, sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(malformed_json, cas_root)

    # Root both (even though one doesn't exist)
    _write_run_roots(runs_dir, [out1, fake_out2])

    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify FAIL with missing
    assert receipt['verdict'] == 'FAIL'
    assert receipt['required_total'] == 2
    assert fake_out2 in receipt['required_missing']
    assert receipt['required_unreachable'] == []  # It's rooted, just missing


def test_b05_required_missing_sorted(isolated_env, monkeypatch):
    """B-05: required_missing is sorted for determinism"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create some real outputs and store them in CAS
    real_outs = [_put_blob_isolated(f"output{i}".encode(), cas_root) for i in range(3)]

    # Create some fake hashes (valid format, but won't be stored in CAS)
    # These will be in the OUTPUT_HASHES record but missing from CAS
    fake_hashes = [
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    ]

    all_hashes = real_outs + fake_hashes

    # Store OUTPUT_HASHES record using isolated helper
    # Since fake_hashes are valid format but don't exist in CAS, this will work
    output_hashes_json = json.dumps(all_hashes, sort_keys=True, separators=(',', ':')).encode('utf-8')
    output_hashes_record = _put_blob_isolated(output_hashes_json, cas_root)

    # Root all hashes (so they're reachable, but fakes don't exist in CAS)
    _write_run_roots(runs_dir, all_hashes)

    receipt = root_audit(output_hashes_record=output_hashes_record, cas_root=cas_root, runs_dir=runs_dir)

    # Verify missing is sorted
    assert receipt['verdict'] == 'FAIL'
    missing = receipt['required_missing']
    assert missing == sorted(missing)
    assert set(missing) == set(fake_hashes)
    # They're rooted, so they're reachable (not in unreachable list)
    assert receipt['required_unreachable'] == []


# ============================================================================
# Additional edge cases
# ============================================================================

def test_mode_a_pass_with_valid_roots(isolated_env, monkeypatch):
    """Mode A: Simple PASS case with valid roots"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create and root some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)

    _write_run_roots(runs_dir, [h1, h2])

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify PASS in Mode A
    assert receipt['mode'] == 'audit'
    assert receipt['verdict'] == 'PASS'
    assert receipt['required_check']['enabled'] is False
    assert receipt['required_total'] == 0
    assert receipt['errors'] == []


def test_dry_run_parameter_always_true(isolated_env, monkeypatch):
    """dry_run parameter is accepted but always True for audit"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    h1 = _put_blob_isolated(b"blob1", cas_root)
    _write_run_roots(runs_dir, [h1])

    # Try both True and False
    receipt1 = root_audit(dry_run=True, cas_root=cas_root, runs_dir=runs_dir)
    receipt2 = root_audit(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)

    # Both should succeed (audit never deletes)
    assert receipt1['verdict'] == 'PASS'
    assert receipt2['verdict'] == 'PASS'

    # Blobs still exist
    obj_path = cas_root / h1[0] / h1[1:3] / h1
    assert obj_path.exists()


def test_cas_snapshot_hash_deterministic(isolated_env, monkeypatch):
    """CAS snapshot hash is deterministic"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    # Create blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)

    _write_run_roots(runs_dir, [h1])

    # Run audit multiple times
    receipt1 = root_audit(cas_root=cas_root, runs_dir=runs_dir)
    receipt2 = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Snapshot hash should be identical
    assert receipt1['cas_snapshot_hash'] == receipt2['cas_snapshot_hash']
    assert receipt1['cas_snapshot_hash'] != ''


def test_root_source_content_hash_computed(isolated_env, monkeypatch):
    """Root source metadata includes content hash"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']


    h1 = _put_blob_isolated(b"blob1", cas_root)
    _write_run_roots(runs_dir, [h1])

    receipt = root_audit(cas_root=cas_root, runs_dir=runs_dir)

    # Verify root sources have content hashes
    run_roots_source = receipt['root_sources'][0]
    assert run_roots_source['name'] == 'RUN_ROOTS'
    assert run_roots_source['exists'] is True
    assert run_roots_source['content_hash'] is not None
    assert len(run_roots_source['content_hash']) == 64

    gc_pins_source = receipt['root_sources'][1]
    assert gc_pins_source['name'] == 'GC_PINS'
    assert gc_pins_source['exists'] is False
    assert gc_pins_source['content_hash'] is None
