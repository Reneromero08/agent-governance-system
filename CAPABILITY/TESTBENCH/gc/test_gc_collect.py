"""
Z.2.5 – GC Tests

Tests for garbage collection implementation covering:
- GC-01 through GC-11: Core GC behavior
- GC-04, GC-05, GC-16, GC-18: Specific edge cases
- GC-12: Policy B enforcement (empty roots fail-closed)

All tests run in isolated temp storage and are deterministic.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
import pytest

from CAPABILITY.CAS.cas import cas_put, cas_get
from CAPABILITY.GC.gc import gc_collect


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def isolated_env(tmp_path):
    """
    Create an isolated environment for GC testing.
    
    Returns a dict with:
    - cas_root: Path to isolated CAS storage
    - runs_dir: Path to isolated RUNS directory
    """
    cas_root = tmp_path / "CAS" / "storage"
    runs_dir = tmp_path / "RUNS"
    
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'cas_root': cas_root,
        'runs_dir': runs_dir,
        'tmp_path': tmp_path
    }


def _put_blob_isolated(data: bytes, cas_root: Path) -> str:
    """Helper to put a blob in isolated CAS storage"""
    import hashlib
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


# ============================================================================
# GC-01: Dry run reports candidates without deleting
# ============================================================================

def test_gc_01_dry_run_no_delete(isolated_env):
    """GC-01: Dry run mode reports candidates but performs no deletions"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)
    
    # Set h1 as root
    _write_run_roots(runs_dir, [h1])
    
    # Run GC in dry-run mode
    result = gc_collect(dry_run=True, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report
    assert result['mode'] == 'dry_run'
    assert result['roots_count'] == 1
    assert result['reachable_hashes_count'] == 1
    assert result['candidate_hashes_count'] == 2
    assert result['deleted_hashes'] == []
    assert len(result['skipped_hashes']) == 2
    assert all(s['reason'] == 'dry_run' for s in result['skipped_hashes'])
    assert result['errors'] == []
    
    # Verify no deletions occurred
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert (cas_root / h2[0] / h2[1:3] / h2).exists()
    assert (cas_root / h3[0] / h3[1:3] / h3).exists()


# ============================================================================
# GC-02: Apply mode deletes unreferenced blobs
# ============================================================================

def test_gc_02_apply_deletes_unreferenced(isolated_env):
    """GC-02: Apply mode deletes unreferenced blobs"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)
    
    # Set h1 as root
    _write_run_roots(runs_dir, [h1])
    
    # Run GC in apply mode
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report
    assert result['mode'] == 'apply'
    assert result['roots_count'] == 1
    assert result['reachable_hashes_count'] == 1
    assert result['candidate_hashes_count'] == 2
    assert len(result['deleted_hashes']) == 2
    assert set(result['deleted_hashes']) == {h2, h3}
    assert result['skipped_hashes'] == []
    assert result['errors'] == []
    
    # Verify deletions occurred
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert not (cas_root / h2[0] / h2[1:3] / h2).exists()
    assert not (cas_root / h3[0] / h3[1:3] / h3).exists()


# ============================================================================
# GC-03: Multiple roots preserve all referenced blobs
# ============================================================================

def test_gc_03_multiple_roots(isolated_env):
    """GC-03: Multiple roots preserve all referenced blobs"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)
    h4 = _put_blob_isolated(b"blob4", cas_root)
    
    # Set h1 and h2 as roots
    _write_run_roots(runs_dir, [h1, h2])
    
    # Run GC in apply mode
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report
    assert result['roots_count'] == 2
    assert result['reachable_hashes_count'] == 2
    assert result['candidate_hashes_count'] == 2
    assert set(result['deleted_hashes']) == {h3, h4}
    
    # Verify correct blobs preserved
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert (cas_root / h2[0] / h2[1:3] / h2).exists()
    assert not (cas_root / h3[0] / h3[1:3] / h3).exists()
    assert not (cas_root / h4[0] / h4[1:3] / h4).exists()


# ============================================================================
# GC-04: Pin file roots are honored
# ============================================================================

def test_gc_04_pin_file_roots(isolated_env):
    """GC-04: Pin file roots are honored and preserved"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)
    
    # Set h1 as run root, h2 as pin
    _write_run_roots(runs_dir, [h1])
    _write_gc_pins(runs_dir, [h2])
    
    # Run GC in apply mode
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report
    assert result['roots_count'] == 2
    assert result['reachable_hashes_count'] == 2
    assert len(result['root_sources']) == 2
    assert result['candidate_hashes_count'] == 1
    assert result['deleted_hashes'] == [h3]
    
    # Verify correct blobs preserved
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert (cas_root / h2[0] / h2[1:3] / h2).exists()
    assert not (cas_root / h3[0] / h3[1:3] / h3).exists()


# ============================================================================
# GC-05: Deduplication across root sources
# ============================================================================

def test_gc_05_dedup_across_sources(isolated_env):
    """GC-05: Same hash in multiple root sources is deduplicated"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    
    # Set h1 in both run roots and pins
    _write_run_roots(runs_dir, [h1])
    _write_gc_pins(runs_dir, [h1])
    
    # Run GC in apply mode
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report: h1 should only be counted once
    assert result['roots_count'] == 1  # Deduplicated
    assert result['reachable_hashes_count'] == 1
    assert result['candidate_hashes_count'] == 1
    assert result['deleted_hashes'] == [h2]


# ============================================================================
# GC-06: Empty CAS is handled gracefully
# ============================================================================

def test_gc_06_empty_cas(isolated_env):
    """GC-06: Empty CAS is handled gracefully"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # No blobs, no roots
    _write_run_roots(runs_dir, [])
    
    # Run GC with allow_empty_roots=True
    result = gc_collect(dry_run=False, allow_empty_roots=True, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report
    assert result['roots_count'] == 0
    assert result['reachable_hashes_count'] == 0
    assert result['candidate_hashes_count'] == 0
    assert result['deleted_hashes'] == []
    assert result['errors'] == []


# ============================================================================
# GC-07: All blobs referenced (no candidates)
# ============================================================================

def test_gc_07_all_referenced(isolated_env):
    """GC-07: When all blobs are referenced, no deletions occur"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    h3 = _put_blob_isolated(b"blob3", cas_root)
    
    # Set all as roots
    _write_run_roots(runs_dir, [h1, h2, h3])
    
    # Run GC in apply mode
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify report
    assert result['roots_count'] == 3
    assert result['reachable_hashes_count'] == 3
    assert result['candidate_hashes_count'] == 0
    assert result['deleted_hashes'] == []
    
    # Verify all blobs still exist
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert (cas_root / h2[0] / h2[1:3] / h2).exists()
    assert (cas_root / h3[0] / h3[1:3] / h3).exists()


# ============================================================================
# GC-08: Deterministic ordering in reports
# ============================================================================

def test_gc_08_deterministic_ordering(isolated_env):
    """GC-08: Reports have deterministic, stable ordering"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create blobs in non-sorted order
    h3 = _put_blob_isolated(b"blob3", cas_root)
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    
    # No roots
    _write_run_roots(runs_dir, [])
    
    # Run GC twice with allow_empty_roots=True
    result1 = gc_collect(dry_run=True, allow_empty_roots=True, cas_root=cas_root, runs_dir=runs_dir)
    result2 = gc_collect(dry_run=True, allow_empty_roots=True, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify deterministic ordering
    assert result1['skipped_hashes'] == result2['skipped_hashes']
    
    # Verify sorted order
    hashes = [s['hash'] for s in result1['skipped_hashes']]
    assert hashes == sorted(hashes)


# ============================================================================
# GC-09: Malformed RUN_ROOTS.json fails closed
# ============================================================================

def test_gc_09_malformed_run_roots(isolated_env):
    """GC-09: Malformed RUN_ROOTS.json causes fail-closed behavior"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    
    # Write malformed RUN_ROOTS.json (not a list)
    root_file = runs_dir / "RUN_ROOTS.json"
    with open(root_file, 'w', encoding='utf-8') as f:
        json.dump({"invalid": "format"}, f)
    
    # Run GC
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify fail-closed
    assert len(result['errors']) > 0
    assert 'Root enumeration failed' in result['errors'][0]
    assert result['deleted_hashes'] == []
    
    # Verify no deletions
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()


# ============================================================================
# GC-10: Invalid hash in roots fails closed
# ============================================================================

def test_gc_10_invalid_hash_in_roots(isolated_env):
    """GC-10: Invalid hash format in roots causes fail-closed behavior"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    
    # Write RUN_ROOTS.json with invalid hash
    _write_run_roots(runs_dir, ["invalid_hash"])
    
    # Run GC
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify fail-closed
    assert len(result['errors']) > 0
    assert 'Root enumeration failed' in result['errors'][0]
    assert result['deleted_hashes'] == []
    
    # Verify no deletions
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()


# ============================================================================
# GC-11: CAS snapshot hash is deterministic
# ============================================================================

def test_gc_11_cas_snapshot_deterministic(isolated_env):
    """GC-11: CAS snapshot hash is deterministic"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    
    # Set roots
    _write_run_roots(runs_dir, [h1, h2])
    
    # Run GC twice
    result1 = gc_collect(dry_run=True, cas_root=cas_root, runs_dir=runs_dir)
    result2 = gc_collect(dry_run=True, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify snapshot hash is deterministic
    assert result1['cas_snapshot_hash'] == result2['cas_snapshot_hash']
    assert result1['cas_snapshot_hash'] != ''


# ============================================================================
# GC-12: Policy B - Empty roots fail-closed
# ============================================================================

def test_gc_12_policy_b_empty_roots_fail_closed(isolated_env):
    """GC-12: Policy B - Empty roots with allow_empty_roots=False fails closed"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    
    # No roots
    _write_run_roots(runs_dir, [])
    
    # Run GC with allow_empty_roots=False (default)
    result = gc_collect(dry_run=False, allow_empty_roots=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify fail-closed
    assert result['roots_count'] == 0
    assert result['deleted_hashes'] == []
    assert len(result['errors']) > 0
    assert 'POLICY_LOCK' in result['errors'][0]
    assert 'Empty roots' in result['errors'][0]
    
    # Verify no deletions
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert (cas_root / h2[0] / h2[1:3] / h2).exists()


def test_gc_12_policy_b_empty_roots_override(isolated_env):
    """GC-12: Policy B - Empty roots with allow_empty_roots=True allows full sweep"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    h2 = _put_blob_isolated(b"blob2", cas_root)
    
    # No roots
    _write_run_roots(runs_dir, [])
    
    # Run GC with allow_empty_roots=True (explicit override)
    result = gc_collect(dry_run=False, allow_empty_roots=True, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify full sweep allowed
    assert result['roots_count'] == 0
    assert result['errors'] == []
    assert len(result['deleted_hashes']) == 2
    assert set(result['deleted_hashes']) == {h1, h2}
    
    # Verify all deleted
    assert not (cas_root / h1[0] / h1[1:3] / h1).exists()
    assert not (cas_root / h2[0] / h2[1:3] / h2).exists()


# ============================================================================
# GC-16: Missing root files are not an error
# ============================================================================

def test_gc_16_missing_root_files_ok(isolated_env):
    """GC-16: Missing RUN_ROOTS.json and GC_PINS.json is not an error"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    
    # Don't create any root files
    
    # Run GC with allow_empty_roots=True
    result = gc_collect(dry_run=False, allow_empty_roots=True, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify success (no errors)
    assert result['errors'] == []
    assert result['roots_count'] == 0
    assert result['deleted_hashes'] == [h1]


# ============================================================================
# GC-18: Malformed GC_PINS.json fails closed
# ============================================================================

def test_gc_18_malformed_gc_pins(isolated_env):
    """GC-18: Malformed GC_PINS.json causes fail-closed behavior"""
    cas_root = isolated_env['cas_root']
    runs_dir = isolated_env['runs_dir']
    
    # Create some blobs
    h1 = _put_blob_isolated(b"blob1", cas_root)
    
    # Write valid RUN_ROOTS.json
    _write_run_roots(runs_dir, [h1])
    
    # Write malformed GC_PINS.json (not a list)
    pin_file = runs_dir / "GC_PINS.json"
    with open(pin_file, 'w', encoding='utf-8') as f:
        json.dump({"invalid": "format"}, f)
    
    # Run GC
    result = gc_collect(dry_run=False, cas_root=cas_root, runs_dir=runs_dir)
    
    # Verify fail-closed
    assert len(result['errors']) > 0
    assert 'Root enumeration failed' in result['errors'][0]
    assert result['deleted_hashes'] == []
    
    # Verify no deletions
    assert (cas_root / h1[0] / h1[1:3] / h1).exists()
