"""
Z.2.4 – CAS Deduplication Proof

Tests proving that CAS implements deduplication:
- Identical content stored twice returns the same hash
- On second put, the underlying stored object is NOT rewritten
- Verification via file metadata (mtime)

These tests mechanically prove that Z.2.4 deduplication is satisfied by:
- Content addressing (same bytes -> same hash)
- Write-once semantics (existing objects are not rewritten)
"""

import hashlib
import os
import time
from pathlib import Path

import pytest

from CAPABILITY.CAS.cas import (
    cas_put,
    cas_get,
    _get_object_path,
)


def test_cas_dedup_same_hash():
    """Test that storing identical content twice returns the same hash."""
    data = b"deduplication test data"
    
    # Store the same data twice
    hash1 = cas_put(data)
    hash2 = cas_put(data)
    
    # Must return identical hashes
    assert hash1 == hash2
    
    # Hash must match expected SHA-256
    expected_hash = hashlib.sha256(data).hexdigest()
    assert hash1 == expected_hash
    assert hash2 == expected_hash


def test_cas_dedup_no_rewrite():
    """
    Test that storing identical content twice does NOT rewrite the underlying object.
    
    Proof mechanism: Check file mtime (modification time).
    - First put creates the file with mtime T1
    - Second put should NOT modify the file, so mtime remains T1
    """
    data = b"immutable dedup test"
    
    # First put: store the data
    hash1 = cas_put(data)
    obj_path = _get_object_path(hash1)
    
    # Verify object exists
    assert obj_path.exists()
    
    # Record the modification time after first write
    stat1 = obj_path.stat()
    mtime1 = stat1.st_mtime
    
    # Sleep briefly to ensure time resolution (some filesystems have coarse timestamps)
    time.sleep(0.1)
    
    # Second put: store identical data again
    hash2 = cas_put(data)
    
    # Hash must be identical
    assert hash2 == hash1
    
    # Check modification time after second put
    stat2 = obj_path.stat()
    mtime2 = stat2.st_mtime
    
    # Modification time MUST NOT change (file was not rewritten)
    assert mtime2 == mtime1, (
        f"File was rewritten on second put! "
        f"mtime1={mtime1}, mtime2={mtime2}"
    )


def test_cas_dedup_multiple_puts():
    """Test that multiple puts of the same data never rewrite the object."""
    data = b"multiple put dedup test"
    
    # First put
    hash1 = cas_put(data)
    obj_path = _get_object_path(hash1)
    mtime1 = obj_path.stat().st_mtime
    
    # Sleep to ensure time resolution
    time.sleep(0.1)
    
    # Multiple subsequent puts
    for i in range(5):
        hash_i = cas_put(data)
        assert hash_i == hash1
        
        # Verify mtime has not changed
        mtime_i = obj_path.stat().st_mtime
        assert mtime_i == mtime1, (
            f"File was rewritten on put #{i+2}! "
            f"Original mtime={mtime1}, current mtime={mtime_i}"
        )
        
        # Brief sleep between puts
        time.sleep(0.05)


def test_cas_dedup_different_data_different_hash():
    """Test that different data produces different hashes (sanity check)."""
    data1 = b"first data"
    data2 = b"second data"
    
    hash1 = cas_put(data1)
    hash2 = cas_put(data2)
    
    # Different data must produce different hashes
    assert hash1 != hash2
    
    # Each hash must be correct
    assert hash1 == hashlib.sha256(data1).hexdigest()
    assert hash2 == hashlib.sha256(data2).hexdigest()


def test_cas_dedup_empty_data():
    """Test deduplication with empty data."""
    data = b""
    
    hash1 = cas_put(data)
    obj_path = _get_object_path(hash1)
    mtime1 = obj_path.stat().st_mtime
    
    time.sleep(0.1)
    
    hash2 = cas_put(data)
    mtime2 = obj_path.stat().st_mtime
    
    # Same hash, no rewrite
    assert hash2 == hash1
    assert mtime2 == mtime1


def test_cas_dedup_large_data():
    """Test deduplication with larger data."""
    data = b"x" * 1024 * 1024  # 1MB
    
    hash1 = cas_put(data)
    obj_path = _get_object_path(hash1)
    mtime1 = obj_path.stat().st_mtime
    
    time.sleep(0.1)
    
    hash2 = cas_put(data)
    mtime2 = obj_path.stat().st_mtime
    
    # Same hash, no rewrite
    assert hash2 == hash1
    assert mtime2 == mtime1


def test_cas_dedup_binary_data():
    """Test deduplication with binary data."""
    data = bytes(range(256))
    
    hash1 = cas_put(data)
    obj_path = _get_object_path(hash1)
    mtime1 = obj_path.stat().st_mtime
    
    time.sleep(0.1)
    
    hash2 = cas_put(data)
    mtime2 = obj_path.stat().st_mtime
    
    # Same hash, no rewrite
    assert hash2 == hash1
    assert mtime2 == mtime1


def test_cas_dedup_retrieval_after_dedup():
    """Test that data can be retrieved correctly after deduplication."""
    data = b"retrieve after dedup"
    
    # Store twice
    hash1 = cas_put(data)
    hash2 = cas_put(data)
    
    assert hash1 == hash2
    
    # Retrieve and verify
    retrieved = cas_get(hash1)
    assert retrieved == data
    
    # Retrieve using second hash (which is identical)
    retrieved2 = cas_get(hash2)
    assert retrieved2 == data
