"""
Z.2.4 – Artifact Store Deduplication Proof

Tests proving that the artifact store implements deduplication:
- store_bytes with identical content twice returns the same sha256 ref
- store_file on identical files returns the same sha256 ref

These tests mechanically prove that Z.2.4 deduplication is satisfied by:
- Content addressing (same bytes -> same sha256 ref)
- CAS write-once semantics (inherited from Z.2.1)
"""

import hashlib
import tempfile
from pathlib import Path

import pytest

from CAPABILITY.ARTIFACTS.store import (
    store_bytes,
    store_file,
    load_bytes,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test: store_bytes deduplication
# ============================================================================

def test_artifact_store_bytes_dedup_same_ref():
    """Test that storing identical bytes twice returns the same sha256 ref."""
    data = b"artifact dedup test data"
    
    # Store the same bytes twice
    ref1 = store_bytes(data)
    ref2 = store_bytes(data)
    
    # Must return identical references
    assert ref1 == ref2
    
    # Reference must be in correct format
    assert ref1.startswith("sha256:")
    assert len(ref1) == len("sha256:") + 64
    
    # Hash portion must match expected SHA-256
    expected_hash = hashlib.sha256(data).hexdigest()
    expected_ref = f"sha256:{expected_hash}"
    assert ref1 == expected_ref
    assert ref2 == expected_ref


def test_artifact_store_bytes_dedup_multiple():
    """Test that multiple store_bytes calls with identical data return the same ref."""
    data = b"multiple store dedup test"
    
    # Store the same data multiple times
    refs = [store_bytes(data) for _ in range(10)]
    
    # All references must be identical
    assert len(set(refs)) == 1, "All refs should be identical"
    
    # Verify the reference is correct
    expected_hash = hashlib.sha256(data).hexdigest()
    expected_ref = f"sha256:{expected_hash}"
    assert refs[0] == expected_ref


def test_artifact_store_bytes_dedup_empty():
    """Test deduplication with empty bytes."""
    data = b""
    
    ref1 = store_bytes(data)
    ref2 = store_bytes(data)
    
    assert ref1 == ref2
    
    # Verify it's the correct hash for empty bytes
    expected_hash = hashlib.sha256(b"").hexdigest()
    assert ref1 == f"sha256:{expected_hash}"


def test_artifact_store_bytes_dedup_large():
    """Test deduplication with larger data."""
    data = b"y" * 1024 * 1024  # 1MB
    
    ref1 = store_bytes(data)
    ref2 = store_bytes(data)
    
    assert ref1 == ref2


def test_artifact_store_bytes_dedup_binary():
    """Test deduplication with binary data."""
    data = bytes(range(256))
    
    ref1 = store_bytes(data)
    ref2 = store_bytes(data)
    
    assert ref1 == ref2


def test_artifact_store_bytes_different_data():
    """Test that different data produces different refs (sanity check)."""
    data1 = b"first artifact"
    data2 = b"second artifact"
    
    ref1 = store_bytes(data1)
    ref2 = store_bytes(data2)
    
    # Different data must produce different refs
    assert ref1 != ref2


# ============================================================================
# Test: store_file deduplication
# ============================================================================

def test_artifact_store_file_dedup_same_ref(temp_dir):
    """Test that storing identical files returns the same sha256 ref."""
    data = b"file dedup test content"
    
    # Create two separate files with identical content
    file1 = temp_dir / "file1.txt"
    file2 = temp_dir / "file2.txt"
    file1.write_bytes(data)
    file2.write_bytes(data)
    
    # Store both files
    ref1 = store_file(str(file1))
    ref2 = store_file(str(file2))
    
    # Must return identical references
    assert ref1 == ref2
    
    # Reference must match expected hash
    expected_hash = hashlib.sha256(data).hexdigest()
    expected_ref = f"sha256:{expected_hash}"
    assert ref1 == expected_ref


def test_artifact_store_file_dedup_same_file_multiple_times(temp_dir):
    """Test that storing the same file multiple times returns the same ref."""
    data = b"same file multiple stores"
    
    file_path = temp_dir / "test.txt"
    file_path.write_bytes(data)
    
    # Store the same file multiple times
    refs = [store_file(str(file_path)) for _ in range(5)]
    
    # All references must be identical
    assert len(set(refs)) == 1


def test_artifact_store_file_dedup_different_paths_same_content(temp_dir):
    """Test that files with identical content but different paths return the same ref."""
    data = b"identical content different paths"
    
    # Create files in different subdirectories
    subdir1 = temp_dir / "dir1"
    subdir2 = temp_dir / "dir2"
    subdir1.mkdir()
    subdir2.mkdir()
    
    file1 = subdir1 / "file.txt"
    file2 = subdir2 / "file.txt"
    file1.write_bytes(data)
    file2.write_bytes(data)
    
    ref1 = store_file(str(file1))
    ref2 = store_file(str(file2))
    
    # Same content -> same ref, regardless of path
    assert ref1 == ref2


def test_artifact_store_file_dedup_different_names_same_content(temp_dir):
    """Test that files with different names but identical content return the same ref."""
    data = b"same content different names"
    
    file1 = temp_dir / "alpha.txt"
    file2 = temp_dir / "beta.dat"
    file3 = temp_dir / "gamma.bin"
    
    file1.write_bytes(data)
    file2.write_bytes(data)
    file3.write_bytes(data)
    
    ref1 = store_file(str(file1))
    ref2 = store_file(str(file2))
    ref3 = store_file(str(file3))
    
    # All must return the same ref
    assert ref1 == ref2 == ref3


def test_artifact_store_file_different_content(temp_dir):
    """Test that files with different content produce different refs (sanity check)."""
    file1 = temp_dir / "file1.txt"
    file2 = temp_dir / "file2.txt"
    
    file1.write_bytes(b"content one")
    file2.write_bytes(b"content two")
    
    ref1 = store_file(str(file1))
    ref2 = store_file(str(file2))
    
    # Different content must produce different refs
    assert ref1 != ref2


# ============================================================================
# Test: Cross-function deduplication
# ============================================================================

def test_artifact_store_bytes_and_file_dedup(temp_dir):
    """Test that store_bytes and store_file deduplicate to the same ref."""
    data = b"cross-function dedup test"
    
    # Store via store_bytes
    ref_bytes = store_bytes(data)
    
    # Store via store_file
    file_path = temp_dir / "test.txt"
    file_path.write_bytes(data)
    ref_file = store_file(str(file_path))
    
    # Both must return the same ref (same content -> same hash)
    assert ref_bytes == ref_file
    
    # Verify both can be loaded and return identical data
    loaded_bytes = load_bytes(ref_bytes)
    loaded_file = load_bytes(ref_file)
    
    assert loaded_bytes == data
    assert loaded_file == data
    assert loaded_bytes == loaded_file


def test_artifact_store_mixed_dedup(temp_dir):
    """Test deduplication across mixed store_bytes and store_file calls."""
    data = b"mixed dedup scenario"
    
    # Create a file
    file_path = temp_dir / "mixed.txt"
    file_path.write_bytes(data)
    
    # Interleave store_bytes and store_file calls
    ref1 = store_bytes(data)
    ref2 = store_file(str(file_path))
    ref3 = store_bytes(data)
    ref4 = store_file(str(file_path))
    
    # All must return the same ref
    assert ref1 == ref2 == ref3 == ref4


# ============================================================================
# Test: Deduplication with retrieval
# ============================================================================

def test_artifact_dedup_retrieval():
    """Test that deduplicated artifacts can be retrieved correctly."""
    data = b"retrieval after dedup"
    
    # Store twice
    ref1 = store_bytes(data)
    ref2 = store_bytes(data)
    
    assert ref1 == ref2
    
    # Load from both refs (which are identical)
    loaded1 = load_bytes(ref1)
    loaded2 = load_bytes(ref2)
    
    # Both must return the original data
    assert loaded1 == data
    assert loaded2 == data
    assert loaded1 == loaded2
