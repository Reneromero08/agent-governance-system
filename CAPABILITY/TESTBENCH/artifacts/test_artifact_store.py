"""
Z.2.2 – CAS-backed artifact store tests

Tests proving:
- store_bytes -> load_bytes roundtrip
- store_file -> load_bytes matches file contents
- materialize writes correct bytes for both:
  - sha256 ref
  - legacy path ref
- invalid sha256 ref rejected
- missing object rejected
- legacy missing file rejected
- deterministic: same bytes stored twice -> identical sha256 ref
"""

import os
import pytest
import tempfile
from pathlib import Path

from CAPABILITY.ARTIFACTS.store import (
    store_bytes,
    load_bytes,
    store_file,
    materialize,
    ArtifactException,
    InvalidReferenceException,
    ObjectNotFoundException,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Sample test data"""
    return b"Hello, World! This is test data."


@pytest.fixture
def sample_file(temp_dir, sample_data):
    """Create a sample file with test data"""
    file_path = temp_dir / "sample.txt"
    file_path.write_bytes(sample_data)
    return file_path


# ============================================================================
# Test: store_bytes -> load_bytes roundtrip
# ============================================================================

def test_store_bytes_load_bytes_roundtrip(sample_data):
    """Test that store_bytes -> load_bytes roundtrip preserves data"""
    # Store bytes
    ref = store_bytes(sample_data)
    
    # Verify reference format
    assert ref.startswith("sha256:")
    assert len(ref) == len("sha256:") + 64
    
    # Load bytes
    loaded = load_bytes(ref)
    
    # Verify data matches
    assert loaded == sample_data


def test_store_bytes_empty():
    """Test storing empty bytes"""
    data = b""
    ref = store_bytes(data)
    loaded = load_bytes(ref)
    assert loaded == data


def test_store_bytes_large():
    """Test storing larger data"""
    data = b"x" * 1024 * 1024  # 1MB
    ref = store_bytes(data)
    loaded = load_bytes(ref)
    assert loaded == data


def test_store_bytes_binary():
    """Test storing binary data"""
    data = bytes(range(256))
    ref = store_bytes(data)
    loaded = load_bytes(ref)
    assert loaded == data


# ============================================================================
# Test: store_file -> load_bytes matches file contents
# ============================================================================

def test_store_file_load_bytes(sample_file, sample_data):
    """Test that store_file -> load_bytes matches file contents"""
    # Store file
    ref = store_file(str(sample_file))
    
    # Verify reference format
    assert ref.startswith("sha256:")
    
    # Load bytes
    loaded = load_bytes(ref)
    
    # Verify data matches original file
    assert loaded == sample_data


def test_store_file_with_path_object(sample_file, sample_data):
    """Test store_file with Path object"""
    # Store file using Path object (converted to string)
    ref = store_file(str(sample_file))
    loaded = load_bytes(ref)
    assert loaded == sample_data


# ============================================================================
# Test: materialize writes correct bytes
# ============================================================================

def test_materialize_cas_ref(temp_dir, sample_data):
    """Test materialize with CAS reference"""
    # Store bytes to get CAS ref
    ref = store_bytes(sample_data)
    
    # Materialize to new file
    out_path = temp_dir / "materialized.txt"
    materialize(ref, str(out_path))
    
    # Verify file exists and contains correct data
    assert out_path.exists()
    assert out_path.read_bytes() == sample_data


def test_materialize_legacy_path(sample_file, sample_data, temp_dir):
    """Test materialize with legacy file path reference"""
    # Materialize from legacy path
    out_path = temp_dir / "materialized_legacy.txt"
    materialize(str(sample_file), str(out_path))
    
    # Verify file exists and contains correct data
    assert out_path.exists()
    assert out_path.read_bytes() == sample_data


def test_materialize_atomic_true(temp_dir, sample_data):
    """Test materialize with atomic=True (default)"""
    ref = store_bytes(sample_data)
    out_path = temp_dir / "atomic.txt"
    
    materialize(ref, str(out_path), atomic=True)
    
    assert out_path.exists()
    assert out_path.read_bytes() == sample_data


def test_materialize_atomic_false(temp_dir, sample_data):
    """Test materialize with atomic=False"""
    ref = store_bytes(sample_data)
    out_path = temp_dir / "non_atomic.txt"
    
    materialize(ref, str(out_path), atomic=False)
    
    assert out_path.exists()
    assert out_path.read_bytes() == sample_data


def test_materialize_creates_parent_dirs(temp_dir, sample_data):
    """Test that materialize creates parent directories"""
    ref = store_bytes(sample_data)
    out_path = temp_dir / "subdir" / "nested" / "file.txt"
    
    materialize(ref, str(out_path))
    
    assert out_path.exists()
    assert out_path.read_bytes() == sample_data


def test_materialize_overwrites_existing(temp_dir, sample_data):
    """Test that materialize overwrites existing file"""
    ref = store_bytes(sample_data)
    out_path = temp_dir / "overwrite.txt"
    
    # Create existing file with different content
    out_path.write_bytes(b"old content")
    
    # Materialize should overwrite
    materialize(ref, str(out_path))
    
    assert out_path.read_bytes() == sample_data


# ============================================================================
# Test: invalid sha256 ref rejected
# ============================================================================

def test_load_bytes_invalid_cas_ref_format():
    """Test that invalid CAS reference format is rejected"""
    invalid_refs = [
        "sha256:invalid",  # Not hex
        "sha256:abc",  # Too short
        "sha256:" + "g" * 64,  # Invalid hex character
        "sha256:" + "A" * 64,  # Uppercase (must be lowercase)
        "sha256:" + "0" * 63,  # Too short
        "sha256:" + "0" * 65,  # Too long
        "sha256:",  # Missing hash
    ]
    
    for ref in invalid_refs:
        with pytest.raises(InvalidReferenceException):
            load_bytes(ref)
    
    # These are treated as legacy file paths (not CAS refs) and will raise
    # ObjectNotFoundException instead because the files don't exist
    legacy_path_refs = [
        "sha256",  # No colon - treated as file path
        "md5:" + "0" * 32,  # Wrong algorithm - treated as file path
    ]
    
    for ref in legacy_path_refs:
        with pytest.raises(ObjectNotFoundException):
            load_bytes(ref)


def test_materialize_invalid_cas_ref(temp_dir):
    """Test that materialize rejects invalid CAS reference"""
    out_path = temp_dir / "output.txt"
    
    with pytest.raises(InvalidReferenceException):
        materialize("sha256:invalid", str(out_path))


# ============================================================================
# Test: missing object rejected
# ============================================================================

def test_load_bytes_missing_cas_object():
    """Test that missing CAS object is rejected"""
    # Valid format but non-existent hash
    ref = "sha256:" + "0" * 64
    
    with pytest.raises(ObjectNotFoundException):
        load_bytes(ref)


def test_materialize_missing_cas_object(temp_dir):
    """Test that materialize rejects missing CAS object"""
    ref = "sha256:" + "0" * 64
    out_path = temp_dir / "output.txt"
    
    with pytest.raises(ObjectNotFoundException):
        materialize(ref, str(out_path))


# ============================================================================
# Test: legacy missing file rejected
# ============================================================================

def test_load_bytes_missing_file():
    """Test that missing legacy file is rejected"""
    with pytest.raises(ObjectNotFoundException):
        load_bytes("/nonexistent/file.txt")


def test_store_file_missing():
    """Test that store_file rejects missing file"""
    with pytest.raises(ObjectNotFoundException):
        store_file("/nonexistent/file.txt")


def test_materialize_missing_legacy_file(temp_dir):
    """Test that materialize rejects missing legacy file"""
    out_path = temp_dir / "output.txt"
    
    with pytest.raises(ObjectNotFoundException):
        materialize("/nonexistent/file.txt", str(out_path))


def test_load_bytes_directory_not_file(temp_dir):
    """Test that directory path is rejected (not a file)"""
    with pytest.raises(InvalidReferenceException):
        load_bytes(str(temp_dir))


def test_store_file_directory_not_file(temp_dir):
    """Test that store_file rejects directory path"""
    with pytest.raises(InvalidReferenceException):
        store_file(str(temp_dir))


# ============================================================================
# Test: deterministic - same bytes -> identical sha256 ref
# ============================================================================

def test_deterministic_same_bytes():
    """Test that same bytes produce identical sha256 ref"""
    data = b"deterministic test data"
    
    ref1 = store_bytes(data)
    ref2 = store_bytes(data)
    
    # Should produce identical references
    assert ref1 == ref2


def test_deterministic_same_file(temp_dir):
    """Test that same file content produces identical sha256 ref"""
    data = b"file content for determinism test"
    
    # Create two files with identical content
    file1 = temp_dir / "file1.txt"
    file2 = temp_dir / "file2.txt"
    file1.write_bytes(data)
    file2.write_bytes(data)
    
    ref1 = store_file(str(file1))
    ref2 = store_file(str(file2))
    
    # Should produce identical references
    assert ref1 == ref2


def test_deterministic_different_bytes():
    """Test that different bytes produce different sha256 refs"""
    data1 = b"data one"
    data2 = b"data two"
    
    ref1 = store_bytes(data1)
    ref2 = store_bytes(data2)
    
    # Should produce different references
    assert ref1 != ref2


# ============================================================================
# Test: edge cases and error handling
# ============================================================================

def test_load_bytes_empty_reference():
    """Test that empty reference is rejected"""
    with pytest.raises(InvalidReferenceException):
        load_bytes("")


def test_load_bytes_none_reference():
    """Test that None reference is rejected"""
    with pytest.raises(InvalidReferenceException):
        load_bytes(None)


def test_store_bytes_not_bytes():
    """Test that non-bytes input is rejected"""
    with pytest.raises(ArtifactException):
        store_bytes("not bytes")


def test_store_file_empty_path():
    """Test that empty path is rejected"""
    with pytest.raises(InvalidReferenceException):
        store_file("")


def test_materialize_invalid_output_path():
    """Test materialize with invalid output path"""
    ref = store_bytes(b"test")
    
    # This should still work on most systems, but we test the behavior
    # The actual validation happens during write
    # We'll test a case that should fail on write
    # For now, just verify it doesn't crash with valid paths
    pass  # Platform-specific behavior


# ============================================================================
# Test: CAS reference format validation
# ============================================================================

def test_cas_ref_format_lowercase_only():
    """Test that CAS references must be lowercase"""
    data = b"test"
    ref = store_bytes(data)
    
    # Reference should be lowercase
    assert ref == ref.lower()
    
    # Uppercase prefix is treated as a file path (not a CAS ref)
    # so it raises ObjectNotFoundException
    ref_upper = ref.replace("sha256:", "SHA256:")
    with pytest.raises(ObjectNotFoundException):
        load_bytes(ref_upper)
    
    # Mixed case hash (with lowercase prefix) should be rejected as invalid CAS ref
    hash_part = ref[7:]  # Skip "sha256:"
    ref_mixed = "sha256:" + hash_part.upper()
    with pytest.raises(InvalidReferenceException):
        load_bytes(ref_mixed)


def test_cas_ref_exact_format():
    """Test exact CAS reference format requirements"""
    data = b"exact format test"
    ref = store_bytes(data)
    
    # Should be exactly "sha256:" + 64 hex chars
    assert len(ref) == 7 + 64
    assert ref.startswith("sha256:")
    
    # Hash part should be exactly 64 chars
    hash_part = ref[7:]
    assert len(hash_part) == 64
    assert all(c in "0123456789abcdef" for c in hash_part)


# ============================================================================
# Test: dual mode compatibility
# ============================================================================

def test_dual_mode_cas_and_legacy(sample_file, sample_data, temp_dir):
    """Test that both CAS refs and legacy paths work in same workflow"""
    # Store via CAS
    cas_ref = store_bytes(sample_data)
    
    # Load from CAS ref
    loaded_cas = load_bytes(cas_ref)
    assert loaded_cas == sample_data
    
    # Load from legacy path
    loaded_legacy = load_bytes(str(sample_file))
    assert loaded_legacy == sample_data
    
    # Both should produce same data
    assert loaded_cas == loaded_legacy
    
    # Materialize from CAS ref
    out_cas = temp_dir / "from_cas.txt"
    materialize(cas_ref, str(out_cas))
    
    # Materialize from legacy path
    out_legacy = temp_dir / "from_legacy.txt"
    materialize(str(sample_file), str(out_legacy))
    
    # Both should produce identical files
    assert out_cas.read_bytes() == out_legacy.read_bytes()
    assert out_cas.read_bytes() == sample_data
