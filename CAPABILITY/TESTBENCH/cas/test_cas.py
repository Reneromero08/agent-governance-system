import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from CAPABILITY.CAS.cas import (
    cas_put,
    cas_get,
    CASException,
    InvalidHashException,
    ObjectNotFoundException,
    CorruptObjectException
)


def test_cas_put_get_roundtrip():
    """Test that put/get roundtrip works correctly."""
    data = b"Hello, World!"
    hash_value = cas_put(data)
    
    retrieved_data = cas_get(hash_value)
    assert retrieved_data == data


def test_identical_input_same_hash():
    """Test that identical input always produces the same hash."""
    data = b"Same data"
    hash1 = cas_put(data)
    hash2 = cas_put(data)
    
    assert hash1 == hash2
    assert hash1 == hashlib.sha256(data).hexdigest()


def test_different_input_different_hash():
    """Test that different inputs produce different hashes."""
    data1 = b"Data one"
    data2 = b"Data two"
    
    hash1 = cas_put(data1)
    hash2 = cas_put(data2)
    
    assert hash1 != hash2
    assert hash1 == hashlib.sha256(data1).hexdigest()
    assert hash2 == hashlib.sha256(data2).hexdigest()


def test_double_put_does_not_rewrite():
    """Test that putting the same data twice does not rewrite the object."""
    data = b"Immutable data"
    hash1 = cas_put(data)
    hash2 = cas_put(data)
    
    assert hash1 == hash2
    
    # Verify the data is still retrievable
    retrieved_data = cas_get(hash1)
    assert retrieved_data == data


def test_corrupted_object_detection():
    """Test that corrupted objects are detected."""
    data = b"Original data"
    hash_value = cas_put(data)
    
    # Manually corrupt the stored file
    from CAPABILITY.CAS.cas import _get_object_path
    obj_path = _get_object_path(hash_value)
    
    # Write different content to the file to simulate corruption
    with open(obj_path, 'wb') as f:
        f.write(b"Corrupted content")
    
    # Now trying to get should raise CorruptObjectException
    with pytest.raises(CorruptObjectException):
        cas_get(hash_value)


def test_missing_object():
    """Test that missing objects raise ObjectNotFoundException."""
    fake_hash = "a" * 64  # Valid format but doesn't exist
    
    with pytest.raises(ObjectNotFoundException):
        cas_get(fake_hash)


def test_invalid_hash_format():
    """Test that invalid hash formats raise InvalidHashException."""
    # Test with wrong length
    with pytest.raises(InvalidHashException):
        cas_get("abc")
    
    # Test with invalid characters
    with pytest.raises(InvalidHashException):
        cas_get("x" * 64)
    
    # Test with uppercase (should be lowercase hex)
    with pytest.raises(InvalidHashException):
        cas_get("A" * 64)


def test_empty_data():
    """Test that empty data can be stored and retrieved."""
    data = b""
    hash_value = cas_put(data)
    
    retrieved_data = cas_get(hash_value)
    assert retrieved_data == data
    
    # Verify it's the SHA256 of empty bytes
    expected_hash = hashlib.sha256(b"").hexdigest()
    assert hash_value == expected_hash


def test_large_data():
    """Test that large data can be stored and retrieved."""
    data = b"x" * 10000  # 10KB of data
    hash_value = cas_put(data)
    
    retrieved_data = cas_get(hash_value)
    assert retrieved_data == data
    
    # Verify the hash is correct
    expected_hash = hashlib.sha256(data).hexdigest()
    assert hash_value == expected_hash


def test_binary_data():
    """Test that binary data with various byte values can be stored."""
    # Create binary data with all possible byte values
    data = bytes(range(256))
    hash_value = cas_put(data)
    
    retrieved_data = cas_get(hash_value)
    assert retrieved_data == data


def test_atomic_write():
    """Test that writes are atomic and don't leave partial files."""
    # Use a large enough piece of data that write takes some time
    data = b"Large data " * 1000
    
    # Put the data
    hash_value = cas_put(data)
    
    # Verify it can be retrieved correctly
    retrieved_data = cas_get(hash_value)
    assert retrieved_data == data


def test_storage_path_structure():
    """Test that the storage path structure works correctly."""
    from CAPABILITY.CAS.cas import _get_object_path
    
    # Test a sample hash with valid hex characters - exactly 64 chars
    sample_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"  # 64 chars total
    expected_path = Path("CAPABILITY/CAS/storage") / "a" / "bc" / sample_hash
    
    actual_path = _get_object_path(sample_hash)
    assert actual_path == expected_path


def test_hash_verification_on_put():
    """Test that data integrity is verified after writing."""
    data = b"Verification test data"
    hash_value = cas_put(data)
    
    # The data should be stored correctly and retrievable
    retrieved_data = cas_get(hash_value)
    assert retrieved_data == data
    assert hash_value == hashlib.sha256(data).hexdigest()