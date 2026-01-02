import hashlib
import os
from pathlib import Path
from typing import Union


class CASException(Exception):
    """Base exception for CAS operations"""
    pass


class InvalidHashException(CASException):
    """Raised when a hash format is invalid"""
    pass


class ObjectNotFoundException(CASException):
    """Raised when an object is not found in CAS"""
    pass


class CorruptObjectException(CASException):
    """Raised when an object is corrupted (hash mismatch)"""
    pass


# Default CAS root directory
_CAS_ROOT = Path("CAPABILITY/CAS/storage")


def _get_object_path(hash_str: str) -> Path:
    """Get the path for an object based on its hash.
    
    Uses a prefix directory structure to avoid having too many files in one directory.
    For example: abcdef... -> a/bc/abcdef...
    """
    if len(hash_str) != 64 or not all(c in '0123456789abcdef' for c in hash_str):
        raise InvalidHashException(f"Invalid hash format: {hash_str}")
    
    # Create path: first char / next 2 chars / full hash
    prefix1 = hash_str[0]
    prefix2 = hash_str[1:3]
    return _CAS_ROOT / prefix1 / prefix2 / hash_str


def cas_put(data: bytes) -> str:
    """Store data in CAS and return its hash.
    
    Args:
        data: The bytes to store
        
    Returns:
        The SHA-256 hash of the data (lowercase hex)
        
    Raises:
        CASException: If there's an error during storage
    """
    # Calculate the hash of the data
    hash_obj = hashlib.sha256(data)
    hash_str = hash_obj.hexdigest()
    
    # Get the path where the object should be stored
    obj_path = _get_object_path(hash_str)
    
    # Create directories if they don't exist
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if object already exists
    if obj_path.exists():
        # If it exists, return the hash without rewriting
        return hash_str
    
    # Write data atomically to a temporary file first
    temp_path = obj_path.with_suffix(obj_path.suffix + '.tmp')
    try:
        with open(temp_path, 'wb') as f:
            f.write(data)
        
        # Atomically move the temp file to the final location
        temp_path.replace(obj_path)
        
        # Re-read and verify integrity
        with open(obj_path, 'rb') as f:
            read_data = f.read()
        
        # Verify the hash of the stored data
        verify_hash = hashlib.sha256(read_data).hexdigest()
        if verify_hash != hash_str:
            # If verification fails, remove the corrupted file
            obj_path.unlink(missing_ok=True)
            raise CorruptObjectException(f"Stored data verification failed for hash: {hash_str}")
        
        return hash_str
    except Exception:
        # Clean up temp file if something went wrong
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def cas_get(hash_str: str) -> bytes:
    """Retrieve data from CAS by its hash.
    
    Args:
        hash_str: The SHA-256 hash (lowercase hex) of the data to retrieve
        
    Returns:
        The stored bytes
        
    Raises:
        InvalidHashException: If the hash format is invalid
        ObjectNotFoundException: If the object is not found
        CorruptObjectException: If the object is corrupted (hash mismatch)
    """
    # Validate hash format
    if len(hash_str) != 64 or not all(c in '0123456789abcdef' for c in hash_str):
        raise InvalidHashException(f"Invalid hash format: {hash_str}")
    
    # Get the path of the object
    obj_path = _get_object_path(hash_str)
    
    # Check if the object exists
    if not obj_path.exists():
        raise ObjectNotFoundException(f"Object not found: {hash_str}")
    
    # Read the data
    try:
        with open(obj_path, 'rb') as f:
            data = f.read()
    except IOError as e:
        raise CASException(f"Failed to read object {hash_str}: {str(e)}")
    
    # Verify the integrity of the data by hashing it and comparing with the expected hash
    calculated_hash = hashlib.sha256(data).hexdigest()
    if calculated_hash != hash_str:
        raise CorruptObjectException(f"Corruption detected for hash: {hash_str}")
    
    return data