"""
Z.2.2 – CAS-backed artifact store

Replaces artifact file-path references with content hashes (CAS addresses),
while preserving backward compatibility.

Artifact reference format:
- CAS refs: "sha256:<64-lowercase-hex>"
- Legacy paths: plain strings (no prefix)

Public API:
1. store_bytes(data: bytes) -> str
   - Stores bytes into CAS and returns "sha256:<hash>"

2. load_bytes(ref: str) -> bytes
   - If ref starts with "sha256:", resolve from CAS (validate hash format; fail closed).
   - Otherwise treat ref as a file path and read bytes from disk (fail closed if missing).

3. store_file(path: str) -> str
   - Reads bytes from path and stores into CAS, returning "sha256:<hash>"

4. materialize(ref: str, out_path: str, *, atomic: bool = True) -> None
   - Writes bytes referenced by ref into out_path.
   - If atomic=True, write to temp then replace.
   - Deterministic, fail closed.

Required behavior:
- Deterministic: same bytes -> same sha256 ref.
- Strict validation: CAS refs must be exactly sha256:<64hex>.
- Fail closed: invalid ref, missing object, or read errors must raise explicit exceptions.
- No silent fallbacks. No "best effort."
"""

import os
import re
from pathlib import Path
from typing import Union

# Import CAS primitives from Z.2.1
from CAPABILITY.CAS.cas import (
    cas_put,
    cas_get,
    InvalidHashException,
    ObjectNotFoundException as CASObjectNotFoundException,
    CorruptObjectException,
)


# ============================================================================
# Exceptions
# ============================================================================

class ArtifactException(Exception):
    """Base exception for artifact store operations"""
    pass


class InvalidReferenceException(ArtifactException):
    """Raised when an artifact reference format is invalid"""
    pass


class ObjectNotFoundException(ArtifactException):
    """Raised when an artifact object is not found"""
    pass


# ============================================================================
# Constants
# ============================================================================

# CAS reference format: "sha256:<64-lowercase-hex>"
CAS_REF_PATTERN = re.compile(r'^sha256:[0-9a-f]{64}$')
CAS_REF_PREFIX = "sha256:"


# ============================================================================
# Reference validation
# ============================================================================

def _is_cas_ref(ref: str) -> bool:
    """Check if a reference is a CAS reference (sha256:...)"""
    return ref.startswith(CAS_REF_PREFIX)


def _validate_cas_ref(ref: str) -> str:
    """
    Validate CAS reference format and extract hash.
    
    Args:
        ref: Reference string to validate
        
    Returns:
        The hash portion (without "sha256:" prefix)
        
    Raises:
        InvalidReferenceException: If format is invalid
    """
    if not CAS_REF_PATTERN.match(ref):
        raise InvalidReferenceException(
            f"Invalid CAS reference format: {ref}. "
            f"Expected 'sha256:<64-lowercase-hex>'"
        )
    return ref[len(CAS_REF_PREFIX):]


def _validate_file_path(path: str) -> Path:
    """
    Validate file path reference.
    
    Args:
        path: File path string
        
    Returns:
        Path object
        
    Raises:
        InvalidReferenceException: If path is invalid
        ObjectNotFoundException: If file does not exist
    """
    if not path:
        raise InvalidReferenceException("Empty file path")
    
    try:
        p = Path(path)
    except Exception as e:
        raise InvalidReferenceException(f"Invalid file path: {path}: {e}")
    
    if not p.exists():
        raise ObjectNotFoundException(f"File not found: {path}")
    
    if not p.is_file():
        raise InvalidReferenceException(f"Path is not a file: {path}")
    
    return p


# ============================================================================
# Public API
# ============================================================================

def store_bytes(data: bytes) -> str:
    """
    Store bytes into CAS and return CAS reference.
    
    Args:
        data: Bytes to store
        
    Returns:
        CAS reference in format "sha256:<hash>"
        
    Raises:
        ArtifactException: If storage fails
    """
    if not isinstance(data, bytes):
        raise ArtifactException(f"Expected bytes, got {type(data).__name__}")
    
    try:
        hash_hex = cas_put(data)
        return f"{CAS_REF_PREFIX}{hash_hex}"
    except Exception as e:
        raise ArtifactException(f"Failed to store bytes: {e}")


def load_bytes(ref: str) -> bytes:
    """
    Load bytes from artifact reference.
    
    If ref starts with "sha256:", resolve from CAS (validate hash format; fail closed).
    Otherwise treat ref as a file path and read bytes from disk (fail closed if missing).
    
    Args:
        ref: Artifact reference (CAS ref or file path)
        
    Returns:
        Bytes content
        
    Raises:
        InvalidReferenceException: If reference format is invalid
        ObjectNotFoundException: If object is not found
        ArtifactException: If read fails
    """
    if not isinstance(ref, str):
        raise InvalidReferenceException(f"Expected string reference, got {type(ref).__name__}")
    
    if not ref:
        raise InvalidReferenceException("Empty reference")
    
    # CAS reference path
    if _is_cas_ref(ref):
        hash_hex = _validate_cas_ref(ref)
        try:
            return cas_get(hash_hex)
        except InvalidHashException as e:
            raise InvalidReferenceException(f"Invalid hash in CAS reference: {e}")
        except CASObjectNotFoundException as e:
            raise ObjectNotFoundException(f"CAS object not found: {ref}")
        except CorruptObjectException as e:
            raise ArtifactException(f"CAS object corrupted: {ref}: {e}")
        except Exception as e:
            raise ArtifactException(f"Failed to load from CAS: {ref}: {e}")
    
    # Legacy file path
    else:
        file_path = _validate_file_path(ref)
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise ArtifactException(f"Failed to read file: {ref}: {e}")


def store_file(path: str) -> str:
    """
    Read bytes from file path and store into CAS.
    
    Args:
        path: File path to read
        
    Returns:
        CAS reference in format "sha256:<hash>"
        
    Raises:
        InvalidReferenceException: If path is invalid
        ObjectNotFoundException: If file does not exist
        ArtifactException: If read or store fails
    """
    file_path = _validate_file_path(path)
    
    try:
        data = file_path.read_bytes()
    except Exception as e:
        raise ArtifactException(f"Failed to read file: {path}: {e}")
    
    return store_bytes(data)


def materialize(ref: str, out_path: str, *, atomic: bool = True) -> None:
    """
    Write bytes referenced by ref into out_path.
    
    Args:
        ref: Artifact reference (CAS ref or file path)
        out_path: Output file path
        atomic: If True, write to temp then replace (default: True)
        
    Raises:
        InvalidReferenceException: If reference format is invalid
        ObjectNotFoundException: If object is not found
        ArtifactException: If write fails
    """
    # Load bytes from reference
    data = load_bytes(ref)
    
    # Validate output path
    try:
        out = Path(out_path)
    except Exception as e:
        raise ArtifactException(f"Invalid output path: {out_path}: {e}")
    
    # Create parent directories if needed
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ArtifactException(f"Failed to create parent directories: {out_path}: {e}")
    
    # Write bytes
    if atomic:
        # Atomic write: write to temp then replace
        temp_path = out.with_suffix(out.suffix + '.tmp')
        try:
            temp_path.write_bytes(data)
            # Atomic replace
            temp_path.replace(out)
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise ArtifactException(f"Failed to write file atomically: {out_path}: {e}")
    else:
        # Direct write
        try:
            out.write_bytes(data)
        except Exception as e:
            raise ArtifactException(f"Failed to write file: {out_path}: {e}")
