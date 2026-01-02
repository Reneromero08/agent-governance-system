"""
Z.2.3 – Immutable run artifacts

Defines and implements immutable CAS-backed run records for:
- TASK_SPEC: Immutable bytes representing the exact task input
- STATUS: Small structured record describing run state
- OUTPUT_HASHES: Deterministic ordered list of CAS hashes produced by the run

All artifacts are:
- Immutable (no updates, no overwrites, no mutation)
- Deterministic (same input -> same CAS hash)
- Fail-closed (invalid input or corrupted data raises exceptions)
- No side effects outside CAS

Public API:
- put_task_spec(spec: dict) -> str
- put_status(status: dict) -> str
- put_output_hashes(hashes: list[str]) -> str
- load_task_spec(hash: str) -> dict
- load_status(hash: str) -> dict
- load_output_hashes(hash: str) -> list[str]
"""

import json
from typing import Any

from CAPABILITY.CAS.cas import cas_put, cas_get, InvalidHashException, ObjectNotFoundException, CorruptObjectException


# ============================================================================
# Exceptions
# ============================================================================

class RunRecordException(Exception):
    """Base exception for run record operations"""
    pass


class InvalidInputException(RunRecordException):
    """Raised when input data is invalid"""
    pass


# ============================================================================
# Canonical encoding
# ============================================================================

def _canonical_encode(obj: Any) -> bytes:
    """
    Encode an object to canonical JSON bytes.

    Ensures deterministic encoding:
    - Sorted keys
    - No whitespace
    - UTF-8 encoding
    - Separators without spaces

    Args:
        obj: JSON-serializable object

    Returns:
        Canonical JSON bytes

    Raises:
        InvalidInputException: If object is not JSON-serializable
    """
    try:
        json_str = json.dumps(
            obj,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        )
        return json_str.encode('utf-8')
    except (TypeError, ValueError) as e:
        raise InvalidInputException(f"Object is not JSON-serializable: {e}")


def _canonical_decode(data: bytes) -> Any:
    """
    Decode canonical JSON bytes to an object.

    Args:
        data: JSON bytes

    Returns:
        Decoded object

    Raises:
        RunRecordException: If decoding fails
    """
    try:
        json_str = data.decode('utf-8')
        return json.loads(json_str)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RunRecordException(f"Failed to decode JSON: {e}")


# ============================================================================
# TASK_SPEC
# ============================================================================

def put_task_spec(spec: dict) -> str:
    """
    Store an immutable task specification in CAS.

    The task spec is canonically encoded to ensure deterministic hashing.
    Same input will always produce the same CAS hash.

    Args:
        spec: Task specification dictionary (must be JSON-serializable)

    Returns:
        CAS hash (lowercase hex, 64 characters)

    Raises:
        InvalidInputException: If spec is not a dict or not JSON-serializable
        RunRecordException: If storage fails
    """
    if not isinstance(spec, dict):
        raise InvalidInputException(f"Task spec must be a dict, got {type(spec).__name__}")

    if not spec:
        raise InvalidInputException("Task spec cannot be empty")

    # Canonically encode
    data = _canonical_encode(spec)

    # Store in CAS
    try:
        return cas_put(data)
    except Exception as e:
        raise RunRecordException(f"Failed to store task spec: {e}")


def load_task_spec(hash: str) -> dict:
    """
    Load an immutable task specification from CAS.

    Args:
        hash: CAS hash (lowercase hex, 64 characters)

    Returns:
        Task specification dictionary

    Raises:
        InvalidInputException: If hash format is invalid
        RunRecordException: If object not found or corrupted
    """
    if not isinstance(hash, str):
        raise InvalidInputException(f"Hash must be a string, got {type(hash).__name__}")

    # Load from CAS
    try:
        data = cas_get(hash)
    except InvalidHashException as e:
        raise InvalidInputException(f"Invalid hash format: {e}")
    except ObjectNotFoundException as e:
        raise RunRecordException(f"Task spec not found: {hash}")
    except CorruptObjectException as e:
        raise RunRecordException(f"Task spec corrupted: {hash}: {e}")
    except Exception as e:
        raise RunRecordException(f"Failed to load task spec: {e}")

    # Decode
    obj = _canonical_decode(data)

    # Validate type
    if not isinstance(obj, dict):
        raise RunRecordException(f"Task spec must be a dict, got {type(obj).__name__}")

    return obj


# ============================================================================
# STATUS
# ============================================================================

def put_status(status: dict) -> str:
    """
    Store an immutable status record in CAS.

    Status record should contain:
    - state: e.g. PENDING, RUNNING, SUCCESS, FAILURE
    - Optional error code or message

    The status is canonically encoded to ensure deterministic hashing.

    Args:
        status: Status dictionary (must be JSON-serializable)

    Returns:
        CAS hash (lowercase hex, 64 characters)

    Raises:
        InvalidInputException: If status is not a dict or not JSON-serializable
        RunRecordException: If storage fails
    """
    if not isinstance(status, dict):
        raise InvalidInputException(f"Status must be a dict, got {type(status).__name__}")

    if not status:
        raise InvalidInputException("Status cannot be empty")

    # Validate required field
    if 'state' not in status:
        raise InvalidInputException("Status must contain 'state' field")

    # Canonically encode
    data = _canonical_encode(status)

    # Store in CAS
    try:
        return cas_put(data)
    except Exception as e:
        raise RunRecordException(f"Failed to store status: {e}")


def load_status(hash: str) -> dict:
    """
    Load an immutable status record from CAS.

    Args:
        hash: CAS hash (lowercase hex, 64 characters)

    Returns:
        Status dictionary

    Raises:
        InvalidInputException: If hash format is invalid
        RunRecordException: If object not found or corrupted
    """
    if not isinstance(hash, str):
        raise InvalidInputException(f"Hash must be a string, got {type(hash).__name__}")

    # Load from CAS
    try:
        data = cas_get(hash)
    except InvalidHashException as e:
        raise InvalidInputException(f"Invalid hash format: {e}")
    except ObjectNotFoundException as e:
        raise RunRecordException(f"Status not found: {hash}")
    except CorruptObjectException as e:
        raise RunRecordException(f"Status corrupted: {hash}: {e}")
    except Exception as e:
        raise RunRecordException(f"Failed to load status: {e}")

    # Decode
    obj = _canonical_decode(data)

    # Validate type
    if not isinstance(obj, dict):
        raise RunRecordException(f"Status must be a dict, got {type(obj).__name__}")

    # Validate required field
    if 'state' not in obj:
        raise RunRecordException("Status must contain 'state' field")

    return obj


# ============================================================================
# OUTPUT_HASHES
# ============================================================================

def put_output_hashes(hashes: list[str]) -> str:
    """
    Store a deterministic ordered list of output CAS hashes.

    The order of hashes is preserved and must be stable.
    The list is canonically encoded to ensure deterministic hashing.

    Args:
        hashes: Ordered list of CAS hashes (lowercase hex, 64 characters each)

    Returns:
        CAS hash (lowercase hex, 64 characters)

    Raises:
        InvalidInputException: If hashes is not a list or contains invalid hashes
        RunRecordException: If storage fails
    """
    if not isinstance(hashes, list):
        raise InvalidInputException(f"Hashes must be a list, got {type(hashes).__name__}")

    # Validate each hash format
    for i, h in enumerate(hashes):
        if not isinstance(h, str):
            raise InvalidInputException(f"Hash at index {i} must be a string, got {type(h).__name__}")

        if len(h) != 64 or not all(c in '0123456789abcdef' for c in h):
            raise InvalidInputException(f"Hash at index {i} has invalid format: {h}")

    # Canonically encode (list is JSON-serializable)
    data = _canonical_encode(hashes)

    # Store in CAS
    try:
        return cas_put(data)
    except Exception as e:
        raise RunRecordException(f"Failed to store output hashes: {e}")


def load_output_hashes(hash: str) -> list[str]:
    """
    Load a deterministic ordered list of output CAS hashes.

    Args:
        hash: CAS hash (lowercase hex, 64 characters)

    Returns:
        Ordered list of CAS hashes

    Raises:
        InvalidInputException: If hash format is invalid
        RunRecordException: If object not found or corrupted
    """
    if not isinstance(hash, str):
        raise InvalidInputException(f"Hash must be a string, got {type(hash).__name__}")

    # Load from CAS
    try:
        data = cas_get(hash)
    except InvalidHashException as e:
        raise InvalidInputException(f"Invalid hash format: {e}")
    except ObjectNotFoundException as e:
        raise RunRecordException(f"Output hashes not found: {hash}")
    except CorruptObjectException as e:
        raise RunRecordException(f"Output hashes corrupted: {hash}: {e}")
    except Exception as e:
        raise RunRecordException(f"Failed to load output hashes: {e}")

    # Decode
    obj = _canonical_decode(data)

    # Validate type
    if not isinstance(obj, list):
        raise RunRecordException(f"Output hashes must be a list, got {type(obj).__name__}")

    # Validate each hash format
    for i, h in enumerate(obj):
        if not isinstance(h, str):
            raise RunRecordException(f"Hash at index {i} must be a string, got {type(h).__name__}")

        if len(h) != 64 or not all(c in '0123456789abcdef' for c in h):
            raise RunRecordException(f"Hash at index {i} has invalid format: {h}")

    return obj
