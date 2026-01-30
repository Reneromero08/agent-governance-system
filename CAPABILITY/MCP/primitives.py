#!/usr/bin/env python3
"""
MCP Safe Primitives - Atomic file operations and task validation.

Extracted from server.py for modularity. Originally ported from CAT LAB server_CATDPT.py.

Provides:
- Platform-aware file locking (Windows msvcrt / Unix fcntl)
- Atomic JSONL write and rewrite operations
- Streaming JSONL reader with pagination
- Task state machine validation
- File hashing utilities
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Optional import for GuardedWriter
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    GuardedWriter = None


# =============================================================================
# FILE LOCKING (Platform-specific)
# =============================================================================

if sys.platform == 'win32':
    import msvcrt
    def lock_file(f, exclusive: bool = True):
        """Lock file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
    def unlock_file(f):
        """Unlock file on Windows."""
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
else:
    import fcntl
    def lock_file(f, exclusive: bool = True):
        """Lock file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    def unlock_file(f):
        """Unlock file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# =============================================================================
# CONSTANTS
# =============================================================================

# Configuration constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit for file reads
MAX_RESULTS_PER_PAGE = 100
DEFAULT_POLL_INTERVAL = 5
MAX_POLL_INTERVAL = 60
BACKOFF_MULTIPLIER = 1.5

# Task state machine transitions (valid state changes)
TASK_STATES = {
    "pending": ["acknowledged", "cancelled"],
    "acknowledged": ["processing", "cancelled"],
    "processing": ["completed", "failed", "timeout", "cancelled"],
    "completed": [],  # Terminal state
    "failed": [],     # Terminal state
    "timeout": [],    # Terminal state
    "cancelled": [],  # Terminal state
}

# SPECTRUM-02 Validator Constants
VALIDATOR_SEMVER = "1.0.0"
SUPPORTED_VALIDATOR_SEMVERS = {"1.0.0", "1.0.1", "1.1.0"}

# Cache for build ID (computed once per process)
_VALIDATOR_BUILD_ID_CACHE: Optional[str] = None


# =============================================================================
# VALIDATOR BUILD ID
# =============================================================================

def get_validator_build_id(project_root: Optional[Path] = None) -> str:
    """Get deterministic validator build fingerprint.

    Preferred: git commit SHA (short) if repo is a git checkout.
    Fallback: SHA-256 of this file's bytes.

    Returns a non-empty string. Result is cached for process lifetime.
    """
    global _VALIDATOR_BUILD_ID_CACHE

    if _VALIDATOR_BUILD_ID_CACHE is not None:
        return _VALIDATOR_BUILD_ID_CACHE

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    # Try git commit SHA first
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            _VALIDATOR_BUILD_ID_CACHE = f"git:{result.stdout.strip()}"
            return _VALIDATOR_BUILD_ID_CACHE
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: hash of this file
    this_file = Path(__file__)
    if this_file.exists():
        sha = hashlib.sha256()
        with open(this_file, "rb") as f:
            sha.update(f.read())
        _VALIDATOR_BUILD_ID_CACHE = f"file:{sha.hexdigest()[:12]}"
        return _VALIDATOR_BUILD_ID_CACHE

    # Ultimate fallback (should never happen)
    _VALIDATOR_BUILD_ID_CACHE = "unknown"
    return _VALIDATOR_BUILD_ID_CACHE


# =============================================================================
# ATOMIC JSONL OPERATIONS
# =============================================================================

def atomic_write_jsonl(file_path: Path, line: str, writer: Optional[Any] = None) -> bool:
    """Atomically append a single line to a JSONL file.

    Uses write-to-temp-then-append pattern with file locking to prevent:
    1. Partial writes from crashes
    2. Interleaved writes from concurrent processes

    Args:
        file_path: Path to JSONL file
        line: JSON line to append
        writer: Optional GuardedWriter for firewall enforcement

    Returns True on success, False on failure.
    """
    try:
        line_bytes = (line.rstrip('\n') + '\n').encode('utf-8')

        if writer and hasattr(writer, 'mkdir_tmp'):
            writer.mkdir_tmp(file_path.parent, parents=True, exist_ok=True)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first (atomic on most filesystems)
        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='jsonl_',
            dir=file_path.parent
        )
        try:
            os.write(fd, line_bytes)
            os.fsync(fd)  # Force flush to disk
        finally:
            os.close(fd)

        # Now append temp content to target with locking
        with open(file_path, 'a', encoding='utf-8') as f:
            lock_file(f, exclusive=True)
            try:
                with open(temp_path, 'r', encoding='utf-8') as tmp:
                    f.write(tmp.read())
                f.flush()
                os.fsync(f.fileno())
            finally:
                unlock_file(f)

        # Clean up temp file
        os.unlink(temp_path)
        return True

    except Exception as e:
        # Clean up temp file on error
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except Exception:
            pass
        return False


def atomic_rewrite_jsonl(
    file_path: Path,
    transform: Callable[[List[Dict]], List[Dict]],
    writer: Optional[Any] = None
) -> bool:
    """Atomically rewrite a JSONL file with a transformation function.

    Uses read-transform-write pattern:
    1. Read all lines (with lock on read)
    2. Apply transformation
    3. Write to temp file
    4. Atomic rename over original

    Args:
        file_path: Path to JSONL file
        transform: Function to transform list of entries
        writer: Optional GuardedWriter for firewall enforcement

    Returns True on success, False on failure.
    """
    try:
        if writer and hasattr(writer, 'mkdir_tmp'):
            writer.mkdir_tmp(file_path.parent, parents=True, exist_ok=True)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create/touch file if it doesn't exist
        if not file_path.exists():
            file_path.touch()

        # Read phase - use lock to read
        entries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lock_file(f, exclusive=False)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
            finally:
                unlock_file(f)

        # Apply transformation
        transformed = transform(entries)

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='jsonl_',
            dir=file_path.parent
        )
        try:
            for entry in transformed:
                os.write(fd, (json.dumps(entry) + '\n').encode('utf-8'))
            os.fsync(fd)
        finally:
            os.close(fd)

        # Atomic rename (works on Unix, best-effort on Windows)
        temp_path_obj = Path(temp_path)
        if sys.platform == 'win32':
            import shutil
            backup_path = file_path.with_suffix('.bak')
            try:
                if file_path.exists():
                    shutil.copy2(file_path, backup_path)
                shutil.move(str(temp_path_obj), str(file_path))
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    backup_path.unlink()
                raise
        else:
            os.replace(temp_path, file_path)

        return True

    except Exception as e:
        try:
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass
        return False


def read_jsonl_streaming(
    file_path: Path,
    filter_fn: Optional[Callable[[Dict], bool]] = None,
    limit: int = MAX_RESULTS_PER_PAGE,
    offset: int = 0
) -> Iterator[Dict]:
    """Stream JSONL file with optional filtering and pagination.

    Yields entries one at a time without loading entire file into memory.
    """
    if not file_path.exists():
        return

    count = 0
    skipped = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lock_file(f, exclusive=False)  # Shared lock for reading
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

                # Apply filter
                if filter_fn and not filter_fn(entry):
                    continue

                # Apply offset
                if skipped < offset:
                    skipped += 1
                    continue

                # Apply limit
                if count >= limit:
                    break

                yield entry
                count += 1
        finally:
            unlock_file(f)


# =============================================================================
# TASK VALIDATION
# =============================================================================

def validate_task_state_transition(current: str, target: str) -> bool:
    """Validate that a task state transition is allowed."""
    if current not in TASK_STATES:
        return False
    return target in TASK_STATES.get(current, [])


def validate_task_spec(task_spec: Dict) -> Dict:
    """Validate task_spec has required fields and valid structure.

    Returns: {"valid": bool, "errors": [...]}
    """
    errors = []

    # Required fields
    required_fields = ["task_id", "task_type"]
    for field in required_fields:
        if field not in task_spec:
            errors.append(f"Missing required field: {field}")

    # Validate task_type
    valid_task_types = ["file_operation", "code_adapt", "validate", "research"]
    if "task_type" in task_spec and task_spec["task_type"] not in valid_task_types:
        errors.append(f"Invalid task_type: {task_spec['task_type']}. Must be one of: {valid_task_types}")

    # Validate task_id format (alphanumeric with hyphens/underscores)
    if "task_id" in task_spec:
        task_id = task_spec["task_id"]
        if not isinstance(task_id, str) or not re.match(r'^[\w\-]+$', task_id):
            errors.append(f"Invalid task_id format: must be alphanumeric with hyphens/underscores")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =============================================================================
# HASHING AND SCHEMA VALIDATION
# =============================================================================

def compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_against_schema(instance: Dict, schema: Dict) -> Dict:
    """Validate JSON instance against schema.

    Basic validation without jsonschema library:
    - Checks required fields
    - Validates enum values for task_type
    - Returns errors list if invalid
    """
    errors = []

    # Check input_schema if present
    input_schema = schema.get("input_schema", schema)

    # Check required fields
    required_fields = input_schema.get("required", [])
    for field in required_fields:
        if field not in instance:
            errors.append(f"Missing required field: {field}")

    # Validate properties with enums
    properties = input_schema.get("properties", {})
    for field, field_schema in properties.items():
        if field in instance and "enum" in field_schema:
            if instance[field] not in field_schema["enum"]:
                errors.append(
                    f"Invalid value for '{field}': '{instance[field]}'. "
                    f"Must be one of: {field_schema['enum']}"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# Aliases for backwards compatibility (used by server.py with underscore prefix)
_lock_file = lock_file
_unlock_file = unlock_file
_atomic_write_jsonl = atomic_write_jsonl
_atomic_rewrite_jsonl = atomic_rewrite_jsonl
_read_jsonl_streaming = read_jsonl_streaming
_validate_task_state_transition = validate_task_state_transition
_validate_task_spec = validate_task_spec
_compute_hash = compute_hash
_validate_against_schema = validate_against_schema
