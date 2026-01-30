#!/usr/bin/env python3
"""
AGS MCP Server - Stub Implementation

This is the seam implementation of the AGS MCP server.
It defines the interface but returns "not implemented" for most features.
Full implementation will be added when MCP integration is needed.

Usage:
  python CAPABILITY/MCP/server.py          # Start server (stdio mode)
  python CAPABILITY/MCP/server.py --http   # Start server (HTTP mode, not implemented)
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Write enforcement
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

# =============================================================================
# SAFE PRIMITIVES (Ported from CAT LAB server_CATDPT.py)
# =============================================================================

# Windows-compatible file locking
if sys.platform == 'win32':
    import msvcrt
    def _lock_file(f, exclusive: bool = True):
        """Lock file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
    def _unlock_file(f):
        """Unlock file on Windows."""
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
else:
    import fcntl
    def _lock_file(f, exclusive: bool = True):
        """Lock file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    def _unlock_file(f):
        """Unlock file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_ROOT = PROJECT_ROOT / "CAPABILITY"
LAW_ROOT = PROJECT_ROOT / "LAW"
NAVIGATION_ROOT = PROJECT_ROOT / "NAVIGATION"

# Load schemas
SCHEMAS_DIR = Path(__file__).parent / "schemas"
# Per ADR: Logs/runs under LAW/CONTRACTS/_runs/
RUNS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
LOGS_DIR = LAW_ROOT / "CONTRACTS" / "_runs" / "mcp_logs"

# =============================================================================
# SPECTRUM-02 Validator Constants
# =============================================================================
VALIDATOR_SEMVER = "1.0.0"
SUPPORTED_VALIDATOR_SEMVERS = {"1.0.0", "1.0.1", "1.1.0"}

# Cache for build ID (computed once per process)
_VALIDATOR_BUILD_ID_CACHE: Optional[str] = None


def get_validator_build_id() -> str:
    """Get deterministic validator build fingerprint.

    Preferred: git commit SHA (short) if repo is a git checkout.
    Fallback: SHA-256 of MCP/server.py file bytes.

    Returns a non-empty string. Result is cached for process lifetime.
    """
    global _VALIDATOR_BUILD_ID_CACHE

    if _VALIDATOR_BUILD_ID_CACHE is not None:
        return _VALIDATOR_BUILD_ID_CACHE

    # Try git commit SHA first
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            _VALIDATOR_BUILD_ID_CACHE = f"git:{result.stdout.strip()}"
            return _VALIDATOR_BUILD_ID_CACHE
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: hash of server.py
    server_path = Path(__file__)
    if server_path.exists():
        sha = hashlib.sha256()
        with open(server_path, "rb") as f:
            sha.update(f.read())
        _VALIDATOR_BUILD_ID_CACHE = f"file:{sha.hexdigest()[:12]}"
        return _VALIDATOR_BUILD_ID_CACHE

    # Ultimate fallback (should never happen)
    _VALIDATOR_BUILD_ID_CACHE = "unknown"
    return _VALIDATOR_BUILD_ID_CACHE


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

# Configuration constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit for file reads
MAX_RESULTS_PER_PAGE = 100
DEFAULT_POLL_INTERVAL = 5
MAX_POLL_INTERVAL = 60
BACKOFF_MULTIPLIER = 1.5

# =============================================================================
# CMP-01 ROOT RULES - Strict path governance (6-bucket structure)
# =============================================================================

# Contracts directory (runs/ledgers)
CONTRACTS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"

# Skills directory
SKILLS_DIR = PROJECT_ROOT / "CAPABILITY" / "SKILLS"

# Durable output roots (only places files may persist after run)
DURABLE_ROOTS = [
    "LAW/CONTRACTS/_runs/",
    "NAVIGATION/CORTEX/_generated/",
    "MEMORY/LLM_PACKER/_packs/",
]

# Catalytic domains (temporary, must be restored byte-identical)
CATALYTIC_ROOTS = [
    "LAW/CONTRACTS/_runs/_tmp/",
    "NAVIGATION/CORTEX/_generated/_tmp/",
    "MEMORY/LLM_PACKER/_packs/_tmp/",
    "CAPABILITY/TOOLS/_tmp/",
    "CAPABILITY/MCP/_tmp/",
]

# Forbidden roots (must never be written to or overlapped)
FORBIDDEN_ROOTS = [
    "LAW/CANON/",
    "AGENTS.md",
]


def _atomic_write_jsonl(file_path: Path, line: str, writer: Optional[GuardedWriter] = None) -> bool:
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

        if writer:
            # Use firewall-protected mkdir
            writer.mkdir_tmp(file_path.parent, parents=True, exist_ok=True)
        else:
            # Legacy path
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
            _lock_file(f, exclusive=True)
            try:
                with open(temp_path, 'r', encoding='utf-8') as tmp:
                    f.write(tmp.read())
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock_file(f)

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


def _atomic_rewrite_jsonl(
    file_path: Path,
    transform: Callable[[List[Dict]], List[Dict]],
    writer: Optional[GuardedWriter] = None
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
        if writer:
            writer.mkdir_tmp(file_path.parent, parents=True, exist_ok=True)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create/touch file if it doesn't exist
        if not file_path.exists():
            file_path.touch()

        # Read phase - use lock to read
        entries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            _lock_file(f, exclusive=False)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
            finally:
                _unlock_file(f)

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
            # Windows needs file to be closed before rename
            import shutil
            backup_path = file_path.with_suffix('.bak')
            try:
                # Create backup
                if file_path.exists():
                    shutil.copy2(file_path, backup_path)
                # Replace original with temp
                shutil.move(str(temp_path_obj), str(file_path))
                # Remove backup
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                # Restore from backup on failure
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    backup_path.unlink()
                raise
        else:
            os.replace(temp_path, file_path)

        return True

    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass
        return False


def _read_jsonl_streaming(
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
        _lock_file(f, exclusive=False)  # Shared lock for reading
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
            _unlock_file(f)


def _validate_task_state_transition(current: str, target: str) -> bool:
    """Validate that a task state transition is allowed."""
    if current not in TASK_STATES:
        return False
    return target in TASK_STATES.get(current, [])


def _validate_task_spec(task_spec: Dict) -> Dict:
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


def _compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _validate_against_schema(instance: Dict, schema: Dict) -> Dict:
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


# =============================================================================
# CMP-01 PATH VALIDATION (Ported from CAT LAB server_CATDPT.py)
# Forbidden overlap containment + output existence checks
# =============================================================================

def _is_path_under_root(path: Path, root: Path) -> bool:
    """Component-safe check if path is under root (not just string prefix)."""
    try:
        # Python 3.9+ has is_relative_to
        return path.is_relative_to(root)
    except AttributeError:
        # Python 3.8 fallback
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def _validate_single_path(
    raw_path: str,
    json_pointer: str,
    allowed_roots: List[str],
    root_error_code: str
) -> List[Dict]:
    """Validate a single path against CMP-01 rules.
    
    Returns list of error dicts (empty if valid).
    """
    errors = []

    # 1. Reject absolute paths
    if Path(raw_path).is_absolute():
        errors.append({
            "code": "PATH_ESCAPES_REPO_ROOT",
            "message": f"Absolute paths are not allowed: {raw_path}",
            "path": json_pointer,
            "details": {"declared": raw_path, "reason": "absolute_path"}
        })
        return errors

    # 2. Reject traversal segments
    path_parts = Path(raw_path).parts
    if ".." in path_parts:
        errors.append({
            "code": "PATH_CONTAINS_TRAVERSAL",
            "message": f"Path contains forbidden traversal segment '..': {raw_path}",
            "path": json_pointer,
            "details": {"declared": raw_path, "segments": list(path_parts)}
        })
        return errors

    # 3. Resolve and check containment under PROJECT_ROOT
    abs_path = (PROJECT_ROOT / raw_path).resolve()
    if not _is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
        errors.append({
            "code": "PATH_ESCAPES_REPO_ROOT",
            "message": f"Path escapes repository root: {raw_path}",
            "path": json_pointer,
            "details": {"declared": raw_path, "resolved": str(abs_path), "repo_root": str(PROJECT_ROOT)}
        })
        return errors

    # 4. Check forbidden overlap
    for forbidden in FORBIDDEN_ROOTS:
        forbidden_abs = (PROJECT_ROOT / forbidden).resolve()
        # Check both directions of containment
        if _is_path_under_root(abs_path, forbidden_abs) or _is_path_under_root(forbidden_abs, abs_path):
            errors.append({
                "code": "FORBIDDEN_PATH_OVERLAP",
                "message": f"Path overlaps forbidden root '{forbidden}': {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "forbidden_root": forbidden}
            })
            return errors

    # 5. Check under allowed roots
    under_allowed = False
    for root in allowed_roots:
        root_abs = (PROJECT_ROOT / root).resolve()
        if _is_path_under_root(abs_path, root_abs):
            under_allowed = True
            break

    if not under_allowed:
        errors.append({
            "code": root_error_code,
            "message": f"Path not under any allowed root: {raw_path}",
            "path": json_pointer,
            "details": {"declared": raw_path, "allowed_roots": allowed_roots}
        })

    return errors


def _check_containment_overlap(
    paths: List[str],
    json_pointer_base: str
) -> List[Dict]:
    """Check for containment overlap between paths in the same list.
    
    Policy: Exact duplicates (same resolved path) are allowed/deduped.
    Only flag when one path strictly contains another.
    """
    errors = []
    abs_paths = []

    # Store (original_index, raw_path, abs_path) to preserve correct indices
    for orig_idx, raw_path in enumerate(paths):
        if not Path(raw_path).is_absolute() and ".." not in Path(raw_path).parts:
            abs_paths.append((orig_idx, raw_path, (PROJECT_ROOT / raw_path).resolve()))

    for i, (idx_a, raw_a, abs_a) in enumerate(abs_paths):
        for j, (idx_b, raw_b, abs_b) in enumerate(abs_paths):
            if i >= j:
                continue
            
            # Allow exact duplicates (same resolved path)
            if abs_a == abs_b:
                continue
            
            # Check if one strictly contains the other (both directions)
            if _is_path_under_root(abs_a, abs_b) or _is_path_under_root(abs_b, abs_a):
                # Use smaller original index for deterministic path pointer
                smaller_idx = min(idx_a, idx_b)
                errors.append({
                    "code": "PATH_OVERLAP",
                    "message": f"Paths have containment overlap: '{raw_a}' and '{raw_b}'",
                    "path": f"{json_pointer_base}/{smaller_idx}",
                    "details": {
                        "index_a": idx_a,
                        "index_b": idx_b,
                        "path_a": raw_a,
                        "path_b": raw_b
                    }
                })

    return errors


def _validate_jobspec_paths(task_spec: Dict) -> Dict:
    """Validate all paths in a JobSpec against CMP-01 rules.
    
    Checks:
    - catalytic_domains must be under CATALYTIC_ROOTS
    - outputs.durable_paths must be under DURABLE_ROOTS
    - No forbidden overlaps
    - No traversal escapes
    - No containment overlap within same list
    
    Returns: {"valid": bool, "errors": [error_dict, ...]}
    """
    errors = []

    # Validate catalytic_domains
    catalytic_domains = task_spec.get("catalytic_domains", [])
    for idx, domain in enumerate(catalytic_domains):
        path_errors = _validate_single_path(
            domain,
            f"/catalytic_domains/{idx}",
            CATALYTIC_ROOTS,
            "CATALYTIC_OUTSIDE_ROOT"
        )
        errors.extend(path_errors)

    # Check containment overlap within catalytic_domains
    if len(catalytic_domains) > 1:
        errors.extend(_check_containment_overlap(
            catalytic_domains,
            "/catalytic_domains"
        ))

    # Validate outputs.durable_paths
    outputs = task_spec.get("outputs", {})
    durable_paths = outputs.get("durable_paths", [])
    for idx, dpath in enumerate(durable_paths):
        path_errors = _validate_single_path(
            dpath,
            f"/outputs/durable_paths/{idx}",
            DURABLE_ROOTS,
            "OUTPUT_OUTSIDE_DURABLE_ROOT"
        )
        errors.extend(path_errors)

    # Check containment overlap within durable_paths
    if len(durable_paths) > 1:
        errors.extend(_check_containment_overlap(
            durable_paths,
            "/outputs/durable_paths"
        ))

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _verify_post_run_outputs(run_id: str) -> Dict:
    """Verify declared outputs exist after run completion.
    
    Called from skill_complete to enforce output existence.
    
    Returns: {"valid": bool, "errors": [error_dict, ...]}
    """
    errors = []
    run_dir = CONTRACTS_DIR / run_id

    # Load TASK_SPEC.json
    task_spec_path = run_dir / "TASK_SPEC.json"
    if not task_spec_path.exists():
        return {
            "valid": False,
            "errors": [{
                "code": "TASK_SPEC_MISSING",
                "message": f"TASK_SPEC.json not found in run directory",
                "path": "/",
                "details": {"run_id": run_id, "expected": str(task_spec_path)}
            }]
        }

    with open(task_spec_path) as f:
        task_spec = json.load(f)

    outputs = task_spec.get("outputs", {})
    durable_paths = outputs.get("durable_paths", [])

    for idx, raw_path in enumerate(durable_paths):
        json_pointer = f"/outputs/durable_paths/{idx}"

        # 0. Report errors for absolute/traversal paths (do not silently skip)
        if Path(raw_path).is_absolute():
            errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Absolute paths are not allowed: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "reason": "absolute_path"}
            })
            continue  # Skip further checks for this entry

        if ".." in Path(raw_path).parts:
            errors.append({
                "code": "PATH_CONTAINS_TRAVERSAL",
                "message": f"Path contains forbidden traversal segment '..': {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "segments": list(Path(raw_path).parts)}
            })
            continue  # Skip further checks for this entry

        abs_path = (PROJECT_ROOT / raw_path).resolve()

        # 0.5 Check containment under PROJECT_ROOT (catches symlink escapes)
        if not _is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
            errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Path escapes repository root (possibly via symlink): {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "resolved": str(abs_path), "repo_root": str(PROJECT_ROOT)}
            })
            continue  # Skip further checks for this entry

        # 1. Check forbidden overlap (hard stop for this entry if found)
        forbidden_hit = False
        for forbidden in FORBIDDEN_ROOTS:
            forbidden_abs = (PROJECT_ROOT / forbidden).resolve()
            if _is_path_under_root(abs_path, forbidden_abs) or _is_path_under_root(forbidden_abs, abs_path):
                errors.append({
                    "code": "FORBIDDEN_PATH_OVERLAP",
                    "message": f"Output overlaps forbidden root '{forbidden}': {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "forbidden_root": forbidden}
                })
                forbidden_hit = True
                break  # Exit forbidden loop
        
        if forbidden_hit:
            continue  # Skip durable/existence checks for this entry

        # 2. Check under DURABLE_ROOTS
        under_durable = False
        for root in DURABLE_ROOTS:
            root_abs = (PROJECT_ROOT / root).resolve()
            if _is_path_under_root(abs_path, root_abs):
                under_durable = True
                break

        if not under_durable:
            errors.append({
                "code": "OUTPUT_OUTSIDE_DURABLE_ROOT",
                "message": f"Output not under any durable root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "durable_roots": DURABLE_ROOTS}
            })
            continue  # Skip existence check for this entry

        # 3. Check existence on disk
        if not abs_path.exists():
            errors.append({
                "code": "OUTPUT_MISSING",
                "message": f"Declared output does not exist: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "resolved": str(abs_path)}
            })

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =============================================================================
# SPECTRUM-02 BUNDLE VERIFICATION (Ported from CAT LAB server_CATDPT.py)
# Adversarial resume without execution history
# =============================================================================

def _generate_output_hashes(run_id: str) -> Dict:
    """Generate OUTPUT_HASHES.json for SPECTRUM-02 bundle.

    Hashes every declared durable output in TASK_SPEC.json.
    - If output is a file: hash that file
    - If output is a directory: hash every file under it

    Returns: {"valid": bool, "errors": [...], "hashes": {...}}
    """
    errors = []
    hashes = {}
    run_dir = CONTRACTS_DIR / run_id

    # Load TASK_SPEC.json
    task_spec_path = run_dir / "TASK_SPEC.json"
    if not task_spec_path.exists():
        return {
            "valid": False,
            "errors": [{
                "code": "TASK_SPEC_MISSING",
                "message": "TASK_SPEC.json not found",
                "path": "/",
                "details": {"run_id": run_id}
            }],
            "hashes": {}
        }

    with open(task_spec_path) as f:
        task_spec = json.load(f)

    outputs_spec = task_spec.get("outputs", {})
    durable_paths = outputs_spec.get("durable_paths", [])

    for idx, raw_path in enumerate(durable_paths):
        json_pointer = f"/outputs/durable_paths/{idx}"

        # Skip invalid paths (already caught by _verify_post_run_outputs)
        if Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
            continue

        abs_path = (PROJECT_ROOT / raw_path).resolve()

        # Skip if path escapes repo root
        if not _is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
            continue

        if not abs_path.exists():
            errors.append({
                "code": "OUTPUT_MISSING",
                "message": f"Declared output does not exist: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path}
            })
            continue

        if abs_path.is_file():
            # Hash single file
            file_hash = _compute_hash(abs_path)
            # Use posix-style path relative to PROJECT_ROOT
            rel_posix = abs_path.relative_to(PROJECT_ROOT).as_posix()
            hashes[rel_posix] = f"sha256:{file_hash}"
        elif abs_path.is_dir():
            # Hash every file under directory
            for file_path in abs_path.rglob("*"):
                if file_path.is_file():
                    file_hash = _compute_hash(file_path)
                    rel_posix = file_path.relative_to(PROJECT_ROOT).as_posix()
                    hashes[rel_posix] = f"sha256:{file_hash}"

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "hashes": hashes
    }


def verify_spectrum02_bundle(
    run_dir: Path,
    strict_build_id: bool = False
) -> Dict:
    """Verify a SPECTRUM-02 resume bundle.

    Checks:
    - TASK_SPEC.json exists
    - STATUS.json exists with status=success and cmp01=pass
    - OUTPUT_HASHES.json exists with supported validator_semver
    - validator_build_id exists and is non-empty
    - All declared hashes verify against actual files

    Args:
        run_dir: Path to the run directory containing the bundle
        strict_build_id: If True, reject if build_id != current get_validator_build_id()

    Returns: {"valid": bool, "errors": [...]}
    """
    errors = []

    # Ensure run_dir is a Path
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    # 1. Check TASK_SPEC.json exists
    task_spec_path = run_dir / "TASK_SPEC.json"
    if not task_spec_path.exists():
        errors.append({
            "code": "BUNDLE_INCOMPLETE",
            "message": "TASK_SPEC.json missing",
            "path": "/",
            "details": {"expected": str(task_spec_path)}
        })
        return {"valid": False, "errors": errors}

    # 2. Check STATUS.json exists and is valid
    status_path = run_dir / "STATUS.json"
    if not status_path.exists():
        errors.append({
            "code": "BUNDLE_INCOMPLETE",
            "message": "STATUS.json missing",
            "path": "/",
            "details": {"expected": str(status_path)}
        })
        return {"valid": False, "errors": errors}

    try:
        with open(status_path) as f:
            status = json.load(f)
    except json.JSONDecodeError as e:
        errors.append({
            "code": "BUNDLE_INCOMPLETE",
            "message": f"STATUS.json invalid JSON: {e}",
            "path": "/",
            "details": {}
        })
        return {"valid": False, "errors": errors}

    if status.get("status") != "success":
        errors.append({
            "code": "STATUS_NOT_SUCCESS",
            "message": f"STATUS.status is '{status.get('status')}', expected 'success'",
            "path": "/status",
            "details": {"actual": status.get("status")}
        })
        return {"valid": False, "errors": errors}

    if status.get("cmp01") != "pass":
        errors.append({
            "code": "CMP01_NOT_PASS",
            "message": f"STATUS.cmp01 is '{status.get('cmp01')}', expected 'pass'",
            "path": "/cmp01",
            "details": {"actual": status.get("cmp01")}
        })
        return {"valid": False, "errors": errors}

    # 3. Check OUTPUT_HASHES.json exists and is valid
    hashes_path = run_dir / "OUTPUT_HASHES.json"
    if not hashes_path.exists():
        errors.append({
            "code": "BUNDLE_INCOMPLETE",
            "message": "OUTPUT_HASHES.json missing",
            "path": "/",
            "details": {"expected": str(hashes_path)}
        })
        return {"valid": False, "errors": errors}

    try:
        with open(hashes_path) as f:
            output_hashes = json.load(f)
    except json.JSONDecodeError as e:
        errors.append({
            "code": "BUNDLE_INCOMPLETE",
            "message": f"OUTPUT_HASHES.json invalid JSON: {e}",
            "path": "/",
            "details": {}
        })
        return {"valid": False, "errors": errors}

    # 4. Check validator semver is supported
    validator_semver = output_hashes.get("validator_semver")
    if validator_semver not in SUPPORTED_VALIDATOR_SEMVERS:
        errors.append({
            "code": "VALIDATOR_UNSUPPORTED",
            "message": f"validator_semver '{validator_semver}' not supported",
            "path": "/validator_semver",
            "details": {
                "actual": validator_semver,
                "supported": list(SUPPORTED_VALIDATOR_SEMVERS)
            }
        })
        return {"valid": False, "errors": errors}

    # 5. Check validator_build_id exists and is non-empty
    validator_build_id = output_hashes.get("validator_build_id")
    if not validator_build_id:
        errors.append({
            "code": "VALIDATOR_BUILD_ID_MISSING",
            "message": "validator_build_id is missing or empty",
            "path": "/validator_build_id",
            "details": {"actual": validator_build_id}
        })
        return {"valid": False, "errors": errors}

    # 6. Strict build ID check (optional)
    if strict_build_id:
        current_build_id = get_validator_build_id()
        if validator_build_id != current_build_id:
            errors.append({
                "code": "VALIDATOR_BUILD_MISMATCH",
                "message": f"validator_build_id mismatch: expected '{current_build_id}', got '{validator_build_id}'",
                "path": "/validator_build_id",
                "details": {
                    "expected": current_build_id,
                    "actual": validator_build_id
                }
            })
            return {"valid": False, "errors": errors}

    # 7. Verify each hash
    hashes = output_hashes.get("hashes", {})
    for rel_path, expected_hash in hashes.items():
        abs_path = PROJECT_ROOT / rel_path

        if not abs_path.exists():
            errors.append({
                "code": "OUTPUT_MISSING",
                "message": f"Output file does not exist: {rel_path}",
                "path": f"/hashes/{rel_path}",
                "details": {"declared": rel_path, "resolved": str(abs_path)}
            })
            continue

        # Compute actual hash
        actual_hash = f"sha256:{_compute_hash(abs_path)}"

        if actual_hash != expected_hash:
            errors.append({
                "code": "HASH_MISMATCH",
                "message": f"Hash mismatch for {rel_path}",
                "path": f"/hashes/{rel_path}",
                "details": {
                    "declared": rel_path,
                    "expected": expected_hash,
                    "actual": actual_hash
                }
            })

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def load_schema(name: str) -> Dict:
    """Load a schema file."""
    schema_path = SCHEMAS_DIR / f"{name}.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    return {}


# MCP Protocol Constants
MCP_VERSION = "2024-11-05"
SERVER_NAME = "ags-mcp-server"
SERVER_VERSION = "0.1.0"


def governed_tool(func):
    """Decorator: Run preflight + admission + critic.py before execution to enforce governance lock."""
    def wrapper(self, args: Dict) -> Dict:
        import os
        import subprocess
        # Mandatory preflight (fail-closed)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        preflight = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "preflight"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        if preflight.returncode != 0:
            payload = ((preflight.stdout or "") + "\n" + (preflight.stderr or "")).strip()
            return {
                "content": [{
                    "type": "text",
                    "text": "⛔ PREFLIGHT BLOCKED ⛔\n\nAction blocked. Repository freshness check failed.\n\n" + (payload or "(no details)")
                }],
                "isError": True
            }

        # Mandatory admission control (fail-closed)
        intent_path = env.get("AGS_INTENT_PATH", "").strip()
        if not intent_path:
            return {
                "content": [{
                    "type": "text",
                    "text": "⛔ ADMISSION BLOCKED ⛔\n\nAction blocked. Missing AGS_INTENT_PATH for admission control."
                }],
                "isError": True
            }
        admit = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "ags.py"), "admit", "--intent", intent_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        if admit.returncode != 0:
            payload = ((admit.stdout or "") + "\n" + (admit.stderr or "")).strip()
            return {
                "content": [{
                    "type": "text",
                    "text": "⛔ ADMISSION BLOCKED ⛔\n\nAction blocked. Admission control rejected intent.\n\n" + (payload or "(no details)")
                }],
                "isError": True
            }
        # Exempt if checking critic itself (avoid infinite loop if critic is broken? No, critic run is separate tool)
        
        # Run critic
        
        res = subprocess.run(
            [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "governance" / "critic.py")],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", cwd=str(PROJECT_ROOT), env=env
        )
        
        if res.returncode != 0:
            output = (res.stdout + res.stderr).strip()
            return {
                "content": [{
                    "type": "text",
                    "text": f"⛔ GOVERNANCE LOCKDOWN ⛔\n\nAction blocked. The repository has governance violations.\nYou must fix these issues before the Agent can Act.\n\nViolations:\n{output}"
                }],
                "isError": True
            }
        return func(self, args)
    return wrapper


class AGSMCPServer:
    """AGS MCP Server implementation."""

    def __init__(self):
        import uuid
        import atexit
        self.tools_schema = load_schema("tools")
        self.resources_schema = load_schema("resources")
        self._initialized = False
        self.session_id = str(uuid.uuid4())
        # Semantic adapter is lazy-initialized on first semantic tool call.
        self.semantic_adapter = None
        self.semantic_available = False
        self._semantic_init_attempted = False
        # Session auditor for ELO file/symbol tracking (E.1.2)
        self.session_auditor = None
        self._session_auditor_available = False
        self._init_session_auditor()
        # Register atexit handler to end session cleanly
        atexit.register(self._end_session_audit)
        # Write enforcement
        self.writer = GuardedWriter(
            project_root=PROJECT_ROOT,
            tmp_roots=[
                "LAW/CONTRACTS/_runs/_tmp",
                "CAPABILITY/PRIMITIVES/_scratch",
                "NAVIGATION/CORTEX/_generated/_tmp",
            ],
            durable_roots=[
                "LAW/CONTRACTS/_runs",
                "NAVIGATION/CORTEX/_generated",
            ],
            exclusions=[
                "LAW/CANON",
                "AGENTS.md",
                "BUILD",
                ".git",
            ],
        )






    def _init_session_auditor(self) -> None:
        """Initialize the session auditor for ELO file/symbol tracking (E.1.2).

        The SessionAuditor tracks:
        - Files accessed via MCP tools (canon_read, context_search, etc.)
        - ADR reads
        - Symbol expansions (codebook_lookup)
        - Search query counts (semantic vs keyword)
        """
        try:
            # Add CAPABILITY to path for import
            sys.path.insert(0, str(CAPABILITY_ROOT))
            from PRIMITIVES.session_auditor import SessionAuditor

            # Initialize auditor with log dir and agent ID
            log_dir = NAVIGATION_ROOT / "CORTEX" / "_generated"
            log_dir.mkdir(parents=True, exist_ok=True)

            self.session_auditor = SessionAuditor(
                log_dir=str(log_dir),
                agent_id="mcp-server"
            )
            # Start the session
            self.session_auditor.start_session()
            self._session_auditor_available = True
            print(f"[INFO] Session auditor initialized (session: {self.session_auditor.current_session_id})", file=sys.stderr)
        except Exception as e:
            self.session_auditor = None
            self._session_auditor_available = False
            print(f"[WARNING] Session auditor unavailable: {e}", file=sys.stderr)

    def _end_session_audit(self) -> None:
        """End the session audit and write to log file."""
        if self.session_auditor and self.session_auditor.is_active:
            try:
                entry = self.session_auditor.end_session()
                print(f"[INFO] Session audit complete: {entry.search_queries} searches, {len(entry.files_accessed)} files", file=sys.stderr)
            except Exception as e:
                print(f"[WARNING] Failed to end session audit: {e}", file=sys.stderr)

    def _audit_file_access(self, file_path: str) -> None:
        """Record a file access for ELO tracking."""
        if self.session_auditor and self._session_auditor_available:
            try:
                # Normalize to relative path from project root
                path = Path(file_path)
                if path.is_absolute():
                    try:
                        rel_path = path.relative_to(PROJECT_ROOT)
                        file_path = str(rel_path)
                    except ValueError:
                        pass  # Keep absolute path if not under project root
                self.session_auditor.record_file_access(file_path)
            except Exception:
                pass  # Don't let audit failures break tool calls

    def _audit_adr_read(self, adr_id: str) -> None:
        """Record an ADR read for ELO tracking."""
        if self.session_auditor and self._session_auditor_available:
            try:
                self.session_auditor.record_adr_read(adr_id)
            except Exception:
                pass

    def _audit_symbol_expansion(self, symbol: str) -> None:
        """Record a symbol expansion for ELO tracking."""
        if self.session_auditor and self._session_auditor_available:
            try:
                self.session_auditor.record_symbol_expansion(symbol)
            except Exception:
                pass

    def _audit_search(self, is_semantic: bool) -> None:
        """Record a search for ELO tracking."""
        if self.session_auditor and self._session_auditor_available:
            try:
                self.session_auditor.record_search(is_semantic=is_semantic)
            except Exception:
                pass

    def _track_tool_access(self, tool_name: str, arguments: Dict, result: Dict) -> None:
        """Track file/symbol/search access based on tool call (E.1.2).

        Maps tool calls to ELO-relevant events:
        - canon_read -> file access + potential ADR read
        - context_search/context_review -> file access (from results)
        - codebook_lookup -> symbol expansion
        - cassette_network_query/memory_query/skill_discovery -> semantic search
        - find_related -> semantic search
        """
        try:
            # Canon read - track file access
            if tool_name == "canon_read":
                file_name = arguments.get("file", "")
                if file_name:
                    file_path = f"LAW/CANON/{file_name.upper()}.md"
                    self._audit_file_access(file_path)

            # Context search/review - track file accesses from results
            elif tool_name in ("context_search", "context_review"):
                self._audit_search(is_semantic=False)  # Keyword search
                # Extract file paths from result
                try:
                    content = result.get("content", [])
                    if content and content[0].get("type") == "text":
                        import json
                        records = json.loads(content[0].get("text", "[]"))
                        for record in records:
                            if isinstance(record, dict) and "path" in record:
                                self._audit_file_access(record["path"])
                                # Check if it's an ADR
                                path = record.get("path", "")
                                if "ADR-" in path:
                                    import re
                                    adr_match = re.search(r"ADR-(\d+)", path)
                                    if adr_match:
                                        self._audit_adr_read(f"ADR-{adr_match.group(1)}")
                except Exception:
                    pass

            # Codebook lookup - track symbol expansion
            elif tool_name == "codebook_lookup":
                symbol_id = arguments.get("id", "")
                if symbol_id:
                    self._audit_symbol_expansion(symbol_id)
                # Also track as keyword search if expand=True
                if arguments.get("expand"):
                    self._audit_search(is_semantic=False)

            # Semantic search tools
            elif tool_name in ("cassette_network_query", "memory_query", "skill_discovery",
                               "find_related", "semantic_neighbors"):
                self._audit_search(is_semantic=True)
                # Extract file paths from cassette_network_query results
                if tool_name == "cassette_network_query":
                    try:
                        content = result.get("content", [])
                        if content and content[0].get("type") == "text":
                            import json
                            data = json.loads(content[0].get("text", "{}"))
                            for res in data.get("results", []):
                                path = res.get("path", res.get("file_path", ""))
                                if path:
                                    self._audit_file_access(path)
                    except Exception:
                        pass

            # ADR creation - track ADR read (reviewing existing ADRs)
            elif tool_name == "adr_create":
                # The ADR being created
                adr_id = arguments.get("id", "")
                if adr_id:
                    self._audit_adr_read(adr_id)

        except Exception:
            pass  # Don't let tracking failures break tool calls

    def _ensure_semantic_adapter(self) -> None:
        """Lazy initialization of the semantic adapter.

        VS Code/Antigravity expect the server to respond quickly to `initialize`.
        Any heavy/optional init must be deferred until a semantic tool is actually called.

        ELO Integration: Passes session_id to adapter for search logging.
        """
        if getattr(self, "_semantic_init_attempted", False):
            return

        self._semantic_init_attempted = True
        try:
            try:
                from .semantic_adapter import SemanticMCPAdapter
            except Exception:
                from semantic_adapter import SemanticMCPAdapter  # type: ignore

            # Pass session_id for ELO search logging
            adapter = SemanticMCPAdapter(session_id=self.session_id)
            init_result = adapter.initialize()
            self.semantic_adapter = adapter
            self.semantic_available = True
            elo_status = "enabled" if init_result.get("elo_available") else "disabled"
            print(f"[INFO] Semantic adapter initialized (ELO: {elo_status})", file=sys.stderr)
        except Exception as e:
            self.semantic_adapter = None
            self.semantic_available = False
            print(f"[INFO] Semantic adapter unavailable: {e}", file=sys.stderr)
    def handle_request(self, request: Dict) -> Optional[Dict]:
        """Handle a JSON-RPC 2.0 request.

        Important: notifications (requests without an `id`, or with `id: null`) must not
        receive a response. Some MCP clients validate strictly and will error if we
        reply with `id: null`.
        """
        method = request.get("method", "") or ""
        params = request.get("params", {}) or {}

        has_id = ("id" in request) and (request.get("id") is not None)
        request_id = request.get("id") if has_id else None

        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,  # legacy alias
            "notifications/initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }

        handler = handlers.get(method)
        if not handler:
            return None if not has_id else self._error_response(request_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
        except Exception as e:
            return None if not has_id else self._error_response(request_id, -32603, str(e))

        return None if not has_id else self._success_response(request_id, result)
    def _success_response(self, request_id: Any, result: Any) -> Dict:
        resp: Dict[str, Any] = {"jsonrpc": "2.0", "result": result}
        if request_id is not None:
            resp["id"] = request_id
        return resp
    def _error_response(self, request_id: Any, code: int, message: str) -> Dict:
        resp: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
        }
        if request_id is not None:
            resp["id"] = request_id
        return resp



    def _audit_log(self, tool: str, args: Dict, result_type: str, result_data: Any = None, duration: float = 0.0) -> None:
        """Append a JSON record to the audit log."""
        from datetime import datetime
        try:
            self.writer.mkdir_tmp(LOGS_DIR, parents=True, exist_ok=True)
            log_file = LOGS_DIR / "audit.jsonl"
            
            # Truncate large args for logging (e.g. file content)
            safe_args = args.copy()
            if "content" in safe_args and len(str(safe_args["content"])) > 200:
                safe_args["content"] = str(safe_args["content"])[:200] + "...(truncated)"
                
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "tool": tool,
                "arguments": safe_args,
                "status": result_type,
                "duration_ms": round(duration * 1000, 2),
                "result_summary": str(result_data)[:200] if result_data else None
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Audit log failure: {e}", file=sys.stderr)

    def _load_cortex_index(self) -> Dict:
        """Load the cached cortex index (NAVIGATION/CORTEX/cortex.json)."""
        cortex_path = NAVIGATION_ROOT / "CORTEX" / "cortex.json"
        if not cortex_path.exists():
            return {}
        try:
            return json.loads(cortex_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return {}

    def _search_cortex_index(self, query: str, limit: int = 20) -> List[Dict]:
        """Simple substring search over cortex index entities."""
        data = self._load_cortex_index()
        entities = data.get("entities", []) if isinstance(data, dict) else []
        if not query:
            return entities[:limit]
        needle = query.lower()
        results = []
        for entity in entities:
            haystack = " ".join(
                str(entity.get(field, "")) for field in ("path", "title", "summary", "tags")
            ).lower()
            if needle in haystack:
                results.append(entity)
            if len(results) >= limit:
                break
        return results

    def _context_records(self, record_type: str) -> List[Dict]:
        """Collect context records from LAW/CONTEXT/<record_type>."""
        context_dir = LAW_ROOT / "CONTEXT" / record_type
        records = []
        if not context_dir.exists():
            return records
        for path in sorted(context_dir.glob("*.md")):
            title = None
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            records.append({
                "path": str(path.relative_to(PROJECT_ROOT)),
                "title": title or path.stem,
            })
        return records

    def _find_skill_dir(self, skill_name: str) -> Optional[Path]:
        """Locate a skill directory by name under CAPABILITY/SKILLS."""
        if not skill_name:
            return None
        skills_root = CAPABILITY_ROOT / "SKILLS"
        if not skills_root.exists():
            return None
        for root, _, files in os.walk(skills_root):
            if Path(root).name == skill_name and "run.py" in files:
                return Path(root)
        return None

    def _handle_initialize(self, params: Dict) -> Dict:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        }

    def _handle_initialized(self, params: Dict) -> None:
        """Handle initialized notification."""
        return None

    def _handle_tools_list(self, params: Dict) -> Dict:
        """List available tools."""
        tools = []
        for tool_def in self.tools_schema.get("definitions", {}).values():
            tools.append({
                "name": tool_def.get("name"),
                "description": tool_def.get("description"),
                "inputSchema": tool_def.get("inputSchema", {})
            })
        return {"tools": tools}

    def _handle_tools_call(self, params: Dict) -> Dict:
        """Call a tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Dispatch to tool handlers
        tool_handlers = {
            # Read tools
            "context_search": self._tool_context_search,
            "context_review": self._tool_context_review,
            "canon_read": self._tool_canon_read,
            "codebook_lookup": self._tool_codebook_lookup,
            # Cassette network (unified query interface)
            "cassette_network_query": self._tool_cassette_network_query,
            "skill_discovery": self._tool_skill_discovery,
            "find_related": self._tool_find_related,
            "semantic_stats": self._tool_semantic_stats,
            "memory_save": self._tool_memory_save,
            "memory_query": self._tool_memory_query,
            "memory_recall": self._tool_memory_recall,
            "semantic_neighbors": self._tool_semantic_neighbors,
            "memory_stats": self._tool_memory_stats,
            # Write tools
            "skill_run": self._tool_skill_run,
            # Session info tool
            "session_info": self._tool_session_info,
        }

        handler = tool_handlers.get(tool_name)
        if handler:
            import time
            start_time = time.time()
            try:
                result = handler(arguments)
                duration = time.time() - start_time
                is_error = result.get("isError", False)
                self._audit_log(tool_name, arguments, "error" if is_error else "success", result, duration)
                # E.1.2: Track file/symbol/search access for ELO
                if not is_error:
                    self._track_tool_access(tool_name, arguments, result)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self._audit_log(tool_name, arguments, "crit_error", str(e), duration)
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Internal tool error: {str(e)}"
                    }],
                    "isError": True
                }
        else:
            self._audit_log(tool_name, arguments, "unknown_tool")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}"
                }],
                "isError": True
            }

    def _handle_resources_list(self, params: Dict) -> Dict:
        """List available resources."""
        resources = self.resources_schema.get("resources", [])
        return {"resources": resources}

    def _handle_resources_read(self, params: Dict) -> Dict:
        """Read a resource."""
        uri = params.get("uri", "")

        # Static file map
        uri_map = {
            "ags://canon/contract": LAW_ROOT / "CANON" / "CONTRACT.md",
            "ags://canon/invariants": LAW_ROOT / "CANON" / "INVARIANTS.md",
            "ags://canon/genesis": LAW_ROOT / "CANON" / "GENESIS.md",
            "ags://canon/versioning": LAW_ROOT / "CANON" / "VERSIONING.md",
            "ags://canon/arbitration": LAW_ROOT / "CANON" / "ARBITRATION.md",
            "ags://canon/deprecation": LAW_ROOT / "CANON" / "DEPRECATION.md",
            "ags://canon/migration": LAW_ROOT / "CANON" / "MIGRATION.md",
            "ags://maps/entrypoints": NAVIGATION_ROOT / "maps" / "ENTRYPOINTS.md",
            "ags://agents": PROJECT_ROOT / "AGENTS.md",
        }

        file_path = uri_map.get(uri)
        if file_path and file_path.exists():
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": content
                }]
            }
        
        # Dynamic resources
        elif uri == "ags://context/decisions":
            return self._dynamic_context_resource("decisions")
        elif uri == "ags://context/preferences":
            return self._dynamic_context_resource("preferences")
        elif uri == "ags://context/rejected":
            return self._dynamic_context_resource("rejected")
        elif uri == "ags://context/open":
            return self._dynamic_context_resource("open")
        elif uri == "ags://cortex/index":
            return self._dynamic_cortex_resource()
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    def _dynamic_context_resource(self, record_type: str) -> Dict:
        """Generate dynamic context resource content."""
        context_dir = LAW_ROOT / "CONTEXT" / record_type
        records = []
        if context_dir.exists():
            for path in sorted(context_dir.glob("*.md")):
                title = None
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                records.append({
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "title": title or path.stem,
                })
        content = json.dumps(records, indent=2)
        return {
            "contents": [{
                "uri": f"ags://context/{record_type}",
                "mimeType": "application/json",
                "text": content
            }]
        }
    
    def _dynamic_cortex_resource(self) -> Dict:
        """Generate dynamic cortex index resource."""
        cortex_path = NAVIGATION_ROOT / "CORTEX" / "cortex.json"
        content = cortex_path.read_text(encoding="utf-8", errors="ignore") if cortex_path.exists() else "{}"
        return {
            "contents": [{
                "uri": "ags://cortex/index",
                "mimeType": "application/json",
                "text": content
            }]
        }

    def _handle_prompts_list(self, params: Dict) -> Dict:
        """List available prompts."""
        return {
            "prompts": [
                {
                    "name": "genesis",
                    "description": "The Genesis Prompt for AGS session bootstrapping"
                },
                {
                    "name": "commit_ceremony",
                    "description": "Checklist for the commit ceremony"
                },
                {
                    "name": "skill_template",
                    "description": "Template for creating a new Skill"
                },
                {
                    "name": "conflict_resolution",
                    "description": "Guide for resolving conflicts in Canon (Arbitration)"
                },
                {
                    "name": "deprecation_workflow",
                    "description": "Checklist for deprecating tokens or features"
                }
            ]
        }

    def _handle_prompts_get(self, params: Dict) -> Dict:
        """Get a specific prompt."""
        prompt_name = params.get("name")

        if prompt_name == "genesis":
            genesis_path = LAW_ROOT / "CANON" / "GENESIS.md"
            if genesis_path.exists():
                content = genesis_path.read_text(encoding="utf-8")
                # Extract the prompt block
                return {
                    "description": "Genesis Prompt for AGS session bootstrapping",
                    "messages": [{
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": content
                        }
                    }]
                }

        if prompt_name == "skill_template":
            skill_md = (CAPABILITY_ROOT / "SKILLS" / "_TEMPLATE" / "SKILL.md").read_text(encoding="utf-8")
            run_py = (CAPABILITY_ROOT / "SKILLS" / "_TEMPLATE" / "run.py").read_text(encoding="utf-8")
            return {
                "description": "Template for creating a new Skill",
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Create a new SKILL following this template:\n\n### SKILL.md\n{skill_md}\n\n### run.py\n{run_py}"
                    }
                }]
            }

        if prompt_name == "conflict_resolution":
            arb_path = LAW_ROOT / "CANON" / "ARBITRATION.md"
            content = arb_path.read_text(encoding="utf-8") if arb_path.exists() else "ARBITRATION.md not found."
            return {
                "description": "Guide for resolving conflicts in Canon",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": content }
                }]
            }

        if prompt_name == "deprecation_workflow":
            dep_path = LAW_ROOT / "CANON" / "DEPRECATION.md"
            content = dep_path.read_text(encoding="utf-8") if dep_path.exists() else "DEPRECATION.md not found."
            return {
                "description": "Checklist for deprecating tokens or features",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": content }
                }]
            }

        return {
            "description": f"Prompt '{prompt_name}' not implemented",
            "messages": []
        }

    # Tool implementations
    def _tool_context_search(self, args: Dict) -> Dict:
        """Search context records in LAW/CONTEXT."""
        try:
            query = (args.get("query") or "").lower()
            record_type = args.get("type")
            tag_filters = [str(tag).lower() for tag in args.get("tags", []) or []]
            status_filter = (args.get("status") or "").lower()

            record_types = [record_type] if record_type else ["decisions", "preferences", "rejected", "open"]
            results = []
            for rtype in record_types:
                records = self._context_records(rtype)
                for record in records:
                    path = PROJECT_ROOT / record["path"]
                    content = path.read_text(encoding="utf-8", errors="ignore").lower()
                    if query and query not in content:
                        continue
                    if status_filter and status_filter not in content:
                        continue
                    if tag_filters and not all(tag in content for tag in tag_filters):
                        continue
                    results.append(record)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(results, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Context search error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_context_review(self, args: Dict) -> Dict:
        """Return a stub review summary for context records."""
        try:
            days = args.get("days")
            payload = {
                "checked_days": days,
                "overdue": [],
                "upcoming": [],
            }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(payload, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Context review error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_canon_read(self, args: Dict) -> Dict:
        """Read a canon file."""
        file_name = args.get("file", "").upper()
        canon_path = LAW_ROOT / "CANON" / f"{file_name}.md"

        if canon_path.exists():
            content = canon_path.read_text(encoding="utf-8", errors="ignore")
            return {
                "content": [{
                    "type": "text",
                    "text": content
                }]
            }
        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Canon file not found: {file_name}"
                }],
                "isError": True
            }

    @governed_tool
    def _tool_skill_run(self, args: Dict) -> Dict:
        """Execute a skill with the given input."""
        import subprocess
        import tempfile
        
        skill_name = args.get("skill", "")
        skill_input = args.get("input", {})
        
        if not skill_name:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: 'skill' parameter is required"
                }],
                "isError": True
            }
        
        # Validate skill exists
        skill_dir = self._find_skill_dir(skill_name)
        run_script = skill_dir / "run.py" if skill_dir else None

        if not skill_dir:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Skill '{skill_name}' not found. Available skills: {self._list_skills()}"
                }],
                "isError": True
            }
        
        if not run_script or not run_script.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Skill '{skill_name}' has no run.py"
                }],
                "isError": True
            }
        
        try:
            # Create temp files for input/output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
                json.dump(skill_input, f_in)
                input_path = f_in.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
                output_path = f_out.name
            
            # Run the skill
            result = subprocess.run(
                [sys.executable, str(run_script), input_path, output_path],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=60  # 60 second timeout
            )
            
            # Read output
            output_content = Path(output_path).read_text() if Path(output_path).exists() else "{}"
            
            # Clean up
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
            
            if result.returncode == 0:
                return {
                    "content": [{
                        "type": "text",
                        "text": output_content
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Skill failed (exit {result.returncode}):\n{result.stderr}\n\nOutput:\n{output_content}"
                    }],
                    "isError": True
                }
        except subprocess.TimeoutExpired:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Skill execution timed out (60s limit)"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Skill execution error: {str(e)}"
                }],
                "isError": True
            }
    
    def _list_skills(self) -> str:
        """List available skills."""
        skills_root = CAPABILITY_ROOT / "SKILLS"
        skills = []
        if skills_root.exists():
            for root, _, files in os.walk(skills_root):
                if "run.py" in files and "SKILL.md" in files:
                    rel = Path(root).relative_to(skills_root).as_posix()
                    skills.append(rel)
        return ", ".join(sorted(skills))

    def _tool_codebook_lookup(self, args: Dict) -> Dict:
        """Look up a codebook entry by ID with optional stacked filtering."""
        import subprocess

        entry_id = args.get("id", "")
        expand = args.get("expand", False)
        query = args.get("query", "")
        semantic = args.get("semantic", "")
        limit = args.get("limit", 10)
        list_all = args.get("list", False)

        cmd = [sys.executable, str(CAPABILITY_ROOT / "TOOLS" / "codebook_lookup.py")]

        if list_all:
            cmd.append("--list")
            cmd.append("--json")
        elif entry_id:
            cmd.append(entry_id)
            if query:
                cmd.extend(["--query", query, "--limit", str(limit), "--json"])
            elif semantic:
                cmd.extend(["--semantic", semantic, "--limit", str(limit), "--json"])
            elif expand:
                cmd.append("--expand")
            else:
                cmd.append("--json")
        else:
            cmd.extend(["--list", "--json"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0:
                return {
                    "content": [{
                        "type": "text",
                        "text": result.stdout
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Codebook lookup failed: {result.stderr or result.stdout}"
                    }],
                    "isError": True
                }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Codebook lookup error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_skill_discovery(self, args: Dict) -> Dict:
        """Find skills by semantic intent query."""
        try:
            from CAPABILITY.PRIMITIVES.skill_index import find_skills_by_intent

            query = args.get("query", "")
            top_k = args.get("top_k", 5)
            threshold = args.get("threshold")

            if not query:
                return {
                    "content": [{"type": "text", "text": "Error: 'query' parameter is required"}],
                    "isError": True
                }

            # Perform skill discovery
            result = find_skills_by_intent(
                query=query,
                top_k=top_k,
                threshold=threshold,
                emit_receipt=True
            )

            # Format results
            if not result.get("results"):
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No skills found matching query: '{query}'"
                    }]
                }

            # Build response text
            lines = [f"Found {len(result['results'])} skills matching '{query}':\n"]

            for i, skill_result in enumerate(result["results"], 1):
                skill_id = skill_result["skill_id"]
                score = skill_result["score"]
                metadata = skill_result.get("metadata", {})

                name = metadata.get("name", skill_id)
                description = metadata.get("description", "No description")
                purpose = metadata.get("purpose", "")

                lines.append(f"{i}. **{name}** (ID: `{skill_id}`, similarity: {score:.3f})")
                lines.append(f"   {description}")
                if purpose and purpose != description:
                    lines.append(f"   Purpose: {purpose[:150]}{'...' if len(purpose) > 150 else ''}")
                lines.append("")

            # Add receipt info
            if "receipt" in result:
                receipt = result["receipt"]
                lines.append(f"\n_Total candidates: {result['total_candidates']}, "
                           f"Model: {receipt.get('model_name', 'unknown')}_")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(lines)
                }]
            }

        except FileNotFoundError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Skill index not found. Run skill embedding first: {str(e)}"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Skill discovery error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_find_related(self, args: Dict) -> Dict:
        """Find related artifacts by embedding similarity."""
        try:
            from CAPABILITY.PRIMITIVES.cross_ref_index import find_related

            artifact_id = args.get("artifact_id", "")
            top_k = args.get("top_k", 5)
            threshold = args.get("threshold")

            if not artifact_id:
                return {
                    "content": [{"type": "text", "text": "Error: 'artifact_id' parameter is required"}],
                    "isError": True
                }

            # Perform cross-reference query
            result = find_related(
                artifact_id=artifact_id,
                top_k=top_k,
                threshold=threshold,
            )

            # Format results
            if not result.get("related"):
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No related artifacts found for: '{artifact_id}'"
                    }]
                }

            # Build response text
            lines = [f"Found {len(result['related'])} related artifacts for `{artifact_id}`:\n"]

            for i, related_item in enumerate(result["related"], 1):
                rel_id = related_item["artifact_id"]
                rel_type = related_item["artifact_type"]
                rel_path = related_item["artifact_path"]
                similarity = related_item["similarity"]
                metadata = related_item.get("metadata", {})

                lines.append(f"{i}. **{rel_type}**: `{rel_path}` (similarity: {similarity:.3f})")

                # Add type-specific metadata
                if rel_type == "canon":
                    tags = metadata.get("tags")
                    if tags:
                        lines.append(f"   Tags: {tags}")
                elif rel_type == "adr":
                    title = metadata.get("title")
                    status = metadata.get("status")
                    if title:
                        lines.append(f"   Title: {title}")
                    if status:
                        lines.append(f"   Status: {status}")
                elif rel_type == "skill":
                    # Metadata for skills is in JSON, parse if present
                    if isinstance(metadata, dict) and metadata:
                        name = metadata.get("name")
                        if name:
                            lines.append(f"   Name: {name}")

                lines.append("")

            # Add stats
            total_candidates = result.get("total_candidates", 0)
            lines.append(f"\n_Total candidates: {total_candidates}_")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(lines)
                }]
            }

        except FileNotFoundError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Cross-reference index not found. Run `build_cross_refs()` first: {str(e)}"
                }],
                "isError": True
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Cross-reference query error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_cassette_network_query(self, args: Dict) -> Dict:
        """Query the cassette network."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Cassette network not available"}],
                "isError": True
            }
        
        try:
            return self.semantic_adapter.cassette_network_query(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Cassette network error: {str(e)}"}],
                "isError": True
            }
    
    def _tool_semantic_stats(self, args: Dict) -> Dict:
        """Get statistics about semantic embeddings and cassette network."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }
        
        try:
            embedding_stats = self.semantic_adapter.get_embedding_stats()
            network_stats = self.semantic_adapter.get_network_status()
            
            stats = {
                "embeddings": embedding_stats,
                "network": network_stats,
                "semantic_available": True,
                "db_exists": (PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db").exists()
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(stats, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Semantic stats error: {str(e)}"}],
                "isError": True
            }
    
    def _tool_memory_save(self, args: Dict) -> Dict:
        """Save a memory to the resident cassette."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }

        try:
            return self.semantic_adapter.memory_save_tool(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory save error: {str(e)}"}],
                "isError": True
            }

    def _tool_memory_query(self, args: Dict) -> Dict:
        """Query memories using semantic search."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }

        try:
            return self.semantic_adapter.memory_query_tool(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory query error: {str(e)}"}],
                "isError": True
            }

    def _tool_memory_recall(self, args: Dict) -> Dict:
        """Retrieve a full memory by its hash."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }

        try:
            return self.semantic_adapter.memory_recall_tool(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory recall error: {str(e)}"}],
                "isError": True
            }

    def _tool_semantic_neighbors(self, args: Dict) -> Dict:
        """Find memories semantically similar to a given memory."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }

        try:
            return self.semantic_adapter.semantic_neighbors_tool(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Semantic neighbors error: {str(e)}"}],
                "isError": True
            }

    def _tool_memory_stats(self, args: Dict) -> Dict:
        """Get statistics about stored memories."""
        self._ensure_semantic_adapter()
        if not self.semantic_available or not self.semantic_adapter:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }

        try:
            return self.semantic_adapter.memory_stats_tool(args)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory stats error: {str(e)}"}],
                "isError": True
            }

    def _tool_session_info(self, args: Dict) -> Dict:
        """Get information about the current MCP session including session_id for ADR-021 compliance."""
        from datetime import datetime
        import os
        
        try:
            include_audit_log = args.get("include_audit_log", False)
            limit = int(args.get("limit", 10))
            
            # Basic session info
            session_info = {
                "session_id": self.session_id,
                "server_name": SERVER_NAME,
                "server_version": SERVER_VERSION,
                "mcp_version": MCP_VERSION,
                "connected_at": datetime.now().isoformat(),
                "project_root": str(PROJECT_ROOT),
                "audit_log_path": str(LOGS_DIR / "audit.jsonl"),
                "adr_021_compliant": True,
                "adr_021_note": "This session_id is automatically logged with all cortex queries and tool calls",
                "semantic_available": self.semantic_available
            }
            
            # Add semantic stats if available
            if self.semantic_available and self.semantic_adapter:
                try:
                    embedding_stats = self.semantic_adapter.get_embedding_stats()
                    session_info["semantic_stats"] = {
                        "embeddings_count": embedding_stats.get("total_embeddings", 0) if isinstance(embedding_stats, dict) else 0,
                        "semantic_ready": True
                    }
                except:
                    session_info["semantic_stats"] = {"semantic_ready": False}
            
            # Add audit log entries if requested
            if include_audit_log:
                audit_entries = []
                log_file = LOGS_DIR / "audit.jsonl"
                if log_file.exists():
                    lines = log_file.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
                    # Filter for this session_id and get most recent
                    for line in reversed(lines):
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("session_id") == self.session_id:
                                audit_entries.append(entry)
                                if len(audit_entries) >= limit:
                                    break
                        except:
                            continue
                session_info["audit_log_entries"] = audit_entries
                session_info["audit_log_count"] = len(audit_entries)
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(session_info, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Session info error: {str(e)}"
                }],
                "isError": True
            }

    def _tool_not_implemented(self, args: Dict) -> Dict:
        """Placeholder for unimplemented tools."""
        return {
            "content": [{
                "type": "text",
                "text": "This tool is staged but not yet implemented. See MCP/MCP_SPEC.md for the implementation roadmap."
            }],
            "isError": False
        }


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a buffered binary stream."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("EOF while reading message body")
        buf.extend(chunk)
    return bytes(buf)


def _read_message(stdin, mode: Optional[str]) -> Tuple[Optional[Dict], Optional[str]]:
    """Read one MCP message in either framed (Content-Length) or JSONL mode.

    Returns: (request_dict_or_none, detected_mode)
    """
    if mode == "jsonl":
        line = stdin.readline()
        if not line:
            return None, None
        # Skip blank lines
        while line in (b"\r\n", b"\n"):
            line = stdin.readline()
            if not line:
                return None, None
        return json.loads(line.decode("utf-8", errors="replace")), "jsonl"

    # framed or auto-detect
    first = stdin.readline()
    if not first:
        return None, None

    # Skip blank lines
    while first in (b"\r\n", b"\n"):
        first = stdin.readline()
        if not first:
            return None, None

    if not first.lower().startswith(b"content-length:"):
        # Auto-detect fallback: treat as JSONL.
        if mode == "framed":
            raise ValueError("Expected Content-Length header, got JSON line")
        return json.loads(first.decode("utf-8", errors="replace")), "jsonl"

    # Framed: read headers until blank line
    headers = [first]
    while True:
        line = stdin.readline()
        if not line:
            raise EOFError("EOF while reading headers")
        if line in (b"\r\n", b"\n"):
            break
        headers.append(line)

    content_length: Optional[int] = None
    for h in headers:
        if h.lower().startswith(b"content-length:"):
            content_length = int(h.split(b":", 1)[1].strip())
            break
    if content_length is None:
        raise ValueError("Missing Content-Length header")

    body = _read_exact(stdin, content_length)
    return json.loads(body.decode("utf-8", errors="replace")), "framed"


def _write_framed_json(stdout, message: Dict) -> None:
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stdout.write(header)
    stdout.write(body)
    stdout.flush()


def run_stdio():
    """Run the server in stdio mode.

    Supports:
    - MCP/LSP Content-Length framing (VS Code, Antigravity, most MCP clients)
    - Legacy newline-delimited JSON (some simpler clients)
    """
    server = AGSMCPServer()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    mode: Optional[str] = None  # "framed" | "jsonl" (auto-detected on first request)

    while True:
        try:
            request, detected = _read_message(stdin, mode)
            if request is None:
                break
            if detected is None:
                break
            if mode is None:
                mode = detected

            if not isinstance(request, dict):
                continue

            response = server.handle_request(request)
            if response is None:
                continue

            if mode == "framed":
                _write_framed_json(stdout, response)
            else:
                stdout.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
                stdout.flush()

        except json.JSONDecodeError:
            if mode == "framed":
                _write_framed_json(stdout, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
            continue
        except EOFError:
            break
        except Exception as e:
            if mode == "framed":
                _write_framed_json(stdout, {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}})
            else:
                print(f"[ERROR] MCP stdio loop: {e}", file=sys.stderr)
            continue


def main():
    parser = argparse.ArgumentParser(description="AGS MCP Server")
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode (not implemented)")
    parser.add_argument("--test", action="store_true", help="Run a test request")
    args = parser.parse_args()

    if args.http:
        print("HTTP mode not implemented. Use stdio mode.", file=sys.stderr)
        sys.exit(1)

    if args.test:
        # Test mode: run sample requests for all implemented tools
        server = AGSMCPServer()

        print("="*60)
        print("AGS MCP SERVER TEST")
        print("="*60)

        # Initialize
        init_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        print("\n[OK] Initialize:", init_response["result"]["serverInfo"])

        # List tools
        tools_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        tool_names = [t["name"] for t in tools_response["result"]["tools"]]
        print(f"\n[OK] Tools available: {tool_names}")

        # List resources
        resources_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        })
        print(f"\n[OK] Resources available: {len(resources_response['result']['resources'])} resources")

        # Test cortex_query
        print("\n--- Testing cortex_query ---")
        cortex_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "cortex_query", "arguments": {"query": "packer"}}
        })
        content = cortex_response["result"]["content"][0]["text"]
        is_error = cortex_response["result"].get("isError", False)
        print(f"[OK] cortex_query('packer'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
        if not is_error and content:
            try:
                results = json.loads(content)
                print(f"  Found {len(results)} results")
            except:
                print(f"  Output: {content[:100]}...")

        # Test context_search
        print("\n--- Testing context_search ---")
        context_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "context_search", "arguments": {"type": "decisions"}}
        })
        content = context_response["result"]["content"][0]["text"]
        is_error = context_response["result"].get("isError", False)
        print(f"[OK] context_search(type='decisions'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
        if not is_error and content:
            try:
                results = json.loads(content)
                print(f"  Found {len(results)} records")
            except:
                print(f"  Output: {content[:100]}...")

        # Test context_review
        print("\n--- Testing context_review ---")
        review_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "context_review", "arguments": {"days": 30}}
        })
        content = review_response["result"]["content"][0]["text"]
        is_error = review_response["result"].get("isError", False)
        print(f"[OK] context_review(days=30): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")
        if not is_error and content:
            try:
                results = json.loads(content)
                overdue = len(results.get("overdue", []))
                upcoming = len(results.get("upcoming", []))
                print(f"  Overdue: {overdue}, Upcoming: {upcoming}")
            except:
                print(f"  Output: {content[:100]}...")

        # Test canon_read
        print("\n--- Testing canon_read ---")
        canon_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "canon_read", "arguments": {"file": "CONTRACT"}}
        })
        content = canon_response["result"]["content"][0]["text"]
        is_error = canon_response["result"].get("isError", False)
        print(f"[OK] canon_read('CONTRACT'): {'ERROR' if is_error else 'OK'} ({len(content)} chars)")

        # Test resource reading
        print("\n--- Testing resources/read ---")
        read_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "resources/read",
            "params": {"uri": "ags://canon/genesis"}
        })
        content = read_response["result"]["contents"][0]["text"]
        print(f"[OK] resources/read('ags://canon/genesis'): {len(content)} chars")

        # Test prompts/get
        print("\n--- Testing prompts/get ---")
        prompt_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "prompts/get",
            "params": {"name": "genesis"}
        })
        messages = prompt_response["result"].get("messages", [])
        print(f"[OK] prompts/get('genesis'): {len(messages)} messages")

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        return

    # Default: stdio mode
    run_stdio()


if __name__ == "__main__":
    main()
