"""
Z.2.6 – ROOT AUDIT Tool

Provides deterministic, fail-closed audit of root completeness and GC safety.

Two modes:
- Mode A: General root safety audit (verify roots enumerate, reachable set computable)
- Mode B: Run completeness check (verify all required outputs are rooted and reachable)

Public API:
    root_audit() - Main audit entry point
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

from CAPABILITY.CAS.cas import _CAS_ROOT, _get_object_path
from CAPABILITY.RUNS.records import load_output_hashes, RunRecordException, InvalidInputException


# ============================================================================
# Root enumeration (reused from GC with audit-specific error handling)
# ============================================================================

def _enumerate_run_roots(runs_dir: Path) -> tuple[List[str], List[str]]:
    """
    Enumerate roots from RUN_ROOTS.json.

    Returns:
        Tuple of (roots, errors) - errors list for fail-closed reporting
    """
    root_file = runs_dir / "RUN_ROOTS.json"

    if not root_file.exists():
        return ([], [])

    try:
        with open(root_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return ([], [f"RUN_ROOTS: Invalid JSON: {e}"])

    if not isinstance(data, list):
        return ([], [f"RUN_ROOTS: Must be a list, got {type(data).__name__}"])

    roots = []
    errors = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            errors.append(f"RUN_ROOTS[{i}]: Must be string, got {type(item).__name__}")
            continue

        if len(item) != 64 or not all(c in '0123456789abcdef' for c in item):
            errors.append(f"RUN_ROOTS[{i}]: Invalid hash format: {item}")
            continue

        roots.append(item)

    return (roots, errors)


def _enumerate_pin_roots(runs_dir: Path) -> tuple[List[str], List[str]]:
    """
    Enumerate roots from GC_PINS.json.

    Returns:
        Tuple of (roots, errors) - errors list for fail-closed reporting
    """
    pin_file = runs_dir / "GC_PINS.json"

    if not pin_file.exists():
        return ([], [])

    try:
        with open(pin_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return ([], [f"GC_PINS: Invalid JSON: {e}"])

    if not isinstance(data, list):
        return ([], [f"GC_PINS: Must be a list, got {type(data).__name__}"])

    pins = []
    errors = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            errors.append(f"GC_PINS[{i}]: Must be string, got {type(item).__name__}")
            continue

        if len(item) != 64 or not all(c in '0123456789abcdef' for c in item):
            errors.append(f"GC_PINS[{i}]: Invalid hash format: {item}")
            continue

        pins.append(item)

    return (pins, errors)


def _compute_file_hash(file_path: Path) -> Optional[str]:
    """
    Compute SHA-256 hash of file contents.

    Returns:
        Hash string or None if file doesn't exist
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


# ============================================================================
# Reachability computation (same semantics as GC)
# ============================================================================

def _traverse_references(roots: Set[str], cas_root: Path) -> Set[str]:
    """
    Traverse references from roots to build reachable set.

    Uses identical semantics to Z.2.5 GC mark phase.
    Current implementation: trivial traversal (roots = reachable set).
    Future: When GC implements deep traversal, this MUST match that logic.

    Args:
        roots: Set of root hashes
        cas_root: Path to CAS storage root

    Returns:
        Set of reachable hashes
    """
    # For Z.2.5/Z.2.6, we only mark the roots as reachable
    # No deep traversal is implemented yet
    reachable = set(roots)

    # Future: If CAS objects contain references to other CAS objects,
    # implement BFS/DFS traversal here (same as GC)

    return reachable


# ============================================================================
# CAS enumeration (reused from GC)
# ============================================================================

def _enumerate_cas_blobs(cas_root: Path) -> List[str]:
    """
    Enumerate all CAS blobs deterministically.

    Returns:
        Sorted list of all CAS blob hashes
    """
    blobs = []

    if not cas_root.exists():
        return blobs

    # Walk the CAS directory structure: a/bc/abcdef...
    for prefix1_dir in sorted(cas_root.iterdir()):
        if not prefix1_dir.is_dir():
            continue

        for prefix2_dir in sorted(prefix1_dir.iterdir()):
            if not prefix2_dir.is_dir():
                continue

            for blob_file in sorted(prefix2_dir.iterdir()):
                if blob_file.is_file() and not blob_file.name.endswith('.tmp'):
                    # Validate hash format
                    hash_str = blob_file.name
                    if len(hash_str) == 64 and all(c in '0123456789abcdef' for c in hash_str):
                        blobs.append(hash_str)

    return sorted(blobs)


def _compute_cas_snapshot_hash(blobs: List[str]) -> str:
    """
    Compute deterministic hash of CAS snapshot.

    Args:
        blobs: Sorted list of blob hashes

    Returns:
        SHA-256 hash of the snapshot
    """
    # Canonical encoding: one hash per line, sorted
    snapshot_bytes = '\n'.join(blobs).encode('utf-8')
    return hashlib.sha256(snapshot_bytes).hexdigest()


# ============================================================================
# Mode B: Required outputs verification
# ============================================================================

def _verify_required_outputs(
    output_hashes_record: str,
    reachable: Set[str],
    cas_blobs: Set[str]
) -> Dict[str, Any]:
    """
    Verify that all required outputs are reachable and exist.

    Args:
        output_hashes_record: CAS hash of OUTPUT_HASHES record
        reachable: Set of reachable hashes
        cas_blobs: Set of all CAS blob hashes

    Returns:
        Dict with:
        - 'required_hashes': list[str] - All required output hashes
        - 'required_missing': list[str] - Hashes not found in CAS
        - 'required_unreachable': list[str] - Hashes not reachable from roots
        - 'errors': list[str] - Any errors encountered
    """
    errors = []
    required_hashes = []
    required_missing = []
    required_unreachable = []

    # Validate output_hashes_record format
    if len(output_hashes_record) != 64 or not all(c in '0123456789abcdef' for c in output_hashes_record):
        errors.append(f"OUTPUT_HASHES record: Invalid hash format: {output_hashes_record}")
        return {
            'required_hashes': [],
            'required_missing': [],
            'required_unreachable': [],
            'errors': errors
        }

    # Load OUTPUT_HASHES record
    try:
        required_hashes = load_output_hashes(output_hashes_record)
    except InvalidInputException as e:
        errors.append(f"OUTPUT_HASHES record: Invalid format: {e}")
        return {
            'required_hashes': [],
            'required_missing': [],
            'required_unreachable': [],
            'errors': errors
        }
    except RunRecordException as e:
        errors.append(f"OUTPUT_HASHES record: {e}")
        return {
            'required_hashes': [],
            'required_missing': [],
            'required_unreachable': [],
            'errors': errors
        }
    except Exception as e:
        errors.append(f"OUTPUT_HASHES record: Unexpected error: {e}")
        return {
            'required_hashes': [],
            'required_missing': [],
            'required_unreachable': [],
            'errors': errors
        }

    # Check each required hash
    for h in required_hashes:
        # Check if hash exists in CAS
        if h not in cas_blobs:
            required_missing.append(h)

        # Check if hash is reachable from roots
        if h not in reachable:
            required_unreachable.append(h)

    return {
        'required_hashes': required_hashes,
        'required_missing': sorted(required_missing),
        'required_unreachable': sorted(required_unreachable),
        'errors': errors
    }


# ============================================================================
# Public API
# ============================================================================

def root_audit(
    *,
    output_hashes_record: Optional[str] = None,
    dry_run: bool = True,
    cas_root: Optional[Path] = None,
    runs_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Deterministic, fail-closed audit of root completeness and GC safety.

    Mode A (output_hashes_record=None):
        Verify roots enumerate, reachable set is computable, receipt is deterministic.

    Mode B (output_hashes_record=<hash>):
        Mode A checks + verify all required outputs are rooted and reachable.

    Args:
        output_hashes_record: Optional CAS hash of OUTPUT_HASHES record (Mode B)
        dry_run: Always True for audit (kept for interface symmetry)
        cas_root: Optional CAS storage root (defaults to CAPABILITY/CAS/storage)
        runs_dir: Optional RUNS directory (defaults to CAPABILITY/RUNS)

    Returns:
        Deterministic receipt dict with verdict ("PASS" or "FAIL")

    Raises:
        None - all errors reported via receipt (fail-closed)
    """
    # Default paths
    if cas_root is None:
        cas_root = _CAS_ROOT
    if runs_dir is None:
        runs_dir = Path("CAPABILITY/RUNS")

    errors = []

    # ========================================================================
    # PHASE 1: Root enumeration
    # ========================================================================

    run_roots, run_errors = _enumerate_run_roots(runs_dir)
    pin_roots, pin_errors = _enumerate_pin_roots(runs_dir)

    errors.extend(run_errors)
    errors.extend(pin_errors)

    # If enumeration failed, fail-closed immediately
    if errors:
        return {
            'mode': 'audit',
            'root_sources': [],
            'roots_count': 0,
            'reachable_hashes_count': 0,
            'required_check': {
                'enabled': output_hashes_record is not None,
                'output_hashes_record': output_hashes_record
            },
            'required_total': 0,
            'required_missing': [],
            'required_unreachable': [],
            'errors': sorted(errors),
            'cas_snapshot_hash': '',
            'verdict': 'FAIL'
        }

    # Build root sources metadata
    run_roots_file = runs_dir / "RUN_ROOTS.json"
    gc_pins_file = runs_dir / "GC_PINS.json"

    root_sources = [
        {
            'name': 'RUN_ROOTS',
            'path': str(run_roots_file),
            'exists': run_roots_file.exists(),
            'content_hash': _compute_file_hash(run_roots_file)
        },
        {
            'name': 'GC_PINS',
            'path': str(gc_pins_file),
            'exists': gc_pins_file.exists(),
            'content_hash': _compute_file_hash(gc_pins_file)
        }
    ]

    # Deduplicate roots
    all_roots = set(run_roots) | set(pin_roots)
    roots_count = len(all_roots)

    # Policy: Empty roots => FAIL for audit (no override)
    if roots_count == 0:
        errors.append("POLICY_LOCK: Empty roots detected. Audit requires at least one root.")
        return {
            'mode': 'audit',
            'root_sources': root_sources,
            'roots_count': 0,
            'reachable_hashes_count': 0,
            'required_check': {
                'enabled': output_hashes_record is not None,
                'output_hashes_record': output_hashes_record
            },
            'required_total': 0,
            'required_missing': [],
            'required_unreachable': [],
            'errors': sorted(errors),
            'cas_snapshot_hash': '',
            'verdict': 'FAIL'
        }

    # ========================================================================
    # PHASE 2: Reachability computation
    # ========================================================================

    try:
        reachable = _traverse_references(all_roots, cas_root)
        reachable_count = len(reachable)
    except Exception as e:
        errors.append(f"Reachability traversal failed: {e}")
        return {
            'mode': 'audit',
            'root_sources': root_sources,
            'roots_count': roots_count,
            'reachable_hashes_count': 0,
            'required_check': {
                'enabled': output_hashes_record is not None,
                'output_hashes_record': output_hashes_record
            },
            'required_total': 0,
            'required_missing': [],
            'required_unreachable': [],
            'errors': sorted(errors),
            'cas_snapshot_hash': '',
            'verdict': 'FAIL'
        }

    # ========================================================================
    # PHASE 3: CAS snapshot
    # ========================================================================

    try:
        cas_blobs = _enumerate_cas_blobs(cas_root)
        cas_snapshot_hash = _compute_cas_snapshot_hash(cas_blobs)
        cas_blobs_set = set(cas_blobs)
    except Exception as e:
        errors.append(f"CAS enumeration failed: {e}")
        return {
            'mode': 'audit',
            'root_sources': root_sources,
            'roots_count': roots_count,
            'reachable_hashes_count': reachable_count,
            'required_check': {
                'enabled': output_hashes_record is not None,
                'output_hashes_record': output_hashes_record
            },
            'required_total': 0,
            'required_missing': [],
            'required_unreachable': [],
            'errors': sorted(errors),
            'cas_snapshot_hash': '',
            'verdict': 'FAIL'
        }

    # ========================================================================
    # PHASE 4: Mode B - Required outputs verification
    # ========================================================================

    required_total = 0
    required_missing = []
    required_unreachable = []

    if output_hashes_record is not None:
        # Mode B: Verify required outputs
        result = _verify_required_outputs(output_hashes_record, reachable, cas_blobs_set)

        errors.extend(result['errors'])
        required_total = len(result['required_hashes'])
        required_missing = result['required_missing']
        required_unreachable = result['required_unreachable']

    # ========================================================================
    # PHASE 5: Verdict computation
    # ========================================================================

    # Sort errors for determinism
    errors = sorted(errors)

    # Mode A: PASS if no errors and roots > 0
    # Mode B: Mode A + required_missing empty + required_unreachable empty
    if errors:
        verdict = 'FAIL'
    elif roots_count == 0:
        verdict = 'FAIL'
    elif output_hashes_record is not None:
        # Mode B checks
        if required_missing or required_unreachable:
            verdict = 'FAIL'
        else:
            verdict = 'PASS'
    else:
        # Mode A
        verdict = 'PASS'

    # ========================================================================
    # Return deterministic receipt
    # ========================================================================

    return {
        'mode': 'audit',
        'root_sources': root_sources,
        'roots_count': roots_count,
        'reachable_hashes_count': reachable_count,
        'required_check': {
            'enabled': output_hashes_record is not None,
            'output_hashes_record': output_hashes_record
        },
        'required_total': required_total,
        'required_missing': required_missing,
        'required_unreachable': required_unreachable,
        'errors': errors,
        'cas_snapshot_hash': cas_snapshot_hash,
        'verdict': verdict
    }
