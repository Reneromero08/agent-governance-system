"""
Z.2.5 – Garbage Collection for CAS

Implements a two-phase mark-and-sweep GC for CAS storage:
1. MARK: Enumerate roots and traverse references to build reachable set
2. SWEEP: Delete unreferenced blobs (or report in dry-run mode)

Policy B (POLICY LOCK):
- If root enumeration yields ZERO roots, GC MUST FAIL-CLOSED and perform ZERO deletions
- A full sweep is ONLY permitted when allow_empty_roots=True is explicitly provided

Deterministic behavior:
- Same inputs => same outputs
- Stable ordering at every step
- Reproducible reports
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Any
import threading

from CAPABILITY.CAS.cas import _CAS_ROOT, _get_object_path, InvalidHashException


# ============================================================================
# Exceptions
# ============================================================================

class GCException(Exception):
    """Base exception for GC operations"""
    pass


class RootEnumerationException(GCException):
    """Raised when root enumeration fails"""
    pass


class LockException(GCException):
    """Raised when GC lock cannot be acquired"""
    pass


# ============================================================================
# Global GC lock (single-instance enforcement)
# ============================================================================

_gc_lock = threading.Lock()


# ============================================================================
# Root enumeration
# ============================================================================

def _enumerate_run_roots(runs_dir: Path) -> List[str]:
    """
    Enumerate roots from run records.
    
    Reads from CAPABILITY/RUNS/RUN_ROOTS.json if it exists.
    Returns empty list if file doesn't exist (not an error).
    
    Args:
        runs_dir: Path to RUNS directory
        
    Returns:
        List of CAS hashes (validated)
        
    Raises:
        RootEnumerationException: If file is malformed or contains invalid hashes
    """
    root_file = runs_dir / "RUN_ROOTS.json"
    
    if not root_file.exists():
        return []
    
    try:
        with open(root_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise RootEnumerationException(f"Failed to read RUN_ROOTS.json: {e}")
    
    if not isinstance(data, list):
        raise RootEnumerationException(f"RUN_ROOTS.json must contain a list, got {type(data).__name__}")
    
    # Validate all hashes
    roots = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise RootEnumerationException(f"Root at index {i} must be a string, got {type(item).__name__}")
        
        if len(item) != 64 or not all(c in '0123456789abcdef' for c in item):
            raise RootEnumerationException(f"Root at index {i} has invalid hash format: {item}")
        
        roots.append(item)
    
    return roots


def _enumerate_pin_roots(runs_dir: Path) -> List[str]:
    """
    Enumerate roots from pin file.
    
    Reads from CAPABILITY/RUNS/GC_PINS.json if it exists.
    Returns empty list if file doesn't exist (not an error).
    
    Args:
        runs_dir: Path to RUNS directory
        
    Returns:
        List of CAS hashes (validated)
        
    Raises:
        RootEnumerationException: If file is malformed or contains invalid hashes
    """
    pin_file = runs_dir / "GC_PINS.json"
    
    if not pin_file.exists():
        return []
    
    try:
        with open(pin_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise RootEnumerationException(f"Failed to read GC_PINS.json: {e}")
    
    if not isinstance(data, list):
        raise RootEnumerationException(f"GC_PINS.json must contain a list, got {type(data).__name__}")
    
    # Validate all hashes
    pins = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise RootEnumerationException(f"Pin at index {i} must be a string, got {type(item).__name__}")
        
        if len(item) != 64 or not all(c in '0123456789abcdef' for c in item):
            raise RootEnumerationException(f"Pin at index {i} has invalid hash format: {item}")
        
        pins.append(item)
    
    return pins


def _enumerate_all_roots(runs_dir: Path) -> Dict[str, Any]:
    """
    Enumerate all roots from all sources.
    
    Args:
        runs_dir: Path to RUNS directory
        
    Returns:
        Dict with:
        - 'sources': list of source identifiers
        - 'roots': deduplicated set of all root hashes
        
    Raises:
        RootEnumerationException: If any source fails to enumerate
    """
    run_roots = _enumerate_run_roots(runs_dir)
    pin_roots = _enumerate_pin_roots(runs_dir)
    
    # Deduplicate and combine
    all_roots = set(run_roots) | set(pin_roots)
    
    sources = []
    if run_roots:
        sources.append(f"RUN_ROOTS.json ({len(run_roots)} roots)")
    if pin_roots:
        sources.append(f"GC_PINS.json ({len(pin_roots)} roots)")
    
    return {
        'sources': sources,
        'roots': all_roots
    }


# ============================================================================
# Mark phase (traverse references)
# ============================================================================

def _traverse_references(roots: Set[str], cas_root: Path) -> Set[str]:
    """
    Traverse references from roots to build reachable set.
    
    For Z.2.5, we implement a simple traversal that only marks the roots themselves
    as reachable. Future phases may implement deeper traversal if CAS objects
    contain references to other CAS objects.
    
    Args:
        roots: Set of root hashes
        cas_root: Path to CAS storage root
        
    Returns:
        Set of reachable hashes (currently just the roots themselves)
    """
    # For Z.2.5, we only mark the roots as reachable
    # No deep traversal is implemented yet
    reachable = set(roots)
    
    # Future: If CAS objects contain references to other CAS objects,
    # implement BFS/DFS traversal here
    
    return reachable


# ============================================================================
# Sweep phase (enumerate and delete candidates)
# ============================================================================

def _enumerate_cas_blobs(cas_root: Path) -> List[str]:
    """
    Enumerate all CAS blobs deterministically.
    
    Args:
        cas_root: Path to CAS storage root
        
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
    Compute a deterministic hash of the CAS snapshot.
    
    Args:
        blobs: Sorted list of blob hashes
        
    Returns:
        SHA-256 hash of the snapshot
    """
    # Canonical encoding: one hash per line, sorted
    snapshot_bytes = '\n'.join(blobs).encode('utf-8')
    return hashlib.sha256(snapshot_bytes).hexdigest()


def _get_object_path_from_root(hash_str: str, cas_root: Path) -> Path:
    """Get the path for an object relative to a specific CAS root."""
    prefix1 = hash_str[0]
    prefix2 = hash_str[1:3]
    return cas_root / prefix1 / prefix2 / hash_str


def _delete_blob(hash_str: str, cas_root: Path) -> bool:
    """
    Delete a single blob from CAS.
    
    Args:
        hash_str: Hash of blob to delete
        cas_root: Path to CAS storage root
        
    Returns:
        True if deleted, False if already missing
    """
    try:
        obj_path = _get_object_path_from_root(hash_str, cas_root)
        if obj_path.exists():
            obj_path.unlink()
            return True
        return False
    except Exception:
        return False


# ============================================================================
# Main GC entry point
# ============================================================================

def gc_collect(
    *,
    dry_run: bool = True,
    allow_empty_roots: bool = False,
    cas_root: Path = None,
    runs_dir: Path = None
) -> Dict[str, Any]:
    """
    Perform garbage collection on CAS storage.
    
    Two-phase model:
    1. MARK: Enumerate roots and traverse references to build reachable set
    2. SWEEP: Delete unreferenced blobs (or report in dry-run mode)
    
    Policy B (POLICY LOCK):
    - If root enumeration yields ZERO roots and allow_empty_roots=False:
      => FAIL-CLOSED: delete nothing, return error
    - If root enumeration yields ZERO roots and allow_empty_roots=True:
      => Full sweep allowed (delete all unreferenced blobs)
    
    Args:
        dry_run: If True, do not delete anything (report only)
        allow_empty_roots: If True, allow full sweep when roots==0
        cas_root: Path to CAS storage root (default: CAPABILITY/CAS/storage)
        runs_dir: Path to RUNS directory (default: CAPABILITY/RUNS)
        
    Returns:
        Dict containing:
        - mode: "dry_run" or "apply"
        - allow_empty_roots: bool
        - root_sources: list of source identifiers
        - roots_count: int
        - reachable_hashes_count: int
        - candidate_hashes_count: int
        - deleted_hashes: list[str] (stable order)
        - skipped_hashes: list[{"hash": str, "reason": str}] (stable order)
        - errors: list[str] (empty on success)
        - cas_snapshot_hash: str
        
    Raises:
        GCException: On critical failures
    """
    # Default paths
    if cas_root is None:
        cas_root = _CAS_ROOT
    if runs_dir is None:
        runs_dir = Path("CAPABILITY/RUNS")
    
    errors = []
    deleted_hashes = []
    skipped_hashes = []
    
    # ========================================================================
    # PHASE 1: MARK (read-only)
    # ========================================================================
    
    try:
        root_data = _enumerate_all_roots(runs_dir)
        root_sources = root_data['sources']
        roots = root_data['roots']
    except RootEnumerationException as e:
        # Root enumeration failure => fail-closed
        return {
            'mode': 'dry_run' if dry_run else 'apply',
            'allow_empty_roots': allow_empty_roots,
            'root_sources': [],
            'roots_count': 0,
            'reachable_hashes_count': 0,
            'candidate_hashes_count': 0,
            'deleted_hashes': [],
            'skipped_hashes': [],
            'errors': [f"Root enumeration failed: {e}"],
            'cas_snapshot_hash': ''
        }
    
    # Policy B: Check for empty roots
    if len(roots) == 0 and not allow_empty_roots:
        return {
            'mode': 'dry_run' if dry_run else 'apply',
            'allow_empty_roots': allow_empty_roots,
            'root_sources': root_sources,
            'roots_count': 0,
            'reachable_hashes_count': 0,
            'candidate_hashes_count': 0,
            'deleted_hashes': [],
            'skipped_hashes': [],
            'errors': ['POLICY_LOCK: Empty roots detected and allow_empty_roots=False. Fail-closed: no deletions.'],
            'cas_snapshot_hash': ''
        }
    
    # Traverse references to build reachable set
    try:
        reachable = _traverse_references(roots, cas_root)
    except Exception as e:
        # Traversal failure => fail-closed
        return {
            'mode': 'dry_run' if dry_run else 'apply',
            'allow_empty_roots': allow_empty_roots,
            'root_sources': root_sources,
            'roots_count': len(roots),
            'reachable_hashes_count': 0,
            'candidate_hashes_count': 0,
            'deleted_hashes': [],
            'skipped_hashes': [],
            'errors': [f"Reference traversal failed: {e}"],
            'cas_snapshot_hash': ''
        }
    
    # ========================================================================
    # PHASE 2: SWEEP
    # ========================================================================
    
    # Enumerate all CAS blobs
    try:
        all_blobs = _enumerate_cas_blobs(cas_root)
        cas_snapshot_hash = _compute_cas_snapshot_hash(all_blobs)
    except Exception as e:
        return {
            'mode': 'dry_run' if dry_run else 'apply',
            'allow_empty_roots': allow_empty_roots,
            'root_sources': root_sources,
            'roots_count': len(roots),
            'reachable_hashes_count': len(reachable),
            'candidate_hashes_count': 0,
            'deleted_hashes': [],
            'skipped_hashes': [],
            'errors': [f"CAS enumeration failed: {e}"],
            'cas_snapshot_hash': ''
        }
    
    # Compute delete candidates
    all_blobs_set = set(all_blobs)
    candidates = sorted(all_blobs_set - reachable)
    
    if dry_run:
        # Dry run: report only, no deletions
        return {
            'mode': 'dry_run',
            'allow_empty_roots': allow_empty_roots,
            'root_sources': root_sources,
            'roots_count': len(roots),
            'reachable_hashes_count': len(reachable),
            'candidate_hashes_count': len(candidates),
            'deleted_hashes': [],
            'skipped_hashes': [{'hash': h, 'reason': 'dry_run'} for h in candidates],
            'errors': [],
            'cas_snapshot_hash': cas_snapshot_hash
        }
    
    # Apply mode: acquire lock and delete
    if not _gc_lock.acquire(blocking=False):
        return {
            'mode': 'apply',
            'allow_empty_roots': allow_empty_roots,
            'root_sources': root_sources,
            'roots_count': len(roots),
            'reachable_hashes_count': len(reachable),
            'candidate_hashes_count': len(candidates),
            'deleted_hashes': [],
            'skipped_hashes': [],
            'errors': ['Failed to acquire GC lock (another GC may be running)'],
            'cas_snapshot_hash': cas_snapshot_hash
        }
    
    try:
        # Delete candidates in stable order
        for hash_str in candidates:
            try:
                if _delete_blob(hash_str, cas_root):
                    deleted_hashes.append(hash_str)
                else:
                    skipped_hashes.append({'hash': hash_str, 'reason': 'already_missing'})
            except Exception as e:
                skipped_hashes.append({'hash': hash_str, 'reason': f'delete_failed: {e}'})
    finally:
        _gc_lock.release()
    
    return {
        'mode': 'apply',
        'allow_empty_roots': allow_empty_roots,
        'root_sources': root_sources,
        'roots_count': len(roots),
        'reachable_hashes_count': len(reachable),
        'candidate_hashes_count': len(candidates),
        'deleted_hashes': deleted_hashes,
        'skipped_hashes': skipped_hashes,
        'errors': [],
        'cas_snapshot_hash': cas_snapshot_hash
    }
