#!/usr/bin/env python3
"""
MCP Validation - CMP-01 path governance and SPECTRUM-02 bundle verification.

Extracted from server.py for modularity. Originally ported from CAT LAB server_CATDPT.py.

Provides:
- CMP-01 path validation (forbidden overlap, durable/catalytic roots)
- JobSpec path validation
- Post-run output verification
- SPECTRUM-02 bundle generation and verification
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .primitives import compute_hash, get_validator_build_id, VALIDATOR_SEMVER, SUPPORTED_VALIDATOR_SEMVERS


# =============================================================================
# PATH CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


# =============================================================================
# CMP-01 PATH VALIDATION
# =============================================================================

def is_path_under_root(path: Path, root: Path) -> bool:
    """Component-safe check if path is under root (not just string prefix)."""
    try:
        return path.is_relative_to(root)
    except AttributeError:
        # Python 3.8 fallback
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def validate_single_path(
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
    if not is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
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
        if is_path_under_root(abs_path, forbidden_abs) or is_path_under_root(forbidden_abs, abs_path):
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
        if is_path_under_root(abs_path, root_abs):
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


def check_containment_overlap(
    paths: List[str],
    json_pointer_base: str
) -> List[Dict]:
    """Check for containment overlap between paths in the same list.

    Policy: Exact duplicates (same resolved path) are allowed/deduped.
    Only flag when one path strictly contains another.
    """
    errors = []
    abs_paths = []

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

            # Check if one strictly contains the other
            if is_path_under_root(abs_a, abs_b) or is_path_under_root(abs_b, abs_a):
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


def validate_jobspec_paths(task_spec: Dict) -> Dict:
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
        path_errors = validate_single_path(
            domain,
            f"/catalytic_domains/{idx}",
            CATALYTIC_ROOTS,
            "CATALYTIC_OUTSIDE_ROOT"
        )
        errors.extend(path_errors)

    # Check containment overlap within catalytic_domains
    if len(catalytic_domains) > 1:
        errors.extend(check_containment_overlap(
            catalytic_domains,
            "/catalytic_domains"
        ))

    # Validate outputs.durable_paths
    outputs = task_spec.get("outputs", {})
    durable_paths = outputs.get("durable_paths", [])
    for idx, dpath in enumerate(durable_paths):
        path_errors = validate_single_path(
            dpath,
            f"/outputs/durable_paths/{idx}",
            DURABLE_ROOTS,
            "OUTPUT_OUTSIDE_DURABLE_ROOT"
        )
        errors.extend(path_errors)

    # Check containment overlap within durable_paths
    if len(durable_paths) > 1:
        errors.extend(check_containment_overlap(
            durable_paths,
            "/outputs/durable_paths"
        ))

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def verify_post_run_outputs(run_id: str) -> Dict:
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

        # Check for absolute/traversal paths
        if Path(raw_path).is_absolute():
            errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Absolute paths are not allowed: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "reason": "absolute_path"}
            })
            continue

        if ".." in Path(raw_path).parts:
            errors.append({
                "code": "PATH_CONTAINS_TRAVERSAL",
                "message": f"Path contains forbidden traversal segment '..': {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "segments": list(Path(raw_path).parts)}
            })
            continue

        abs_path = (PROJECT_ROOT / raw_path).resolve()

        # Check containment under PROJECT_ROOT
        if not is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
            errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Path escapes repository root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "resolved": str(abs_path)}
            })
            continue

        # Check forbidden overlap
        forbidden_hit = False
        for forbidden in FORBIDDEN_ROOTS:
            forbidden_abs = (PROJECT_ROOT / forbidden).resolve()
            if is_path_under_root(abs_path, forbidden_abs) or is_path_under_root(forbidden_abs, abs_path):
                errors.append({
                    "code": "FORBIDDEN_PATH_OVERLAP",
                    "message": f"Output overlaps forbidden root '{forbidden}': {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "forbidden_root": forbidden}
                })
                forbidden_hit = True
                break

        if forbidden_hit:
            continue

        # Check under DURABLE_ROOTS
        under_durable = False
        for root in DURABLE_ROOTS:
            root_abs = (PROJECT_ROOT / root).resolve()
            if is_path_under_root(abs_path, root_abs):
                under_durable = True
                break

        if not under_durable:
            errors.append({
                "code": "OUTPUT_OUTSIDE_DURABLE_ROOT",
                "message": f"Output not under any durable root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "durable_roots": DURABLE_ROOTS}
            })
            continue

        # Check existence
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
# SPECTRUM-02 BUNDLE VERIFICATION
# =============================================================================

def generate_output_hashes(run_id: str) -> Dict:
    """Generate OUTPUT_HASHES.json for SPECTRUM-02 bundle.

    Hashes every declared durable output in TASK_SPEC.json.

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

        # Skip invalid paths
        if Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
            continue

        abs_path = (PROJECT_ROOT / raw_path).resolve()

        # Skip if path escapes repo root
        if not is_path_under_root(abs_path, PROJECT_ROOT.resolve()):
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
            file_hash = compute_hash(abs_path)
            rel_posix = abs_path.relative_to(PROJECT_ROOT).as_posix()
            hashes[rel_posix] = f"sha256:{file_hash}"
        elif abs_path.is_dir():
            for file_path in abs_path.rglob("*"):
                if file_path.is_file():
                    file_hash = compute_hash(file_path)
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

    Returns: {"valid": bool, "errors": [...]}
    """
    errors = []

    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    # 1. Check TASK_SPEC.json
    task_spec_path = run_dir / "TASK_SPEC.json"
    if not task_spec_path.exists():
        errors.append({
            "code": "BUNDLE_INCOMPLETE",
            "message": "TASK_SPEC.json missing",
            "path": "/",
            "details": {"expected": str(task_spec_path)}
        })
        return {"valid": False, "errors": errors}

    # 2. Check STATUS.json
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

    # 3. Check OUTPUT_HASHES.json
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

    # 4. Check validator semver
    validator_semver = output_hashes.get("validator_semver")
    if validator_semver not in SUPPORTED_VALIDATOR_SEMVERS:
        errors.append({
            "code": "VALIDATOR_UNSUPPORTED",
            "message": f"validator_semver '{validator_semver}' not supported",
            "path": "/validator_semver",
            "details": {"actual": validator_semver, "supported": list(SUPPORTED_VALIDATOR_SEMVERS)}
        })
        return {"valid": False, "errors": errors}

    # 5. Check validator_build_id
    validator_build_id = output_hashes.get("validator_build_id")
    if not validator_build_id:
        errors.append({
            "code": "VALIDATOR_BUILD_ID_MISSING",
            "message": "validator_build_id is missing or empty",
            "path": "/validator_build_id",
            "details": {"actual": validator_build_id}
        })
        return {"valid": False, "errors": errors}

    # 6. Strict build ID check
    if strict_build_id:
        current_build_id = get_validator_build_id()
        if validator_build_id != current_build_id:
            errors.append({
                "code": "VALIDATOR_BUILD_MISMATCH",
                "message": f"validator_build_id mismatch",
                "path": "/validator_build_id",
                "details": {"expected": current_build_id, "actual": validator_build_id}
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

        actual_hash = f"sha256:{compute_hash(abs_path)}"
        if actual_hash != expected_hash:
            errors.append({
                "code": "HASH_MISMATCH",
                "message": f"Hash mismatch for {rel_path}",
                "path": f"/hashes/{rel_path}",
                "details": {"expected": expected_hash, "actual": actual_hash}
            })

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# Aliases for backwards compatibility
_is_path_under_root = is_path_under_root
_validate_single_path = validate_single_path
_check_containment_overlap = check_containment_overlap
_validate_jobspec_paths = validate_jobspec_paths
_verify_post_run_outputs = verify_post_run_outputs
_generate_output_hashes = generate_output_hashes
