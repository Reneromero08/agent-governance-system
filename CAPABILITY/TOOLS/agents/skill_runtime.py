#!/usr/bin/env python3

"""
Runtime helpers for skills.

Z.1.6: Canonical Skill Execution with CMP-01 Pre-Validation

This module provides the canonical entry point for all skill execution.
Every skill run MUST pass through execute_skill(), which enforces:
- CMP-01 path validation (pre-run)
- Skill manifest integrity
- Canon version compatibility
- Deterministic execution envelope
- Ledger receipts for validation

No skill may execute without passing CMP-01 validation.
"""

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Version = Tuple[int, int, int]


def _parse_version(value: str) -> Version:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", value.strip())
    if not match:
        raise ValueError(f"Invalid version: {value}")
    return tuple(int(part) for part in match.groups())


def _parse_constraints(range_str: str) -> Iterable[Tuple[str, Version]]:
    cleaned = range_str.strip().strip('"').strip("'")
    if not cleaned:
        return []
    constraints = []
    for token in cleaned.split():
        match = re.match(r"^(>=|<=|>|<|==|=)?(\d+\.\d+\.\d+)$", token)
        if not match:
            raise ValueError(f"Invalid range token: {token}")
        op = match.group(1) or "=="
        constraints.append((op, _parse_version(match.group(2))))
    return constraints


def _satisfies(version: Version, constraints: Iterable[Tuple[str, Version]]) -> bool:
    for op, bound in constraints:
        if op == ">=" and not (version >= bound):
            return False
        if op == ">" and not (version > bound):
            return False
        if op == "<=" and not (version <= bound):
            return False
        if op == "<" and not (version < bound):
            return False
        if op in ("==", "=") and not (version == bound):
            return False
    return True


def _read_required_range(skill_dir: Path) -> Optional[str]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    lines = skill_md.read_text(errors="ignore").splitlines()[:50]
    for line in lines:
        match = re.search(r"^\*\*required_canon_version:\*\*\s*(.+)$", line)
        if match:
            return match.group(1).strip()
    return None


def _find_repo_root(start: Path) -> Optional[Path]:
    for candidate in [start] + list(start.parents):
        if (candidate / "LAW" / "CANON" / "VERSIONING.md").exists():
            return candidate
        if (candidate / "CANON" / "VERSIONING.md").exists():
            return candidate
    return None


def _read_canon_version(project_root: Path) -> Optional[str]:
    versioning = project_root / "LAW" / "CANON" / "VERSIONING.md"
    if not versioning.exists():
        versioning = project_root / "CANON" / "VERSIONING.md"
    if not versioning.exists():
        return None
    content = versioning.read_text(errors="ignore")
    match = re.search(r"canon_version:\s*(\d+\.\d+\.\d+)", content)
    return match.group(1) if match else None


def ensure_canon_compat(skill_dir: Path) -> bool:
    project_root = _find_repo_root(skill_dir.resolve())
    if not project_root:
        print("[skill] Could not locate repository root from skill path.")
        return False
    required_range = _read_required_range(skill_dir)
    if not required_range:
        print(f"[skill] Missing required_canon_version in {skill_dir / 'SKILL.md'}")
        return False

    canon_version_str = _read_canon_version(project_root)
    if not canon_version_str:
        print(f"[skill] Missing canon_version in {project_root / 'CANON' / 'VERSIONING.md'}")
        return False

    try:
        constraints = _parse_constraints(required_range)
        canon_version = _parse_version(canon_version_str)
    except ValueError as exc:
        print(f"[skill] Version parsing error: {exc}")
        return False

    if not _satisfies(canon_version, constraints):
        print(
            "[skill] Canon version not supported: "
            f"{canon_version_str} not in {required_range}"
        )
        return False

    return True


def _canonical_json(obj: Any) -> bytes:
    """Produce canonical JSON encoding for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def _compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CMP01ValidationReceipt:
    """Receipt of CMP-01 validation execution.

    This receipt is deterministic and can be verified post-hoc.
    """
    validator_id: str  # "CMP-01-skill-runtime-v1"
    skill_manifest_hash: str  # SHA-256 of skill SKILL.md content
    task_spec_hash: str  # SHA-256 of canonical JobSpec JSON
    verdict: str  # "PASS" or "FAIL"
    timestamp: str  # ISO 8601
    errors: List[Dict[str, Any]]  # Empty if PASS, structured errors if FAIL


@dataclass(frozen=True)
class SkillExecutionResult:
    """Result of skill execution through canonical path."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    output_data: Optional[Dict[str, Any]]
    cmp01_receipt: CMP01ValidationReceipt
    execution_time_ms: int


class SkillExecutionError(Exception):
    """Base exception for skill execution failures."""
    pass


class CMP01ValidationError(SkillExecutionError):
    """CMP-01 validation failed - execution is prohibited."""
    def __init__(self, receipt: CMP01ValidationReceipt):
        self.receipt = receipt
        error_summary = "; ".join(e.get("message", "Unknown error") for e in receipt.errors)
        super().__init__(f"CMP-01 validation FAILED: {error_summary}")


def _validate_cmp01_paths(task_spec: Dict[str, Any], project_root: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate JobSpec paths against CMP-01 rules.

    Enforces:
    - No absolute paths
    - No traversal (..)
    - No escapes from project root
    - No forbidden path overlaps (CANON/, AGENTS.md, BUILD/, .git)
    - catalytic_domains under allowed catalytic roots
    - outputs.durable_paths under allowed durable roots
    - No containment overlap within same list

    Returns:
        Tuple of (valid: bool, errors: List[Dict])
    """
    errors = []

    # Constants per INTEGRITY.md
    DURABLE_ROOTS = [
        "LAW/CONTRACTS/_runs",
        "NAVIGATION/CORTEX/_generated",
        "MEMORY/LLM_PACKER/_packs",
        "CAPABILITY/PRIMITIVES/_scratch",
    ]

    CATALYTIC_ROOTS = [
        "LAW/CONTRACTS/_runs/_tmp",
        "CAPABILITY/PRIMITIVES/_scratch",
    ]

    FORBIDDEN_ROOTS = [
        "LAW/CANON",
        "CANON",
        "AGENTS.md",
        "BUILD",
        ".git",
    ]

    def _is_path_under_root(path: Path, root: Path) -> bool:
        """Check if path is contained under root."""
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
    ) -> List[Dict[str, Any]]:
        """Validate a single path. Returns list of errors (empty if valid)."""
        path_errors = []

        # 1. Reject absolute paths
        if Path(raw_path).is_absolute():
            path_errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Absolute paths are not allowed: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "reason": "absolute_path"}
            })
            return path_errors

        # 2. Reject traversal
        path_parts = Path(raw_path).parts
        if ".." in path_parts:
            path_errors.append({
                "code": "PATH_CONTAINS_TRAVERSAL",
                "message": f"Path contains forbidden traversal segment '..': {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "segments": list(path_parts)}
            })
            return path_errors

        # 3. Resolve and check containment under project root
        abs_path = (project_root / raw_path).resolve()
        if not _is_path_under_root(abs_path, project_root.resolve()):
            path_errors.append({
                "code": "PATH_ESCAPES_REPO_ROOT",
                "message": f"Path escapes repository root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "resolved": str(abs_path)}
            })
            return path_errors

        # 4. Check forbidden overlap
        for forbidden in FORBIDDEN_ROOTS:
            forbidden_abs = (project_root / forbidden).resolve()
            if _is_path_under_root(abs_path, forbidden_abs) or _is_path_under_root(forbidden_abs, abs_path):
                path_errors.append({
                    "code": "FORBIDDEN_PATH_OVERLAP",
                    "message": f"Path overlaps forbidden root '{forbidden}': {raw_path}",
                    "path": json_pointer,
                    "details": {"declared": raw_path, "forbidden_root": forbidden}
                })
                return path_errors

        # 5. Check under allowed roots
        under_allowed = False
        for root in allowed_roots:
            root_abs = (project_root / root).resolve()
            if _is_path_under_root(abs_path, root_abs):
                under_allowed = True
                break

        if not under_allowed:
            path_errors.append({
                "code": root_error_code,
                "message": f"Path not under any allowed root: {raw_path}",
                "path": json_pointer,
                "details": {"declared": raw_path, "allowed_roots": allowed_roots}
            })

        return path_errors

    def _check_containment_overlap(paths: List[str], json_pointer_base: str) -> List[Dict[str, Any]]:
        """Check for containment overlap within same list."""
        overlap_errors = []
        abs_paths = []

        for orig_idx, raw_path in enumerate(paths):
            if not Path(raw_path).is_absolute() and ".." not in Path(raw_path).parts:
                abs_paths.append((orig_idx, raw_path, (project_root / raw_path).resolve()))

        for i, (idx_a, raw_a, abs_a) in enumerate(abs_paths):
            for j, (idx_b, raw_b, abs_b) in enumerate(abs_paths):
                if i >= j:
                    continue

                # Allow exact duplicates
                if abs_a == abs_b:
                    continue

                # Check strict containment
                if _is_path_under_root(abs_a, abs_b) or _is_path_under_root(abs_b, abs_a):
                    smaller_idx = min(idx_a, idx_b)
                    overlap_errors.append({
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

        return overlap_errors

    # Validate catalytic_domains
    catalytic_domains = task_spec.get("catalytic_domains", [])
    for idx, domain in enumerate(catalytic_domains):
        errors.extend(_validate_single_path(
            domain,
            f"/catalytic_domains/{idx}",
            CATALYTIC_ROOTS,
            "CATALYTIC_OUTSIDE_ROOT"
        ))

    if len(catalytic_domains) > 1:
        errors.extend(_check_containment_overlap(catalytic_domains, "/catalytic_domains"))

    # Validate outputs.durable_paths
    outputs = task_spec.get("outputs", {})
    durable_paths = outputs.get("durable_paths", [])
    for idx, dpath in enumerate(durable_paths):
        errors.extend(_validate_single_path(
            dpath,
            f"/outputs/durable_paths/{idx}",
            DURABLE_ROOTS,
            "OUTPUT_OUTSIDE_DURABLE_ROOT"
        ))

    if len(durable_paths) > 1:
        errors.extend(_check_containment_overlap(durable_paths, "/outputs/durable_paths"))

    return (len(errors) == 0, errors)


def _validate_skill_manifest(skill_dir: Path) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Validate skill manifest integrity.

    Returns:
        Tuple of (valid: bool, manifest_hash: str, errors: List[Dict])
    """
    errors = []
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        errors.append({
            "code": "SKILL_MANIFEST_MISSING",
            "message": f"SKILL.md not found in {skill_dir}",
            "path": "/skill_manifest",
            "details": {"skill_dir": str(skill_dir)}
        })
        return (False, "", errors)

    try:
        manifest_content = skill_md.read_bytes()
        manifest_hash = _compute_sha256(manifest_content)
    except Exception as e:
        errors.append({
            "code": "SKILL_MANIFEST_UNREADABLE",
            "message": f"Failed to read SKILL.md: {e}",
            "path": "/skill_manifest",
            "details": {"skill_dir": str(skill_dir), "error": str(e)}
        })
        return (False, "", errors)

    # Validate run.py exists
    run_script = skill_dir / "run.py"
    if not run_script.exists():
        errors.append({
            "code": "SKILL_RUN_SCRIPT_MISSING",
            "message": f"run.py not found in {skill_dir}",
            "path": "/run_script",
            "details": {"skill_dir": str(skill_dir)}
        })
        return (False, manifest_hash, errors)

    return (len(errors) == 0, manifest_hash, errors)


def _execute_cmp01_validation(
    skill_dir: Path,
    task_spec: Dict[str, Any],
    project_root: Path
) -> CMP01ValidationReceipt:
    """
    Execute CMP-01 pre-validation.

    This is the enforcement boundary. If this returns FAIL, execution MUST NOT proceed.

    Returns:
        CMP01ValidationReceipt with deterministic verdict
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    all_errors = []

    # 1. Validate skill manifest
    manifest_valid, manifest_hash, manifest_errors = _validate_skill_manifest(skill_dir)
    all_errors.extend(manifest_errors)

    # 2. Validate canon compatibility
    if manifest_valid and not ensure_canon_compat(skill_dir):
        all_errors.append({
            "code": "CANON_VERSION_INCOMPATIBLE",
            "message": "Skill canon version requirements not satisfied",
            "path": "/canon_version",
            "details": {}
        })

    # 3. Compute task_spec hash
    task_spec_hash = _compute_sha256(_canonical_json(task_spec))

    # 4. Validate JobSpec paths (CMP-01 core)
    paths_valid, path_errors = _validate_cmp01_paths(task_spec, project_root)
    all_errors.extend(path_errors)

    # Determine verdict
    verdict = "PASS" if len(all_errors) == 0 else "FAIL"

    return CMP01ValidationReceipt(
        validator_id="CMP-01-skill-runtime-v1",
        skill_manifest_hash=manifest_hash,
        task_spec_hash=task_spec_hash,
        verdict=verdict,
        timestamp=timestamp,
        errors=all_errors
    )


def execute_skill(
    skill_dir: Path,
    task_spec: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
    timeout_seconds: int = 60
) -> SkillExecutionResult:
    """
    CANONICAL SKILL EXECUTION ENTRY POINT (Z.1.6)

    This is the ONLY permitted path for skill execution.
    All skills MUST pass through this function, which enforces:

    1. CMP-01 pre-validation (MANDATORY, fail-closed)
    2. Skill manifest integrity
    3. Canon version compatibility
    4. Deterministic execution envelope
    5. Ledger receipts

    If CMP-01 validation fails, execution is PROHIBITED and CMP01ValidationError is raised.

    Args:
        skill_dir: Path to skill directory (must contain SKILL.md and run.py)
        task_spec: JobSpec dictionary with catalytic_domains and outputs
        input_data: Optional input data dict to pass to skill
        project_root: Project root (auto-detected if None)
        timeout_seconds: Execution timeout (default 60s)

    Returns:
        SkillExecutionResult with execution data and CMP-01 receipt

    Raises:
        CMP01ValidationError: If CMP-01 validation fails (execution prohibited)
        SkillExecutionError: If execution fails after passing validation
    """
    import tempfile
    import time

    # Auto-detect project root if not provided
    if project_root is None:
        project_root = _find_repo_root(skill_dir.resolve())
        if project_root is None:
            raise SkillExecutionError(f"Could not locate repository root from {skill_dir}")

    # MANDATORY CMP-01 PRE-VALIDATION
    # This is the enforcement boundary - execution cannot proceed if this fails
    start_time = time.time()
    cmp01_receipt = _execute_cmp01_validation(skill_dir, task_spec, project_root)

    # FAIL-CLOSED: If CMP-01 fails, raise exception and HALT
    if cmp01_receipt.verdict == "FAIL":
        raise CMP01ValidationError(cmp01_receipt)

    # CMP-01 PASSED - Execution is now permitted
    run_script = skill_dir / "run.py"
    input_data = input_data or {}

    try:
        # Create temporary input/output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
            json.dump(input_data, f_in, indent=2)
            input_path = f_in.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
            output_path = f_out.name

        # Execute skill
        result = subprocess.run(
            [sys.executable, str(run_script), input_path, output_path],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout_seconds
        )

        # Read output
        output_data = None
        if Path(output_path).exists():
            try:
                with open(output_path, 'r') as f:
                    output_data = json.load(f)
            except (json.JSONDecodeError, Exception):
                pass  # Output may not be JSON

        # Cleanup temp files
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return SkillExecutionResult(
            success=(result.returncode == 0),
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            output_data=output_data,
            cmp01_receipt=cmp01_receipt,
            execution_time_ms=execution_time_ms
        )

    except subprocess.TimeoutExpired as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return SkillExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Skill execution timed out after {timeout_seconds}s",
            output_data=None,
            cmp01_receipt=cmp01_receipt,
            execution_time_ms=execution_time_ms
        )
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        raise SkillExecutionError(f"Skill execution error: {e}")


def write_ledger_receipt(
    receipt: CMP01ValidationReceipt,
    ledger_path: Path
) -> None:
    """
    Write CMP-01 validation receipt to append-only ledger.

    Ledger format: JSONL (one JSON object per line, canonical encoding)

    Args:
        receipt: CMP01ValidationReceipt to log
        ledger_path: Path to ledger file (will be created if doesn't exist)
    """
    ledger_entry = {
        "type": "CMP01_VALIDATION",
        "validator_id": receipt.validator_id,
        "skill_manifest_hash": receipt.skill_manifest_hash,
        "task_spec_hash": receipt.task_spec_hash,
        "verdict": receipt.verdict,
        "timestamp": receipt.timestamp,
        "errors": receipt.errors
    }

    ledger_line = _canonical_json(ledger_entry) + b'\n'

    # Append to ledger (create parent dirs if needed)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, 'ab') as f:
        f.write(ledger_line)


__all__ = ["ensure_canon_compat", "execute_skill", "CMP01ValidationError", "SkillExecutionResult", "write_ledger_receipt"]
