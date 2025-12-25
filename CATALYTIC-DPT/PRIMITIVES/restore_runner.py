#!/usr/bin/env python3
"""
Restore Runner - SPECTRUM-06 Enforcement

Implements restore semantics exactly per:
- CATALYTIC-DPT/SPECTRUM/SPECTRUM-06.md (LAW)
  - Ticket A (core semantics)
  - B1 (success result artifacts)
  - B2 (failure codes + rollback + threat model)

Public API:
  - restore_bundle(run_dir: Path, restore_root: Path, *, strict: bool = True) -> dict
  - restore_chain(run_dirs: list[Path], restore_root: Path, *, strict: bool = True) -> dict

Return shape (stable):
  {
    "ok": bool,
    "code": str,
    "details": dict
  }
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PRIMITIVES.verify_bundle import BundleVerifier


# -----------------------------------------------------------------------------
# Restore-specific error codes (SPECTRUM-06 Section 8.3)
# -----------------------------------------------------------------------------

RESTORE_CODES = {
    "OK": "OK",
    "RESTORE_VERIFY_STRICT_FAILED": "RESTORE_VERIFY_STRICT_FAILED",
    "RESTORE_PROOF_MISSING": "RESTORE_PROOF_MISSING",
    "RESTORE_PROOF_MALFORMED": "RESTORE_PROOF_MALFORMED",
    "RESTORE_PROOF_RESTORATION_RESULT_MISSING": "RESTORE_PROOF_RESTORATION_RESULT_MISSING",
    "RESTORE_PROOF_NOT_VERIFIED": "RESTORE_PROOF_NOT_VERIFIED",
    "RESTORE_OUTPUT_HASHES_MISSING": "RESTORE_OUTPUT_HASHES_MISSING",
    "RESTORE_OUTPUT_HASHES_MALFORMED": "RESTORE_OUTPUT_HASHES_MALFORMED",
    "RESTORE_OUTPUT_HASHES_HASHES_MISSING": "RESTORE_OUTPUT_HASHES_HASHES_MISSING",
    "RESTORE_OUTPUT_HASHES_HASHES_EMPTY": "RESTORE_OUTPUT_HASHES_HASHES_EMPTY",
    "RESTORE_TARGET_MISSING": "RESTORE_TARGET_MISSING",
    "RESTORE_TARGET_NOT_ABSOLUTE": "RESTORE_TARGET_NOT_ABSOLUTE",
    "RESTORE_TARGET_NOT_EXIST": "RESTORE_TARGET_NOT_EXIST",
    "RESTORE_TARGET_NOT_DIRECTORY": "RESTORE_TARGET_NOT_DIRECTORY",
    "RESTORE_TARGET_NOT_WRITABLE": "RESTORE_TARGET_NOT_WRITABLE",
    "RESTORE_PATH_TRAVERSAL_DETECTED": "RESTORE_PATH_TRAVERSAL_DETECTED",
    "RESTORE_PATH_NULL_BYTE_DETECTED": "RESTORE_PATH_NULL_BYTE_DETECTED",
    "RESTORE_SYMLINK_ESCAPE_DETECTED": "RESTORE_SYMLINK_ESCAPE_DETECTED",
    "RESTORE_SOURCE_MISSING": "RESTORE_SOURCE_MISSING",
    "RESTORE_SOURCE_NOT_REGULAR_FILE": "RESTORE_SOURCE_NOT_REGULAR_FILE",
    "RESTORE_TARGET_PATH_EXISTS": "RESTORE_TARGET_PATH_EXISTS",
    "RESTORE_CHAIN_RUN_ID_DUPLICATE": "RESTORE_CHAIN_RUN_ID_DUPLICATE",
    "RESTORE_CHAIN_TARGET_DIR_EXISTS": "RESTORE_CHAIN_TARGET_DIR_EXISTS",
    "RESTORE_STAGING_HASH_MISMATCH": "RESTORE_STAGING_HASH_MISMATCH",
    "RESTORE_FINALIZE_FAILED": "RESTORE_FINALIZE_FAILED",
    "RESTORE_OUTPUT_MISSING_AFTER_RESTORE": "RESTORE_OUTPUT_MISSING_AFTER_RESTORE",
    "RESTORE_HASH_MISMATCH_AFTER_RESTORE": "RESTORE_HASH_MISMATCH_AFTER_RESTORE",
    "RESTORE_RESULT_ARTIFACT_WRITE_FAILED": "RESTORE_RESULT_ARTIFACT_WRITE_FAILED",
    "RESTORE_ROLLBACK_FAILED": "RESTORE_ROLLBACK_FAILED",
    "RESTORE_INTERNAL_ERROR": "RESTORE_INTERNAL_ERROR",
}


PHASE_PREFLIGHT = "PREFLIGHT"
PHASE_PLAN = "PLAN"
PHASE_EXECUTE = "EXECUTE"
PHASE_VERIFY = "VERIFY"


@dataclass(frozen=True)
class RestorePlanEntry:
    relative_path: str
    source_path: Path
    target_path: Path
    expected_hash: str


def _canonical_json_bytes(obj: Any) -> bytes:
    # SPECTRUM-06 Section 7.3 canonical JSON bytes (same rules as SPECTRUM-04 canonicalization).
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_hex(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def _sorted_paths(paths: Iterable[str]) -> List[str]:
    return sorted(paths, key=lambda s: s.encode("utf-8"))


def _normalize_relative_path(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize per SPECTRUM-06 Section 4.3.

    Returns:
        (normalized_path, error_code_if_any)
    """
    if "\x00" in raw:
        return None, RESTORE_CODES["RESTORE_PATH_NULL_BYTE_DETECTED"]

    # Step 1: backslashes -> slashes
    path = raw.replace("\\", "/")
    # Step 2: collapse multiple slashes
    while "//" in path:
        path = path.replace("//", "/")
    # Step 3: remove leading slash
    while path.startswith("/"):
        path = path[1:]

    # Step 4: reject empty
    if path == "":
        return None, RESTORE_CODES["RESTORE_PATH_TRAVERSAL_DETECTED"]

    parts = path.split("/")
    # Step 5: reject paths containing only '.' or '..' components, and reject any '.' or '..' component
    for part in parts:
        if part in (".", "..") or part == "":
            return None, RESTORE_CODES["RESTORE_PATH_TRAVERSAL_DETECTED"]

    # If normalization changes the raw key, treat as ambiguity and reject.
    if path != raw:
        return None, RESTORE_CODES["RESTORE_PATH_TRAVERSAL_DETECTED"]

    return path, None


def _is_lexically_under(root: Path, candidate: Path) -> bool:
    root_parts = root.parts
    cand_parts = candidate.parts
    return len(cand_parts) >= len(root_parts) and cand_parts[: len(root_parts)] == root_parts


def _symlink_escapes_root(root: Path, target: Path) -> bool:
    """
    Detect if any existing symlink in the path prefix would cause escape outside root.
    """
    root_real = root.resolve()

    current = root_real
    rel_parts = target.relative_to(root).parts
    for part in rel_parts:
        current = current / part
        if current.exists() and current.is_symlink():
            resolved = current.resolve()
            if not _is_lexically_under(root_real, resolved):
                return True
    return False


def _validate_restore_root(restore_root: Path) -> Optional[str]:
    if restore_root is None:
        return RESTORE_CODES["RESTORE_TARGET_MISSING"]
    if not Path(restore_root).is_absolute():
        return RESTORE_CODES["RESTORE_TARGET_NOT_ABSOLUTE"]
    if not restore_root.exists():
        return RESTORE_CODES["RESTORE_TARGET_NOT_EXIST"]
    if not restore_root.is_dir():
        return RESTORE_CODES["RESTORE_TARGET_NOT_DIRECTORY"]
    if not os.access(str(restore_root), os.W_OK):
        return RESTORE_CODES["RESTORE_TARGET_NOT_WRITABLE"]
    return None


def _load_json_required(path: Path, missing_code: str, malformed_code: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, missing_code
    try:
        data = path.read_bytes()
        return json.loads(data.decode("utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, malformed_code


def _extract_output_hashes(output_hashes_obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    hashes = output_hashes_obj.get("hashes")
    if not isinstance(hashes, dict):
        return None, RESTORE_CODES["RESTORE_OUTPUT_HASHES_HASHES_MISSING"]
    if len(hashes) == 0:
        return None, RESTORE_CODES["RESTORE_OUTPUT_HASHES_HASHES_EMPTY"]
    for k, v in hashes.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return None, RESTORE_CODES["RESTORE_OUTPUT_HASHES_HASHES_MISSING"]
    return hashes, None


def _check_proof_verified(proof_obj: Dict[str, Any]) -> Optional[str]:
    rr = proof_obj.get("restoration_result")
    if not isinstance(rr, dict):
        return RESTORE_CODES["RESTORE_PROOF_RESTORATION_RESULT_MISSING"]
    if rr.get("verified") is not True:
        return RESTORE_CODES["RESTORE_PROOF_NOT_VERIFIED"]
    return None


def _result(code: str, phase: str, *, ok: bool, details: Optional[Dict[str, Any]] = None, cause_code: Optional[str] = None) -> Dict[str, Any]:
    details_out: Dict[str, Any] = {} if details is None else dict(details)
    details_out["phase"] = phase
    if cause_code is not None:
        details_out["cause_code"] = cause_code
    return {"ok": ok, "code": code, "details": details_out}


def _cleanup_success_artifacts(result_root: Path) -> None:
    for name in ("RESTORE_MANIFEST.json", "RESTORE_REPORT.json"):
        p = result_root / name
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _rollback_bundle(result_root: Path, entries: List[RestorePlanEntry], staging_dir: Optional[Path]) -> bool:
    ok = True

    if staging_dir is not None:
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=False)
        except Exception:
            ok = False

    for entry in entries:
        try:
            if entry.target_path.exists():
                entry.target_path.unlink()
        except Exception:
            ok = False

    # Remove now-empty directories created under result_root (best effort).
    try:
        for entry in sorted(entries, key=lambda e: len(e.target_path.parts), reverse=True):
            parent = entry.target_path.parent
            while parent != result_root and _is_lexically_under(result_root, parent):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
    except Exception:
        ok = False

    _cleanup_success_artifacts(result_root)
    return ok


def _write_restore_artifacts(
    *,
    result_root: Path,
    entries: List[RestorePlanEntry],
    bundle_root: str,
    chain_root: Optional[str],
) -> Tuple[bool, Optional[str]]:
    manifest_entries: List[Dict[str, Any]] = []
    restored_bytes = 0

    for entry in entries:
        try:
            size = entry.target_path.stat().st_size
        except FileNotFoundError:
            return False, RESTORE_CODES["RESTORE_RESULT_ARTIFACT_WRITE_FAILED"]
        restored_bytes += size
        manifest_entries.append(
            {"bytes": int(size), "relative_path": entry.relative_path, "sha256": entry.expected_hash}
        )

    manifest = {"entries": manifest_entries}
    report = {
        "bundle_roots": [bundle_root],
        "chain_root": chain_root,
        "ok": True,
        "restored_bytes": int(restored_bytes),
        "restored_files_count": int(len(entries)),
    }

    try:
        (result_root / "RESTORE_MANIFEST.json").write_bytes(_canonical_json_bytes(manifest))
        (result_root / "RESTORE_REPORT.json").write_bytes(_canonical_json_bytes(report))
    except Exception:
        return False, RESTORE_CODES["RESTORE_RESULT_ARTIFACT_WRITE_FAILED"]

    return True, None


def _build_restore_plan(
    *,
    project_root: Path,
    result_root: Path,
    hashes: Dict[str, str],
) -> Tuple[Optional[List[RestorePlanEntry]], Optional[str]]:
    entries: List[RestorePlanEntry] = []

    for relative_path in _sorted_paths(hashes.keys()):
        expected_hash = hashes[relative_path]
        if not (isinstance(expected_hash, str) and expected_hash.startswith("sha256:") and len(expected_hash) == 71):
            return None, RESTORE_CODES["RESTORE_OUTPUT_HASHES_HASHES_MISSING"]

        source_path = project_root / relative_path
        target_path = result_root / relative_path
        entries.append(
            RestorePlanEntry(
                relative_path=relative_path,
                source_path=source_path,
                target_path=target_path,
                expected_hash=expected_hash,
            )
        )

    # Validate sources (deterministic order already).
    for entry in entries:
        if not entry.source_path.exists():
            return None, RESTORE_CODES["RESTORE_SOURCE_MISSING"]
        if not entry.source_path.is_file():
            return None, RESTORE_CODES["RESTORE_SOURCE_NOT_REGULAR_FILE"]

    return entries, None


def _validate_path_safety_from_hashes(result_root: Path, hashes: Dict[str, str]) -> Optional[str]:
    for relative_path in _sorted_paths(hashes.keys()):
        normalized, err = _normalize_relative_path(relative_path)
        if err is not None:
            return err
        assert normalized == relative_path

        target_abs = result_root / relative_path
        if _symlink_escapes_root(result_root, target_abs):
            return RESTORE_CODES["RESTORE_SYMLINK_ESCAPE_DETECTED"]

    return None


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def restore_bundle(run_dir: Path, restore_root: Path, *, strict: bool = True) -> Dict[str, Any]:
    """
    Restore outputs for a single run bundle into restore_root (single mode).
    """
    return _restore_bundle_impl(run_dir=run_dir, restore_root=restore_root, strict=strict, chain_root=None)


def _restore_bundle_impl(*, run_dir: Path, restore_root: Path, strict: bool, chain_root: Optional[str]) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    verifier = BundleVerifier(project_root=project_root)

    run_dir = Path(run_dir)
    restore_root = Path(restore_root)
    phase = PHASE_PREFLIGHT

    restore_root_err = _validate_restore_root(restore_root)
    if restore_root_err is not None:
        return _result(restore_root_err, phase, ok=False)

    # Eligibility checks (SPECTRUM-06 8.4.3 order).
    verify_result = verifier.verify_bundle_spectrum05(run_dir=run_dir, strict=strict, check_proof=False)
    if not verify_result.get("ok", False):
        return _result(
            RESTORE_CODES["RESTORE_VERIFY_STRICT_FAILED"],
            phase,
            ok=False,
            details={"verifier_code": verify_result.get("code"), "verifier_details": verify_result.get("details", {})},
        )

    proof_obj, proof_err = _load_json_required(
        run_dir / "PROOF.json",
        RESTORE_CODES["RESTORE_PROOF_MISSING"],
        RESTORE_CODES["RESTORE_PROOF_MALFORMED"],
    )
    if proof_err is not None:
        return _result(proof_err, phase, ok=False)
    assert proof_obj is not None
    proof_verified_err = _check_proof_verified(proof_obj)
    if proof_verified_err is not None:
        return _result(proof_verified_err, phase, ok=False)

    output_hashes_obj, oh_err = _load_json_required(
        run_dir / "OUTPUT_HASHES.json",
        RESTORE_CODES["RESTORE_OUTPUT_HASHES_MISSING"],
        RESTORE_CODES["RESTORE_OUTPUT_HASHES_MALFORMED"],
    )
    if oh_err is not None:
        return _result(oh_err, phase, ok=False)
    assert output_hashes_obj is not None
    hashes, hashes_err = _extract_output_hashes(output_hashes_obj)
    if hashes_err is not None:
        return _result(hashes_err, phase, ok=False)
    assert hashes is not None

    # Path safety checks (SPECTRUM-06 8.4.5). Evaluated after restore_root validation and eligibility checks.
    path_err = _validate_path_safety_from_hashes(restore_root, hashes)
    if path_err is not None:
        return _result(path_err, PHASE_PREFLIGHT, ok=False)

    # PLAN
    phase = PHASE_PLAN
    plan, plan_err = _build_restore_plan(project_root=project_root, result_root=restore_root, hashes=hashes)
    if plan_err is not None:
        return _result(plan_err, phase, ok=False)
    assert plan is not None

    # EXECUTE
    phase = PHASE_EXECUTE
    for entry in plan:
        if entry.target_path.exists():
            return _result(RESTORE_CODES["RESTORE_TARGET_PATH_EXISTS"], phase, ok=False, details={"path": entry.relative_path})

    staging_dir = restore_root / f".spectrum06_staging_{uuid.uuid4().hex}"
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        return _result(RESTORE_CODES["RESTORE_INTERNAL_ERROR"], phase, ok=False)

    try:
        for entry in plan:
            staged_path = staging_dir / entry.relative_path
            _copy_file(entry.source_path, staged_path)
            staged_hash = "sha256:" + _sha256_file_hex(staged_path)
            if staged_hash != entry.expected_hash:
                rollback_ok = _rollback_bundle(restore_root, plan, staging_dir)
                if not rollback_ok:
                    return _result(
                        RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                        phase,
                        ok=False,
                        cause_code=RESTORE_CODES["RESTORE_STAGING_HASH_MISMATCH"],
                    )
                return _result(RESTORE_CODES["RESTORE_STAGING_HASH_MISMATCH"], phase, ok=False, details={"path": entry.relative_path})

        # Finalize move
        created_targets: List[Path] = []
        try:
            for entry in plan:
                src = staging_dir / entry.relative_path
                dst = entry.target_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(src), str(dst))
                created_targets.append(dst)
        except Exception:
            # Rollback any created targets + staging
            rollback_ok = _rollback_bundle(restore_root, plan, staging_dir)
            if not rollback_ok:
                return _result(
                    RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                    phase,
                    ok=False,
                    cause_code=RESTORE_CODES["RESTORE_FINALIZE_FAILED"],
                )
            return _result(RESTORE_CODES["RESTORE_FINALIZE_FAILED"], phase, ok=False)

        # Remove staging directory after move
        try:
            shutil.rmtree(staging_dir, ignore_errors=False)
        except Exception:
            rollback_ok = _rollback_bundle(restore_root, plan, staging_dir)
            if not rollback_ok:
                return _result(
                    RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                    phase,
                    ok=False,
                    cause_code=RESTORE_CODES["RESTORE_FINALIZE_FAILED"],
                )
            return _result(RESTORE_CODES["RESTORE_FINALIZE_FAILED"], phase, ok=False)

    except Exception:
        rollback_ok = _rollback_bundle(restore_root, plan, staging_dir)
        if not rollback_ok:
            return _result(
                RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                phase,
                ok=False,
                cause_code=RESTORE_CODES["RESTORE_INTERNAL_ERROR"],
            )
        return _result(RESTORE_CODES["RESTORE_INTERNAL_ERROR"], phase, ok=False)

    # VERIFY
    phase = PHASE_VERIFY
    for entry in plan:
        if not entry.target_path.exists():
            rollback_ok = _rollback_bundle(restore_root, plan, None)
            if not rollback_ok:
                return _result(
                    RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                    phase,
                    ok=False,
                    cause_code=RESTORE_CODES["RESTORE_OUTPUT_MISSING_AFTER_RESTORE"],
                )
            return _result(RESTORE_CODES["RESTORE_OUTPUT_MISSING_AFTER_RESTORE"], phase, ok=False, details={"path": entry.relative_path})

        actual_hash = "sha256:" + _sha256_file_hex(entry.target_path)
        if actual_hash != entry.expected_hash:
            rollback_ok = _rollback_bundle(restore_root, plan, None)
            if not rollback_ok:
                return _result(
                    RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                    phase,
                    ok=False,
                    cause_code=RESTORE_CODES["RESTORE_HASH_MISMATCH_AFTER_RESTORE"],
                )
            return _result(RESTORE_CODES["RESTORE_HASH_MISMATCH_AFTER_RESTORE"], phase, ok=False, details={"path": entry.relative_path})

    bundle_root = verify_result.get("bundle_root")
    if not isinstance(bundle_root, str) or len(bundle_root) != 64:
        rollback_ok = _rollback_bundle(restore_root, plan, None)
        if not rollback_ok:
            return _result(
                RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                phase,
                ok=False,
                cause_code=RESTORE_CODES["RESTORE_INTERNAL_ERROR"],
            )
        return _result(RESTORE_CODES["RESTORE_INTERNAL_ERROR"], phase, ok=False)

    artifacts_ok, artifacts_err = _write_restore_artifacts(
        result_root=restore_root,
        entries=plan,
        bundle_root=bundle_root,
        chain_root=chain_root,
    )
    if not artifacts_ok:
        rollback_ok = _rollback_bundle(restore_root, plan, None)
        if not rollback_ok:
            return _result(
                RESTORE_CODES["RESTORE_ROLLBACK_FAILED"],
                phase,
                ok=False,
                cause_code=RESTORE_CODES["RESTORE_RESULT_ARTIFACT_WRITE_FAILED"],
            )
        return _result(artifacts_err or RESTORE_CODES["RESTORE_RESULT_ARTIFACT_WRITE_FAILED"], phase, ok=False)

    return _result(
        RESTORE_CODES["OK"],
        phase,
        ok=True,
        details={
            "restored_files_count": len(plan),
            "bundle_root": bundle_root,
            "restore_root": str(restore_root),
        },
    )


def restore_chain(run_dirs: List[Path], restore_root: Path, *, strict: bool = True) -> Dict[str, Any]:
    """
    Restore outputs for a chain of run bundles into restore_root/<run_id>/ (chain mode).
    """
    project_root = Path(__file__).resolve().parents[2]
    verifier = BundleVerifier(project_root=project_root)

    restore_root = Path(restore_root)
    phase = PHASE_PREFLIGHT

    restore_root_err = _validate_restore_root(restore_root)
    if restore_root_err is not None:
        return _result(restore_root_err, phase, ok=False)

    run_dirs = [Path(d) for d in run_dirs]
    run_ids = [d.name for d in run_dirs]

    if not run_dirs:
        # Mirror strict chain verification failure without introducing a new restore code.
        return _result(
            RESTORE_CODES["RESTORE_VERIFY_STRICT_FAILED"],
            phase,
            ok=False,
            details={"verifier_code": "CHAIN_EMPTY"},
        )

    if len(run_ids) != len(set(run_ids)):
        return _result(RESTORE_CODES["RESTORE_CHAIN_RUN_ID_DUPLICATE"], phase, ok=False, details={"run_ids": run_ids})

    for rid in run_ids:
        if (restore_root / rid).exists():
            return _result(RESTORE_CODES["RESTORE_CHAIN_TARGET_DIR_EXISTS"], phase, ok=False, details={"run_id": rid})

    # Chain verification gating (strict)
    chain_verify = verifier.verify_chain_spectrum05(run_dirs=run_dirs, strict=strict, check_proof=False)
    if not chain_verify.get("ok", False):
        return _result(
            RESTORE_CODES["RESTORE_VERIFY_STRICT_FAILED"],
            phase,
            ok=False,
            details={"verifier_code": chain_verify.get("code"), "verifier_details": chain_verify.get("details", {})},
        )
    chain_root = chain_verify.get("chain_root")
    if not isinstance(chain_root, str) or len(chain_root) != 64:
        return _result(RESTORE_CODES["RESTORE_INTERNAL_ERROR"], phase, ok=False)

    chain_manifest = restore_root / f".spectrum06_chain_{uuid.uuid4().hex}.json"
    try:
        chain_manifest.write_bytes(_canonical_json_bytes({"run_ids": run_ids}))
    except Exception:
        return _result(RESTORE_CODES["RESTORE_INTERNAL_ERROR"], phase, ok=False)

    created_bundle_roots: List[str] = []
    created_dirs: List[Path] = []

    try:
        for run_dir in run_dirs:
            rid = run_dir.name
            bundle_root_dir = restore_root / rid
            bundle_root_dir.mkdir(parents=True, exist_ok=False)
            created_dirs.append(bundle_root_dir)

            # Run bundle restore into per-run result_root.
            bundle_result = _restore_bundle_impl(run_dir=run_dir, restore_root=bundle_root_dir, strict=strict, chain_root=chain_root)
            if not bundle_result["ok"]:
                cause = bundle_result.get("code", RESTORE_CODES["RESTORE_INTERNAL_ERROR"])
                rollback_ok = True
                for d in reversed(created_dirs):
                    try:
                        shutil.rmtree(d, ignore_errors=False)
                    except Exception:
                        rollback_ok = False
                try:
                    if chain_manifest.exists():
                        chain_manifest.unlink()
                except Exception:
                    rollback_ok = False

                if not rollback_ok:
                    return _result(RESTORE_CODES["RESTORE_ROLLBACK_FAILED"], PHASE_EXECUTE, ok=False, cause_code=cause)
                return _result(cause, bundle_result["details"].get("phase", PHASE_EXECUTE), ok=False, details=bundle_result.get("details", {}))

            created_bundle_roots.append(bundle_result["details"].get("bundle_root", ""))

        # Remove chain manifest on success.
        try:
            chain_manifest.unlink()
        except Exception:
            # Treat as internal error (no spec code) - rollback is not required since restore succeeded.
            return _result(RESTORE_CODES["RESTORE_INTERNAL_ERROR"], PHASE_VERIFY, ok=False)

        return _result(
            RESTORE_CODES["OK"],
            PHASE_VERIFY,
            ok=True,
            details={
                "restore_root": str(restore_root),
                "chain_root": chain_root,
                "run_ids": run_ids,
            },
        )
    finally:
        # Ensure chain manifest is never left behind on failure.
        if not any(r for r in created_dirs if (r / "RESTORE_REPORT.json").exists()):
            try:
                if chain_manifest.exists():
                    chain_manifest.unlink()
            except Exception:
                pass
