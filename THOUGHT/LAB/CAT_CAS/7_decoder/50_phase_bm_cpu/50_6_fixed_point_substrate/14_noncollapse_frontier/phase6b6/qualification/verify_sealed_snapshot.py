"""Software-only Git-provenance sealed snapshot verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from .qualification_contract import (
        AUTHORITY_FALSE_FIELDS,
        AUTHORITY_TRUE_FIELDS,
        PHASE6B6_RELATIVE_ROOT,
        QUALIFIED_V2_SOURCE,
        REPO_ROOT,
        SNAPSHOT_SUBJECT_COMMIT,
        V2_RELATIVE_SOURCE,
        build_expected_snapshot_binding,
        digest,
        qualification_contract,
        validate_schema,
    )
except ImportError:  # pragma: no cover
    from qualification_contract import (  # type: ignore
        AUTHORITY_FALSE_FIELDS,
        AUTHORITY_TRUE_FIELDS,
        PHASE6B6_RELATIVE_ROOT,
        QUALIFIED_V2_SOURCE,
        REPO_ROOT,
        SNAPSHOT_SUBJECT_COMMIT,
        V2_RELATIVE_SOURCE,
        build_expected_snapshot_binding,
        digest,
        qualification_contract,
        validate_schema,
    )


class SnapshotVerificationError(ValueError):
    """Raised when a sealed snapshot fails closed."""


PROHIBITED_PATH_PREFIXES = (
    f"{PHASE6B6_RELATIVE_ROOT}/contracts/sessions/",
    f"{PHASE6B6_RELATIVE_ROOT}/evidence/",
    f"{PHASE6B6_RELATIVE_ROOT}/target/",
    f"{PHASE6B6_RELATIVE_ROOT}/acquisition/",
    f"{PHASE6B6_RELATIVE_ROOT}/calibration/",
    f"{PHASE6B6_RELATIVE_ROOT}/restoration/",
    f"{PHASE6B6_RELATIVE_ROOT}/target_coupling/",
    f"{PHASE6B6_RELATIVE_ROOT}/small_wall/",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _check_case_collisions(paths: list[str]) -> None:
    seen: dict[str, str] = {}
    for path in paths:
        key = path.casefold()
        previous = seen.get(key)
        if previous is not None and previous != path:
            raise SnapshotVerificationError(f"case-colliding snapshot paths: {previous} and {path}")
        seen[key] = path


def _git_hash_object(repo_root: Path, path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", "-w", str(path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SnapshotVerificationError((result.stderr or result.stdout or "git hash-object failed").strip())
    return result.stdout.strip()


def _write_tree_from_entries(repo_root: Path, entries: list[dict[str, Any]]) -> str:
    with tempfile.TemporaryDirectory(prefix="phase6b6-observed-index-") as tmp:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(tmp) / "index")
        init = subprocess.run(["git", "read-tree", "--empty"], cwd=str(repo_root), env=env, capture_output=True, check=False)
        if init.returncode != 0:
            raise SnapshotVerificationError("could not initialize observed temporary git index")
        for entry in entries:
            result = subprocess.run(
                [
                    "git",
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"{entry['mode']},{entry['git_object_sha']},{entry['path']}",
                ],
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise SnapshotVerificationError((result.stderr or result.stdout or "git update-index failed").strip())
        result = subprocess.run(["git", "write-tree"], cwd=str(repo_root), env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SnapshotVerificationError((result.stderr or result.stdout or "git write-tree failed").strip())
        return result.stdout.strip()


def _scan_observed_scope(snapshot_dir: Path, trusted: dict[str, Any]) -> list[dict[str, Any]]:
    root = snapshot_dir.resolve()
    if not root.is_dir():
        raise SnapshotVerificationError("snapshot directory does not exist")

    expected_paths = {entry["path"]: entry for entry in trusted["path_mode_blob_inventory"]}
    expected_dirs: set[str] = set()
    for path in expected_paths:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            expected_dirs.add("/".join(parts[:index]))

    observed: list[dict[str, Any]] = []
    all_seen_paths: list[str] = []

    for base, dirs, files in os.walk(root, topdown=True, followlinks=False):
        base_path = Path(base)
        rel_base = "" if base_path == root else _relative(base_path, root)
        kept_dirs = []
        for dirname in dirs:
            rel = f"{rel_base}/{dirname}" if rel_base else dirname
            full = base_path / dirname
            st = full.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise SnapshotVerificationError(f"symlink substitution rejected: {rel}")
            if rel in expected_paths:
                raise SnapshotVerificationError(f"directory replaces tracked file: {rel}")
            if not stat.S_ISDIR(st.st_mode):
                raise SnapshotVerificationError(f"unsupported directory entry type: {rel}")
            if rel in expected_dirs or any(path.startswith(f"{rel}/") for path in expected_paths):
                kept_dirs.append(dirname)
            elif rel.startswith(PHASE6B6_RELATIVE_ROOT):
                all_seen_paths.append(rel)
                kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        for filename in files:
            full = base_path / filename
            rel = f"{rel_base}/{filename}" if rel_base else filename
            st = full.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise SnapshotVerificationError(f"symlink substitution rejected: {rel}")
            if stat.S_ISDIR(st.st_mode):
                raise SnapshotVerificationError(f"directory replaces tracked file: {rel}")
            if not stat.S_ISREG(st.st_mode):
                raise SnapshotVerificationError(f"unsupported snapshot file type: {rel}")
            if getattr(st, "st_nlink", 1) > 1:
                raise SnapshotVerificationError(f"hard-link ambiguity rejected: {rel}")
            all_seen_paths.append(rel)
            if rel not in expected_paths and (
                rel.startswith(f"{PHASE6B6_RELATIVE_ROOT}/") or rel == V2_RELATIVE_SOURCE
            ):
                observed_mode = "100755" if (st.st_mode & stat.S_IXUSR) else "100644"
                observed.append(
                    {
                        "path": rel,
                        "mode": observed_mode,
                        "git_object_type": "blob",
                        "git_object_sha": _git_hash_object(REPO_ROOT, full),
                        "sha256": file_sha256(full),
                        "size": st.st_size,
                    }
                )
            elif rel in expected_paths:
                observed_mode = "100755" if (st.st_mode & stat.S_IXUSR) else "100644"
                observed.append(
                    {
                        "path": rel,
                        "mode": observed_mode,
                        "git_object_type": "blob",
                        "git_object_sha": _git_hash_object(REPO_ROOT, full),
                        "sha256": file_sha256(full),
                        "size": st.st_size,
                    }
                )

    _check_case_collisions(all_seen_paths)
    observed.sort(key=lambda entry: entry["path"])
    return observed


def scan_prohibited_paths(observed_paths: list[str]) -> dict[str, Any]:
    matches = sorted(
        path
        for path in observed_paths
        if any(path.startswith(prefix) for prefix in PROHIBITED_PATH_PREFIXES)
    )
    result = {
        "schema_id": "CAT_CAS_PHASE6B6_PROHIBITED_PATH_SCAN_RESULT_V1",
        "status": "PASS" if not matches else "FAIL",
        "prohibited_prefixes": list(PROHIBITED_PATH_PREFIXES),
        "matches": matches,
        "generated_final_campaign_sessions_present": any(
            path.startswith(f"{PHASE6B6_RELATIVE_ROOT}/contracts/sessions/") for path in matches
        ),
    }
    validate_schema("prohibited_path_scan_result.schema.json", result)
    return result


def derive_authority_from_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    approval_path = snapshot_dir / PHASE6B6_RELATIVE_ROOT / "PHASE6B6_SOFTWARE_ENTRY_APPROVAL.json"
    if not approval_path.is_file():
        raise SnapshotVerificationError("missing Phase 6B.6 approval JSON")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotVerificationError("approval JSON is not valid JSON") from exc
    for field in AUTHORITY_TRUE_FIELDS:
        if approval.get(field) is not True:
            raise SnapshotVerificationError(f"approval authority true flag missing or false: {field}")
    for field in AUTHORITY_FALSE_FIELDS:
        if approval.get(field) is not False:
            raise SnapshotVerificationError(f"approval authority false flag is true: {field}")
    result = {
        "schema_id": "CAT_CAS_PHASE6B6_DERIVED_AUTHORITY_RESULT_V1",
        "source_path": f"{PHASE6B6_RELATIVE_ROOT}/PHASE6B6_SOFTWARE_ENTRY_APPROVAL.json",
        "source_sha256": file_sha256(approval_path),
        "authority": {field: approval[field] for field in (*AUTHORITY_TRUE_FIELDS, *AUTHORITY_FALSE_FIELDS)},
    }
    validate_schema("derived_authority_result.schema.json", result)
    return result


def _package_identity(snapshot_dir: Path, observed_entries: list[dict[str, Any]], phase6b6_digest: str) -> dict[str, Any]:
    by_path = {entry["path"]: entry for entry in observed_entries}

    def source_sha(relative: str) -> str:
        path = f"{PHASE6B6_RELATIVE_ROOT}/{relative}"
        if path not in by_path:
            raise SnapshotVerificationError(f"missing Phase 6B.6 package file: {path}")
        return by_path[path]["sha256"]

    def inventory_for(prefix: str) -> str:
        return digest([entry for entry in observed_entries if entry["path"].startswith(f"{PHASE6B6_RELATIVE_ROOT}/{prefix}")])

    return {
        "root_path": PHASE6B6_RELATIVE_ROOT,
        "tracked_file_count": len([entry for entry in observed_entries if entry["path"].startswith(f"{PHASE6B6_RELATIVE_ROOT}/")]),
        "subtree_inventory_digest": phase6b6_digest,
        "contract_source_sha256": source_sha("contracts/contract.py"),
        "schedule_source_sha256": source_sha("contracts/schedule.py"),
        "runtime_source_inventory_digest": inventory_for("runtime/"),
        "analysis_source_inventory_digest": inventory_for("analysis/"),
        "schema_inventory_digest": inventory_for("schemas/"),
    }


def observed_snapshot_identity(snapshot_dir: Path, trusted: dict[str, Any]) -> dict[str, Any]:
    observed_entries = _scan_observed_scope(snapshot_dir, trusted)
    observed_paths = [entry["path"] for entry in observed_entries]
    prohibited = scan_prohibited_paths(observed_paths)
    if prohibited["status"] != "PASS":
        raise SnapshotVerificationError(f"prohibited generated snapshot content: {prohibited['matches']}")

    phase6b6_entries = [entry for entry in observed_entries if entry["path"].startswith(f"{PHASE6B6_RELATIVE_ROOT}/")]
    phase6b6_digest = digest(phase6b6_entries)
    v2_entry = next((entry for entry in observed_entries if entry["path"] == V2_RELATIVE_SOURCE), None)
    if v2_entry is None:
        raise SnapshotVerificationError("missing observed V2 source")
    if v2_entry["sha256"] != QUALIFIED_V2_SOURCE["physical_interface_source_sha256"]:
        raise SnapshotVerificationError("observed V2 source SHA-256 mismatch")

    for relative in ("contracts/v2_interface.py", "contracts/contract.py", "contracts/schedule.py"):
        if f"{PHASE6B6_RELATIVE_ROOT}/{relative}" not in observed_paths:
            raise SnapshotVerificationError(f"missing observed Phase 6B.6 source: {relative}")

    authority = derive_authority_from_snapshot(snapshot_dir)
    identity = {
        "schema_id": "CAT_CAS_PHASE6B6_OBSERVED_SNAPSHOT_IDENTITY_V1",
        "calculated_path_mode_blob_inventory": observed_entries,
        "calculated_inventory_sha256": digest(observed_entries),
        "calculated_tree": _write_tree_from_entries(REPO_ROOT, observed_entries),
        "calculated_phase6b6_subtree_inventory_sha256": phase6b6_digest,
        "calculated_v2_source_sha256": v2_entry["sha256"],
        "phase6b6_package_identity": _package_identity(snapshot_dir, observed_entries, phase6b6_digest),
        "derived_authority": authority,
        "prohibited_path_scan": prohibited,
    }
    validate_schema("observed_snapshot_identity.schema.json", identity)
    return identity


def verify_snapshot_directory(snapshot_dir: Path) -> dict[str, Any]:
    trusted = build_expected_snapshot_binding(REPO_ROOT, SNAPSHOT_SUBJECT_COMMIT)
    observed = observed_snapshot_identity(snapshot_dir, trusted)
    expected_inventory = trusted["path_mode_blob_inventory"]
    observed_inventory = observed["calculated_path_mode_blob_inventory"]
    if observed_inventory != expected_inventory:
        expected_paths = {entry["path"]: entry for entry in expected_inventory}
        observed_paths = {entry["path"]: entry for entry in observed_inventory}
        missing = sorted(set(expected_paths) - set(observed_paths))
        extra = sorted(set(observed_paths) - set(expected_paths))
        changed = sorted(
            path
            for path in set(expected_paths) & set(observed_paths)
            if expected_paths[path] != observed_paths[path]
        )
        raise SnapshotVerificationError(f"snapshot inventory mismatch missing={missing} extra={extra} changed={changed}")
    if observed["calculated_tree"] != trusted["expected_scoped_tree"]:
        raise SnapshotVerificationError("observed Git tree does not match trusted scoped tree")
    if observed["calculated_phase6b6_subtree_inventory_sha256"] != trusted["expected_phase6b6_subtree_inventory_sha256"]:
        raise SnapshotVerificationError("Phase 6B.6 subtree inventory digest mismatch")
    contract = qualification_contract()
    result = {
        "schema_id": "CAT_CAS_PHASE6B6_SNAPSHOT_VERIFICATION_RESULT_V1",
        "status": "PHASE6B6_SEALED_SNAPSHOT_VERIFICATION_PASS",
        "trusted_snapshot_binding": trusted,
        "observed_snapshot_identity": observed,
        "qualification_contract_digest": contract["qualification_contract_sha256"],
        "checked": [
            "frozen_snapshot_subject_commit",
            "trusted_git_inventory",
            "observed_path_mode_blob_inventory",
            "observed_scoped_tree",
            "phase6b6_subtree_inventory",
            "observed_v2_source_sha256",
            "derived_authority",
            "prohibited_path_scan",
        ],
    }
    result["snapshot_verification_sha256"] = digest(result)
    validate_schema("snapshot_verification_result.schema.json", result)
    return result


def materialize_trusted_snapshot(binding: dict[str, Any], destination: Path, repo_root: Path = REPO_ROOT) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise SnapshotVerificationError("snapshot destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    for entry in binding["path_mode_blob_inventory"]:
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "cat-file", "blob", entry["git_object_sha"]],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise SnapshotVerificationError(f"could not materialize blob: {entry['path']}")
        target.write_bytes(result.stdout)
        if entry["mode"] == "100755":
            target.chmod(0o755)
        else:
            target.chmod(0o644)
