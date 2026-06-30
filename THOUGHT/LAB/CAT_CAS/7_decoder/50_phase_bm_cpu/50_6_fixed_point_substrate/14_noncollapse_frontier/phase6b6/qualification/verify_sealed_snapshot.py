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

ENTRY_TYPE_BY_MODE = (
    (stat.S_ISREG, "regular_file"),
    (stat.S_ISDIR, "directory"),
    (stat.S_ISLNK, "symlink"),
    (stat.S_ISFIFO, "fifo"),
    (stat.S_ISSOCK, "socket"),
    (stat.S_ISCHR, "character_device"),
    (stat.S_ISBLK, "block_device"),
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


def _entry_type(mode: int) -> str:
    for predicate, name in ENTRY_TYPE_BY_MODE:
        if predicate(mode):
            return name
    return "other"


def _git_path(repo_root: Path, args: list[str]) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SnapshotVerificationError((result.stderr or result.stdout or "git rev-parse --git-path failed").strip())
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _git_common_dir(repo_root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SnapshotVerificationError((result.stderr or result.stdout or "git rev-parse --git-common-dir failed").strip())
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def trusted_repository_object_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Record trusted repository object/ref/index/worktree state without mutation."""
    common_dir = _git_common_dir(repo_root)
    objects_dir = _git_path(repo_root, ["objects"])
    pack_dir = objects_dir / "pack"
    refs_dir = common_dir / "refs"
    index_path = _git_path(repo_root, ["index"])
    packed_refs = common_dir / "packed-refs"

    loose_objects: list[str] = []
    if objects_dir.exists():
        for path in objects_dir.glob("[0-9a-f][0-9a-f]/*"):
            if path.is_file():
                loose_objects.append(path.relative_to(objects_dir).as_posix())

    pack_files = []
    if pack_dir.exists():
        pack_files = [_file_fingerprint(path) for path in sorted(pack_dir.iterdir()) if path.is_file()]

    ref_files = []
    if refs_dir.exists():
        ref_files = [_file_fingerprint(path) for path in sorted(refs_dir.rglob("*")) if path.is_file()]
    if packed_refs.is_file():
        ref_files.append(_file_fingerprint(packed_refs))

    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise SnapshotVerificationError((status.stderr or status.stdout or "git status failed").strip())

    return {
        "loose_object_paths": sorted(loose_objects),
        "pack_index_files": sorted(pack_files, key=lambda item: item["path"]),
        "refs": sorted(ref_files, key=lambda item: item["path"]),
        "index": _file_fingerprint(index_path) if index_path.is_file() else None,
        "worktree_status": status.stdout,
    }


def assert_trusted_repository_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before != after:
        raise SnapshotVerificationError("trusted repository object/ref/index/worktree state changed")


def _init_observed_git_repo(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "init", "--quiet"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SnapshotVerificationError((result.stderr or result.stdout or "git init failed").strip())


def _git_hash_object(observed_repo: Path, path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", "-w", str(path)],
        cwd=str(observed_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SnapshotVerificationError((result.stderr or result.stdout or "git hash-object failed").strip())
    return result.stdout.strip()


def _write_tree_from_entries(observed_repo: Path, entries: list[dict[str, Any]]) -> str:
    init = subprocess.run(["git", "read-tree", "--empty"], cwd=str(observed_repo), capture_output=True, text=True, check=False)
    if init.returncode != 0:
        raise SnapshotVerificationError((init.stderr or init.stdout or "could not initialize observed git index").strip())
    for entry in entries:
        result = subprocess.run(
            [
                "git",
                "update-index",
                "--add",
                "--cacheinfo",
                f"{entry['mode']},{entry['git_object_sha']},{entry['path']}",
            ],
            cwd=str(observed_repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SnapshotVerificationError((result.stderr or result.stdout or "git update-index failed").strip())
    result = subprocess.run(["git", "write-tree"], cwd=str(observed_repo), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SnapshotVerificationError((result.stderr or result.stdout or "git write-tree failed").strip())
    return result.stdout.strip()


def _expected_ancestor_dirs(expected_paths: set[str]) -> list[str]:
    expected_dirs: set[str] = {"."}
    for path in expected_paths:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            expected_dirs.add("/".join(parts[:index]))
    return sorted(expected_dirs)


def _scan_observed_scope(snapshot_dir: Path, trusted: dict[str, Any], observed_repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = snapshot_dir.resolve()
    if not root.is_dir():
        raise SnapshotVerificationError("snapshot directory does not exist")

    expected_paths = {entry["path"]: entry for entry in trusted["path_mode_blob_inventory"]}
    expected_dirs = set(_expected_ancestor_dirs(set(expected_paths)))

    observed: list[dict[str, Any]] = []
    all_seen_paths: list[str] = []
    unexpected_entries: list[dict[str, str]] = []

    for base, dirs, files in os.walk(root, topdown=True, followlinks=False):
        base_path = Path(base)
        rel_base = "" if base_path == root else _relative(base_path, root)
        recurse_dirs: list[str] = []
        for dirname in dirs:
            rel = f"{rel_base}/{dirname}" if rel_base else dirname
            full = base_path / dirname
            st = full.lstat()
            entry_type = _entry_type(st.st_mode)
            all_seen_paths.append(rel)
            if entry_type == "directory":
                recurse_dirs.append(dirname)
            if rel in expected_paths:
                unexpected_entries.append(
                    {"path": rel, "entry_type": entry_type, "reason": "expected_file_replaced_by_non_file"}
                )
                continue
            if rel not in expected_dirs:
                unexpected_entries.append(
                    {"path": rel, "entry_type": entry_type, "reason": "not_trusted_file_or_ancestor_directory"}
                )
        dirs[:] = recurse_dirs

        for filename in files:
            full = base_path / filename
            rel = f"{rel_base}/{filename}" if rel_base else filename
            st = full.lstat()
            entry_type = _entry_type(st.st_mode)
            all_seen_paths.append(rel)
            if rel not in expected_paths:
                unexpected_entries.append(
                    {"path": rel, "entry_type": entry_type, "reason": "not_trusted_file_or_ancestor_directory"}
                )
                continue
            if entry_type != "regular_file":
                unexpected_entries.append(
                    {"path": rel, "entry_type": entry_type, "reason": "expected_file_replaced_by_non_file"}
                )
                continue
            if getattr(st, "st_nlink", 1) > 1:
                raise SnapshotVerificationError(f"hard-link ambiguity rejected: {rel}")
            observed_mode = "100755" if (st.st_mode & stat.S_IXUSR) else "100644"
            observed.append(
                {
                    "path": rel,
                    "mode": observed_mode,
                    "git_object_type": "blob",
                    "git_object_sha": _git_hash_object(observed_repo, full),
                    "sha256": file_sha256(full),
                    "size": st.st_size,
                }
            )

    _check_case_collisions(all_seen_paths)
    observed.sort(key=lambda entry: entry["path"])
    scan = {
        "schema_id": "CAT_CAS_PHASE6B6_UNEXPECTED_ENTRY_SCAN_V1",
        "expected_file_paths": sorted(expected_paths),
        "expected_ancestor_directories": sorted(expected_dirs),
        "unexpected_entries": sorted(unexpected_entries, key=lambda entry: entry["path"]),
    }
    if scan["unexpected_entries"]:
        raise SnapshotVerificationError(f"unexpected snapshot entries: {scan['unexpected_entries']}")
    return observed, scan


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
    with tempfile.TemporaryDirectory(prefix="phase6b6-observed-git-") as tmp:
        observed_repo = Path(tmp)
        _init_observed_git_repo(observed_repo)
        observed_entries, unexpected_entry_scan = _scan_observed_scope(snapshot_dir, trusted, observed_repo)
        calculated_tree = _write_tree_from_entries(observed_repo, observed_entries)
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
        "calculated_tree": calculated_tree,
        "calculated_phase6b6_subtree_inventory_sha256": phase6b6_digest,
        "calculated_v2_source_sha256": v2_entry["sha256"],
        "phase6b6_package_identity": _package_identity(snapshot_dir, observed_entries, phase6b6_digest),
        "derived_authority": authority,
        "prohibited_path_scan": prohibited,
        "unexpected_entry_scan": unexpected_entry_scan,
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
            "unexpected_entry_scan",
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
