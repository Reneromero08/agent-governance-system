"""Local exporter for copy-only Phase 6B.6 target qualification packages."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .qualification_contract import (
    PHASE6B6_RELATIVE_ROOT,
    QUALIFICATION_ROOT,
    REPO_ROOT,
    SNAPSHOT_SUBJECT_COMMIT,
    V2_RELATIVE_SOURCE,
    _git,
    _git_text,
    build_expected_snapshot_binding,
    digest,
    qualification_contract,
)


PORTABLE_SCHEMA_ID = "CAT_CAS_PHASE6B6_PORTABLE_TARGET_PACKAGE_MANIFEST_V1"
QUALIFICATION_REVIEWED_HEAD = "5ad5b5f07bd31e368de56ab3c721f20498fb7aa1"
QUALIFICATION_MERGE_HEAD = "3c6a5dd344a58d729ea84d23cc90e9e34d6f8336"
SNAPSHOT_SUBJECT_TREE = "1a927b20cb2d712a7220a823621c8fc83cbc984d"
EXPECTED_SCOPED_TREE = "408ee35257417898a992510b0f260602117a15af"
EXPECTED_INVENTORY_SHA256 = "e47dea4c3467835a425d9d553803da48f672a8799970db4fc9b83e98596f50d8"
EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256 = "24789f0df9afa2d9f6a243a9050ff8f265cf22ffb42ab33bbe2f67521dbf44b5"
EXPECTED_V2_SOURCE_SHA256 = "c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976"
PACKAGE_ROOT_DIR = "phase6b6_portable_target_package"
V2_HEADER_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.h"
)
CAPTURED_FILE_HEADER_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/captured_file.h"
)
CAPTURE_QUALITY_HEADER_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/capture_quality_contract.h"
)


class PortablePackageError(ValueError):
    """Raised when portable package export cannot be sealed."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _write_file(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def _case_collision_check(paths: list[str]) -> None:
    seen: dict[str, str] = {}
    for path in paths:
        key = path.casefold()
        previous = seen.get(key)
        if previous is not None and previous != path:
            raise PortablePackageError(f"case-colliding portable package paths: {previous} and {path}")
        seen[key] = path


def _reject_bad_relative_path(path: str) -> None:
    if path.startswith("/") or "\\" in path or path == "" or path.split("/").count(".."):
        raise PortablePackageError(f"unsafe portable package path: {path}")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise PortablePackageError(f"unsafe portable package path: {path}")


def _copy_git_blob(
    package_root: Path,
    package_path: str,
    git_path: str,
    mode: str,
    git_object_sha: str,
    *,
    role: str,
    source_commit: str,
) -> dict[str, Any]:
    _reject_bad_relative_path(package_path)
    if mode not in ("100644", "100755"):
        raise PortablePackageError(f"unsupported file mode for portable package: {mode} {git_path}")
    data = _git(REPO_ROOT, ["cat-file", "blob", git_object_sha])
    fs_mode = 0o755 if mode == "100755" else 0o644
    _write_file(package_root / package_path, data, fs_mode)
    return {
        "path": package_path,
        "source_path": git_path,
        "mode": mode,
        "size": len(data),
        "sha256": _sha256_bytes(data),
        "git_blob_sha1": git_blob_sha1(data),
        "role": role,
        "source_commit": source_commit,
    }


def _copy_worktree_file(package_root: Path, package_path: str, source_path: Path, *, role: str) -> dict[str, Any]:
    _reject_bad_relative_path(package_path)
    st = source_path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise PortablePackageError(f"portable support file is not regular: {source_path}")
    data = source_path.read_bytes()
    mode = "100755" if (st.st_mode & stat.S_IXUSR) else "100644"
    source_commit = _git_text(REPO_ROOT, ["rev-parse", "HEAD"])
    _write_file(package_root / package_path, data, 0o755 if mode == "100755" else 0o644)
    return {
        "path": package_path,
        "source_path": source_path.relative_to(REPO_ROOT).as_posix(),
        "mode": mode,
        "size": len(data),
        "sha256": _sha256_bytes(data),
        "git_blob_sha1": git_blob_sha1(data),
        "role": role,
        "source_commit": source_commit,
    }


def _generated_file_record(package_root: Path, package_path: str, data: bytes, *, role: str) -> dict[str, Any]:
    _reject_bad_relative_path(package_path)
    _write_file(package_root / package_path, data, 0o644)
    return {
        "path": package_path,
        "source_path": "generated_by_portable_package_exporter",
        "mode": "100644",
        "size": len(data),
        "sha256": _sha256_bytes(data),
        "git_blob_sha1": git_blob_sha1(data),
        "role": role,
        "source_commit": _git_text(REPO_ROOT, ["rev-parse", "HEAD"]),
    }


def _iter_tar_paths(root: Path) -> list[Path]:
    paths = [root]
    paths.extend(sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()))
    return paths


def _assert_regular_tree(root: Path) -> None:
    seen: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        _reject_bad_relative_path(rel)
        seen.append(rel)
        st = path.lstat()
        if path.is_dir():
            continue
        if not stat.S_ISREG(st.st_mode):
            raise PortablePackageError(f"non-regular portable package entry rejected: {rel}")
        if rel.endswith(".bundle") or "/.git/" in f"/{rel}/" or rel == ".git":
            raise PortablePackageError(f"forbidden Git package content rejected: {rel}")
    _case_collision_check(seen)


def _write_deterministic_tar(package_root: Path, out: Path) -> None:
    _assert_regular_tree(package_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in _iter_tar_paths(package_root):
            arcname = path.relative_to(package_root.parent).as_posix()
            info = archive.gettarinfo(str(path), arcname)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_dir():
                info.mode = 0o755
                archive.addfile(info)
            else:
                info.mode = 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)


def build_portable_package_tree(package_root: Path) -> dict[str, Any]:
    binding = build_expected_snapshot_binding(REPO_ROOT, SNAPSHOT_SUBJECT_COMMIT)
    contract = qualification_contract()
    if binding["snapshot_subject_tree"] != SNAPSHOT_SUBJECT_TREE:
        raise PortablePackageError("snapshot subject tree mismatch")
    if binding["expected_scoped_tree"] != EXPECTED_SCOPED_TREE:
        raise PortablePackageError("scoped tree mismatch")
    if binding["expected_inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise PortablePackageError("snapshot inventory digest mismatch")
    if binding["expected_phase6b6_subtree_inventory_sha256"] != EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256:
        raise PortablePackageError("Phase 6B.6 subtree inventory digest mismatch")

    files: list[dict[str, Any]] = []
    binding_bytes = _canonical_json(binding)
    contract_bytes = _canonical_json(contract)
    files.append(_generated_file_record(package_root, "TRUSTED_SNAPSHOT_BINDING.json", binding_bytes, role="trusted_snapshot_binding"))
    files.append(_generated_file_record(package_root, "QUALIFICATION_CONTRACT.json", contract_bytes, role="qualification_contract"))

    for entry in binding["path_mode_blob_inventory"]:
        source_path = entry["path"]
        role = "qualified_v2_source" if source_path == V2_RELATIVE_SOURCE else "phase6b6_scoped_source"
        files.append(
            _copy_git_blob(
                package_root,
                f"snapshot/{source_path}",
                source_path,
                entry["mode"],
                entry["git_object_sha"],
                role=role,
                source_commit=binding["snapshot_subject_commit"],
            )
        )

    header_blob = _git_text(REPO_ROOT, ["rev-parse", f"{binding['snapshot_subject_commit']}:{V2_HEADER_RELATIVE_SOURCE}"])
    files.append(
        _copy_git_blob(
            package_root,
            "combined_pdn_hardware.h",
            V2_HEADER_RELATIVE_SOURCE,
            "100644",
            header_blob,
            role="qualified_v2_support_header",
            source_commit=binding["snapshot_subject_commit"],
        )
    )
    captured_file_blob = _git_text(REPO_ROOT, ["rev-parse", f"{binding['snapshot_subject_commit']}:{CAPTURED_FILE_HEADER_RELATIVE_SOURCE}"])
    files.append(
        _copy_git_blob(
            package_root,
            "captured_file.h",
            CAPTURED_FILE_HEADER_RELATIVE_SOURCE,
            "100644",
            captured_file_blob,
            role="qualified_v2_support_header",
            source_commit=binding["snapshot_subject_commit"],
        )
    )
    capture_quality_blob = _git_text(REPO_ROOT, ["rev-parse", f"{binding['snapshot_subject_commit']}:{CAPTURE_QUALITY_HEADER_RELATIVE_SOURCE}"])
    files.append(
        _copy_git_blob(
            package_root,
            "capture_quality_contract.h",
            CAPTURE_QUALITY_HEADER_RELATIVE_SOURCE,
            "100644",
            capture_quality_blob,
            role="qualified_v2_support_header",
            source_commit=binding["snapshot_subject_commit"],
        )
    )
    files.append(
        _copy_worktree_file(
            package_root,
            "portable_target_qualification.py",
            QUALIFICATION_ROOT / "portable_target_qualification.py",
            role="portable_target_runner",
        )
    )
    files.append(
        _copy_worktree_file(
            package_root,
            "emit_v2_reference_table.c",
            QUALIFICATION_ROOT / "emit_v2_reference_table.c",
            role="c_reference_emitter",
        )
    )

    package_paths = [entry["path"] for entry in files]
    _case_collision_check(package_paths)
    manifest = {
        "schema_id": PORTABLE_SCHEMA_ID,
        "format_version": 1,
        "package_root": PACKAGE_ROOT_DIR,
        "qualification_reviewed_head": QUALIFICATION_REVIEWED_HEAD,
        "qualification_merge": QUALIFICATION_MERGE_HEAD,
        "snapshot_subject_commit": binding["snapshot_subject_commit"],
        "snapshot_subject_tree": binding["snapshot_subject_tree"],
        "expected_scoped_tree": binding["expected_scoped_tree"],
        "expected_inventory_sha256": binding["expected_inventory_sha256"],
        "expected_phase6b6_subtree_inventory_sha256": binding["expected_phase6b6_subtree_inventory_sha256"],
        "qualified_v2_source_sha256": EXPECTED_V2_SOURCE_SHA256,
        "target_executes_git": False,
        "target_requires_jsonschema": False,
        "target_requires_repository_history": False,
        "copied_files": sorted(files, key=lambda entry: entry["path"]),
        "snapshot_file_count": len(binding["path_mode_blob_inventory"]),
        "portable_qualification_scope": [
            "portable package manifest verification",
            "copied-file inventory verification",
            "pure-Python Git blob and scoped-tree reconstruction",
            "qualification contract verification",
            "strict C reference-emitter compile",
            "deterministic C reference emission A and B",
            "byte comparison",
            "C versus Python tone-codeword equivalence",
            "ASan reference-emitter execution",
            "UBSan reference-emitter execution",
            "runtime validate-only",
            "hardware-option rejection",
            "sender-process absence check",
        ],
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    _write_file(package_root / "PORTABLE_PACKAGE_MANIFEST.json", manifest_bytes, 0o644)
    _write_file(package_root / "PORTABLE_PACKAGE_MANIFEST.sha256", f"{manifest_sha}  PORTABLE_PACKAGE_MANIFEST.json\n".encode("ascii"), 0o644)
    return {"manifest": manifest, "manifest_sha256": manifest_sha}


def export_portable_target_package(out: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase6b6-portable-package-") as temp:
        staging = Path(temp) / PACKAGE_ROOT_DIR
        staging.mkdir(parents=True)
        result = build_portable_package_tree(staging)
        _write_deterministic_tar(staging, out)
    archive_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{archive_sha}  {out.name}\n", encoding="ascii")
    return {
        "schema_id": "CAT_CAS_PHASE6B6_PORTABLE_TARGET_PACKAGE_EXPORT_RESULT_V1",
        "status": "PHASE6B6_PORTABLE_TARGET_PACKAGE_EXPORT_OK",
        "archive_path": str(out),
        "archive_sha256": archive_sha,
        "portable_manifest_sha256": result["manifest_sha256"],
        "manifest": result["manifest"],
    }
