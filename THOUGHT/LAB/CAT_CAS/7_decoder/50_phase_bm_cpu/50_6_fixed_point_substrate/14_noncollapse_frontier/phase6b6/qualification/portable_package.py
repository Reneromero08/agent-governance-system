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
    qualification_contract,
)


PORTABLE_SCHEMA_ID = "CAT_CAS_PHASE6B6_PORTABLE_TARGET_PACKAGE_MANIFEST_V1"
BASE_QUALIFICATION_REVIEWED_HEAD = "5ad5b5f07bd31e368de56ab3c721f20498fb7aa1"
BASE_QUALIFICATION_MERGE = "3c6a5dd344a58d729ea84d23cc90e9e34d6f8336"
SNAPSHOT_SUBJECT_TREE = "1a927b20cb2d712a7220a823621c8fc83cbc984d"
EXPECTED_SCOPED_TREE = "408ee35257417898a992510b0f260602117a15af"
EXPECTED_INVENTORY_SHA256 = "e47dea4c3467835a425d9d553803da48f672a8799970db4fc9b83e98596f50d8"
EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256 = "24789f0df9afa2d9f6a243a9050ff8f265cf22ffb42ab33bbe2f67521dbf44b5"
EXPECTED_V2_SOURCE_SHA256 = "c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976"
PACKAGE_ROOT_DIR = "phase6b6_portable_target_package"
QUALIFICATION_CONTRACT_DIGEST = "986d1eb27e6e715da0ed8765f58566b0608e464b94dbd0d58ab3d130d80fd0d2"
GENERATED_CONTROL_FILES = (
    "PORTABLE_PACKAGE_MANIFEST.json",
    "PORTABLE_PACKAGE_MANIFEST.sha256",
    "TRUSTED_SNAPSHOT_BINDING.json",
    "QUALIFICATION_CONTRACT.json",
)

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

PORTABLE_TARGET_RUNNER_SOURCE = (
    f"{PHASE6B6_RELATIVE_ROOT}/qualification/portable_target_qualification.py"
)
C_REFERENCE_EMITTER_SOURCE = f"{PHASE6B6_RELATIVE_ROOT}/qualification/emit_v2_reference_table.c"

PORTABLE_QUALIFICATION_SCOPE = [
    "portable package manifest verification",
    "copied-file inventory verification",
    "pure-Python Git blob and scoped-tree reconstruction",
    "qualification contract verification",
    "strict C reference-emitter compile",
    "deterministic raw C reference emission A and B",
    "byte comparison",
    "C versus Python tone-codeword equivalence",
    "ASan reference-emitter execution",
    "UBSan reference-emitter execution",
    "runtime validate-only",
    "hardware-option rejection",
    "sender-process absence check",
    "final result validation",
]


class PortablePackageError(ValueError):
    """Raised when portable package export cannot be sealed."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_package_permissions(git_mode: str) -> int:
    if git_mode == "100644":
        return 0o644
    if git_mode == "100755":
        return 0o755
    raise PortablePackageError(f"unsupported Git source mode for portable package: {git_mode}")


def _format_permission(mode: int) -> str:
    return f"{mode:04o}"


def _posix_permissions_supported() -> bool:
    return os.name != "nt"


def _require_exact_permission(path: Path, expected: int, *, rel: str, kind: str) -> None:
    if not _posix_permissions_supported():
        return
    observed = stat.S_IMODE(path.lstat().st_mode)
    if observed != expected:
        raise PortablePackageError(
            f"{kind} permission mismatch: {rel} expected={_format_permission(expected)} "
            f"observed={_format_permission(observed)}"
        )


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


def _require_clean_tracked_worktree() -> None:
    unstaged = _git(REPO_ROOT, ["diff", "--quiet", "HEAD", "--"], input_bytes=None)
    staged = _git(REPO_ROOT, ["diff", "--cached", "--quiet"], input_bytes=None)
    if unstaged or staged:
        raise PortablePackageError("unexpected git diff output while checking clean worktree")


def _source_tree(commit: str) -> str:
    return _git_text(REPO_ROOT, ["rev-parse", f"{commit}^{{tree}}"])


def _ls_tree_blob(commit: str, git_path: str) -> tuple[str, str]:
    raw = _git(REPO_ROOT, ["ls-tree", "-z", "--full-tree", commit, "--", git_path])
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        raise PortablePackageError(f"expected exactly one Git tree record for {git_path}")
    meta, path_bytes = records[0].split(b"\t", 1)
    mode, object_type, object_sha = meta.decode("ascii").split(" ")
    path = path_bytes.decode("utf-8")
    if path != git_path or object_type != "blob" or mode not in ("100644", "100755"):
        raise PortablePackageError(f"unsupported Git object for portable package: {mode} {object_type} {path}")
    rev_parse_blob = _git_text(REPO_ROOT, ["rev-parse", f"{commit}:{git_path}"])
    if rev_parse_blob != object_sha:
        raise PortablePackageError(f"Git blob resolution mismatch for {git_path}")
    return mode, object_sha


def _file_record(
    package_root: Path,
    *,
    package_path: str,
    source_path: str,
    source_commit: str,
    source_tree: str,
    source_blob_sha1: str,
    mode: str,
    role: str,
) -> dict[str, Any]:
    _reject_bad_relative_path(package_path)
    _reject_bad_relative_path(source_path)
    package_permission = canonical_package_permissions(mode)
    data = _git(REPO_ROOT, ["cat-file", "blob", source_blob_sha1])
    if git_blob_sha1(data) != source_blob_sha1:
        raise PortablePackageError(f"exported bytes do not match source Git blob: {source_path}")
    _write_file(package_root / package_path, data, package_permission)
    return {
        "path": package_path,
        "source_path": source_path,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_blob_sha1": source_blob_sha1,
        "content_sha256": _sha256_bytes(data),
        "mode": mode,
        "size": len(data),
        "role": role,
    }


def _copy_git_object_file(
    package_root: Path,
    *,
    package_path: str,
    source_path: str,
    source_commit: str,
    source_tree: str,
    role: str,
) -> dict[str, Any]:
    mode, source_blob_sha1 = _ls_tree_blob(source_commit, source_path)
    return _file_record(
        package_root,
        package_path=package_path,
        source_path=source_path,
        source_commit=source_commit,
        source_tree=source_tree,
        source_blob_sha1=source_blob_sha1,
        mode=mode,
        role=role,
    )


def _copy_snapshot_inventory_file(
    package_root: Path,
    entry: dict[str, Any],
    *,
    source_commit: str,
    source_tree: str,
    role: str,
) -> dict[str, Any]:
    return _file_record(
        package_root,
        package_path=f"snapshot/{entry['path']}",
        source_path=entry["path"],
        source_commit=source_commit,
        source_tree=source_tree,
        source_blob_sha1=entry["git_object_sha"],
        mode=entry["mode"],
        role=role,
    )


def _write_generated_file(package_root: Path, package_path: str, data: bytes) -> None:
    _reject_bad_relative_path(package_path)
    _write_file(package_root / package_path, data, 0o644)


def _iter_tar_paths(root: Path) -> list[Path]:
    paths = [root]
    paths.extend(sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()))
    return paths


def _assert_regular_tree(root: Path) -> None:
    seen: list[str] = []
    for path in _iter_tar_paths(root):
        rel = path.relative_to(root).as_posix() if path != root else ""
        label = rel or PACKAGE_ROOT_DIR
        if rel:
            _reject_bad_relative_path(rel)
            seen.append(rel)
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode):
            _require_exact_permission(path, 0o755, rel=label, kind="directory")
            continue
        if not stat.S_ISREG(st.st_mode):
            raise PortablePackageError(f"non-regular portable package entry rejected: {label}")
        mode = stat.S_IMODE(st.st_mode)
        if _posix_permissions_supported() and mode not in (0o644, 0o755):
            raise PortablePackageError(
                f"regular file permission mismatch: {label} expected=0644-or-0755 observed={_format_permission(mode)}"
            )
        if rel.endswith(".bundle") or "/.git/" in f"/{rel}/" or rel == ".git":
            raise PortablePackageError(f"forbidden Git package content rejected: {rel}")
    _case_collision_check(seen)


def _canonicalize_directory_permissions(root: Path) -> None:
    for path in _iter_tar_paths(root):
        if path.is_dir():
            path.chmod(0o755)


def _assert_manifest_file_permissions(package_root: Path, manifest: dict[str, Any]) -> None:
    for item in manifest["copied_files"]:
        rel = item["path"]
        expected = canonical_package_permissions(item["mode"])
        path = package_root / rel
        _reject_bad_relative_path(rel)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise PortablePackageError(f"manifest-bound regular file required: {rel}")
        _require_exact_permission(path, expected, rel=rel, kind=f"manifest file Git source mode {item['mode']}")


def _expected_package_permissions(manifest: dict[str, Any]) -> dict[str, int]:
    permissions = {item["path"]: canonical_package_permissions(item["mode"]) for item in manifest["copied_files"]}
    for rel in GENERATED_CONTROL_FILES:
        permissions[rel] = 0o644
    return permissions


def _write_deterministic_tar(package_root: Path, out: Path, manifest: dict[str, Any]) -> None:
    _assert_regular_tree(package_root)
    file_permissions = _expected_package_permissions(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in _iter_tar_paths(package_root):
            arcname = path.relative_to(package_root.parent).as_posix()
            rel = path.relative_to(package_root).as_posix() if path != package_root else ""
            info = archive.gettarinfo(str(path), arcname)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            st = path.lstat()
            if stat.S_ISDIR(st.st_mode):
                info.mode = 0o755
                archive.addfile(info)
            elif stat.S_ISREG(st.st_mode):
                if rel not in file_permissions:
                    raise PortablePackageError(f"unexpected portable package file rejected before tar: {rel}")
                expected = file_permissions[rel]
                _require_exact_permission(path, expected, rel=rel, kind="regular file")
                info.mode = expected
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                raise PortablePackageError(f"unsupported portable package archive entry: {arcname}")


def build_portable_package_tree(package_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    portable_export_commit = _git_text(REPO_ROOT, ["rev-parse", "HEAD"])
    portable_export_tree = _git_text(REPO_ROOT, ["rev-parse", "HEAD^{tree}"])
    snapshot_source_tree = _source_tree(SNAPSHOT_SUBJECT_COMMIT)
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
    if contract["qualification_contract_sha256"] != QUALIFICATION_CONTRACT_DIGEST:
        raise PortablePackageError("qualification contract digest mismatch")

    files: list[dict[str, Any]] = []
    binding_bytes = _canonical_json(binding)
    contract_bytes = _canonical_json(contract)
    _write_generated_file(package_root, "TRUSTED_SNAPSHOT_BINDING.json", binding_bytes)
    _write_generated_file(package_root, "QUALIFICATION_CONTRACT.json", contract_bytes)

    for entry in binding["path_mode_blob_inventory"]:
        source_path = entry["path"]
        role = "qualified_v2_source" if source_path == V2_RELATIVE_SOURCE else "phase6b6_scoped_source"
        files.append(
            _copy_snapshot_inventory_file(
                package_root,
                entry,
                source_commit=binding["snapshot_subject_commit"],
                source_tree=snapshot_source_tree,
                role=role,
            )
        )

    for package_path, source_path in (
        ("combined_pdn_hardware.h", V2_HEADER_RELATIVE_SOURCE),
        ("captured_file.h", CAPTURED_FILE_HEADER_RELATIVE_SOURCE),
        ("capture_quality_contract.h", CAPTURE_QUALITY_HEADER_RELATIVE_SOURCE),
    ):
        files.append(
            _copy_git_object_file(
                package_root,
                package_path=package_path,
                source_path=source_path,
                source_commit=binding["snapshot_subject_commit"],
                source_tree=snapshot_source_tree,
                role="qualified_v2_support_header",
            )
        )

    for package_path, source_path, role in (
        ("portable_target_qualification.py", PORTABLE_TARGET_RUNNER_SOURCE, "portable_target_runner"),
        ("emit_v2_reference_table.c", C_REFERENCE_EMITTER_SOURCE, "c_reference_emitter"),
    ):
        files.append(
            _copy_git_object_file(
                package_root,
                package_path=package_path,
                source_path=source_path,
                source_commit=portable_export_commit,
                source_tree=portable_export_tree,
                role=role,
            )
        )

    package_paths = [entry["path"] for entry in files]
    source_paths = [entry["source_path"] for entry in files]
    _case_collision_check(package_paths)
    if len(source_paths) != len(set(source_paths)):
        raise PortablePackageError("duplicate portable source paths")
    copied_files = sorted(files, key=lambda entry: entry["path"])
    portable_support_bindings = [
        entry for entry in copied_files if entry["role"] in ("portable_target_runner", "c_reference_emitter")
    ]
    manifest = {
        "schema_id": PORTABLE_SCHEMA_ID,
        "format_version": 1,
        "package_root": PACKAGE_ROOT_DIR,
        "base_qualification_reviewed_head": BASE_QUALIFICATION_REVIEWED_HEAD,
        "base_qualification_merge": BASE_QUALIFICATION_MERGE,
        "portable_export_commit": portable_export_commit,
        "portable_export_tree": portable_export_tree,
        "portable_support_blob_bindings": portable_support_bindings,
        "snapshot_subject_commit": binding["snapshot_subject_commit"],
        "snapshot_subject_tree": binding["snapshot_subject_tree"],
        "expected_scoped_tree": binding["expected_scoped_tree"],
        "expected_inventory_sha256": binding["expected_inventory_sha256"],
        "expected_phase6b6_subtree_inventory_sha256": binding["expected_phase6b6_subtree_inventory_sha256"],
        "qualified_v2_source_sha256": EXPECTED_V2_SOURCE_SHA256,
        "target_executes_git": False,
        "target_requires_jsonschema": False,
        "target_requires_repository_history": False,
        "copied_files": copied_files,
        "copied_file_count": len(copied_files),
        "snapshot_file_count": len(binding["path_mode_blob_inventory"]),
        "portable_qualification_scope": PORTABLE_QUALIFICATION_SCOPE,
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
        staging.chmod(0o755)
        result = build_portable_package_tree(staging)
        _canonicalize_directory_permissions(staging)
        _assert_manifest_file_permissions(staging, result["manifest"])
        _write_deterministic_tar(staging, out, result["manifest"])
    archive_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{archive_sha}  {out.name}\n", encoding="ascii")
    return {
        "schema_id": "CAT_CAS_PHASE6B6_PORTABLE_TARGET_PACKAGE_EXPORT_RESULT_V1",
        "status": "PHASE6B6_PORTABLE_TARGET_PACKAGE_EXPORT_OK",
        "archive_path": str(out),
        "archive_sha256": archive_sha,
        "portable_manifest_sha256": result["manifest_sha256"],
        "portable_export_commit": result["manifest"]["portable_export_commit"],
        "portable_export_tree": result["manifest"]["portable_export_tree"],
        "manifest": result["manifest"],
    }
