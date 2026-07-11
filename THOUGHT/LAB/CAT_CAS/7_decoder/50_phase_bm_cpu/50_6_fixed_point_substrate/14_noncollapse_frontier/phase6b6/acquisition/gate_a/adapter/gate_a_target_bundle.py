#!/usr/bin/env python3
"""Git-free target-side Gate A bundle validator.

This module is packaged inside the deterministic execution bundle. It validates
an extracted bundle against its manifest without any dependency on a .git
repository, Git objects, Git modes, or Git blob reconstruction. Host-side Git
custody remains exact and is enforced by build_gate_a_execution_bundle.py; this
module establishes target-side byte integrity purely from local files.

Digest layering (non-circular):

  payload manifest core -> execution_bundle_sha256
    core = manifest minus {execution_bundle_sha256, deterministic_archive_sha256}
    execution_bundle_sha256 = sha256(canonical_bytes(core))

  payload-only PAX tar  -> deterministic_archive_sha256
    the tar contains only the manifest-declared payload files, in manifest order;
    deterministic_archive_sha256 = sha256(payload_archive_bytes(...))

  deployment archive     -> transport container only (no stored self-digest)
    the on-disk / extracted bundle contains the payload files PLUS a detached
    manifest envelope (adapter/GATE_A_EXECUTION_BUNDLE_MANIFEST.json). The
    manifest envelope stores deterministic_archive_sha256 for the PAYLOAD tar,
    never for the deployment archive, so there is no digest cycle.

The manifest carries immutable Git-object authority bindings (git_mode,
git_blob_sha1) established by the host. The target cannot reconstruct Git state,
so it preserves those bindings and verifies they are syntactically valid, while
establishing byte integrity through package_path + byte_size + sha256 + payload
archive digest + execution bundle digest.
"""

from __future__ import annotations

import hashlib
import json
import stat
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
MANIFEST_ENVELOPE_PACKAGE_PATH = "adapter/" + MANIFEST_FILENAME

BASE_MAIN = "03985a74d27e654c151cccad28c6221d91f70180"
REVIEWED_PLAN_HEAD = "65d20b4bc65ddd9260a3c90d92612d2da48763a6"
PLAN_REVIEW_ID = 4617290767
SCHEDULE_SHA256 = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
NAMESPACE_SHA256 = "5b3090f642d28492e182630e6349eccd8181704f08129d40d886c8f529dfd50e"
TARGET_IDENTITY_SHA256 = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"

MANIFEST_KEYS = {
    "schema_id",
    "base_main_commit",
    "reviewed_gate_a_plan_head",
    "gate_a_plan_review",
    "schedule_sha256",
    "target_namespace_sha256",
    "target_identity_stdout_sha256",
    "authority_artifact_created",
    "engineering_smoke_executor_implemented",
    "execution_bundle_target_qualified",
    "engineering_smoke_authorized",
    "hardware_ran",
    "files",
    "execution_bundle_sha256",
    "deterministic_archive_sha256",
}

FILE_ENTRY_KEYS = {
    "package_path",
    "source_repository_path",
    "git_mode",
    "git_blob_sha1",
    "byte_size",
    "sha256",
    "role",
}

REQUIRED_ROLES = {
    "host_adapter", "target_runner", "target_worker", "target_execution_gate",
    "physical_runtime", "physical_runtime_gate_a", "physical_runtime_gate_a_header",
    "captured_file_runtime", "capture_quality_contract",
}


class TargetBundleError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetBundleError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in "0123456789abcdef" for character in value)


def payload_archive_bytes(named_blobs: list[tuple[str, bytes]]) -> bytes:
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in named_blobs:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, BytesIO(data))
    return stream.getvalue()


def payload_archive_digest(named_blobs: list[tuple[str, bytes]]) -> str:
    return sha256_bytes(payload_archive_bytes(named_blobs))


def manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    core = dict(manifest)
    core.pop("execution_bundle_sha256", None)
    core.pop("deterministic_archive_sha256", None)
    return core


def recompute_execution_bundle_sha256(manifest: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(manifest_core(manifest)))


def _safe_package_path(package_path: str) -> bool:
    if not isinstance(package_path, str) or not package_path:
        return False
    pure = Path(package_path)
    if pure.is_absolute():
        return False
    parts = pure.parts
    if ".." in parts:
        return False
    if any(part in ("", ".") for part in parts):
        return False
    if package_path.startswith("/") or ":" in package_path or "\\" in package_path:
        return False
    return True


def validate_manifest_shape(manifest: dict[str, Any]) -> None:
    require(isinstance(manifest, dict), "manifest must be an object")
    require(set(manifest) == MANIFEST_KEYS, "manifest top-level key set mismatch")
    require(manifest["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_BUNDLE_MANIFEST_V1", "manifest schema mismatch")
    require(manifest["base_main_commit"] == BASE_MAIN, "manifest base main mismatch")
    require(manifest["reviewed_gate_a_plan_head"] == REVIEWED_PLAN_HEAD, "manifest reviewed plan head mismatch")
    require(manifest["gate_a_plan_review"] == PLAN_REVIEW_ID, "manifest plan review mismatch")
    require(manifest["schedule_sha256"] == SCHEDULE_SHA256, "manifest schedule digest mismatch")
    require(manifest["target_namespace_sha256"] == NAMESPACE_SHA256, "manifest namespace digest mismatch")
    require(manifest["target_identity_stdout_sha256"] == TARGET_IDENTITY_SHA256, "manifest target identity mismatch")
    require(manifest["authority_artifact_created"] is False, "manifest authority-artifact flag must be false")
    require(manifest["engineering_smoke_executor_implemented"] is True, "manifest executor implementation flag missing")
    require(manifest["execution_bundle_target_qualified"] is True, "manifest target-qualified flag missing")
    require(manifest["engineering_smoke_authorized"] is False, "manifest smoke flag must be false")
    require(manifest["hardware_ran"] is False, "manifest hardware-ran flag must be false")

    execution_digest = manifest["execution_bundle_sha256"]
    archive_digest = manifest["deterministic_archive_sha256"]
    require(is_hex(execution_digest, 64), "execution bundle digest malformed")
    require(is_hex(archive_digest, 64), "deterministic archive digest malformed")

    files = manifest["files"]
    require(isinstance(files, list) and files, "manifest files must be a non-empty list")
    seen_package: set[str] = set()
    seen_lower: set[str] = set()
    roles: set[str] = set()
    for entry in files:
        require(isinstance(entry, dict), "manifest file entry must be an object")
        require(set(entry) == FILE_ENTRY_KEYS, f"manifest file entry key set mismatch: {entry}")
        package_path = entry["package_path"]
        require(_safe_package_path(package_path), f"unsafe package path: {package_path}")
        require(package_path not in seen_package, f"duplicate package path: {package_path}")
        require(package_path.lower() not in seen_lower, f"case-colliding package path: {package_path}")
        seen_package.add(package_path)
        seen_lower.add(package_path.lower())
        source_path = entry["source_repository_path"]
        require(isinstance(source_path, str), f"source path must be a string: {package_path}")
        require(_safe_package_path(source_path), f"unsafe source path: {source_path}")
        require(entry["git_mode"] == "100644", f"unexpected Git mode for {package_path}: {entry['git_mode']}")
        require(is_hex(entry["git_blob_sha1"], 40), f"git blob sha1 malformed: {package_path}")
        require(is_hex(entry["sha256"], 64), f"file sha256 malformed: {package_path}")
        require(isinstance(entry["byte_size"], int) and entry["byte_size"] >= 0, f"byte size malformed: {package_path}")
        require(isinstance(entry["role"], str) and entry["role"], f"role malformed: {package_path}")
        roles.add(entry["role"])
    require(REQUIRED_ROLES <= roles, f"manifest missing required roles: {REQUIRED_ROLES - roles}")

    require(recompute_execution_bundle_sha256(manifest) == execution_digest, "execution bundle digest mismatch")


def entries_by_role(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        role = entry["role"]
        require(role not in roles, f"duplicate manifest role: {role}")
        roles[role] = entry
    for role in REQUIRED_ROLES:
        require(role in roles, f"manifest role missing: {role}")
    return roles


def load_manifest(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / "adapter" / MANIFEST_FILENAME
    require(manifest_path.is_file() and not manifest_path.is_symlink(), f"manifest envelope missing: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "manifest envelope must be an object")
    return value


def default_permitted_extras() -> set[str]:
    """The only non-payload file allowed inside a pristine extracted bundle.

    The extracted bundle is treated as immutable during strict validation.
    Generated runtime outputs (compiled worker, sanitizer binaries, Python
    bytecode, temporary authority artifacts, receipts) must be written outside
    the bundle root. No wildcard categories are accepted: any extra beyond this
    exact set, or an explicit caller-supplied allowlist of exact relative paths,
    is rejected.
    """
    return {MANIFEST_ENVELOPE_PACKAGE_PATH}


def permitted_extra_set(permitted_runtime_outputs: set[str] | None) -> set[str]:
    permitted = default_permitted_extras()
    if permitted_runtime_outputs:
        for rel in permitted_runtime_outputs:
            require(isinstance(rel, str) and rel and _safe_package_path(rel), f"invalid permitted runtime output: {rel!r}")
            permitted.add(rel)
    return permitted


def validate_payload_files(bundle_root: Path, manifest: dict[str, Any]) -> str:
    named_blobs: list[tuple[str, bytes]] = []
    root_resolved = bundle_root.resolve()
    for entry in manifest["files"]:
        package_path = entry["package_path"]
        target = bundle_root / package_path
        require(not target.is_symlink(), f"symlink rejected for payload file: {package_path}")
        require(target.exists(), f"missing payload file: {package_path}")
        resolved = target.resolve()
        require(str(resolved).startswith(str(root_resolved)), f"payload path escapes bundle root: {package_path}")
        mode = target.stat().st_mode
        require(stat.S_ISREG(mode), f"payload file is not a regular file: {package_path}")
        data = target.read_bytes()
        require(len(data) == entry["byte_size"], f"payload byte size mismatch: {package_path}")
        require(sha256_bytes(data) == entry["sha256"], f"payload sha256 mismatch: {package_path}")
        named_blobs.append((package_path, data))
    archive_digest = payload_archive_digest(named_blobs)
    require(archive_digest == manifest["deterministic_archive_sha256"], "deterministic archive digest mismatch")
    return archive_digest


def _enumerate_files(bundle_root: Path) -> list[str]:
    found: list[str] = []
    root_resolved = bundle_root.resolve()
    for path in sorted(bundle_root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(bundle_root).as_posix()
        require(not path.is_symlink(), f"symlink present in bundle: {rel}")
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"special file present in bundle: {rel}")
        if path.is_file():
            resolved = path.resolve()
            require(str(resolved).startswith(str(root_resolved)), f"file escapes bundle root: {rel}")
            found.append(rel)
    return found


def validate_extracted_bundle(
    bundle_root: Path,
    manifest: dict[str, Any],
    *,
    strict: bool = True,
    permitted_runtime_outputs: set[str] | None = None,
) -> dict[str, Any]:
    validate_manifest_shape(manifest)
    archive_digest = validate_payload_files(bundle_root, manifest)
    declared = {entry["package_path"] for entry in manifest["files"]}
    permitted = permitted_extra_set(permitted_runtime_outputs)
    if strict:
        actual = _enumerate_files(bundle_root)
        seen_lower: dict[str, str] = {}
        for rel in actual:
            lower = rel.lower()
            require(lower not in seen_lower, f"case collision on disk: {rel} vs {seen_lower.get(lower)}")
            seen_lower[lower] = rel
        actual_set = set(actual)
        extras = {rel for rel in actual_set if rel not in declared and rel not in permitted}
        require(not extras, f"unexpected files in extracted bundle: {sorted(extras)}")
    return {
        "status": "GATE_A_TARGET_BUNDLE_VALIDATED",
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "deterministic_archive_sha256": archive_digest,
        "strict": strict,
        "payload_file_count": len(declared),
    }
