#!/usr/bin/env python3
"""Git-free validator for the frequency-preparation target payload."""

from __future__ import annotations

import hashlib
import json
import stat
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "GATE_A_FREQUENCY_PREPARATION_BUNDLE_MANIFEST.json"
MANIFEST_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_BUNDLE_MANIFEST_V1"
MANIFEST_KEYS = {
    "schema_id",
    "files",
    "bundle_sha256",
    "deterministic_archive_sha256",
    "authority_artifact_created",
    "live_frequency_preparation_authorized",
    "target_contact_authorized",
}
FILE_KEYS = {
    "package_path",
    "source_repository_path",
    "git_mode",
    "git_blob_sha1",
    "byte_size",
    "sha256",
    "role",
}
REQUIRED_ROLES = {
    "reviewed_preparation_core",
    "authority_validator",
    "live_transaction",
    "target_runner",
    "target_bundle",
}


class BundleError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise BundleError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(char in "0123456789abcdef" for char in value)


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
            info.pax_headers = {}
            archive.addfile(info, BytesIO(data))
    return stream.getvalue()


def validate_manifest_shape(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(set(manifest) == MANIFEST_KEYS, "manifest key set mismatch")
    require(manifest["schema_id"] == MANIFEST_SCHEMA_ID, "manifest schema mismatch")
    require(manifest["authority_artifact_created"] is False, "source manifest contains authority")
    require(manifest["live_frequency_preparation_authorized"] is False, "source manifest authorizes live preparation")
    require(manifest["target_contact_authorized"] is False, "source manifest authorizes target contact")
    require(is_hex(manifest["bundle_sha256"], 64), "bundle digest malformed")
    require(is_hex(manifest["deterministic_archive_sha256"], 64), "archive digest malformed")
    files = manifest["files"]
    require(isinstance(files, list) and len(files) == 5, "manifest file count mismatch")
    roles: dict[str, dict[str, Any]] = {}
    previous = ""
    for entry in files:
        require(isinstance(entry, dict) and set(entry) == FILE_KEYS, "manifest file entry mismatch")
        package_path = entry["package_path"]
        require(isinstance(package_path, str) and package_path > previous, "package paths not sorted")
        require(not package_path.startswith("/") and ".." not in Path(package_path).parts, "unsafe package path")
        require(entry["git_mode"] == "100644", "package mode mismatch")
        require(is_hex(entry["git_blob_sha1"], 40), "package blob malformed")
        require(isinstance(entry["byte_size"], int) and entry["byte_size"] > 0, "package size malformed")
        require(is_hex(entry["sha256"], 64), "package SHA-256 malformed")
        role = entry["role"]
        require(role in REQUIRED_ROLES and role not in roles, "package role mismatch")
        roles[role] = entry
        previous = package_path
    require(set(roles) == REQUIRED_ROLES, "manifest roles incomplete")
    core = {key: value for key, value in manifest.items() if key not in {"bundle_sha256", "deterministic_archive_sha256"}}
    require(sha256_bytes(canonical_bytes(core)) == manifest["bundle_sha256"], "bundle digest mismatch")
    return roles


def load_manifest(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / MANIFEST_FILENAME
    require(path.is_file() and not path.is_symlink(), "manifest envelope missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "manifest must be object")
    return value


def validate_extracted_bundle(bundle_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    validate_manifest_shape(manifest)
    root = bundle_root.resolve()
    require(root.is_dir() and not root.is_symlink(), "bundle root invalid")
    expected_paths = {entry["package_path"] for entry in manifest["files"]} | {MANIFEST_FILENAME}
    observed_paths: set[str] = set()
    blobs: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), f"invalid bundle path: {path}")
        observed_paths.add(path.relative_to(root).as_posix())
    require(observed_paths == expected_paths, f"bundle path mismatch: expected={sorted(expected_paths)} observed={sorted(observed_paths)}")
    for entry in manifest["files"]:
        path = root / entry["package_path"]
        mode = stat.S_IMODE(path.stat().st_mode)
        require(mode & 0o222 == 0 or mode == 0o644, "unexpected extracted payload mode")
        data = path.read_bytes()
        require(len(data) == entry["byte_size"], f"payload size mismatch: {entry['package_path']}")
        require(sha256_bytes(data) == entry["sha256"], f"payload digest mismatch: {entry['package_path']}")
        blobs.append((entry["package_path"], data))
    archive = payload_archive_bytes(blobs)
    require(sha256_bytes(archive) == manifest["deterministic_archive_sha256"], "payload archive digest mismatch")
    return {
        "status": "FREQUENCY_PREPARATION_BUNDLE_EXACT",
        "bundle_sha256": manifest["bundle_sha256"],
        "deterministic_archive_sha256": manifest["deterministic_archive_sha256"],
        "file_count": len(manifest["files"]),
    }
