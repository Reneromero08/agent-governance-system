#!/usr/bin/env python3
"""Build and verify the deterministic Gate A execution-bundle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

BASE_MAIN = "03985a74d27e654c151cccad28c6221d91f70180"
REVIEWED_PLAN_HEAD = "65d20b4bc65ddd9260a3c90d92612d2da48763a6"
PLAN_REVIEW_ID = 4617290767
SCHEDULE_SHA256 = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
NAMESPACE_SHA256 = "5b3090f642d28492e182630e6349eccd8181704f08129d40d886c8f529dfd50e"
TARGET_IDENTITY_SHA256 = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
MANIFEST_PATH = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"

PACKAGE_FILES = [
    ("adapter/gate_a_authority.py", HERE / "gate_a_authority.py", "shared_authority_validator"),
    ("adapter/gate_a_hardware_adapter.py", HERE / "gate_a_hardware_adapter.py", "host_adapter"),
    ("adapter/gate_a_target_runner.py", HERE / "gate_a_target_runner.py", "target_runner"),
    ("adapter/gate_a_worker.c", HERE / "gate_a_worker.c", "target_worker"),
    ("adapter/GATE_A_ADAPTER_QUALIFICATION_CONTRACT.json", HERE / "GATE_A_ADAPTER_QUALIFICATION_CONTRACT.json", "qualification_contract"),
    ("adapter/schemas/gate_a_execution_authority.schema.json", HERE / "schemas" / "gate_a_execution_authority.schema.json", "future_authority_schema"),
    ("GATE_A_ENGINEERING_SMOKE_SCHEDULE.json", GATE_A / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json", "schedule"),
    ("GATE_A_TARGET_NAMESPACE_BINDING.json", GATE_A / "GATE_A_TARGET_NAMESPACE_BINDING.json", "target_namespace"),
]

MANIFEST_KEYS = {
    "schema_id",
    "base_main_commit",
    "reviewed_gate_a_plan_head",
    "gate_a_plan_review",
    "schedule_sha256",
    "target_namespace_sha256",
    "target_identity_stdout_sha256",
    "authority_artifact_created",
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


class BundleError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise BundleError(message)


def repo_root() -> Path:
    for parent in (HERE, *HERE.parents):
        if (parent / ".git").exists():
            return parent
    raise BundleError("repository root not found")


def rel(path: Path) -> str:
    return path.resolve().relative_to(repo_root().resolve()).as_posix()


def git(args: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root(), input=input_text, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BlobSource:
    mode: str
    blob: str


def git_index_source(path: Path) -> BlobSource:
    output = git(["ls-files", "--stage", "--", rel(path)]).strip()
    require(output, f"path is not staged in Git index: {rel(path)}")
    parts = output.split()
    require(len(parts) >= 4, f"unexpected index entry: {output}")
    return BlobSource(mode=parts[0], blob=parts[1])


def git_tree_source(treeish: str, path: Path) -> BlobSource:
    output = git(["ls-tree", treeish, "--", rel(path)]).strip()
    require(output, f"path is not present in {treeish}: {rel(path)}")
    meta = output.split("\t", 1)[0].split()
    require(len(meta) == 3, f"unexpected tree entry: {output}")
    return BlobSource(mode=meta[0], blob=meta[2])


def blob_bytes(blob: str) -> bytes:
    return subprocess.run(["git", "cat-file", "blob", blob], cwd=repo_root(), check=True, stdout=subprocess.PIPE).stdout


def assert_clean_for_paths(paths: list[Path]) -> None:
    rels = [rel(path) for path in paths]
    unstaged = git(["diff", "--name-only", "--", *rels]).splitlines()
    staged = git(["diff", "--cached", "--name-only", "--", *rels]).splitlines()
    require(not unstaged, f"worktree-only modifications rejected: {unstaged}")
    require(not staged, f"index-only modifications rejected for committed-tree build: {staged}")


def build_entries(treeish: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_package: set[str] = set()
    seen_lower: set[str] = set()
    for package_path, source_path, role in PACKAGE_FILES:
        require(package_path not in seen_package, f"duplicate package path: {package_path}")
        require(package_path.lower() not in seen_lower, f"case collision: {package_path}")
        seen_package.add(package_path)
        seen_lower.add(package_path.lower())
        require(not Path(package_path).is_absolute() and ".." not in Path(package_path).parts, f"unsafe package path: {package_path}")
        source = git_index_source(source_path) if treeish == ":" else git_tree_source(treeish, source_path)
        require(source.mode == "100644", f"unexpected Git mode for {rel(source_path)}: {source.mode}")
        data = blob_bytes(source.blob)
        entries.append({
            "package_path": package_path,
            "source_repository_path": rel(source_path),
            "git_mode": source.mode,
            "git_blob_sha1": source.blob,
            "byte_size": len(data),
            "sha256": sha256_bytes(data),
            "role": role,
        })
    return entries


def validate_package_paths(entries: list[dict[str, Any]]) -> None:
    seen_package: set[str] = set()
    seen_lower: set[str] = set()
    for entry in entries:
        require(set(entry) == FILE_ENTRY_KEYS, f"manifest file entry key set mismatch: {entry}")
        package_path = entry["package_path"]
        require(isinstance(package_path, str), "package path must be a string")
        require(package_path not in seen_package, f"duplicate package path: {package_path}")
        require(package_path.lower() not in seen_lower, f"case-colliding package path: {package_path}")
        seen_package.add(package_path)
        seen_lower.add(package_path.lower())
        require(not Path(package_path).is_absolute() and ".." not in Path(package_path).parts, f"unsafe package path: {package_path}")
        require(entry["git_mode"] == "100644", f"unexpected Git mode for {package_path}: {entry['git_mode']}")
        require(isinstance(entry["source_repository_path"], str), f"source path must be a string: {package_path}")
        require(not Path(entry["source_repository_path"]).is_absolute() and ".." not in Path(entry["source_repository_path"]).parts, f"unsafe source path: {entry['source_repository_path']}")


def archive_digest(entries: list[dict[str, Any]]) -> str:
    validate_package_paths(entries)
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            data = blob_bytes(entry["git_blob_sha1"])
            info = tarfile.TarInfo(name=entry["package_path"])
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, BytesIO(data))
    return sha256_bytes(stream.getvalue())


def render_manifest(treeish: str) -> dict[str, Any]:
    paths = [source_path for _, source_path, _ in PACKAGE_FILES]
    if treeish != ":":
        assert_clean_for_paths(paths)
    entries = build_entries(treeish)
    core = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_BUNDLE_MANIFEST_V1",
        "base_main_commit": BASE_MAIN,
        "reviewed_gate_a_plan_head": REVIEWED_PLAN_HEAD,
        "gate_a_plan_review": PLAN_REVIEW_ID,
        "schedule_sha256": SCHEDULE_SHA256,
        "target_namespace_sha256": NAMESPACE_SHA256,
        "target_identity_stdout_sha256": TARGET_IDENTITY_SHA256,
        "authority_artifact_created": False,
        "engineering_smoke_authorized": False,
        "hardware_ran": False,
        "files": entries,
    }
    core["execution_bundle_sha256"] = sha256_bytes(canonical_bytes(core))
    core["deterministic_archive_sha256"] = archive_digest(entries)
    return core


def validate_manifest(manifest: dict[str, Any]) -> None:
    validate_committed_manifest_exact(manifest, "HEAD")


def validate_committed_manifest_exact(manifest: dict[str, Any], treeish: str) -> dict[str, Any]:
    require(set(manifest) == MANIFEST_KEYS, "manifest top-level key set mismatch")
    require(isinstance(manifest["files"], list), "manifest files must be a list")
    validate_package_paths(manifest["files"])
    execution_digest = manifest["execution_bundle_sha256"]
    archive = manifest["deterministic_archive_sha256"]
    require(isinstance(execution_digest, str) and len(execution_digest) == 64, "execution bundle digest malformed")
    require(isinstance(archive, str) and len(archive) == 64, "deterministic archive digest malformed")
    expected = render_manifest(treeish)
    require(manifest == expected, "committed execution bundle manifest mismatch")
    return expected


def validate_manifest_digests(manifest: dict[str, Any]) -> None:
    execution_digest = manifest["execution_bundle_sha256"]
    archive = manifest["deterministic_archive_sha256"]
    unsigned = dict(manifest)
    unsigned.pop("execution_bundle_sha256")
    unsigned.pop("deterministic_archive_sha256")
    require(sha256_bytes(canonical_bytes(unsigned)) == execution_digest, "execution bundle digest mismatch")
    require(archive == archive_digest(manifest["files"]), "deterministic archive digest mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Gate A deterministic execution bundle")
    parser.add_argument("--treeish", default="HEAD", help="Git treeish to read; ':' reads staged index")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--compare-twice", action="store_true")
    args = parser.parse_args(argv)
    manifest = render_manifest(args.treeish)
    validate_manifest_digests(manifest)
    validate_committed_manifest_exact(manifest, args.treeish)
    if args.compare_twice:
        with tempfile.TemporaryDirectory(prefix="gate_a_bundle_"):
            manifest_b = render_manifest(args.treeish)
        require(manifest == manifest_b, "bundle manifest is not stable across two builds")
    if args.write_manifest:
        MANIFEST_PATH.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BundleError, subprocess.CalledProcessError) as exc:
        print(f"build_gate_a_execution_bundle: {exc}", file=sys.stderr)
        raise SystemExit(1)
