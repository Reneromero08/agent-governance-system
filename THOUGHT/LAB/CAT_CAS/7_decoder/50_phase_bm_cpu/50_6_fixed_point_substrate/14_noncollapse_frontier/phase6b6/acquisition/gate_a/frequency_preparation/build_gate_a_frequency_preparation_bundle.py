#!/usr/bin/env python3
"""Deterministic Git-bound bundle builder for frequency preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import gate_a_frequency_preparation_bundle as target_bundle

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / target_bundle.MANIFEST_FILENAME
PACKAGE_FILES = (
    ("gate_a_frequency_preparation.py", HERE / "gate_a_frequency_preparation.py", "reviewed_preparation_core"),
    ("gate_a_frequency_preparation_authority.py", HERE / "gate_a_frequency_preparation_authority.py", "authority_validator"),
    ("gate_a_frequency_preparation_bundle.py", HERE / "gate_a_frequency_preparation_bundle.py", "target_bundle"),
    ("gate_a_frequency_preparation_live.py", HERE / "gate_a_frequency_preparation_live.py", "live_transaction"),
    ("gate_a_frequency_preparation_target.py", HERE / "gate_a_frequency_preparation_target.py", "target_runner"),
)


class BuildError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise BuildError(message)


def repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / ".git").exists():
            return candidate
    raise BuildError("repository root not found")


def rel(path: Path) -> str:
    return path.resolve().relative_to(repo_root().resolve()).as_posix()


def run_git(*args: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(["git", *args], cwd=repo_root(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr if text else result.stderr[:200]!r}")
    return result


def git_blob(treeish: str, path: Path) -> tuple[str, str, bytes]:
    relative = rel(path)
    listing = run_git("ls-tree", treeish, "--", relative).stdout.strip().split()
    require(len(listing) >= 3, f"Git tree entry missing: {relative}")
    mode, kind, blob = listing[:3]
    require(mode == "100644" and kind == "blob", f"unexpected Git object for {relative}")
    raw = run_git("cat-file", "blob", blob, text=False).stdout
    return mode, blob, raw


def render_manifest(treeish: str = "HEAD") -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    named_blobs: list[tuple[str, bytes]] = []
    for package_path, source, role in sorted(PACKAGE_FILES):
        mode, blob, raw = git_blob(treeish, source)
        files.append({
            "package_path": package_path,
            "source_repository_path": rel(source),
            "git_mode": mode,
            "git_blob_sha1": blob,
            "byte_size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "role": role,
        })
        named_blobs.append((package_path, raw))
    core = {
        "schema_id": target_bundle.MANIFEST_SCHEMA_ID,
        "files": files,
        "authority_artifact_created": False,
        "live_frequency_preparation_authorized": False,
        "target_contact_authorized": False,
    }
    manifest = dict(core)
    manifest["bundle_sha256"] = hashlib.sha256(target_bundle.canonical_bytes(core)).hexdigest()
    manifest["deterministic_archive_sha256"] = hashlib.sha256(target_bundle.payload_archive_bytes(named_blobs)).hexdigest()
    target_bundle.validate_manifest_shape(manifest)
    return manifest


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def validate_committed_manifest_exact(treeish: str = "HEAD") -> dict[str, Any]:
    expected = render_manifest(treeish)
    actual = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    require(actual == expected, "committed frequency-preparation manifest is stale")
    return actual


def payload_archive(treeish: str = "HEAD") -> bytes:
    manifest = render_manifest(treeish)
    blobs = []
    for entry in manifest["files"]:
        raw = run_git("cat-file", "blob", entry["git_blob_sha1"], text=False).stdout
        blobs.append((entry["package_path"], raw))
    archive = target_bundle.payload_archive_bytes(blobs)
    require(hashlib.sha256(archive).hexdigest() == manifest["deterministic_archive_sha256"], "payload archive digest mismatch")
    return archive


def deployment_archive(treeish: str = "HEAD") -> bytes:
    manifest = render_manifest(treeish)
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in manifest["files"]:
            raw = run_git("cat-file", "blob", entry["git_blob_sha1"], text=False).stdout
            info = tarfile.TarInfo(entry["package_path"])
            info.size = len(raw)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.pax_headers = {}
            archive.addfile(info, BytesIO(raw))
        envelope = manifest_bytes(manifest)
        info = tarfile.TarInfo(target_bundle.MANIFEST_FILENAME)
        info.size = len(envelope)
        info.mode = 0o644
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.pax_headers = {}
        archive.addfile(info, BytesIO(envelope))
    return stream.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treeish", default="HEAD")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--compare-twice", action="store_true")
    parser.add_argument("--emit-deployment-archive", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_manifest(args.treeish)
    if args.write_manifest:
        MANIFEST_PATH.write_bytes(manifest_bytes(rendered))
    if MANIFEST_PATH.exists() and not args.write_manifest:
        validate_committed_manifest_exact(args.treeish)
    if args.compare_twice:
        require(render_manifest(args.treeish) == render_manifest(args.treeish), "manifest double render mismatch")
        require(deployment_archive(args.treeish) == deployment_archive(args.treeish), "deployment archive double build mismatch")
    if args.emit_deployment_archive is not None:
        data = deployment_archive(args.treeish)
        args.emit_deployment_archive.write_bytes(data)
    print(json.dumps(rendered, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
