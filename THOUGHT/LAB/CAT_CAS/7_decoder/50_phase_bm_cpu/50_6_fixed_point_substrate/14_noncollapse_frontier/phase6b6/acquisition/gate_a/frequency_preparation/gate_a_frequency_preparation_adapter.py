#!/usr/bin/env python3
"""Host entry point for one authority-gated preparation/restoration qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import build_gate_a_frequency_preparation_bundle as bundle
import gate_a_frequency_preparation_authority as authority
import gate_a_frequency_preparation_transport as transport

HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "GATE_A_FREQUENCY_PREPARATION_AUTHORITY.json"
MANIFEST_PATH = HERE / "GATE_A_FREQUENCY_PREPARATION_BUNDLE_MANIFEST.json"
PROTECTED_PATHS = tuple(sorted({
    *(bundle.rel(source) for _package, source, _role in bundle.PACKAGE_FILES),
    bundle.rel(bundle.MANIFEST_PATH),
    bundle.rel(HERE / "build_gate_a_frequency_preparation_bundle.py"),
    bundle.rel(HERE / "gate_a_frequency_preparation_transport.py"),
    bundle.rel(HERE / "gate_a_frequency_preparation_adapter.py"),
    bundle.rel(HERE / "schemas" / "gate_a_frequency_preparation_authority.schema.json"),
}))


class AdapterError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AdapterError(message)


def git(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, check=False)


def validate_git_custody(authority_path: Path, authority_value: dict[str, Any]) -> dict[str, str]:
    root = bundle.repo_root()
    exact_path = authority_path.resolve()
    canonical = AUTHORITY_PATH.resolve()
    require(exact_path == canonical, "preparation authority path is not canonical")
    require(exact_path.is_file() and not exact_path.is_symlink(), "preparation authority must be a regular file")
    relative = exact_path.relative_to(root).as_posix()

    head_result = git(root, "rev-parse", "HEAD")
    require(head_result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head_result.stdout.strip()) is not None, "authority-bearing HEAD unavailable")
    head = head_result.stdout.strip()
    reviewed = authority_value["reviewed_source_commit"]
    require(reviewed != head, "authority must be committed after reviewed source")
    require(git(root, "cat-file", "-e", f"{reviewed}^{{commit}}").returncode == 0, "reviewed source commit missing")
    require(git(root, "merge-base", "--is-ancestor", reviewed, head).returncode == 0, "reviewed source is not ancestor")

    parent_result = git(root, "rev-list", "--parents", "-n", "1", head)
    parent_fields = parent_result.stdout.strip().split()
    require(parent_result.returncode == 0 and parent_fields == [head, reviewed], "authority-bearing commit must have exactly the reviewed source as its sole parent")
    changed_result = git(root, "diff", "--name-only", reviewed, head)
    changed_paths = [line for line in changed_result.stdout.splitlines() if line]
    require(changed_result.returncode == 0 and changed_paths == [relative], "authority-bearing commit must change only the canonical authority artifact")

    require(git(root, "cat-file", "-e", f"{reviewed}:{relative}").returncode != 0, "authority existed at reviewed source")
    require(git(root, "diff", "--quiet", reviewed, head, "--", *PROTECTED_PATHS).returncode == 0, "protected preparation source drifted after review")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", *PROTECTED_PATHS, relative)
    require(status.returncode == 0 and status.stdout == "", "authority or protected source differs from HEAD")

    blob_result = git(root, "rev-parse", f"HEAD:{relative}")
    require(blob_result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", blob_result.stdout.strip()) is not None, "authority blob unavailable")
    blob = blob_result.stdout.strip()
    committed = git(root, "cat-file", "blob", blob, text=False)
    require(committed.returncode == 0 and committed.stdout == exact_path.read_bytes(), "authority bytes differ from committed blob")
    reviewed_tree = git(root, "rev-parse", f"{reviewed}^{{tree}}").stdout.strip()
    head_tree = git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    require(re.fullmatch(r"[0-9a-f]{40}", reviewed_tree) is not None, "reviewed tree unavailable")
    require(re.fullmatch(r"[0-9a-f]{40}", head_tree) is not None, "authority tree unavailable")
    return {
        "reviewed_source_commit": reviewed,
        "reviewed_source_tree": reviewed_tree,
        "authority_bearing_commit": head,
        "authority_bearing_tree": head_tree,
        "authority_git_blob_sha1": blob,
    }


def source_blob_bindings(treeish: str) -> dict[str, str]:
    mapping = {
        "host_adapter": HERE / "gate_a_frequency_preparation_adapter.py",
        "host_transport": HERE / "gate_a_frequency_preparation_transport.py",
        "authority_validator": HERE / "gate_a_frequency_preparation_authority.py",
        "live_transaction": HERE / "gate_a_frequency_preparation_live.py",
        "target_runner": HERE / "gate_a_frequency_preparation_target.py",
        "target_bundle": HERE / "gate_a_frequency_preparation_bundle.py",
        "reviewed_preparation_core": HERE / "gate_a_frequency_preparation.py",
    }
    result = {}
    for role, path in mapping.items():
        blob = git(bundle.repo_root(), "rev-parse", f"{treeish}:{bundle.rel(path)}")
        require(blob.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", blob.stdout.strip()) is not None, f"source blob unavailable: {role}")
        result[role] = blob.stdout.strip()
    return result


def validate_source_blob_bindings(authority_value: dict[str, Any], reviewed_source: str) -> None:
    require(authority_value["source_git_blobs"] == source_blob_bindings(reviewed_source), "authority source blob bindings mismatch")


def build_source_review_binding(authority_value: dict[str, Any], custody: dict[str, str], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_SOURCE_REVIEW_BINDING_V1",
        **custody,
        "independent_review_id": authority_value["independent_review_id"],
        "authority_sha256": hashlib.sha256(AUTHORITY_PATH.read_bytes()).hexdigest(),
        "source_git_blobs": authority_value["source_git_blobs"],
        "bundle_sha256": manifest["bundle_sha256"],
        "deterministic_archive_sha256": manifest["deterministic_archive_sha256"],
        "manifest_sha256": hashlib.sha256(bundle.manifest_bytes(manifest)).hexdigest(),
        "target": authority_value["target"],
        "target_identity_sha256": authority_value["target_identity_sha256"],
        "remote_execution_root": authority_value["remote_execution_root"],
        "remote_output_root": authority_value["remote_output_root"],
        "remote_claim_root": authority_value["remote_claim_root"],
    }


def execute_authorized(args: argparse.Namespace) -> dict[str, Any]:
    authority_bytes = args.authority_artifact.read_bytes()
    authority_value = json.loads(authority_bytes.decode("utf-8"))
    require(isinstance(authority_value, dict), "authority artifact must be object")
    custody = validate_git_custody(args.authority_artifact, authority_value)
    reviewed = authority_value["reviewed_source_commit"]
    manifest = bundle.render_manifest(reviewed)
    committed_manifest = json.loads(git(bundle.repo_root(), "show", f"{reviewed}:{bundle.rel(MANIFEST_PATH)}").stdout)
    require(committed_manifest == manifest, "reviewed source manifest is stale")
    manifest_bytes = bundle.manifest_bytes(manifest)
    require(hashlib.sha256(manifest_bytes).hexdigest() == authority_value["manifest_sha256"], "authority manifest SHA-256 mismatch")
    validate_source_blob_bindings(authority_value, reviewed)
    permit = authority.validate_authority(
        authority_value,
        authority_bytes=authority_bytes,
        authority_sha256=args.authority_sha256,
        exact_manifest=manifest,
        expected_reviewed_source_commit=reviewed,
        expected_independent_review_id=args.independent_review_id,
    )
    deployment = bundle.deployment_archive(reviewed)
    require(hashlib.sha256(bundle.payload_archive(reviewed)).hexdigest() == manifest["deterministic_archive_sha256"], "reviewed source archive mismatch")
    request = transport.TransportRequest(
        permit=permit,
        authority_path=args.authority_artifact,
        authority_bytes=authority_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        deployment_archive=deployment,
        local_evidence_root=args.local_evidence_root,
        source_review_binding=build_source_review_binding(authority_value, custody, manifest),
    )
    return transport.run_transport(request)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate A frequency preparation authority adapter")
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--authority-artifact", type=Path)
    parser.add_argument("--authority-sha256")
    parser.add_argument("--independent-review-id", type=int)
    parser.add_argument("--local-evidence-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute_authorized:
        print("FREQUENCY_PREPARATION_AUTHORITY_REQUIRED", file=sys.stderr)
        return 2
    try:
        require(args.authority_artifact is not None, "authority artifact required")
        require(args.authority_sha256 is not None, "authority SHA-256 required")
        require(args.independent_review_id is not None, "independent review ID required")
        require(args.local_evidence_root is not None, "local evidence root required")
        result = execute_authorized(args)
    except BaseException as exc:
        print(json.dumps({"status": "FAILED_CLOSED_ADAPTER", "failure": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
