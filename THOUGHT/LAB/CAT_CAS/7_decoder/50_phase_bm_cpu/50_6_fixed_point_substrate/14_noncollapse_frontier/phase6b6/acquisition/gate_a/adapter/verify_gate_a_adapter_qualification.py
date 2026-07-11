#!/usr/bin/env python3
"""Dedicated Gate A adapter no-drive qualification verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import build_gate_a_execution_bundle as bundle
import gate_a_engineering_smoke_transport as smoke_transport
import gate_a_hardware_adapter as adapter
import gate_a_isolated_qualification as isolated_harness
import gate_a_process_custody as process_custody
import gate_a_target_bundle as target_bundle
import gate_a_target_runner

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
RESULT = HERE / "GATE_A_ADAPTER_QUALIFICATION_RESULT.json"
CANDIDATE_V2 = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V2.json"
MANIFEST = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
AUTHORITY_SCHEMA = HERE / "schemas" / "gate_a_execution_authority.schema.json"
AUTHORITY_NAME = "GATE_A_EXECUTION_AUTHORITY.json"
AUTHORITY = HERE / AUTHORITY_NAME
CONTRACT = HERE / "GATE_A_ADAPTER_QUALIFICATION_CONTRACT.json"
HISTORICAL_ADAPTER_COMMIT = "6f243b1aaf7cfaa09f21b8d5816ddd9097612f72"
HISTORICAL_MANIFEST_SHA256 = "ccb7866db67170083cb00d546c334b61772c8ef909131ec9c62ed21115facc94"
HISTORICAL_RESULT_SHA256 = "1d9d2c62cbf81f72eeb9c40f02841f9f507d52eae8229da73fc2f81eb0a15223"
HISTORICAL_CANDIDATE_V2_SHA256 = "d8f190bc7f8c9904659cd697ed091b192843efe18f5f1d74d713282e889b060e"
REVIEWED_EXECUTION_SOURCE_COMMIT = "4d281112d9da56cd3c99e6860121d1ab1b9d3e47"
REVIEWED_EXECUTION_SOURCE_TREE = "7a41d145eed40c79f068f01223f6f1e1b076c6bc"
REVIEWED_EXECUTION_SOURCE_REVIEW_ID = 4676060226
REVIEWED_MANIFEST_BLOB_SHA1 = "9fa8a9e573b482c75df18a3da99669467dbbdd13"
REVIEWED_MANIFEST_SHA256 = "117cf39db81b1e1a84948eb6ceaf40912be1f3e2e0b2f6e1c489e5f21944fb71"
EXECUTION_AUTHORITY_SHA256 = "7e1e8835bd67590e4e554ae112a2c8a6ca99dd8b9b3a9aafdb23fee31907d682"
EXECUTION_AUTHORITY_BLOB_SHA1 = "709c799f60e30984d3c80715af480fbe5deac952"
ENGINEERING_SMOKE_EVIDENCE = (
    GATE_A.parents[1] / "evidence" / "gate_a_engineering_smoke_7e1e8835"
)
ENGINEERING_SMOKE_EXECUTION_COMMIT = "9fd5ae7fccf76956b63afaa77442fa8bc337170a"
ENGINEERING_SMOKE_EXECUTION_TREE = "1fa555764ea50de60176d3e8e2670cdaa0cde1e2"
ENGINEERING_SMOKE_FINAL_INVENTORY_SHA256 = "c87b9b2b392c9918ba3af01ca80afa3fc53fca5beae8ae30ee06055158f71c44"
ENGINEERING_SMOKE_FINAL_INVENTORY_CANONICAL_SHA256 = "9e6dbffd166f34204fed9d38369daee2670c592aa335751d3cefe11e4521a239"
ENGINEERING_SMOKE_FINAL_BINDINGS_SHA256 = "c0a5e30334c568dd6e2b37a090b4637c102a24936e2aa8e760c840f24ec4314d"
ENGINEERING_SMOKE_HOST_COMMANDS_SHA256 = "d6277d83bce7b8608e9c775bf0195c4a60b2d11cda7cba1e26df167e08b1a3af"
ENGINEERING_SMOKE_TARGET_INVENTORY_SHA256 = "91f9d8b953b919049f8078c0a8ce1cf87f7ea35a150b87cdececd67caabf0646"
ENGINEERING_SMOKE_AUTHORITY_CLAIM_SHA256 = "fdaa4b71e2e37dea828d6a0a1ba9494c6bddc36b885ff6cab7b4a9be72959dbf"
ENGINEERING_SMOKE_EXECUTION_STARTED_SHA256 = "80e2362808ce3defbe6e5c20431f3f4802fff28df1b846634ad6cb4ffb7f6121"
ENGINEERING_SMOKE_NEXT_BOUNDARY = "INDEPENDENT_GATE_A_ENGINEERING_SMOKE_EVIDENCE_REVIEW"


class VerifyError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerifyError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def committed_sha256(path: Path, treeish: str = "HEAD") -> str:
    if treeish == ":":
        blob = bundle.git_index_source(path).blob
    else:
        blob = run(["git", "rev-parse", f"{treeish}:{git_rel(path)}"], cwd=bundle.repo_root()).stdout.strip()
    data = subprocess.run(["git", "cat-file", "blob", blob], cwd=bundle.repo_root(), stdout=subprocess.PIPE, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def committed_object(path: Path, treeish: str) -> dict[str, Any]:
    blob = run(["git", "rev-parse", f"{treeish}:{git_rel(path)}"], cwd=bundle.repo_root()).stdout.strip()
    data = subprocess.run(["git", "cat-file", "blob", blob], cwd=bundle.repo_root(), stdout=subprocess.PIPE, check=True).stdout
    value = json.loads(data)
    require(isinstance(value, dict), f"committed object required: {treeish}:{git_rel(path)}")
    return value


def active_treeish() -> str:
    return os.environ.get("GATE_A_BUNDLE_TREEISH", "HEAD")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def run(argv: list[str], *, cwd: Path = HERE, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, input=input_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def git_blob(path: Path) -> str:
    return run(["git", "hash-object", str(path)]).stdout.strip()


def git_rel(path: Path) -> str:
    return path.resolve().relative_to(bundle.repo_root().resolve()).as_posix()


def validate_execution_authority_inventory(
    filesystem_hits: list[str],
    tracked_hits: list[str],
    canonical_rel: str,
) -> bool:
    if not filesystem_hits and not tracked_hits:
        return False
    require(filesystem_hits == [canonical_rel], f"alternate or worktree-only execution authority present: {filesystem_hits}")
    require(tracked_hits == [canonical_rel], f"execution authority tracked-path set mismatch: {tracked_hits}")
    return True


def validate_execution_authority_state(manifest: dict[str, Any]) -> dict[str, Any]:
    """Accept absence or validate the one exact canonical committed authority."""

    root = bundle.repo_root().resolve()
    authority_search_root = GATE_A.parents[1].resolve()
    canonical_rel = git_rel(AUTHORITY)
    schema_rel = git_rel(AUTHORITY_SCHEMA)
    filesystem_hits: list[str] = []

    def fail_walk(error: OSError) -> None:
        raise VerifyError(f"cannot inspect execution-authority namespace: {error}")

    for directory, _child_directories, filenames in os.walk(
        authority_search_root,
        topdown=True,
        onerror=fail_walk,
        followlinks=False,
    ):
        for filename in filenames:
            folded = filename.casefold()
            if not folded.startswith("gate_a_execution_authority") or not folded.endswith(".json"):
                continue
            relative = (Path(directory) / filename).relative_to(root).as_posix()
            if relative != schema_rel:
                filesystem_hits.append(relative)
    filesystem_hits.sort()
    tracked_result = run(
        ["git", "ls-files"],
        cwd=root,
        check=False,
    )
    require(tracked_result.returncode == 0, f"cannot enumerate tracked execution authorities: {tracked_result.stderr}")
    tracked_hits = sorted(
        line
        for line in tracked_result.stdout.splitlines()
        if line != schema_rel
        and Path(line).name.casefold().startswith("gate_a_execution_authority")
        and Path(line).name.casefold().endswith(".json")
    )

    status = run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            ":(icase,glob)**/gate_a_execution_authority*.json",
            f":(exclude){schema_rel}",
        ],
        cwd=root,
        check=False,
    )
    require(status.returncode == 0 and status.stdout == "", "execution authority differs from current HEAD")

    head = run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    reviewed_source_available = run(
        ["git", "cat-file", "-e", f"{REVIEWED_EXECUTION_SOURCE_COMMIT}^{{commit}}"],
        cwd=root,
        check=False,
    ).returncode == 0
    reviewed_source_lane = False
    if reviewed_source_available:
        ancestor = run(
            ["git", "merge-base", "--is-ancestor", REVIEWED_EXECUTION_SOURCE_COMMIT, head],
            cwd=root,
            check=False,
        )
        require(ancestor.returncode in (0, 1), "cannot determine reviewed-source ancestry")
        reviewed_source_lane = ancestor.returncode == 0

    protected_paths = tuple(sorted({
        *(bundle.rel(source) for _package, source, _role in bundle.PACKAGE_FILES),
        bundle.rel(bundle.MANIFEST_PATH),
    }))
    if reviewed_source_lane:
        reviewed_tree = run(
            ["git", "rev-parse", f"{REVIEWED_EXECUTION_SOURCE_COMMIT}^{{tree}}"],
            cwd=root,
        ).stdout.strip()
        require(reviewed_tree == REVIEWED_EXECUTION_SOURCE_TREE, "reviewed source tree mismatch")
        manifest_rel = git_rel(MANIFEST)
        reviewed_manifest_blob = run(
            ["git", "rev-parse", f"{REVIEWED_EXECUTION_SOURCE_COMMIT}:{manifest_rel}"],
            cwd=root,
        ).stdout.strip()
        head_manifest_blob = run(
            ["git", "rev-parse", f"HEAD:{manifest_rel}"],
            cwd=root,
        ).stdout.strip()
        require(reviewed_manifest_blob == REVIEWED_MANIFEST_BLOB_SHA1, "reviewed execution manifest blob mismatch")
        require(head_manifest_blob == REVIEWED_MANIFEST_BLOB_SHA1, "current execution manifest differs from reviewed source")
        require(
            committed_sha256(MANIFEST, REVIEWED_EXECUTION_SOURCE_COMMIT) == REVIEWED_MANIFEST_SHA256,
            "reviewed execution manifest SHA-256 mismatch",
        )
        require(committed_sha256(MANIFEST, "HEAD") == REVIEWED_MANIFEST_SHA256, "current execution manifest bytes drifted")
        protected_drift = run(
            ["git", "diff", "--quiet", REVIEWED_EXECUTION_SOURCE_COMMIT, head, "--", *protected_paths],
            cwd=root,
            check=False,
        )
        require(protected_drift.returncode == 0, "protected execution source drifted after review")
        protected_status = run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *protected_paths],
            cwd=root,
            check=False,
        )
        require(
            protected_status.returncode == 0 and protected_status.stdout == "",
            "protected execution source differs from current HEAD",
        )

    authority_present = validate_execution_authority_inventory(filesystem_hits, tracked_hits, canonical_rel)
    if not authority_present:
        require(
            not reviewed_source_lane or head == REVIEWED_EXECUTION_SOURCE_COMMIT,
            "reviewed-source descendant is missing its committed execution authority",
        )
        return {
            "status": "GATE_A_EXECUTION_AUTHORITY_ABSENT",
            "authority_artifact_present": False,
        }

    require(reviewed_source_lane, "execution authority is not descended from the reviewed source")
    require(AUTHORITY.is_file() and not AUTHORITY.is_symlink(), "canonical execution authority must be a real regular file")

    authority_bytes = AUTHORITY.read_bytes()
    authority = load(AUTHORITY)
    require(sha256_file(AUTHORITY) == EXECUTION_AUTHORITY_SHA256, "execution authority SHA-256 mismatch")
    require(git_blob(AUTHORITY) == EXECUTION_AUTHORITY_BLOB_SHA1, "execution authority worktree blob mismatch")
    head_authority_blob = run(
        ["git", "rev-parse", f"HEAD:{canonical_rel}"],
        cwd=root,
    ).stdout.strip()
    require(head_authority_blob == EXECUTION_AUTHORITY_BLOB_SHA1, "execution authority committed blob mismatch")

    exact_manifest = bundle.validate_committed_manifest_exact(manifest, "HEAD")
    validation = adapter.gate_a_authority.validate_execution_authority(
        authority,
        authority_sha256=EXECUTION_AUTHORITY_SHA256,
        authority_bytes=authority_bytes,
        expected_reviewed_adapter_head=REVIEWED_EXECUTION_SOURCE_COMMIT,
        expected_independent_review_id=REVIEWED_EXECUTION_SOURCE_REVIEW_ID,
        exact_manifest=exact_manifest,
    )
    custody = adapter.validate_authority_git_custody(AUTHORITY, authority)
    require(validation["status"] == "GATE_A_EXECUTION_AUTHORITY_EXACT", "production authority validation failed")
    require(validation["reviewed_adapter_head"] == REVIEWED_EXECUTION_SOURCE_COMMIT, "authority reviewed source mismatch")
    require(validation["independent_review_id"] == REVIEWED_EXECUTION_SOURCE_REVIEW_ID, "authority review binding mismatch")
    require(custody["reviewed_source_tree"] == REVIEWED_EXECUTION_SOURCE_TREE, "reviewed source tree mismatch")
    require(custody["authority_git_blob_sha1"] == EXECUTION_AUTHORITY_BLOB_SHA1, "authority custody blob mismatch")
    require(custody["protected_path_count"] == len(protected_paths), "protected source count mismatch")
    require(authority["project_owner_approved"] is True, "project-owner execution approval missing")
    require(authority["maximum_execution_count"] == 1, "execution authority count must be one")
    require(authority["consumed"] is False, "execution authority must be unconsumed")
    require(authority["authority_state"]["automatic_retry"] is False, "automatic retry must remain false")
    for field in (
        "calibration_authorized",
        "scientific_acquisition_authorized",
        "restoration_authorized",
        "target_coupling_authorized",
        "small_wall_authorized",
    ):
        require(authority["authority_state"][field] is False, f"downstream authority must remain false: {field}")
    return {
        "status": "GATE_A_EXECUTION_AUTHORITY_COMMITTED_EXACT",
        "authority_artifact_present": True,
        "authority_artifact_path": canonical_rel,
        "authority_sha256": EXECUTION_AUTHORITY_SHA256,
        "authority_git_blob_sha1": EXECUTION_AUTHORITY_BLOB_SHA1,
        "reviewed_source_commit": REVIEWED_EXECUTION_SOURCE_COMMIT,
        "reviewed_source_tree": REVIEWED_EXECUTION_SOURCE_TREE,
        "independent_review_id": REVIEWED_EXECUTION_SOURCE_REVIEW_ID,
        "authority_bearing_head": custody["authority_bearing_head"],
        "authority_bearing_tree": custody["authority_bearing_tree"],
        "protected_path_count": custody["protected_path_count"],
        "execution_bundle_sha256": validation["execution_bundle_sha256"],
        "maximum_execution_count": authority["maximum_execution_count"],
        "consumed": authority["consumed"],
        "automatic_retry": authority["authority_state"]["automatic_retry"],
        "downstream_authority_false": True,
    }


def validate_engineering_smoke_evidence_git_custody(
    evidence_root: Path,
    *,
    treeish: str,
) -> dict[str, Any]:
    """Require one exact committed packet whose blobs match the working tree."""

    root = bundle.repo_root().resolve()
    require(
        evidence_root.resolve() == ENGINEERING_SMOKE_EVIDENCE.resolve(),
        "engineering-smoke evidence is not at its canonical path",
    )
    evidence_rel = git_rel(evidence_root)
    actual_files: dict[str, Path] = {}
    actual_directories: set[str] = set()
    for path in evidence_root.rglob("*"):
        require(not path.is_symlink(), f"engineering-smoke evidence contains symlink: {path}")
        relative = path.relative_to(evidence_root).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files[git_rel(path)] = path
        else:
            raise VerifyError(f"engineering-smoke evidence contains non-file object: {path}")
    require(actual_directories == {"TARGET_OUTPUT"}, "engineering-smoke evidence directory set mismatch")
    require(len(actual_files) == 20, "engineering-smoke evidence file count mismatch")

    if treeish == ":":
        listing = run(["git", "ls-files", "--stage", "--", evidence_rel], cwd=root, check=False)
    else:
        listing = run(["git", "ls-tree", "-r", treeish, "--", evidence_rel], cwd=root, check=False)
    require(listing.returncode == 0, f"cannot enumerate committed engineering-smoke evidence: {listing.stderr}")
    committed: dict[str, tuple[str, str]] = {}
    for line in listing.stdout.splitlines():
        metadata, separator, path = line.partition("\t")
        require(separator == "\t", "malformed committed engineering-smoke evidence entry")
        fields = metadata.split()
        require(len(fields) == 3, "malformed committed engineering-smoke evidence metadata")
        mode = fields[0]
        if treeish == ":":
            blob = fields[1]
        else:
            require(fields[1] == "blob", "engineering-smoke evidence Git object is not a blob")
            blob = fields[2]
        committed[path] = (mode, blob)
    require(set(committed) == set(actual_files), "committed engineering-smoke evidence tree mismatch")
    for relative, path in actual_files.items():
        mode, blob = committed[relative]
        require(mode == "100644", f"engineering-smoke evidence mode mismatch: {relative}")
        require(blob == git_blob(path), f"engineering-smoke evidence blob differs from working tree: {relative}")

    if treeish == ":":
        clean = run(["git", "diff", "--quiet", "--", evidence_rel], cwd=root, check=False)
        require(clean.returncode == 0, "engineering-smoke evidence working tree differs from staged packet")
    else:
        status = run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", evidence_rel],
            cwd=root,
            check=False,
        )
        require(
            status.returncode == 0 and status.stdout == "",
            "engineering-smoke evidence differs from current HEAD",
        )
    return {
        "treeish": treeish,
        "file_count": len(actual_files),
        "working_tree_matches_committed_blobs": True,
    }


def validate_engineering_smoke_evidence_state(
    evidence_root: Path = ENGINEERING_SMOKE_EVIDENCE,
    *,
    require_committed: bool = True,
) -> dict[str, Any]:
    """Validate and project current state from the sealed one-shot packet."""

    require(evidence_root.is_dir() and not evidence_root.is_symlink(), "engineering-smoke evidence directory missing")
    try:
        packet = smoke_transport.validate_final_packet(evidence_root)
    except smoke_transport.TransportError as exc:
        raise VerifyError(f"engineering-smoke final packet invalid: {exc}") from exc
    require(packet["status"] == "GATE_A_FINAL_PACKET_VALID", "engineering-smoke final packet status mismatch")
    require(packet["file_count"] == 19, "engineering-smoke final inventory entry count mismatch")
    require(
        packet["final_inventory_sha256"] == ENGINEERING_SMOKE_FINAL_INVENTORY_CANONICAL_SHA256,
        "engineering-smoke canonical inventory digest mismatch",
    )
    require(
        sha256_file(evidence_root / "FINAL_EVIDENCE_INVENTORY.json")
        == ENGINEERING_SMOKE_FINAL_INVENTORY_SHA256,
        "engineering-smoke final inventory file SHA-256 mismatch",
    )
    require(
        sha256_file(evidence_root / "FINAL_BINDINGS.json") == ENGINEERING_SMOKE_FINAL_BINDINGS_SHA256,
        "engineering-smoke final bindings SHA-256 mismatch",
    )
    require(
        sha256_file(evidence_root / "HOST_COMMANDS.jsonl") == ENGINEERING_SMOKE_HOST_COMMANDS_SHA256,
        "engineering-smoke host command ledger SHA-256 mismatch",
    )

    execution_tree = run(
        ["git", "rev-parse", f"{ENGINEERING_SMOKE_EXECUTION_COMMIT}^{{tree}}"],
        cwd=bundle.repo_root(),
    ).stdout.strip()
    require(execution_tree == ENGINEERING_SMOKE_EXECUTION_TREE, "engineering-smoke execution tree mismatch")
    execution_ancestor = run(
        ["git", "merge-base", "--is-ancestor", ENGINEERING_SMOKE_EXECUTION_COMMIT, "HEAD"],
        cwd=bundle.repo_root(),
        check=False,
    )
    require(execution_ancestor.returncode == 0, "engineering-smoke execution commit is not an ancestor of HEAD")

    manifest = load(evidence_root / "EXECUTION_BUNDLE_MANIFEST.json")
    authority = load(AUTHORITY)
    authority_copy = evidence_root / "AUTHORITY_ARTIFACT.json"
    schedule_copy = evidence_root / "SCHEDULE.json"
    schedule_source = GATE_A / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
    require(authority_copy.read_bytes() == AUTHORITY.read_bytes(), "sealed authority copy differs from canonical artifact")
    require(sha256_file(authority_copy) == EXECUTION_AUTHORITY_SHA256, "sealed authority copy SHA-256 mismatch")
    require(manifest == load(MANIFEST), "sealed execution manifest differs from reviewed manifest")
    require(sha256_file(evidence_root / "EXECUTION_BUNDLE_MANIFEST.json") == REVIEWED_MANIFEST_SHA256, "sealed manifest SHA-256 mismatch")
    require(schedule_copy.read_bytes() == schedule_source.read_bytes(), "sealed schedule differs from frozen schedule")

    source_binding = load(evidence_root / "SOURCE_REVIEW_BINDING.json")
    request = smoke_transport.HostExecutionRequest(
        target=authority["target"],
        authority_path=AUTHORITY,
        authority_sha256=EXECUTION_AUTHORITY_SHA256,
        reviewed_adapter_head=REVIEWED_EXECUTION_SOURCE_COMMIT,
        independent_review_id=REVIEWED_EXECUTION_SOURCE_REVIEW_ID,
        execution_bundle_sha256=manifest["execution_bundle_sha256"],
        schedule_sha256=bundle.SCHEDULE_SHA256,
        namespace_sha256=bundle.NAMESPACE_SHA256,
        remote_execution_root=authority["remote_execution_root"],
        remote_output_root=authority["remote_output_root"],
        local_evidence_root=evidence_root,
        authority_bytes=authority_copy.read_bytes(),
        schedule_bytes=schedule_copy.read_bytes(),
        manifest_bytes=(evidence_root / "EXECUTION_BUNDLE_MANIFEST.json").read_bytes(),
        source_review_binding=source_binding,
        authority_bearing_execution_commit=ENGINEERING_SMOKE_EXECUTION_COMMIT,
        reviewed_source_tree=REVIEWED_EXECUTION_SOURCE_TREE,
        authority_bearing_execution_tree=ENGINEERING_SMOKE_EXECUTION_TREE,
        authority_git_blob_sha1=EXECUTION_AUTHORITY_BLOB_SHA1,
    )
    try:
        smoke_transport.validate_source_review_binding(source_binding, request=request, manifest=manifest)
    except smoke_transport.TransportError as exc:
        raise VerifyError(f"engineering-smoke source-review binding invalid: {exc}") from exc

    bindings = load(evidence_root / "FINAL_BINDINGS.json")
    require(set(bindings) == {
        "artifact_sha256", "authority_bearing_execution_commit", "authority_sha256",
        "automatic_retry", "completed_stages", "execution_bundle_sha256", "failed_stage",
        "independent_review_id", "namespace_sha256", "primary_error", "retry_count",
        "reviewed_source_commit", "runner_start_count", "schedule_sha256", "schema_id",
        "secondary_errors", "status", "transport_execution_count",
    }, "engineering-smoke final bindings key set mismatch")
    require(bindings["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_FINAL_BINDINGS_V1", "engineering-smoke bindings schema mismatch")
    require(bindings["status"] == "GATE_A_AUTHORIZED_TRANSPORT_FAILED_CLOSED", "engineering-smoke transport result mismatch")
    require(bindings["failed_stage"] == "target_result_verification", "engineering-smoke failed stage mismatch")
    require(bindings["primary_error"] == "authorized runtime failed after evidence custody", "engineering-smoke primary error mismatch")
    require(bindings["secondary_errors"] == [], "engineering-smoke secondary errors present")
    require(bindings["authority_bearing_execution_commit"] == ENGINEERING_SMOKE_EXECUTION_COMMIT, "engineering-smoke execution commit binding mismatch")
    require(bindings["reviewed_source_commit"] == REVIEWED_EXECUTION_SOURCE_COMMIT, "engineering-smoke reviewed source binding mismatch")
    require(bindings["independent_review_id"] == REVIEWED_EXECUTION_SOURCE_REVIEW_ID, "engineering-smoke review binding mismatch")
    require(bindings["authority_sha256"] == EXECUTION_AUTHORITY_SHA256, "engineering-smoke authority binding mismatch")
    require(bindings["execution_bundle_sha256"] == manifest["execution_bundle_sha256"], "engineering-smoke bundle binding mismatch")
    require(bindings["schedule_sha256"] == bundle.SCHEDULE_SHA256, "engineering-smoke schedule binding mismatch")
    require(bindings["namespace_sha256"] == bundle.NAMESPACE_SHA256, "engineering-smoke namespace binding mismatch")
    require(bindings["transport_execution_count"] == 1, "engineering-smoke transport execution count mismatch")
    require(bindings["runner_start_count"] == 1, "engineering-smoke runner start count mismatch")
    require(bindings["retry_count"] == 0, "engineering-smoke retry count changed")
    require(bindings["automatic_retry"] is False, "engineering-smoke automatic retry changed")
    require(bindings["completed_stages"] == [
        "authority_validated", "remote_namespace_inspected", "authority_claimed", "bundle_staged",
        "authority_staged", "target_runner_started", "post_runtime_process_scanned",
        "evidence_archived", "evidence_copied_back", "copy_back_verified",
        "copy_back_receipt_uploaded", "remote_cleanup_attempted", "post_cleanup_process_scanned",
    ], "engineering-smoke completed-stage sequence mismatch")
    artifact_digests = bindings["artifact_sha256"]
    expected_artifacts = {
        "AUTHORITY_ARTIFACT.json", "AUTHORITY_CLAIM_RECEIPT.json", "CLEANUP_RECEIPT.json",
        "COPY_BACK_RECEIPT.json", "EXECUTION_BUNDLE_MANIFEST.json", "HOST_COMMANDS.jsonl",
        "POST_CLEANUP_PROCESS_RECEIPT.json", "POST_RUNTIME_PROCESS_RECEIPT.json", "SCHEDULE.json",
        "SOURCE_REVIEW_BINDING.json", "TARGET_EVIDENCE_INVENTORY.json",
        "TARGET_EXECUTION_RECEIPT.json", "TRANSPORT_FAILURE_RECEIPT.json",
    }
    require(isinstance(artifact_digests, dict) and set(artifact_digests) == expected_artifacts, "engineering-smoke artifact digest set mismatch")
    for name, digest in artifact_digests.items():
        require(digest == sha256_file(evidence_root / name), f"engineering-smoke artifact digest mismatch: {name}")

    failure_receipt = load(evidence_root / "TRANSPORT_FAILURE_RECEIPT.json")
    require(set(failure_receipt) == {
        "authority_claim_preserved", "authority_claim_state", "automatic_retry", "cleanup_attempted",
        "copy_back_verified", "failed_stage", "mutation_attempted", "primary_error", "retry_count",
        "runner_start_count", "schema_id", "secondary_errors",
    }, "engineering-smoke transport failure receipt key set mismatch")
    require(failure_receipt == {
        "authority_claim_preserved": True,
        "authority_claim_state": "confirmed",
        "automatic_retry": False,
        "cleanup_attempted": True,
        "copy_back_verified": True,
        "failed_stage": "target_result_verification",
        "mutation_attempted": True,
        "primary_error": "authorized runtime failed after evidence custody",
        "retry_count": 0,
        "runner_start_count": 1,
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_TRANSPORT_FAILURE_V1",
        "secondary_errors": [],
    }, "engineering-smoke transport failure receipt mismatch")

    claim = load(evidence_root / "AUTHORITY_CLAIM_RECEIPT.json")
    require(set(claim) == {"claim_created", "claim_root", "claim_sha256"}, "engineering-smoke authority claim key set mismatch")
    require(claim["claim_created"] is True, "engineering-smoke authority claim was not created")
    require(claim["claim_sha256"] == ENGINEERING_SMOKE_AUTHORITY_CLAIM_SHA256, "engineering-smoke authority claim digest mismatch")
    require(claim["claim_root"].endswith(EXECUTION_AUTHORITY_SHA256), "engineering-smoke authority claim root mismatch")

    attempt = load(evidence_root / "TARGET_OUTPUT" / "ATTEMPT.json")
    require(set(attempt) == {
        "authority_sha256", "automatic_retry", "execution_bundle_sha256",
        "maximum_execution_count", "preflight", "schema_id", "sequence",
    }, "engineering-smoke attempt key set mismatch")
    require(attempt["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ATTEMPT_V1", "engineering-smoke attempt schema mismatch")
    require(attempt["authority_sha256"] == EXECUTION_AUTHORITY_SHA256, "engineering-smoke attempt authority mismatch")
    require(attempt["execution_bundle_sha256"] == manifest["execution_bundle_sha256"], "engineering-smoke attempt bundle mismatch")
    require(attempt["maximum_execution_count"] == 1, "engineering-smoke attempt execution cap mismatch")
    require(attempt["automatic_retry"] is False, "engineering-smoke attempt automatic retry changed")
    require(attempt["sequence"] == [
        "I", "I", "I", "I", "C0", "D0", "S0E", "S0E", "S0E", "S0E",
        "O0", "O0", "A0P", "A0N", "T", "T",
    ], "engineering-smoke frozen sequence mismatch")
    require(attempt["preflight"] == {
        "frequency_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "namespace_state": "absent",
        "preflight_complete": False,
        "voltage_writes": 0,
    }, "engineering-smoke preflight receipt mismatch")
    failure = load(evidence_root / "TARGET_OUTPUT" / "FAILURE.json")
    require(failure == {
        "automatic_retry": False,
        "partial_evidence_preserved": True,
        "reason": "temperature unobservable",
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FAILURE_V1",
    }, "engineering-smoke target failure mismatch")

    events = [json.loads(line) for line in (evidence_root / "TARGET_OUTPUT" / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(events) == 2, "engineering-smoke event count mismatch")
    require(events[0] == {
        "event": "pre_runtime_process_scan",
        "forbidden_process_hits": [],
        "return_code": 0,
        "scan_complete": True,
        "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "stdout_sha256": "c44baa2c70fdf4a9a38e61ad86c4a57f53a08eb4b5534da7338a708fc03e0458",
    }, "engineering-smoke pre-runtime event mismatch")
    require(events[1] == {"event": "runtime_failed", "reason": "temperature unobservable"}, "engineering-smoke failure event mismatch")

    target_inventory = load(evidence_root / "TARGET_EVIDENCE_INVENTORY.json")
    try:
        target_inventory_digest = smoke_transport._validate_inventory_shape(target_inventory)
    except smoke_transport.TransportError as exc:
        raise VerifyError(f"engineering-smoke target inventory invalid: {exc}") from exc
    require(target_inventory_digest == ENGINEERING_SMOKE_TARGET_INVENTORY_SHA256, "engineering-smoke target inventory digest mismatch")
    target_output = evidence_root / "TARGET_OUTPUT"
    require(
        {item["path"] for item in target_inventory["files"]}
        == {path.name for path in target_output.iterdir() if path.is_file()},
        "engineering-smoke target inventory does not close over target output",
    )
    for item in target_inventory["files"]:
        path = target_output / item["path"]
        require(path.stat().st_size == item["size"], f"engineering-smoke target evidence size mismatch: {item['path']}")
        require(sha256_file(path) == item["sha256"], f"engineering-smoke target evidence SHA-256 mismatch: {item['path']}")

    target_execution = load(evidence_root / "TARGET_EXECUTION_RECEIPT.json")
    require(set(target_execution) == {
        "evidence_archive_created", "post_runtime_process_receipt", "runner_command",
        "runner_return_code", "runner_stderr", "runner_stderr_sha256", "runner_stdout",
        "runner_stdout_sha256", "target_evidence_inventory", "target_evidence_inventory_sha256",
        "target_timeout",
    }, "engineering-smoke target execution receipt key set mismatch")
    require(target_execution["runner_return_code"] == 1, "engineering-smoke target runner return code mismatch")
    require(target_execution["target_timeout"] is False, "engineering-smoke target runner timed out")
    require(target_execution["evidence_archive_created"] is True, "engineering-smoke target archive missing")
    require(target_execution["runner_stdout"] == "", "engineering-smoke target runner stdout changed")
    require(target_execution["runner_stderr"] == "gate_a_target_runner: temperature unobservable\n", "engineering-smoke target runner stderr mismatch")
    require(target_execution["runner_stdout_sha256"] == sha256_text(target_execution["runner_stdout"]), "engineering-smoke target stdout digest mismatch")
    require(target_execution["runner_stderr_sha256"] == sha256_text(target_execution["runner_stderr"]), "engineering-smoke target stderr digest mismatch")
    require(target_execution["target_evidence_inventory"] == target_inventory, "engineering-smoke embedded target inventory mismatch")
    require(target_execution["target_evidence_inventory_sha256"] == ENGINEERING_SMOKE_TARGET_INVENTORY_SHA256, "engineering-smoke embedded target inventory digest mismatch")
    runner_command = target_execution["runner_command"]
    require(isinstance(runner_command, list) and runner_command.count("--execute-authorized") == 1, "engineering-smoke target runner command count mismatch")
    require(runner_command[0:3] == ["/usr/bin/python3", "-B", f"{authority['remote_execution_root']}/adapter/gate_a_target_runner.py"], "engineering-smoke target runner prefix mismatch")
    for flag, expected in (
        ("--authority-sha256", EXECUTION_AUTHORITY_SHA256),
        ("--execution-bundle-sha256", manifest["execution_bundle_sha256"]),
        ("--source-head", REVIEWED_EXECUTION_SOURCE_COMMIT),
        ("--independent-review-id", str(REVIEWED_EXECUTION_SOURCE_REVIEW_ID)),
        ("--schedule-sha256", bundle.SCHEDULE_SHA256),
        ("--target", authority["target"]),
        ("--namespace-sha256", bundle.NAMESPACE_SHA256),
        ("--output-root", authority["remote_output_root"]),
        ("--transport-claim-root", claim["claim_root"]),
    ):
        require(runner_command.count(flag) == 1, f"engineering-smoke target runner flag count mismatch: {flag}")
        index = runner_command.index(flag)
        require(index + 1 < len(runner_command) and runner_command[index + 1] == expected, f"engineering-smoke target runner binding mismatch: {flag}")

    pre_runtime = load(target_output / "PRE_RUNTIME_PROCESS_RECEIPT.json")
    target_post_runtime = load(target_output / "POST_RUNTIME_PROCESS_RECEIPT.json")
    post_runtime = load(evidence_root / "POST_RUNTIME_PROCESS_RECEIPT.json")
    post_cleanup = load(evidence_root / "POST_CLEANUP_PROCESS_RECEIPT.json")
    try:
        process_custody.validate_process_receipt(pre_runtime, expected_phase="pre_runtime")
        process_custody.validate_process_receipt(target_post_runtime, expected_phase="post_runtime")
        process_custody.validate_process_receipt(post_runtime, expected_phase="post_runtime")
        process_custody.validate_process_receipt(post_cleanup, expected_phase="post_cleanup")
    except process_custody.ProcessCustodyError as exc:
        raise VerifyError(f"engineering-smoke process custody invalid: {exc}") from exc
    require(target_post_runtime == post_runtime, "engineering-smoke post-runtime process receipts differ")
    require(target_execution["post_runtime_process_receipt"] == post_runtime, "engineering-smoke embedded post-runtime receipt differs")

    copy_back = load(evidence_root / "COPY_BACK_RECEIPT.json")
    require(set(copy_back) == {
        "archive_sha256", "authority_sha256", "copy_back_complete",
        "downloaded_evidence_inventory_sha256", "evidence_inventory_sha256",
        "execution_bundle_sha256", "remote_output_root", "retained_evidence_custody_verified",
        "schema_id", "target_evidence_inventory_sha256",
    }, "engineering-smoke copy-back receipt key set mismatch")
    require(copy_back["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1", "engineering-smoke copy-back schema mismatch")
    require(copy_back["authority_sha256"] == EXECUTION_AUTHORITY_SHA256, "engineering-smoke copy-back authority mismatch")
    require(copy_back["execution_bundle_sha256"] == manifest["execution_bundle_sha256"], "engineering-smoke copy-back bundle mismatch")
    require(copy_back["remote_output_root"] == authority["remote_output_root"], "engineering-smoke copy-back output root mismatch")
    require(copy_back["copy_back_complete"] is True and copy_back["retained_evidence_custody_verified"] is True, "engineering-smoke copy-back custody unverified")
    for field in (
        "downloaded_evidence_inventory_sha256", "evidence_inventory_sha256",
        "target_evidence_inventory_sha256",
    ):
        require(copy_back[field] == ENGINEERING_SMOKE_TARGET_INVENTORY_SHA256, f"engineering-smoke copy-back inventory mismatch: {field}")

    cleanup = load(evidence_root / "CLEANUP_RECEIPT.json")
    require(set(cleanup) == {
        "parsed", "raw_response", "raw_response_sha256", "raw_stderr",
        "raw_stderr_sha256", "schema_id",
    }, "engineering-smoke cleanup receipt key set mismatch")
    require(cleanup["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V1", "engineering-smoke cleanup schema mismatch")
    require(cleanup["raw_response_sha256"] == sha256_text(cleanup["raw_response"]), "engineering-smoke cleanup response digest mismatch")
    require(cleanup["raw_stderr_sha256"] == sha256_text(cleanup["raw_stderr"]), "engineering-smoke cleanup stderr digest mismatch")
    require(cleanup["raw_stderr"] == "", "engineering-smoke cleanup command wrote stderr")
    require(json.loads(cleanup["raw_response"]) == cleanup["parsed"], "engineering-smoke cleanup parsed/raw observations differ")
    try:
        smoke_transport.validate_cleanup_result(cleanup["parsed"], request=request)
    except smoke_transport.TransportError as exc:
        raise VerifyError(f"engineering-smoke cleanup receipt invalid: {exc}") from exc
    require(cleanup["parsed"]["cleanup_mode"] == "verified_copyback", "engineering-smoke cleanup mode mismatch")
    require(cleanup["parsed"]["execution_started_sha256"] == ENGINEERING_SMOKE_EXECUTION_STARTED_SHA256, "engineering-smoke execution-start marker mismatch")

    command_lines = (evidence_root / "HOST_COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()
    commands = [json.loads(line) for line in command_lines]
    expected_operations = [
        ("remote_namespace_inspected", "remote_namespace_inspected", "ssh"),
        ("authority_claimed", "authority_claimed", "ssh"),
        ("bundle_staged", "bundle_staged", "scp"),
        ("authority_staged", "authority_staged", "scp"),
        ("target_runner_started", "target_runner_started", "ssh"),
        ("evidence_download", "evidence_copied_back", "scp"),
        ("copy_back_receipt_upload", "copy_back_receipt_uploaded", "scp"),
        ("remote_cleanup_attempted", "remote_cleanup_attempted", "ssh"),
        ("post_cleanup_process_scan", "post_cleanup_process_scanned", "ssh"),
    ]
    require(len(commands) == len(expected_operations), "engineering-smoke host command count mismatch")
    for sequence, (command, expected) in enumerate(zip(commands, expected_operations), start=1):
        operation, stage, program = expected
        require(set(command) == {
            "command", "command_sha256", "end_monotonic_ns", "end_utc_ns", "failure",
            "operation", "raw_stderr", "raw_stdout", "return_code", "schema_id", "sequence",
            "stage", "start_monotonic_ns", "start_utc_ns", "stderr_sha256", "stdin_sha256",
            "stdin_size", "stdout_sha256", "timed_out", "timeout_seconds",
        }, f"engineering-smoke host command key set mismatch: {sequence}")
        require(command["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_HOST_COMMAND_V1", f"engineering-smoke host command schema mismatch: {sequence}")
        require(command["sequence"] == sequence, f"engineering-smoke host command sequence mismatch: {sequence}")
        require(command["operation"] == operation and command["stage"] == stage, f"engineering-smoke host command operation mismatch: {sequence}")
        require(isinstance(command["command"], list) and command["command"][0] == program, f"engineering-smoke host command transport mismatch: {sequence}")
        require(command["command_sha256"] == canonical_sha256(command["command"]), f"engineering-smoke host command digest mismatch: {sequence}")
        require(command["return_code"] == 0 and command["timed_out"] is False and command["failure"] is None, f"engineering-smoke host command failed: {sequence}")
        require(command["stdout_sha256"] == sha256_text(command["raw_stdout"]), f"engineering-smoke host stdout digest mismatch: {sequence}")
        require(command["stderr_sha256"] == sha256_text(command["raw_stderr"]), f"engineering-smoke host stderr digest mismatch: {sequence}")
    require(sum(command["operation"] == "target_runner_started" for command in commands) == 1, "engineering-smoke runner was not started exactly once")
    require(sum(command["command"][0] == "ssh" for command in commands) == 5, "engineering-smoke SSH command count mismatch")
    require(sum(command["command"][0] == "scp" for command in commands) == 4, "engineering-smoke SCP command count mismatch")

    expected_target_files = {
        "ATTEMPT.json", "EVENTS.jsonl", "FAILURE.json",
        "POST_RUNTIME_PROCESS_RECEIPT.json", "PRE_RUNTIME_PROCESS_RECEIPT.json",
    }
    require({path.name for path in target_output.iterdir()} == expected_target_files, "physical-runtime, sender, or capture artifact unexpectedly present")

    git_custody: dict[str, Any] = {"required": require_committed}
    if require_committed:
        git_custody.update(
            validate_engineering_smoke_evidence_git_custody(
                evidence_root,
                treeish=active_treeish(),
            )
        )
    return {
        "status": "GATE_A_ENGINEERING_SMOKE_FAILED_CLOSED__TEMPERATURE_UNOBSERVABLE",
        "result": "FAILED_CLOSED",
        "failure_reason": "temperature unobservable",
        "smoke_attempted": True,
        "orchestrator_invocation_count": 1,
        "target_contact_occurred": True,
        "target_contact_attempt_count": 1,
        "authority_artifact_consumed_field": authority["consumed"],
        "authority_consumed": True,
        "engineering_smoke_authorized": False,
        "execution_count": 1,
        "transport_execution_count": 1,
        "runner_start_count": 1,
        "physical_runtime_execution_count": 0,
        "retry_count": 0,
        "automatic_retry": False,
        "preflight_complete": False,
        "hardware_ran": False,
        "sender_started": False,
        "capture_started": False,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "copy_back_verified": True,
        "cleanup_verified": True,
        "durable_authority_claim_preserved": True,
        "pre_runtime_process_custody": "PASS",
        "post_runtime_process_custody": "PASS",
        "post_cleanup_process_custody": "PASS",
        "sender_lifecycle_custody": "NOT_APPLICABLE__PHYSICAL_RUNTIME_NOT_ENTERED",
        "raw_derived_iq_custody": "NOT_APPLICABLE__PHYSICAL_RUNTIME_NOT_ENTERED",
        "host_command_count": len(commands),
        "ssh_command_count": 5,
        "scp_command_count": 4,
        "evidence_path": git_rel(evidence_root),
        "execution_head": ENGINEERING_SMOKE_EXECUTION_COMMIT,
        "execution_tree": ENGINEERING_SMOKE_EXECUTION_TREE,
        "final_evidence_inventory_sha256": ENGINEERING_SMOKE_FINAL_INVENTORY_SHA256,
        "final_bindings_sha256": ENGINEERING_SMOKE_FINAL_BINDINGS_SHA256,
        "host_commands_sha256": ENGINEERING_SMOKE_HOST_COMMANDS_SHA256,
        "target_evidence_inventory_sha256": ENGINEERING_SMOKE_TARGET_INVENTORY_SHA256,
        "git_custody": git_custody,
        "next_boundary": ENGINEERING_SMOKE_NEXT_BOUNDARY,
    }


def validate_schema_closed(schema: dict[str, Any]) -> None:
    require(schema["additionalProperties"] is False, "authority schema top level open")
    require(set(schema["required"]) == set(schema["properties"]), "authority schema required/properties mismatch")
    state = schema["properties"]["authority_state"]
    require(state["additionalProperties"] is False, "authority state schema open")
    require(set(state["required"]) == set(state["properties"]), "authority state required/properties mismatch")


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ADAPTER_QUALIFICATION_CONTRACT_V1", "contract schema mismatch")
    require(contract["base_main_commit"] == bundle.BASE_MAIN, "contract base mismatch")
    require(contract["reviewed_gate_a_plan_head"] == bundle.REVIEWED_PLAN_HEAD, "contract reviewed plan mismatch")
    require(contract["gate_a_plan_review"] == bundle.PLAN_REVIEW_ID, "contract review mismatch")
    require(contract["schedule_sha256"] == bundle.SCHEDULE_SHA256, "contract schedule mismatch")
    require(contract["target_namespace_sha256"] == bundle.NAMESPACE_SHA256, "contract namespace mismatch")
    require(contract["target_identity_stdout_sha256"] == bundle.TARGET_IDENTITY_SHA256, "contract target identity mismatch")
    require(contract["target_geometry"]["automatic_retry"] is False, "contract retry boundary mismatch")
    require(contract["target_geometry"]["maximum_execution_count"] == 1, "contract execution count mismatch")
    for key, value in contract["authority_false_state"].items():
        require(value is False, f"contract authority flag must be false: {key}")
    required_tests = set(contract["required_negative_tests"])
    for expected in (
        "worktree-byte mutation behavior",
        "index-byte mutation detection",
        "missing authority rejects before transport",
        "committed two-commit authority custody",
        "ordinary worker live execution rejection",
        "second execution rejection",
        "automatic retry rejection",
        "target-local timeout rejection",
        "four-way remote namespace preflight rejection before transfer",
        "complete capture requirement",
        "partial evidence preservation",
        "cleanup requires verified copy-back",
        "cleanup inventory digest recomputation",
        "physically absent sender lifecycle custody",
        "one contiguous STEP sender epoch",
        "distinct bounded anchor epochs",
        "continuous capture across sender lifecycle",
        "raw-derived per-slot lock-in I/Q recomputation",
        "altered lock-in range, tone, slot, I or Q rejection",
        "pre-runtime process receipt required",
        "post-runtime process receipt required on success and failure",
        "post-cleanup process receipt required",
        "process command, return code, raw streams and hashes bound",
        "complete retained host evidence packet",
        "final evidence inventory closure",
        "host command ledger closure",
        "durable authority claim survives cleanup",
        "transport failure injection preserves local failure receipt",
        "transport failure state machine never retries runtime",
        "zero network contact in tests",
    ):
        require(expected in required_tests, f"contract negative test missing: {expected}")
    expected_sources = set(contract["expected_source_files"])
    for expected in (
        "gate_a_engineering_smoke_executor.py",
        "gate_a_process_custody.py",
        "gate_a_engineering_smoke_transport.py",
        "../../../../holo_runtime_v2/combined_pdn_hardware.c",
        "../../../../holo_runtime_v2/gate_a_engineering_smoke_runtime.c",
        "../../../../holo_runtime_v2/gate_a_engineering_smoke_runtime.h",
        "test_gate_a_engineering_smoke_executor.py",
    ):
        require(expected in expected_sources, f"contract source file missing: {expected}")


def static_forbidden_surface_scan() -> dict[str, Any]:
    blocked_regexes = [
        ("shell_true", "shell=True"),
        ("os_system", "os.system("),
        ("eval_call", "eval("),
        ("exec_call", "exec("),
    ]
    implementation_files = [
        HERE / "gate_a_authority.py",
        HERE / "gate_a_target_bundle.py",
        HERE / "gate_a_engineering_smoke_executor.py",
        HERE / "gate_a_process_custody.py",
        HERE / "gate_a_engineering_smoke_transport.py",
        HERE / "gate_a_hardware_adapter.py",
        HERE / "gate_a_target_runner.py",
        HERE / "gate_a_worker.c",
        HERE / "build_gate_a_execution_bundle.py",
        HERE / "gate_a_isolated_qualification.py",
    ]
    matches: list[dict[str, str]] = []
    for path in implementation_files:
        text = path.read_text(encoding="utf-8")
        for name, needle in blocked_regexes:
            if needle in text:
                matches.append({"file": path.name, "pattern": name})
    require(not matches, f"forbidden implementation surface present: {matches}")
    runner_text = (HERE / "gate_a_target_runner.py").read_text(encoding="utf-8")
    worker_text = (HERE / "gate_a_worker.c").read_text(encoding="utf-8")
    host_text = (HERE / "gate_a_hardware_adapter.py").read_text(encoding="utf-8")
    executor_text = (HERE / "gate_a_engineering_smoke_executor.py").read_text(encoding="utf-8")
    transport_text = (HERE / "gate_a_engineering_smoke_transport.py").read_text(encoding="utf-8")
    process_text = (HERE / "gate_a_process_custody.py").read_text(encoding="utf-8")
    require("authorized live execution path is intentionally unused" not in runner_text, "target execution sentinel remains")
    require("live execution unavailable" not in worker_text, "worker execution sentinel remains")
    require("run_gate_a_engineering_smoke" in worker_text, "worker does not call the bounded physical runtime")
    require("transport_factory" in host_text and "validate_future_authority" in host_text, "host authority-before-transport seam missing")
    require("validate_authority_git_custody" in host_text, "host committed-authority custody gate missing")
    require("timeout=self.timeout_s" in executor_text, "target worker timeout missing")
    require("start_new_session=True" in transport_text and "os.killpg" in transport_text and "signal.SIGKILL" in transport_text, "target process-group timeout cleanup missing")
    require("GATE_A_COMPILED_AUTHORITY_SHA256" in worker_text, "worker compile-time authority binding missing")
    require("PRE_RUNTIME_PROCESS_RECEIPT.json" in executor_text and "POST_RUNTIME_PROCESS_RECEIPT.json" in executor_text, "target process receipts are not retained")
    require("POST_CLEANUP_PROCESS_RECEIPT.json" in transport_text, "post-cleanup process receipt is not retained")
    require("raw_stdout_base64" in process_text and "stdout_sha256" in process_text and "parsed_forbidden_hits" in process_text, "shared process custody is incomplete")
    for retained in (
        "AUTHORITY_ARTIFACT.json", "SCHEDULE.json", "EXECUTION_BUNDLE_MANIFEST.json",
        "SOURCE_REVIEW_BINDING.json", "HOST_COMMANDS.jsonl", "TARGET_EXECUTION_RECEIPT.json",
        "TARGET_EVIDENCE_INVENTORY.json", "COPY_BACK_RECEIPT.json",
        "POST_RUNTIME_PROCESS_RECEIPT.json", "POST_CLEANUP_PROCESS_RECEIPT.json",
        "CLEANUP_RECEIPT.json",
        "FINAL_EVIDENCE_INVENTORY.json", "FINAL_BINDINGS.json",
    ):
        require(retained in transport_text, f"retained host packet artifact missing: {retained}")

    runtime = HERE.parents[3] / "holo_runtime_v2" / "gate_a_engineering_smoke_runtime.c"
    runtime_text = runtime.read_text(encoding="utf-8")
    marker = "int run_gate_a_engineering_smoke("
    require(marker in runtime_text, "bounded Gate A physical-runtime entry point missing")
    gate_a_body = runtime_text[runtime_text.index(marker):]
    require("GATE_A_COMPILED_OUTPUT_ROOT" in worker_text and "gate_a_runtime_output_root" in gate_a_body, "worker one-shot output binding missing")
    require("LOCKIN_IQ.jsonl" in runtime_text and "lockin(" in runtime_text, "raw-derived lock-in custody missing")
    require("SENDER_LIFECYCLE.jsonl" in runtime_text and "gate-a:anchor:positive" in runtime_text and "gate-a:anchor:negative" in runtime_text, "bounded sender lifecycle custody missing")
    require("16.0 * sender->slot_s" not in runtime_text, "one sender thread still spans the complete sequence")
    for name, needle in (
        ("frequency_control", "pin_frequency("),
        ("msr_access", "msr_read("),
        ("min_frequency_control", "scaling_min_freq"),
        ("max_frequency_control", "scaling_max_freq"),
        ("boost_control", "/cpufreq/boost"),
        ("msr_device", "/dev/cpu/"),
    ):
        require(needle not in gate_a_body, f"bounded Gate A runtime exposes forbidden surface: {name}")
    return {
        "status": "BOUNDED_EXECUTOR_SURFACE_SCAN_PASS",
        "generic_unsafe_matches": 0,
        "frequency_control_calls": 0,
        "msr_access_calls": 0,
        "intentional_execution_sentinels": 0,
    }


def assert_rejects(name: str, func: Callable[[], None]) -> str:
    try:
        func()
    except Exception:
        return name
    raise VerifyError(f"mutation accepted: {name}")


def target_args(manifest: dict[str, Any], reviewed_head: str, review_id: int, output_root: str | None = None) -> Any:
    return type("Args", (), {
        "authority_artifact": None,
        "authority_sha256": None,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "source_head": reviewed_head,
        "independent_review_id": review_id,
        "schedule_sha256": adapter.SCHEDULE_SHA256,
        "target": "root@192.168.137.100",
        "namespace_sha256": adapter.NAMESPACE_SHA256,
        "output_root": output_root or adapter.REMOTE_OUTPUT_ROOT,
        "claim_root": None,
    })()


def authority_template(manifest: dict[str, Any], reviewed_head: str, review_id: int) -> dict[str, Any]:
    files_by_role = {entry["role"]: entry for entry in manifest["files"]}
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_AUTHORITY_V1",
        "reviewed_adapter_head": reviewed_head,
        "independent_review_id": review_id,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "host_adapter_git_blob_sha1": files_by_role["host_adapter"]["git_blob_sha1"],
        "target_runner_git_blob_sha1": files_by_role["target_runner"]["git_blob_sha1"],
        "target_worker_git_blob_sha1": files_by_role["target_worker"]["git_blob_sha1"],
        "schedule_sha256": adapter.SCHEDULE_SHA256,
        "target_namespace_sha256": adapter.NAMESPACE_SHA256,
        "target_identity_sha256": adapter.TARGET_IDENTITY_SHA256,
        "target": "root@192.168.137.100",
        "remote_execution_root": adapter.REMOTE_EXECUTION_ROOT,
        "remote_output_root": adapter.REMOTE_OUTPUT_ROOT,
        "maximum_execution_count": 1,
        "consumed": False,
        "project_owner_approved": True,
        "authority_state": {
            "authorization_artifact_created": True,
            "engineering_smoke_authorized": True,
            "hardware_ran": False,
            "calibration_authorized": False,
            "scientific_acquisition_authorized": False,
            "restoration_authorized": False,
            "target_coupling_authorized": False,
            "small_wall_authorized": False,
            "automatic_retry": False,
        },
    }


def validate_authority_host(path: Path, digest: str, manifest: dict[str, Any], reviewed_head: str, review_id: int) -> dict[str, Any]:
    authority = load(path)
    return adapter.validate_future_authority(
        authority,
        authority_sha256=digest,
        authority_bytes=path.read_bytes(),
        expected_reviewed_adapter_head=reviewed_head,
        expected_independent_review_id=review_id,
        expected_manifest=manifest,
    )


def validate_authority_both(path: Path, digest: str, manifest: dict[str, Any], reviewed_head: str, review_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    host_result = validate_authority_host(path, digest, manifest, reviewed_head, review_id)
    exact_manifest = bundle.validate_committed_manifest_exact(manifest, active_treeish())
    runner_result = gate_a_target_runner.validate_authority(path, digest, target_args(manifest, reviewed_head, review_id), exact_manifest)
    return host_result, runner_result


def manifest_mutation_tests(manifest: dict[str, Any]) -> dict[str, Any]:
    cases: list[str] = []

    def mutate_file(field: str, value: Any) -> Callable[[], None]:
        def inner() -> None:
            changed = copy.deepcopy(manifest)
            changed["files"][0][field] = value
            adapter.validate_bundle_manifest(changed)
        return inner

    def remove_file() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"].pop()
        adapter.validate_bundle_manifest(changed)

    def add_file() -> None:
        changed = copy.deepcopy(manifest)
        extra = copy.deepcopy(changed["files"][0])
        extra["package_path"] = "adapter/extra.txt"
        changed["files"].append(extra)
        adapter.validate_bundle_manifest(changed)

    def duplicate_path() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"].append(copy.deepcopy(changed["files"][0]))
        adapter.validate_bundle_manifest(changed)

    def case_collision() -> None:
        changed = copy.deepcopy(manifest)
        extra = copy.deepcopy(changed["files"][0])
        extra["package_path"] = changed["files"][0]["package_path"].upper()
        changed["files"].append(extra)
        adapter.validate_bundle_manifest(changed)

    def unsafe_path() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"][0]["package_path"] = "../gate_a_hardware_adapter.py"
        adapter.validate_bundle_manifest(changed)

    def extra_top_level() -> None:
        changed = copy.deepcopy(manifest)
        changed["extra"] = True
        adapter.validate_bundle_manifest(changed)

    for name, func in (
        ("manifest_execution_bundle_sha256_changed", lambda: changed_top(manifest, "execution_bundle_sha256", "0" * 64)),
        ("manifest_deterministic_archive_sha256_changed", lambda: changed_top(manifest, "deterministic_archive_sha256", "0" * 64)),
        ("manifest_per_file_sha256_changed", mutate_file("sha256", "0" * 64)),
        ("manifest_byte_size_changed", mutate_file("byte_size", 1)),
        ("manifest_source_repository_path_changed", mutate_file("source_repository_path", "THOUGHT/changed.py")),
        ("manifest_package_path_changed", mutate_file("package_path", "adapter/changed.py")),
        ("manifest_role_changed", mutate_file("role", "changed_role")),
        ("manifest_git_blob_sha1_changed", mutate_file("git_blob_sha1", "0" * 40)),
        ("manifest_git_mode_changed", mutate_file("git_mode", "100755")),
        ("manifest_missing_file", remove_file),
        ("manifest_extra_file", add_file),
        ("manifest_duplicate_package_path", duplicate_path),
        ("manifest_case_collision", case_collision),
        ("manifest_unsafe_relative_path", unsafe_path),
        ("manifest_symlink_mode", mutate_file("git_mode", "120000")),
        ("manifest_submodule_mode", mutate_file("git_mode", "160000")),
        ("manifest_extra_top_level_property", extra_top_level),
    ):
        cases.append(assert_rejects(name, func))
    return {"status": "MANIFEST_MUTATION_TESTS_PASS", "negative_tests": len(cases), "cases": cases}


def changed_top(source: dict[str, Any], key: str, value: Any) -> None:
    changed = copy.deepcopy(source)
    changed[key] = value
    adapter.validate_bundle_manifest(changed)


def authority_mutation_tests(manifest: dict[str, Any]) -> dict[str, Any]:
    reviewed_head = "1234567890abcdef1234567890abcdef12345678"
    review_id = 4618767711
    cases: list[str] = []
    equivalence_result: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="gate_a_authority_mutations_") as tmp:
        path = Path(tmp) / AUTHORITY_NAME

        def write_authority(value: dict[str, Any]) -> str:
            path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            return sha256_file(path)

        valid = authority_template(manifest, reviewed_head, review_id)
        digest = write_authority(valid)
        host_result, runner_result = validate_authority_both(path, digest, manifest, reviewed_head, review_id)
        require(host_result == runner_result, "host/target authority validation result mismatch")
        equivalence_result = host_result
        exact_manifest = bundle.validate_committed_manifest_exact(manifest, active_treeish())

        def reject_both(name: str, changed: dict[str, Any], expected_head: str = reviewed_head, expected_review_id: int = review_id, digest_override: str | None = None) -> str:
            actual_digest = write_authority(changed)
            use_digest = digest_override or actual_digest
            def host_case() -> None:
                validate_authority_host(path, use_digest, manifest, expected_head, expected_review_id)
            def runner_case() -> None:
                gate_a_target_runner.validate_authority(path, use_digest, target_args(manifest, expected_head, expected_review_id), exact_manifest)
            assert_rejects(name + ":host", host_case)
            assert_rejects(name + ":target", runner_case)
            return name

        def mutate(name: str, mutator: Callable[[dict[str, Any]], None], expected_head: str = reviewed_head, expected_review_id: int = review_id) -> None:
            changed = copy.deepcopy(valid)
            mutator(changed)
            cases.append(reject_both(name, changed, expected_head, expected_review_id))

        mutate("authority_extra_top_level_field", lambda v: v.__setitem__("extra", True))
        mutate("authority_missing_top_level_field", lambda v: v.pop("target"))
        mutate("authority_extra_state_field", lambda v: v["authority_state"].__setitem__("extra", True))
        mutate("authority_missing_state_field", lambda v: v["authority_state"].pop("hardware_ran"))
        mutate("authority_wrong_schema_id", lambda v: v.__setitem__("schema_id", "WRONG"))
        mutate("authority_wrong_reviewed_adapter_head", lambda v: v.__setitem__("reviewed_adapter_head", "0" * 40))
        mutate("authority_wrong_independent_review_id", lambda v: v.__setitem__("independent_review_id", review_id + 1))
        mutate("authority_review_id_zero", lambda v: v.__setitem__("independent_review_id", 0), expected_review_id=0)
        mutate("authority_wrong_bundle_digest", lambda v: v.__setitem__("execution_bundle_sha256", "0" * 64))
        mutate("authority_wrong_host_adapter_blob", lambda v: v.__setitem__("host_adapter_git_blob_sha1", "0" * 40))
        mutate("authority_wrong_target_runner_blob", lambda v: v.__setitem__("target_runner_git_blob_sha1", "0" * 40))
        mutate("authority_wrong_target_worker_blob", lambda v: v.__setitem__("target_worker_git_blob_sha1", "0" * 40))
        mutate("authority_wrong_schedule_digest", lambda v: v.__setitem__("schedule_sha256", "0" * 64))
        mutate("authority_wrong_namespace_digest", lambda v: v.__setitem__("target_namespace_sha256", "0" * 64))
        mutate("authority_wrong_target_identity_digest", lambda v: v.__setitem__("target_identity_sha256", "0" * 64))
        mutate("authority_wrong_target", lambda v: v.__setitem__("target", "root@127.0.0.1"))
        mutate("authority_wrong_remote_execution_root", lambda v: v.__setitem__("remote_execution_root", "/root/wrong"))
        mutate("authority_wrong_remote_output_root", lambda v: v.__setitem__("remote_output_root", "/root/wrong/evidence"))
        mutate("authority_maximum_execution_count_gt_one", lambda v: v.__setitem__("maximum_execution_count", 2))
        mutate("authority_consumed_true", lambda v: v.__setitem__("consumed", True))
        mutate("authority_project_owner_approved_false", lambda v: v.__setitem__("project_owner_approved", False))
        mutate("authority_artifact_created_false", lambda v: v["authority_state"].__setitem__("authorization_artifact_created", False))
        mutate("authority_engineering_smoke_authorized_false", lambda v: v["authority_state"].__setitem__("engineering_smoke_authorized", False))
        mutate("authority_hardware_ran_true", lambda v: v["authority_state"].__setitem__("hardware_ran", True))
        mutate("authority_automatic_retry_true", lambda v: v["authority_state"].__setitem__("automatic_retry", True))
        mutate("authority_calibration_authorized_true", lambda v: v["authority_state"].__setitem__("calibration_authorized", True))
        mutate("authority_scientific_acquisition_authorized_true", lambda v: v["authority_state"].__setitem__("scientific_acquisition_authorized", True))
        mutate("authority_restoration_authorized_true", lambda v: v["authority_state"].__setitem__("restoration_authorized", True))
        mutate("authority_target_coupling_authorized_true", lambda v: v["authority_state"].__setitem__("target_coupling_authorized", True))
        mutate("authority_small_wall_authorized_true", lambda v: v["authority_state"].__setitem__("small_wall_authorized", True))
        cases.append(reject_both("authority_file_sha256_mismatch", valid, digest_override="0" * 64))

    return {
        "status": "AUTHORITY_MUTATION_TESTS_PASS",
        "negative_tests": len(cases),
        "host_target_equivalence": equivalence_result,
        "cases": cases,
    }


def other_mutation_tests(ctx: adapter.AdapterContext, manifest: dict[str, Any]) -> dict[str, Any]:
    cases: list[str] = []
    canonical_authority = git_rel(AUTHORITY)
    alternate_authority = f"alternate/{AUTHORITY_NAME}"
    uppercase_extension_authority = "alternate/GATE_A_EXECUTION_AUTHORITY.JSON"
    require(
        validate_execution_authority_inventory([], [], canonical_authority) is False,
        "absent execution-authority inventory was not accepted",
    )
    require(
        validate_execution_authority_inventory(
            [canonical_authority],
            [canonical_authority],
            canonical_authority,
        ) is True,
        "canonical execution-authority inventory was not accepted",
    )
    for name, filesystem_hits, tracked_hits in (
        ("untracked_execution_authority_rejection", [canonical_authority], []),
        ("tracked_missing_execution_authority_rejection", [], [canonical_authority]),
        ("alternate_execution_authority_rejection", [alternate_authority], [alternate_authority]),
        ("uppercase_extension_execution_authority_rejection", [uppercase_extension_authority], []),
        (
            "sibling_execution_authority_rejection",
            [canonical_authority, alternate_authority],
            [canonical_authority],
        ),
    ):
        cases.append(assert_rejects(
            name,
            lambda filesystem_hits=filesystem_hits, tracked_hits=tracked_hits: validate_execution_authority_inventory(
                filesystem_hits,
                tracked_hits,
                canonical_authority,
            ),
        ))

    def wrong_schedule() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["slot_sequence"][6] = "I"
        adapter.validate_schedule(changed)

    def wrong_target() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["target"]["ssh_target"] = "root@127.0.0.1"
        adapter.validate_schedule(changed)

    def wrong_namespace() -> None:
        changed = copy.deepcopy(ctx.namespace)
        changed["binding_sha256"] = "0" * 64
        adapter.validate_namespace(changed)

    def off_or_sham_drive_mutation() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["slot_definitions"]["D0"]["executed"]["drive_on"] = True
        adapter.validate_schedule(changed)

    def step_sender_epoch_mutation() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["slot_definitions"]["S0E"]["executed"]["sender_epoch_id"] = "gate-a:step:epoch1"
        adapter.validate_schedule(changed)

    def extra_namespace_property() -> None:
        changed = copy.deepcopy(ctx.namespace)
        changed["extra"] = True
        adapter.validate_namespace(changed)

    def worktree_byte_mutation_behavior() -> None:
        target = HERE / "gate_a_worker.c"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n/* mutation */\n")
            bundle.assert_clean_for_paths([target])
        finally:
            target.write_bytes(original)

    def index_byte_mutation_detection() -> None:
        target = HERE / "gate_a_worker.c"
        original = target.read_bytes()
        mutated_blob = run(["git", "hash-object", "-w", "--stdin"], input_text=(original + b"\n/* staged mutation */\n").decode("utf-8")).stdout.strip()
        try:
            run(["git", "update-index", "--cacheinfo", "100644", mutated_blob, git_rel(target)], cwd=bundle.repo_root())
            changed = bundle.render_manifest(":")
            require(changed == manifest, "index mutation changed bundle identity")
        finally:
            run(["git", "add", git_rel(target)], cwd=bundle.repo_root())

    def cleanup_without_receipt() -> None:
        args = type("Args", (), {"copy_back_receipt": None})()
        gate_a_target_runner.cleanup_after_verified_copy(args)

    for name, func in (
        ("schedule_slot_sequence_rejection", wrong_schedule),
        ("schedule_target_rejection", wrong_target),
        ("namespace_digest_rejection", wrong_namespace),
        ("slot_drive_mutation_rejection", off_or_sham_drive_mutation),
        ("step_sender_epoch_mutation_rejection", step_sender_epoch_mutation),
        ("namespace_extra_property_rejection", extra_namespace_property),
        ("worktree_byte_mutation_behavior", worktree_byte_mutation_behavior),
        ("index_byte_mutation_detection", index_byte_mutation_detection),
        ("cleanup_without_copy_back_receipt_rejection", cleanup_without_receipt),
    ):
        cases.append(assert_rejects(name, func))

    for flag in (
        "--deploy",
        "--connect",
        "--start-sender",
        "--start-capture",
        "--write-control",
        "--cleanup-after-verified-copy",
    ):
        proc = run([sys.executable, str(HERE / "gate_a_hardware_adapter.py"), flag], check=False)
        require(proc.returncode != 0, f"adapter bypass accepted: {flag}")
        cases.append(f"authority_bypass_rejection:{flag}")
    return {"status": "OTHER_MUTATION_TESTS_PASS", "negative_tests": len(cases), "cases": cases}


def mutation_tests() -> dict[str, Any]:
    ctx = adapter.load_context()
    manifest = copy.deepcopy(ctx.manifest)
    manifest_result = manifest_mutation_tests(manifest)
    authority_result = authority_mutation_tests(manifest)
    other_result = other_mutation_tests(ctx, manifest)
    total = manifest_result["negative_tests"] + authority_result["negative_tests"] + other_result["negative_tests"]
    return {
        "status": "MUTATION_TESTS_PASS",
        "negative_tests": total,
        "manifest": manifest_result,
        "authority": authority_result,
        "other": other_result,
    }


def isolated_bundle_qualification() -> dict[str, Any]:
    compile_c = shutil.which("cc") is not None
    with tempfile.TemporaryDirectory(prefix="gate_a_extract_") as tmp:
        root = Path(tmp) / "bundle"
        bundle.write_extracted_tree(root, active_treeish())
        require(not (root / ".git").exists(), "extracted bundle must not contain .git")
        report = isolated_harness.build_isolated_report(root, require_isolated_origin=False, compile_c=compile_c)
    require(report["status"] == "GATE_A_ISOLATED_BUNDLE_QUALIFICATION_PASS", "isolated bundle qualification failed")
    report["c_compiler_present"] = compile_c
    return report


def validate_records() -> dict[str, Any]:
    treeish = active_treeish()
    manifest = load(MANIFEST)
    result = load(RESULT)
    candidate = load(CANDIDATE_V2)
    schema = load(AUTHORITY_SCHEMA)
    contract = load(CONTRACT)
    require(set(manifest) == bundle.MANIFEST_KEYS, "manifest record key set mismatch")
    bundle.validate_committed_manifest_exact(manifest, treeish)
    validate_schema_closed(schema)
    validate_contract(contract)
    require(set(candidate) == {
        "schema_id",
        "status",
        "base_main_commit",
        "reviewed_gate_a_plan_head",
        "gate_a_plan_review",
        "plan_reviewed",
        "adapter_implemented",
        "hosted_adapter_qualification_complete",
        "target_adapter_qualification_complete",
        "execution_bundle_ready",
        "execution_bundle_target_qualified",
        "project_owner_execution_approval_recorded",
        "authorization_artifact_created",
        "engineering_smoke_authorized",
        "hardware_ran",
        "schedule_sha256",
        "target_namespace_sha256",
        "target_identity_stdout_sha256",
        "host_adapter_git_blob_sha1",
        "target_runner_git_blob_sha1",
        "target_worker_git_blob_sha1",
        "execution_bundle_sha256",
        "deterministic_archive_sha256",
        "bundle_manifest_sha256",
        "next_boundary",
        "authority_false_state",
    }, "candidate V2 key set mismatch")
    require(set(result) == {
        "schema_id",
        "status",
        "adapter_implementation_complete",
        "hosted_nonexecuting_qualification_complete",
        "target_nonexecuting_qualification_complete",
        "execution_bundle_ready",
        "execution_bundle_target_qualified",
        "authority_artifact_created",
        "engineering_smoke_authorized",
        "hardware_ran",
        "no_target_connection_occurred",
        "no_ssh_occurred",
        "no_sender_ran",
        "no_receiver_capture_ran",
        "no_control_write_occurred",
        "next_boundary",
        "execution_bundle_sha256",
        "deterministic_archive_sha256",
        "bundle_manifest_sha256",
        "authority_false_state",
    }, "qualification result key set mismatch")
    manifest_digest = committed_sha256(MANIFEST, treeish)
    result_digest = committed_sha256(RESULT, treeish)
    candidate_digest = committed_sha256(CANDIDATE_V2, treeish)
    historical_manifest = committed_object(MANIFEST, HISTORICAL_ADAPTER_COMMIT)
    historical_manifest_digest = committed_sha256(MANIFEST, HISTORICAL_ADAPTER_COMMIT)
    require(historical_manifest_digest == HISTORICAL_MANIFEST_SHA256, "historical manifest changed")
    require(result_digest == HISTORICAL_RESULT_SHA256, "historical qualification result bytes changed")
    require(candidate_digest == HISTORICAL_CANDIDATE_V2_SHA256, "historical Candidate V2 bytes changed")
    require(result["execution_bundle_sha256"] == historical_manifest["execution_bundle_sha256"], "historical result bundle binding mismatch")
    require(result["deterministic_archive_sha256"] == historical_manifest["deterministic_archive_sha256"], "historical result archive binding mismatch")
    require(result["bundle_manifest_sha256"] == historical_manifest_digest, "historical result manifest binding mismatch")
    require(candidate["execution_bundle_sha256"] == historical_manifest["execution_bundle_sha256"], "historical candidate bundle binding mismatch")
    require(candidate["deterministic_archive_sha256"] == historical_manifest["deterministic_archive_sha256"], "historical candidate archive binding mismatch")
    require(candidate["bundle_manifest_sha256"] == historical_manifest_digest, "historical candidate manifest binding mismatch")
    historical_roles = {entry["role"]: entry for entry in historical_manifest["files"]}
    require(candidate["host_adapter_git_blob_sha1"] == historical_roles["host_adapter"]["git_blob_sha1"], "historical candidate adapter blob mismatch")
    require(candidate["target_runner_git_blob_sha1"] == historical_roles["target_runner"]["git_blob_sha1"], "historical candidate runner blob mismatch")
    require(candidate["target_worker_git_blob_sha1"] == historical_roles["target_worker"]["git_blob_sha1"], "historical candidate worker blob mismatch")
    require(manifest["engineering_smoke_executor_implemented"] is True, "executor implementation status mismatch")
    require(manifest["execution_bundle_target_qualified"] is True, "target bundle qualification status mismatch")
    require(manifest["engineering_smoke_authorized"] is False, "engineering smoke authority changed")
    require(manifest["hardware_ran"] is False, "hardware run state changed")
    require(result["status"] == "TARGET_NONEXECUTING_QUALIFICATION_REQUIRED", "result boundary mismatch")
    require(candidate["status"] == "CANDIDATE__BLOCKED_PENDING_TARGET_NONEXECUTING_QUALIFICATION", "candidate status mismatch")
    for key, value in result["authority_false_state"].items():
        require(value is False, f"result authority flag must be false: {key}")
    execution_authority = validate_execution_authority_state(manifest)
    current_engineering_smoke = validate_engineering_smoke_evidence_state()
    require(execution_authority["authority_artifact_present"] is True, "consumed smoke authority input missing")
    require(execution_authority["consumed"] is False, "historical authority input consumed field changed")
    return {
        "status": "RECORDS_VALID",
        "manifest_sha256": manifest_digest,
        "result_sha256": result_digest,
        "candidate_v2_sha256": candidate_digest,
        "historical_manifest_sha256": historical_manifest_digest,
        "historical_records_immutable": True,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "execution_authority": execution_authority,
        "current_engineering_smoke": current_engineering_smoke,
    }


def focused_executor_tests() -> dict[str, Any]:
    completed = run([
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        str(HERE),
        "-p",
        "test_gate_a_engineering_smoke_executor.py",
        "-v",
    ], cwd=bundle.repo_root(), check=False)
    require(completed.returncode == 0, f"focused executor tests failed:\n{completed.stdout}\n{completed.stderr}")
    count = sum(1 for line in completed.stderr.splitlines() if line.rstrip().endswith("... ok"))
    require(count >= 30, "focused executor test count below required minimum")
    return {
        "status": "GATE_A_ENGINEERING_SMOKE_EXECUTOR_TESTS_PASS",
        "tests_run": count,
        "network_connections_opened": 0,
        "target_contact_count": 0,
        "hardware_execution_count": 0,
    }


def main() -> int:
    context = adapter.load_context()
    no_drive = adapter.qualify_no_drive(context)
    require(no_drive["transport"] == "NO_DRIVE", "adapter transport not no-drive")
    treeish = active_treeish()
    manifest_a = bundle.render_manifest(treeish)
    manifest_b = bundle.render_manifest(treeish)
    require(manifest_a == manifest_b, "bundle double-build mismatch")
    records = validate_records()
    mutations = mutation_tests()
    isolated = isolated_bundle_qualification()
    scan = static_forbidden_surface_scan()
    executor_tests = focused_executor_tests()
    runtime_path = HERE.parents[2] / "runtime" / "explicit_slot_runtime.py"
    require("SOFTWARE_ENTRY_ONLY_AUTHORITY: real hardware execution is not authorized" in runtime_path.read_text(encoding="utf-8"), "runtime hardware rejection marker missing")
    total_negative_tests = mutations["negative_tests"] + isolated["isolated_negative_tests"]
    result = {
        "status": "GATE_A_ADAPTER_QUALIFICATION_PASS",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "adapter_no_drive": no_drive["status"],
        "bundle_double_build_equivalence": True,
        "mutation_tests": mutations,
        "isolated_bundle_qualification": isolated,
        "total_negative_tests": total_negative_tests,
        "forbidden_surface_scan": scan,
        "executor_tests": executor_tests,
        "records": records,
        "authority_artifact_present": records["execution_authority"]["authority_artifact_present"],
        "execution_authority": records["execution_authority"],
        "current_engineering_smoke": records["current_engineering_smoke"],
        "engineering_smoke_attempted": records["current_engineering_smoke"]["smoke_attempted"],
        "authority_consumed": records["current_engineering_smoke"]["authority_consumed"],
        "engineering_smoke_authorized": records["current_engineering_smoke"]["engineering_smoke_authorized"],
        "hardware_ran": records["current_engineering_smoke"]["hardware_ran"],
        "verification_target_connection_count": 0,
        "sender_start_count": 0,
        "control_write_count": 0,
        "msr_access_count": 0,
        "hardware_execution_count": 0,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        VerifyError,
        adapter.AdapterError,
        adapter.gate_a_authority.AuthorityError,
        bundle.BundleError,
        process_custody.ProcessCustodyError,
        smoke_transport.TransportError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"verify_gate_a_adapter_qualification: {exc}", file=sys.stderr)
        raise SystemExit(1)
