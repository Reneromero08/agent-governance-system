#!/usr/bin/env python3
"""Verify the sealed second Gate A attempt and consumed-authority state.

This qualification-only verifier opens no network connection and contains no
execution surface. It closes the authority namespace, exact failed-closed
packet, and protected-source immutability after the 800 MHz frequency veto.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import build_gate_a_execution_bundle as bundle
import gate_a_engineering_smoke_transport as transport

HERE = Path(__file__).resolve().parent
REPO_ROOT = bundle.repo_root().resolve()
REVIEWED_SOURCE = "040b80fbb10c6a8fa63bb590a86c6dc8d4ff4d59"
AUTHORITY_COMMIT = "06f9604f3811034d304413980c71d97ac09ca5f3"
SEALED_EVIDENCE_COMMIT = "e023f3d319c5aa0e7434533d3acccfa0fa475e9b"
SEALED_EVIDENCE_TREE = "4ae7c2331cdc5fe4466cb09f978e67535b868169"

ACTIVE_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY.json"
FIRST_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY_CONSUMED_7e1e8835.json"
SECOND_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY_CONSUMED_1dabfc7b.json"
AUTHORITY_SCHEMA = HERE / "schemas" / "gate_a_execution_authority.schema.json"

FIRST_AUTHORITY_SHA256 = "7e1e8835bd67590e4e554ae112a2c8a6ca99dd8b9b3a9aafdb23fee31907d682"
FIRST_AUTHORITY_BLOB = "709c799f60e30984d3c80715af480fbe5deac952"
SECOND_AUTHORITY_SHA256 = "1dabfc7bbfc65e988542b0c4580f031309c5abfc53521dd849e5c0ac71e24fd4"
SECOND_AUTHORITY_BLOB = "b69b1fa6c7d2c710c76fac0115c2227006fd2212"
EXECUTION_BUNDLE_SHA256 = "353f7e2d865508ebc018cb72648d3d3f227dc1c1128681fd9b4e99d81c9aa47f"

EVIDENCE_ROOT = HERE.parents[2] / "evidence" / "gate_a_engineering_smoke_1dabfc7b"
FIRST_SMOKE_ROOT = HERE.parents[2] / "evidence" / "gate_a_engineering_smoke_7e1e8835"
ORIGINAL_TARGET_ROOT = HERE.parents[2] / "evidence" / "gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a"
REPLACEMENT_TARGET_ROOT = HERE.parents[2] / "evidence" / "gate_a_target_nonexecuting_qualification_replacement_gate_a_replacement_593e9920_02"
HISTORICAL_TREES = {
    FIRST_SMOKE_ROOT: "287be635034e89b2df85f93cad5506a81762268b",
    ORIGINAL_TARGET_ROOT: "a02edbfb85bb2b1816a3be92089112f29c639da9",
    REPLACEMENT_TARGET_ROOT: "a5781e637d878b067f505c0485beddda9ccd6893",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: Path, treeish: str = "HEAD") -> str:
    return run("git", "rev-parse", f"{treeish}:{rel(path)}").stdout.strip()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_authority_archive(path: Path, expected_sha256: str, expected_blob: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"authority archive missing: {path.name}")
    require(sha256(path) == expected_sha256, f"authority SHA-256 mismatch: {path.name}")
    require(git_blob(path) == expected_blob, f"authority committed blob mismatch: {path.name}")
    require(run("git", "hash-object", str(path)).stdout.strip() == expected_blob, f"authority worktree blob mismatch: {path.name}")
    value = load_object(path)
    require(value["maximum_execution_count"] == 1, f"maximum execution count drift: {path.name}")
    require(value["consumed"] is False, f"historical authority input changed: {path.name}")
    require(value["authority_state"]["automatic_retry"] is False, f"retry state drift: {path.name}")
    return {"path": rel(path), "sha256": expected_sha256, "git_blob_sha1": expected_blob}


def verify_authority_namespace() -> dict[str, Any]:
    require(not ACTIVE_AUTHORITY.exists(), "canonical active authority path must be absent")
    first = verify_authority_archive(FIRST_AUTHORITY, FIRST_AUTHORITY_SHA256, FIRST_AUTHORITY_BLOB)
    second = verify_authority_archive(SECOND_AUTHORITY, SECOND_AUTHORITY_SHA256, SECOND_AUTHORITY_BLOB)
    status = run(
        "git", "status", "--porcelain=v1", "--untracked-files=all", "--",
        ":(icase,glob)**/gate_a_execution_authority*.json",
        f":(exclude){rel(AUTHORITY_SCHEMA)}",
        check=False,
    )
    require(status.returncode == 0 and status.stdout == "", "execution-authority namespace differs from HEAD")
    tracked = sorted(
        line for line in run("git", "ls-files").stdout.splitlines()
        if Path(line).name.casefold().startswith("gate_a_execution_authority")
        and Path(line).name.casefold().endswith(".json")
        and line != rel(AUTHORITY_SCHEMA)
    )
    require(tracked == sorted([rel(FIRST_AUTHORITY), rel(SECOND_AUTHORITY)]), f"authority tracked set mismatch: {tracked}")
    return {"status": "TWO_CONSUMED_AUTHORITIES_ARCHIVED_EXACT", "archives": [first, second]}


def verify_evidence_packet() -> dict[str, Any]:
    evidence_rel = rel(EVIDENCE_ROOT)
    require(EVIDENCE_ROOT.is_dir() and not EVIDENCE_ROOT.is_symlink(), "second evidence root missing")
    require(run("git", "merge-base", "--is-ancestor", AUTHORITY_COMMIT, SEALED_EVIDENCE_COMMIT, check=False).returncode == 0, "authority is not ancestor of sealed evidence")
    sealed_tree = run("git", "rev-parse", f"{SEALED_EVIDENCE_COMMIT}:{evidence_rel}").stdout.strip()
    current_tree = run("git", "rev-parse", f"HEAD:{evidence_rel}").stdout.strip()
    require(sealed_tree == SEALED_EVIDENCE_TREE, "sealed second-evidence tree mismatch")
    require(current_tree == SEALED_EVIDENCE_TREE, "current second-evidence tree changed")
    require(run("git", "diff", "--quiet", SEALED_EVIDENCE_COMMIT, "HEAD", "--", evidence_rel, check=False).returncode == 0, "second evidence changed after sealing")
    status = run("git", "status", "--porcelain=v1", "--untracked-files=all", "--", evidence_rel, check=False)
    require(status.returncode == 0 and status.stdout == "", "second evidence differs from HEAD")

    packet = transport.validate_final_packet(EVIDENCE_ROOT)
    require(packet["status"] == "GATE_A_FINAL_PACKET_VALID", "final packet validator failed")
    bindings = load_object(EVIDENCE_ROOT / "FINAL_BINDINGS.json")
    require(bindings["status"] == "GATE_A_AUTHORIZED_TRANSPORT_FAILED_CLOSED", "final status mismatch")
    require(bindings["failed_stage"] == "target_result_verification", "terminal stage mismatch")
    require(bindings["primary_error"] == "authorized runtime failed after evidence custody", "primary error mismatch")
    require(bindings["authority_sha256"] == SECOND_AUTHORITY_SHA256, "final authority binding mismatch")
    require(bindings["authority_bearing_execution_commit"] == AUTHORITY_COMMIT, "authority commit binding mismatch")
    require(bindings["reviewed_source_commit"] == REVIEWED_SOURCE, "reviewed source binding mismatch")
    require(bindings["execution_bundle_sha256"] == EXECUTION_BUNDLE_SHA256, "bundle binding mismatch")
    require(bindings["transport_execution_count"] == 1 and bindings["runner_start_count"] == 1, "execution count mismatch")
    require(bindings["retry_count"] == 0 and bindings["automatic_retry"] is False, "retry boundary mismatch")

    attempt = load_object(EVIDENCE_ROOT / "TARGET_OUTPUT" / "ATTEMPT.json")
    require(attempt["authority_sha256"] == SECOND_AUTHORITY_SHA256, "attempt authority mismatch")
    require(attempt["maximum_execution_count"] == 1 and attempt["automatic_retry"] is False, "attempt retry boundary mismatch")
    require(attempt["preflight"] == {
        "frequency_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "namespace_state": "absent",
        "preflight_complete": False,
        "voltage_writes": 0,
    }, "attempt preflight mismatch")

    failure = load_object(EVIDENCE_ROOT / "TARGET_OUTPUT" / "FAILURE.json")
    require(failure == {
        "automatic_retry": False,
        "partial_evidence_preserved": True,
        "reason": "frequency veto",
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FAILURE_V1",
    }, "frequency failure receipt mismatch")

    events = [json.loads(line) for line in (EVIDENCE_ROOT / "TARGET_OUTPUT" / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
    require([event["event"] for event in events] == [
        "pre_runtime_process_scan", "temperature_preflight", "frequency_preflight", "runtime_failed"
    ], "event sequence mismatch")
    require(not any(event["event"] == "runtime_start" for event in events), "physical runtime unexpectedly started")
    temperature = events[1]
    require(temperature["observation_complete"] is True and temperature["veto_passed"] is True, "temperature preflight incomplete")
    require(temperature["temperature_c"] == 38.5, "temperature observation mismatch")
    require(events[2]["frequency_khz"] == {"4": 800000, "5": 800000}, "frequency veto observation mismatch")
    require(events[3]["reason"] == "frequency veto", "runtime failure reason mismatch")

    temperature_receipt = load_object(EVIDENCE_ROOT / "TARGET_OUTPUT" / "TEMPERATURE_PREFLIGHT_RECEIPT.json")
    require(temperature_receipt["selected_driver_name"] == "k10temp", "temperature driver mismatch")
    require(temperature_receipt["normalized_temperature_c"] == 38.5, "temperature receipt mismatch")
    require(temperature_receipt["veto_temperature_c"] == 68.0 and temperature_receipt["veto_passed"] is True, "temperature veto mismatch")

    cleanup = load_object(EVIDENCE_ROOT / "CLEANUP_RECEIPT.json")["parsed"]
    require(cleanup["cleanup_return_code"] == 0 and cleanup["cleanup_mode"] == "verified_copyback", "cleanup result mismatch")
    require(cleanup["claim_retained"] is True, "durable claim not retained")
    for key in ("execution_root_absent", "output_root_absent", "stage_absent", "authority_absent", "archive_absent", "receipt_absent", "target_inventory_absent"):
        require(cleanup[key] is True, f"cleanup absence proof failed: {key}")

    copy_back = load_object(EVIDENCE_ROOT / "COPY_BACK_RECEIPT.json")
    require(copy_back["copy_back_complete"] is True and copy_back["retained_evidence_custody_verified"] is True, "copy-back custody mismatch")
    require(copy_back["evidence_inventory_sha256"] == copy_back["downloaded_evidence_inventory_sha256"] == copy_back["target_evidence_inventory_sha256"], "copy-back inventory mismatch")

    for name, phase in (("POST_RUNTIME_PROCESS_RECEIPT.json", "post_runtime"), ("POST_CLEANUP_PROCESS_RECEIPT.json", "post_cleanup")):
        receipt = load_object(EVIDENCE_ROOT / name)
        require(receipt["phase"] == phase and receipt["scan_complete"] is True, f"process receipt incomplete: {phase}")
        require(receipt["parsed_forbidden_hits"] == [] and receipt["failure"] is None, f"forbidden process hit: {phase}")

    return {
        "status": "SECOND_GATE_A_ATTEMPT_VALID_FAILED_CLOSED_FREQUENCY_VETO",
        "sealed_commit": SEALED_EVIDENCE_COMMIT,
        "tree_sha1": SEALED_EVIDENCE_TREE,
        "packet": packet,
        "temperature_c": 38.5,
        "frequency_khz": {"4": 800000, "5": 800000},
        "runtime_execution_count": 0,
        "retry_count": 0,
    }


def verify_protected_sources() -> dict[str, Any]:
    require(run("git", "merge-base", "--is-ancestor", REVIEWED_SOURCE, "HEAD", check=False).returncode == 0, "reviewed source is not an ancestor")
    protected = tuple(sorted({
        *(bundle.rel(source) for _package, source, _role in bundle.PACKAGE_FILES),
        bundle.rel(bundle.MANIFEST_PATH),
    }))
    require(run("git", "diff", "--quiet", REVIEWED_SOURCE, "HEAD", "--", *protected, check=False).returncode == 0, "protected execution source drifted")
    manifest = load_object(bundle.MANIFEST_PATH)
    exact = bundle.validate_committed_manifest_exact(manifest, "HEAD")
    require(exact["execution_bundle_sha256"] == EXECUTION_BUNDLE_SHA256, "execution bundle digest changed")
    return {"status": "PROTECTED_EXECUTION_SOURCE_BYTE_IDENTICAL", "path_count": len(protected), "execution_bundle_sha256": EXECUTION_BUNDLE_SHA256}


def verify_historical_evidence() -> dict[str, Any]:
    values: dict[str, str] = {}
    for path, expected in HISTORICAL_TREES.items():
        current = run("git", "rev-parse", f"HEAD:{rel(path)}").stdout.strip()
        require(current == expected, f"historical evidence tree changed: {path.name}")
        values[path.name] = expected
    return {"status": "HISTORICAL_EVIDENCE_TREES_EXACT", "trees": values}


def main() -> int:
    report = {
        "status": "GATE_A_SECOND_ATTEMPT_CLOSURE_VALID",
        "head": run("git", "rev-parse", "HEAD").stdout.strip(),
        "authorities": verify_authority_namespace(),
        "second_attempt": verify_evidence_packet(),
        "protected_sources": verify_protected_sources(),
        "historical_evidence": verify_historical_evidence(),
        "active_authority_created": False,
        "target_contacts_during_verification": 0,
        "hardware_executions_during_verification": 0,
        "next_boundary": "GATE_A_FREQUENCY_OBSERVATION_CONTRACT_REVIEW",
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerifyError, transport.TransportError, bundle.BundleError, subprocess.CalledProcessError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"verify_gate_a_second_attempt_closure: {exc}", file=sys.stderr)
        raise SystemExit(1)
