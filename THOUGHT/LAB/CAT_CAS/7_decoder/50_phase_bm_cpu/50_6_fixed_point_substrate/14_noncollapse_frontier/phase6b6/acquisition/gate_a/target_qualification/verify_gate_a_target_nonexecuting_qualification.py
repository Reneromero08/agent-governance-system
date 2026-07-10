#!/usr/bin/env python3
"""Verify the preserved Gate A attempt and its fail-closed adjudication.

This verifier is local-only.  It never opens SSH/SCP or contacts the target.  A
green result means the historical packet is byte-for-byte unchanged, the one
historical attempt is still represented accurately, the process-evidence gaps
are detected, the old completion interpretation is superseded, and every
current or downstream authority field remains false.
"""

from __future__ import annotations

import base64
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
from unittest import mock

import run_target_nonexecuting_qualification as future_runner

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
ADAPTER = GATE_A / "adapter"
PHASE6B6 = HERE.parents[2]
REPO_ROOT = PHASE6B6.parents[7]
EVID = PHASE6B6 / "evidence" / "gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a"
EVID_REL = EVID.relative_to(REPO_ROOT).as_posix()

AUTHORIZATION = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
CONTRACT = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_CONTRACT.json"
RESULT = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT.json"
CANDIDATE_V3 = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3.json"
ADJUDICATION = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_ADJUDICATION.json"
CANDIDATE_V4 = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V4.json"

RESULT_SCHEMA = HERE / "schemas" / "gate_a_target_nonexecuting_qualification_result.schema.json"
ADJUDICATION_SCHEMA = HERE / "schemas" / "gate_a_target_nonexecuting_qualification_adjudication.schema.json"
CANDIDATE_V4_SCHEMA = HERE / "schemas" / "gate_a_engineering_smoke_authority_candidate_v4.schema.json"

PRE_REPAIR_HEAD = "310efd0ec4103654b122a961b001bc5b79cf5896"
INTEGRATED_MAIN = "6f243b1aaf7cfaa09f21b8d5816ddd9097612f72"
REVIEWED_HEAD = "653d225594af088dc5a72b1655b8b6192b019fc3"
REVIEW_ID = 4621557139
EXECUTION_BUNDLE = "abc9e50a517d764c553adc5096378992028b29a8f62480a9ae217ebbd5202bba"
DETERMINISTIC_ARCHIVE = "04eaf73336f373865f4e837baca9ff4fe893d3b1b16dd8b8288af1259ff96f9c"
MANIFEST_FILE_SHA = "ccb7866db67170083cb00d546c334b61772c8ef909131ec9c62ed21115facc94"
PREDECESSOR_ID_SHA = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"
EXPECTED_HOSTNAME = "catcas"
EXPECTED_ARCH = "x86_64"
EXPECTED_CPU = "AMD Phenom(tm) II X6 1090T Processor"
EXPECTED_OWNER = "Raúl Romero"
PREDECESSOR_RESULT_SHA = "1d9d2c62cbf81f72eeb9c40f02841f9f507d52eae8229da73fc2f81eb0a15223"
PREDECESSOR_CANDIDATE_V2_SHA = "d8f190bc7f8c9904659cd697ed091b192843efe18f5f1d74d713282e889b060e"
PREDECESSOR_DIGEST_MEANING = (
    "PR #36 adapter-package result and Candidate V2 digests, not PR #37 target qualification result "
    "or Candidate V3 committed file hashes"
)
CURRENT_STATUS = (
    "TARGET_NONEXECUTING_QUALIFICATION_ATTEMPT_PRESERVED__FAIL_CLOSED_PROCESS_ABSENCE_NOT_PROVEN"
)
NEXT_BOUNDARY = "PROJECT_OWNER_DECISION_FOR_ONE_REPLACEMENT_GATE_A_TARGET_NONEXECUTING_QUALIFICATION"

DIGEST_SEMANTICS_NAME = "DIGEST_SEMANTICS.json"

EXPECTED_BLOBS = {
    "gate_a_hardware_adapter.py": "08ea193e341f2d1934fe1955f224b2566c5945e5",
    "gate_a_target_runner.py": "7a17a6a514944888b594d37389e7d6105fbe97c4",
    "gate_a_worker.c": "a1022c0109e0c4406d84f19a9777d8a0008696fb",
    "gate_a_target_bundle.py": "369613f6153c71dddba9aaf1e4086a04519d8a37",
    "gate_a_authority.py": "13d657941e20919281a18c4d31dbd174addcfae3",
    "build_gate_a_execution_bundle.py": "2d67bf6e8ce293ddc12cfe562302d7b00619cff7",
    "verify_gate_a_adapter_qualification.py": "62817afd128fcc2d6a9f25c654fca3174b2a5643",
    "gate_a_isolated_qualification.py": "1dbeb06a6dceb6f1155d447835b3ef265571d19d",
}

FORBIDDEN_COMMAND_SUBSTRINGS = [
    "--probe-only",
    "--execute-authorized",
    "--cleanup-after-verified-copy",
    "rdmsr",
    "wrmsr",
    "cpupower",
    "turbostat",
    "modprobe",
    "insmod",
    "rmmod",
    "apt-get",
    "apt ",
    "combined_pdn_runner",
    "run_combined_campaign",
    "--hardware",
    " git ",
    "git clone",
    "git checkout",
    "git bundle",
]

DOWNSTREAM_FALSE_FIELDS = (
    "engineering_smoke_authorized",
    "hardware_ran",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
)


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


def git_blob(path: Path) -> str:
    proc = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    return proc.stdout.strip()


def validate_schema_closed(schema: dict[str, Any], nested_keys: tuple[str, ...] = ()) -> None:
    require(schema["type"] == "object", "schema top level is not object")
    require(schema["additionalProperties"] is False, "schema top level open")
    require(set(schema["required"]) == set(schema["properties"]), "schema required/properties mismatch")
    for key in nested_keys:
        sub = schema["properties"][key]
        require(sub["type"] == "object", f"schema {key} is not object")
        require(sub["additionalProperties"] is False, f"schema {key} open")
        require(set(sub["required"]) == set(sub["properties"]), f"schema {key} required/properties mismatch")


def validate_const_instance(value: Any, schema: dict[str, Any], context: str) -> None:
    if "const" in schema:
        require(value == schema["const"], f"{context} const mismatch")
        return
    if schema.get("type") == "object":
        require(isinstance(value, dict), f"{context} must be object")
        properties = schema["properties"]
        require(set(value) == set(properties), f"{context} key closure mismatch")
        for key, sub in properties.items():
            validate_const_instance(value[key], sub, f"{context}.{key}")
        return
    if schema.get("type") == "string":
        require(isinstance(value, str), f"{context} must be string")
    elif schema.get("type") == "integer":
        require(isinstance(value, int) and not isinstance(value, bool), f"{context} must be integer")
    elif schema.get("type") == "array":
        require(isinstance(value, list), f"{context} must be array")


def assert_historical_evidence_immutable() -> dict[str, Any]:
    commands = [
        ["git", "diff", "--exit-code", f"{PRE_REPAIR_HEAD}...HEAD", "--", EVID_REL],
        ["git", "diff", "--exit-code", "--", EVID_REL],
        ["git", "diff", "--cached", "--exit-code", "--", EVID_REL],
    ]
    labels = ("committed_range", "worktree", "index")
    proof: dict[str, Any] = {
        "baseline_head": PRE_REPAIR_HEAD,
        "historical_evidence_dir": EVID_REL,
    }
    for label, argv in zip(labels, commands, strict=True):
        proc = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(
            proc.returncode == 0,
            f"historical evidence changed ({label}): {(proc.stdout + proc.stderr)[:500]}",
        )
        proof[f"{label}_exit_code"] = proc.returncode
    proof["byte_for_byte_unchanged"] = True
    return proof


def validate_historical_authorization(auth: dict[str, Any]) -> None:
    require(auth["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION_V1", "historical authorization schema mismatch")
    require(auth["decision"] == "AUTHORIZED_FOR_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_ONLY", "historical authorization decision mismatch")
    require(auth["project_owner"] == EXPECTED_OWNER, "historical authorization owner identity mismatch")
    require(auth["integrated_main"] == INTEGRATED_MAIN, "historical authorization integrated main mismatch")
    require(auth["maximum_target_qualification_executions"] == 1, "historical authorization max executions must be 1")
    require(auth["automatic_retry"] is False, "historical authorization retry must be false")
    for key in ("ssh_authorized", "copy_authorized", "target_filesystem_staging_authorized", "compile_validate_only_authorized", "no_drive_runner_authorized"):
        require(auth[key] is True, f"historical authorization {key} must be true")
    for key in (
        "probe_authorized",
        "engineering_smoke_authorized",
        "hardware_execution_authorized",
        "calibration_authorized",
        "scientific_acquisition_authorized",
        "restoration_authorized",
        "target_coupling_authorized",
        "small_wall_authorized",
        "execution_authority_artifact_creation_authorized",
    ):
        require(auth[key] is False, f"historical authorization {key} must be false")


def validate_predecessor_adapter_package_digests(digests: dict[str, Any], context: str) -> None:
    require(set(digests) == {
        "predecessor_adapter_qualification_result_sha256",
        "predecessor_candidate_v2_sha256",
        "meaning",
    }, f"{context} predecessor digests open")
    require(digests["predecessor_adapter_qualification_result_sha256"] == PREDECESSOR_RESULT_SHA, f"{context} predecessor adapter result digest mismatch")
    require(digests["predecessor_candidate_v2_sha256"] == PREDECESSOR_CANDIDATE_V2_SHA, f"{context} predecessor Candidate V2 digest mismatch")
    require(digests["meaning"] == PREDECESSOR_DIGEST_MEANING, f"{context} predecessor digest meaning mismatch")


def validate_historical_contract(contract: dict[str, Any]) -> None:
    require(contract["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_CONTRACT_V1", "historical contract schema mismatch")
    require(contract["integrated_main"] == INTEGRATED_MAIN, "historical contract integrated main mismatch")
    require(contract["reviewed_adapter_head"] == REVIEWED_HEAD, "historical contract reviewed head mismatch")
    require(contract["maximum_target_qualification_executions"] == 1, "historical contract max executions")
    require(contract["automatic_retry"] is False, "historical contract retry")
    require(contract["git_on_target_forbidden"] is True, "historical contract git-on-target must be forbidden")
    blobs = contract["integrated_blobs"]
    require(blobs["host_adapter"] == EXPECTED_BLOBS["gate_a_hardware_adapter.py"], "contract host adapter blob mismatch")
    require(blobs["target_runner"] == EXPECTED_BLOBS["gate_a_target_runner.py"], "contract target runner blob mismatch")
    require(blobs["target_worker"] == EXPECTED_BLOBS["gate_a_worker.c"], "contract target worker blob mismatch")


HISTORICAL_CONST_TRUE = (
    "project_owner_target_qualification_approval_recorded",
    "target_connection_occurred",
    "ssh_occurred",
    "bundle_transferred",
    "bundle_transfer_verified",
    "target_identity_verified",
    "strict_bundle_validation_before",
    "target_no_drive_qualification_complete",
    "worker_validate_only_complete",
    "strict_bundle_validation_after",
    "bundle_tree_unchanged",
    "copy_back_verified",
    "cleanup_verified",
    "target_nonexecuting_qualification_complete",
    "execution_bundle_target_qualified",
)
HISTORICAL_CONST_FALSE = (
    "project_owner_execution_approval_recorded",
    "authorization_artifact_created",
    "engineering_smoke_authorized",
    "hardware_ran",
    "automatic_retry",
)
HISTORICAL_CONST_ZERO = (
    "probe_count",
    "execute_authorized_count",
    "network_connection_count",
    "hardware_probe_count",
    "sender_start_count",
    "receiver_capture_count",
    "control_write_count",
    "msr_access_count",
    "hardware_execution_count",
)


def validate_historical_result(result: dict[str, Any]) -> None:
    """Validate the preserved pre-adjudication record without promoting it."""
    require(result["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT_V1", "historical result schema mismatch")
    require(result["status"] == "TARGET_NONEXECUTING_QUALIFICATION_COMPLETE__ENGINEERING_SMOKE_STILL_UNAUTHORIZED", "historical result status mismatch")
    require(result["integrated_main"] == INTEGRATED_MAIN, "historical result integrated main mismatch")
    require(result["reviewed_adapter_head"] == REVIEWED_HEAD, "historical result reviewed head mismatch")
    require(result["review_id"] == REVIEW_ID, "historical result review id mismatch")
    require(result["target_qualification_execution_count"] == 1, "historical result execution count must be 1")
    for key in HISTORICAL_CONST_TRUE:
        require(result[key] is True, f"historical result {key} must be true")
    for key in HISTORICAL_CONST_FALSE:
        require(result[key] is False, f"historical result {key} must be false")
    for key in HISTORICAL_CONST_ZERO:
        require(result[key] == 0, f"historical result {key} must be 0")
    target = result["target"]
    require(target["ssh_target"] == "root@192.168.137.100", "historical result target mismatch")
    require(target["hostname"] == EXPECTED_HOSTNAME, "historical result hostname mismatch")
    require(target["architecture"] == EXPECTED_ARCH, "historical result arch mismatch")
    require(target["cpu_model"] == EXPECTED_CPU, "historical result cpu mismatch")
    require(target["remote_execution_root"] == "/root/catcas_phase6b6_gate_a_smoke_9c416379", "historical result exec root mismatch")
    require(target["execution_root_final_state"] == "ABSENT", "historical result exec root not absent")
    require(target["transfer_stage_final_state"] == "ABSENT", "historical result transfer stage not absent")
    bindings = result["bindings"]
    require(bindings["execution_bundle_sha256"] == EXECUTION_BUNDLE, "historical result execution bundle digest mismatch")
    require(bindings["deterministic_archive_sha256"] == DETERMINISTIC_ARCHIVE, "historical result archive digest mismatch")
    require(bindings["bundle_manifest_sha256"] == MANIFEST_FILE_SHA, "historical result manifest digest mismatch")
    require(bindings["predecessor_target_identity_stdout_sha256"] == PREDECESSOR_ID_SHA, "historical result predecessor identity mismatch")
    require(bindings["deployment_archive_host_sha256"] == bindings["deployment_archive_target_sha256"], "historical transfer digest mismatch")
    require(bindings["current_target_identity_before_sha256"] == bindings["current_target_identity_after_sha256"], "historical identity before/after mismatch")
    require(bindings["before_tree_canonical_sha256"] == bindings["after_tree_canonical_sha256"], "historical before/after tree mismatch")
    for name, blob in EXPECTED_BLOBS.items():
        key = {
            "gate_a_hardware_adapter.py": "host_adapter_git_blob_sha1",
            "gate_a_target_runner.py": "target_runner_git_blob_sha1",
            "gate_a_worker.c": "target_worker_git_blob_sha1",
            "gate_a_target_bundle.py": "target_bundle_validator_git_blob_sha1",
            "gate_a_authority.py": "authority_validator_git_blob_sha1",
            "build_gate_a_execution_bundle.py": "bundle_builder_git_blob_sha1",
            "verify_gate_a_adapter_qualification.py": "qualification_verifier_git_blob_sha1",
            "gate_a_isolated_qualification.py": "isolated_qualification_harness_git_blob_sha1",
        }[name]
        require(bindings[key] == blob, f"historical result blob {key} mismatch")
    for key, value in result["authority_false_state"].items():
        require(value is False, f"historical result authority_false_state {key} must be false")
    validate_predecessor_adapter_package_digests(result["predecessor_adapter_package_digests"], "historical result")
    require(result["next_boundary"] == "GATE_A_ENGINEERING_SMOKE_AUTHORITY_REVIEW_AND_OWNER_DECISION", "historical result next boundary mismatch")


def validate_historical_candidate_v3(candidate: dict[str, Any]) -> None:
    require(candidate["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3", "historical Candidate V3 schema mismatch")
    require(candidate["status"] == "CANDIDATE__BLOCKED_PENDING_INDEPENDENT_REVIEW_AND_PROJECT_OWNER_EXECUTION_DECISION", "historical Candidate V3 status mismatch")
    for key in (
        "plan_reviewed",
        "adapter_implemented",
        "hosted_adapter_qualification_complete",
        "target_adapter_qualification_complete",
        "execution_bundle_ready",
        "execution_bundle_target_qualified",
        "project_owner_target_qualification_approval_recorded",
    ):
        require(candidate[key] is True, f"historical Candidate V3 {key} must be true")
    for key in (
        "project_owner_execution_approval_recorded",
        "authorization_artifact_created",
        "engineering_smoke_authorized",
        "hardware_ran",
        "automatic_retry",
        "calibration_authorized",
        "scientific_acquisition_authorized",
        "restoration_authorized",
        "target_coupling_authorized",
        "small_wall_authorized",
    ):
        require(candidate[key] is False, f"historical Candidate V3 {key} must be false")
    require(candidate["next_boundary"] == "GATE_A_ENGINEERING_SMOKE_AUTHORITY_REVIEW_AND_OWNER_DECISION", "historical Candidate V3 next boundary mismatch")
    require(candidate["execution_bundle_sha256"] == EXECUTION_BUNDLE, "historical Candidate V3 execution bundle mismatch")
    validate_predecessor_adapter_package_digests(candidate["predecessor_adapter_package_digests"], "historical Candidate V3")


def validate_qualification_json(qualification: dict[str, Any]) -> None:
    require(qualification["status"] == "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED", "historical qualification status mismatch")
    require(qualification["git_free"] is True, "historical qualification not git-free")
    require(qualification["compiled"] is True, "historical qualification not compiled")
    local_validation = qualification["local_bundle_validation"]
    require(local_validation["status"] == "GATE_A_TARGET_BUNDLE_VALIDATED", "historical qualification bundle status mismatch")
    require(local_validation["strict"] is True, "historical qualification bundle validation not strict")
    require(local_validation["execution_bundle_sha256"] == EXECUTION_BUNDLE, "historical qualification bundle mismatch")
    require(local_validation["deterministic_archive_sha256"] == DETERMINISTIC_ARCHIVE, "historical qualification archive mismatch")
    require(qualification["worker_validate_only"]["status"] == "GATE_A_WORKER_VALIDATE_ONLY_OK", "historical worker validate-only mismatch")
    for key in ("network_connections_opened", "hardware_probes", "sender_starts", "receiver_captures", "control_writes", "msr_accesses", "hardware_executions"):
        require(qualification[key] == 0, f"historical qualification counter {key} nonzero")


def validate_digest_semantics(
    semantics: dict[str, Any],
    final_bindings: dict[str, Any],
    evidence_dir: Path,
) -> None:
    require(semantics["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_DIGEST_SEMANTICS_V1", "digest semantics schema mismatch")
    require(semantics["review_comments"] == [4929354450, 4929446568], "digest semantics review comments mismatch")
    require(semantics["predecessor_adapter_qualification_result_sha256"] == PREDECESSOR_RESULT_SHA, "digest semantics predecessor adapter result mismatch")
    require(semantics["predecessor_candidate_v2_sha256"] == PREDECESSOR_CANDIDATE_V2_SHA, "digest semantics predecessor Candidate V2 mismatch")
    require(semantics["predecessor_digest_meaning"] == PREDECESSOR_DIGEST_MEANING, "digest semantics meaning mismatch")
    require(semantics["target_qualification_result_committed_sha256"] == sha256_file(RESULT), "digest semantics historical result committed hash mismatch")
    require(semantics["candidate_v3_committed_sha256"] == sha256_file(CANDIDATE_V3), "digest semantics historical Candidate V3 hash mismatch")
    require(semantics["final_bindings_committed_sha256"] == sha256_file(evidence_dir / "FINAL_BINDINGS.json"), "digest semantics FINAL_BINDINGS hash mismatch")
    validate_predecessor_adapter_package_digests(final_bindings["predecessor_adapter_package_digests"], "final bindings")
    require(final_bindings["owner_identity"] == EXPECTED_OWNER, "final bindings owner identity mismatch")
    require(final_bindings["target_qualification_result_committed_sha256"] == semantics["target_qualification_result_committed_sha256"], "final bindings historical result hash mismatch")
    require(final_bindings["candidate_v3_committed_sha256"] == semantics["candidate_v3_committed_sha256"], "final bindings historical Candidate V3 hash mismatch")
    require(final_bindings["digest_semantics_path"] == DIGEST_SEMANTICS_NAME, "final bindings digest semantics path mismatch")


def validate_historical_evidence(evidence_dir: Path = EVID) -> dict[str, Any]:
    require(evidence_dir.is_dir(), f"historical evidence dir missing: {evidence_dir}")
    final_bindings = load(evidence_dir / "FINAL_BINDINGS.json")
    semantics = load(evidence_dir / DIGEST_SEMANTICS_NAME)
    validate_digest_semantics(semantics, final_bindings, evidence_dir)
    require(final_bindings["overall_status"] == "SUCCESS", "historical final bindings not SUCCESS")
    require(final_bindings["qualification_execution_count"] == 1, "historical final bindings execution count != 1")
    require(final_bindings["qualification_exit_code"] == 0, "historical final bindings exit code != 0")
    require(final_bindings["bundle_tree_unchanged"] is True, "historical final bindings tree changed")
    require(final_bindings["copy_back_verified"] is True, "historical final bindings copy-back unverified")
    require(final_bindings["cleanup_verified"] is True, "historical final bindings cleanup flag missing")

    copy_back = load(evidence_dir / "COPY_BACK_RECEIPT.json")
    require(copy_back["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1", "historical copy-back schema mismatch")
    require(copy_back["retained_evidence_custody_verified"] is True, "historical copy-back custody unverified")
    require(copy_back["inventory_verified"] is True, "historical copy-back inventory unverified")
    require(copy_back["unexpected_entries"] == [], "historical copy-back unexpected entries present")

    cleanup = load(evidence_dir / "CLEANUP_RECEIPT.json")
    require(cleanup["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V1", "historical cleanup schema mismatch")
    for key in ("exact_execution_root_removed", "exact_transfer_stage_removed", "execution_root_absence_proven", "transfer_stage_absence_proven"):
        require(cleanup[key] is True, f"historical cleanup {key} not true")
    require(cleanup["forbidden_processes_remaining"] == [], "historical cleanup list changed")

    inventory = load(evidence_dir / "EVIDENCE_INVENTORY.json")
    listed = {entry["path"] for entry in inventory["files"]}
    require(DIGEST_SEMANTICS_NAME in listed, "digest semantics missing from historical evidence inventory")
    for entry in inventory["files"]:
        file_path = evidence_dir / entry["path"]
        require(file_path.is_file(), f"historical evidence inventory missing file: {entry['path']}")
        require(file_path.stat().st_size == entry["size"], f"historical evidence size mismatch: {entry['path']}")
        require(sha256_file(file_path) == entry["sha256"], f"historical evidence sha256 mismatch: {entry['path']}")
    actual = {
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file() and path.name not in ("EVIDENCE_INVENTORY.json", "FINAL_BINDINGS.json")
    }
    require(not (actual - listed), f"unexpected historical evidence files: {sorted(actual - listed)}")

    copy_dir = evidence_dir / "copy_back" / "target_evidence"
    require(copy_dir.is_dir(), "copied-back historical target evidence dir missing")
    target_inventory = json.loads((copy_dir / "TARGET_EVIDENCE_INVENTORY.json").read_text(encoding="utf-8"))
    require(isinstance(target_inventory, list), "historical target evidence inventory must be a list")
    target_names = {entry["path"] for entry in target_inventory}
    for entry in target_inventory:
        file_path = copy_dir / entry["path"]
        require(file_path.is_file(), f"historical target evidence missing: {entry['path']}")
        require(file_path.stat().st_size == entry["size"], f"historical target evidence size mismatch: {entry['path']}")
        require(sha256_file(file_path) == entry["sha256"], f"historical target evidence sha256 mismatch: {entry['path']}")
    copied_actual = {path.name for path in copy_dir.iterdir() if path.is_file()}
    require(target_names <= copied_actual, f"copied-back historical evidence missing entries: {target_names - copied_actual}")
    require((copied_actual - target_names) <= {"TARGET_EVIDENCE_INVENTORY.json"}, "copied-back historical evidence has unexpected files")

    qualification = load(copy_dir / "TARGET_QUALIFICATION_RESULT.json")
    validate_qualification_json(qualification)
    tree_before = json.loads((copy_dir / "TARGET_TREE_BEFORE.json").read_text(encoding="utf-8"))
    tree_after = json.loads((copy_dir / "TARGET_TREE_AFTER.json").read_text(encoding="utf-8"))
    require(tree_before == tree_after, "historical copied-back before/after tree differ")
    identity_before = load(copy_dir / "TARGET_IDENTITY_BEFORE.json")
    identity_after = load(copy_dir / "TARGET_IDENTITY_AFTER.json")
    for key in ("hostname", "architecture", "cpu_model"):
        require(identity_before[key] == identity_after[key], f"historical identity {key} changed")
    require(identity_before["hostname"] == EXPECTED_HOSTNAME, "historical identity hostname mismatch")
    require(identity_before["architecture"] == EXPECTED_ARCH, "historical identity arch mismatch")
    require(identity_before["cpu_model"] == EXPECTED_CPU, "historical identity cpu mismatch")

    commands = [
        json.loads(line)
        for line in (evidence_dir / "COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qualification_commands = [
        command
        for command in commands
        if "--qualify-no-drive" in str(command.get("argv")) and "gate_a_target_runner.py" in str(command.get("argv"))
    ]
    require(len(qualification_commands) == 1, f"expected exactly one historical qualification execution, found {len(qualification_commands)}")
    for command in commands:
        argv_text = " ".join(command.get("argv", [])) if isinstance(command.get("argv"), list) else str(command.get("argv"))
        for forbidden in FORBIDDEN_COMMAND_SUBSTRINGS:
            require(forbidden not in argv_text, f"forbidden historical command substring {forbidden!r} in {argv_text[:120]}")

    return {
        "final_bindings": final_bindings,
        "digest_semantics": semantics,
        "copy_back": copy_back,
        "cleanup": cleanup,
        "qualification": qualification,
    }


def detect_historical_process_evidence_defect(evidence_dir: Path = EVID) -> dict[str, Any]:
    process_scripts = [
        evidence_dir / "target" / "logs" / "012_process_before.script.py",
        evidence_dir / "target" / "logs" / "017_process_after.script.py",
    ]
    for script in process_scripts:
        text = script.read_text(encoding="utf-8")
        require('subprocess.run(["ps","-eo","pid,comm,args"], stdout=subprocess.PIPE).stdout' in text, f"historical scanner shape changed: {script}")
        require("returncode" not in text and "check=True" not in text, f"historical scanner unexpectedly binds return status: {script}")
        require("raw_process_listing" not in text, f"historical scanner unexpectedly binds raw listing: {script}")

    copied_process_files = [
        evidence_dir / "copy_back" / "target_evidence" / "TARGET_PROCESS_STATE_BEFORE.txt",
        evidence_dir / "copy_back" / "target_evidence" / "TARGET_PROCESS_STATE_AFTER.txt",
    ]
    for process_file in copied_process_files:
        value = json.loads(process_file.read_text(encoding="utf-8"))
        require(set(value) == {"forbidden_process_hits"}, f"historical process state unexpectedly complete: {process_file}")

    cleanup_script = (evidence_dir / "cleanup" / "logs" / "021_cleanup_namespace.script.py").read_text(encoding="utf-8")
    require('res["forbidden_processes_remaining"]=[]' in cleanup_script, "historical cleanup hardcoded assertion not detected")
    cleanup_receipt = load(evidence_dir / "CLEANUP_RECEIPT.json")
    require(cleanup_receipt["forbidden_processes_remaining"] == [], "historical cleanup receipt changed")
    return {
        "process_scan_return_code_bound": False,
        "raw_process_listing_bound": False,
        "cleanup_process_absence_proven": False,
        "defect_detected": True,
        "process_scripts": [path.relative_to(evidence_dir).as_posix() for path in process_scripts],
    }


def validate_adjudication(
    adjudication: dict[str, Any],
    schema: dict[str, Any],
    defect: dict[str, Any],
) -> None:
    validate_const_instance(adjudication, schema, "adjudication")
    require(adjudication["status"] == CURRENT_STATUS, "adjudication current status mismatch")
    require(adjudication["process_scan_return_code_bound"] == defect["process_scan_return_code_bound"], "adjudication return-code gap mismatch")
    require(adjudication["raw_process_listing_bound"] == defect["raw_process_listing_bound"], "adjudication raw-list gap mismatch")
    require(adjudication["cleanup_process_absence_proven"] == defect["cleanup_process_absence_proven"], "adjudication cleanup gap mismatch")
    require(adjudication["target_nonexecuting_qualification_complete"] is False, "adjudication must keep target qualification false")
    require(adjudication["execution_bundle_target_qualified"] is False, "adjudication must keep target bundle qualification false")
    require(adjudication["original_authority_consumed"] is True, "adjudication must consume original authority")
    require(adjudication["replacement_qualification_authorized"] is False, "adjudication must not authorize replacement")
    require(adjudication["replacement_authority_artifact_created"] is False, "adjudication replacement authority artifact must be false")
    require(adjudication["project_owner_execution_approval_recorded"] is False, "adjudication execution approval must be false")
    require(adjudication["authorization_artifact_created"] is False, "adjudication execution authority artifact must be false")
    require(adjudication["automatic_retry"] is False, "adjudication automatic retry must be false")
    require(adjudication["next_boundary"] == NEXT_BOUNDARY, "adjudication next boundary mismatch")


def validate_candidate_v4(candidate: dict[str, Any], schema: dict[str, Any]) -> None:
    validate_const_instance(candidate, schema, "Candidate V4")
    validate_predecessor_adapter_package_digests(candidate["predecessor_adapter_package_digests"], "Candidate V4")
    require(candidate["target_adapter_qualification_complete"] is False, "Candidate V4 target qualification must be false")
    require(candidate["execution_bundle_target_qualified"] is False, "Candidate V4 target bundle qualification must be false")
    require(candidate["replacement_target_qualification_authorized"] is False, "Candidate V4 replacement authority must be false")
    require(candidate["replacement_target_qualification_authorization_artifact_created"] is False, "Candidate V4 replacement authority artifact must be false")
    require(candidate["project_owner_execution_approval_recorded"] is False, "Candidate V4 execution approval must be false")
    require(candidate["authorization_artifact_created"] is False, "Candidate V4 execution artifact must be false")
    require(candidate["engineering_smoke_authorized"] is False, "Candidate V4 engineering smoke must be false")
    require(candidate["hardware_ran"] is False, "Candidate V4 hardware_ran must be false")
    require(candidate["automatic_retry"] is False, "Candidate V4 automatic retry must be false")
    require(candidate["supersedes_current_authority_interpretation_of"] == [CANDIDATE_V3.name], "Candidate V4 must supersede V3 current interpretation")
    require(candidate["next_boundary"] == NEXT_BOUNDARY, "Candidate V4 next boundary mismatch")


def require_no_execution_authority(search_root: Path, *, check_tracked: bool) -> None:
    hits = list(search_root.rglob("GATE_A_EXECUTION_AUTHORITY.json"))
    require(not hits, f"execution authority artifact present: {hits}")
    if check_tracked:
        proc = subprocess.run(
            ["git", "ls-files", "*GATE_A_EXECUTION_AUTHORITY.json"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(proc.returncode == 0, f"git ls-files authority check failed: {proc.stderr}")
        require(proc.stdout.strip() == "", f"execution authority artifact tracked: {proc.stdout.strip()}")


def require_no_replacement_authority_artifact() -> None:
    name = "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
    hits = list(PHASE6B6.rglob(name))
    require(not hits, f"replacement target-qualification authority artifact present: {hits}")
    proc = subprocess.run(
        ["git", "ls-files", f"*{name}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(proc.returncode == 0, f"git ls-files replacement authority check failed: {proc.stderr}")
    require(proc.stdout.strip() == "", f"replacement target-qualification authority artifact tracked: {proc.stdout.strip()}")


def require_no_unaccented_owner() -> None:
    forbidden_owner = "Ra" + "ul Romero"
    for root in (HERE, EVID):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            require(forbidden_owner not in text, f"unaccented owner identity in scoped artifact: {path}")


def verify_adapter_blobs() -> None:
    for name, blob in EXPECTED_BLOBS.items():
        actual = git_blob(ADAPTER / name)
        require(actual == blob, f"adapter blob {name} mismatch: {actual} != {blob}")
    manifest_sha = sha256_file(ADAPTER / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json")
    require(manifest_sha == MANIFEST_FILE_SHA, f"adapter manifest file sha mismatch: {manifest_sha}")


def assert_rejects(name: str, func: Callable[[], None]) -> str:
    try:
        func()
    except Exception:
        return name
    raise VerifyError(f"mutation accepted: {name}")


def future_scanner_nonzero_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gate_a_fake_ps_") as temp_dir:
        root = Path(temp_dir)
        fake_ps = root / "fake_ps.py"
        receipt_path = root / "scan.json"
        fake_ps.write_text(
            "import sys\nsys.stdout.write('PARTIAL_PROCESS_LIST\\n')\nsys.stderr.write('fake ps failure\\n')\nraise SystemExit(7)\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PROCESS_SCAN_COMMAND_JSON"] = json.dumps([sys.executable, str(fake_ps)])
        env["OUT"] = str(receipt_path)
        proc = subprocess.run(
            [sys.executable, "-c", future_runner.process_script()],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(proc.returncode == 70, f"future scanner did not abort on fake ps failure: rc={proc.returncode}")
        receipt = load(receipt_path)
        require(receipt["return_code"] == 7, "future scanner did not bind fake ps return code")
        require(receipt["scan_complete"] is False, "future scanner marked failed ps complete")
        require(receipt["ps_executed_successfully"] is False, "future scanner marked failed ps successful")
        require(receipt["forbidden_process_filter_evaluated"] is False, "future scanner filtered failed ps output")
        require(receipt["forbidden_process_hits"] == [], "future scanner emitted misleading hits on failed ps")
        require("PARTIAL_PROCESS_LIST" in receipt["raw_process_listing"], "future scanner lost partial stdout")
        require("fake ps failure" in receipt["raw_process_stderr"], "future scanner lost partial stderr")
        require(receipt["stdout_sha256"] == sha256_text(receipt["raw_process_listing"]), "future scanner stdout hash mismatch")
        require(receipt["stderr_sha256"] == sha256_text(receipt["raw_process_stderr"]), "future scanner stderr hash mismatch")
        require(base64.b64decode(receipt["raw_process_listing_base64"]) == receipt["raw_process_listing"].encode("utf-8"), "future scanner raw stdout base64 mismatch")
        require(base64.b64decode(receipt["raw_process_stderr_base64"]) == receipt["raw_process_stderr"].encode("utf-8"), "future scanner raw stderr base64 mismatch")
        return {
            "status": "FAKE_PS_NONZERO_REJECTED",
            "fake_ps_return_code": receipt["return_code"],
            "scanner_return_code": proc.returncode,
            "scan_complete": receipt["scan_complete"],
            "raw_stdout_preserved": True,
            "raw_stderr_preserved": True,
        }


def future_replacement_authority_contract_test() -> dict[str, Any]:
    source_commit = future_runner.current_head()
    authority_id = "synthetic_owner_decision_test"
    local_evidence_dir = (
        future_runner.EVIDENCE_PARENT.relative_to(future_runner.REPO_ROOT).as_posix()
        + f"/gate_a_target_nonexecuting_qualification_replacement_{authority_id}"
    )
    authority: dict[str, Any] = {
        "schema_id": future_runner.REPLACEMENT_AUTHORITY_SCHEMA_ID,
        "decision": future_runner.REPLACEMENT_AUTHORITY_DECISION,
        "project_owner": EXPECTED_OWNER,
        "owner_instruction": "Authorize one synthetic contract-validation replacement only",
        "authority_id": authority_id,
        "authorized_source_commit": source_commit,
        "historical_authorization_path": future_runner.HISTORICAL_AUTHORIZATION_REL,
        "historical_authority_consumed": True,
        "historical_evidence_dir": future_runner.HISTORICAL_EVIDENCE_REL,
        "local_evidence_dir": local_evidence_dir,
        "ssh_target": future_runner.SSH_TARGET,
        "expected_hostname": future_runner.EXPECTED_HOSTNAME,
        "expected_architecture": future_runner.EXPECTED_ARCH,
        "expected_cpu_model": future_runner.EXPECTED_CPU,
        "remote_execution_root": f"/root/catcas_phase6b6_gate_a_target_nonexec_{authority_id}",
        "remote_evidence_root": f"/root/catcas_phase6b6_gate_a_target_nonexec_{authority_id}/evidence",
        "remote_transfer_stage": f"/tmp/catcas_gate_a_bundle_{authority_id}.deploy.tar",
        "remote_evidence_archive": f"/tmp/catcas_gate_a_evidence_{authority_id}.tar",
        "remote_temp_prefix": f"/tmp/catcas_gate_a_tq_{authority_id}_",
        "execution_bundle_sha256": future_runner.EXPECTED_EXECUTION_BUNDLE,
        "deterministic_archive_sha256": future_runner.EXPECTED_ARCHIVE,
        "bundle_manifest_sha256": future_runner.EXPECTED_MANIFEST_FILE,
        "maximum_target_qualification_executions": 1,
        "automatic_retry": False,
        "replacement_qualification_authorized": True,
        "ssh_authorized": True,
        "copy_authorized": True,
        "target_filesystem_staging_authorized": True,
        "compile_validate_only_authorized": True,
        "no_drive_runner_authorized": True,
        "probe_authorized": False,
        "engineering_smoke_authorized": False,
        "hardware_execution_authorized": False,
        "calibration_authorized": False,
        "scientific_acquisition_authorized": False,
        "restoration_authorized": False,
        "target_coupling_authorized": False,
        "small_wall_authorized": False,
        "execution_authority_artifact_creation_authorized": False,
    }
    with tempfile.TemporaryDirectory(prefix="gate_a_authority_contract_") as temp_dir:
        authority_path = Path(temp_dir) / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
        evidence_path = future_runner.validate_replacement_authority(
            authority,
            authority_path,
            source_commit=source_commit,
        )
    require(evidence_path == future_runner.REPO_ROOT / local_evidence_dir, "future authority resolved wrong evidence namespace")
    future_runner.ensure_new_evidence_namespace(evidence_path)
    return {
        "status": "SYNTHETIC_REPLACEMENT_AUTHORITY_CONTRACT_VALID",
        "closed_key_count": len(authority),
        "historical_namespace_reused": False,
        "automatic_retry": authority["automatic_retry"],
        "downstream_authority_false": True,
    }


def future_process_scan_receipt_contract_test() -> dict[str, Any]:
    command = ["ps", "-eo", "pid,comm,args"]
    raw_stdout = b"    PID COMMAND         COMMAND\n      1 init            /sbin/init\n"
    raw_stderr = b""
    command_bytes = json.dumps(
        command,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    scan = {
        "command": command,
        "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
        "return_code": 0,
        "stdout_sha256": hashlib.sha256(raw_stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(raw_stderr).hexdigest(),
        "raw_process_listing": raw_stdout.decode("utf-8"),
        "raw_process_listing_base64": base64.b64encode(raw_stdout).decode("ascii"),
        "raw_process_listing_sha256": hashlib.sha256(raw_stdout).hexdigest(),
        "raw_process_stderr": "",
        "raw_process_stderr_base64": base64.b64encode(raw_stderr).decode("ascii"),
        "ps_executed_successfully": True,
        "raw_process_listing_preserved": True,
        "forbidden_process_filter_evaluated": True,
        "forbidden_process_hits": [],
        "scan_complete": True,
    }
    future_runner.validate_process_scan(scan, "synthetic successful scan")
    return {
        "status": "SUCCESSFUL_PROCESS_SCAN_RECEIPT_CONTRACT_VALID",
        "exact_command_bound": True,
        "raw_stdout_bound": True,
        "raw_stderr_bound": True,
        "scan_complete": True,
    }


def verify_scp_timeout_recording(direction: str) -> str:
    with tempfile.TemporaryDirectory(prefix=f"gate_a_scp_{direction}_") as temp_dir:
        evidence = Path(temp_dir)
        future_runner.REC = future_runner.Recorder(evidence)
        timeout = subprocess.TimeoutExpired(
            cmd=["scp"],
            timeout=300,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        with mock.patch.object(future_runner.subprocess, "run", side_effect=timeout):
            try:
                if direction == "upload":
                    future_runner.scp_to("timeout_upload", Path("bundle.tar"), "/tmp/bundle.tar")
                else:
                    future_runner.scp_from("timeout_download", "/tmp/evidence.tar", Path("evidence.tar"))
            except future_runner.QualError:
                pass
            else:
                raise VerifyError(f"scp {direction} timeout did not raise QualError")
        entries = [json.loads(line) for line in (evidence / "COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()]
        require(len(entries) == 1, f"scp {direction} timeout command not recorded exactly once")
        require(entries[0]["exit_code"] == 124, f"scp {direction} timeout exit code not 124")
        stdout = (evidence / entries[0]["stdout_path"]).read_bytes()
        stderr = (evidence / entries[0]["stderr_path"]).read_bytes()
        require(stdout == b"partial stdout", f"scp {direction} partial stdout lost")
        require(b"partial stderr" in stderr and b"[timeout]" in stderr, f"scp {direction} partial stderr lost")
    return f"scp_{direction}_timeout_recorded"


def mutation_tests(
    result: dict[str, Any],
    authorization: dict[str, Any],
    candidate_v3: dict[str, Any],
    qualification: dict[str, Any],
    semantics: dict[str, Any],
    final_bindings: dict[str, Any],
    adjudication: dict[str, Any],
    adjudication_schema: dict[str, Any],
    candidate_v4: dict[str, Any],
    candidate_v4_schema: dict[str, Any],
    defect: dict[str, Any],
) -> dict[str, Any]:
    cases: list[str] = []

    def mutated(base: dict[str, Any], mutator: Callable[[dict[str, Any]], None], validator: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        def inner() -> None:
            value = copy.deepcopy(base)
            mutator(value)
            validator(value)
        return inner

    historical_result = lambda value: validate_historical_result(value)
    historical_auth = lambda value: validate_historical_authorization(value)
    historical_candidate = lambda value: validate_historical_candidate_v3(value)
    qualification_validator = lambda value: validate_qualification_json(value)
    adjudication_validator = lambda value: validate_adjudication(value, adjudication_schema, defect)
    candidate_v4_validator = lambda value: validate_candidate_v4(value, candidate_v4_schema)

    def semantics_validator(value: dict[str, Any]) -> None:
        validate_digest_semantics(value, final_bindings, EVID)

    checks = [
        ("second_historical_qualification_execution", mutated(result, lambda value: value.__setitem__("target_qualification_execution_count", 2), historical_result)),
        ("historical_automatic_retry_enabled", mutated(result, lambda value: value.__setitem__("automatic_retry", True), historical_result)),
        ("historical_authorization_retry_enabled", mutated(authorization, lambda value: value.__setitem__("automatic_retry", True), historical_auth)),
        ("historical_probe_invocation", mutated(result, lambda value: value.__setitem__("probe_count", 1), historical_result)),
        ("historical_execute_authorized_invocation", mutated(result, lambda value: value.__setitem__("execute_authorized_count", 1), historical_result)),
        ("historical_wrong_target", mutated(result, lambda value: value["target"].__setitem__("ssh_target", "root@127.0.0.1"), historical_result)),
        ("historical_wrong_bundle_digest", mutated(result, lambda value: value["bindings"].__setitem__("execution_bundle_sha256", "0" * 64), historical_result)),
        ("historical_wrong_worker_result", mutated(qualification, lambda value: value["worker_validate_only"].__setitem__("status", "BAD"), qualification_validator)),
        ("historical_nonzero_hardware_probe", mutated(qualification, lambda value: value.__setitem__("hardware_probes", 1), qualification_validator)),
        ("historical_nonzero_sender", mutated(qualification, lambda value: value.__setitem__("sender_starts", 1), qualification_validator)),
        ("historical_nonzero_receiver", mutated(qualification, lambda value: value.__setitem__("receiver_captures", 1), qualification_validator)),
        ("historical_nonzero_control_write", mutated(qualification, lambda value: value.__setitem__("control_writes", 1), qualification_validator)),
        ("historical_nonzero_msr", mutated(qualification, lambda value: value.__setitem__("msr_accesses", 1), qualification_validator)),
        ("historical_nonzero_hardware_execution", mutated(qualification, lambda value: value.__setitem__("hardware_executions", 1), qualification_validator)),
        ("historical_candidate_execution_approval_true", mutated(candidate_v3, lambda value: value.__setitem__("project_owner_execution_approval_recorded", True), historical_candidate)),
        ("wrong_predecessor_adapter_result_digest", mutated(semantics, lambda value: value.__setitem__("predecessor_adapter_qualification_result_sha256", "0" * 64), semantics_validator)),
        ("wrong_predecessor_candidate_v2_digest", mutated(semantics, lambda value: value.__setitem__("predecessor_candidate_v2_sha256", "0" * 64), semantics_validator)),
        ("adjudication_target_qualification_promoted", mutated(adjudication, lambda value: value.__setitem__("target_nonexecuting_qualification_complete", True), adjudication_validator)),
        ("adjudication_target_bundle_promoted", mutated(adjudication, lambda value: value.__setitem__("execution_bundle_target_qualified", True), adjudication_validator)),
        ("candidate_v4_advances_to_smoke_review", mutated(candidate_v4, lambda value: value.__setitem__("next_boundary", "GATE_A_ENGINEERING_SMOKE_AUTHORITY_REVIEW_AND_OWNER_DECISION"), candidate_v4_validator)),
        ("adjudication_automatic_retry_true", mutated(adjudication, lambda value: value.__setitem__("automatic_retry", True), adjudication_validator)),
        ("candidate_v4_automatic_retry_true", mutated(candidate_v4, lambda value: value.__setitem__("automatic_retry", True), candidate_v4_validator)),
        ("replacement_authorized_without_new_owner_artifact", mutated(adjudication, lambda value: value.__setitem__("replacement_qualification_authorized", True), adjudication_validator)),
        ("candidate_v4_replacement_authorized", mutated(candidate_v4, lambda value: value.__setitem__("replacement_target_qualification_authorized", True), candidate_v4_validator)),
        ("adjudication_replacement_authority_artifact_created", mutated(adjudication, lambda value: value.__setitem__("replacement_authority_artifact_created", True), adjudication_validator)),
        ("candidate_v4_replacement_authority_artifact_created", mutated(candidate_v4, lambda value: value.__setitem__("replacement_target_qualification_authorization_artifact_created", True), candidate_v4_validator)),
        ("original_authority_treated_as_reusable", mutated(adjudication, lambda value: value.__setitem__("original_authority_consumed", False), adjudication_validator)),
        ("process_return_code_gap_omitted", mutated(adjudication, lambda value: value.__setitem__("process_scan_return_code_bound", True), adjudication_validator)),
        ("raw_process_list_gap_omitted", mutated(adjudication, lambda value: value.__setitem__("raw_process_listing_bound", True), adjudication_validator)),
        ("cleanup_process_absence_claimed_proven", mutated(adjudication, lambda value: value.__setitem__("cleanup_process_absence_proven", True), adjudication_validator)),
        ("candidate_v4_target_qualification_promoted", mutated(candidate_v4, lambda value: value.__setitem__("target_adapter_qualification_complete", True), candidate_v4_validator)),
        ("candidate_v4_target_bundle_promoted", mutated(candidate_v4, lambda value: value.__setitem__("execution_bundle_target_qualified", True), candidate_v4_validator)),
        ("adjudication_execution_approval_true", mutated(adjudication, lambda value: value.__setitem__("project_owner_execution_approval_recorded", True), adjudication_validator)),
        ("adjudication_execution_artifact_created", mutated(adjudication, lambda value: value.__setitem__("authorization_artifact_created", True), adjudication_validator)),
        ("candidate_v4_execution_artifact_created", mutated(candidate_v4, lambda value: value.__setitem__("authorization_artifact_created", True), candidate_v4_validator)),
    ]
    for field in DOWNSTREAM_FALSE_FIELDS:
        checks.append((f"adjudication_downstream_{field}_true", mutated(adjudication, lambda value, key=field: value.__setitem__(key, True), adjudication_validator)))
        checks.append((f"candidate_v4_downstream_{field}_true", mutated(candidate_v4, lambda value, key=field: value.__setitem__(key, True), candidate_v4_validator)))
    for name, check in checks:
        cases.append(assert_rejects(name, check))

    def validate_mutated_evidence(mutator: Callable[[Path], None]) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_evidence_mutation_") as temp_dir:
            copied = Path(temp_dir) / EVID.name
            shutil.copytree(EVID, copied)
            mutator(copied)
            validate_historical_evidence(copied)

    cases.append(assert_rejects(
        "historical_evidence_directory_changes",
        lambda: validate_mutated_evidence(lambda copied: (copied / "README.md").write_bytes((copied / "README.md").read_bytes() + b"\n")),
    ))
    cases.append(assert_rejects(
        "historical_evidence_file_removed",
        lambda: validate_mutated_evidence(lambda copied: (copied / "target" / "logs" / "012_process_before.stdout.txt").unlink()),
    ))
    cases.append(assert_rejects(
        "historical_evidence_extra_file",
        lambda: validate_mutated_evidence(lambda copied: (copied / "UNEXPECTED_EXTRA.txt").write_text("x", encoding="utf-8")),
    ))

    def execution_authority_appears() -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_authority_mutation_") as temp_dir:
            root = Path(temp_dir)
            (root / "GATE_A_EXECUTION_AUTHORITY.json").write_text("{}\n", encoding="utf-8")
            require_no_execution_authority(root, check_tracked=False)

    cases.append(assert_rejects("execution_authority_artifact_appears", execution_authority_appears))

    def original_authority_reused_by_future_runner() -> None:
        future_runner.validate_replacement_authority(
            authorization,
            AUTHORIZATION,
            source_commit=PRE_REPAIR_HEAD,
        )

    cases.append(assert_rejects("future_runner_reuses_original_authority", original_authority_reused_by_future_runner))

    def existing_replacement_namespace() -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_existing_namespace_") as temp_dir:
            future_runner.ensure_new_evidence_namespace(Path(temp_dir))

    cases.append(assert_rejects("future_runner_reuses_existing_namespace", existing_replacement_namespace))

    def uncommitted_replacement_authority() -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_uncommitted_authority_") as temp_dir:
            authority_path = Path(temp_dir) / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
            authority_path.write_text("{}\n", encoding="utf-8")
            future_runner.validate_replacement_authority_custody(
                authority_path,
                future_runner.current_head(),
            )

    cases.append(assert_rejects("future_runner_rejects_uncommitted_owner_authority", uncommitted_replacement_authority))
    cases.append(verify_scp_timeout_recording("upload"))
    cases.append(verify_scp_timeout_recording("download"))

    return {
        "status": "TARGET_NONEXEC_ADJUDICATION_MUTATION_TESTS_PASS",
        "negative_tests": len(cases),
        "cases": cases,
    }


def main() -> int:
    immutability = assert_historical_evidence_immutable()

    authorization = load(AUTHORIZATION)
    contract = load(CONTRACT)
    result = load(RESULT)
    candidate_v3 = load(CANDIDATE_V3)
    adjudication = load(ADJUDICATION)
    candidate_v4 = load(CANDIDATE_V4)
    result_schema = load(RESULT_SCHEMA)
    adjudication_schema = load(ADJUDICATION_SCHEMA)
    candidate_v4_schema = load(CANDIDATE_V4_SCHEMA)

    validate_schema_closed(
        result_schema,
        ("target", "bindings", "predecessor_adapter_package_digests", "authority_false_state"),
    )
    validate_schema_closed(adjudication_schema)
    validate_schema_closed(candidate_v4_schema, ("predecessor_adapter_package_digests",))

    validate_historical_authorization(authorization)
    validate_historical_contract(contract)
    validate_historical_result(result)
    validate_historical_candidate_v3(candidate_v3)
    verify_adapter_blobs()
    evidence = validate_historical_evidence()
    defect = detect_historical_process_evidence_defect()
    validate_adjudication(adjudication, adjudication_schema, defect)
    validate_candidate_v4(candidate_v4, candidate_v4_schema)
    require_no_unaccented_owner()
    require_no_execution_authority(PHASE6B6, check_tracked=True)
    require_no_replacement_authority_artifact()

    scanner_test = future_scanner_nonzero_test()
    scanner_receipt_test = future_process_scan_receipt_contract_test()
    future_authority_test = future_replacement_authority_contract_test()
    mutations = mutation_tests(
        result,
        authorization,
        candidate_v3,
        evidence["qualification"],
        evidence["digest_semantics"],
        evidence["final_bindings"],
        adjudication,
        adjudication_schema,
        candidate_v4,
        candidate_v4_schema,
        defect,
    )

    output = {
        "status": CURRENT_STATUS,
        "integrated_main": INTEGRATED_MAIN,
        "pre_repair_pr_head": PRE_REPAIR_HEAD,
        "historical_attempt_authorized": True,
        "historical_attempt_execution_count": 1,
        "historical_evidence_immutable": immutability,
        "process_evidence_defect": defect,
        "target_nonexecuting_qualification_complete": False,
        "execution_bundle_target_qualified": False,
        "original_authority_consumed": True,
        "replacement_qualification_authorized": False,
        "project_owner_execution_approval_recorded": False,
        "authorization_artifact_created": False,
        "replacement_authority_artifact_created": False,
        "engineering_smoke_authorized": False,
        "hardware_ran": False,
        "automatic_retry": False,
        "future_scanner_nonzero_test": scanner_test,
        "future_process_scan_receipt_contract_test": scanner_receipt_test,
        "future_replacement_authority_contract_test": future_authority_test,
        "mutation_tests": mutations,
        "historical_evidence_inventory_sha256": sha256_file(EVID / "EVIDENCE_INVENTORY.json"),
        "next_boundary": NEXT_BOUNDARY,
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        VerifyError,
        future_runner.QualError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        ValueError,
    ) as exc:
        print(f"verify_gate_a_target_nonexecuting_qualification: {exc}", file=sys.stderr)
        raise SystemExit(1)
