#!/usr/bin/env python3
"""Verify the preserved Gate A attempt and its fail-closed adjudication.

This verifier is local-only.  It never opens SSH/SCP or contacts the target.  A
green result means the historical packet is byte-for-byte unchanged, the one
historical attempt is still represented accurately, the process-evidence gaps
are detected, the old completion interpretation is superseded, historical and
downstream authority fields remain false, and any current Gate A execution
authority is the one exact production-validated committed artifact.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
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
REPLACEMENT_AUTHORITY = HERE / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
SUPERSEDED_REPLACEMENT_AUTHORITY = (
    HERE
    / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION_SUPERSEDED_71ab1528_01.json"
)
REPLACEMENT_AUTHORITY_STATE = HERE / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORITY_STATE.json"
REPLACEMENT_EVIDENCE = (
    PHASE6B6
    / "evidence"
    / "gate_a_target_nonexecuting_qualification_replacement_gate_a_replacement_593e9920_02"
)

RESULT_SCHEMA = HERE / "schemas" / "gate_a_target_nonexecuting_qualification_result.schema.json"
ADJUDICATION_SCHEMA = HERE / "schemas" / "gate_a_target_nonexecuting_qualification_adjudication.schema.json"
CANDIDATE_V4_SCHEMA = HERE / "schemas" / "gate_a_engineering_smoke_authority_candidate_v4.schema.json"
REPLACEMENT_AUTHORITY_STATE_SCHEMA = HERE / "schemas" / "gate_a_replacement_target_nonexecuting_qualification_authority_state.schema.json"

PRE_REPAIR_HEAD = "310efd0ec4103654b122a961b001bc5b79cf5896"
MERGED_PR37_BASELINE = "e03502b1859d6a3b79699fa56252710ba85f3595"
HISTORICAL_EVIDENCE_TREE_SHA1 = "a02edbfb85bb2b1816a3be92089112f29c639da9"
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
COMPLETED_CURRENT_STATUS = "REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_COMPLETE__AUTHORITY_CONSUMED"
COMPLETED_NEXT_BOUNDARY = (
    "STOP__REPLACEMENT_QUALIFICATION_COMPLETE__ALL_DOWNSTREAM_WORK_REQUIRES_NEW_OWNER_AUTHORITY"
)
REPAIRED_SOURCE_COMMIT = "593e9920be533603217cee93572d79b86cc65cf9"
REPAIRED_SOURCE_TREE = "d9488e9968023a98fabc0808530fdbc731d5831a"
REPAIRED_SOURCE_REVIEW_ID = "PR_37_GATE_A_FOUR_WAY_REMOTE_NAMESPACE_PREFLIGHT_REVIEW"
ACTIVE_AUTHORITY_ID = "gate_a_replacement_593e9920_02"
ACTIVE_AUTHORITY_SHA256 = "ecfa7d590d393c96e4f4f31180045660eef68f42b53c2de5f01faf3e0f933286"
ACTIVE_AUTHORITY_BLOB = "f7935a84b9cc11570b93a22551cce0fb2706aa97"
EXECUTION_HEAD = "1ea708cfdc93083cc9386a6b1b14cf51d1ed8367"
FINAL_BINDINGS_SHA256 = "a584f34b677e5e0d0a8fc5c057e2831ef4f60e865218c674edcebcd322bd5dca"
EVIDENCE_INVENTORY_SHA256 = "1c882900775358c634353b34394d79bcd19c509a003190fb214b1f2985505b20"
COMMANDS_SHA256 = "006507e525eedba79b230099532a0b64756d186af23ae5b383916a8ea48dbc91"
TARGET_EVIDENCE_ARCHIVE_SHA256 = "960426405b81571b762dfff833fc3b99d726667fd64f222bea41ddcebb056d6d"
SUPERSEDED_SOURCE_COMMIT = "71ab1528e44fe6181e72850a0bd93a131b7a6335"
SUPERSEDED_SOURCE_TREE = "527f5a275c9af59e3fd85716d5e752fa79db74d8"
SUPERSEDED_SOURCE_REVIEW_ID = "PR_37_REPLACEMENT_AUTHORITY_TWO_COMMIT_SOURCE_BINDING_REVIEW"
SUPERSEDED_AUTHORITY_BEARING_COMMIT = "1c293410d97b6ed7579cec94dd813890cda45f98"
SUPERSEDED_AUTHORITY_SHA256 = "13fde796b2a3caa71db4d08e02c6d62e785cfabc4301b7e8ad94b4946f215d87"
SUPERSEDED_AUTHORITY_BLOB = "289e3a9745337a5ed2b7bcc6230b8423eb49ce16"

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

if str(ADAPTER) not in sys.path:
    sys.path.insert(0, str(ADAPTER))
import verify_gate_a_adapter_qualification as adapter_qualification


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


def git_blob_at(treeish: str, path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    proc = subprocess.run(
        ["git", "rev-parse", f"{treeish}:{relative}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    return proc.stdout.strip()


def git_sha256_at(treeish: str, path: Path) -> str:
    blob = git_blob_at(treeish, path)
    data = subprocess.run(
        ["git", "cat-file", "blob", blob],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    return hashlib.sha256(data).hexdigest()


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
    baseline = subprocess.run(
        ["git", "rev-parse", f"{MERGED_PR37_BASELINE}:{EVID_REL}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    current = subprocess.run(
        ["git", "rev-parse", f"HEAD:{EVID_REL}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(baseline.returncode == 0, f"cannot read historical evidence baseline tree: {baseline.stderr}")
    require(current.returncode == 0, f"cannot read current historical evidence tree: {current.stderr}")
    require(baseline.stdout.strip() == HISTORICAL_EVIDENCE_TREE_SHA1, "historical evidence baseline tree binding changed")
    require(current.stdout.strip() == HISTORICAL_EVIDENCE_TREE_SHA1, "historical evidence committed tree changed")

    commands = [
        ["git", "diff", "--exit-code", "--", EVID_REL],
        ["git", "diff", "--cached", "--exit-code", "--", EVID_REL],
    ]
    labels = ("worktree", "index")
    proof: dict[str, Any] = {
        "baseline_head": MERGED_PR37_BASELINE,
        "historical_evidence_tree_sha1": HISTORICAL_EVIDENCE_TREE_SHA1,
        "committed_tree_exit_code": 0,
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


def validate_gate_a_execution_authority_state() -> dict[str, Any]:
    """Recognize absence or the one exact current Gate A execution authority."""

    try:
        manifest = adapter_qualification.load(adapter_qualification.MANIFEST)
        state = adapter_qualification.validate_execution_authority_state(manifest)
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        raise VerifyError(f"Gate A execution authority state invalid: {exc}") from exc
    if state["authority_artifact_present"]:
        require(
            state["authority_artifact_path"]
            == adapter_qualification.git_rel(adapter_qualification.AUTHORITY),
            "Gate A execution authority canonical path mismatch",
        )
        require(
            state["authority_sha256"] == adapter_qualification.EXECUTION_AUTHORITY_SHA256,
            "Gate A execution authority SHA-256 mismatch",
        )
        require(
            state["authority_git_blob_sha1"] == adapter_qualification.EXECUTION_AUTHORITY_BLOB_SHA1,
            "Gate A execution authority blob mismatch",
        )
        require(
            state["reviewed_source_commit"] == adapter_qualification.REVIEWED_EXECUTION_SOURCE_COMMIT,
            "Gate A reviewed source mismatch",
        )
        require(
            state["reviewed_source_tree"] == adapter_qualification.REVIEWED_EXECUTION_SOURCE_TREE,
            "Gate A reviewed source tree mismatch",
        )
        require(
            state["independent_review_id"] == adapter_qualification.REVIEWED_EXECUTION_SOURCE_REVIEW_ID,
            "Gate A independent review mismatch",
        )
        require(state["maximum_execution_count"] == 1, "Gate A maximum execution count mismatch")
        require(state["consumed"] is False, "Gate A execution authority must be unconsumed")
        require(state["automatic_retry"] is False, "Gate A automatic retry must remain false")
        require(state["downstream_authority_false"] is True, "Gate A downstream authority changed")
    return state


def validate_superseded_replacement_authority(authority: dict[str, Any]) -> dict[str, Any]:
    """Preserve _01 exactly while proving that it is no longer active authority."""
    require(
        SUPERSEDED_REPLACEMENT_AUTHORITY.is_file() and not SUPERSEDED_REPLACEMENT_AUTHORITY.is_symlink(),
        "superseded replacement authority must be a real file",
    )
    require(authority == load(SUPERSEDED_REPLACEMENT_AUTHORITY), "superseded authority object differs from preserved artifact")
    require(authority["project_owner"] == EXPECTED_OWNER, "superseded authority owner mismatch")
    require(authority["authority_id"] == "gate_a_replacement_71ab1528_01", "superseded authority id mismatch")
    require(authority["authorized_source_commit"] == SUPERSEDED_SOURCE_COMMIT, "superseded authority source commit mismatch")
    require(authority["authorized_source_tree_sha1"] == SUPERSEDED_SOURCE_TREE, "superseded authority source tree mismatch")
    require(authority["authorized_source_review_id"] == SUPERSEDED_SOURCE_REVIEW_ID, "superseded authority review label mismatch")
    require(authority["maximum_target_qualification_executions"] == 1, "superseded authority maximum execution mismatch")
    require(authority["automatic_retry"] is False, "superseded authority automatic retry must be false")
    for field in future_runner.DOWNSTREAM_FALSE_FIELDS:
        require(authority[field] is False, f"superseded authority downstream field must be false: {field}")

    archived_rel = SUPERSEDED_REPLACEMENT_AUTHORITY.relative_to(REPO_ROOT).as_posix()
    original_rel = REPLACEMENT_AUTHORITY.relative_to(REPO_ROOT).as_posix()
    require(sha256_file(SUPERSEDED_REPLACEMENT_AUTHORITY) == SUPERSEDED_AUTHORITY_SHA256, "superseded authority SHA-256 mismatch")
    require(git_blob(SUPERSEDED_REPLACEMENT_AUTHORITY) == SUPERSEDED_AUTHORITY_BLOB, "superseded authority worktree blob mismatch")
    current_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{archived_rel}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(current_blob.returncode == 0 and current_blob.stdout.strip() == SUPERSEDED_AUTHORITY_BLOB, "superseded authority committed blob mismatch")
    original_blob = subprocess.run(
        ["git", "rev-parse", f"{SUPERSEDED_AUTHORITY_BEARING_COMMIT}:{original_rel}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(original_blob.returncode == 0 and original_blob.stdout.strip() == SUPERSEDED_AUTHORITY_BLOB, "original authority-bearing blob mismatch")
    evidence_path = REPO_ROOT / authority["local_evidence_dir"]
    require(not evidence_path.exists() and not evidence_path.is_symlink(), "superseded authority evidence exists despite zero executions")
    protected_drift = subprocess.run(
        ["git", "diff", "--quiet", SUPERSEDED_SOURCE_COMMIT, "HEAD", "--", future_runner.RUNNER_REL],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(protected_drift.returncode == 1, "superseded authority source unexpectedly remains current")
    return {
        "status": "SUPERSEDED_REPLACEMENT_AUTHORITY_PRESERVED_UNCONSUMED",
        "authority_id": authority["authority_id"],
        "authority_artifact_path": archived_rel,
        "authority_artifact_sha256": SUPERSEDED_AUTHORITY_SHA256,
        "authority_artifact_git_blob_sha1": SUPERSEDED_AUTHORITY_BLOB,
        "authorized_source_commit": SUPERSEDED_SOURCE_COMMIT,
        "authorized_source_tree_sha1": SUPERSEDED_SOURCE_TREE,
        "authorized_source_review_id": SUPERSEDED_SOURCE_REVIEW_ID,
        "authority_bearing_commit": SUPERSEDED_AUTHORITY_BEARING_COMMIT,
        "authority_consumed": False,
        "replacement_execution_count": 0,
        "replacement_evidence_dir": evidence_path.relative_to(REPO_ROOT).as_posix(),
        "replacement_evidence_present": False,
        "protected_runner_drift_invalidates_target_contact": True,
    }


def validate_historical_replacement_authority_custody(
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Validate consumed authority against its recorded execution commit.

    This is intentionally verifier-local.  Production custody continues to
    require current HEAD; a squash merge must not rewrite the HEAD that
    actually carried and consumed this historical authority.
    """
    authority_rel = REPLACEMENT_AUTHORITY.relative_to(REPO_ROOT).as_posix()
    source_commit = authority["authorized_source_commit"]
    source_tree = authority["authorized_source_tree_sha1"]
    for commit, label in ((source_commit, "authorized source"), (EXECUTION_HEAD, "recorded execution HEAD")):
        object_check = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(object_check.returncode == 0, f"{label} is not an exact commit object")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, EXECUTION_HEAD],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(ancestor.returncode == 0 and source_commit != EXECUTION_HEAD, "historical authority source is not a strict execution ancestor")
    tree = subprocess.run(
        ["git", "rev-parse", f"{source_commit}^{{tree}}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(tree.returncode == 0 and tree.stdout.strip() == source_tree, "historical authority source tree mismatch")
    source_authority = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}:{authority_rel}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(source_authority.returncode != 0, "historical authority existed in reviewed source commit")
    for rel in future_runner.PROTECTED_EXECUTION_SOURCE_PATHS:
        for commit, label in ((source_commit, "source"), (EXECUTION_HEAD, "execution")):
            present = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}:{rel}"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            require(present.returncode == 0, f"historical protected path absent from {label}: {rel}")
    drift = subprocess.run(
        ["git", "diff", "--quiet", source_commit, EXECUTION_HEAD, "--", *future_runner.PROTECTED_EXECUTION_SOURCE_PATHS],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(drift.returncode == 0, "historical protected source drifted before execution")
    blob = subprocess.run(
        ["git", "rev-parse", f"{EXECUTION_HEAD}:{authority_rel}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(blob.returncode == 0 and blob.stdout.strip() == ACTIVE_AUTHORITY_BLOB, "historical execution authority blob mismatch")
    current_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{authority_rel}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(current_blob.returncode == 0 and current_blob.stdout.strip() == ACTIVE_AUTHORITY_BLOB, "preserved authority blob differs from historical execution")
    committed_bytes = subprocess.run(
        ["git", "cat-file", "blob", ACTIVE_AUTHORITY_BLOB],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(committed_bytes.returncode == 0 and committed_bytes.stdout == REPLACEMENT_AUTHORITY.read_bytes(), "preserved authority working-tree bytes changed")
    require(json.loads(committed_bytes.stdout) == authority, "historical authority object differs from execution blob")
    return {
        "authorized_source_commit": source_commit,
        "authorized_source_tree_sha1": source_tree,
        "authorized_source_review_id": REPAIRED_SOURCE_REVIEW_ID,
        "execution_head": EXECUTION_HEAD,
        "replacement_authority_git_blob_sha1": ACTIVE_AUTHORITY_BLOB,
        "protected_execution_source_paths": list(future_runner.PROTECTED_EXECUTION_SOURCE_PATHS),
    }


def validate_committed_replacement_authority(authority: dict[str, Any]) -> dict[str, Any]:
    """Validate the one active authority against reviewed source commit A."""
    require(REPLACEMENT_AUTHORITY.is_file() and not REPLACEMENT_AUTHORITY.is_symlink(), "active replacement authority must be a real file")
    require(
        authority == future_runner.load_replacement_authority(REPLACEMENT_AUTHORITY),
        "active replacement authority object differs from exact artifact",
    )
    evidence_path = future_runner.validate_replacement_authority(authority, REPLACEMENT_AUTHORITY)
    custody = validate_historical_replacement_authority_custody(authority)
    require(authority["project_owner"] == EXPECTED_OWNER, "active authority owner mismatch")
    require(authority["authority_id"] == ACTIVE_AUTHORITY_ID, "active authority id mismatch")
    require(authority["authorized_source_commit"] == REPAIRED_SOURCE_COMMIT, "active authority source commit mismatch")
    require(authority["authorized_source_tree_sha1"] == REPAIRED_SOURCE_TREE, "active authority source tree mismatch")
    require(authority["authorized_source_review_id"] == REPAIRED_SOURCE_REVIEW_ID, "active authority source review mismatch")
    require(authority["maximum_target_qualification_executions"] == 1, "active authority execution cap mismatch")
    require(authority["automatic_retry"] is False, "active authority automatic retry must be false")
    require(authority["replacement_qualification_authorized"] is True, "active replacement qualification must be authorized")
    for field in future_runner.DOWNSTREAM_FALSE_FIELDS:
        require(authority[field] is False, f"active authority downstream field must be false: {field}")
    require(sha256_file(REPLACEMENT_AUTHORITY) == ACTIVE_AUTHORITY_SHA256, "active authority SHA-256 mismatch")
    require(git_blob(REPLACEMENT_AUTHORITY) == ACTIVE_AUTHORITY_BLOB, "active authority worktree blob mismatch")
    require(custody["replacement_authority_git_blob_sha1"] == ACTIVE_AUTHORITY_BLOB, "active authority committed blob mismatch")
    require(custody["authorized_source_commit"] == REPAIRED_SOURCE_COMMIT, "active custody source commit mismatch")
    require(custody["authorized_source_tree_sha1"] == REPAIRED_SOURCE_TREE, "active custody source tree mismatch")
    require(custody["authorized_source_review_id"] == REPAIRED_SOURCE_REVIEW_ID, "active custody source review mismatch")
    require(custody["execution_head"] == EXECUTION_HEAD, "active custody execution HEAD mismatch")
    require(evidence_path.resolve() == REPLACEMENT_EVIDENCE.resolve(), "active authority evidence namespace mismatch")
    require(evidence_path.is_dir() and not evidence_path.is_symlink(), "consumed active authority evidence directory missing")
    return {
        **custody,
        "status": "COMMITTED_REPLACEMENT_AUTHORITY_VALID_CONSUMED",
        "authority_id": authority["authority_id"],
        "authority_artifact_path": REPLACEMENT_AUTHORITY.relative_to(REPO_ROOT).as_posix(),
        "authority_artifact_sha256": ACTIVE_AUTHORITY_SHA256,
        "authority_consumed": True,
        "replacement_execution_count": 1,
        "replacement_evidence_dir": evidence_path.relative_to(REPO_ROOT).as_posix(),
        "replacement_evidence_present": True,
    }


def validate_replacement_evidence(
    evidence_dir: Path = REPLACEMENT_EVIDENCE,
    *,
    expected_binding: str = REPLACEMENT_EVIDENCE.relative_to(REPO_ROOT).as_posix(),
    require_committed: bool = True,
) -> dict[str, Any]:
    """Independently adjudicate the one replacement evidence namespace."""
    require(evidence_dir.is_dir() and not evidence_dir.is_symlink(), "replacement evidence directory missing")
    for path in evidence_dir.rglob("*"):
        require(not path.is_symlink(), f"replacement evidence contains symlink: {path}")

    inventory_path = evidence_dir / "EVIDENCE_INVENTORY.json"
    bindings_path = evidence_dir / "FINAL_BINDINGS.json"
    commands_path = evidence_dir / "COMMANDS.jsonl"
    require(sha256_file(inventory_path) == EVIDENCE_INVENTORY_SHA256, "replacement evidence inventory SHA-256 mismatch")
    require(sha256_file(bindings_path) == FINAL_BINDINGS_SHA256, "replacement final bindings SHA-256 mismatch")
    require(sha256_file(commands_path) == COMMANDS_SHA256, "replacement commands SHA-256 mismatch")

    inventory = load(inventory_path)
    require(set(inventory) == {"schema_id", "file_count", "files"}, "replacement evidence inventory key closure mismatch")
    require(inventory["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXEC_EVIDENCE_INVENTORY_V1", "replacement evidence inventory schema mismatch")
    require(isinstance(inventory["files"], list), "replacement evidence inventory files must be a list")
    require(inventory["file_count"] == len(inventory["files"]) == 78, "replacement evidence inventory count mismatch")
    listed_paths: set[str] = set()
    for entry in inventory["files"]:
        require(isinstance(entry, dict) and set(entry) == {"path", "sha256", "size"}, "replacement inventory entry malformed")
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and entry["path"] == relative.as_posix(), "unsafe replacement inventory path")
        require(entry["path"] not in listed_paths, "duplicate replacement inventory path")
        listed_paths.add(entry["path"])
        path = evidence_dir / relative
        require(path.is_file() and not path.is_symlink(), f"replacement inventory file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"], f"replacement inventory size mismatch: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"replacement inventory SHA-256 mismatch: {entry['path']}")
    actual_paths = {
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file() and path.name not in {"EVIDENCE_INVENTORY.json", "FINAL_BINDINGS.json"}
    }
    require(listed_paths == actual_paths, "replacement evidence inventory does not close over actual files")

    bindings = load(bindings_path)
    require(bindings["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXEC_ORCHESTRATION_V2", "replacement binding schema mismatch")
    require(bindings["overall_status"] == "SUCCESS", "replacement orchestration did not succeed")
    require(bindings["execution_head"] == EXECUTION_HEAD, "replacement execution HEAD mismatch")
    require(bindings["authorized_source_commit"] == REPAIRED_SOURCE_COMMIT, "replacement evidence source commit mismatch")
    require(bindings["authorized_source_tree_sha1"] == REPAIRED_SOURCE_TREE, "replacement evidence source tree mismatch")
    require(bindings["authorized_source_review_id"] == REPAIRED_SOURCE_REVIEW_ID, "replacement evidence review id mismatch")
    require(bindings["replacement_authority_id"] == ACTIVE_AUTHORITY_ID, "replacement evidence authority id mismatch")
    require(bindings["replacement_authority_sha256"] == ACTIVE_AUTHORITY_SHA256, "replacement evidence authority SHA-256 mismatch")
    require(bindings["replacement_authority_git_blob_sha1"] == ACTIVE_AUTHORITY_BLOB, "replacement evidence authority blob mismatch")
    require(bindings["replacement_evidence_dir"] == expected_binding, "replacement evidence path binding mismatch")
    require(bindings["_evidence_inventory_sha256"] == EVIDENCE_INVENTORY_SHA256, "final bindings inventory digest mismatch")
    require(bindings["maximum_target_qualification_executions"] == 1, "replacement maximum execution count mismatch")
    require(bindings["qualification_execution_count"] == 1, "replacement qualification execution count mismatch")
    require(bindings["automatic_retry"] is False, "replacement evidence automatic retry must be false")
    require(bindings["qualification_exit_code"] == 0, "replacement qualification exit code nonzero")
    require(bindings["qualification_status"] == "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED", "replacement qualification status mismatch")
    require(bindings["worker_validate_only_status"] == "GATE_A_WORKER_VALIDATE_ONLY_OK", "replacement worker status mismatch")

    authority = load(REPLACEMENT_AUTHORITY)
    with (
        mock.patch.object(future_runner, "EXEC_ROOT", authority["remote_execution_root"], create=True),
        mock.patch.object(future_runner, "TRANSFER_STAGE", authority["remote_transfer_stage"], create=True),
        mock.patch.object(future_runner, "EV_ARCHIVE", authority["remote_evidence_archive"], create=True),
        mock.patch.object(future_runner, "TP", authority["remote_temp_prefix"], create=True),
    ):
        future_runner.validate_remote_namespace_preflight(bindings["remote_namespace_preflight"])
    require(bindings["execution_root_predeploy_state"] == "ABSENT", "replacement execution root was not absent at preflight")
    require(bindings["transfer_stage_predeploy_state"] == "ABSENT", "replacement transfer stage was not absent at preflight")
    require(bindings["evidence_archive_predeploy_state"] == "ABSENT", "replacement evidence archive was not absent at preflight")
    require(bindings["temp_prefix_predeploy_state"] == "ABSENT", "replacement temp prefix was not absent at preflight")
    require(bindings["temp_prefix_predeploy_match_count"] == 0, "replacement temp prefix match count nonzero")
    require(bindings["temp_prefix_predeploy_matches"] == [], "replacement temp prefix matches present")

    for key in ("process_scan_before", "process_scan_after", "process_scan_after_cleanup"):
        future_runner.validate_process_scan(bindings[key], f"replacement {key}")
        require(bindings[key]["forbidden_process_hits"] == [], f"replacement {key} has forbidden process hits")
    qualification = bindings["qualification_json"]
    validate_qualification_json(qualification)
    require(qualification == load(evidence_dir / "copy_back" / "target_evidence" / "TARGET_QUALIFICATION_RESULT.json"), "copied qualification differs from final bindings")
    require(qualification["hardware_probes"] == 0, "replacement qualification probed hardware")
    require(qualification["sender_starts"] == 0, "replacement qualification started sender")
    require(qualification["receiver_captures"] == 0, "replacement qualification captured receiver")
    require(qualification["control_writes"] == 0, "replacement qualification wrote controls")
    require(qualification["msr_accesses"] == 0, "replacement qualification accessed MSRs")
    require(qualification["hardware_executions"] == 0, "replacement qualification executed hardware")
    require(bindings["strict_validation_before"] == bindings["strict_validation_after"], "strict bundle validation changed")
    require(bindings["bundle_tree_unchanged"] is True, "replacement bundle tree changed")
    require(bindings["before_tree_canonical_sha256"] == bindings["after_tree_canonical_sha256"], "replacement bundle tree digest changed")
    require(bindings["transfer_digest_match"] is True, "replacement transfer digest mismatch")
    require(bindings["copy_back_verified"] is True, "replacement copy-back not verified")
    require(bindings["cleanup_verified"] is True, "replacement cleanup not verified")
    require(bindings["execution_root_final_state"] == "ABSENT", "replacement execution root cleanup not proven")
    require(bindings["transfer_stage_final_state"] == "ABSENT", "replacement transfer stage cleanup not proven")
    require(bindings["current_identity_before"] == bindings["current_identity_after"], "replacement target identity changed")
    require(bindings["current_identity_before"]["hostname"] == EXPECTED_HOSTNAME, "replacement target hostname mismatch")
    require(bindings["current_identity_before"]["architecture"] == EXPECTED_ARCH, "replacement target architecture mismatch")
    require(bindings["current_identity_before"]["cpu_model"] == EXPECTED_CPU, "replacement target CPU mismatch")

    copy_back = load(evidence_dir / "COPY_BACK_RECEIPT.json")
    require(set(copy_back) == {"inventory_entry_count", "inventory_verified", "retained_evidence_custody_verified", "schema_id", "target_evidence_archive_sha256", "unexpected_entries"}, "replacement copy-back receipt key closure mismatch")
    require(copy_back["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1", "replacement copy-back schema mismatch")
    require(copy_back["inventory_entry_count"] == 15, "replacement copy-back inventory count mismatch")
    require(copy_back["inventory_verified"] is True and copy_back["retained_evidence_custody_verified"] is True, "replacement copy-back custody unverified")
    require(copy_back["unexpected_entries"] == [], "replacement copy-back contains unexpected entries")
    require(copy_back["target_evidence_archive_sha256"] == TARGET_EVIDENCE_ARCHIVE_SHA256, "replacement target archive receipt mismatch")
    require(sha256_file(evidence_dir / "copy_back" / "target_evidence.tar") == TARGET_EVIDENCE_ARCHIVE_SHA256, "replacement copied archive SHA-256 mismatch")

    copied_dir = evidence_dir / "copy_back" / "target_evidence"
    target_inventory_path = copied_dir / "TARGET_EVIDENCE_INVENTORY.json"
    target_inventory = json.loads(target_inventory_path.read_text(encoding="utf-8"))
    require(isinstance(target_inventory, list) and len(target_inventory) == 14, "replacement target inventory count mismatch")
    target_names: set[str] = set()
    for entry in target_inventory:
        require(isinstance(entry, dict) and set(entry) == {"mode", "path", "sha256", "size"}, "replacement target inventory entry malformed")
        require("/" not in entry["path"] and "\\" not in entry["path"] and entry["path"] not in {".", ".."}, "unsafe replacement target inventory path")
        require(entry["path"] not in target_names, "duplicate replacement target inventory path")
        target_names.add(entry["path"])
        path = copied_dir / entry["path"]
        require(path.is_file() and not path.is_symlink(), f"replacement copied target file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"] and sha256_file(path) == entry["sha256"], f"replacement copied target file mismatch: {entry['path']}")
    actual_target_names = {path.name for path in copied_dir.iterdir() if path.is_file()}
    require(target_names | {"TARGET_EVIDENCE_INVENTORY.json"} == actual_target_names, "replacement copied target inventory is not closed")
    bound_target_inventory = bindings["target_evidence_inventory"]
    require(isinstance(bound_target_inventory, list) and len(bound_target_inventory) == 15, "replacement bound target inventory count mismatch")
    bound_names: set[str] = set()
    for entry in bound_target_inventory:
        require(isinstance(entry, dict) and set(entry) == {"mode", "path", "sha256", "size"}, "replacement bound target inventory entry malformed")
        require(entry["path"] not in bound_names, "duplicate replacement bound target inventory path")
        bound_names.add(entry["path"])
        path = copied_dir / entry["path"]
        require(path.is_file() and not path.is_symlink(), f"replacement bound target file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"] and sha256_file(path) == entry["sha256"], f"replacement bound target file mismatch: {entry['path']}")
    require(bound_names == actual_target_names, "replacement bound target inventory is not closed")
    require(
        [entry for entry in bound_target_inventory if entry["path"] != "TARGET_EVIDENCE_INVENTORY.json"] == target_inventory,
        "replacement target inventory binding mismatch",
    )

    cleanup = load(evidence_dir / "CLEANUP_RECEIPT.json")
    require(set(cleanup) == {"schema_id", "exact_execution_root_removed", "exact_transfer_stage_removed", "execution_root_absence_proven", "transfer_stage_absence_proven", "forbidden_processes_remaining", "process_scan"}, "replacement cleanup receipt key closure mismatch")
    require(cleanup["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V2", "replacement cleanup schema mismatch")
    for key in ("exact_execution_root_removed", "exact_transfer_stage_removed", "execution_root_absence_proven", "transfer_stage_absence_proven"):
        require(cleanup[key] is True, f"replacement cleanup field false: {key}")
    require(cleanup["forbidden_processes_remaining"] == [], "replacement cleanup left forbidden processes")
    future_runner.validate_process_scan(cleanup["process_scan"], "replacement cleanup receipt process scan")
    require(cleanup["process_scan"] == bindings["process_scan_after_cleanup"], "cleanup process receipt differs from final bindings")

    command_lines = commands_path.read_text(encoding="utf-8").splitlines()
    commands = [json.loads(line) for line in command_lines]
    require(len(commands) == 22, "replacement command count mismatch")
    require([command["sequence"] for command in commands] == list(range(1, 23)), "replacement command sequence mismatch")
    require(all(command["exit_code"] == 0 for command in commands), "replacement command returned nonzero")
    qualification_commands = [
        command
        for command in commands
        if "gate_a_target_runner.py --qualify-no-drive" in " ".join(command["argv"])
    ]
    require(len(qualification_commands) == 1, "replacement qualification command did not execute exactly once")
    preflight_command = commands[4]
    require(preflight_command["stdin_script_path"].endswith("005_prove_four_way_absence.script.py"), "four-way preflight was not first target operation")
    require(set(preflight_command["environment"]) == {"ROOT", "STAGE", "EVARCHIVE", "TP"}, "four-way preflight environment was not read-only")
    require("006_target_identity_before" in commands[5]["stdout_path"], "temp-prefix write did not follow four-way preflight")
    for command in commands:
        argv_text = " ".join(command["argv"])
        for forbidden in FORBIDDEN_COMMAND_SUBSTRINGS:
            require(forbidden not in argv_text, f"forbidden replacement command detected: {forbidden}")

    git_custody: dict[str, Any] = {"required": require_committed}
    if require_committed:
        evidence_rel = REPLACEMENT_EVIDENCE.relative_to(REPO_ROOT).as_posix()
        require(evidence_dir.resolve() == REPLACEMENT_EVIDENCE.resolve(), "committed evidence custody requires canonical evidence path")
        tree = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", evidence_rel],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(tree.returncode == 0, f"cannot enumerate committed replacement evidence: {tree.stderr}")
        committed = {line for line in tree.stdout.splitlines() if line}
        expected_committed = {
            f"{evidence_rel}/{path.relative_to(evidence_dir).as_posix()}"
            for path in evidence_dir.rglob("*")
            if path.is_file()
        }
        require(committed == expected_committed, "current HEAD replacement evidence tree mismatch")
        worktree = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", evidence_rel],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(worktree.returncode == 0 and worktree.stdout == "", "replacement evidence differs from current HEAD")
        git_custody = {
            "required": True,
            "committed_file_count": len(committed),
            "current_head": future_runner.current_head(),
            "worktree_clean": True,
        }

    return {
        "status": "REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_COMPLETE",
        "execution_head": EXECUTION_HEAD,
        "authority_id": ACTIVE_AUTHORITY_ID,
        "qualification_execution_count": 1,
        "automatic_retry": False,
        "four_way_namespace_preflight": True,
        "process_scans_complete": 3,
        "forbidden_process_hits": 0,
        "copy_back_verified": True,
        "cleanup_verified": True,
        "evidence_file_count": inventory["file_count"] + 2,
        "evidence_inventory_sha256": EVIDENCE_INVENTORY_SHA256,
        "final_bindings_sha256": FINAL_BINDINGS_SHA256,
        "target_evidence_archive_sha256": TARGET_EVIDENCE_ARCHIVE_SHA256,
        "git_custody": git_custody,
        "target_nonexecuting_qualification_complete": True,
        "execution_bundle_target_qualified": True,
    }


def validate_replacement_authority_state(
    state: dict[str, Any],
    schema: dict[str, Any],
    active_custody: dict[str, Any],
    superseded_custody: dict[str, Any],
) -> None:
    validate_const_instance(state, schema, "replacement authority state")
    require(state["status"] == COMPLETED_CURRENT_STATUS, "completed current status mismatch")
    require(state["superseded_authority_id"] == superseded_custody["authority_id"], "superseded authority state id mismatch")
    require(state["superseded_authority_artifact_path"] == superseded_custody["authority_artifact_path"], "superseded authority path mismatch")
    require(state["superseded_authority_artifact_sha256"] == superseded_custody["authority_artifact_sha256"], "superseded authority SHA-256 mismatch")
    require(state["superseded_authority_artifact_git_blob_sha1"] == superseded_custody["authority_artifact_git_blob_sha1"], "superseded authority blob mismatch")
    require(state["superseded_authority_consumed"] is False, "superseded authority cannot be consumed")
    require(state["superseded_replacement_execution_count"] == 0, "superseded authority execution count must remain zero")
    require(state["active_authority_id"] == active_custody["authority_id"], "active authority state id mismatch")
    require(state["active_authority_artifact_path"] == active_custody["authority_artifact_path"], "active authority path mismatch")
    require(state["active_authority_artifact_sha256"] == active_custody["authority_artifact_sha256"], "active authority SHA-256 mismatch")
    require(state["active_authority_artifact_git_blob_sha1"] == active_custody["replacement_authority_git_blob_sha1"], "active authority blob mismatch")
    require(state["authorized_source_commit"] == active_custody["authorized_source_commit"], "active authority source commit mismatch")
    require(state["authorized_source_tree_sha1"] == active_custody["authorized_source_tree_sha1"], "active authority source tree mismatch")
    require(state["authorized_source_review_id"] == active_custody["authorized_source_review_id"], "active authority review id mismatch")
    require(state["authority_execution_head_recorded_dynamically"] is True, "authority state must use dynamic execution HEAD")
    require(state["execution_head"] == EXECUTION_HEAD, "authority state execution HEAD mismatch")
    require(state["active_replacement_authority_present"] is True, "active authority must be present")
    require(state["new_replacement_authority_artifact_created"] is True, "new authority artifact must be recorded")
    require(state["replacement_qualification_was_authorized"] is True, "replacement qualification authorization history missing")
    require(state["replacement_qualification_authorized"] is False, "consumed authority must not authorize another target contact")
    require(state["orchestrator_only_no_drive_qualification"] is True, "replacement authority must remain orchestrator-only and no-drive")
    require(state["authority_consumed"] is True, "active authority must be consumed after execution")
    require(state["replacement_execution_count"] == 1, "active replacement execution count must be one")
    require(state["replacement_evidence_present"] is True, "active replacement evidence must be present")
    require(state["replacement_evidence_dir"] == active_custody["replacement_evidence_dir"], "active replacement evidence namespace mismatch")
    require(state["automatic_retry"] is False, "active authority automatic retry must be false")
    require(state["replacement_result_status"] == "SUCCESS", "replacement result status mismatch")
    require(state["final_bindings_sha256"] == FINAL_BINDINGS_SHA256, "authority state final bindings digest mismatch")
    require(state["evidence_inventory_sha256"] == EVIDENCE_INVENTORY_SHA256, "authority state evidence inventory digest mismatch")
    require(state["copy_back_verified"] is True, "authority state copy-back must be verified")
    require(state["cleanup_verified"] is True, "authority state cleanup must be verified")
    require(state["process_absence_proven"] is True, "authority state process absence must be proven")
    require(state["four_way_remote_namespace_preflight_passed"] is True, "authority state four-way preflight must pass")
    require(state["target_nonexecuting_qualification_complete"] is True, "target qualification must be complete")
    require(state["execution_bundle_target_qualified"] is True, "target bundle must be qualified")
    for field in DOWNSTREAM_FALSE_FIELDS:
        require(state[field] is False, f"active authority downstream field must be false: {field}")
    require(state["next_boundary"] == COMPLETED_NEXT_BOUNDARY, "completed authority next boundary mismatch")


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
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", INTEGRATED_MAIN, "HEAD"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(ancestry.returncode == 0, "historical adapter commit is not an ancestor of HEAD")
    for name, blob in EXPECTED_BLOBS.items():
        actual = git_blob_at(INTEGRATED_MAIN, ADAPTER / name)
        require(actual == blob, f"historical adapter blob {name} mismatch: {actual} != {blob}")
    manifest_sha = git_sha256_at(INTEGRATED_MAIN, ADAPTER / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json")
    require(manifest_sha == MANIFEST_FILE_SHA, f"historical adapter manifest file sha mismatch: {manifest_sha}")


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


def synthetic_replacement_authority(
    source_commit: str,
    source_tree_sha1: str,
    *,
    authority_id: str = "synthetic_owner_decision_test",
) -> dict[str, Any]:
    local_evidence_dir = (
        future_runner.EVIDENCE_PARENT.relative_to(future_runner.REPO_ROOT).as_posix()
        + f"/gate_a_target_nonexecuting_qualification_replacement_{authority_id}"
    )
    return {
        "schema_id": future_runner.REPLACEMENT_AUTHORITY_SCHEMA_ID,
        "decision": future_runner.REPLACEMENT_AUTHORITY_DECISION,
        "project_owner": EXPECTED_OWNER,
        "owner_instruction": "Authorize one synthetic contract-validation replacement only",
        "authority_id": authority_id,
        "authorized_source_commit": source_commit,
        "authorized_source_tree_sha1": source_tree_sha1,
        "authorized_source_review_id": future_runner.AUTHORIZED_SOURCE_REVIEW_ID,
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


def future_replacement_authority_contract_test() -> dict[str, Any]:
    source_commit = future_runner.current_head()
    source_tree = subprocess.run(
        ["git", "rev-parse", f"{source_commit}^{{tree}}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    authority = synthetic_replacement_authority(source_commit, source_tree)
    with tempfile.TemporaryDirectory(prefix="gate_a_authority_contract_") as temp_dir:
        authority_path = Path(temp_dir) / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
        evidence_path = future_runner.validate_replacement_authority(
            authority,
            authority_path,
        )
    expected_evidence_path = future_runner.REPO_ROOT / authority["local_evidence_dir"]
    require(evidence_path == expected_evidence_path, "future authority resolved wrong evidence namespace")
    future_runner.ensure_new_evidence_namespace(evidence_path)
    return {
        "status": "SYNTHETIC_REPLACEMENT_AUTHORITY_CONTRACT_VALID",
        "closed_key_count": len(authority),
        "two_commit_source_binding": True,
        "authorized_source_review_id": authority["authorized_source_review_id"],
        "historical_namespace_reused": False,
        "automatic_retry": authority["automatic_retry"],
        "downstream_authority_false": True,
    }


def replacement_authority_two_commit_git_integration_test() -> dict[str, Any]:
    """Exercise the production custody validator in a disposable Git repository."""

    def git(root: Path, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise VerifyError(f"temporary Git command failed: git {' '.join(args)}: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def write_authority(path: Path, source_commit: str, source_tree: str, authority_id: str) -> dict[str, Any]:
        authority = synthetic_replacement_authority(
            source_commit,
            source_tree,
            authority_id=authority_id,
        )
        path.write_text(json.dumps(authority, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        future_runner.validate_replacement_authority(authority, path)
        return authority

    def custody(root: Path, path: Path, authority: dict[str, Any]) -> dict[str, Any]:
        return future_runner.validate_replacement_authority_custody(
            path,
            authority["authorized_source_commit"],
            authority["authorized_source_tree_sha1"],
            expected_authority=authority,
            repo_root=root,
            protected_paths=("reviewed_runner.py",),
        )

    with tempfile.TemporaryDirectory(prefix="gate_a_two_commit_git_") as temp_dir:
        root = Path(temp_dir)
        git(root, "init", "-b", "main")
        git(root, "config", "user.name", "Gate A Integration Test")
        git(root, "config", "user.email", "gate-a-test@example.invalid")
        git(root, "config", "core.autocrlf", "false")

        runner_path = root / "reviewed_runner.py"
        authority_path = root / "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
        runner_path.write_text("print('reviewed runner')\n", encoding="utf-8")
        git(root, "add", "reviewed_runner.py")
        git(root, "commit", "-m", "commit A reviewed source")
        commit_a = git(root, "rev-parse", "HEAD")
        tree_a = git(root, "rev-parse", f"{commit_a}^{{tree}}")

        unrelated = git(root, "commit-tree", tree_a, "-m", "unrelated source object")

        authority_b = write_authority(authority_path, commit_a, tree_a, "synthetic_commit_b_success")
        git(root, "add", authority_path.name)
        git(root, "commit", "-m", "commit B owner authority")
        commit_b = git(root, "rev-parse", "HEAD")
        success = custody(root, authority_path, authority_b)
        require(success["authorized_source_commit"] == commit_a, "commit A source binding lost")
        require(success["execution_head"] == commit_b, "commit B execution HEAD not recorded")

        self_referential_authority = copy.deepcopy(authority_b)
        self_referential_authority["authorized_source_commit"] = commit_b
        self_referential_authority["authorized_source_tree_sha1"] = git(root, "rev-parse", f"{commit_b}^{{tree}}")
        self_reference_case = assert_rejects(
            "two_commit_self_reference_rejected",
            lambda: custody(root, authority_path, self_referential_authority),
        )

        original_authority_bytes = authority_path.read_bytes()
        authority_path.write_bytes(original_authority_bytes + b" ")
        modified_authority_case = assert_rejects(
            "two_commit_modified_authority_rejected",
            lambda: custody(root, authority_path, authority_b),
        )
        authority_path.write_bytes(original_authority_bytes)

        git(root, "switch", "-c", "runner-drift", commit_a)
        runner_path.write_text("print('drifted runner')\n", encoding="utf-8")
        drift_authority = write_authority(authority_path, commit_a, tree_a, "synthetic_commit_b_drift")
        git(root, "add", "reviewed_runner.py", authority_path.name)
        git(root, "commit", "-m", "commit B with forbidden runner drift")
        runner_drift_case = assert_rejects(
            "two_commit_runner_drift_rejected",
            lambda: custody(root, authority_path, drift_authority),
        )

        git(root, "switch", "-c", "nonancestor", commit_a)
        unrelated_authority = write_authority(
            authority_path,
            unrelated,
            tree_a,
            "synthetic_nonancestor_source",
        )
        git(root, "add", authority_path.name)
        git(root, "commit", "-m", "authority naming unrelated source")
        nonancestor_case = assert_rejects(
            "two_commit_nonancestor_source_rejected",
            lambda: custody(root, authority_path, unrelated_authority),
        )

        git(root, "switch", "-c", "uncommitted-authority", commit_a)
        (root / "authority_later_marker.txt").write_text("later commit without authority\n", encoding="utf-8")
        git(root, "add", "authority_later_marker.txt")
        git(root, "commit", "-m", "later commit without authority")
        uncommitted_authority = write_authority(
            authority_path,
            commit_a,
            tree_a,
            "synthetic_uncommitted_authority",
        )
        uncommitted_case = assert_rejects(
            "two_commit_uncommitted_authority_rejected",
            lambda: custody(root, authority_path, uncommitted_authority),
        )

    return {
        "status": "TWO_COMMIT_REPLACEMENT_AUTHORITY_GIT_INTEGRATION_PASS",
        "commit_a_reviewed_source": True,
        "commit_b_authority_blob_bound": True,
        "execution_head_recorded_dynamically": True,
        "rejection_cases": [
            runner_drift_case,
            nonancestor_case,
            uncommitted_case,
            modified_authority_case,
            self_reference_case,
        ],
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


def remote_namespace_preflight_contract_test() -> dict[str, Any]:
    """Exercise the four-way, read-only, fail-closed remote preflight locally."""
    cases: list[str] = []

    def snapshot(root: Path) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for path in [root, *sorted(root.rglob("*"))]:
            stat_result = path.lstat()
            relative = "." if path == root else path.relative_to(root).as_posix()
            entry: dict[str, Any] = {
                "mode": stat_result.st_mode,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
            }
            if path.is_file() and not path.is_symlink():
                entry["sha256"] = sha256_file(path)
            if path.is_symlink():
                entry["symlink_target"] = os.readlink(path)
            result[relative] = entry
        return result

    def execute_script(execution_root: Path, transfer_stage: Path, evidence_archive: Path, temp_prefix: Path) -> dict[str, Any]:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(execution_root),
                "STAGE": str(transfer_stage),
                "EVARCHIVE": str(evidence_archive),
                "TP": str(temp_prefix),
            }
        )
        proc = subprocess.run(
            [sys.executable, "-c", future_runner.absence_script()],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(proc.returncode == 0, f"remote namespace preflight script failed closed without receipt: {proc.stderr}")
        value = json.loads(proc.stdout)
        require(isinstance(value, dict), "remote namespace preflight script did not return an object")
        return value

    def validate_for(
        report: dict[str, Any],
        execution_root: Path,
        transfer_stage: Path,
        evidence_archive: Path,
        temp_prefix: Path,
    ) -> None:
        with (
            mock.patch.object(future_runner, "EXEC_ROOT", str(execution_root), create=True),
            mock.patch.object(future_runner, "TRANSFER_STAGE", str(transfer_stage), create=True),
            mock.patch.object(future_runner, "EV_ARCHIVE", str(evidence_archive), create=True),
            mock.patch.object(future_runner, "TP", str(temp_prefix), create=True),
        ):
            future_runner.validate_remote_namespace_preflight(report)

    with tempfile.TemporaryDirectory(prefix="gate_a_namespace_preflight_") as temp_dir:
        root = Path(temp_dir)
        execution_root = root / "execution_root"
        transfer_stage = root / "transfer_stage.deploy.tar"
        evidence_archive = root / "evidence_archive.tar"
        temp_prefix = root / "remote_temp_prefix_"
        sentinel = root / "unrelated_sentinel.txt"
        sentinel.write_text("unchanged\n", encoding="utf-8")

        before = snapshot(root)
        valid = execute_script(execution_root, transfer_stage, evidence_archive, temp_prefix)
        after = snapshot(root)
        require(before == after, "remote namespace preflight changed the inspected filesystem")
        validate_for(valid, execution_root, transfer_stage, evidence_archive, temp_prefix)
        require(valid["temp_prefix"]["match_count"] == 0, "valid preflight did not prove zero prefix matches")
        cases.extend(("all_four_namespaces_absent", "preflight_script_filesystem_immutable"))

        execution_root.mkdir()
        cases.append(assert_rejects("execution_root_present", lambda: validate_for(execute_script(execution_root, transfer_stage, evidence_archive, temp_prefix), execution_root, transfer_stage, evidence_archive, temp_prefix)))
        execution_root.rmdir()

        transfer_stage.write_bytes(b"collision")
        cases.append(assert_rejects("transfer_stage_present", lambda: validate_for(execute_script(execution_root, transfer_stage, evidence_archive, temp_prefix), execution_root, transfer_stage, evidence_archive, temp_prefix)))
        transfer_stage.unlink()

        evidence_archive.write_bytes(b"collision")
        cases.append(assert_rejects("evidence_archive_present", lambda: validate_for(execute_script(execution_root, transfer_stage, evidence_archive, temp_prefix), execution_root, transfer_stage, evidence_archive, temp_prefix)))
        evidence_archive.unlink()

        prefix_file = root / "remote_temp_prefix_one"
        prefix_dir = root / "remote_temp_prefix_two"
        prefix_file.write_bytes(b"collision")
        prefix_dir.mkdir()
        prefix_collision = execute_script(execution_root, transfer_stage, evidence_archive, temp_prefix)
        require(prefix_collision["temp_prefix"]["match_count"] == 2, "prefix collision count not bound")
        cases.append(assert_rejects("temp_prefix_matches_present", lambda: validate_for(prefix_collision, execution_root, transfer_stage, evidence_archive, temp_prefix)))
        prefix_file.unlink()
        prefix_dir.rmdir()

        unobservable_prefix = root / "missing_parent" / "remote_temp_prefix_"
        unobservable = execute_script(execution_root, transfer_stage, evidence_archive, unobservable_prefix)
        require(unobservable["temp_prefix"]["state"] == "UNOBSERVABLE", "prefix inspection error was not bound")
        cases.append(assert_rejects("temp_prefix_unobservable", lambda: validate_for(unobservable, execution_root, transfer_stage, evidence_archive, unobservable_prefix)))

        malformed_cases: list[tuple[str, Any]] = []
        missing = copy.deepcopy(valid)
        del missing["evidence_archive"]
        malformed_cases.append(("preflight_missing_key", missing))
        opened = copy.deepcopy(valid)
        opened["unexpected"] = False
        malformed_cases.append(("preflight_extra_key", opened))
        incomplete = copy.deepcopy(valid)
        incomplete["inspection_complete"] = False
        malformed_cases.append(("preflight_incomplete", incomplete))
        exact_unobservable = copy.deepcopy(valid)
        exact_unobservable["execution_root"].update(
            {"state": "UNOBSERVABLE", "error_type": "PermissionError", "error_message": "denied"}
        )
        malformed_cases.append(("execution_root_unobservable", exact_unobservable))
        inconsistent_count = copy.deepcopy(valid)
        inconsistent_count["temp_prefix"]["match_count"] = 1
        malformed_cases.append(("temp_prefix_count_inconsistent", inconsistent_count))
        boolean_count = copy.deepcopy(valid)
        boolean_count["temp_prefix"]["match_count"] = False
        malformed_cases.append(("temp_prefix_count_boolean", boolean_count))
        prefix_open = copy.deepcopy(valid)
        prefix_open["temp_prefix"]["unexpected"] = 0
        malformed_cases.append(("temp_prefix_observation_open", prefix_open))
        for name, malformed in malformed_cases:
            cases.append(assert_rejects(name, lambda value=malformed: validate_for(value, execution_root, transfer_stage, evidence_archive, temp_prefix)))
        cases.append(assert_rejects("preflight_non_object", lambda: validate_for([], execution_root, transfer_stage, evidence_archive, temp_prefix)))  # type: ignore[arg-type]

        with (
            mock.patch.object(future_runner, "EXEC_ROOT", str(execution_root), create=True),
            mock.patch.object(future_runner, "TRANSFER_STAGE", str(transfer_stage), create=True),
            mock.patch.object(future_runner, "EV_ARCHIVE", str(evidence_archive), create=True),
            mock.patch.object(future_runner, "TP", str(temp_prefix), create=True),
            mock.patch.object(future_runner, "run_ssh_py", return_value={"receipt": True}) as remote_call,
            mock.patch.object(future_runner, "parse_json_stdout", return_value=valid),
        ):
            require(future_runner.run_remote_namespace_preflight() == valid, "preflight helper changed validated receipt")
            remote_call.assert_called_once_with(
                "prove_four_way_absence",
                future_runner.absence_script(),
                {
                    "ROOT": str(execution_root),
                    "STAGE": str(transfer_stage),
                    "EVARCHIVE": str(evidence_archive),
                    "TP": str(temp_prefix),
                },
                subdir="target",
            )
            require("OUT" not in remote_call.call_args.args[2], "read-only preflight unexpectedly received an output path")
        cases.append("preflight_helper_exact_environment_no_output_path")

        with (
            mock.patch.object(future_runner, "EXEC_ROOT", str(execution_root), create=True),
            mock.patch.object(future_runner, "TRANSFER_STAGE", str(transfer_stage), create=True),
            mock.patch.object(future_runner, "EV_ARCHIVE", str(evidence_archive), create=True),
            mock.patch.object(future_runner, "TP", str(temp_prefix), create=True),
            mock.patch.object(future_runner, "run_ssh_py", return_value={"receipt": True}) as failed_remote_call,
            mock.patch.object(future_runner, "parse_json_stdout", return_value=exact_unobservable),
        ):
            cases.append(assert_rejects("failed_preflight_stops_helper", future_runner.run_remote_namespace_preflight))
            require(failed_remote_call.call_count == 1, "failed preflight issued more than one remote operation")

    main_source = inspect.getsource(future_runner.main)
    preflight_index = main_source.index("namespace_preflight = run_remote_namespace_preflight()")
    identity_index = main_source.index('run_ssh_py("target_identity_before"')
    require(preflight_index < identity_index, "temp-prefix identity write precedes remote namespace preflight")
    before_preflight = main_source[:preflight_index]
    for forbidden_call in ("run_ssh(", "run_ssh_py(", "scp_to(", "scp_from("):
        require(forbidden_call not in before_preflight, f"remote operation precedes namespace preflight: {forbidden_call}")
    preflight_script = future_runner.absence_script()
    for forbidden_write in ("open(", "mkdir", "makedirs", "unlink", "remove(", "rmdir", "rename(", "replace("):
        require(forbidden_write not in preflight_script, f"preflight script contains filesystem write primitive: {forbidden_write}")
    cases.append("preflight_is_first_remote_operation_before_temp_prefix_write")

    return {
        "status": "FOUR_WAY_REMOTE_NAMESPACE_PREFLIGHT_CONTRACT_PASS",
        "case_count": len(cases),
        "cases": cases,
        "filesystem_write_count": 0,
        "remote_temp_prefix_match_requirement": 0,
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
    replacement_authority: dict[str, Any],
    superseded_replacement_authority: dict[str, Any],
    replacement_authority_state: dict[str, Any],
    replacement_authority_state_schema: dict[str, Any],
    replacement_authority_custody: dict[str, Any],
    superseded_authority_custody: dict[str, Any],
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
    replacement_authority_validator = lambda value: validate_committed_replacement_authority(value)
    superseded_authority_validator = lambda value: validate_superseded_replacement_authority(value)
    replacement_authority_state_validator = lambda value: validate_replacement_authority_state(
        value,
        replacement_authority_state_schema,
        replacement_authority_custody,
        superseded_authority_custody,
    )

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
        ("replacement_authority_wrong_source_commit", mutated(replacement_authority, lambda value: value.__setitem__("authorized_source_commit", "0" * 40), replacement_authority_validator)),
        ("replacement_authority_wrong_source_tree", mutated(replacement_authority, lambda value: value.__setitem__("authorized_source_tree_sha1", "0" * 40), replacement_authority_validator)),
        ("replacement_authority_wrong_review_label", mutated(replacement_authority, lambda value: value.__setitem__("authorized_source_review_id", "WRONG_REVIEW"), replacement_authority_validator)),
        ("replacement_authority_execution_count_above_one", mutated(replacement_authority, lambda value: value.__setitem__("maximum_target_qualification_executions", 2), replacement_authority_validator)),
        ("replacement_authority_automatic_retry_true", mutated(replacement_authority, lambda value: value.__setitem__("automatic_retry", True), replacement_authority_validator)),
        ("superseded_authority_bytes_changed", mutated(superseded_replacement_authority, lambda value: value.__setitem__("authority_id", "gate_a_replacement_mutated_01"), superseded_authority_validator)),
        ("superseded_authority_falsely_consumed", mutated(replacement_authority_state, lambda value: value.__setitem__("superseded_authority_consumed", True), replacement_authority_state_validator)),
        ("superseded_authority_execution_count_nonzero", mutated(replacement_authority_state, lambda value: value.__setitem__("superseded_replacement_execution_count", 1), replacement_authority_state_validator)),
        ("authority_state_automatic_retry_true", mutated(replacement_authority_state, lambda value: value.__setitem__("automatic_retry", True), replacement_authority_state_validator)),
        ("authority_state_active_authority_hidden", mutated(replacement_authority_state, lambda value: value.__setitem__("active_replacement_authority_present", False), replacement_authority_state_validator)),
        ("authority_state_new_authority_artifact_hidden", mutated(replacement_authority_state, lambda value: value.__setitem__("new_replacement_authority_artifact_created", False), replacement_authority_state_validator)),
        ("authority_state_reauthorized_after_consumption", mutated(replacement_authority_state, lambda value: value.__setitem__("replacement_qualification_authorized", True), replacement_authority_state_validator)),
        ("authority_state_authorization_history_removed", mutated(replacement_authority_state, lambda value: value.__setitem__("replacement_qualification_was_authorized", False), replacement_authority_state_validator)),
        ("authority_state_orchestrator_only_removed", mutated(replacement_authority_state, lambda value: value.__setitem__("orchestrator_only_no_drive_qualification", False), replacement_authority_state_validator)),
        ("authority_state_unconsumed_after_execution", mutated(replacement_authority_state, lambda value: value.__setitem__("authority_consumed", False), replacement_authority_state_validator)),
        ("authority_state_execution_count_reset", mutated(replacement_authority_state, lambda value: value.__setitem__("replacement_execution_count", 0), replacement_authority_state_validator)),
        ("authority_state_evidence_hidden", mutated(replacement_authority_state, lambda value: value.__setitem__("replacement_evidence_present", False), replacement_authority_state_validator)),
        ("authority_state_result_status_changed", mutated(replacement_authority_state, lambda value: value.__setitem__("replacement_result_status", "FAILURE"), replacement_authority_state_validator)),
        ("authority_state_copy_back_unverified", mutated(replacement_authority_state, lambda value: value.__setitem__("copy_back_verified", False), replacement_authority_state_validator)),
        ("authority_state_cleanup_unverified", mutated(replacement_authority_state, lambda value: value.__setitem__("cleanup_verified", False), replacement_authority_state_validator)),
        ("authority_state_target_qualification_demoted", mutated(replacement_authority_state, lambda value: value.__setitem__("target_nonexecuting_qualification_complete", False), replacement_authority_state_validator)),
        ("authority_state_target_bundle_demoted", mutated(replacement_authority_state, lambda value: value.__setitem__("execution_bundle_target_qualified", False), replacement_authority_state_validator)),
    ]
    for field in future_runner.DOWNSTREAM_FALSE_FIELDS:
        checks.append((f"replacement_authority_downstream_{field}_true", mutated(replacement_authority, lambda value, key=field: value.__setitem__(key, True), replacement_authority_validator)))
    for field in DOWNSTREAM_FALSE_FIELDS:
        checks.append((f"adjudication_downstream_{field}_true", mutated(adjudication, lambda value, key=field: value.__setitem__(key, True), adjudication_validator)))
        checks.append((f"candidate_v4_downstream_{field}_true", mutated(candidate_v4, lambda value, key=field: value.__setitem__(key, True), candidate_v4_validator)))
        checks.append((f"authority_state_downstream_{field}_true", mutated(replacement_authority_state, lambda value, key=field: value.__setitem__(key, True), replacement_authority_state_validator)))
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

    def validate_mutated_replacement_evidence(mutator: Callable[[Path], None]) -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_replacement_evidence_mutation_") as temp_dir:
            copied = Path(temp_dir) / REPLACEMENT_EVIDENCE.name
            shutil.copytree(REPLACEMENT_EVIDENCE, copied)
            mutator(copied)
            validate_replacement_evidence(copied, require_committed=False)

    cases.append(assert_rejects(
        "replacement_evidence_file_changes",
        lambda: validate_mutated_replacement_evidence(
            lambda copied: (copied / "FINAL_BINDINGS.json").write_bytes(
                (copied / "FINAL_BINDINGS.json").read_bytes() + b"\n"
            )
        ),
    ))
    cases.append(assert_rejects(
        "replacement_evidence_file_removed",
        lambda: validate_mutated_replacement_evidence(
            lambda copied: (copied / "target" / "logs" / "005_prove_four_way_absence.stdout.txt").unlink()
        ),
    ))
    cases.append(assert_rejects(
        "replacement_evidence_extra_file",
        lambda: validate_mutated_replacement_evidence(
            lambda copied: (copied / "UNEXPECTED_EXTRA.txt").write_text("x", encoding="utf-8")
        ),
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
                "0" * 40,
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
    replacement_authority = load(REPLACEMENT_AUTHORITY)
    superseded_replacement_authority = load(SUPERSEDED_REPLACEMENT_AUTHORITY)
    replacement_authority_state = load(REPLACEMENT_AUTHORITY_STATE)
    result_schema = load(RESULT_SCHEMA)
    adjudication_schema = load(ADJUDICATION_SCHEMA)
    candidate_v4_schema = load(CANDIDATE_V4_SCHEMA)
    replacement_authority_state_schema = load(REPLACEMENT_AUTHORITY_STATE_SCHEMA)

    validate_schema_closed(
        result_schema,
        ("target", "bindings", "predecessor_adapter_package_digests", "authority_false_state"),
    )
    validate_schema_closed(adjudication_schema)
    validate_schema_closed(candidate_v4_schema, ("predecessor_adapter_package_digests",))
    validate_schema_closed(replacement_authority_state_schema)

    validate_historical_authorization(authorization)
    validate_historical_contract(contract)
    validate_historical_result(result)
    validate_historical_candidate_v3(candidate_v3)
    verify_adapter_blobs()
    evidence = validate_historical_evidence()
    defect = detect_historical_process_evidence_defect()
    validate_adjudication(adjudication, adjudication_schema, defect)
    validate_candidate_v4(candidate_v4, candidate_v4_schema)
    superseded_authority_custody = validate_superseded_replacement_authority(superseded_replacement_authority)
    replacement_authority_custody = validate_committed_replacement_authority(replacement_authority)
    replacement_evidence = validate_replacement_evidence()
    validate_replacement_authority_state(
        replacement_authority_state,
        replacement_authority_state_schema,
        replacement_authority_custody,
        superseded_authority_custody,
    )
    require_no_unaccented_owner()
    gate_a_execution_authority = validate_gate_a_execution_authority_state()

    scanner_test = future_scanner_nonzero_test()
    scanner_receipt_test = future_process_scan_receipt_contract_test()
    namespace_preflight_test = remote_namespace_preflight_contract_test()
    future_authority_test = future_replacement_authority_contract_test()
    two_commit_git_test = replacement_authority_two_commit_git_integration_test()
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
        replacement_authority,
        superseded_replacement_authority,
        replacement_authority_state,
        replacement_authority_state_schema,
        replacement_authority_custody,
        superseded_authority_custody,
    )

    output = {
        "status": COMPLETED_CURRENT_STATUS,
        "integrated_main": INTEGRATED_MAIN,
        "pre_repair_pr_head": PRE_REPAIR_HEAD,
        "historical_attempt_authorized": True,
        "historical_attempt_execution_count": 1,
        "historical_evidence_immutable": immutability,
        "process_evidence_defect": defect,
        "target_nonexecuting_qualification_complete": True,
        "execution_bundle_target_qualified": True,
        "original_authority_consumed": True,
        "replacement_qualification_authorized": False,
        "project_owner_execution_approval_recorded": gate_a_execution_authority["authority_artifact_present"],
        "authorization_artifact_created": gate_a_execution_authority["authority_artifact_present"],
        "active_replacement_authority_present": True,
        "new_replacement_authority_artifact_created": True,
        "superseded_replacement_authority_preserved": True,
        "superseded_replacement_authority_state": "SUPERSEDED_UNCONSUMED",
        "replacement_authority_consumed": True,
        "replacement_execution_count": 1,
        "replacement_authority_custody": replacement_authority_custody,
        "replacement_evidence_adjudication": replacement_evidence,
        "superseded_replacement_authority_custody": superseded_authority_custody,
        "engineering_smoke_authorized": gate_a_execution_authority["authority_artifact_present"],
        "hardware_ran": False,
        "automatic_retry": False,
        "gate_a_execution_authority": gate_a_execution_authority,
        "future_scanner_nonzero_test": scanner_test,
        "future_process_scan_receipt_contract_test": scanner_receipt_test,
        "remote_namespace_preflight_contract_test": namespace_preflight_test,
        "future_replacement_authority_contract_test": future_authority_test,
        "replacement_authority_two_commit_git_integration_test": two_commit_git_test,
        "mutation_tests": mutations,
        "historical_evidence_inventory_sha256": sha256_file(EVID / "EVIDENCE_INVENTORY.json"),
        "candidate_v4_preserved_predecision_state": True,
        "repaired_source_review_id": REPAIRED_SOURCE_REVIEW_ID,
        "repaired_source_commit": REPAIRED_SOURCE_COMMIT,
        "repaired_source_tree_sha1": REPAIRED_SOURCE_TREE,
        "execution_head": EXECUTION_HEAD,
        "current_verification_head": future_runner.current_head(),
        "next_boundary": COMPLETED_NEXT_BOUNDARY,
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
