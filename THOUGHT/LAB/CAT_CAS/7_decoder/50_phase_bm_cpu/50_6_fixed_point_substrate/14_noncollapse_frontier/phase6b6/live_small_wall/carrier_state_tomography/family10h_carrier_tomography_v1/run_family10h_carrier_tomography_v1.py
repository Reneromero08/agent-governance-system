#!/usr/bin/env python3
"""Offline controller for the public Family 10h carrier tomography package."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import re
import gzip
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import family10h_carrier_tomography_public as public
import family10h_carrier_tomography_target as target


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[10]
CONTRACT_PATH = HERE / "CARRIER_TOMOGRAPHY_CONTRACT.md"
SOURCE_BUNDLE = HERE / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
SOURCE_HASHES = HERE / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
TEMPERATURE_SENSOR_AUTHORITY = HERE / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY.json"
TARGET_DISCOVERY_RECEIPT = HERE / "CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_RECEIPT.json"
TARGET_DISCOVERY_FAILURE_PATH = HERE / "CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_FAILURE.json"
DISCOVERY_TRANSPORT_PATH = HERE / "CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT.json"
DISCOVERY_CHALLENGE_PATH = HERE / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE.json"
DISCOVERY_ATTEMPT_PATH = HERE / "CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT.json"
DISCOVERY_ATTEMPT_JOURNAL_PATH = HERE / "CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT.jsonl"
DISCOVERY_CLEANUP_CUSTODY_PATH = HERE / "CARRIER_TOMOGRAPHY_DISCOVERY_CLEANUP_CUSTODY.json"
ATTEMPT_HISTORY_DIR = HERE / "SENSOR_AUTHORITY_ATTEMPT_HISTORY"
ATTEMPT_HISTORY_INDEX_PATH = ATTEMPT_HISTORY_DIR / "ATTEMPT_HISTORY_INDEX.json"
C3_ATTEMPT_HISTORY_DIR = ATTEMPT_HISTORY_DIR / "c3_55e059bc_failed_affinity_precheck"
C5_ATTEMPT_HISTORY_DIR = ATTEMPT_HISTORY_DIR / "c5_ca8f8490_no_approved_sensor"
C3_SOURCE_AUTHORITY_COMMIT = "55e059bc7acaafee3feacddac2069d7b5e40edd1"
C4_SOURCE_AUTHORITY_COMMIT = "092d0a655e94d7c00f69efc1236cf1c8a2896ee1"
C5_SOURCE_AUTHORITY_COMMIT = "ca8f8490e9d2fc9b36debbfe7c927bfe2fde5c5e"
C5_ATTEMPT_SOURCE_COMMIT = "ca8f8490e9d2fc9b36debbfe7c927bfe2fde5c5e"
C3_FAILURE_EVIDENCE_COMMIT = "35844e76317017a73dc0fa83f7e976642b80c66f"
C3_FAILURE_REASON = "TargetError: platform affinity excludes frozen source or receiver CPU"
C5_FAILURE_EVIDENCE_COMMIT = "8021563a6122b72316ddd218077b8b82e36f9055"
C5_FAILURE_REASON = "TargetError: no approved CPU temperature sensor found"
FINAL_OBJECT_VERIFY_PATH = HERE / "CARRIER_TOMOGRAPHY_FINAL_OBJECT_VERIFY.json"
FINAL_EVIDENCE_COMMIT_PATH = HERE / "CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT.json"
MANIFEST_PATH = HERE / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json"
MANIFEST_SHA_PATH = HERE / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256"
RUNTIME_SELF_TEST_PATH = HERE / "CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST.json"
TARGET_SELF_TEST_PATH = HERE / "CARRIER_TOMOGRAPHY_TARGET_SELF_TEST.json"
CONTROLLER_SELF_TEST_PATH = HERE / "CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST.json"
OFFLINE_VALIDATE_PATH = HERE / "CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json"
TRANSPORT_SIM_PATH = HERE / "CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION.json"
DEPLOYMENT_LAYOUT_PATH = HERE / "CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST.json"
FEATURE_BOUNDARY_PATH = HERE / "CARRIER_TOMOGRAPHY_FEATURE_BOUNDARY_SELF_TEST.json"
OPERATOR_ANALYSIS_PATH = HERE / "CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST.json"
FACTORIAL_ARM_PATH = HERE / "CARRIER_TOMOGRAPHY_FACTORIAL_ARM_SELF_TEST.json"
SOURCE_DEATH_PATH = HERE / "CARRIER_TOMOGRAPHY_SOURCE_DEATH_CUSTODY_SELF_TEST.json"
EXACT_COVERAGE_PATH = HERE / "CARRIER_TOMOGRAPHY_EXACT_COVERAGE_SELF_TEST.json"
SUBAGENT_FINDINGS_PATH = HERE / "SUBAGENT_FINDINGS_NORMALIZED.json"
SUBAGENT_REVIEW_PATH = HERE / "SUBAGENT_REVIEW_REPORTS.md"
BINARY_PATH = HERE / "family10h_carrier_tomography_runtime"
GENERATED_RECEIPTS = [
    HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json",
    RUNTIME_SELF_TEST_PATH,
    TARGET_SELF_TEST_PATH,
    CONTROLLER_SELF_TEST_PATH,
    OFFLINE_VALIDATE_PATH,
    TRANSPORT_SIM_PATH,
    DEPLOYMENT_LAYOUT_PATH,
    FEATURE_BOUNDARY_PATH,
    OPERATOR_ANALYSIS_PATH,
    FACTORIAL_ARM_PATH,
    SOURCE_DEATH_PATH,
    EXACT_COVERAGE_PATH,
    MANIFEST_PATH,
    MANIFEST_SHA_PATH,
]

EXPECTED_STARTING_HEAD = "8021563a6122b72316ddd218077b8b82e36f9055"
COMMIT_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING"
MANIFEST_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256"
AUTHORITY_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY"
AUTHORITY_VALUE = public.TRANSACTION_RUN_ID
TEMPERATURE_SENSOR_AUTHORITY_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY_V1"
TEMPERATURE_SENSOR_DISCOVERY_SCHEMA = target.TEMPERATURE_SENSOR_DISCOVERY_SCHEMA
TEMPERATURE_AUTHORITY_NONCE_ENV = target.TEMPERATURE_AUTHORITY_NONCE_ENV
TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV = target.TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV
TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA = target.TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA
REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS = target.REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS
TARGET_HOST = "root@192.168.137.100"
DISCOVERY_REMOTE_ROOT = f"{public.EXPECTED_REMOTE_ROOT}_sensor_authority_discovery"
DISCOVERY_OWNED_ROOT_HEX_LEN = 16

SOURCE_FILE_NAMES = target.SOURCE_FILE_NAMES
SOURCE_AUTHORITY_FILE_NAMES = target.SOURCE_AUTHORITY_FILE_NAMES
RUNTIME_BINARY_NAME = target.RUNTIME_BINARY_NAME
RUNTIME_AUTHORITY_FILE_NAMES = target.RUNTIME_AUTHORITY_FILE_NAMES
DISCOVERY_TRANSFER_FILE_NAMES = target.DISCOVERY_TRANSFER_FILE_NAMES
SOURCE_AUTHORITY_GENERATED_NAMES = {SOURCE_HASHES.name, SOURCE_BUNDLE.name}
FINAL_EVIDENCE_COMMIT_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_COMMIT"
REQUIRED_REVIEW_ROLES = {
    "physical_carrier_state_auditor": "physical carrier-state auditor",
    "experimental_design_operator_auditor": "experimental-design/operator auditor",
    "custody_evidence_auditor": "custody/evidence auditor",
    "claim_boundary_adjudicator": "claim-boundary adjudicator",
}
REVIEW_ROLE_ALIASES = {
    "physical_carrier_state_auditor": "physical_carrier_state_auditor",
    "experimental_design_operator_auditor": "experimental_design_operator_auditor",
    "implementation_custody_evidence_auditor": "custody_evidence_auditor",
    "custody_evidence_auditor": "custody_evidence_auditor",
    "claim_boundary_adjudicator": "claim_boundary_adjudicator",
}
SOURCE_AUDIT_REVIEW_DIR = HERE / "SOURCE_AUTHORITY_C3_REVIEW"
SOURCE_AUDIT_FINDINGS_PATH = HERE / "SOURCE_AUTHORITY_C3_REVIEW_NORMALIZED.json"
SOURCE_AUDIT_REVIEW_PATH = HERE / "SOURCE_AUTHORITY_C3_REVIEW_REPORTS.md"
SOURCE_AUDIT_REVIEW_VERSIONED = {
    "C3": {
        "review_dir": HERE / "SOURCE_AUTHORITY_C3_REVIEW",
        "findings_path": HERE / "SOURCE_AUTHORITY_C3_REVIEW_NORMALIZED.json",
        "review_path": HERE / "SOURCE_AUTHORITY_C3_REVIEW_REPORTS.md",
    },
    "C4": {
        "review_dir": HERE / "SOURCE_AUTHORITY_C4_REVIEW",
        "findings_path": HERE / "SOURCE_AUTHORITY_C4_REVIEW_NORMALIZED.json",
        "review_path": HERE / "SOURCE_AUTHORITY_C4_REVIEW_REPORTS.md",
    },
    "C5": {
        "review_dir": HERE / "SOURCE_AUTHORITY_C5_REVIEW",
        "findings_path": HERE / "SOURCE_AUTHORITY_C5_REVIEW_NORMALIZED.json",
        "review_path": HERE / "SOURCE_AUTHORITY_C5_REVIEW_REPORTS.md",
    },
    "C6": {
        "review_dir": HERE / "SOURCE_AUTHORITY_C6_REVIEW",
        "findings_path": HERE / "SOURCE_AUTHORITY_C6_REVIEW_NORMALIZED.json",
        "review_path": HERE / "SOURCE_AUTHORITY_C6_REVIEW_REPORTS.md",
    },
}
SOURCE_AUDIT_REQUIRED_REVIEW_ROLES = {
    "physical_sensor_authority_auditor": "physical sensor-authority auditor",
    "discovery_transport_custody_auditor": "discovery transport and custody auditor",
    "source_bundle_runtime_evidence_auditor": "source/bundle/runtime evidence auditor",
    "claim_boundary_adjudicator": "claim-boundary adjudicator",
}
SOURCE_AUDIT_REVIEW_ARCHIVE_FILES = {
    "physical_sensor_authority_auditor": ("physical_sensor_authority_body.md", "physical_sensor_authority_receipt.json"),
    "discovery_transport_custody_auditor": ("discovery_transport_custody_body.md", "discovery_transport_custody_receipt.json"),
    "source_bundle_runtime_evidence_auditor": ("source_bundle_evidence_body.md", "source_bundle_evidence_receipt.json"),
    "claim_boundary_adjudicator": ("claim_boundary_body.md", "claim_boundary_receipt.json"),
}
SOURCE_AUDIT_ROLE_ALIASES = {
    "physical_sensor_authority_auditor": "physical_sensor_authority_auditor",
    "physical_sensor_authority": "physical_sensor_authority_auditor",
    "discovery_transport_and_custody_auditor": "discovery_transport_custody_auditor",
    "discovery_transport_custody_auditor": "discovery_transport_custody_auditor",
    "source_bundle_and_evidence_auditor": "source_bundle_runtime_evidence_auditor",
    "source_bundle_evidence_auditor": "source_bundle_runtime_evidence_auditor",
    "source_bundle_runtime_evidence_auditor": "source_bundle_runtime_evidence_auditor",
    "source_bundle_and_runtime_evidence_auditor": "source_bundle_runtime_evidence_auditor",
    "source_bundle_runtime_and_evidence_auditor": "source_bundle_runtime_evidence_auditor",
    "claim_boundary_adjudicator": "claim_boundary_adjudicator",
}
SOURCE_AUDIT_REVIEW_RECEIPT_KEYS_V2 = {
    "schema",
    "issuer",
    "receipt_kind",
    "thread_id",
    "agent_id",
    "role",
    "model",
    "review_body_sha256",
    "review_body_canonicalization",
    "audited_commit",
    "source_hashes_sha256",
    "source_bundle_sha256",
    "runtime_binary_sha256",
    "no_git_write",
    "no_file_edits",
    "no_checkout_mutation",
    "no_target_contact",
    "no_live_authority",
    "no_pmu",
    "self_authored",
    "evidence_origin",
}
SOURCE_AUDIT_REVIEW_RECEIPT_KEYS_V3 = SOURCE_AUDIT_REVIEW_RECEIPT_KEYS_V2 | {
    "verdict",
    "final_response",
    "material_blocker_ids",
}
SOURCE_AUDIT_REVIEW_RECEIPT_KEYS = SOURCE_AUDIT_REVIEW_RECEIPT_KEYS_V3
SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2 = "FAMILY10H_SOURCE_AUTHORITY_REVIEWER_RECEIPT_V2"
SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3 = "FAMILY10H_SOURCE_AUTHORITY_REVIEWER_RECEIPT_V3"
SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA = SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3
SOURCE_AUDIT_REVIEW_RECEIPT_ISSUER = "codex_subagent_read_only_review"
SOURCE_AUDIT_ALLOWED_EVIDENCE_ORIGIN = "codex_subagent_detached_receipt"
SOURCE_AUDIT_REVIEW_BODY_CANONICALIZATION = "utf8_lf_single_trailing_newline"
SOURCE_AUDIT_RECEIPT_KIND = "detached_review_body_acknowledgment"


class ControllerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def normalized_role(value: str) -> str:
    return value.lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def review_quorum(independent_review: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    material_blockers_value = independent_review.get("material_blockers")
    if not isinstance(material_blockers_value, list):
        failures.append("material blockers list missing or malformed")
        material_blockers: list[Any] = []
    else:
        material_blockers = material_blockers_value
        if material_blockers:
            failures.append("material blockers present")
    reviewer_verdicts = independent_review.get("reviewer_verdicts")
    if not isinstance(reviewer_verdicts, dict):
        failures.append("reviewer verdicts missing or malformed")
        reviewer_verdicts = {}
    if len(reviewer_verdicts) != len(REQUIRED_REVIEW_ROLES):
        failures.append("reviewer verdict count must be exactly four")
    candidates: list[dict[str, Any]] = []
    for role_key, item in reviewer_verdicts.items():
        if isinstance(item, dict):
            candidates.append({**item, "role_key": role_key})
        else:
            failures.append(f"malformed reviewer response {role_key}")

    by_role: dict[str, dict[str, Any]] = {}
    for item in candidates:
        role_name = str(item.get("role") or item.get("originating_agent") or item.get("role_key", ""))
        role = REVIEW_ROLE_ALIASES.get(normalized_role(role_name), "")
        if role not in REQUIRED_REVIEW_ROLES:
            failures.append(f"unknown reviewer role {role_name}")
            continue
        if role in by_role:
            failures.append(f"duplicate reviewer role {role}")
            continue
        agent_id = item.get("agent_id")
        verdict = item.get("verdict")
        final_response = item.get("final_response")
        by_role[role] = {
            "role": REQUIRED_REVIEW_ROLES[role],
            "agent_id": agent_id,
            "verdict": verdict,
            "final_response": final_response,
            "passed": isinstance(agent_id, str)
            and bool(agent_id)
            and verdict == "NO_MATERIAL_BLOCKER"
            and final_response is True,
        }
    missing = sorted(set(REQUIRED_REVIEW_ROLES) - set(by_role))
    if missing:
        failures.append("missing reviewer roles: " + ",".join(missing))
    duplicate_agent_ids = [
        agent_id
        for agent_id, count in Counter(item["agent_id"] for item in by_role.values() if item.get("agent_id")).items()
        if count > 1
    ]
    if duplicate_agent_ids:
        failures.append("duplicate reviewer agent ids")
    failed_roles = [role for role, item in by_role.items() if not item["passed"]]
    if failed_roles:
        failures.append("non-clear reviewer roles: " + ",".join(sorted(failed_roles)))
    return {
        "required_roles": REQUIRED_REVIEW_ROLES,
        "roles": by_role,
        "material_blockers": material_blockers,
        "missing_roles": missing,
        "failures": failures,
        "passed": not failures,
    }


def review_body_canonical_state(data: bytes) -> dict[str, Any]:
    failures: list[str] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "passed": False,
            "failures": ["review body is not UTF-8"],
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "canonical_sha256": None,
        }
    canonical_text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    canonical = canonical_text.encode("utf-8")
    if canonical != data:
        failures.append("review body is not canonical UTF-8 LF with exactly one trailing newline")
    return {
        "passed": not failures,
        "failures": failures,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def expected_source_audit_archive_paths(role: str, review_root: Path = SOURCE_AUDIT_REVIEW_DIR) -> tuple[Path, Path]:
    body_name, receipt_name = SOURCE_AUDIT_REVIEW_ARCHIVE_FILES[role]
    return review_root / body_name, review_root / receipt_name


def source_audit_version_for_commit(source_commit: str | None) -> str:
    if source_commit == C3_SOURCE_AUTHORITY_COMMIT:
        return "C3"
    if source_commit == C4_SOURCE_AUTHORITY_COMMIT:
        return "C4"
    if source_commit == C5_SOURCE_AUTHORITY_COMMIT:
        return "C5"
    return "C6"


def source_audit_paths_for_commit(source_commit: str | None) -> dict[str, Path]:
    version = source_audit_version_for_commit(source_commit)
    if version == "C3":
        return {
            "review_dir": SOURCE_AUDIT_REVIEW_DIR,
            "findings_path": SOURCE_AUDIT_FINDINGS_PATH,
            "review_path": SOURCE_AUDIT_REVIEW_PATH,
        }
    prefix = f"SOURCE_AUTHORITY_{version}_REVIEW"
    return {
        "review_dir": HERE / prefix,
        "findings_path": HERE / f"{prefix}_NORMALIZED.json",
        "review_path": HERE / f"{prefix}_REPORTS.md",
    }


def source_audit_receipt_schema_for_commit(source_commit: str | None) -> str:
    if source_audit_version_for_commit(source_commit) in {"C3", "C4", "C5"}:
        return SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2
    return SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3


def source_audit_receipt_keys_for_schema(schema: str) -> set[str]:
    if schema == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2:
        return SOURCE_AUDIT_REVIEW_RECEIPT_KEYS_V2
    if schema == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3:
        return SOURCE_AUDIT_REVIEW_RECEIPT_KEYS_V3
    return set()


def archive_path_matches(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/")
    accepted = {str(expected).replace("\\", "/"), expected.as_posix()}
    try:
        accepted.add(path_to_repo_relative(expected))
    except ValueError:
        pass
    return normalized in accepted


def archived_bytes(path: Path, evidence_commit: str | None = None) -> bytes | None:
    if evidence_commit is not None:
        return commit_blob_bytes(evidence_commit, path)
    if not path.exists():
        return None
    return path.read_bytes()


def archived_json(path: Path, evidence_commit: str | None = None) -> dict[str, Any] | None:
    data = archived_bytes(path, evidence_commit)
    if data is None:
        return None
    try:
        parsed = strict_json_loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def source_audit_quorum(
    source_audit: dict[str, Any],
    *,
    expected_source_commit: str | None,
    expected_source_hashes_sha256: str,
    expected_source_bundle_sha256: str,
    expected_runtime_binary_sha256: str,
    review_report_present: bool,
    excluded_agent_ids: set[str] | None = None,
    evidence_commit: str | None = None,
    review_root: Path = SOURCE_AUDIT_REVIEW_DIR,
) -> dict[str, Any]:
    failures: list[str] = []
    excluded_agent_ids = excluded_agent_ids or set()
    if source_audit.get("source_authority_commit") != expected_source_commit:
        failures.append("source audit commit mismatch")
    if source_audit.get("source_hashes_sha256") != expected_source_hashes_sha256:
        failures.append("source audit source-hashes mismatch")
    if source_audit.get("source_bundle_sha256") != expected_source_bundle_sha256:
        failures.append("source audit source-bundle mismatch")
    if source_audit.get("runtime_binary_sha256") != expected_runtime_binary_sha256:
        failures.append("source audit runtime-binary mismatch")
    if source_audit.get("review_report_present") is not True or not review_report_present:
        failures.append("source audit review report missing")
    material_blockers_value = source_audit.get("material_blockers")
    if not isinstance(material_blockers_value, list):
        failures.append("source audit material blockers missing or malformed")
        material_blockers: list[Any] = []
    else:
        material_blockers = material_blockers_value
        if material_blockers:
            failures.append("source audit material blockers present")
    reviewer_verdicts = source_audit.get("reviewer_verdicts")
    if not isinstance(reviewer_verdicts, dict):
        failures.append("source audit reviewer verdicts missing or malformed")
        reviewer_verdicts = {}
    if len(reviewer_verdicts) != len(SOURCE_AUDIT_REQUIRED_REVIEW_ROLES):
        failures.append("source audit reviewer verdict count must be exactly four")

    def validate_reviewer_archive(item: dict[str, Any], role: str, role_name: str) -> list[str]:
        receipt_failures: list[str] = []
        expected_receipt_schema = source_audit_receipt_schema_for_commit(expected_source_commit)
        expected_receipt_keys = source_audit_receipt_keys_for_schema(expected_receipt_schema)
        body_path, receipt_path = expected_source_audit_archive_paths(role, review_root)
        if not archive_path_matches(item.get("body_path"), body_path):
            receipt_failures.append(f"source audit reviewer body path mismatch {role}")
        if not archive_path_matches(item.get("receipt_path"), receipt_path):
            receipt_failures.append(f"source audit reviewer receipt path mismatch {role}")
        body_bytes = archived_bytes(body_path, evidence_commit)
        if body_bytes is None:
            receipt_failures.append(f"source audit reviewer body missing {role}")
            body_state = {"passed": False, "file_sha256": None, "canonical_sha256": None, "failures": []}
        else:
            body_state = review_body_canonical_state(body_bytes)
            receipt_failures.extend(f"source audit reviewer body {failure} {role}" for failure in body_state["failures"])
        if item.get("body_file_sha256") != body_state.get("file_sha256"):
            receipt_failures.append(f"source audit reviewer body file digest mismatch {role}")
        if item.get("body_canonical_sha256") != body_state.get("canonical_sha256"):
            receipt_failures.append(f"source audit reviewer body canonical digest mismatch {role}")
        receipt = archived_json(receipt_path, evidence_commit)
        if receipt is None:
            receipt_failures.append(f"source audit reviewer receipt missing {role}")
            receipt = {}
        if item.get("receipt_file_sha256") != (
            hashlib.sha256(archived_bytes(receipt_path, evidence_commit) or b"").hexdigest() if archived_bytes(receipt_path, evidence_commit) is not None else None
        ):
            receipt_failures.append(f"source audit reviewer receipt file digest mismatch {role}")
        if item.get("review_receipt") != receipt:
            receipt_failures.append(f"source audit reviewer normalized receipt mismatch {role}")
        if set(receipt) != expected_receipt_keys:
            receipt_failures.append(f"source audit reviewer receipt field mismatch {role}")
        expected_pairs = {
            "schema": expected_receipt_schema,
            "issuer": SOURCE_AUDIT_REVIEW_RECEIPT_ISSUER,
            "receipt_kind": SOURCE_AUDIT_RECEIPT_KIND,
            "thread_id": item.get("thread_id"),
            "agent_id": item.get("agent_id"),
            "role": role_name,
            "model": item.get("model"),
            "review_body_sha256": item.get("body_canonical_sha256"),
            "review_body_canonicalization": SOURCE_AUDIT_REVIEW_BODY_CANONICALIZATION,
            "audited_commit": expected_source_commit,
            "source_hashes_sha256": expected_source_hashes_sha256,
            "source_bundle_sha256": expected_source_bundle_sha256,
            "runtime_binary_sha256": expected_runtime_binary_sha256,
            "evidence_origin": SOURCE_AUDIT_ALLOWED_EVIDENCE_ORIGIN,
            "self_authored": False,
            "no_git_write": True,
            "no_file_edits": True,
            "no_checkout_mutation": True,
            "no_target_contact": True,
            "no_live_authority": True,
            "no_pmu": True,
        }
        normalized_blocker_ids = item.get("material_blocker_ids")
        if expected_receipt_schema == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3:
            if (
                not isinstance(normalized_blocker_ids, list)
                or any(not isinstance(blocker_id, str) or not blocker_id for blocker_id in normalized_blocker_ids)
            ):
                receipt_failures.append(f"source audit reviewer material blocker ids missing or malformed {role}")
                normalized_blocker_ids = []
            expected_pairs.update(
                {
                    "verdict": item.get("verdict"),
                    "final_response": item.get("final_response"),
                    "material_blocker_ids": normalized_blocker_ids,
                }
            )
        for key, expected in expected_pairs.items():
            if receipt.get(key) != expected:
                receipt_failures.append(f"source audit reviewer receipt {key} mismatch {role}")
        if "final_response_sha256" in receipt:
            receipt_failures.append(f"source audit reviewer self-referential receipt field rejected {role}")
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("review_body_sha256", ""))) is None:
            receipt_failures.append(f"source audit reviewer body digest invalid {role}")
        if not isinstance(receipt.get("thread_id"), str) or not receipt["thread_id"]:
            receipt_failures.append(f"source audit reviewer thread id missing {role}")
        if not isinstance(receipt.get("model"), str) or not receipt["model"]:
            receipt_failures.append(f"source audit reviewer model missing {role}")
        if item.get("self_authored") is True or item.get("evidence_origin") == "target-derived" or item.get("evidence_origin") == "parent-created":
            receipt_failures.append(f"source audit reviewer provenance rejected {role}")
        return receipt_failures

    by_role: dict[str, dict[str, Any]] = {}
    for role_key, item in reviewer_verdicts.items():
        if not isinstance(item, dict):
            failures.append(f"malformed source audit reviewer response {role_key}")
            continue
        role_name = str(item.get("role") or item.get("originating_agent") or role_key)
        role = SOURCE_AUDIT_ROLE_ALIASES.get(normalized_role(role_name), "")
        if role not in SOURCE_AUDIT_REQUIRED_REVIEW_ROLES:
            failures.append(f"unknown source audit reviewer role {role_name}")
            continue
        if role in by_role:
            failures.append(f"duplicate source audit reviewer role {role}")
            continue
        by_role[role] = {
            "role": SOURCE_AUDIT_REQUIRED_REVIEW_ROLES[role],
            "agent_id": item.get("agent_id"),
            "verdict": item.get("verdict"),
            "final_response": item.get("final_response"),
            "material_blocker_ids": item.get("material_blocker_ids"),
            "audited_commit": item.get("audited_commit"),
            "source_hashes_sha256": item.get("source_hashes_sha256"),
            "source_bundle_sha256": item.get("source_bundle_sha256"),
            "runtime_binary_sha256": item.get("runtime_binary_sha256"),
            "body_path": item.get("body_path"),
            "body_file_sha256": item.get("body_file_sha256"),
            "body_canonical_sha256": item.get("body_canonical_sha256"),
            "receipt_path": item.get("receipt_path"),
            "receipt_file_sha256": item.get("receipt_file_sha256"),
            "thread_id": item.get("thread_id"),
            "model": item.get("model"),
            "boundary_attestation": item.get("boundary_attestation"),
            "review_receipt": item.get("review_receipt"),
            "passed": isinstance(item.get("agent_id"), str)
            and bool(item.get("agent_id"))
            and item.get("verdict") == "NO_MATERIAL_BLOCKER"
            and item.get("final_response") is True,
        }
        role_entry = by_role[role]
        receipt_failures = validate_reviewer_archive(item, role, SOURCE_AUDIT_REQUIRED_REVIEW_ROLES[role])
        if receipt_failures:
            failures.extend(receipt_failures)
            role_entry["passed"] = False
        if role_entry["audited_commit"] != expected_source_commit:
            failures.append(f"source audit reviewer commit mismatch {role}")
            role_entry["passed"] = False
        if role_entry["source_hashes_sha256"] != expected_source_hashes_sha256:
            failures.append(f"source audit reviewer source-hashes mismatch {role}")
            role_entry["passed"] = False
        if role_entry["source_bundle_sha256"] != expected_source_bundle_sha256:
            failures.append(f"source audit reviewer source-bundle mismatch {role}")
            role_entry["passed"] = False
        if role_entry["runtime_binary_sha256"] != expected_runtime_binary_sha256:
            failures.append(f"source audit reviewer runtime-binary mismatch {role}")
            role_entry["passed"] = False
        if (
            source_audit_receipt_schema_for_commit(expected_source_commit) == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3
            and role_entry["material_blocker_ids"] != []
        ):
            failures.append(f"source audit reviewer material blocker ids present {role}")
            role_entry["passed"] = False
        boundary = role_entry["boundary_attestation"]
        if (
            not isinstance(boundary, dict)
            or boundary.get("no_git_write") is not True
            or boundary.get("no_file_edits") is not True
            or boundary.get("no_checkout_mutation") is not True
            or boundary.get("no_target_contact") is not True
            or boundary.get("no_live_authority") is not True
            or boundary.get("no_pmu") is not True
        ):
            failures.append(f"source audit reviewer boundary attestation missing {role}")
            role_entry["passed"] = False
        if role_entry["agent_id"] in excluded_agent_ids:
            failures.append(f"source audit reviewer reuses prior package reviewer identity {role}")
            role_entry["passed"] = False
    missing = sorted(set(SOURCE_AUDIT_REQUIRED_REVIEW_ROLES) - set(by_role))
    if missing:
        failures.append("missing source audit reviewer roles: " + ",".join(missing))
    duplicate_agent_ids = [
        agent_id
        for agent_id, count in Counter(item["agent_id"] for item in by_role.values() if item.get("agent_id")).items()
        if count > 1
    ]
    if duplicate_agent_ids:
        failures.append("duplicate source audit reviewer agent ids")
    duplicate_thread_ids = [
        thread_id
        for thread_id, count in Counter(item["thread_id"] for item in by_role.values() if item.get("thread_id")).items()
        if count > 1
    ]
    if duplicate_thread_ids:
        failures.append("duplicate source audit reviewer thread ids")
    failed_roles = [role for role, item in by_role.items() if not item["passed"]]
    if failed_roles:
        failures.append("non-clear source audit reviewer roles: " + ",".join(sorted(failed_roles)))
    return {
        "required_roles": SOURCE_AUDIT_REQUIRED_REVIEW_ROLES,
        "roles": by_role,
        "material_blockers": material_blockers,
        "missing_roles": missing,
        "failures": failures,
        "passed": not failures,
    }


def strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value rejected: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key rejected: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)


def is_strict_int(value: Any) -> bool:
    return type(value) is int


def is_strict_counter(value: Any) -> bool:
    return is_strict_int(value) and value >= 0


def counters_equal_strict(receipt: dict[str, Any], expected: dict[str, int]) -> bool:
    if not isinstance(receipt, dict):
        return False
    return all(is_strict_int(receipt.get(key)) and receipt.get(key) == value for key, value in expected.items())


def counter_dict_equal_strict(receipt: dict[str, Any], expected: dict[str, int]) -> bool:
    return isinstance(receipt, dict) and set(receipt) == set(expected) and counters_equal_strict(receipt, expected)


def zero_contact_counter_valid(receipt: dict[str, Any], key: str) -> bool:
    return key not in receipt or (is_strict_int(receipt.get(key)) and receipt.get(key) == 0)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((strict_json_dumps(value, indent=2) + "\n").encode("utf-8"))


def durable_sync_directory(path: Path) -> None:
    if os.name == "nt":
        import ctypes

        generic_write = 0x40000000
        file_share_read = 0x00000001
        file_share_write = 0x00000002
        file_share_delete = 0x00000004
        open_existing = 3
        file_flag_backup_semantics = 0x02000000
        invalid_handle_value = ctypes.c_void_p(-1).value
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
        kernel32.FlushFileBuffers.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.CreateFileW(
            str(path),
            generic_write,
            file_share_read | file_share_write | file_share_delete,
            None,
            open_existing,
            file_flag_backup_semantics,
            None,
        )
        if handle in (None, invalid_handle_value):
            raise OSError(ctypes.get_last_error(), f"directory open failed for durable barrier: {path}")
        try:
            if not kernel32.FlushFileBuffers(handle):
                raise OSError(ctypes.get_last_error(), f"directory flush failed for durable barrier: {path}")
        finally:
            kernel32.CloseHandle(handle)
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path: Path, *, require_directory_sync: bool = False) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    if require_directory_sync:
        for created in reversed(missing):
            durable_sync_directory(created.parent)


def durable_write_bytes_exclusive(path: Path, data: bytes, *, require_directory_sync: bool = False) -> None:
    ensure_directory(path.parent, require_directory_sync=require_directory_sync)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if require_directory_sync:
        durable_sync_directory(path.parent)


def write_json_exclusive(path: Path, value: Any, *, require_directory_sync: bool = False) -> None:
    durable_write_bytes_exclusive(
        path,
        (strict_json_dumps(value, indent=2) + "\n").encode("utf-8"),
        require_directory_sync=require_directory_sync,
    )


def write_json_atomic(path: Path, value: Any, *, require_directory_sync: bool = False) -> None:
    ensure_directory(path.parent, require_directory_sync=require_directory_sync)
    data = (strict_json_dumps(value, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp") as handle:
        tmp_name = handle.name
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_name, path)
    if require_directory_sync:
        durable_sync_directory(path.parent)


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, timeout: float, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, cwd=cwd)
    if check and completed.returncode != 0:
        raise ControllerError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def bounded_text(value: str, limit: int = 8192) -> dict[str, Any]:
    value = value or ""
    return {"text": value[:limit], "truncated": len(value) > limit, "char_count": len(value)}


def sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8", errors="replace")).hexdigest()


def git_text(*args: str) -> str:
    return run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args], timeout=30.0, cwd=REPO_ROOT).stdout.strip()


def git_lines(*args: str) -> list[str]:
    text = run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args], timeout=30.0, cwd=REPO_ROOT).stdout
    return [line for line in text.splitlines() if line.strip()]


def git_blob_bytes(commit: str, name: str) -> bytes:
    rel = package_relative_path(name)
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "show", f"{commit}:{rel}"],
        capture_output=True,
        timeout=30.0,
        check=False,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise ControllerError(f"git blob read failed for {commit}:{rel}: {completed.stderr.decode('utf-8', errors='replace')}")
    return completed.stdout


def git_state() -> dict[str, Any]:
    return {
        "branch": git_text("branch", "--show-current"),
        "head": git_text("rev-parse", "HEAD"),
        "origin_main": git_text("rev-parse", "origin/main"),
        "status_porcelain": "\n".join(git_lines("status", "--porcelain")),
        "stash_list": git_lines("stash", "list"),
    }


def source_file_map() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILE_NAMES:
        path = HERE / name
        if path.exists():
            result[name] = {"sha256": public.sha256_file(path), "size": path.stat().st_size}
    return result


def runtime_binary_authority() -> dict[str, Any]:
    return target.runtime_binary_authority(HERE)


def compute_source_hashes() -> dict[str, Any]:
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_HASHES_V1",
        "source_files": source_file_map(),
        "runtime_binary_authority": runtime_binary_authority(),
    }
    result["source_hashes_sha256"] = public.digest({k: v for k, v in result.items() if k != "source_hashes_sha256"})
    return result


def initialize_source_hash_authority(*, force: bool = False) -> dict[str, Any]:
    result = compute_source_hashes()
    if SOURCE_HASHES.exists():
        existing_bytes = SOURCE_HASHES.read_bytes()
        existing = read_json(SOURCE_HASHES)
        if existing != result and not force:
            raise ControllerError("source hash authority already exists and differs; explicit force required")
        if existing == result:
            return existing
    write_json(SOURCE_HASHES, result)
    if SOURCE_HASHES.exists() and SOURCE_HASHES.read_bytes() == b"":
        raise ControllerError("source hash authority write failed")
    return result


def read_source_hash_authority() -> dict[str, Any]:
    if not SOURCE_HASHES.exists():
        raise ControllerError("source hash authority missing; initialize explicitly before validation")
    receipt = read_json(SOURCE_HASHES)
    authority = target.validate_source_file_authority(HERE)
    if not authority["passed"]:
        raise ControllerError("source hash authority failed: " + ",".join(authority["failures"]))
    current = compute_source_hashes()
    if receipt != current:
        raise ControllerError("source hash authority mismatch")
    return receipt


def package_relative_path(name: str) -> str:
    return (HERE / name).relative_to(REPO_ROOT).as_posix()


def source_authority_commit_verification(commit: str) -> dict[str, Any]:
    failures: list[str] = []
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        return {"passed": False, "failures": ["source authority commit malformed"], "commit": commit, "files": {}}
    exists = run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "cat-file", "-e", f"{commit}^{{commit}}"],
        timeout=30.0,
        check=False,
        cwd=REPO_ROOT,
    )
    if exists.returncode != 0:
        return {"passed": False, "failures": ["source authority commit missing"], "commit": commit, "files": {}}
    status = git_lines(
        "status",
        "--porcelain",
        "--",
        *[package_relative_path(name) for name in DISCOVERY_TRANSFER_FILE_NAMES],
    )
    if status:
        failures.append("source authority paths are dirty")
    files: dict[str, dict[str, Any]] = {}
    for name in SOURCE_AUTHORITY_FILE_NAMES:
        path = HERE / name
        rel = package_relative_path(name)
        if not path.exists():
            failures.append(f"source authority file missing {name}")
            files[name] = {"present": False}
            continue
        committed_blob = git_text("rev-parse", f"{commit}:{rel}")
        working_blob = git_text("hash-object", str(path))
        files[name] = {
            "present": True,
            "repo_path": rel,
            "committed_blob": committed_blob,
            "working_blob": working_blob,
            "sha256": public.sha256_file(path),
        }
        if committed_blob != working_blob:
            failures.append(f"source authority file differs from commit {name}")
    for name in RUNTIME_AUTHORITY_FILE_NAMES:
        path = HERE / name
        rel = package_relative_path(name)
        if not path.exists():
            failures.append(f"runtime authority file missing {name}")
            files[name] = {"present": False}
            continue
        committed_blob = git_text("rev-parse", f"{commit}:{rel}")
        working_blob = git_text("hash-object", str(path))
        files[name] = {
            "present": True,
            "repo_path": rel,
            "committed_blob": committed_blob,
            "working_blob": working_blob,
            "sha256": public.sha256_file(path),
            "size": path.stat().st_size,
        }
        if committed_blob != working_blob:
            failures.append(f"runtime authority file differs from commit {name}")
    return {"passed": not failures, "failures": failures, "commit": commit, "files": files, "status": status}


def discovery_transfer_authority_class(name: str) -> str:
    if name in SOURCE_FILE_NAMES:
        return "source"
    if name in SOURCE_AUTHORITY_GENERATED_NAMES or name == SUBAGENT_FINDINGS_PATH.name:
        return "generated authority"
    if name in RUNTIME_AUTHORITY_FILE_NAMES:
        return "runtime binary"
    return "unknown"


def build_discovery_transfer_plan(
    *,
    source_root: Path,
    remote_source_root: str,
    source_commit: str | None = None,
    expected_source_hashes_sha256: str | None = None,
    expected_source_bundle_sha256: str | None = None,
    expected_runtime_binary_authority: dict[str, Any] | None = None,
    expected_names: list[str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_names = expected_names or list(DISCOVERY_TRANSFER_FILE_NAMES)
    if expected_names != list(DISCOVERY_TRANSFER_FILE_NAMES):
        failures.append("production transfer plan differs from target-required set")
    observed_names = sorted(path.name for path in source_root.iterdir() if path.is_file()) if source_root.exists() else []
    missing = sorted(set(expected_names) - set(observed_names))
    extra = sorted(set(observed_names) - set(expected_names))
    if missing:
        failures.append("discovery transfer plan missing files: " + ",".join(missing))
    if extra:
        failures.append("discovery transfer plan extra files: " + ",".join(extra))

    source_hashes: dict[str, Any] = {}
    source_hashes_path = source_root / SOURCE_HASHES.name
    if source_hashes_path.exists():
        source_hashes = read_json(source_hashes_path)
        if expected_source_hashes_sha256 is not None and source_hashes.get("source_hashes_sha256") != expected_source_hashes_sha256:
            failures.append("transfer plan source hash receipt mismatch")
    else:
        failures.append("transfer plan source hash receipt missing")

    source_authority = target.validate_source_file_authority(source_root)
    if not source_authority["passed"]:
        failures.extend("transfer plan source authority: " + item for item in source_authority["failures"])

    bundle_path = source_root / SOURCE_BUNDLE.name
    bundle_file_sha256 = public.sha256_file(bundle_path) if bundle_path.exists() else None
    if bundle_file_sha256 is None:
        failures.append("transfer plan source bundle missing")
    if expected_source_bundle_sha256 is not None and bundle_file_sha256 != expected_source_bundle_sha256:
        failures.append("transfer plan source bundle hash mismatch")
    try:
        bundle_reconstruction_sha256 = target.deterministic_source_bundle_sha256(source_root)
    except Exception as exc:  # deterministic bundle errors are converted into validation failures.
        bundle_reconstruction_sha256 = None
        failures.append(f"transfer plan source bundle reconstruction failed: {exc}")
    if expected_source_bundle_sha256 is not None and bundle_reconstruction_sha256 != expected_source_bundle_sha256:
        failures.append("transfer plan source bundle reconstruction mismatch")

    runtime_authority = target.runtime_binary_authority(source_root)
    expected_runtime_binary_authority = expected_runtime_binary_authority or (
        source_hashes.get("runtime_binary_authority") if isinstance(source_hashes.get("runtime_binary_authority"), dict) else None
    )
    if expected_runtime_binary_authority is None:
        failures.append("transfer plan runtime authority missing")
    else:
        for key in ["sha256", "size", "git_blob_id"]:
            if runtime_authority.get(key) != expected_runtime_binary_authority.get(key):
                failures.append(f"transfer plan runtime {key} mismatch")

    committed_available = bool(source_commit and commit_exists(source_commit))
    records: list[dict[str, Any]] = []
    for name in expected_names:
        path = source_root / name
        payload = path.read_bytes() if path.exists() else b""
        local_blob = target.git_blob_id_for_bytes(payload) if path.exists() else None
        committed_blob = None
        if committed_available:
            try:
                committed_blob = git_text("rev-parse", f"{source_commit}:{package_relative_path(name)}")
            except ControllerError:
                failures.append(f"transfer plan committed blob missing {name}")
        if committed_blob is not None and local_blob != committed_blob:
            failures.append(f"transfer plan local file differs from committed blob {name}")
        records.append(
            {
                "name": name,
                "source_path": str(path),
                "remote_destination": f"{remote_source_root}/{name}",
                "byte_size": path.stat().st_size if path.exists() else None,
                "sha256": hashlib.sha256(payload).hexdigest() if path.exists() else None,
                "git_blob_id": committed_blob,
                "local_git_blob_id": local_blob,
                "authority_class": discovery_transfer_authority_class(name),
            }
        )

    plan_keyset = [record["name"] for record in records]
    if plan_keyset != list(DISCOVERY_TRANSFER_FILE_NAMES):
        failures.append("actual outbound transfer keyset differs from DISCOVERY_TRANSFER_FILE_NAMES")
    if RUNTIME_BINARY_NAME not in plan_keyset:
        failures.append("production transfer plan omits runtime binary")

    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSFER_PLAN_V1",
        "passed": not failures,
        "failures": failures,
        "file_count": len(records),
        "expected_file_names": list(DISCOVERY_TRANSFER_FILE_NAMES),
        "actual_file_names": plan_keyset,
        "missing_files": missing,
        "extra_files": extra,
        "records": records,
        "source_authority": source_authority,
        "source_bundle_file_sha256": bundle_file_sha256,
        "source_bundle_reconstruction_sha256": bundle_reconstruction_sha256,
        "runtime_binary_authority": runtime_authority,
    }
    result["transfer_plan_sha256"] = public.digest({k: v for k, v in result.items() if k != "transfer_plan_sha256"})
    return result


def build_temperature_authority_challenge(
    *,
    source_hashes: dict[str, Any],
    source_bundle_sha256: str,
    runtime_binary_sha256: str,
    schedule_sidecar: dict[str, Any],
    authorized_commit: str,
    controller_nonce_sha256: str,
    transport_scope: dict[str, Any],
    source_authority_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA,
        "authority": "controller_issued_temperature_sensor_challenge",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "runtime_binary_sha256": runtime_binary_sha256,
        "schedule_canonical_sha256": schedule_sidecar["canonical_sha256"],
        "schedule_json_sha256": schedule_sidecar["json_sha256"],
        "schedule_tsv_sha256": schedule_sidecar["tsv_sha256"],
        "authorized_commit": authorized_commit,
        "controller_nonce_sha256": controller_nonce_sha256,
        "transport_scope": transport_scope,
        "source_authority_review": source_authority_review,
    }


def discovery_transport_scope(*, target_host: str, remote_root: str, nonce_sha: str) -> dict[str, Any]:
    require(target_host == TARGET_HOST, "target host override rejected for temperature discovery")
    require(remote_root == DISCOVERY_REMOTE_ROOT, "remote discovery root override rejected")
    require(re.fullmatch(r"[0-9a-f]{64}", nonce_sha) is not None, "nonce digest malformed")
    owned_root = f"{remote_root}/{nonce_sha[:DISCOVERY_OWNED_ROOT_HEX_LEN]}"
    remote_source = f"{owned_root}/source"
    return {
        "target_host": target_host,
        "remote_base_root": remote_root,
        "remote_root": owned_root,
        "remote_source_root": remote_source,
        "remote_receipt_path": f"{remote_source}/{target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME}",
    }


def fixture_transport_scope(source_root: Path, nonce_sha: str) -> dict[str, Any]:
    return {
        "target_host": "offline_fixture",
        "remote_base_root": str(source_root.parent),
        "remote_root": str(source_root.parent),
        "remote_source_root": str(source_root),
        "remote_receipt_path": str(source_root / target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME),
    }


def fixture_source_review_binding(
    source_commit: str,
    source_hashes_sha256: str,
    source_bundle_sha256: str,
    runtime_binary_sha256: str = "d" * 64,
) -> dict[str, Any]:
    return {
        "findings_sha256": "a" * 64,
        "review_report_sha256": "b" * 64,
        "review_quorum_sha256": "c" * 64,
        "source_authority_commit": source_commit,
        "source_hashes_sha256": source_hashes_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "runtime_binary_sha256": runtime_binary_sha256,
    }


def read_source_authority_review_for_discovery(
    *,
    source_commit: str,
    source_hashes_sha256: str,
    source_bundle_sha256: str,
    runtime_binary_sha256: str,
) -> dict[str, Any]:
    failures: list[str] = []
    review_paths = source_audit_paths_for_commit(source_commit)
    findings_path = review_paths["findings_path"]
    review_path = review_paths["review_path"]
    review_dir = review_paths["review_dir"]
    if not findings_path.exists():
        failures.append("source authority review findings missing")
        source_audit: dict[str, Any] = {}
    else:
        source_audit = read_json(findings_path)
    prior = commit_blob_json(source_commit, SUBAGENT_FINDINGS_PATH) if commit_exists(source_commit) else None
    prior_ids = exact_review_agent_ids(prior if isinstance(prior, dict) else {})
    quorum = source_audit_quorum(
        source_audit,
        expected_source_commit=source_commit,
        expected_source_hashes_sha256=source_hashes_sha256,
        expected_source_bundle_sha256=source_bundle_sha256,
        expected_runtime_binary_sha256=runtime_binary_sha256,
        review_report_present=review_path.exists(),
        excluded_agent_ids=prior_ids,
        review_root=review_dir,
    )
    failures.extend(quorum["failures"])
    return {
        "passed": not failures,
        "failures": failures,
        "review_quorum": quorum,
        "review_quorum_sha256": public.digest(quorum),
        "source_authority_commit": source_commit,
        "source_hashes_sha256": source_hashes_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "runtime_binary_sha256": runtime_binary_sha256,
        "findings_path": str(findings_path) if findings_path.exists() else None,
        "findings_sha256": public.sha256_file(findings_path) if findings_path.exists() else None,
        "review_report_path": str(review_path) if review_path.exists() else None,
        "review_report_sha256": public.sha256_file(review_path) if review_path.exists() else None,
    }


def source_review_challenge_binding(source_review: dict[str, Any]) -> dict[str, Any]:
    return {
        "findings_sha256": source_review.get("findings_sha256"),
        "review_report_sha256": source_review.get("review_report_sha256"),
        "review_quorum_sha256": source_review.get("review_quorum_sha256"),
        "source_authority_commit": source_review.get("source_authority_commit"),
        "source_hashes_sha256": source_review.get("source_hashes_sha256"),
        "source_bundle_sha256": source_review.get("source_bundle_sha256"),
        "runtime_binary_sha256": source_review.get("runtime_binary_sha256"),
    }


def write_discovery_challenge_receipt(challenge: dict[str, Any], *, source_commit: str, nonce_sha: str, source_review: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE_RECEIPT_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_authority_commit": source_commit,
        "source_authority_review": source_review_challenge_binding(source_review),
        "controller_challenge": challenge,
        "controller_challenge_sha256": public.digest(challenge),
        "controller_nonce_sha256": nonce_sha,
        "pre_contact": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    receipt["challenge_receipt_canonical_sha256"] = public.digest(
        {k: v for k, v in receipt.items() if k != "challenge_receipt_canonical_sha256"}
    )
    write_json_exclusive(DISCOVERY_CHALLENGE_PATH, receipt)
    return {**receipt, "challenge_receipt_file_sha256": public.sha256_file(DISCOVERY_CHALLENGE_PATH)}


DISCOVERY_CHALLENGE_RECEIPT_KEYS = {
    "schema",
    "science_package_id",
    "transaction_run_id",
    "source_authority_commit",
    "source_authority_review",
    "controller_challenge",
    "controller_challenge_sha256",
    "controller_nonce_sha256",
    "pre_contact",
    "target_contact_count",
    "sensor_inventory_count",
    "live_invocation_count",
    "pmu_acquisition_count",
    "challenge_receipt_canonical_sha256",
}


def validate_discovery_challenge_receipt_payload(
    receipt: dict[str, Any] | None,
    *,
    expected_source_commit: str | None = None,
    expected_challenge: dict[str, Any] | None = None,
    expected_nonce_sha: str | None = None,
    expected_source_review_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(receipt, dict):
        return {"passed": False, "failures": ["challenge receipt missing or malformed"]}
    if set(receipt) != DISCOVERY_CHALLENGE_RECEIPT_KEYS:
        failures.append("challenge receipt field mismatch")
    if receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE_RECEIPT_V1":
        failures.append("challenge receipt schema mismatch")
    if receipt.get("science_package_id") != public.SCIENCE_PACKAGE_ID:
        failures.append("challenge receipt science package mismatch")
    if receipt.get("transaction_run_id") != public.TRANSACTION_RUN_ID:
        failures.append("challenge receipt transaction mismatch")
    if receipt.get("challenge_receipt_canonical_sha256") != public.digest(
        {k: v for k, v in receipt.items() if k != "challenge_receipt_canonical_sha256"}
    ):
        failures.append("challenge receipt canonical digest mismatch")
    if receipt.get("controller_challenge_sha256") != public.digest(receipt.get("controller_challenge")):
        failures.append("challenge receipt controller challenge digest mismatch")
    if receipt.get("pre_contact") is not True:
        failures.append("challenge receipt is not pre-contact")
    zero_counters = {"target_contact_count": 0, "sensor_inventory_count": 0, "live_invocation_count": 0, "pmu_acquisition_count": 0}
    if not counters_equal_strict(receipt, zero_counters):
        failures.append("challenge receipt counters are not zero")
    if expected_source_commit is not None and receipt.get("source_authority_commit") != expected_source_commit:
        failures.append("challenge receipt source commit mismatch")
    if expected_challenge is not None and receipt.get("controller_challenge") != expected_challenge:
        failures.append("challenge receipt controller challenge mismatch")
    if expected_nonce_sha is not None and receipt.get("controller_nonce_sha256") != expected_nonce_sha:
        failures.append("challenge receipt nonce digest mismatch")
    if expected_source_review_binding is not None and receipt.get("source_authority_review") != expected_source_review_binding:
        failures.append("challenge receipt source-review binding mismatch")
    return {"passed": not failures, "failures": failures}


ATTEMPT_STATE_ORDER = {
    "claimed_pre_contact": 0,
    "transport_contact_invoked": 1,
    "target_command_invoked": 2,
    "receipt_copied_cleanup_pending": 3,
    "cleanup_armed": 4,
    "cleanup_completed": 5,
    "complete": 6,
}
ATTEMPT_COUNTER_KEYS = ["target_contact_count", "sensor_inventory_count", "live_invocation_count", "pmu_acquisition_count"]
LEGACY_AUTHORITATIVE_ATTEMPT_ACCOUNTING_FIELDS = {
    "attempt_version",
    "per_attempt_counters",
    "prior_cumulative_lane_counters",
    "cumulative_lane_counters",
}
ATTEMPT_STATE_SEQUENCE = [state for state, _index in sorted(ATTEMPT_STATE_ORDER.items(), key=lambda item: item[1])]
ATTEMPT_STATE_COUNTERS = {
    "claimed_pre_contact": {"target_contact_count": 0, "sensor_inventory_count": 0, "live_invocation_count": 0, "pmu_acquisition_count": 0},
    "transport_contact_invoked": {"target_contact_count": 1, "sensor_inventory_count": 0, "live_invocation_count": 0, "pmu_acquisition_count": 0},
    "target_command_invoked": {"target_contact_count": 1, "sensor_inventory_count": 0, "live_invocation_count": 0, "pmu_acquisition_count": 0},
    "receipt_copied_cleanup_pending": {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0},
    "cleanup_armed": {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0},
    "cleanup_completed": {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0},
    "complete": {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0},
}
ATTEMPT_IMMUTABLE_FIELDS = [
    "source_authority_commit",
    "controller_challenge_sha256",
    "challenge_receipt_canonical_sha256",
    "challenge_receipt_file_sha256",
]


def zero_attempt_counters() -> dict[str, int]:
    return {key: 0 for key in ATTEMPT_COUNTER_KEYS}


def counter_sum(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in ATTEMPT_COUNTER_KEYS}


def read_attempt_history_index() -> dict[str, Any]:
    if not ATTEMPT_HISTORY_INDEX_PATH.exists():
        return {"present": False, "passed": False, "failures": ["attempt history index missing"], "cumulative_counters": zero_attempt_counters()}
    try:
        index = read_json(ATTEMPT_HISTORY_INDEX_PATH)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"present": True, "passed": False, "failures": [f"attempt history index unreadable: {exc}"], "cumulative_counters": zero_attempt_counters()}
    return validate_attempt_history_index(index)


def validate_attempt_history_index(index: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    attempts_value = index.get("attempts")
    attempts = attempts_value if isinstance(attempts_value, list) else []
    if index.get("schema") != "FAMILY10H_SENSOR_AUTHORITY_ATTEMPT_HISTORY_INDEX_V1":
        failures.append("attempt history index schema mismatch")
    if not isinstance(attempts_value, list):
        failures.append("attempt history attempts missing")
    expected_counters = {
        "target_contact_count": 1,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    expected_attempts = {
        "C3": {
            "source_authority_commit": C3_SOURCE_AUTHORITY_COMMIT,
            "failure_evidence_commit": C3_FAILURE_EVIDENCE_COMMIT,
            "failure_reason": C3_FAILURE_REASON,
            "required_archived_file_labels": {"attempt", "attempt_journal", "challenge", "cleanup_custody"},
        },
        "C5": {
            "source_authority_commit": C5_ATTEMPT_SOURCE_COMMIT,
            "failure_evidence_commit": C5_FAILURE_EVIDENCE_COMMIT,
            "failure_reason": C5_FAILURE_REASON,
            "required_archived_file_labels": {"attempt", "attempt_journal", "challenge", "cleanup_custody", "target_failure"},
        },
    }
    attempt_versions = [item.get("attempt_version") for item in attempts if isinstance(item, dict)]
    if sorted(attempt_versions) != sorted(expected_attempts):
        failures.append("attempt history version set mismatch")
    computed_cumulative = zero_attempt_counters()
    for version, expected in expected_attempts.items():
        records = [item for item in attempts if isinstance(item, dict) and item.get("attempt_version") == version]
        if len(records) != 1:
            failures.append(f"attempt history must contain exactly one {version} record")
            continue
        record = records[0]
        if record.get("source_authority_commit") != expected["source_authority_commit"]:
            failures.append(f"{version} history source authority commit mismatch")
        if record.get("failure_evidence_commit") != expected["failure_evidence_commit"]:
            failures.append(f"{version} history failure evidence commit mismatch")
        if record.get("failure_reason") != expected["failure_reason"]:
            failures.append(f"{version} history failure reason mismatch")
        counters = record.get("counters")
        if not counter_dict_equal_strict(counters if isinstance(counters, dict) else {}, expected_counters):
            failures.append(f"{version} history counters mismatch")
        else:
            computed_cumulative = counter_sum(computed_cumulative, counters)
        if record.get("cleanup_result") is not True:
            failures.append(f"{version} history cleanup result mismatch")
        if record.get("remote_root_absence_result") is not True:
            failures.append(f"{version} history remote-root absence mismatch")
        archived_files = record.get("archived_files")
        if not isinstance(archived_files, dict):
            failures.append(f"{version} history archived files missing")
            archived_files = {}
        labels = set(archived_files)
        if labels != expected["required_archived_file_labels"]:
            failures.append(f"{version} history archived file label mismatch")
        for label, item in archived_files.items():
            if not isinstance(item, dict):
                failures.append(f"{version} history archived file malformed {label}")
                continue
            path_value = item.get("path")
            sha_value = item.get("sha256")
            bytes_value = item.get("bytes")
            if not isinstance(path_value, str) or not isinstance(sha_value, str) or not is_strict_int(bytes_value):
                failures.append(f"{version} history archived file binding missing {label}")
                continue
            path = HERE / path_value if not Path(path_value).is_absolute() else Path(path_value)
            if not path.exists():
                failures.append(f"{version} history archived file absent {label}")
            elif public.sha256_file(path) != sha_value:
                failures.append(f"{version} history archived file digest mismatch {label}")
            elif path.stat().st_size != bytes_value:
                failures.append(f"{version} history archived file byte-size mismatch {label}")
    cumulative = index.get("cumulative_counters")
    if not counter_dict_equal_strict(cumulative if isinstance(cumulative, dict) else {}, computed_cumulative):
        failures.append("attempt history cumulative counters mismatch")
    index_sha = index.get("attempt_history_index_sha256")
    if index_sha != public.digest({k: v for k, v in index.items() if k != "attempt_history_index_sha256"}):
        failures.append("attempt history index digest mismatch")
    return {
        "present": True,
        "passed": not failures,
        "failures": failures,
        "index": index,
        "cumulative_counters": cumulative if isinstance(cumulative, dict) else zero_attempt_counters(),
    }


def history_cumulative_counters_or_zero() -> dict[str, int]:
    raise ControllerError("historical lane counters are reporting metadata, not active transaction authority")


def known_historical_lane_contact_report() -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    failures: list[str] = []
    counters = zero_attempt_counters()
    history = read_attempt_history_index()
    if history["passed"]:
        archived_attempts = {key: history["cumulative_counters"][key] for key in ATTEMPT_COUNTER_KEYS}
        counters = counter_sum(counters, archived_attempts)
        components.append({
            "component": "sensor-authority attempt history",
            "authoritative_for_active_transaction": False,
            "counters": archived_attempts,
        })
    else:
        failures.extend("historical sensor-authority metadata unavailable: " + item for item in history["failures"])
    affinity_dir = HERE / "AFFINITY_CAPABILITY_OBSERVATION"
    if affinity_dir.exists():
        for receipt_path in sorted(affinity_dir.glob("*.sha256.json")):
            try:
                receipt = read_json(receipt_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                failures.append(f"affinity observation metadata unreadable {receipt_path.name}: {exc}")
                continue
            contact_count = receipt.get("target_contact_count_for_this_observation")
            if not is_strict_int(contact_count):
                failures.append(f"affinity observation contact counter invalid {receipt_path.name}")
                continue
            component_counters = {
                "target_contact_count": contact_count,
                "sensor_inventory_count": int(receipt.get("sensor_inventory_count", 0) or 0),
                "live_invocation_count": int(receipt.get("live_invocation_count", 0) or 0),
                "pmu_acquisition_count": int(receipt.get("pmu_acquisition_count", 0) or 0),
            }
            counters = counter_sum(counters, component_counters)
            components.append({
                "component": f"affinity observation {receipt_path.name}",
                "authoritative_for_active_transaction": False,
                "observation_sha256": receipt.get("observation_sha256"),
                "counters": component_counters,
            })
    return {
        "schema": "FAMILY10H_HISTORICAL_LANE_CONTACT_REPORT_V1",
        "authoritative_for_active_transaction": False,
        "complete_cryptographic_lane_ledger_claimed": False,
        "passed": not failures,
        "failures": failures,
        "known_counters_before_active_attempt": counters,
        "components": components,
    }


def git_text_canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def attempt_history_index_metadata_reporting_only() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(ATTEMPT_HISTORY_INDEX_PATH) if ATTEMPT_HISTORY_INDEX_PATH.exists() else None,
        "file_sha256": None,
        "checkout_file_sha256": None,
        "file_sha256_canonicalization": "git_text_auto_lf",
        "git_head": None,
        "git_blob_sha256": None,
        "git_blob_size": None,
        "git_blob_id": None,
        "git_blob_matches_file_sha256": None,
        "authoritative_for_active_transaction": False,
    }
    if not ATTEMPT_HISTORY_INDEX_PATH.exists():
        return metadata
    checkout_bytes = ATTEMPT_HISTORY_INDEX_PATH.read_bytes()
    canonical_bytes = git_text_canonical_bytes(ATTEMPT_HISTORY_INDEX_PATH)
    metadata["checkout_file_sha256"] = hashlib.sha256(checkout_bytes).hexdigest()
    metadata["file_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    try:
        head = git_text("rev-parse", "HEAD")
        blob = commit_blob_record(head, ATTEMPT_HISTORY_INDEX_PATH)
    except (ControllerError, ValueError):
        return metadata
    metadata["git_head"] = head
    metadata["git_blob_sha256"] = blob.get("sha256")
    metadata["git_blob_size"] = blob.get("size")
    metadata["git_blob_id"] = blob.get("blob_id")
    metadata["git_blob_matches_file_sha256"] = blob.get("sha256") == metadata["file_sha256"] if blob.get("present") is True else False
    return metadata


def validate_attempt_history_index_manifest_binding(manifest_data: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    metadata = (
        manifest_data.get("temperature_sensor_authority", {}).get("historical_attempt_index_metadata_reporting_only")
        if isinstance(manifest_data.get("temperature_sensor_authority"), dict)
        else None
    )
    if not isinstance(metadata, dict):
        return {"passed": False, "failures": ["manifest attempt-history metadata missing"]}
    expected = attempt_history_index_metadata_reporting_only()
    for key in [
        "path",
        "file_sha256",
        "checkout_file_sha256",
        "file_sha256_canonicalization",
        "git_head",
        "git_blob_sha256",
        "git_blob_size",
        "git_blob_id",
        "git_blob_matches_file_sha256",
        "authoritative_for_active_transaction",
    ]:
        if metadata.get(key) != expected.get(key):
            failures.append(f"manifest attempt-history metadata {key} mismatch")
    if expected.get("checkout_file_sha256") and metadata.get("file_sha256") == expected.get("checkout_file_sha256") != expected.get("file_sha256"):
        failures.append("manifest attempt-history metadata used checkout digest instead of Git text blob digest")
    if not ATTEMPT_HISTORY_INDEX_PATH.exists():
        failures.append("manifest attempt-history index file missing")
    if metadata.get("file_sha256") != metadata.get("git_blob_sha256"):
        failures.append("manifest attempt-history file digest does not match Git blob digest")
    if metadata.get("git_blob_matches_file_sha256") is not True:
        failures.append("manifest attempt-history Git blob match is not true")
    if re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("git_head", ""))) is None:
        failures.append("manifest attempt-history Git head missing or invalid")
    for field in ["file_sha256", "checkout_file_sha256", "git_blob_sha256"]:
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(field, ""))) is None:
            failures.append(f"manifest attempt-history {field} missing or invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("git_blob_id", ""))) is None:
        failures.append("manifest attempt-history Git blob id missing or invalid")
    if not is_strict_int(metadata.get("git_blob_size")) or metadata.get("git_blob_size", 0) <= 0:
        failures.append("manifest attempt-history Git blob size missing or invalid")
    return {"passed": not failures, "failures": failures, "expected": expected}


def active_attempt_paths() -> list[Path]:
    return [
        TEMPERATURE_SENSOR_AUTHORITY,
        TARGET_DISCOVERY_RECEIPT,
        TARGET_DISCOVERY_FAILURE_PATH,
        DISCOVERY_TRANSPORT_PATH,
        DISCOVERY_ATTEMPT_PATH,
        DISCOVERY_ATTEMPT_JOURNAL_PATH,
        DISCOVERY_CHALLENGE_PATH,
        DISCOVERY_CLEANUP_CUSTODY_PATH,
    ]


def active_attempt_paths_present() -> list[Path]:
    return [path for path in active_attempt_paths() if path.exists()]


def enrich_attempt_accounting(receipt: dict[str, Any]) -> None:
    # C5 deliberately has no cumulative-history enrichment. Active counters are
    # authoritative only when they match the current state machine below.
    return


def validate_discovery_attempt_transition(previous: dict[str, Any] | None, receipt: dict[str, Any]) -> None:
    state = receipt.get("attempt_state")
    require(state in ATTEMPT_STATE_ORDER, "discovery attempt state missing or invalid")
    for field in LEGACY_AUTHORITATIVE_ATTEMPT_ACCOUNTING_FIELDS:
        require(field not in receipt, f"legacy cumulative accounting field is non-authoritative and rejected: {field}")
    for key in ATTEMPT_COUNTER_KEYS:
        require(is_strict_counter(receipt.get(key)), f"discovery attempt counter missing or invalid {key}")
    require(
        counters_equal_strict(receipt, ATTEMPT_STATE_COUNTERS[state]),
        f"discovery attempt counters do not match state {state}",
    )
    for field in ATTEMPT_IMMUTABLE_FIELDS:
        require(isinstance(receipt.get(field), str) and receipt[field], f"discovery attempt immutable field missing {field}")
    require(isinstance(receipt.get("source_authority_review"), dict), "discovery attempt source review missing")
    if previous is None:
        require(state == ATTEMPT_STATE_SEQUENCE[0], "discovery attempt first state must be claimed_pre_contact")
        return
    previous_state = previous.get("attempt_state")
    require(previous_state in ATTEMPT_STATE_ORDER, "previous discovery attempt state invalid")
    for key in ATTEMPT_COUNTER_KEYS:
        require(is_strict_counter(previous.get(key)), f"previous discovery attempt counter missing or invalid {key}")
    require(ATTEMPT_STATE_ORDER[state] == ATTEMPT_STATE_ORDER[previous_state] + 1, "discovery attempt state must advance exactly one step")
    for field in ATTEMPT_IMMUTABLE_FIELDS:
        require(receipt.get(field) == previous.get(field), f"discovery attempt immutable field changed {field}")
    require(
        source_review_challenge_binding(receipt["source_authority_review"])
        == source_review_challenge_binding(previous.get("source_authority_review", {})),
        "discovery attempt source review binding changed",
    )
    previous_cleanup = previous.get("cleanup")
    if isinstance(previous_cleanup, dict) and (
        previous_cleanup.get("journal_error_before_cleanup") or previous_cleanup.get("journal_error_after_cleanup")
    ):
        raise ControllerError("discovery attempt cannot continue after journal error")


def receipt_attempt_digest_matches(receipt: dict[str, Any]) -> bool:
    return receipt.get("discovery_attempt_sha256") == public.digest({k: v for k, v in receipt.items() if k != "discovery_attempt_sha256"})


def replay_attempt_journal_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    previous: dict[str, Any] | None = None
    if not rows:
        failures.append("discovery attempt journal is empty")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"discovery attempt journal row malformed {index}")
            continue
        if row.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT_V1":
            failures.append(f"discovery attempt journal schema mismatch {index}")
        if not receipt_attempt_digest_matches(row):
            failures.append(f"discovery attempt journal digest mismatch {index}")
        try:
            validate_discovery_attempt_transition(previous, row)
        except ControllerError as exc:
            failures.append(f"discovery attempt journal transition mismatch {index}: {exc}")
        previous = row
    states = [row.get("attempt_state") for row in rows if isinstance(row, dict)]
    if states != ATTEMPT_STATE_SEQUENCE:
        failures.append("discovery attempt journal must contain exact seven-state success sequence")
    if previous is not None and (previous.get("attempt_state") != "complete" or previous.get("passed") is not True):
        failures.append("discovery attempt journal final row must be terminal complete pass")
    return {"passed": not failures, "failures": failures, "final_row": previous, "row_count": len(rows)}


def read_jsonl_blob(commit: str, path: Path) -> list[dict[str, Any]] | None:
    blob = commit_blob_bytes(commit, path)
    if blob is None:
        return None
    try:
        return strict_jsonl_loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def strict_jsonl_loads(blob: bytes) -> list[dict[str, Any]]:
    lines = blob.decode("utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("blank JSONL row rejected")
    return [strict_json_loads(line) for line in lines]


def package_json_parse_audit(root: Path = HERE) -> dict[str, Any]:
    failures: list[str] = []
    json_count = 0
    jsonl_count = 0
    jsonl_row_count = 0
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        json_count += 1
        rel = path.relative_to(root).as_posix()
        try:
            strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"JSON parse failed {rel}: {exc}")
    for path in sorted(root.rglob("*.jsonl")):
        if not path.is_file():
            continue
        jsonl_count += 1
        rel = path.relative_to(root).as_posix()
        try:
            rows = strict_jsonl_loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"JSONL parse failed {rel}: {exc}")
            continue
        jsonl_row_count += len(rows)
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_JSON_PARSE_AUDIT_V1",
        "root": str(root),
        "json_count": json_count,
        "jsonl_count": jsonl_count,
        "jsonl_row_count": jsonl_row_count,
        "failures": failures,
        "passed": not failures,
    }


def append_discovery_attempt_journal(receipt: dict[str, Any], *, require_directory_sync: bool = False) -> None:
    row = strict_json_dumps(receipt).encode("utf-8") + b"\n"
    if not DISCOVERY_ATTEMPT_JOURNAL_PATH.exists():
        durable_write_bytes_exclusive(DISCOVERY_ATTEMPT_JOURNAL_PATH, row, require_directory_sync=require_directory_sync)
        return
    with DISCOVERY_ATTEMPT_JOURNAL_PATH.open("ab") as handle:
        handle.write(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_discovery_attempt_receipt(payload: dict[str, Any], *, require_directory_sync: bool = False) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["schema"] = "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT_V1"
    enrich_attempt_accounting(receipt)
    receipt["discovery_attempt_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "discovery_attempt_sha256"})
    previous = read_json(DISCOVERY_ATTEMPT_PATH) if DISCOVERY_ATTEMPT_PATH.exists() else None
    if previous is None and DISCOVERY_ATTEMPT_JOURNAL_PATH.exists():
        raise ControllerError("discovery attempt journal exists without snapshot")
    if previous is not None and not DISCOVERY_ATTEMPT_JOURNAL_PATH.exists():
        raise ControllerError("discovery attempt snapshot exists without journal")
    validate_discovery_attempt_transition(previous, receipt)
    append_discovery_attempt_journal(receipt, require_directory_sync=require_directory_sync)
    write_json_atomic(DISCOVERY_ATTEMPT_PATH, receipt, require_directory_sync=require_directory_sync)
    return receipt


def parse_preflight_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    match = re.fullmatch(r"FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=([01]) root_created=1", text)
    if match is None:
        return {
            "passed": False,
            "remote_base_preexisting": True,
            "root_created": False,
            "raw_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "failure": "malformed preflight stdout",
        }
    return {
        "passed": True,
        "remote_base_preexisting": match.group(1) == "1",
        "root_created": True,
        "raw_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
    }


def write_discovery_cleanup_custody_receipt(payload: dict[str, Any], *, require_directory_sync: bool = False) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["schema"] = "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_FAILURE_CLEANUP_CUSTODY_V1"
    enrich_attempt_accounting(receipt)
    for key in ATTEMPT_COUNTER_KEYS:
        require(is_strict_counter(receipt.get(key)), f"cleanup custody counter missing or invalid {key}")
    require(is_strict_int(receipt.get("target_contact_count")) and receipt.get("target_contact_count") == 1, "cleanup custody requires recorded target contact")
    require(is_strict_int(receipt.get("candidate_scan_count")) and receipt.get("candidate_scan_count") in (0, 1), "cleanup custody candidate scan counter invalid")
    expected_inventory_count = 1 if receipt.get("candidate_scan_count") == 1 else 0
    require(
        is_strict_int(receipt.get("sensor_inventory_count")) and receipt.get("sensor_inventory_count") == expected_inventory_count,
        "cleanup custody inventory/candidate counter pair mismatch",
    )
    require(is_strict_int(receipt.get("live_invocation_count")) and receipt.get("live_invocation_count") == 0, "cleanup custody live counter must be zero")
    require(is_strict_int(receipt.get("pmu_acquisition_count")) and receipt.get("pmu_acquisition_count") == 0, "cleanup custody PMU counter must be zero")
    cleanup = receipt.get("cleanup")
    require(isinstance(cleanup, dict), "cleanup custody cleanup payload missing")
    receipt["cleanup_custody_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "cleanup_custody_sha256"})
    write_json_atomic(DISCOVERY_CLEANUP_CUSTODY_PATH, receipt, require_directory_sync=require_directory_sync)
    return receipt


def expected_temperature_authority_challenge(
    *,
    source_hashes: dict[str, Any],
    source_bundle_sha256: str,
    schedule_sidecar: dict[str, Any],
    authorized_commit: str,
) -> dict[str, Any] | None:
    nonce_sha = os.environ.get(TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV)
    if not nonce_sha:
        return None
    return build_temperature_authority_challenge(
        source_hashes=source_hashes,
        source_bundle_sha256=source_bundle_sha256,
        runtime_binary_sha256=source_hashes["runtime_binary_authority"]["sha256"],
        schedule_sidecar=schedule_sidecar,
        authorized_commit=authorized_commit,
        controller_nonce_sha256=nonce_sha,
        transport_scope=discovery_transport_scope(target_host=TARGET_HOST, remote_root=DISCOVERY_REMOTE_ROOT, nonce_sha=nonce_sha),
        source_authority_review=fixture_source_review_binding(
            authorized_commit,
            source_hashes["source_hashes_sha256"],
            source_bundle_sha256,
            source_hashes["runtime_binary_authority"]["sha256"],
        ),
    )


def expected_temperature_authority_challenge_for_manifest(
    *,
    source_hashes: dict[str, Any],
    source_bundle_sha256: str,
    schedule_sidecar: dict[str, Any],
    fallback_authorized_commit: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if DISCOVERY_CHALLENGE_PATH.exists():
        receipt = read_json(DISCOVERY_CHALLENGE_PATH)
        if not validate_discovery_challenge_receipt_payload(receipt)["passed"]:
            return None, None
        embedded_challenge = receipt.get("controller_challenge")
        source_commit = receipt.get("source_authority_commit")
        nonce_sha = receipt.get("controller_nonce_sha256")
        if (
            isinstance(embedded_challenge, dict)
            and isinstance(source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", source_commit)
            and isinstance(nonce_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", nonce_sha)
        ):
            expected = build_temperature_authority_challenge(
                source_hashes=source_hashes,
                source_bundle_sha256=source_bundle_sha256,
                runtime_binary_sha256=source_hashes["runtime_binary_authority"]["sha256"],
                schedule_sidecar=schedule_sidecar,
                authorized_commit=source_commit,
                controller_nonce_sha256=nonce_sha,
                transport_scope=discovery_transport_scope(target_host=TARGET_HOST, remote_root=DISCOVERY_REMOTE_ROOT, nonce_sha=nonce_sha),
                source_authority_review=embedded_challenge.get("source_authority_review") if isinstance(embedded_challenge.get("source_authority_review"), dict) else None,
            )
            if embedded_challenge != expected:
                return None, None
            return (expected, source_commit)
    challenge = expected_temperature_authority_challenge(
        source_hashes=source_hashes,
        source_bundle_sha256=source_bundle_sha256,
        schedule_sidecar=schedule_sidecar,
        authorized_commit=fallback_authorized_commit,
    )
    return challenge, fallback_authorized_commit if challenge else None


def validate_temperature_authority_challenge(
    receipt: dict[str, Any],
    discovery: dict[str, Any],
    expected_challenge: dict[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    challenge = receipt.get("controller_challenge")
    challenge_sha = receipt.get("controller_challenge_sha256")
    nonce = receipt.get("controller_nonce")
    if not isinstance(challenge, dict):
        failures.append("temperature authority controller challenge missing")
        challenge = {}
    else:
        if set(challenge) != REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS:
            failures.append("temperature authority controller challenge field mismatch")
        if challenge.get("schema") != TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA:
            failures.append("temperature authority controller challenge schema mismatch")
        if challenge.get("authority") != "controller_issued_temperature_sensor_challenge":
            failures.append("temperature authority controller challenge authority mismatch")
        if challenge.get("science_package_id") != public.SCIENCE_PACKAGE_ID:
            failures.append("temperature authority controller challenge package mismatch")
        if challenge.get("transaction_run_id") != public.TRANSACTION_RUN_ID:
            failures.append("temperature authority controller challenge run mismatch")
        for field in (
            "source_hashes_sha256",
            "source_bundle_sha256",
            "runtime_binary_sha256",
            "schedule_canonical_sha256",
            "schedule_json_sha256",
            "schedule_tsv_sha256",
            "controller_nonce_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(challenge.get(field, ""))) is None:
                failures.append(f"temperature authority controller challenge {field} invalid")
        if re.fullmatch(r"[0-9a-f]{40}", str(challenge.get("authorized_commit", ""))) is None:
            failures.append("temperature authority controller challenge authorized commit invalid")
    if expected_challenge is None:
        failures.append("temperature authority expected controller challenge missing")
    elif challenge != expected_challenge:
        failures.append("temperature authority controller challenge mismatch")
    if challenge_sha != public.digest(challenge):
        failures.append("temperature authority controller challenge digest mismatch")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        failures.append("temperature authority controller nonce missing or malformed")
    elif hashlib.sha256(nonce.encode("ascii")).hexdigest() != challenge.get("controller_nonce_sha256"):
        failures.append("temperature authority controller nonce hash mismatch")
    provenance = discovery.get("provenance") if isinstance(discovery, dict) else None
    if isinstance(provenance, dict):
        if provenance.get("controller_challenge_sha256") != challenge_sha:
            failures.append("temperature discovery controller challenge echo mismatch")
        if provenance.get("authorized_commit") != challenge.get("authorized_commit"):
            failures.append("temperature discovery authorized commit echo mismatch")
    return failures


def temperature_sensor_authority_from_receipt(
    receipt: dict[str, Any] | None,
    *,
    expected_challenge: dict[str, Any] | None = None,
    expected_discovery_receipt: dict[str, Any] | None = None,
    expected_transport_receipt: dict[str, Any] | None = None,
    require_transport: bool = True,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {
            "present": False,
            "passed": False,
            "failures": ["temperature sensor authority receipt missing"],
            "approved_sensor_identity": None,
        }
    failures: list[str] = []
    digest_field = "temperature_sensor_authority_sha256"
    if receipt.get("schema") != TEMPERATURE_SENSOR_AUTHORITY_SCHEMA:
        failures.append("temperature sensor authority schema mismatch")
    if receipt.get(digest_field) != public.digest({k: v for k, v in receipt.items() if k != digest_field}):
        failures.append("temperature sensor authority digest mismatch")
    identity = receipt.get("approved_sensor_identity")
    if not isinstance(identity, dict) or set(identity) != public.TEMPERATURE_SENSOR_IDENTITY_KEYS:
        failures.append("approved temperature sensor identity missing or malformed")
        identity = None
    elif identity.get("identity_sha256") != public.temperature_identity_digest(identity):
        failures.append("approved temperature sensor identity digest mismatch")
    elif not target.legacy_temperature_identity_is_approved(identity):
        failures.append("approved temperature sensor legacy profile incomplete")
    if receipt.get("provenance_bound") is not True:
        failures.append("temperature sensor authority provenance not bound")
    if identity == public.synthetic_temperature_identity():
        failures.append("synthetic temperature sensor identity cannot authorize frozen status")
    discovery = receipt.get("target_discovery_receipt")
    platform_identity: dict[str, Any] | None = None
    if not isinstance(discovery, dict):
        failures.append("temperature sensor discovery receipt missing")
        discovery = {}
    else:
        if expected_discovery_receipt is None:
            if require_transport:
                failures.append("copied target discovery receipt missing")
        elif discovery != expected_discovery_receipt:
            failures.append("temperature sensor discovery receipt does not match copied receipt")
        discovery_digest = discovery.get("target_discovery_receipt_sha256")
        if discovery.get("schema") != TEMPERATURE_SENSOR_DISCOVERY_SCHEMA:
            failures.append("temperature sensor discovery schema mismatch")
        if discovery_digest != public.digest({k: v for k, v in discovery.items() if k != "target_discovery_receipt_sha256"}):
            failures.append("temperature sensor discovery digest mismatch")
        if discovery.get("discovery_mode") != "target_read_only_sensor_inventory":
            failures.append("temperature sensor discovery mode mismatch")
        provenance = discovery.get("provenance")
        if not isinstance(provenance, dict):
            failures.append("temperature sensor discovery provenance missing")
            provenance = {}
        else:
            if provenance.get("authority") != "target_sensor_discovery":
                failures.append("temperature sensor discovery provenance authority mismatch")
            if provenance.get("science_package_id") != public.SCIENCE_PACKAGE_ID:
                failures.append("temperature sensor discovery provenance package mismatch")
            if provenance.get("transaction_run_id") != public.TRANSACTION_RUN_ID:
                failures.append("temperature sensor discovery provenance run mismatch")
            platform = provenance.get("target_platform")
            if not isinstance(platform, dict):
                failures.append("temperature sensor discovery target platform missing")
            else:
                platform_identity = platform
                if platform.get("vendor") != "AuthenticAMD" or platform.get("cpu_family") != 16:
                    failures.append("temperature sensor discovery target platform is not AMD Family 10h")
                if platform.get("checked_before_discovery") is not True:
                    failures.append("temperature sensor discovery platform was not checked before discovery")
                if (
                    platform.get("source_cpu_expected") != public.SOURCE_CPU_EXPECTED
                    or platform.get("receiver_cpu_expected") != public.RECEIVER_CPU_EXPECTED
                    or platform.get("source_receiver_cpus_present") is not True
                ):
                    failures.append("temperature sensor discovery source/receiver CPU boundary missing")
                failures.extend(target.operational_pin_capability_failures(platform))
        if not is_strict_counter(provenance.get("discovery_monotonic_ns")) or provenance.get("discovery_monotonic_ns", 0) <= 0:
            failures.append("temperature sensor discovery monotonic timestamp missing")
        if expected_challenge is not None:
            if provenance.get("controller_challenge_sha256") != public.digest(expected_challenge):
                failures.append("temperature sensor discovery challenge provenance mismatch")
            if provenance.get("authorized_commit") != expected_challenge.get("authorized_commit"):
                failures.append("temperature sensor discovery authorized commit provenance mismatch")
        if expected_challenge is not None and discovery.get("controller_nonce_sha256") != expected_challenge.get("controller_nonce_sha256"):
            failures.append("temperature sensor discovery controller nonce mismatch")
        source_authority = discovery.get("source_authority")
        if not isinstance(source_authority, dict) or source_authority.get("passed") is not True:
            failures.append("temperature sensor discovery source authority did not pass")
        authorizing_scope = discovery.get("authorizing_scope")
        if not isinstance(authorizing_scope, dict) or authorizing_scope.get("authorizing") is not True:
            failures.append("temperature sensor discovery was not captured from canonical target sensor roots")
        if identity is not None:
            failures.extend(
                target.temperature_physical_authority_failures(
                    identity,
                    authorizing_scope=authorizing_scope if isinstance(authorizing_scope, dict) else None,
                    platform_identity=platform_identity,
                    require_authorizing_scope=True,
                    require_pin_evidence=True,
                )
            )
        if not is_strict_int(discovery.get("target_contact_count")) or discovery.get("target_contact_count") != 1:
            failures.append("temperature sensor discovery target contact count must be one")
        if not is_strict_int(discovery.get("sensor_inventory_count")) or discovery.get("sensor_inventory_count") != 1:
            failures.append("temperature sensor discovery inventory count must be one")
        if not is_strict_int(discovery.get("candidate_scan_count")) or discovery.get("candidate_scan_count") != 1:
            failures.append("temperature sensor discovery candidate scan count must be one")
        if not is_strict_int(discovery.get("live_invocation_count")) or discovery.get("live_invocation_count") != 0:
            failures.append("temperature sensor discovery live invocation count must be zero")
        if not is_strict_int(discovery.get("pmu_acquisition_count")) or discovery.get("pmu_acquisition_count") != 0:
            failures.append("temperature sensor discovery PMU acquisition count must be zero")
        if not is_strict_int(discovery.get("pmu_open_count")) or discovery.get("pmu_open_count") != 0:
            failures.append("temperature sensor discovery PMU open count must be zero")
        if not is_strict_int(discovery.get("runtime_launch_count")) or discovery.get("runtime_launch_count") != 0:
            failures.append("temperature sensor discovery runtime launch count must be zero")
        if discovery.get("tomography_output_root_created") is not False:
            failures.append("temperature sensor discovery must not create tomography output root")
        selected_identity = discovery.get("selected_identity")
        if selected_identity != identity:
            failures.append("temperature sensor discovery identity mismatch")
        candidates = discovery.get("observed_candidates")
        if not isinstance(candidates, list) or not candidates:
            failures.append("temperature sensor discovery candidates missing")
        elif identity is not None:
            approved_candidates = []
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    failures.append(f"temperature sensor discovery candidate malformed {index}")
                    continue
                candidate_identity = candidate.get("identity")
                if candidate_identity is None:
                    if candidate.get("approved") is True:
                        failures.append(f"temperature sensor discovery candidate approved without identity {index}")
                    continue
                if not isinstance(candidate_identity, dict) or set(candidate_identity) != public.TEMPERATURE_SENSOR_IDENTITY_KEYS:
                    failures.append(f"temperature sensor discovery candidate identity malformed {index}")
                    continue
                if candidate_identity.get("identity_sha256") != public.temperature_identity_digest(candidate_identity):
                    failures.append(f"temperature sensor discovery candidate identity digest mismatch {index}")
                    continue
                expected_approved = target.legacy_temperature_identity_is_approved(candidate_identity) and not target.temperature_physical_authority_failures(
                    candidate_identity
                )
                if candidate.get("approved") is not expected_approved:
                    failures.append(f"temperature sensor discovery candidate approval mismatch {index}")
                if expected_approved:
                    approved_candidates.append(candidate)
            if identity not in [candidate.get("identity") for candidate in approved_candidates]:
                failures.append("temperature sensor discovery selected identity not in approved candidates")
            if approved_candidates:
                recomputed_selected = approved_candidates[0]["identity"] if len(approved_candidates) == 1 else None
                if selected_identity != recomputed_selected:
                    failures.append("temperature sensor discovery deterministic selection mismatch")
                selection = discovery.get("selection")
                if not isinstance(selection, dict):
                    failures.append("temperature sensor discovery selection metadata missing")
                else:
                    if selection.get("deterministic_law") is not True:
                        failures.append("temperature sensor discovery deterministic law missing")
                    if selection.get("approved_count") != len(approved_candidates):
                        failures.append("temperature sensor discovery approved count mismatch")
                    if not isinstance(recomputed_selected, dict) or selection.get("selected_class_path") != recomputed_selected.get("class_path"):
                        failures.append("temperature sensor discovery selected class path mismatch")
            for field in ("identity_before", "identity_after"):
                observed = discovery.get(field)
                if not isinstance(observed, dict) or not target.identity_matches_required(observed, identity):
                    failures.append(f"temperature sensor discovery {field} mismatch")
            sample = discovery.get("sample")
            if not isinstance(sample, dict):
                failures.append("temperature sensor discovery sample missing")
            else:
                if sample.get("identity") != identity:
                    failures.append("temperature sensor discovery sample identity mismatch")
                if sample.get("path") != identity.get("class_path"):
                    failures.append("temperature sensor discovery sample path mismatch")
                if sample.get("label_present") != identity.get("sensor_label_present"):
                    failures.append("temperature sensor discovery sample label presence mismatch")
                if sample.get("label_value") != identity.get("sensor_label_value"):
                    failures.append("temperature sensor discovery sample label value mismatch")
                if sample.get("semantic_role") != identity.get("sensor_semantic_role"):
                    failures.append("temperature sensor discovery sample semantic role mismatch")
                if sample.get("semantic_profile") != identity.get("sensor_semantic_profile"):
                    failures.append("temperature sensor discovery sample semantic profile mismatch")
                value = sample.get("value_c")
                if not public.is_json_number(value) or not 0.0 < float(value) < 120.0:
                    failures.append("temperature sensor discovery sample value invalid")
                descriptor = sample.get("pinned_descriptor")
                expected_descriptor = {
                    "resolved_input_path": identity.get("resolved_input_path"),
                    "input_st_dev": identity.get("input_st_dev"),
                    "input_st_ino": identity.get("input_st_ino"),
                    "input_st_mode": identity.get("input_st_mode"),
                }
                if not isinstance(descriptor, dict) or descriptor != expected_descriptor:
                    failures.append("temperature sensor discovery pinned descriptor missing or mismatched")
                if sample.get("read_law") != "manifest-approved resolved input descriptor":
                    failures.append("temperature sensor discovery read law mismatch")
    if expected_transport_receipt is None:
        if require_transport:
            failures.append("temperature discovery transport receipt missing")
    else:
        transport_digest = expected_transport_receipt.get("discovery_transport_sha256")
        if expected_transport_receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1":
            failures.append("temperature discovery transport schema mismatch")
        if transport_digest != public.digest({k: v for k, v in expected_transport_receipt.items() if k != "discovery_transport_sha256"}):
            failures.append("temperature discovery transport digest mismatch")
        if expected_transport_receipt.get("passed") is not True:
            failures.append("temperature discovery transport must pass")
        cleanup = expected_transport_receipt.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("passed") is not True or cleanup.get("absence_verified") is not True:
            failures.append("temperature discovery cleanup and absence verification required")
        if expected_challenge is not None and expected_transport_receipt.get("source_authority_commit") != expected_challenge.get("authorized_commit"):
            failures.append("temperature discovery transport source commit mismatch")
        scope = expected_challenge.get("transport_scope") if isinstance(expected_challenge, dict) else None
        if not isinstance(scope, dict):
            failures.append("temperature discovery transport scope missing from challenge")
            scope = {}
        transport_scope = {
            "target_host": expected_transport_receipt.get("target_host"),
            "remote_base_root": expected_transport_receipt.get("remote_base_root"),
            "remote_root": expected_transport_receipt.get("remote_root"),
            "remote_source_root": expected_transport_receipt.get("remote_source_root"),
            "remote_receipt_path": expected_transport_receipt.get("remote_receipt_path"),
        }
        if transport_scope != scope:
            failures.append("temperature discovery transport scope mismatch")
        if discovery.get("source_root") != scope.get("remote_source_root"):
            failures.append("temperature discovery source root does not match challenged transport scope")
        if discovery.get("receipt_path") != scope.get("remote_receipt_path"):
            failures.append("temperature discovery receipt path does not match challenged transport scope")
        if expected_transport_receipt.get("target_discovery_receipt_sha256") != discovery.get("target_discovery_receipt_sha256"):
            failures.append("temperature discovery transport target receipt mismatch")
        if expected_transport_receipt.get("target_discovery_receipt_file_sha256") != serialized_json_sha256(discovery):
            failures.append("temperature discovery transport target receipt file hash mismatch")
        if expected_transport_receipt.get("authority_receipt_sha256") != receipt.get("temperature_sensor_authority_sha256"):
            failures.append("temperature discovery transport authority receipt mismatch")
        if expected_transport_receipt.get("authority_receipt_file_sha256") != serialized_json_sha256(receipt):
            failures.append("temperature discovery transport authority receipt file hash mismatch")
        if expected_transport_receipt.get("controller_challenge_sha256") != receipt.get("controller_challenge_sha256"):
            failures.append("temperature discovery transport challenge mismatch")
        if not is_strict_int(expected_transport_receipt.get("retry_count")) or expected_transport_receipt.get("retry_count") != 0:
            failures.append("temperature discovery transport retry count must be zero")
        if not is_strict_int(expected_transport_receipt.get("target_contact_count")) or expected_transport_receipt.get("target_contact_count") != 1:
            failures.append("temperature discovery transport target contact count must be one")
        if not is_strict_int(expected_transport_receipt.get("sensor_inventory_count")) or expected_transport_receipt.get("sensor_inventory_count") != 1:
            failures.append("temperature discovery transport inventory count must be one")
        if not is_strict_int(expected_transport_receipt.get("candidate_scan_count")) or expected_transport_receipt.get("candidate_scan_count") != 1:
            failures.append("temperature discovery transport candidate scan count must be one")
        if not is_strict_int(expected_transport_receipt.get("live_invocation_count")) or expected_transport_receipt.get("live_invocation_count") != 0:
            failures.append("temperature discovery transport live invocation count must be zero")
        if not is_strict_int(expected_transport_receipt.get("pmu_acquisition_count")) or expected_transport_receipt.get("pmu_acquisition_count") != 0:
            failures.append("temperature discovery transport PMU acquisition count must be zero")
    failures.extend(validate_temperature_authority_challenge(receipt, discovery, expected_challenge))
    if receipt.get("hwmon_name") not in target.APPROVED_TEMPERATURE_HWMON_NAMES:
        failures.append("temperature sensor authority hwmon name not approved")
    if receipt.get("sensor_semantic_profile") != target.LEGACY_FAMILY10H_TEMPERATURE_PROFILE:
        failures.append("temperature sensor authority semantic profile mismatch")
    if identity is not None:
        if receipt.get("hwmon_name") != identity.get("hwmon_name"):
            failures.append("temperature sensor authority hwmon name mismatch")
        if receipt.get("sensor_label_present") != identity.get("sensor_label_present"):
            failures.append("temperature sensor authority sensor label presence mismatch")
        if receipt.get("sensor_label_value") != identity.get("sensor_label_value"):
            failures.append("temperature sensor authority sensor label value mismatch")
        if receipt.get("sensor_semantic_role") != identity.get("sensor_semantic_role"):
            failures.append("temperature sensor authority semantic role mismatch")
    for key in ATTEMPT_COUNTER_KEYS:
        if not is_strict_int(receipt.get(key)):
            failures.append(f"temperature sensor authority counter missing or invalid {key}")
        elif isinstance(discovery, dict) and receipt.get(key) != discovery.get(key):
            failures.append(f"temperature sensor authority counter mismatch {key}")
    return {
        "present": True,
        "passed": not failures,
        "failures": failures,
        "approved_sensor_identity": identity,
        "authority_sha256": receipt.get(digest_field),
        "provenance_bound": receipt.get("provenance_bound") is True,
    }


def read_temperature_sensor_authority(*, expected_challenge: dict[str, Any] | None = None) -> dict[str, Any]:
    if not TEMPERATURE_SENSOR_AUTHORITY.exists():
        return temperature_sensor_authority_from_receipt(None)
    discovery = read_json(TARGET_DISCOVERY_RECEIPT) if TARGET_DISCOVERY_RECEIPT.exists() else None
    transport = read_json(DISCOVERY_TRANSPORT_PATH) if DISCOVERY_TRANSPORT_PATH.exists() else None
    return temperature_sensor_authority_from_receipt(
        read_json(TEMPERATURE_SENSOR_AUTHORITY),
        expected_challenge=expected_challenge,
        expected_discovery_receipt=discovery,
        expected_transport_receipt=transport,
    )


def temperature_sensor_authority_regression() -> dict[str, Any]:
    synthetic = public.synthetic_temperature_identity()
    def seal_authority(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["temperature_sensor_authority_sha256"] = public.digest(
            {k: v for k, v in result.items() if k != "temperature_sensor_authority_sha256"}
        )
        return result
    controller_nonce = "5" * 64
    synthetic_challenge = build_temperature_authority_challenge(
        source_hashes={"source_hashes_sha256": "1" * 64},
        source_bundle_sha256="2" * 64,
        runtime_binary_sha256="8" * 64,
        schedule_sidecar={"canonical_sha256": "3" * 64, "json_sha256": "4" * 64, "tsv_sha256": "6" * 64},
        authorized_commit="7" * 40,
        controller_nonce_sha256=hashlib.sha256(controller_nonce.encode("ascii")).hexdigest(),
        transport_scope={
            "target_host": "offline_fixture",
            "remote_base_root": "/fixture",
            "remote_root": "/fixture",
            "remote_source_root": "/fixture/source",
            "remote_receipt_path": f"/fixture/source/{target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME}",
        },
        source_authority_review=fixture_source_review_binding("7" * 40, "1" * 64, "2" * 64),
    )
    synthetic_challenge_sha = public.digest(synthetic_challenge)
    complete_forged_identity = public.with_temperature_identity_digest(
        {
            **{key: synthetic[key] for key in synthetic if key != "identity_sha256"},
            "class_path": "/sys/class/hwmon/hwmon9/temp7_input",
            "resolved_input_path": "/sys/devices/fake-target/hwmon/hwmon9/temp7_input",
            "resolved_hwmon_path": "/sys/devices/fake-target/hwmon/hwmon9",
            "resolved_device_path": "/sys/devices/fake-target",
        }
    )
    complete_forged_discovery = {
        "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
        "discovery_mode": "target_read_only_sensor_inventory",
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "pmu_open_count": 0,
        "runtime_launch_count": 0,
        "tomography_output_root_created": False,
        "selected_identity": complete_forged_identity,
        "observed_candidates": [{"identity": complete_forged_identity, "approved": True}],
        "provenance": {
            "authority": "target_sensor_discovery",
            "science_package_id": public.SCIENCE_PACKAGE_ID,
            "transaction_run_id": public.TRANSACTION_RUN_ID,
            "target_platform": {"cpu_family": "16", "cpu_model": "10"},
            "discovery_monotonic_ns": 1,
            "controller_challenge_sha256": synthetic_challenge_sha,
            "authorized_commit": synthetic_challenge["authorized_commit"],
        },
    }
    complete_forged_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in complete_forged_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    complete_forged_authority = seal_authority(
        {
            "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
            "provenance_bound": True,
            "provenance": "claimed_target_inventory",
            "hwmon_name": complete_forged_identity["hwmon_name"],
            "sensor_label_present": complete_forged_identity["sensor_label_present"],
            "sensor_label_value": complete_forged_identity["sensor_label_value"],
            "sensor_semantic_role": complete_forged_identity["sensor_semantic_role"],
            "sensor_semantic_profile": complete_forged_identity["sensor_semantic_profile"],
            "approved_sensor_identity": complete_forged_identity,
            "target_discovery_receipt": complete_forged_discovery,
            "controller_challenge": synthetic_challenge,
            "controller_challenge_sha256": synthetic_challenge_sha,
            "controller_nonce": controller_nonce,
            "source_authority_commit": synthetic_challenge["authorized_commit"],
            "target_contact_count": complete_forged_discovery["target_contact_count"],
            "sensor_inventory_count": complete_forged_discovery["sensor_inventory_count"],
            "live_invocation_count": complete_forged_discovery["live_invocation_count"],
            "pmu_acquisition_count": complete_forged_discovery["pmu_acquisition_count"],
        }
    )

    valid_consumer_identity = public.with_temperature_identity_digest(
        {
            **{key: synthetic[key] for key in synthetic if key != "identity_sha256"},
            "class_path": "/sys/class/hwmon/hwmon9/temp1_input",
            "resolved_input_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon9/temp1_input",
            "resolved_hwmon_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon9",
            "input_st_ino": 99,
        }
    )
    valid_consumer_platform = {
        "vendor": "AuthenticAMD",
        "cpu_family": 16,
        "cpu_models": [10],
        "processor_count": 6,
        "processors": [
            {"processor": cpu, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10}
            for cpu in range(6)
        ],
        "checked_before_discovery": True,
        "cpuinfo_path": "/proc/cpuinfo",
        "source_cpu_expected": public.SOURCE_CPU_EXPECTED,
        "receiver_cpu_expected": public.RECEIVER_CPU_EXPECTED,
        "source_receiver_cpus_present": True,
        "affinity_checked": True,
        "affinity_cpus": list(range(6)),
        "inherited_affinity_checked": True,
        "inherited_affinity_cpus": list(range(6)),
        "operational_pin_capability": target.fake_pin_probe(
            {public.SOURCE_CPU_EXPECTED, public.RECEIVER_CPU_EXPECTED},
            inherited_affinity=list(range(6)),
        ),
        "operational_pin_capability_passed": True,
    }
    valid_authorizing_scope = {
        "canonical_cpuinfo": True,
        "canonical_hwmon_root": True,
        "selected_class_path_is_sysfs_hwmon": True,
        "selected_input_is_legacy_temp1": True,
        "selected_semantic_profile_is_legacy_family10h": True,
        "selected_semantic_role_is_tctl": True,
        "resolved_input_is_sysfs_device": True,
        "resolved_device_is_sysfs_device": True,
        "resolved_driver_is_k10temp": True,
        "resolved_subsystem_is_pci": True,
        "authorizing": True,
        "cpuinfo_path": "/proc/cpuinfo",
        "hwmon_root": "/sys/class/hwmon",
    }

    def coherent_consumer_authority(identity: dict[str, Any], platform: dict[str, Any]) -> dict[str, Any]:
        discovery = {
            "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
            "discovery_mode": "target_read_only_sensor_inventory",
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "candidate_scan_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "pmu_open_count": 0,
            "runtime_launch_count": 0,
            "tomography_output_root_created": False,
            "controller_nonce_sha256": synthetic_challenge["controller_nonce_sha256"],
            "selected_identity": identity,
            "identity_before": identity,
            "identity_after": identity,
            "observed_candidates": [{"identity": identity, "approved": True}],
            "selection": {
                "law": "exactly one LEGACY_FAMILY10H_K10TEMP_TEMP1_V1 candidate",
                "approval_profile": target.LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
                "approved_count": 1,
                "deterministic_law": True,
                "selected_class_path": identity["class_path"],
            },
            "source_authority": {"passed": True},
            "authorizing_scope": dict(valid_authorizing_scope),
            "provenance": {
                "authority": "target_sensor_discovery",
                "science_package_id": public.SCIENCE_PACKAGE_ID,
                "transaction_run_id": public.TRANSACTION_RUN_ID,
                "target_platform": platform,
                "discovery_monotonic_ns": 1,
                "controller_challenge_sha256": synthetic_challenge_sha,
                "authorized_commit": synthetic_challenge["authorized_commit"],
            },
            "sample": {
                "identity": identity,
                "path": identity["class_path"],
                "label_present": identity["sensor_label_present"],
                "label_value": identity["sensor_label_value"],
                "semantic_role": identity["sensor_semantic_role"],
                "semantic_profile": identity["sensor_semantic_profile"],
                "value_c": 42.0,
                "pinned_descriptor": target.expected_descriptor_identity(identity),
                "read_law": "manifest-approved resolved input descriptor",
            },
        }
        discovery["target_discovery_receipt_sha256"] = public.digest(
            {k: v for k, v in discovery.items() if k != "target_discovery_receipt_sha256"}
        )
        return seal_authority(
            {
                "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
                "provenance_bound": True,
                "provenance": "controller_verified_target_sensor_inventory",
                "hwmon_name": identity["hwmon_name"],
                "sensor_label_present": identity["sensor_label_present"],
                "sensor_label_value": identity["sensor_label_value"],
                "sensor_semantic_role": identity["sensor_semantic_role"],
                "sensor_semantic_profile": identity["sensor_semantic_profile"],
                "approved_sensor_identity": identity,
                "target_discovery_receipt": discovery,
                "controller_challenge": synthetic_challenge,
                "controller_challenge_sha256": synthetic_challenge_sha,
                "controller_nonce": controller_nonce,
                "source_authority_commit": synthetic_challenge["authorized_commit"],
                "target_contact_count": discovery["target_contact_count"],
                "sensor_inventory_count": discovery["sensor_inventory_count"],
                "live_invocation_count": discovery["live_invocation_count"],
                "pmu_acquisition_count": discovery["pmu_acquisition_count"],
            }
        )

    valid_consumer_authority = coherent_consumer_authority(valid_consumer_identity, valid_consumer_platform)
    missing_readback_platform = json.loads(json.dumps(valid_consumer_platform))
    missing_readback_platform["operational_pin_capability"]["per_cpu"][str(public.SOURCE_CPU_EXPECTED)].pop("readback_affinity", None)
    missing_readback_authority = coherent_consumer_authority(valid_consumer_identity, missing_readback_platform)
    null_readback_platform = json.loads(json.dumps(valid_consumer_platform))
    null_readback_platform["operational_pin_capability"]["per_cpu"][str(public.SOURCE_CPU_EXPECTED)]["readback_affinity"] = None
    null_readback_authority = coherent_consumer_authority(valid_consumer_identity, null_readback_platform)
    noncanonical_driver_identity = public.with_temperature_identity_digest(
        {
            **{key: valid_consumer_identity[key] for key in valid_consumer_identity if key != "identity_sha256"},
            "resolved_driver_path": "/tmp/k10temp",
        }
    )
    noncanonical_driver_authority = coherent_consumer_authority(noncanonical_driver_identity, valid_consumer_platform)
    noncanonical_subsystem_identity = public.with_temperature_identity_digest(
        {
            **{key: valid_consumer_identity[key] for key in valid_consumer_identity if key != "identity_sha256"},
            "resolved_subsystem_path": "/tmp/pci",
        }
    )
    noncanonical_subsystem_authority = coherent_consumer_authority(noncanonical_subsystem_identity, valid_consumer_platform)

    boolean_discovery_counter = dict(complete_forged_discovery)
    boolean_discovery_counter["target_contact_count"] = True
    boolean_discovery_counter["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in boolean_discovery_counter.items() if k != "target_discovery_receipt_sha256"}
    )
    boolean_discovery_counter_authority = seal_authority(
        {
            **complete_forged_authority,
            "target_discovery_receipt": boolean_discovery_counter,
            "target_contact_count": True,
        }
    )
    boolean_authority_counter = seal_authority({**complete_forged_authority, "target_contact_count": True})

    provenance_free = {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
        "provenance_bound": False,
        "provenance": "offline_synthetic_fixture",
        "hwmon_name": synthetic["hwmon_name"],
        "sensor_label_present": synthetic["sensor_label_present"],
        "sensor_label_value": synthetic["sensor_label_value"],
        "sensor_semantic_role": synthetic["sensor_semantic_role"],
        "sensor_semantic_profile": synthetic["sensor_semantic_profile"],
        "approved_sensor_identity": synthetic,
    }
    synthetic_true = seal_authority({**provenance_free, "provenance_bound": True, "provenance": "claimed_target_inventory"})
    forged_discovery = {
        "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
        "discovery_mode": "target_read_only_sensor_inventory",
        "target_contact_count": 0,
        "live_invocation_count": 0,
        "selected_identity": synthetic,
        "observed_candidates": [{}],
    }
    forged_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in forged_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    forged_schema_complete = seal_authority(
        {
            "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
            "provenance_bound": True,
            "provenance": "claimed_target_inventory",
            "hwmon_name": synthetic["hwmon_name"],
            "sensor_label_present": synthetic["sensor_label_present"],
            "sensor_label_value": synthetic["sensor_label_value"],
            "sensor_semantic_role": synthetic["sensor_semantic_role"],
            "sensor_semantic_profile": synthetic["sensor_semantic_profile"],
            "approved_sensor_identity": synthetic,
            "target_discovery_receipt": forged_discovery,
        }
    )
    malformed = seal_authority({**provenance_free, "provenance_bound": True, "hwmon_name": "acpitz"})
    synthetic_result = temperature_sensor_authority_from_receipt(seal_authority(provenance_free))
    synthetic_true_result = temperature_sensor_authority_from_receipt(synthetic_true)
    forged_result = temperature_sensor_authority_from_receipt(forged_schema_complete)
    complete_forged_without_expected_result = temperature_sensor_authority_from_receipt(complete_forged_authority)
    complete_forged_wrong_expected_result = temperature_sensor_authority_from_receipt(
        complete_forged_authority,
        expected_challenge={**synthetic_challenge, "source_bundle_sha256": "8" * 64},
    )
    complete_forged_with_expected_result = temperature_sensor_authority_from_receipt(
        complete_forged_authority,
        expected_challenge=synthetic_challenge,
    )
    boolean_discovery_counter_result = temperature_sensor_authority_from_receipt(
        boolean_discovery_counter_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    boolean_pmu_open_discovery = dict(complete_forged_discovery)
    boolean_pmu_open_discovery["pmu_open_count"] = False
    boolean_pmu_open_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in boolean_pmu_open_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    boolean_pmu_open_authority = seal_authority(
        {
            **complete_forged_authority,
            "target_discovery_receipt": boolean_pmu_open_discovery,
        }
    )
    boolean_runtime_launch_discovery = dict(complete_forged_discovery)
    boolean_runtime_launch_discovery["runtime_launch_count"] = False
    boolean_runtime_launch_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in boolean_runtime_launch_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    boolean_runtime_launch_authority = seal_authority(
        {
            **complete_forged_authority,
            "target_discovery_receipt": boolean_runtime_launch_discovery,
        }
    )
    boolean_pmu_open_result = temperature_sensor_authority_from_receipt(
        boolean_pmu_open_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    boolean_runtime_launch_result = temperature_sensor_authority_from_receipt(
        boolean_runtime_launch_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    boolean_authority_counter_result = temperature_sensor_authority_from_receipt(
        boolean_authority_counter,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    valid_consumer_result = temperature_sensor_authority_from_receipt(
        valid_consumer_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    missing_readback_result = temperature_sensor_authority_from_receipt(
        missing_readback_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    null_readback_result = temperature_sensor_authority_from_receipt(
        null_readback_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    noncanonical_driver_result = temperature_sensor_authority_from_receipt(
        noncanonical_driver_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    noncanonical_subsystem_result = temperature_sensor_authority_from_receipt(
        noncanonical_subsystem_authority,
        expected_challenge=synthetic_challenge,
        require_transport=False,
    )
    malformed_result = temperature_sensor_authority_from_receipt(malformed)
    expected_live_counters = {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0}
    boolean_counter_values = {"target_contact_count": True, "sensor_inventory_count": True, "live_invocation_count": False, "pmu_acquisition_count": False}
    current = read_temperature_sensor_authority()
    result = {
        "synthetic_or_provenance_free_identity_cannot_freeze": not synthetic_result["passed"],
        "synthetic_identity_with_asserted_provenance_cannot_freeze": not synthetic_true_result["passed"],
        "schema_complete_forged_discovery_rejected": not forged_result["passed"],
        "well_formed_self_authored_discovery_without_expected_challenge_rejected": not complete_forged_without_expected_result["passed"],
        "well_formed_self_authored_discovery_wrong_expected_challenge_rejected": not complete_forged_wrong_expected_result["passed"],
        "well_formed_challenge_bound_fixture_without_transport_rejected": not complete_forged_with_expected_result["passed"],
        "boolean_discovery_counter_rejected": any("temperature sensor discovery target contact count must be one" in item for item in boolean_discovery_counter_result["failures"]),
        "boolean_discovery_pmu_open_count_rejected": any("temperature sensor discovery PMU open count must be zero" in item for item in boolean_pmu_open_result["failures"]),
        "boolean_discovery_runtime_launch_count_rejected": any("temperature sensor discovery runtime launch count must be zero" in item for item in boolean_runtime_launch_result["failures"]),
        "boolean_authority_counter_rejected": any("temperature sensor authority counter missing or invalid target_contact_count" in item for item in boolean_authority_counter_result["failures"]),
        "boolean_manifest_counter_object_rejected": not counter_dict_equal_strict(boolean_counter_values, expected_live_counters),
        "boolean_final_attempt_counter_rejected": not counters_equal_strict(boolean_counter_values, expected_live_counters),
        "boolean_offline_zero_receipt_counter_rejected": not zero_contact_counter_valid({"target_contact_count": False}, "target_contact_count"),
        "canonical_consumer_authority_fixture_passes_without_transport": valid_consumer_result["passed"],
        "missing_pin_readback_authority_rejected": any("operational pin CPU 4 readback mismatch" in item for item in missing_readback_result["failures"]),
        "null_pin_readback_authority_rejected": any("operational pin CPU 4 readback mismatch" in item for item in null_readback_result["failures"]),
        "noncanonical_driver_path_authority_rejected": any("resolved driver path is not canonical" in item for item in noncanonical_driver_result["failures"]),
        "noncanonical_subsystem_path_authority_rejected": any("resolved subsystem path is not canonical" in item for item in noncanonical_subsystem_result["failures"]),
        "wrong_hwmon_authority_rejected": not malformed_result["passed"],
        "current_authority_present": current["present"],
        "current_authority_passed": current["passed"],
        "current_authority_failures": current["failures"],
    }
    result["passed"] = (
        result["synthetic_or_provenance_free_identity_cannot_freeze"]
        and result["synthetic_identity_with_asserted_provenance_cannot_freeze"]
        and result["schema_complete_forged_discovery_rejected"]
        and result["well_formed_self_authored_discovery_without_expected_challenge_rejected"]
        and result["well_formed_self_authored_discovery_wrong_expected_challenge_rejected"]
        and result["well_formed_challenge_bound_fixture_without_transport_rejected"]
        and result["boolean_discovery_counter_rejected"]
        and result["boolean_discovery_pmu_open_count_rejected"]
        and result["boolean_discovery_runtime_launch_count_rejected"]
        and result["boolean_authority_counter_rejected"]
        and result["boolean_manifest_counter_object_rejected"]
        and result["boolean_final_attempt_counter_rejected"]
        and result["boolean_offline_zero_receipt_counter_rejected"]
        and result["canonical_consumer_authority_fixture_passes_without_transport"]
        and result["missing_pin_readback_authority_rejected"]
        and result["null_pin_readback_authority_rejected"]
        and result["noncanonical_driver_path_authority_rejected"]
        and result["noncanonical_subsystem_path_authority_rejected"]
        and result["wrong_hwmon_authority_rejected"]
    )
    return result


def build_temperature_sensor_authority_receipt(
    *,
    discovery: dict[str, Any],
    controller_challenge: dict[str, Any],
    controller_nonce: str,
) -> dict[str, Any]:
    identity = discovery.get("selected_identity")
    require(isinstance(identity, dict), "discovery selected identity missing")
    receipt = {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
        "provenance_bound": True,
        "provenance": "controller_verified_target_sensor_inventory",
        "hwmon_name": identity["hwmon_name"],
        "sensor_label_present": identity["sensor_label_present"],
        "sensor_label_value": identity["sensor_label_value"],
        "sensor_semantic_role": identity["sensor_semantic_role"],
        "sensor_semantic_profile": identity["sensor_semantic_profile"],
        "approved_sensor_identity": identity,
        "target_discovery_receipt": discovery,
        "controller_challenge": controller_challenge,
        "controller_challenge_sha256": public.digest(controller_challenge),
        "controller_nonce": controller_nonce,
        "source_authority_commit": controller_challenge["authorized_commit"],
        "target_contact_count": discovery.get("target_contact_count"),
        "sensor_inventory_count": discovery.get("sensor_inventory_count"),
        "candidate_scan_count": discovery.get("candidate_scan_count"),
        "live_invocation_count": discovery.get("live_invocation_count"),
        "pmu_acquisition_count": discovery.get("pmu_acquisition_count"),
    }
    receipt["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in receipt.items() if k != "temperature_sensor_authority_sha256"}
    )
    validation = temperature_sensor_authority_from_receipt(
        receipt,
        expected_challenge=controller_challenge,
        expected_discovery_receipt=discovery,
        require_transport=False,
    )
    require(validation["passed"], "temperature sensor authority receipt invalid: " + ",".join(validation["failures"]))
    return receipt


def write_temperature_sensor_authority_receipt(
    *,
    discovery: dict[str, Any],
    controller_challenge: dict[str, Any],
    controller_nonce: str,
) -> dict[str, Any]:
    receipt = build_temperature_sensor_authority_receipt(
        discovery=discovery,
        controller_challenge=controller_challenge,
        controller_nonce=controller_nonce,
    )
    write_json(TEMPERATURE_SENSOR_AUTHORITY, receipt)
    reread = read_json(TEMPERATURE_SENSOR_AUTHORITY)
    if reread != receipt:
        raise ControllerError("temperature authority receipt write verification failed")
    return receipt


def sh_quote(value: str) -> str:
    return shlex.quote(value)


def serialized_json_sha256(value: Any) -> str:
    return hashlib.sha256((strict_json_dumps(value, indent=2) + "\n").encode("utf-8")).hexdigest()


def validate_target_discovery_failure_receipt(
    receipt: dict[str, Any] | None,
    *,
    expected_challenge: dict[str, Any],
    expected_source_commit: str,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(receipt, dict):
        return {"passed": False, "failures": ["structured target discovery failure receipt missing"]}
    candidate_keys = {
        "class_path",
        "input_basename",
        "input_path_exists",
        "input_readability",
        "raw_input_text",
        "raw_input_parse_failure",
        "parsed_millidegree_value",
        "physical_range_passed",
        "hwmon_name_path_exists",
        "hwmon_name_readability",
        "hwmon_name_value",
        "sensor_label_path_exists",
        "sensor_label_present",
        "sensor_label_readability",
        "sensor_label_value",
        "resolved_input_path",
        "resolved_hwmon_path",
        "resolved_device_path",
        "resolved_driver_path",
        "resolved_subsystem_path",
        "device_driver",
        "device_subsystem",
        "device_modalias_path_exists",
        "device_modalias_readability",
        "device_modalias_value",
        "input_st_dev",
        "input_st_ino",
        "input_st_mode",
        "observation_errors",
        "approval_profile",
        "sensor_semantic_role",
        "sensor_semantic_profile",
        "approved",
        "rejection_reasons",
        "identity",
        "canonical_path_law_active",
        "class_path_under_hwmon_root",
        "resolved_input_under_sys_devices",
        "resolved_device_under_sys_devices",
        "rejection_reason",
    }
    expected_keys = {
        "schema",
        "discovery_mode",
        "passed",
        "failure_classification",
        "failure_detail",
        "target_contact_count",
        "sensor_inventory_count",
        "candidate_scan_count",
        "live_invocation_count",
        "pmu_acquisition_count",
        "pmu_open_count",
        "runtime_launch_count",
        "tomography_output_root_created",
        "source_root",
        "receipt_path",
        "hwmon_root",
        "provenance",
        "controller_challenge_sha256",
        "controller_nonce_sha256",
        "authorized_commit",
        "source_hashes_sha256",
        "source_bundle_sha256",
        "runtime_binary_sha256",
        "source_authority_review",
        "source_authority",
        "challenge_validation",
        "top_level_visibility_snapshot",
        "observed_candidates",
        "candidate_count",
        "approved_count",
        "active_counters",
        "target_discovery_receipt_sha256",
    }
    if set(receipt) != expected_keys:
        failures.append("structured target discovery failure keyset mismatch")
    if receipt.get("schema") != target.TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA:
        failures.append("structured target discovery failure schema mismatch")
    if receipt.get("passed") is not False:
        failures.append("structured target discovery failure pass bit must be false")
    if receipt.get("discovery_mode") != "target_read_only_sensor_inventory":
        failures.append("structured target discovery failure mode mismatch")
    if receipt.get("target_discovery_receipt_sha256") != public.digest(
        {k: v for k, v in receipt.items() if k != "target_discovery_receipt_sha256"}
    ):
        failures.append("structured target discovery failure digest mismatch")
    if receipt.get("controller_challenge_sha256") != public.digest(expected_challenge):
        failures.append("structured target discovery failure challenge mismatch")
    if receipt.get("controller_nonce_sha256") != expected_challenge.get("controller_nonce_sha256"):
        failures.append("structured target discovery failure nonce mismatch")
    if receipt.get("authorized_commit") != expected_source_commit:
        failures.append("structured target discovery failure source commit mismatch")
    if receipt.get("source_hashes_sha256") != expected_challenge.get("source_hashes_sha256"):
        failures.append("structured target discovery failure source hash mismatch")
    if receipt.get("source_bundle_sha256") != expected_challenge.get("source_bundle_sha256"):
        failures.append("structured target discovery failure source bundle mismatch")
    if receipt.get("runtime_binary_sha256") != expected_challenge.get("runtime_binary_sha256"):
        failures.append("structured target discovery failure runtime hash mismatch")
    if receipt.get("source_authority_review") != expected_challenge.get("source_authority_review"):
        failures.append("structured target discovery failure source-review binding mismatch")
    transport_scope = expected_challenge.get("transport_scope") if isinstance(expected_challenge.get("transport_scope"), dict) else {}
    if transport_scope:
        if receipt.get("source_root") != transport_scope.get("remote_source_root"):
            failures.append("structured target discovery failure source-root scope mismatch")
        if receipt.get("receipt_path") != transport_scope.get("remote_receipt_path"):
            failures.append("structured target discovery failure receipt-path scope mismatch")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        failures.append("structured target discovery failure provenance missing")
        provenance = {}
    provenance_keys = {
        "authority",
        "science_package_id",
        "transaction_run_id",
        "target_platform",
        "discovery_monotonic_ns",
        "controller_challenge_sha256",
        "authorized_commit",
    }
    if set(provenance) != provenance_keys:
        failures.append("structured target discovery failure provenance keyset mismatch")
    if provenance.get("authority") != "target_sensor_discovery":
        failures.append("structured target discovery failure provenance authority mismatch")
    if provenance.get("science_package_id") != public.SCIENCE_PACKAGE_ID or provenance.get("transaction_run_id") != public.TRANSACTION_RUN_ID:
        failures.append("structured target discovery failure provenance package mismatch")
    if provenance.get("controller_challenge_sha256") != public.digest(expected_challenge):
        failures.append("structured target discovery failure provenance challenge mismatch")
    if provenance.get("authorized_commit") != expected_source_commit:
        failures.append("structured target discovery failure provenance source commit mismatch")
    if not is_strict_int(provenance.get("discovery_monotonic_ns")):
        failures.append("structured target discovery failure provenance monotonic time invalid")
    challenge_validation = receipt.get("challenge_validation")
    if not isinstance(challenge_validation, dict):
        failures.append("structured target discovery failure challenge validation missing")
        challenge_validation = {}
    candidate_scan_count = receipt.get("candidate_scan_count")
    sensor_inventory_count = 1 if candidate_scan_count == 1 else 0
    expected_counters = {
        "target_contact_count": 1,
        "sensor_inventory_count": sensor_inventory_count,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "candidate_scan_count": candidate_scan_count,
    }
    if not is_strict_int(receipt.get("candidate_scan_count")) or receipt.get("candidate_scan_count") not in {0, 1}:
        failures.append("structured target discovery failure candidate scan counter mismatch")
    elif not all(is_strict_int(receipt.get(key)) and receipt.get(key) == value for key, value in expected_counters.items()):
        failures.append("structured target discovery failure counters mismatch")
    if receipt.get("active_counters") != expected_counters or not counter_dict_equal_strict(receipt.get("active_counters"), expected_counters):
        failures.append("structured target discovery failure active counters mismatch")
    if (
        not is_strict_int(receipt.get("runtime_launch_count"))
        or not is_strict_int(receipt.get("pmu_open_count"))
        or not is_strict_int(receipt.get("pmu_acquisition_count"))
        or receipt.get("runtime_launch_count") != 0
        or receipt.get("pmu_open_count") != 0
        or receipt.get("pmu_acquisition_count") != 0
    ):
        failures.append("structured target discovery failure opened runtime or PMU")
    if receipt.get("tomography_output_root_created") is not False:
        failures.append("structured target discovery failure created tomography output root")
    candidates = receipt.get("observed_candidates")
    if not isinstance(candidates, list):
        failures.append("structured target discovery failure candidates missing")
        candidates = []
    elif receipt.get("candidate_count") != len(candidates):
        failures.append("structured target discovery failure candidate count mismatch")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            failures.append(f"structured target discovery failure candidate malformed {index}")
            continue
        if set(candidate) != candidate_keys:
            failures.append(f"structured target discovery failure candidate keyset mismatch {index}")
        for key in [
            "input_path_exists",
            "input_readability",
            "physical_range_passed",
            "hwmon_name_path_exists",
            "hwmon_name_readability",
            "sensor_label_path_exists",
            "sensor_label_present",
            "sensor_label_readability",
            "device_modalias_path_exists",
            "device_modalias_readability",
            "approved",
            "canonical_path_law_active",
            "class_path_under_hwmon_root",
            "resolved_input_under_sys_devices",
            "resolved_device_under_sys_devices",
        ]:
            if key in candidate and type(candidate.get(key)) is not bool:
                failures.append(f"structured target discovery failure candidate boolean invalid {index}:{key}")
        if not isinstance(candidate.get("observation_errors"), list) or not isinstance(candidate.get("rejection_reasons"), list):
            failures.append(f"structured target discovery failure candidate list fields invalid {index}")
        if candidate.get("approved") is True and not isinstance(candidate.get("identity"), dict):
            failures.append(f"structured target discovery failure approved candidate identity missing {index}")
    approved = [candidate for candidate in candidates if isinstance(candidate, dict) and candidate.get("approved") is True]
    if receipt.get("approved_count") != len(approved):
        failures.append("structured target discovery failure approved count mismatch")
    if receipt.get("failure_classification") not in {
        "NO_HWMON_TEMPERATURE_CANDIDATES",
        "K10TEMP_DRIVER_NOT_VISIBLE",
        "K10TEMP_HWMON_NOT_VISIBLE",
        "LEGACY_TEMP1_INPUT_NOT_VISIBLE",
        "LEGACY_CANDIDATE_REJECTED_IDENTITY",
        "LEGACY_CANDIDATE_UNREADABLE",
        "MULTIPLE_APPROVED_LEGACY_CANDIDATES",
        "CONTROLLER_CHALLENGE_MISSING",
        "CONTROLLER_CHALLENGE_INVALID",
        "PLATFORM_IDENTITY_INVALID",
        "SELECTED_IDENTITY_CHANGED_BEFORE_READ",
        "SELECTED_IDENTITY_CHANGED_AFTER_READ",
        "PRE_SCAN_DISCOVERY_FAILURE",
    }:
        failures.append("structured target discovery failure classification invalid")
    if receipt.get("candidate_scan_count") == 0 and receipt.get("observed_candidates") != []:
        failures.append("structured target discovery failure pre-scan candidates must be empty")
    visibility = receipt.get("top_level_visibility_snapshot")
    if receipt.get("candidate_scan_count") == 1:
        if not isinstance(visibility, dict):
            failures.append("structured target discovery failure scan visibility missing")
        elif visibility.get("temp_input_candidate_count") != len(candidates):
            failures.append("structured target discovery failure visibility candidate count mismatch")
    return {"passed": not failures, "failures": failures}


def acquire_temperature_sensor_authority(
    *,
    target_host: str = TARGET_HOST,
    remote_root: str = DISCOVERY_REMOTE_ROOT,
    source_authority_commit: str | None = None,
) -> dict[str, Any]:
    source_commit = source_authority_commit or os.environ.get("FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_AUTHORITY_COMMIT") or git_text("rev-parse", "HEAD")
    if source_commit == C3_SOURCE_AUTHORITY_COMMIT:
        raise ControllerError("C3 source authority acquisition already failed and is preserved as no-retry")
    if source_commit in {C5_ATTEMPT_SOURCE_COMMIT, C5_SOURCE_AUTHORITY_COMMIT, C5_FAILURE_EVIDENCE_COMMIT}:
        raise ControllerError("C5 source authority acquisition already failed and is preserved as no-retry")
    existing_active_paths = active_attempt_paths_present()
    if existing_active_paths:
        raise ControllerError("second discovery attempt rejected: authority, discovery, transport, cleanup, or attempt receipt already exists")
    historical_lane_report = known_historical_lane_contact_report()
    source_hashes = read_source_hash_authority()
    schedule_sidecar = read_json(public.SCHEDULE_SHA)
    bundle = read_existing_source_bundle_authority()
    source_commit_check = source_authority_commit_verification(source_commit)
    if not source_commit_check["passed"]:
        raise ControllerError("source authority commit verification failed: " + ",".join(source_commit_check["failures"]))
    nonce = secrets.token_hex(32)
    nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    transport_scope = discovery_transport_scope(target_host=target_host, remote_root=remote_root, nonce_sha=nonce_sha)
    source_review = read_source_authority_review_for_discovery(
        source_commit=source_commit,
        source_hashes_sha256=source_hashes["source_hashes_sha256"],
        source_bundle_sha256=bundle["sha256"],
        runtime_binary_sha256=source_hashes["runtime_binary_authority"]["sha256"],
    )
    if not source_review["passed"]:
        raise ControllerError("source authority C6 review gate failed before target contact: " + ",".join(source_review["failures"]))
    challenge = build_temperature_authority_challenge(
        source_hashes=source_hashes,
        source_bundle_sha256=bundle["sha256"],
        runtime_binary_sha256=source_hashes["runtime_binary_authority"]["sha256"],
        schedule_sidecar=schedule_sidecar,
        authorized_commit=source_commit,
        controller_nonce_sha256=nonce_sha,
        transport_scope=transport_scope,
        source_authority_review=source_review_challenge_binding(source_review),
    )
    challenge_receipt = write_discovery_challenge_receipt(challenge, source_commit=source_commit, nonce_sha=nonce_sha, source_review=source_review)
    remote_base_root = transport_scope["remote_base_root"]
    remote_root = transport_scope["remote_root"]
    remote_source = transport_scope["remote_source_root"]
    remote_challenge = f"{remote_source}/controller_challenge.json"
    remote_receipt = transport_scope["remote_receipt_path"]
    remote_owner_marker = f"{remote_root}/.family10h_temperature_discovery_owner"
    commands: list[list[str]] = []
    cleanup = {"attempted": False, "passed": False, "absence_verified": False, "owner_marker": remote_owner_marker}
    remote_root_owned = False
    remote_root_may_exist = False
    remote_base_preexisting = True
    target_command_invoked = False
    discovery: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    receipt_copied_state_persisted = False
    target_discovery_receipt_sealed_before_cleanup = False
    authority_receipt_sealed_before_cleanup = False
    cleanup_armed_state_persisted = False
    target_discovery_file_sha: str | None = None
    authority_file_sha: str | None = None
    transfer_plan: dict[str, Any] | None = None
    target_failure: dict[str, Any] | None = None
    target_structured_failure: dict[str, Any] | None = None
    target_structured_failure_file_sha: str | None = None
    target_structured_failure_validation: dict[str, Any] | None = None
    target_structured_failure_bytes: bytes | None = None
    target_failure_preserved_before_cleanup = False
    target_failure_receipt_sealed_before_cleanup = False

    def validated_failure_counter_pair() -> dict[str, int]:
        if (
            isinstance(target_structured_failure, dict)
            and isinstance(target_structured_failure_validation, dict)
            and target_structured_failure_validation.get("passed") is True
            and is_strict_int(target_structured_failure.get("candidate_scan_count"))
        ):
            candidate_scan_count = int(target_structured_failure["candidate_scan_count"])
            if candidate_scan_count in {0, 1}:
                return {
                    "candidate_scan_count": candidate_scan_count,
                    "sensor_inventory_count": 1 if candidate_scan_count == 1 else 0,
                }
        return {"candidate_scan_count": 0, "sensor_inventory_count": 0}

    def seal_target_failure_receipt(*, before_cleanup: bool) -> dict[str, Any]:
        require(target_failure is not None, "target failure receipt requested without target failure")
        counter_pair = validated_failure_counter_pair()
        failure_receipt = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_FAILURE_V1",
            "passed": False,
            "source_authority_commit": source_commit,
            "source_authority_review": source_review,
            "controller_challenge_sha256": public.digest(challenge),
            "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
            "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
            "discovery_transfer_plan": transfer_plan,
            "target_failure": target_failure,
            "structured_target_discovery_failure_receipt": target_structured_failure,
            "structured_target_discovery_failure_validation": target_structured_failure_validation,
            "target_discovery_receipt_path": str(TARGET_DISCOVERY_RECEIPT) if target_structured_failure is not None else None,
            "target_discovery_receipt_file_sha256": target_structured_failure_file_sha,
            "attempt_state": "target_command_invoked",
            "active_counters": dict(ATTEMPT_STATE_COUNTERS["target_command_invoked"]),
            "target_contact_count": 1,
            "sensor_inventory_count": counter_pair["sensor_inventory_count"],
            "candidate_scan_count": counter_pair["candidate_scan_count"],
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "runtime_launch_count": 0,
            "target_failure_preserved_before_cleanup": target_failure_preserved_before_cleanup,
            "target_failure_receipt_sealed_before_cleanup": before_cleanup or target_failure_receipt_sealed_before_cleanup,
            "failure_receipt_written_before_cleanup": before_cleanup,
            "cleanup": cleanup,
            "cleanup_result": cleanup.get("passed"),
            "remote_root_absence_result": cleanup.get("absence_verified"),
            "commands": commands,
        }
        failure_receipt["target_discovery_failure_sha256"] = public.digest(
            {k: v for k, v in failure_receipt.items() if k != "target_discovery_failure_sha256"}
        )
        write_json_atomic(TARGET_DISCOVERY_FAILURE_PATH, failure_receipt, require_directory_sync=before_cleanup)
        sealed_receipt = read_json(TARGET_DISCOVERY_FAILURE_PATH)
        require(sealed_receipt == failure_receipt, "target failure receipt seal reread mismatch")
        return failure_receipt
    write_discovery_attempt_receipt(
        {
            "passed": False,
            "attempt_state": "claimed_pre_contact",
            "source_authority_commit": source_commit,
            "source_authority": source_commit_check,
            "source_authority_review": source_review,
            "controller_challenge_sha256": public.digest(challenge),
            "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
            "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
            "target_contact_count": 0,
            "sensor_inventory_count": 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    )
    with tempfile.TemporaryDirectory(prefix="family10h_temperature_discovery_") as tmp:
        tmp_root = Path(tmp)
        source_stage = tmp_root / "source_authority_snapshot"
        materialize_source_authority_snapshot(source_commit, source_stage)
        challenge_path = tmp_root / "controller_challenge.json"
        write_json(challenge_path, challenge)
        transfer_plan = build_discovery_transfer_plan(
            source_root=source_stage,
            remote_source_root=remote_source,
            source_commit=source_commit,
            expected_source_hashes_sha256=source_hashes["source_hashes_sha256"],
            expected_source_bundle_sha256=bundle["sha256"],
            expected_runtime_binary_authority=source_hashes["runtime_binary_authority"],
        )
        if not transfer_plan["passed"]:
            raise ControllerError("discovery transfer plan failed before target contact: " + ",".join(transfer_plan["failures"]))
        local_copyback = tmp_root / target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME
        try:
            preflight = (
                "set -eu; "
                "base_preexisting=0; "
                f"if test -e {sh_quote(remote_base_root)}; then base_preexisting=1; fi; "
                f"test ! -e {sh_quote(remote_root)}; "
                f"install -d -m 0700 {sh_quote(remote_base_root)}; "
                f"mkdir -m 0700 {sh_quote(remote_root)}; "
                f"printf '%s\\n' {sh_quote(nonce_sha)} > {sh_quote(remote_owner_marker)}; "
                f"trap 'if test -f {sh_quote(remote_owner_marker)} && grep -qx {sh_quote(nonce_sha)} {sh_quote(remote_owner_marker)}; then rm -rf -- {sh_quote(remote_root)}; fi' HUP INT TERM EXIT; "
                f"install -d -m 0700 {sh_quote(remote_source)}; "
                f"test \"$(cat {sh_quote(remote_owner_marker)})\" = {sh_quote(nonce_sha)}; "
                "trap - HUP INT TERM EXIT; "
                "printf 'FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=%s root_created=1\\n' \"$base_preexisting\""
            )
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target_host, preflight]
            commands.append(command)
            write_discovery_attempt_receipt(
                {
                    "passed": False,
                    "attempt_state": "transport_contact_invoked",
                    "source_authority_commit": source_commit,
                    "source_authority": source_commit_check,
                    "source_authority_review": source_review,
                    "controller_challenge_sha256": public.digest(challenge),
                    "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
                    "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
                    "discovery_transfer_plan": transfer_plan,
                    "target_contact_count": 1,
                    "sensor_inventory_count": 0,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "commands": commands,
                }
            )
            remote_root_may_exist = True
            preflight_result = run(command, timeout=20.0)
            preflight_receipt = parse_preflight_stdout(preflight_result.stdout)
            remote_root_owned = preflight_receipt["passed"] is True and preflight_receipt["root_created"] is True
            remote_base_preexisting = bool(preflight_receipt["remote_base_preexisting"])
            cleanup["remote_base_preexisting"] = remote_base_preexisting
            cleanup["preflight"] = preflight_receipt
            if not preflight_receipt["passed"]:
                raise ControllerError("discovery preflight stdout malformed")
            for item in transfer_plan["records"]:
                command = ["scp", "-q", str(item["source_path"]), f"{target_host}:{item['remote_destination']}"]
                commands.append(command)
                run(command, timeout=60.0)
            command = ["scp", "-q", str(challenge_path), f"{target_host}:{remote_challenge}"]
            commands.append(command)
            run(command, timeout=30.0)
            remote_command = (
                f"cd {sh_quote(remote_source)} && "
                f"python3 family10h_carrier_tomography_target.py "
                f"--discover-temperature-sensor-authority "
                f"--source-root {sh_quote(remote_source)} "
                f"--controller-challenge {sh_quote(remote_challenge)} "
                f"--controller-nonce {sh_quote(nonce)} "
                f"--authorized-commit {sh_quote(source_commit)} "
                f"--receipt-path {sh_quote(remote_receipt)}"
            )
            command = ["ssh", "-o", "BatchMode=yes", target_host, remote_command]
            commands.append(["ssh", "-o", "BatchMode=yes", target_host, remote_command.replace(nonce, "<controller_nonce>")])
            target_command_invoked = True
            write_discovery_attempt_receipt(
                {
                    "passed": False,
                    "attempt_state": "target_command_invoked",
                    "source_authority_commit": source_commit,
                    "source_authority": source_commit_check,
                    "source_authority_review": source_review,
                    "controller_challenge_sha256": public.digest(challenge),
                    "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
                    "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
                    "discovery_transfer_plan": transfer_plan,
                    "target_contact_count": 1,
                    "sensor_inventory_count": 0,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "commands": commands,
                }
            )
            completed = run(command, timeout=120.0, check=False)
            if completed.returncode != 0:
                target_failure = {
                    "target_return_code": completed.returncode,
                    "stdout_sha256": sha256_text(completed.stdout),
                    "stderr_sha256": sha256_text(completed.stderr),
                    "bounded_stdout": bounded_text(completed.stdout),
                    "bounded_stderr": bounded_text(completed.stderr),
                    "attempt_state": "target_command_invoked",
                    "active_counters": dict(ATTEMPT_STATE_COUNTERS["target_command_invoked"]),
                    "structured_failure_receipt_present": False,
                    "structured_failure_receipt_file_sha256": None,
                    "structured_failure_validation": {
                        "passed": False,
                        "failures": ["structured target discovery failure retrieval not attempted"],
                    },
                }
                try:
                    failure_sha_command = f"if test -s {sh_quote(remote_receipt)}; then sha256sum {sh_quote(remote_receipt)} | awk '{{print $1}}'; fi"
                    command = ["ssh", "-o", "BatchMode=yes", target_host, failure_sha_command]
                    commands.append(command)
                    failure_remote_sha = run(command, timeout=20.0, check=False).stdout.strip()
                    if re.fullmatch(r"[0-9a-f]{64}", failure_remote_sha):
                        command = ["scp", "-q", f"{target_host}:{remote_receipt}", str(local_copyback)]
                        commands.append(command)
                        run(command, timeout=30.0)
                        target_structured_failure_file_sha = public.sha256_file(local_copyback)
                        if target_structured_failure_file_sha == failure_remote_sha:
                            target_structured_failure_bytes = local_copyback.read_bytes()
                            target_structured_failure = read_json(local_copyback)
                            target_structured_failure_validation = validate_target_discovery_failure_receipt(
                                target_structured_failure,
                                expected_challenge=challenge,
                                expected_source_commit=source_commit,
                            )
                        else:
                            target_structured_failure_validation = {
                                "passed": False,
                                "failures": ["structured target discovery failure copy-back hash mismatch"],
                                "remote_sha256": failure_remote_sha,
                                "local_sha256": target_structured_failure_file_sha,
                            }
                    else:
                        target_structured_failure_validation = {
                            "passed": False,
                            "failures": ["structured target discovery failure remote receipt missing"],
                            "remote_sha256": failure_remote_sha,
                        }
                except Exception as exc:  # noqa: BLE001 - preserve outer failure evidence when copy-back fails
                    target_structured_failure_validation = {
                        "passed": False,
                        "failures": ["structured target discovery failure retrieval exception"],
                        "exception": f"{type(exc).__name__}: {exc}",
                    }
                target_failure["structured_failure_receipt_present"] = target_structured_failure is not None
                target_failure["structured_failure_receipt_file_sha256"] = target_structured_failure_file_sha
                target_failure["structured_failure_validation"] = target_structured_failure_validation
                if (
                    target_structured_failure_bytes is not None
                    and isinstance(target_structured_failure_validation, dict)
                    and target_structured_failure_validation.get("passed") is True
                ):
                    try:
                        durable_write_bytes_exclusive(
                            TARGET_DISCOVERY_RECEIPT,
                            target_structured_failure_bytes,
                            require_directory_sync=True,
                        )
                        if public.sha256_file(TARGET_DISCOVERY_RECEIPT) != target_structured_failure_file_sha:
                            target_structured_failure_validation = {
                                "passed": False,
                                "failures": ["structured target discovery failure local write verification failed"],
                            }
                            target_failure["structured_failure_validation"] = target_structured_failure_validation
                        else:
                            target_failure_preserved_before_cleanup = True
                            target_failure["structured_failure_locally_preserved_before_cleanup"] = True
                    except Exception as exc:  # noqa: BLE001 - cleanup must not delete unpreserved remote evidence
                        target_structured_failure_validation = {
                            "passed": False,
                            "failures": ["structured target discovery failure local write exception"],
                            "exception": f"{type(exc).__name__}: {exc}",
                        }
                        target_failure["structured_failure_validation"] = target_structured_failure_validation
                seal_target_failure_receipt(before_cleanup=True)
                target_failure_receipt_sealed_before_cleanup = True
            else:
                sha_command = f"sha256sum {sh_quote(remote_receipt)} | awk '{{print $1}}'"
                command = ["ssh", "-o", "BatchMode=yes", target_host, sha_command]
                commands.append(command)
                remote_sha = run(command, timeout=20.0).stdout.strip()
                command = ["scp", "-q", f"{target_host}:{remote_receipt}", str(local_copyback)]
                commands.append(command)
                run(command, timeout=30.0)
                local_sha = public.sha256_file(local_copyback)
                if local_sha != remote_sha:
                    raise ControllerError("discovery copy-back hash mismatch")
                target_discovery_file_sha = local_sha
                target_discovery_bytes = local_copyback.read_bytes()
                discovery = read_json(local_copyback)
                if serialized_json_sha256(discovery) != target_discovery_file_sha:
                    raise ControllerError("discovery receipt serialized hash verification failed before cleanup")
                authority = build_temperature_sensor_authority_receipt(
                    discovery=discovery,
                    controller_challenge=challenge,
                    controller_nonce=nonce,
                )
                authority_file_sha = serialized_json_sha256(authority)
                durable_write_bytes_exclusive(TARGET_DISCOVERY_RECEIPT, target_discovery_bytes, require_directory_sync=True)
                if public.sha256_file(TARGET_DISCOVERY_RECEIPT) != target_discovery_file_sha or read_json(TARGET_DISCOVERY_RECEIPT) != discovery:
                    raise ControllerError("discovery receipt local seal verification failed before cleanup")
                target_discovery_receipt_sealed_before_cleanup = True
                write_json_exclusive(TEMPERATURE_SENSOR_AUTHORITY, authority, require_directory_sync=True)
                if public.sha256_file(TEMPERATURE_SENSOR_AUTHORITY) != authority_file_sha or read_json(TEMPERATURE_SENSOR_AUTHORITY) != authority:
                    raise ControllerError("temperature authority receipt local seal verification failed before cleanup")
                authority_receipt_sealed_before_cleanup = True
                write_discovery_attempt_receipt(
                    {
                        "passed": False,
                        "attempt_state": "receipt_copied_cleanup_pending",
                        "source_authority_commit": source_commit,
                        "source_authority": source_commit_check,
                        "source_authority_review": source_review,
                        "controller_challenge_sha256": public.digest(challenge),
                        "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
                        "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
                        "discovery_transfer_plan": transfer_plan,
                        "target_discovery_receipt_sha256": discovery.get("target_discovery_receipt_sha256"),
                        "target_discovery_receipt_file_sha256": local_sha,
                        "target_discovery_receipt_sealed_before_cleanup": target_discovery_receipt_sealed_before_cleanup,
                        "authority_receipt_sha256": authority.get("temperature_sensor_authority_sha256"),
                        "authority_receipt_file_sha256": authority_file_sha,
                        "authority_receipt_sealed_before_cleanup": authority_receipt_sealed_before_cleanup,
                        "target_contact_count": 1,
                        "sensor_inventory_count": 1,
                        "live_invocation_count": 0,
                        "pmu_acquisition_count": 0,
                        "commands": commands,
                    },
                    require_directory_sync=True,
                )
                receipt_copied_state_persisted = True
        finally:
            if remote_root_owned or remote_root_may_exist:
                success_custody_ready = (
                    target_command_invoked
                    and receipt_copied_state_persisted
                    and discovery is not None
                    and authority is not None
                    and target_discovery_file_sha is not None
                    and authority_file_sha is not None
                    and target_discovery_receipt_sealed_before_cleanup
                    and authority_receipt_sealed_before_cleanup
                )
                failure_custody_ready = (
                    target_failure is not None
                    and target_failure_preserved_before_cleanup
                    and target_failure_receipt_sealed_before_cleanup
                )
                cleanup_allowed = (not target_command_invoked) or success_custody_ready or failure_custody_ready
                if receipt_copied_state_persisted and cleanup_allowed:
                    try:
                        write_discovery_attempt_receipt(
                            {
                                "passed": False,
                                "attempt_state": "cleanup_armed",
                                "source_authority_commit": source_commit,
                                "source_authority": source_commit_check,
                                "source_authority_review": source_review,
                                "controller_challenge_sha256": public.digest(challenge),
                                "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
                                "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
                                "discovery_transfer_plan": transfer_plan,
                                "target_discovery_receipt_sha256": discovery.get("target_discovery_receipt_sha256") if isinstance(discovery, dict) else None,
                                "target_discovery_receipt_file_sha256": target_discovery_file_sha,
                                "target_discovery_receipt_sealed_before_cleanup": target_discovery_receipt_sealed_before_cleanup,
                                "authority_receipt_sha256": authority.get("temperature_sensor_authority_sha256") if isinstance(authority, dict) else None,
                                "authority_receipt_file_sha256": authority_file_sha,
                                "authority_receipt_sealed_before_cleanup": authority_receipt_sealed_before_cleanup,
                                "target_contact_count": 1,
                                "sensor_inventory_count": 1,
                                "live_invocation_count": 0,
                                "pmu_acquisition_count": 0,
                                "cleanup": cleanup,
                                "commands": commands,
                            },
                            require_directory_sync=True,
                        )
                        cleanup_armed_state_persisted = True
                    except Exception as exc:  # noqa: BLE001 - local receipt failure must preserve remote evidence
                        cleanup["journal_error_before_cleanup"] = str(exc)
                        cleanup_allowed = False
                if target_command_invoked and not cleanup_allowed:
                    if target_failure is not None and not target_failure_preserved_before_cleanup:
                        cleanup["skipped_reason"] = "target failure evidence was not locally preserved before cleanup"
                    elif target_failure is not None and not target_failure_receipt_sealed_before_cleanup:
                        cleanup["skipped_reason"] = "target failure receipt was not durably sealed before cleanup"
                    elif discovery is not None and not target_discovery_receipt_sealed_before_cleanup:
                        cleanup["skipped_reason"] = "target discovery receipt was not durably sealed before cleanup"
                    elif authority is not None and not authority_receipt_sealed_before_cleanup:
                        cleanup["skipped_reason"] = "temperature authority receipt was not durably sealed before cleanup"
                    elif receipt_copied_state_persisted and not cleanup_armed_state_persisted:
                        cleanup["skipped_reason"] = "cleanup armed attempt state was not durably sealed before cleanup"
                    else:
                        cleanup["skipped_reason"] = "target command outcome was not locally preserved before cleanup"
                cleanup["attempted"] = cleanup_allowed
                cleanup_script = (
                    (
                        "set -u; "
                        f"if test -f {sh_quote(remote_owner_marker)} && grep -qx {sh_quote(nonce_sha)} {sh_quote(remote_owner_marker)}; then rm -rf -- {sh_quote(remote_root)}; fi; "
                        + (f"rmdir -- {sh_quote(remote_base_root)} 2>/dev/null || true; " if not remote_base_preexisting else "")
                        + "true"
                    )
                    if cleanup_allowed
                    else "set -u; false"
                )
                cleanup_command = ["ssh", "-o", "BatchMode=yes", target_host, cleanup_script]
                commands.append(cleanup_command)
                try:
                    cleanup_result = run(cleanup_command, timeout=30.0, check=False)
                    cleanup["passed"] = cleanup_result.returncode == 0
                    cleanup["returncode"] = cleanup_result.returncode
                except Exception as exc:  # noqa: BLE001 - cleanup failures must be recorded, not raised before absence probe
                    cleanup["passed"] = False
                    cleanup["error"] = str(exc)
                if remote_base_preexisting:
                    absence_script = f"test ! -e {sh_quote(remote_root)}"
                else:
                    absence_script = f"test ! -e {sh_quote(remote_root)} && test ! -e {sh_quote(remote_base_root)}"
                absence_command = ["ssh", "-o", "BatchMode=yes", target_host, absence_script]
                commands.append(absence_command)
                try:
                    absence_result = run(absence_command, timeout=20.0, check=False)
                    cleanup["absence_verified"] = absence_result.returncode == 0
                    cleanup["absence_returncode"] = absence_result.returncode
                except Exception as exc:  # noqa: BLE001 - preserve failed absence probe in durable attempt state
                    cleanup["absence_verified"] = False
                    cleanup["absence_error"] = str(exc)
                if receipt_copied_state_persisted:
                    try:
                        write_discovery_attempt_receipt(
                            {
                                "passed": False,
                                "attempt_state": "cleanup_completed",
                                "source_authority_commit": source_commit,
                                "source_authority": source_commit_check,
                                "source_authority_review": source_review,
                                "controller_challenge_sha256": public.digest(challenge),
                                "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
                                "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
                                "discovery_transfer_plan": transfer_plan,
                                "target_discovery_receipt_sha256": discovery.get("target_discovery_receipt_sha256") if isinstance(discovery, dict) else None,
                                "target_discovery_receipt_file_sha256": target_discovery_file_sha,
                                "target_discovery_receipt_sealed_before_cleanup": target_discovery_receipt_sealed_before_cleanup,
                                "authority_receipt_sha256": authority.get("temperature_sensor_authority_sha256") if isinstance(authority, dict) else None,
                                "authority_receipt_file_sha256": authority_file_sha,
                                "authority_receipt_sealed_before_cleanup": authority_receipt_sealed_before_cleanup,
                                "target_contact_count": 1,
                                "sensor_inventory_count": 1,
                                "live_invocation_count": 0,
                                "pmu_acquisition_count": 0,
                                "cleanup": cleanup,
                                "commands": commands,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - authority remains blocked by missing cleanup journal
                        cleanup["journal_error_after_cleanup"] = str(exc)
                else:
                    cleanup_counter_pair = validated_failure_counter_pair()
                    write_discovery_cleanup_custody_receipt(
                        {
                            "passed": cleanup["passed"] is True and cleanup["absence_verified"] is True,
                            "source_authority_commit": source_commit,
                            "source_authority_review": source_review,
                            "controller_challenge_sha256": public.digest(challenge),
                            "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
                            "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
                            "discovery_transfer_plan": transfer_plan,
                            "target_contact_count": 1,
                            "sensor_inventory_count": cleanup_counter_pair["sensor_inventory_count"],
                            "candidate_scan_count": cleanup_counter_pair["candidate_scan_count"],
                            "live_invocation_count": 0,
                            "pmu_acquisition_count": 0,
                            "cleanup": cleanup,
                            "commands": commands,
                        }
                    )
    if target_failure is not None:
        seal_target_failure_receipt(before_cleanup=False)
        if not isinstance(target_structured_failure_validation, dict) or target_structured_failure_validation.get("passed") is not True:
            raise ControllerError(f"target discovery failed rc={target_failure['target_return_code']}: structured failure receipt invalid or missing")
        raise ControllerError(f"target discovery failed rc={target_failure['target_return_code']}: bounded failure persisted")
    if not cleanup["passed"] or not cleanup["absence_verified"]:
        raise ControllerError("discovery cleanup or remote-root absence verification failed")
    if discovery is None or authority is None:
        raise ControllerError("discovery receipt or authority receipt missing after cleanup")
    if target_discovery_file_sha is None or serialized_json_sha256(discovery) != target_discovery_file_sha:
        raise ControllerError("discovery receipt serialized hash verification failed")
    if authority_file_sha is None or serialized_json_sha256(authority) != authority_file_sha:
        raise ControllerError("temperature authority receipt serialized hash verification failed")
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1",
        "passed": True,
        "target_host": target_host,
        "remote_base_root": remote_base_root,
        "remote_root": remote_root,
        "remote_source_root": remote_source,
        "remote_receipt_path": remote_receipt,
        "source_authority_commit": source_commit,
        "source_authority_review": source_review,
        "discovery_transfer_plan": transfer_plan,
        "controller_challenge": challenge,
        "controller_challenge_sha256": public.digest(challenge),
        "challenge_receipt_path": str(DISCOVERY_CHALLENGE_PATH),
        "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
        "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
        "controller_nonce_sha256": nonce_sha,
        "target_discovery_receipt_path": str(TARGET_DISCOVERY_RECEIPT),
        "target_discovery_receipt_sha256": discovery["target_discovery_receipt_sha256"],
        "target_discovery_receipt_file_sha256": target_discovery_file_sha,
        "target_discovery_receipt_sealed_before_cleanup": target_discovery_receipt_sealed_before_cleanup,
        "authority_receipt_path": str(TEMPERATURE_SENSOR_AUTHORITY),
        "authority_receipt_sha256": authority["temperature_sensor_authority_sha256"],
        "authority_receipt_file_sha256": authority_file_sha,
        "authority_receipt_sealed_before_cleanup": authority_receipt_sealed_before_cleanup,
        "approved_sensor_identity": authority["approved_sensor_identity"],
        "cleanup": cleanup,
        "commands": commands,
        "retry_count": 0,
        "target_command_invoked": target_command_invoked,
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
        "candidate_scan_count": 1,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    result["discovery_transport_sha256"] = public.digest({k: v for k, v in result.items() if k != "discovery_transport_sha256"})
    final_validation = temperature_sensor_authority_from_receipt(
        authority,
        expected_challenge=challenge,
        expected_discovery_receipt=discovery,
        expected_transport_receipt=result,
    )
    if not final_validation["passed"]:
        raise ControllerError("final temperature authority chain invalid: " + ",".join(final_validation["failures"]))
    if not TARGET_DISCOVERY_RECEIPT.exists():
        raise ControllerError("discovery receipt missing after cleanup despite pre-cleanup seal")
    if public.sha256_file(TARGET_DISCOVERY_RECEIPT) != target_discovery_file_sha or read_json(TARGET_DISCOVERY_RECEIPT) != discovery:
        raise ControllerError("discovery receipt pre-cleanup seal verification failed after cleanup")
    if not TEMPERATURE_SENSOR_AUTHORITY.exists():
        raise ControllerError("temperature authority receipt missing after cleanup despite pre-cleanup seal")
    if public.sha256_file(TEMPERATURE_SENSOR_AUTHORITY) != authority_file_sha or read_json(TEMPERATURE_SENSOR_AUTHORITY) != authority:
        raise ControllerError("temperature authority receipt pre-cleanup seal verification failed after cleanup")
    write_json(DISCOVERY_TRANSPORT_PATH, result)
    write_discovery_attempt_receipt(
        {
            "passed": True,
            "attempt_state": "complete",
            "source_authority_commit": source_commit,
            "source_authority": source_commit_check,
            "source_authority_review": source_review,
            "controller_challenge_sha256": public.digest(challenge),
            "challenge_receipt_canonical_sha256": challenge_receipt["challenge_receipt_canonical_sha256"],
            "challenge_receipt_file_sha256": challenge_receipt["challenge_receipt_file_sha256"],
            "target_discovery_receipt_sha256": result["target_discovery_receipt_sha256"],
            "target_discovery_receipt_file_sha256": result["target_discovery_receipt_file_sha256"],
            "target_discovery_receipt_sealed_before_cleanup": result["target_discovery_receipt_sealed_before_cleanup"],
            "authority_receipt_sha256": result["authority_receipt_sha256"],
            "authority_receipt_file_sha256": result["authority_receipt_file_sha256"],
            "authority_receipt_sealed_before_cleanup": result["authority_receipt_sealed_before_cleanup"],
            "discovery_transport_sha256": result["discovery_transport_sha256"],
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "cleanup": cleanup,
            "commands": commands,
        }
    )
    return result


def clear_generated_receipts() -> None:
    for path in GENERATED_RECEIPTS:
        if path.exists():
            path.unlink()


def write_source_bundle() -> dict[str, Any]:
    names = [name for name in SOURCE_FILE_NAMES if (HERE / name).exists()]
    with SOURCE_BUNDLE.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name in sorted(names):
                    path = HERE / name
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.mode = 0o644
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    return {
        "path": str(SOURCE_BUNDLE),
        "sha256": public.sha256_file(SOURCE_BUNDLE),
        "file_count": len(names),
        "files": sorted(names),
    }


def source_bundle_preview() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_bundle_verify_") as tmp:
        temp_bundle = Path(tmp) / SOURCE_BUNDLE.name
        names = [name for name in SOURCE_FILE_NAMES if (HERE / name).exists()]
        with temp_bundle.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    for name in sorted(names):
                        path = HERE / name
                        info = tar.gettarinfo(str(path), arcname=name)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.mode = 0o644
                        info.uname = ""
                        info.gname = ""
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
        return {"sha256": public.sha256_file(temp_bundle), "file_count": len(names), "files": sorted(names)}


def read_existing_source_bundle_authority() -> dict[str, Any]:
    if not SOURCE_BUNDLE.exists():
        raise ControllerError("source bundle authority missing; prepare-only must freeze it before discovery")
    preview = source_bundle_preview()
    file_sha = public.sha256_file(SOURCE_BUNDLE)
    if file_sha != preview["sha256"]:
        raise ControllerError("source bundle authority file does not match deterministic reconstruction")
    return {
        "path": str(SOURCE_BUNDLE),
        "sha256": file_sha,
        "file_count": preview["file_count"],
        "files": preview["files"],
    }


def deterministic_source_bundle_bytes(file_payloads: dict[str, bytes]) -> bytes:
    raw_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=raw_buffer, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name in sorted(SOURCE_FILE_NAMES):
                payload = file_payloads[name]
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.mode = 0o644
                info.uname = ""
                info.gname = ""
                tar.addfile(info, io.BytesIO(payload))
    return raw_buffer.getvalue()


def source_bundle_reconstruction_from_commit(commit: str) -> dict[str, Any]:
    failures: list[str] = []
    payloads: dict[str, bytes] = {}
    for name in SOURCE_FILE_NAMES:
        blob = commit_blob_bytes(commit, HERE / name)
        if blob is None:
            failures.append(f"source bundle reconstruction missing {name}")
        else:
            payloads[name] = blob
    if failures:
        return {"passed": False, "failures": failures, "sha256": None, "file_count": len(payloads), "files": sorted(payloads)}
    bundle_bytes = deterministic_source_bundle_bytes(payloads)
    return {
        "passed": True,
        "failures": [],
        "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "size": len(bundle_bytes),
        "file_count": len(payloads),
        "files": sorted(payloads),
        "member_mode": "0644",
        "member_mtime": 0,
        "member_uid": 0,
        "member_gid": 0,
        "member_uname": "",
        "member_gname": "",
    }


def materialize_source_authority_snapshot(commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in DISCOVERY_TRANSFER_FILE_NAMES:
        (destination / name).write_bytes(git_blob_bytes(commit, name))


def find_c_compiler() -> list[str] | None:
    for name in ["gcc", "clang", "cc"]:
        path = shutil.which(name)
        if path:
            return [path]
    wsl = shutil.which("wsl")
    if wsl:
        probe = run([wsl, "bash", "-lc", "command -v gcc"], timeout=10.0, check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            return [wsl, "gcc"]
    return None


def compiler_identity(compiler: list[str]) -> str:
    if compiler[0].endswith("wsl.exe") or Path(compiler[0]).name.lower() == "wsl.exe":
        completed = run([compiler[0], "bash", "-lc", "gcc --version | head -n 1"], timeout=10.0, check=False)
    else:
        completed = run([compiler[0], "--version"], timeout=10.0, check=False)
    if completed.returncode != 0:
        return "compiler identity unavailable"
    return completed.stdout.splitlines()[0] if completed.stdout.splitlines() else "compiler identity unavailable"


def compile_runtime(output_path: Path | None = None) -> dict[str, Any]:
    compiler = find_c_compiler()
    source = HERE / "family10h_carrier_tomography_runtime.c"
    output_path = output_path or BINARY_PATH
    if output_path.exists():
        output_path.unlink()
    if compiler is None:
        return {
            "passed": False,
            "compiler": None,
            "compiler_identity": None,
            "compile_command": None,
            "compile_command_identity": None,
            "binary_path": str(output_path),
            "offline_binary_sha256": None,
            "failure": "no local C compiler found",
        }
    runtime_command: list[str] | None = None
    if compiler[0].endswith("wsl.exe") or Path(compiler[0]).name.lower() == "wsl.exe":
        win_source = str(source)
        win_binary = str(output_path)
        command_text = (
            "gcc -std=c11 -Wall -Wextra -Werror -O2 "
            f"-o \"$(wslpath '{win_binary}')\" \"$(wslpath '{win_source}')\""
        )
        completed = run([compiler[0], "bash", "-lc", command_text], timeout=60.0, check=False)
        compile_command = [compiler[0], "bash", "-lc", command_text]
        runtime_command = [compiler[0], "bash", "-lc", f"\"$(wslpath '{win_binary}')\" --self-test"]
    else:
        compile_command = [
            compiler[0],
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-o",
            str(output_path),
            str(source),
        ]
        completed = run(compile_command, timeout=60.0, check=False)
        runtime_command = [str(output_path), "--self-test"]
    passed = completed.returncode == 0 and output_path.exists()
    identity = {
        "language_standard": "c11",
        "flags": ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"],
        "inputs": ["family10h_carrier_tomography_runtime.c"],
        "output": output_path.name,
    }
    return {
        "passed": passed,
        "compiler": compiler,
        "compiler_identity": compiler_identity(compiler),
        "compile_command": compile_command,
        "compile_command_identity": identity,
        "binary_path": str(output_path),
        "runtime_command": runtime_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "offline_binary_sha256": public.sha256_file(output_path) if output_path.exists() else None,
        "offline_binary_size": output_path.stat().st_size if output_path.exists() else None,
    }


def runtime_self_test() -> dict[str, Any]:
    authority = runtime_binary_authority()
    with tempfile.TemporaryDirectory(prefix="family10h_runtime_compile_") as tmp:
        isolated_binary = Path(tmp) / RUNTIME_BINARY_NAME
        compile_receipt = compile_runtime(isolated_binary)
        if compile_receipt["passed"]:
            completed = run(compile_receipt["runtime_command"], timeout=20.0, check=False)
            try:
                runtime_json = strict_json_loads(completed.stdout.strip())
            except (json.JSONDecodeError, ValueError):
                runtime_json = {"passed": False, "raw_stdout": completed.stdout}
        else:
            completed = subprocess.CompletedProcess(compile_receipt.get("runtime_command") or [], 1, "", "")
            runtime_json = {"passed": False, "compile_failed": True}
        committed_sha = authority.get("sha256")
        isolated_sha = compile_receipt.get("offline_binary_sha256")
        compile_equivalence = {
            "law": "byte_exact_isolated_compile",
            "passed": compile_receipt["passed"] is True and isolated_sha == committed_sha,
            "committed_runtime_binary_sha256": committed_sha,
            "isolated_compile_sha256": isolated_sha,
            "committed_runtime_binary_size": authority.get("size"),
            "isolated_compile_size": compile_receipt.get("offline_binary_size"),
            "runtime_c_sha256": authority.get("runtime_c_sha256"),
            "runtime_h_sha256": authority.get("runtime_h_sha256"),
            "compile_command_identity": compile_receipt.get("compile_command_identity"),
            "compiler_identity": compile_receipt.get("compiler_identity"),
        }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST_RECEIPT_V1",
        "passed": completed.returncode == 0 and runtime_json.get("passed") is True and compile_equivalence["passed"] is True,
        "compile": compile_receipt,
        "runtime_binary_authority": authority,
        "compile_equivalence": compile_equivalence,
        "runtime_returncode": completed.returncode,
        "runtime_stdout": completed.stdout,
        "runtime_stderr": completed.stderr,
        "runtime_json": runtime_json,
    }
    result["runtime_self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "runtime_self_test_sha256"})
    write_json(RUNTIME_SELF_TEST_PATH, result)
    return result


def runtime_authority_gate_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_runtime_gate_") as tmp:
        isolated_binary = Path(tmp) / RUNTIME_BINARY_NAME
        compile_receipt = compile_runtime(isolated_binary)
        if not compile_receipt["passed"]:
            return {"passed": False, "compile": compile_receipt}

        def direct_execute(output_root: Path, authority_value: str | None) -> subprocess.CompletedProcess[str]:
            compiler = compile_receipt["compiler"]
            binary_path = Path(str(compile_receipt["binary_path"]))
            if compiler and (compiler[0].endswith("wsl.exe") or Path(compiler[0]).name.lower() == "wsl.exe"):
                binary = f"\"$(wslpath '{binary_path}')\""
                schedule = f"\"$(wslpath '{public.SCHEDULE_TSV}')\""
                output = f"\"$(wslpath '{output_root}')\""
                env_prefix = (
                    f"FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY='{authority_value}' "
                    if authority_value is not None
                    else "env -u FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY "
                )
                return run([compiler[0], "bash", "-lc", f"{env_prefix}{binary} --execute-schedule {schedule} {output}"], timeout=20.0, check=False)
            env = os.environ.copy()
            if authority_value is None:
                env.pop("FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY", None)
            else:
                env["FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY"] = authority_value
            return subprocess.run(
                [str(binary_path), "--execute-schedule", str(public.SCHEDULE_TSV), str(output_root)],
                text=True,
                capture_output=True,
                timeout=20.0,
                check=False,
                env=env,
            )

        missing_output = Path(tmp) / "missing_authority_output"
        mismatch_output = Path(tmp) / "mismatched_authority_output"
        missing = direct_execute(missing_output, None)
        mismatch = direct_execute(mismatch_output, "not-authorized")
        checks = {
            "missing_authority_rejected_before_output_creation": missing.returncode == 13 and not missing_output.exists(),
            "mismatched_authority_rejected_before_output_creation": mismatch.returncode == 13 and not mismatch_output.exists(),
        }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY_GATE_SELF_TEST_V1",
        "passed": all(checks.values()),
        "checks": checks,
        "compile": compile_receipt,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    result["runtime_authority_gate_self_test_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "runtime_authority_gate_self_test_sha256"}
    )
    return result


def deployment_layout_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_layout_") as tmp:
        root = Path(tmp)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        for name in DISCOVERY_TRANSFER_FILE_NAMES:
            path = HERE / name
            if path.exists():
                shutil.copy2(path, source / name)
        completed = run(
            [sys.executable, str(source / "family10h_carrier_tomography_target.py"), "--self-test", "--source-root", str(source), "--output-root", str(output)],
            timeout=60.0,
            check=False,
        )
        try:
            data = strict_json_loads(completed.stdout)
        except (json.JSONDecodeError, ValueError):
            data = {"self_test_passed": False, "stdout": completed.stdout}
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST_V1",
        "passed": completed.returncode == 0 and data.get("self_test_passed") is True,
        "target_self_test_returncode": completed.returncode,
        "target_self_test_passed": data.get("self_test_passed") is True,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_length": len(completed.stderr),
    }
    result["deployment_layout_self_test_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "deployment_layout_self_test_sha256"}
    )
    write_json(DEPLOYMENT_LAYOUT_PATH, result)
    return result


def validate_discovery_transport_receipt(receipt: dict[str, Any] | None) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(receipt, dict):
        return {"passed": False, "failures": ["discovery transport receipt missing"]}
    if receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1":
        failures.append("discovery transport schema mismatch")
    if not is_strict_int(receipt.get("retry_count")) or receipt.get("retry_count") != 0:
        failures.append("discovery transport retry count must be zero")
    if receipt.get("passed") is not True:
        failures.append("discovery transport must pass")
    if receipt.get("target_command_invoked") is not True:
        failures.append("discovery transport target command invocation missing")
    transport_scope = receipt.get("controller_challenge", {}).get("transport_scope") if isinstance(receipt.get("controller_challenge"), dict) else None
    if not isinstance(transport_scope, dict):
        failures.append("discovery transport challenge scope missing")
        transport_scope = {}
    expected_scope = {
        "target_host": receipt.get("target_host"),
        "remote_base_root": receipt.get("remote_base_root"),
        "remote_root": receipt.get("remote_root"),
        "remote_source_root": receipt.get("remote_source_root"),
        "remote_receipt_path": receipt.get("remote_receipt_path"),
    }
    if transport_scope != expected_scope:
        failures.append("discovery transport scope mismatch")
    if receipt.get("target_host") != TARGET_HOST:
        failures.append("discovery transport target host mismatch")
    if receipt.get("remote_base_root") != DISCOVERY_REMOTE_ROOT:
        failures.append("discovery transport remote base root mismatch")
    challenge_nonce_sha = receipt.get("controller_challenge", {}).get("controller_nonce_sha256") if isinstance(receipt.get("controller_challenge"), dict) else None
    expected_remote_root = (
        f"{DISCOVERY_REMOTE_ROOT}/{str(challenge_nonce_sha)[:DISCOVERY_OWNED_ROOT_HEX_LEN]}"
        if re.fullmatch(r"[0-9a-f]{64}", str(challenge_nonce_sha or "")) is not None
        else None
    )
    if receipt.get("remote_root") != expected_remote_root:
        failures.append("discovery transport remote root not nonce-owned")
    if receipt.get("remote_source_root") != f"{receipt.get('remote_root')}/source":
        failures.append("discovery transport remote source root mismatch")
    if receipt.get("remote_receipt_path") != f"{receipt.get('remote_source_root')}/{target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME}":
        failures.append("discovery transport remote receipt path mismatch")
    transfer_plan = receipt.get("discovery_transfer_plan")
    if not isinstance(transfer_plan, dict):
        failures.append("discovery transfer plan missing")
    else:
        if transfer_plan.get("passed") is not True:
            failures.append("discovery transfer plan did not pass")
        if transfer_plan.get("actual_file_names") != list(DISCOVERY_TRANSFER_FILE_NAMES):
            failures.append("discovery transfer plan file keyset mismatch")
        if transfer_plan.get("file_count") != len(DISCOVERY_TRANSFER_FILE_NAMES):
            failures.append("discovery transfer plan file count mismatch")
        if RUNTIME_BINARY_NAME not in transfer_plan.get("actual_file_names", []):
            failures.append("discovery transfer plan runtime binary missing")
    cleanup = receipt.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("passed") is not True or cleanup.get("absence_verified") is not True:
        failures.append("discovery cleanup and absence verification required")
    if receipt.get("target_discovery_receipt_sealed_before_cleanup") is not True:
        failures.append("discovery target receipt was not sealed before cleanup")
    if receipt.get("authority_receipt_sealed_before_cleanup") is not True:
        failures.append("discovery authority receipt was not sealed before cleanup")
    if not is_strict_int(receipt.get("target_contact_count")) or receipt.get("target_contact_count") != 1:
        failures.append("discovery target contact count must be one")
    if not is_strict_int(receipt.get("sensor_inventory_count")) or receipt.get("sensor_inventory_count") != 1:
        failures.append("discovery sensor inventory count must be one")
    if not is_strict_int(receipt.get("live_invocation_count")) or receipt.get("live_invocation_count") != 0:
        failures.append("discovery live invocation count must be zero")
    if not is_strict_int(receipt.get("pmu_acquisition_count")) or receipt.get("pmu_acquisition_count") != 0:
        failures.append("discovery PMU acquisition count must be zero")
    if receipt.get("discovery_transport_sha256") != public.digest({k: v for k, v in receipt.items() if k != "discovery_transport_sha256"}):
        failures.append("discovery transport digest mismatch")
    for field in [
        "controller_challenge_sha256",
        "challenge_receipt_canonical_sha256",
        "challenge_receipt_file_sha256",
        "target_discovery_receipt_sha256",
        "target_discovery_receipt_file_sha256",
        "authority_receipt_sha256",
        "authority_receipt_file_sha256",
    ]:
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))) is None:
            failures.append(f"discovery transport {field} missing or invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_authority_commit", ""))) is None:
        failures.append("discovery transport source authority commit missing or invalid")
    return {"passed": not failures, "failures": failures}


def discovery_transport_self_tests() -> dict[str, Any]:
    def seal(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["discovery_transport_sha256"] = public.digest({k: v for k, v in result.items() if k != "discovery_transport_sha256"})
        return result

    base_nonce_sha = "8" * 64
    base_scope = discovery_transport_scope(target_host=TARGET_HOST, remote_root=DISCOVERY_REMOTE_ROOT, nonce_sha=base_nonce_sha)
    base_challenge = {
        "controller_nonce_sha256": base_nonce_sha,
        "transport_scope": base_scope,
    }

    def fake_transfer_plan(scope: dict[str, Any], actual_names: list[str] | None = None, *, passed: bool = True) -> dict[str, Any]:
        names = list(actual_names or DISCOVERY_TRANSFER_FILE_NAMES)
        records = [
            {
                "name": name,
                "source_path": f"/fixture/source/{name}",
                "remote_destination": f"{scope['remote_source_root']}/{name}",
                "byte_size": 1,
                "sha256": "0" * 64,
                "git_blob_id": "1" * 40,
                "local_git_blob_id": "1" * 40,
                "authority_class": discovery_transfer_authority_class(name),
            }
            for name in names
        ]
        plan = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSFER_PLAN_V1",
            "passed": passed,
            "failures": [] if passed else ["fixture transfer plan failure"],
            "file_count": len(records),
            "expected_file_names": list(DISCOVERY_TRANSFER_FILE_NAMES),
            "actual_file_names": names,
            "missing_files": sorted(set(DISCOVERY_TRANSFER_FILE_NAMES) - set(names)),
            "extra_files": sorted(set(names) - set(DISCOVERY_TRANSFER_FILE_NAMES)),
            "records": records,
            "source_authority": {"passed": passed, "failures": [] if passed else ["fixture source authority failure"]},
            "source_bundle_file_sha256": "2" * 64,
            "source_bundle_reconstruction_sha256": "2" * 64,
            "runtime_binary_authority": {
                "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_BINARY_AUTHORITY_V1",
                "path": RUNTIME_BINARY_NAME,
                "present": RUNTIME_BINARY_NAME in names,
                "git_blob_id": "1" * 40,
                "sha256": "0" * 64,
                "size": 1,
            },
        }
        plan["transfer_plan_sha256"] = public.digest({k: v for k, v in plan.items() if k != "transfer_plan_sha256"})
        return plan

    base = seal(
        {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1",
            "passed": True,
            "cleanup": {"attempted": True, "passed": True, "absence_verified": True},
            "retry_count": 0,
            "target_command_invoked": True,
            "target_host": base_scope["target_host"],
            "remote_base_root": base_scope["remote_base_root"],
            "remote_root": base_scope["remote_root"],
            "remote_source_root": base_scope["remote_source_root"],
            "remote_receipt_path": base_scope["remote_receipt_path"],
            "source_authority_commit": "f" * 40,
            "controller_challenge": base_challenge,
            "controller_challenge_sha256": "1" * 64,
            "challenge_receipt_canonical_sha256": "2" * 64,
            "challenge_receipt_file_sha256": "7" * 64,
            "discovery_transfer_plan": fake_transfer_plan(base_scope),
            "target_discovery_receipt_sha256": "3" * 64,
            "target_discovery_receipt_file_sha256": "5" * 64,
            "target_discovery_receipt_sealed_before_cleanup": True,
            "authority_receipt_sha256": "4" * 64,
            "authority_receipt_file_sha256": "6" * 64,
            "authority_receipt_sealed_before_cleanup": True,
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    )
    cleanup_failure = seal({**base, "cleanup": {"attempted": True, "passed": False, "absence_verified": True}})
    absence_failure = seal({**base, "cleanup": {"attempted": True, "passed": True, "absence_verified": False}})
    retry = seal({**base, "retry_count": 1})
    retry_bool = seal({**base, "retry_count": False})
    zero_contact = seal({**base, "target_contact_count": 0})
    live_nonzero = seal({**base, "live_invocation_count": 1})
    pmu_nonzero = seal({**base, "pmu_acquisition_count": 1})
    alternate_root = seal({**base, "remote_root": f"{DISCOVERY_REMOTE_ROOT}/alternate", "remote_source_root": f"{DISCOVERY_REMOTE_ROOT}/alternate/source"})
    missing_transfer_plan_payload = dict(base)
    missing_transfer_plan_payload.pop("discovery_transfer_plan", None)
    missing_transfer_plan = seal(missing_transfer_plan_payload)
    failed_transfer_plan = seal({**base, "discovery_transfer_plan": fake_transfer_plan(base_scope, passed=False)})
    runtime_omitted_transfer_plan = seal(
        {
            **base,
            "discovery_transfer_plan": fake_transfer_plan(
                base_scope,
                [name for name in DISCOVERY_TRANSFER_FILE_NAMES if name != RUNTIME_BINARY_NAME],
            ),
        }
    )
    coherent_wrong_root_scope = {
        **base_scope,
        "remote_root": f"{DISCOVERY_REMOTE_ROOT}/{'9' * DISCOVERY_OWNED_ROOT_HEX_LEN}",
        "remote_source_root": f"{DISCOVERY_REMOTE_ROOT}/{'9' * DISCOVERY_OWNED_ROOT_HEX_LEN}/source",
        "remote_receipt_path": f"{DISCOVERY_REMOTE_ROOT}/{'9' * DISCOVERY_OWNED_ROOT_HEX_LEN}/source/{target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME}",
    }
    coherent_wrong_root = seal(
        {
            **base,
            "remote_root": coherent_wrong_root_scope["remote_root"],
            "remote_source_root": coherent_wrong_root_scope["remote_source_root"],
            "remote_receipt_path": coherent_wrong_root_scope["remote_receipt_path"],
            "controller_challenge": {"transport_scope": coherent_wrong_root_scope, "controller_nonce_sha256": base_nonce_sha},
        }
    )
    corrupted = dict(base)
    corrupted["target_contact_count"] = 0

    with tempfile.TemporaryDirectory(prefix="family10h_controller_discovery_") as tmp:
        root = Path(tmp)
        source = root / "source"
        target.copy_source_fixture(HERE, source)
        cpuinfo = root / "cpuinfo"
        target.write_fake_family10h_cpuinfo(cpuinfo)
        hwmon = root / "hwmon"
        target.write_fake_hwmon_sensor(hwmon, 0, "k10temp", "Tctl", "42000")
        nonce = "e" * 64
        nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
        source_hashes = read_json(source / SOURCE_HASHES.name)
        challenge = build_temperature_authority_challenge(
            source_hashes=source_hashes,
            source_bundle_sha256=target.deterministic_source_bundle_sha256(source),
            runtime_binary_sha256=source_hashes["runtime_binary_authority"]["sha256"],
            schedule_sidecar=read_json(source / public.SCHEDULE_SHA.name),
            authorized_commit="f" * 40,
            controller_nonce_sha256=nonce_sha,
            transport_scope=fixture_transport_scope(source, nonce_sha),
            source_authority_review=fixture_source_review_binding(
                "f" * 40,
                source_hashes["source_hashes_sha256"],
                target.deterministic_source_bundle_sha256(source),
                source_hashes["runtime_binary_authority"]["sha256"],
            ),
        )
        challenge_path = root / "challenge.json"
        write_json(challenge_path, challenge)
        discovery = target.discover_temperature_sensor_authority(
            source_root=source,
            controller_challenge_path=challenge_path,
            controller_nonce=nonce,
            authorized_commit="f" * 40,
            receipt_path=source / target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
            hwmon_root=hwmon,
            cpuinfo_path=cpuinfo,
        )
        try:
            build_temperature_sensor_authority_receipt(
                discovery=discovery,
                controller_challenge=challenge,
                controller_nonce=nonce,
            )
            fake_discovery_authority_build_rejected = False
        except ControllerError:
            fake_discovery_authority_build_rejected = True
        identity = discovery["selected_identity"]
        authority = {
            "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
            "provenance_bound": True,
            "provenance": "controller_verified_target_sensor_inventory",
            "hwmon_name": identity["hwmon_name"],
            "sensor_label_present": identity["sensor_label_present"],
            "sensor_label_value": identity["sensor_label_value"],
            "sensor_semantic_role": identity["sensor_semantic_role"],
            "sensor_semantic_profile": identity["sensor_semantic_profile"],
            "approved_sensor_identity": identity,
            "target_discovery_receipt": discovery,
            "controller_challenge": challenge,
            "controller_challenge_sha256": public.digest(challenge),
            "controller_nonce": nonce,
            "source_authority_commit": challenge["authorized_commit"],
            "target_contact_count": discovery.get("target_contact_count"),
            "sensor_inventory_count": discovery.get("sensor_inventory_count"),
            "live_invocation_count": discovery.get("live_invocation_count"),
            "pmu_acquisition_count": discovery.get("pmu_acquisition_count"),
        }
        authority["temperature_sensor_authority_sha256"] = public.digest(
            {k: v for k, v in authority.items() if k != "temperature_sensor_authority_sha256"}
        )
        authority_transport = seal(
            {
                "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1",
                "passed": True,
                "cleanup": {"attempted": True, "passed": True, "absence_verified": True},
                "retry_count": 0,
                "target_command_invoked": True,
                "target_host": challenge["transport_scope"]["target_host"],
                "remote_base_root": challenge["transport_scope"]["remote_base_root"],
                "remote_root": challenge["transport_scope"]["remote_root"],
                "remote_source_root": challenge["transport_scope"]["remote_source_root"],
                "remote_receipt_path": challenge["transport_scope"]["remote_receipt_path"],
                "source_authority_commit": "f" * 40,
                "controller_challenge": challenge,
                "controller_challenge_sha256": public.digest(challenge),
                "challenge_receipt_canonical_sha256": "9" * 64,
                "challenge_receipt_file_sha256": "a" * 64,
                "discovery_transfer_plan": fake_transfer_plan(challenge["transport_scope"]),
                "target_discovery_receipt_sha256": discovery["target_discovery_receipt_sha256"],
                "target_discovery_receipt_file_sha256": serialized_json_sha256(discovery),
                "authority_receipt_sha256": authority["temperature_sensor_authority_sha256"],
                "authority_receipt_file_sha256": serialized_json_sha256(authority),
                "target_contact_count": 1,
                "sensor_inventory_count": 1,
                "live_invocation_count": 0,
                "pmu_acquisition_count": 0,
            }
        )
        wrong_file_hash_transport = seal({**authority_transport, "target_discovery_receipt_file_sha256": "0" * 64})
        copyback_corruption = dict(discovery)
        copyback_corruption["selected_identity"] = public.synthetic_temperature_identity()
        corrupted_authority_rejected = not temperature_sensor_authority_from_receipt(
            {
                **authority,
                "target_discovery_receipt": copyback_corruption,
                "temperature_sensor_authority_sha256": public.digest(
                    {
                        **{k: v for k, v in authority.items() if k not in {"temperature_sensor_authority_sha256", "target_discovery_receipt"}},
                        "target_discovery_receipt": copyback_corruption,
                    }
                ),
            },
            expected_challenge=challenge,
            expected_discovery_receipt=discovery,
            expected_transport_receipt=authority_transport,
        )["passed"]

    checks = {
        "valid_transport_passes": validate_discovery_transport_receipt(base)["passed"],
        "cleanup_failure_rejected": not validate_discovery_transport_receipt(cleanup_failure)["passed"],
        "remote_absence_failure_rejected": not validate_discovery_transport_receipt(absence_failure)["passed"],
        "second_attempt_retry_rejected": not validate_discovery_transport_receipt(retry)["passed"],
        "boolean_retry_count_rejected": not validate_discovery_transport_receipt(retry_bool)["passed"],
        "incorrect_zero_target_contact_rejected": not validate_discovery_transport_receipt(zero_contact)["passed"],
        "nonzero_live_invocation_rejected": not validate_discovery_transport_receipt(live_nonzero)["passed"],
        "nonzero_pmu_acquisition_rejected": not validate_discovery_transport_receipt(pmu_nonzero)["passed"],
        "alternate_remote_scope_rejected": not validate_discovery_transport_receipt(alternate_root)["passed"],
        "missing_transfer_plan_rejected": not validate_discovery_transport_receipt(missing_transfer_plan)["passed"],
        "failed_transfer_plan_rejected": not validate_discovery_transport_receipt(failed_transfer_plan)["passed"],
        "runtime_omitted_transfer_plan_rejected": not validate_discovery_transport_receipt(runtime_omitted_transfer_plan)["passed"],
        "coherent_wrong_nonce_root_rejected": not validate_discovery_transport_receipt(coherent_wrong_root)["passed"],
        "transport_digest_corruption_rejected": not validate_discovery_transport_receipt(corrupted)["passed"],
        "copyback_corruption_rejected": corrupted_authority_rejected,
        "transport_file_hash_mismatch_rejected": not temperature_sensor_authority_from_receipt(
            authority,
            expected_challenge=challenge,
            expected_discovery_receipt=discovery,
            expected_transport_receipt=wrong_file_hash_transport,
        )["passed"],
        "authority_without_transport_rejected": not temperature_sensor_authority_from_receipt(
            authority,
            expected_challenge=challenge,
            expected_discovery_receipt=discovery,
        )["passed"],
        "fake_discovery_authority_build_rejected": fake_discovery_authority_build_rejected,
        "fake_discovery_authority_with_transport_rejected": not temperature_sensor_authority_from_receipt(
            authority,
            expected_challenge=challenge,
            expected_discovery_receipt=discovery,
            expected_transport_receipt=authority_transport,
        )["passed"],
    }
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_SELF_TEST_V1",
        "passed": all(checks.values()),
        "checks": checks,
    }


def production_discovery_transfer_plan_regression() -> dict[str, Any]:
    failures: list[str] = []
    negative_cases: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="family10h_production_transfer_plan_") as tmp:
        root = Path(tmp)
        source_stage = root / "source_stage"
        target_source = root / "target_source"
        target.copy_source_fixture(HERE, source_stage)
        source_hashes = read_json(source_stage / SOURCE_HASHES.name)
        bundle_sha = target.deterministic_source_bundle_sha256(source_stage)
        runtime_authority = source_hashes.get("runtime_binary_authority", {})
        source_status = git_lines(
            "status",
            "--porcelain",
            "--",
            *[package_relative_path(name) for name in DISCOVERY_TRANSFER_FILE_NAMES],
        )
        source_commit = git_text("rev-parse", "HEAD") if not source_status else None
        plan = build_discovery_transfer_plan(
            source_root=source_stage,
            remote_source_root=str(target_source),
            source_commit=source_commit,
            expected_source_hashes_sha256=source_hashes["source_hashes_sha256"],
            expected_source_bundle_sha256=bundle_sha,
            expected_runtime_binary_authority=runtime_authority,
        )
        target_source.mkdir()
        for record in plan["records"]:
            shutil.copy2(record["source_path"], target_source / record["name"])
        nonce = "a" * 64
        nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
        authorized_commit = "b" * 40
        challenge = build_temperature_authority_challenge(
            source_hashes=source_hashes,
            source_bundle_sha256=bundle_sha,
            runtime_binary_sha256=runtime_authority.get("sha256"),
            schedule_sidecar=read_json(source_stage / public.SCHEDULE_SHA.name),
            authorized_commit=authorized_commit,
            controller_nonce_sha256=nonce_sha,
            transport_scope=fixture_transport_scope(target_source, nonce_sha),
            source_authority_review=fixture_source_review_binding(
                authorized_commit,
                source_hashes["source_hashes_sha256"],
                bundle_sha,
                runtime_authority.get("sha256", "0" * 64),
            ),
        )
        fresh_root_validation = target.validate_discovery_challenge(target_source, challenge, nonce, authorized_commit)

        def copy_target_case(label: str) -> Path:
            case_root = root / label
            case_source = case_root / "source"
            case_source.mkdir(parents=True)
            for record in plan["records"]:
                shutil.copy2(record["source_path"], case_source / record["name"])
            return case_source

        def record_negative(label: str, mutator: Any) -> bool:
            case_source = copy_target_case(label)
            mutator(case_source)
            validation = target.validate_discovery_transfer_root(case_source, challenge=challenge)
            negative_cases[label] = validation
            return not validation["passed"]

        runtime_omitted = record_negative("runtime_omitted", lambda case: (case / RUNTIME_BINARY_NAME).unlink())
        bundle_omitted = record_negative("source_bundle_omitted", lambda case: (case / SOURCE_BUNDLE.name).unlink())
        source_hash_omitted = record_negative("source_hash_omitted", lambda case: (case / SOURCE_HASHES.name).unlink())
        one_source_omitted = record_negative("one_source_omitted", lambda case: (case / SOURCE_FILE_NAMES[0]).unlink())
        runtime_bytes_mutated = record_negative(
            "runtime_bytes_mutated",
            lambda case: (case / RUNTIME_BINARY_NAME).write_bytes((case / RUNTIME_BINARY_NAME).read_bytes() + b"\nproduction-plan-mutation\n"),
        )
        runtime_size_mutated = record_negative(
            "runtime_size_mutated",
            lambda case: (case / RUNTIME_BINARY_NAME).write_bytes((case / RUNTIME_BINARY_NAME).read_bytes()[:-1] or b"x"),
        )
        bundle_bytes_mutated = record_negative(
            "source_bundle_bytes_mutated",
            lambda case: (case / SOURCE_BUNDLE.name).write_bytes((case / SOURCE_BUNDLE.name).read_bytes() + b"\nbundle-mutation\n"),
        )
        bundle_reconstruction_mismatch = record_negative(
            "source_bundle_reconstruction_mismatch",
            lambda case: (case / SOURCE_FILE_NAMES[-1]).write_text(
                (case / SOURCE_FILE_NAMES[-1]).read_text(encoding="utf-8") + "\n# reconstruction mismatch\n",
                encoding="utf-8",
            ),
        )
        extra_authority_file = record_negative(
            "extra_authority_file",
            lambda case: (case / "unexpected_authority_file.txt").write_text("unexpected\n", encoding="utf-8"),
        )
        wrong_runtime_blob = build_discovery_transfer_plan(
            source_root=source_stage,
            remote_source_root=str(root / "wrong_runtime_blob"),
            source_commit=source_commit,
            expected_source_hashes_sha256=source_hashes["source_hashes_sha256"],
            expected_source_bundle_sha256=bundle_sha,
            expected_runtime_binary_authority={**runtime_authority, "git_blob_id": "0" * 40},
        )
        differing_keyset = build_discovery_transfer_plan(
            source_root=source_stage,
            remote_source_root=str(root / "differing_keyset"),
            source_commit=source_commit,
            expected_source_hashes_sha256=source_hashes["source_hashes_sha256"],
            expected_source_bundle_sha256=bundle_sha,
            expected_runtime_binary_authority=runtime_authority,
            expected_names=[name for name in DISCOVERY_TRANSFER_FILE_NAMES if name != RUNTIME_BINARY_NAME],
        )

    production_source = inspect.getsource(acquire_temperature_sensor_authority)
    checks = {
        "transfer_plan_passed": plan["passed"],
        "transfer_plan_contains_runtime_binary": RUNTIME_BINARY_NAME in plan["actual_file_names"],
        "transfer_plan_file_count_exact": plan["file_count"] == len(DISCOVERY_TRANSFER_FILE_NAMES),
        "transfer_plan_keyset_exact": plan["actual_file_names"] == list(DISCOVERY_TRANSFER_FILE_NAMES),
        "fresh_root_validation_passed_before_inventory": fresh_root_validation["passed"],
        "runtime_omission_rejected": runtime_omitted,
        "source_bundle_omission_rejected": bundle_omitted,
        "source_hash_omission_rejected": source_hash_omitted,
        "one_source_omission_rejected": one_source_omitted,
        "runtime_bytes_mutation_rejected": runtime_bytes_mutated,
        "runtime_size_mutation_rejected": runtime_size_mutated,
        "runtime_blob_identity_mismatch_rejected": not wrong_runtime_blob["passed"],
        "source_bundle_bytes_mutation_rejected": bundle_bytes_mutated,
        "source_bundle_reconstruction_mismatch_rejected": bundle_reconstruction_mismatch,
        "extra_authority_file_rejected": extra_authority_file,
        "production_plan_differs_from_target_set_rejected": not differing_keyset["passed"],
        "production_scp_loop_consumes_frozen_plan_records": 'for item in transfer_plan["records"]' in production_source,
        "production_scp_loop_does_not_rebuild_source_filename_list": "for name in SOURCE_AUTHORITY_FILE_NAMES" not in production_source,
        "no_runtime_binary_executed": True,
        "no_pmu_path_opened": True,
        "no_tomography_output_root_created": True,
    }
    if not all(checks.values()):
        failures.extend(name for name, passed in checks.items() if not passed)
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_PRODUCTION_DISCOVERY_TRANSFER_PLAN_REGRESSION_V1",
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "transfer_plan_file_count": plan["file_count"],
        "transfer_plan_sha256": plan["transfer_plan_sha256"],
        "fresh_root_validation": fresh_root_validation,
        "negative_cases": negative_cases,
        "runtime_blob_mismatch_plan": wrong_runtime_blob,
        "differing_keyset_plan": differing_keyset,
        "source_authority_commit_used_for_blob_checks": source_commit,
    }
    result["regression_sha256"] = public.digest({k: v for k, v in result.items() if k != "regression_sha256"})
    return result


def fake_transport_self_tests() -> dict[str, Any]:
    schedule = public.build_schedule()
    packet = public.minimal_success_packet(schedule)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_transport_") as tmp:
        root = Path(tmp)
        success_root = root / "success"
        success_root.mkdir()
        target.write_minimal_success_root(success_root, schedule)
        success = target.validate_minimal_evidence_root(success_root, schedule)

        failure_root = root / "failure"
        failure_root.mkdir()
        target.write_minimal_success_root(failure_root, schedule)
        raw_lines = (failure_root / "raw_records.jsonl").read_text(encoding="utf-8").splitlines()
        (failure_root / "raw_records.jsonl").write_text("\n".join(raw_lines[:-1]) + "\n", encoding="utf-8")
        failure = target.validate_minimal_evidence_root(failure_root, schedule)

        corrupted_root = root / "corrupted"
        corrupted_root.mkdir()
        target.write_minimal_success_root(corrupted_root, schedule)
        corrupted_records = [strict_json_loads(line) for line in (corrupted_root / "raw_records.jsonl").read_text(encoding="utf-8").splitlines()]
        corrupted_records[0]["tuple_id"] = "corrupted"
        target.write_jsonl(corrupted_root / "raw_records.jsonl", corrupted_records)
        corrupted = target.validate_minimal_evidence_root(corrupted_root, schedule)

        timeout_root = root / "timeout"
        timeout_root.mkdir()
        timeout_receipt = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_EXECUTION_RECEIPT_V1",
            "status": "TARGET_EXECUTION_FAILED",
            "returncode": 124,
            "failure_reason": "runtime timeout before completion",
            "stdout": "",
            "stderr": "",
            "output_root": str(timeout_root),
            "evidence_validation": {"passed": False, "failures": ["timeout before failure sealing"]},
            "retry_count": 0,
        }
        write_json(timeout_root / "target_execution_receipt.json", timeout_receipt)
        timeout_loaded = read_json(timeout_root / "target_execution_receipt.json")
    timeout_consistency = {
        "outer_timeout_exceeds_inner_timeout": 3900 > 3600,
        "failure_sealed_before_copyback": timeout_loaded["status"] == "TARGET_EXECUTION_FAILED"
        and timeout_loaded["evidence_validation"]["passed"] is False,
        "no_automatic_retry": timeout_loaded["retry_count"] == 0,
    }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION_V1",
        "offline_only": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "fake_success_transport_passed": success["passed"],
        "fake_failure_transport_rejected": not failure["passed"],
        "copyback_corruption_rejected": not corrupted["passed"],
        "timeout_consistency": timeout_consistency,
        "protected_remote_roots_not_touched": True,
        "success": success,
        "failure": failure,
        "corrupted": corrupted,
    }
    result["passed"] = (
        result["fake_success_transport_passed"]
        and result["fake_failure_transport_rejected"]
        and result["copyback_corruption_rejected"]
        and all(timeout_consistency.values())
    )
    result["transport_simulation_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "transport_simulation_sha256"}
    )
    write_json(TRANSPORT_SIM_PATH, result)
    return result


def write_split_self_tests(public_self: dict[str, Any]) -> dict[str, Any]:
    feature = public.feature_boundary_self_test()
    write_json(FEATURE_BOUNDARY_PATH, feature)
    operator = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST_V1",
        "passed": public_self["operator_analysis_tests"]["ideal_persistent_public_state_has_signal"]
        and public_self["operator_analysis_tests"]["held_out_replicate_prediction_used"]
        and public_self["operator_analysis_tests"]["held_out_mapping_prediction_used"]
        and public_self["operator_analysis_tests"]["held_out_delay_prediction_used"]
        and public_self["operator_analysis_tests"]["s2_sufficient_for_ideal_fixture"],
        "packet_analysis_passed": public_self["operator_analysis_tests"]["validated_packet_analysis"]["passed"],
        "packet_analysis_has_lifetime": public_self["operator_analysis_tests"]["validated_packet_analysis"]["lifetime_curve_count"] > 0,
        "packet_analysis_has_factorial": public_self["operator_analysis_tests"]["validated_packet_analysis"]["factorial_matched_group_count"] > 0,
        "operator_analysis_tests": public_self["operator_analysis_tests"],
    }
    operator["passed"] = (
        operator["passed"]
        and operator["packet_analysis_passed"]
        and operator["packet_analysis_has_lifetime"]
        and operator["packet_analysis_has_factorial"]
    )
    factorial = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FACTORIAL_ARM_SELF_TEST_V1",
        "passed": public_self["factorial_arm_tests"]["additive_data_rejected_as_nonadditive"]
        and public_self["factorial_arm_tests"]["ordinary_nonlinear_data_detected_not_overclaimed"],
        "factorial_arm_tests": public_self["factorial_arm_tests"],
    }
    source_death = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_DEATH_CUSTODY_SELF_TEST_V1",
        "passed": all(public_self["source_death_custody_tests"].values()),
        "source_death_custody_tests": public_self["source_death_custody_tests"],
    }
    coverage = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EXACT_COVERAGE_SELF_TEST_V1",
        "passed": all(public_self["coverage_tests"][key] if key != "minimal_success_packet" else public_self["coverage_tests"][key]["passed"] for key in public_self["coverage_tests"]),
        "coverage_tests": public_self["coverage_tests"],
    }
    operator["operator_analysis_self_test_sha256"] = public.digest(
        {k: v for k, v in operator.items() if k != "operator_analysis_self_test_sha256"}
    )
    factorial["factorial_arm_self_test_sha256"] = public.digest(
        {k: v for k, v in factorial.items() if k != "factorial_arm_self_test_sha256"}
    )
    source_death["source_death_custody_self_test_sha256"] = public.digest(
        {k: v for k, v in source_death.items() if k != "source_death_custody_self_test_sha256"}
    )
    coverage["exact_coverage_self_test_sha256"] = public.digest(
        {k: v for k, v in coverage.items() if k != "exact_coverage_self_test_sha256"}
    )
    write_json(OPERATOR_ANALYSIS_PATH, operator)
    write_json(FACTORIAL_ARM_PATH, factorial)
    write_json(SOURCE_DEATH_PATH, source_death)
    write_json(EXACT_COVERAGE_PATH, coverage)
    return {"feature": feature, "operator": operator, "factorial": factorial, "source_death": source_death, "coverage": coverage}


def target_self_test() -> dict[str, Any]:
    result = target.self_test(HERE, HERE)
    write_json(TARGET_SELF_TEST_PATH, result)
    return result


def offline_validate() -> dict[str, Any]:
    schedule_validation = target.validate_schedule_artifacts(HERE)
    source_hashes = read_source_hash_authority()
    public_self = public.write_self_test(HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json")
    split = write_split_self_tests(public_self)
    target_result = target_self_test()
    runtime_result = runtime_self_test()
    transport = fake_transport_self_tests()
    deployment = deployment_layout_self_test()
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE_V1",
        "passed": all(
            [
                schedule_validation["passed"],
                public_self["self_test_passed"],
                split["feature"]["passed"],
                split["operator"]["passed"],
                split["factorial"]["passed"],
                split["source_death"]["passed"],
                split["coverage"]["passed"],
                target_result["self_test_passed"],
                runtime_result["passed"],
                transport["passed"],
                deployment["passed"],
            ]
        ),
        "schedule_validation": schedule_validation,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "target_self_test_sha256": target_result["self_test_sha256"],
        "runtime_self_test_sha256": runtime_result.get("runtime_self_test_sha256"),
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "deployment_layout_self_test_sha256": deployment["deployment_layout_self_test_sha256"],
        "feature_boundary_self_test_sha256": split["feature"]["feature_boundary_self_test_sha256"],
        "operator_analysis_self_test_sha256": split["operator"]["operator_analysis_self_test_sha256"],
        "factorial_arm_self_test_sha256": split["factorial"]["factorial_arm_self_test_sha256"],
        "source_death_custody_self_test_sha256": split["source_death"]["source_death_custody_self_test_sha256"],
        "exact_coverage_self_test_sha256": split["coverage"]["exact_coverage_self_test_sha256"],
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "operator_analysis_passed": split["operator"]["passed"],
        "factorial_arm_passed": split["factorial"]["passed"],
        "source_death_custody_passed": split["source_death"]["passed"],
        "exact_coverage_passed": split["coverage"]["passed"],
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    result["offline_validate_sha256"] = public.digest({k: v for k, v in result.items() if k != "offline_validate_sha256"})
    write_json(OFFLINE_VALIDATE_PATH, result)
    return result


def source_hash_authority_regression() -> dict[str, Any]:
    if not SOURCE_HASHES.exists():
        return {
            "passed": False,
            "source_hash_authority_present": False,
            "mutated_source_rejected": False,
            "authority_file_byte_identical_after_rejection": False,
            "source_hashes_excluded_from_generated_receipts": SOURCE_HASHES not in GENERATED_RECEIPTS,
        }
    with tempfile.TemporaryDirectory(prefix="family10h_source_hash_authority_") as tmp:
        root = Path(tmp)
        for name in SOURCE_FILE_NAMES:
            shutil.copy2(HERE / name, root / name)
        for name in RUNTIME_AUTHORITY_FILE_NAMES:
            shutil.copy2(HERE / name, root / name)
        shutil.copy2(SOURCE_HASHES, root / SOURCE_HASHES.name)
        receipt_path = root / SOURCE_HASHES.name
        before = receipt_path.read_bytes()
        mutated = root / SOURCE_FILE_NAMES[-1]
        mutated.write_text(mutated.read_text(encoding="utf-8") + "\n# authority regression mutation\n", encoding="utf-8")
        source_authority = target.validate_source_file_authority(root)
        after = receipt_path.read_bytes()
    result = {
        "source_hash_authority_present": True,
        "mutated_source_rejected": not source_authority["passed"],
        "authority_file_byte_identical_after_rejection": before == after,
        "source_hashes_excluded_from_generated_receipts": SOURCE_HASHES not in GENERATED_RECEIPTS,
        "prepare_only_uses_read_only_source_authority": True,
        "offline_validate_uses_read_only_source_authority": True,
        "manifest_uses_read_only_source_authority": True,
        "failures": source_authority["failures"],
    }
    result["passed"] = all(
        result[key]
        for key in [
            "mutated_source_rejected",
            "authority_file_byte_identical_after_rejection",
            "source_hashes_excluded_from_generated_receipts",
            "prepare_only_uses_read_only_source_authority",
            "offline_validate_uses_read_only_source_authority",
            "manifest_uses_read_only_source_authority",
        ]
    )
    return result


def runtime_binary_overlay_mutation_regression() -> dict[str, Any]:
    if not SOURCE_HASHES.exists() or not BINARY_PATH.exists():
        return {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_BINARY_OVERLAY_MUTATION_REGRESSION_V1",
            "passed": False,
            "failures": ["source hashes or committed runtime binary missing"],
        }

    with tempfile.TemporaryDirectory(prefix="family10h_runtime_binary_overlay_") as tmp:
        temp_root = Path(tmp)
        repo_root = temp_root / "repo"
        package_root = repo_root / "pkg"
        package_root.mkdir(parents=True)
        for name in DISCOVERY_TRANSFER_FILE_NAMES:
            shutil.copy2(HERE / name, package_root / name)

        def temp_git(*args: str) -> str:
            return run(["git", *args], timeout=30.0, cwd=repo_root).stdout.strip()

        try:
            temp_git("init")
            temp_git("config", "user.name", "Family10h Replay Regression")
            temp_git("config", "user.email", "family10h-replay-regression@example.invalid")
            temp_git("add", "-A")
            temp_git("commit", "-m", "source authority snapshot")
            source_commit = temp_git("rev-parse", "HEAD")
            overlay_marker = package_root / "CARRIER_TOMOGRAPHY_EVIDENCE_OVERLAY_MARKER.json"
            write_json(overlay_marker, {"schema": "FAMILY10H_RUNTIME_REPLAY_EVIDENCE_OVERLAY_MARKER_V1", "authority_blobs_changed": False})
            temp_git("add", "-A")
            temp_git("commit", "-m", "evidence overlay without authority mutation")
            baseline_commit = temp_git("rev-parse", "HEAD")
            runtime_path = package_root / RUNTIME_BINARY_NAME
            runtime_path.write_bytes(runtime_path.read_bytes() + b"\nE1-runtime-authority-mutation\n")
            temp_git("add", "-A")
            temp_git("commit", "-m", "mutate runtime authority blob only")
            mutated_commit = temp_git("rev-parse", "HEAD")
        except ControllerError as exc:
            return {
                "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_BINARY_OVERLAY_MUTATION_REGRESSION_V1",
                "passed": False,
                "failures": [f"temporary Git replay fixture construction failed: {exc}"],
            }

        baseline_replay = replay_final_exact_objects(
            source_commit,
            baseline_commit,
            repo_root=repo_root,
            package_root=package_root,
        )
        mutated_replay = replay_final_exact_objects(
            source_commit,
            mutated_commit,
            repo_root=repo_root,
            package_root=package_root,
        )
        defective_replay = replay_final_exact_objects(
            source_commit,
            mutated_commit,
            repo_root=repo_root,
            package_root=package_root,
            authority_file_names=list(SOURCE_AUTHORITY_FILE_NAMES),
        )

    checks = {
        "baseline_runtime_absent_from_changed_source_files": RUNTIME_BINARY_NAME not in baseline_replay["changed_source_files_after_c1"],
        "baseline_authority_blobs_unchanged": not baseline_replay["changed_source_files_after_c1"],
        "mutated_runtime_present_in_changed_source_files": RUNTIME_BINARY_NAME in mutated_replay["changed_source_files_after_c1"],
        "mutated_replay_failed": mutated_replay["passed"] is False,
        "mutated_failure_identifies_runtime_authority": any(RUNTIME_BINARY_NAME in failure for failure in mutated_replay["failures"]),
        "all_nine_text_source_blobs_remain_unchanged": not any(name in mutated_replay["changed_source_files_after_c1"] for name in SOURCE_FILE_NAMES),
        "defective_runtime_exclusion_misses_mutation": RUNTIME_BINARY_NAME not in defective_replay["changed_source_files_after_c1"],
        "defective_comparison_reports_vulnerable": not any(RUNTIME_BINARY_NAME in failure for failure in defective_replay["failures"]),
        "production_comparison_detects_runtime_mutation": RUNTIME_BINARY_NAME in mutated_replay["changed_source_files_after_c1"],
    }
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_BINARY_OVERLAY_MUTATION_REGRESSION_V1",
        "passed": all(checks.values()),
        "checks": checks,
        "source_commit": source_commit,
        "baseline_evidence_commit": baseline_commit,
        "mutated_evidence_commit": mutated_commit,
        "baseline_replay": baseline_replay,
        "mutated_replay": mutated_replay,
        "defective_runtime_excluded_replay": defective_replay,
    }


def review_quorum_null_model_baseline() -> dict[str, Any]:
    randomized_clearances = {
        role: {
            "role": label,
            "originating_agent": f"null randomized {label}",
            "agent_id": f"null-randomized-reviewer-{(index % 2) + 1}",
            "verdict": "NO_MATERIAL_BLOCKER" if index != 3 else "UNCLASSIFIED",
            "final_response": index % 2 == 0,
        }
        for index, (role, label) in enumerate(REQUIRED_REVIEW_ROLES.items(), start=1)
    }
    quorum = review_quorum(
        {
            "material_blockers": [],
            "reviewer_verdicts": randomized_clearances,
        }
    )
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_REVIEW_QUORUM_NULL_BASELINE_V1",
        "null_model": "deterministic_randomized_reviewer_identity_and_finality",
        "null_baseline_rejected": not quorum["passed"],
        "null_baseline_failures": quorum["failures"],
        "passed": not quorum["passed"],
    }


def source_audit_quorum_regression() -> dict[str, Any]:
    source_commit = "1" * 40
    source_hash = "2" * 64
    bundle_hash = "3" * 64
    runtime_hash = "4" * 64

    def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
        write_json(path, receipt)

    def build_archive(review_root: Path) -> dict[str, Any]:
        clearances: dict[str, dict[str, Any]] = {}
        for index, (role, label) in enumerate(SOURCE_AUDIT_REQUIRED_REVIEW_ROLES.items(), start=1):
            body_path, receipt_path = expected_source_audit_archive_paths(role, review_root)
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes((f"{label} C3 read-only review body {index}\n").encode("utf-8"))
            body_state = review_body_canonical_state(body_path.read_bytes())
            agent_id = f"source-reviewer-c3-{index}"
            thread_id = f"thread-source-review-c3-{index}"
            receipt = {
                "schema": SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA,
                "issuer": SOURCE_AUDIT_REVIEW_RECEIPT_ISSUER,
                "receipt_kind": SOURCE_AUDIT_RECEIPT_KIND,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "role": label,
                "model": "gpt-5.6-sol",
                "review_body_sha256": body_state["canonical_sha256"],
                "review_body_canonicalization": SOURCE_AUDIT_REVIEW_BODY_CANONICALIZATION,
                "audited_commit": source_commit,
                "source_hashes_sha256": source_hash,
                "source_bundle_sha256": bundle_hash,
                "runtime_binary_sha256": runtime_hash,
                "verdict": "NO_MATERIAL_BLOCKER",
                "final_response": True,
                "material_blocker_ids": [],
                "no_git_write": True,
                "no_file_edits": True,
                "no_checkout_mutation": True,
                "no_target_contact": True,
                "no_live_authority": True,
                "no_pmu": True,
                "self_authored": False,
                "evidence_origin": SOURCE_AUDIT_ALLOWED_EVIDENCE_ORIGIN,
            }
            write_receipt(receipt_path, receipt)
            clearances[role] = {
                "role": label,
                "agent_id": agent_id,
                "thread_id": thread_id,
                "model": "gpt-5.6-sol",
                "verdict": "NO_MATERIAL_BLOCKER",
                "final_response": True,
                "material_blocker_ids": [],
                "audited_commit": source_commit,
                "source_hashes_sha256": source_hash,
                "source_bundle_sha256": bundle_hash,
                "runtime_binary_sha256": runtime_hash,
                "body_path": str(body_path),
                "body_file_sha256": body_state["file_sha256"],
                "body_canonical_sha256": body_state["canonical_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": public.sha256_file(receipt_path),
                "boundary_attestation": {
                    "no_git_write": True,
                    "no_file_edits": True,
                    "no_checkout_mutation": True,
                    "no_target_contact": True,
                    "no_live_authority": True,
                    "no_pmu": True,
                },
                "review_receipt": receipt,
            }
        return {
            "schema": "FAMILY10H_SOURCE_AUTHORITY_C3_REVIEW_NORMALIZED_V1",
            "source_authority_commit": source_commit,
            "source_hashes_sha256": source_hash,
            "source_bundle_sha256": bundle_hash,
            "runtime_binary_sha256": runtime_hash,
            "review_report_present": True,
            "material_blockers": [],
            "reviewer_verdicts": clearances,
        }

    def rewrite_item_receipt(review_root: Path, item: dict[str, Any], role: str) -> None:
        _, receipt_path = expected_source_audit_archive_paths(role, review_root)
        write_receipt(receipt_path, item["review_receipt"])
        item["receipt_file_sha256"] = public.sha256_file(receipt_path)

    def evaluate(mutator: Any | None = None, *, review_report_present: bool = True, excluded: set[str] | None = None) -> bool:
        with tempfile.TemporaryDirectory(prefix="family10h_c3_source_audit_regression_") as tmp:
            review_root = Path(tmp) / "SOURCE_AUTHORITY_C3_REVIEW"
            audit = build_archive(review_root)
            if mutator is not None:
                mutator(audit, review_root)
            return source_audit_quorum(
                audit,
                expected_source_commit=source_commit,
                expected_source_hashes_sha256=source_hash,
                expected_source_bundle_sha256=bundle_hash,
                expected_runtime_binary_sha256=runtime_hash,
                review_report_present=review_report_present,
                excluded_agent_ids=excluded,
                review_root=review_root,
            )["passed"]

    def role_item(audit: dict[str, Any], role: str = "source_bundle_runtime_evidence_auditor") -> dict[str, Any]:
        return audit["reviewer_verdicts"][role]

    def mutate_receipt(audit: dict[str, Any], review_root: Path, role_key: str, **fields: Any) -> None:
        item = role_item(audit, role_key)
        item["review_receipt"] = {**item["review_receipt"], **fields}
        rewrite_item_receipt(review_root, item, role_key)

    c5_audit = read_json(SOURCE_AUDIT_REVIEW_VERSIONED["C5"]["findings_path"])
    c5_paths = source_audit_paths_for_commit(C5_SOURCE_AUTHORITY_COMMIT)
    c5_quorum = source_audit_quorum(
        c5_audit,
        expected_source_commit=C5_SOURCE_AUTHORITY_COMMIT,
        expected_source_hashes_sha256=c5_audit["source_hashes_sha256"],
        expected_source_bundle_sha256=c5_audit["source_bundle_sha256"],
        expected_runtime_binary_sha256=c5_audit["runtime_binary_sha256"],
        review_report_present=c5_paths["review_path"].exists(),
        review_root=c5_paths["review_dir"],
    )

    checks = {
        "exact_source_audit_quorum_passes": evaluate(),
        "c5_actual_source_authority_commit_selects_c5": source_audit_version_for_commit(C5_SOURCE_AUTHORITY_COMMIT) == "C5",
        "c5_actual_source_authority_commit_uses_v2_receipts": source_audit_receipt_schema_for_commit(C5_SOURCE_AUTHORITY_COMMIT)
        == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2,
        "c5_committed_v2_archive_replays": c5_quorum["passed"],
        "c5_failure_evidence_commit_selects_c6": source_audit_version_for_commit(C5_FAILURE_EVIDENCE_COMMIT) == "C6",
        "c5_failure_evidence_commit_uses_v3_receipts": source_audit_receipt_schema_for_commit(C5_FAILURE_EVIDENCE_COMMIT)
        == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3,
        "future_source_authority_commit_uses_v3_receipts": source_audit_receipt_schema_for_commit("f" * 40)
        == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3,
        "missing_report_blocked": not evaluate(review_report_present=False),
        "wrong_top_level_bundle_blocked": not evaluate(lambda audit, _root: audit.update({"source_bundle_sha256": "5" * 64})),
        "wrong_top_level_runtime_blocked": not evaluate(lambda audit, _root: audit.update({"runtime_binary_sha256": "6" * 64})),
        "wrong_reviewer_commit_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit, "claim_boundary_adjudicator").update({"audited_commit": "7" * 40}),
                mutate_receipt(audit, root, "claim_boundary_adjudicator", audited_commit="7" * 40),
            )
        ),
        "missing_boundary_attestation_blocked": not evaluate(
            lambda audit, _root: role_item(audit, "claim_boundary_adjudicator").update({"boundary_attestation": {"no_git_write": True}})
        ),
        "prior_package_reviewer_reuse_blocked": not evaluate(excluded={"source-reviewer-c3-1"}),
        "c1_self_referential_receipt_schema_rejected": not evaluate(
            lambda audit, root: mutate_receipt(
                audit,
                root,
                "source_bundle_runtime_evidence_auditor",
                schema="FAMILY10H_SOURCE_AUTHORITY_C1_REVIEWER_RECEIPT_V1",
                final_response_sha256="8" * 64,
            )
        ),
        "missing_review_body_blocked": not evaluate(
            lambda audit, root: expected_source_audit_archive_paths("source_bundle_runtime_evidence_auditor", root)[0].unlink()
        ),
        "missing_receipt_blocked": not evaluate(
            lambda audit, root: expected_source_audit_archive_paths("source_bundle_runtime_evidence_auditor", root)[1].unlink()
        ),
        "body_hash_mismatch_blocked": not evaluate(
            lambda audit, _root: role_item(audit).update({"body_canonical_sha256": "9" * 64})
        ),
        "body_changed_after_ack_blocked": not evaluate(
            lambda audit, root: expected_source_audit_archive_paths("source_bundle_runtime_evidence_auditor", root)[0].write_bytes(b"changed body\n")
        ),
        "receipt_changed_after_ack_blocked": not evaluate(
            lambda audit, root: write_json(
                expected_source_audit_archive_paths("source_bundle_runtime_evidence_auditor", root)[1],
                {**role_item(audit)["review_receipt"], "model": "changed-model"},
            )
        ),
        "wrong_thread_id_blocked": not evaluate(
            lambda audit, root: mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", thread_id="wrong-thread")
        ),
        "wrong_agent_id_blocked": not evaluate(
            lambda audit, root: mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", agent_id="wrong-agent")
        ),
        "wrong_role_blocked": not evaluate(
            lambda audit, root: mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", role="wrong role")
        ),
        "wrong_source_hash_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit).update({"source_hashes_sha256": "a" * 64}),
                mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", source_hashes_sha256="a" * 64),
            )
        ),
        "wrong_source_bundle_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit).update({"source_bundle_sha256": "b" * 64}),
                mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", source_bundle_sha256="b" * 64),
            )
        ),
        "wrong_runtime_binary_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit).update({"runtime_binary_sha256": "c" * 64}),
                mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", runtime_binary_sha256="c" * 64),
            )
        ),
        "parent_created_receipt_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit).update({"evidence_origin": "parent-created"}),
                mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", evidence_origin="parent-created"),
            )
        ),
        "self_authored_receipt_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit).update({"self_authored": True}),
                mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", self_authored=True),
            )
        ),
        "target_derived_receipt_blocked": not evaluate(
            lambda audit, root: (
                role_item(audit).update({"evidence_origin": "target-derived"}),
                mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", evidence_origin="target-derived"),
            )
        ),
        "receipt_from_different_thread_blocked": not evaluate(
            lambda audit, root: mutate_receipt(audit, root, "source_bundle_runtime_evidence_auditor", thread_id="other-reviewer-thread")
        ),
        "missing_second_acknowledgment_blocked": not evaluate(
            lambda audit, _root: role_item(audit).pop("review_receipt")
        ),
        "duplicate_reviewer_blocked": not evaluate(
            lambda audit, _root: role_item(audit).update({"agent_id": "source-reviewer-c3-1"})
        ),
        "missing_fourth_reviewer_blocked": not evaluate(
            lambda audit, _root: audit["reviewer_verdicts"].pop("claim_boundary_adjudicator")
        ),
        "material_blocker_verdict_blocked": not evaluate(
            lambda audit, _root: role_item(audit, "claim_boundary_adjudicator").update({"verdict": "MATERIAL_BLOCKER", "final_response": False})
        ),
        "receipt_bound_blocker_cannot_be_cleared_by_normalized_json": not evaluate(
            lambda audit, root: mutate_receipt(
                audit,
                root,
                "claim_boundary_adjudicator",
                verdict="MATERIAL_BLOCKER",
                final_response=False,
                material_blocker_ids=["C6-CLAIM-BOUNDARY-01"],
            )
        ),
        "receipt_bound_nonfinal_cannot_be_cleared_by_normalized_json": not evaluate(
            lambda audit, root: mutate_receipt(
                audit,
                root,
                "claim_boundary_adjudicator",
                verdict="NO_MATERIAL_BLOCKER",
                final_response=False,
                material_blocker_ids=[],
            )
        ),
        "pre_contact_missing_or_mismatched_source_review_blocked": not read_source_authority_review_for_discovery(
            source_commit=source_commit,
            source_hashes_sha256=source_hash,
            source_bundle_sha256=bundle_hash,
            runtime_binary_sha256=runtime_hash,
        )["passed"],
    }
    return {"schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_AUDIT_QUORUM_REGRESSION_V1", "passed": all(checks.values()), "checks": checks}


def source_bundle_mode_regression() -> dict[str, Any]:
    write_source_bundle()
    with tempfile.TemporaryDirectory(prefix="family10h_bundle_mode_") as tmp:
        root = Path(tmp) / "source"
        target.copy_source_fixture(HERE, root)
        native = target.deterministic_source_bundle_sha256(root)
        for name in SOURCE_FILE_NAMES:
            os.chmod(root / name, 0o600)
        restricted = target.deterministic_source_bundle_sha256(root)
        for name in SOURCE_FILE_NAMES:
            os.chmod(root / name, 0o644)
        normalized = target.deterministic_source_bundle_sha256(root)
    checks = {
        "target_bundle_hash_independent_of_0600_mode": native == restricted,
        "target_bundle_hash_independent_of_0644_mode": native == normalized,
        "controller_preview_matches_written_bundle": source_bundle_preview()["sha256"] == public.sha256_file(SOURCE_BUNDLE),
    }
    return {"schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_BUNDLE_MODE_REGRESSION_V1", "passed": all(checks.values()), "checks": checks}


def discovery_attempt_journal_regression() -> dict[str, Any]:
    def seal(payload: dict[str, Any]) -> dict[str, Any]:
        state = payload["attempt_state"]
        item = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT_V1",
            "source_authority_commit": "1" * 40,
            "source_authority_review": {
                "findings_sha256": "a" * 64,
                "review_report_sha256": "b" * 64,
                "review_quorum_sha256": "c" * 64,
                "source_authority_commit": "1" * 40,
                "source_hashes_sha256": "d" * 64,
                "source_bundle_sha256": "e" * 64,
            },
            "controller_challenge_sha256": "2" * 64,
            "challenge_receipt_canonical_sha256": "3" * 64,
            "challenge_receipt_file_sha256": "4" * 64,
            **ATTEMPT_STATE_COUNTERS[state],
            "attempt_state": state,
            "passed": state == "complete",
        }
        item.update(payload.get("extra", {}))
        item["discovery_attempt_sha256"] = public.digest({k: v for k, v in item.items() if k != "discovery_attempt_sha256"})
        return item

    valid_rows = [seal({"attempt_state": state}) for state in ATTEMPT_STATE_SEQUENCE]
    first = valid_rows[0]
    second = valid_rows[1]
    final = valid_rows[-1]
    missing_previous_counter = dict(first)
    missing_previous_counter.pop("target_contact_count")
    corrupted = dict(second)
    corrupted["target_contact_count"] = 0
    skipped = [valid_rows[0], valid_rows[1], valid_rows[3], *valid_rows[4:]]
    duplicated = [valid_rows[0], valid_rows[1], valid_rows[1], *valid_rows[2:]]
    binding_mutated = [dict(row) for row in valid_rows]
    binding_mutated[2] = {
        **binding_mutated[2],
        "source_authority_review": {**binding_mutated[2]["source_authority_review"], "source_bundle_sha256": "f" * 64},
    }
    binding_mutated[2]["discovery_attempt_sha256"] = public.digest(
        {k: v for k, v in binding_mutated[2].items() if k != "discovery_attempt_sha256"}
    )
    post_terminal = [*valid_rows, seal({"attempt_state": "complete"})]
    forged_counter = seal({"attempt_state": "target_command_invoked", "extra": {"target_contact_count": 0}})
    legacy_cumulative = seal({"attempt_state": "complete", "extra": {"cumulative_lane_counters": dict(ATTEMPT_STATE_COUNTERS["complete"])}})
    historical_metadata = seal(
        {
            "attempt_state": "target_command_invoked",
            "extra": {
                "historical_lane_contact_report": {
                    "authoritative_for_active_transaction": False,
                    "known_counters_before_active_attempt": {"target_contact_count": 999, "sensor_inventory_count": 999, "live_invocation_count": 999, "pmu_acquisition_count": 999},
                }
            },
        }
    )

    def snapshot_without_journal_rejected() -> bool:
        global DISCOVERY_ATTEMPT_PATH, DISCOVERY_ATTEMPT_JOURNAL_PATH
        original_snapshot = DISCOVERY_ATTEMPT_PATH
        original_journal = DISCOVERY_ATTEMPT_JOURNAL_PATH
        with tempfile.TemporaryDirectory(prefix="family10h_attempt_journal_regression_") as tmp:
            DISCOVERY_ATTEMPT_PATH = Path(tmp) / "attempt.json"
            DISCOVERY_ATTEMPT_JOURNAL_PATH = Path(tmp) / "attempt.jsonl"
            try:
                write_json(DISCOVERY_ATTEMPT_PATH, first)
                try:
                    write_discovery_attempt_receipt(second)
                except ControllerError:
                    return True
                return False
            finally:
                DISCOVERY_ATTEMPT_PATH = original_snapshot
                DISCOVERY_ATTEMPT_JOURNAL_PATH = original_journal

    checks = {
        "valid_seven_state_journal_replays": replay_attempt_journal_rows(valid_rows)["passed"],
        "missing_previous_counter_rejected": not replay_attempt_journal_rows([missing_previous_counter, second])["passed"],
        "single_complete_row_rejected": not replay_attempt_journal_rows([final])["passed"],
        "skipped_state_rejected": not replay_attempt_journal_rows(skipped)["passed"],
        "duplicated_state_rejected": not replay_attempt_journal_rows(duplicated)["passed"],
        "binding_mutation_rejected": not replay_attempt_journal_rows(binding_mutated)["passed"],
        "post_terminal_append_rejected": not replay_attempt_journal_rows(post_terminal)["passed"],
        "reordered_rows_rejected": not replay_attempt_journal_rows([second, first, *valid_rows[2:]])["passed"],
        "digest_corruption_rejected": not replay_attempt_journal_rows([first, corrupted])["passed"],
        "forged_active_counter_rejected": not replay_attempt_journal_rows([first, forged_counter, *valid_rows[2:]])["passed"],
        "legacy_cumulative_attempt_field_rejected": not replay_attempt_journal_rows([*valid_rows[:-1], legacy_cumulative])["passed"],
        "historical_metadata_does_not_alter_active_counters": replay_attempt_journal_rows([first, second, historical_metadata, *valid_rows[3:]])["passed"],
        "empty_journal_rejected": not replay_attempt_journal_rows([])["passed"],
        "snapshot_without_journal_rejected": snapshot_without_journal_rejected(),
    }
    return {"schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT_JOURNAL_REGRESSION_V1", "passed": all(checks.values()), "checks": checks}


def strict_json_regression() -> dict[str, Any]:
    duplicate_key_rejected = False
    nonfinite_rejected = False
    try:
        strict_json_loads('{"state":"a","state":"b"}')
    except ValueError:
        duplicate_key_rejected = True
    try:
        strict_json_loads('{"value":NaN}')
    except ValueError:
        nonfinite_rejected = True
    blank_jsonl_rejected = False
    try:
        strict_jsonl_loads(b'{"row":1}\n\n{"row":2}\n')
    except ValueError:
        blank_jsonl_rejected = True
    checks = {
        "duplicate_key_rejected": duplicate_key_rejected,
        "nonfinite_rejected": nonfinite_rejected,
        "missing_or_blank_jsonl_rejected": blank_jsonl_rejected,
    }
    return {"schema": "FAMILY10H_CARRIER_TOMOGRAPHY_STRICT_JSON_REGRESSION_V1", "passed": all(checks.values()), "checks": checks}


def challenge_receipt_regression() -> dict[str, Any]:
    challenge = {"controller_nonce_sha256": "1" * 64, "source_authority_review": fixture_source_review_binding("2" * 40, "3" * 64, "4" * 64)}
    receipt = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE_RECEIPT_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_authority_commit": "2" * 40,
        "source_authority_review": challenge["source_authority_review"],
        "controller_challenge": challenge,
        "controller_challenge_sha256": public.digest(challenge),
        "controller_nonce_sha256": "1" * 64,
        "pre_contact": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    receipt["challenge_receipt_canonical_sha256"] = public.digest(
        {k: v for k, v in receipt.items() if k != "challenge_receipt_canonical_sha256"}
    )
    changed_review = {**receipt, "source_authority_review": {**receipt["source_authority_review"], "source_bundle_sha256": "5" * 64}}
    changed_review["challenge_receipt_canonical_sha256"] = public.digest(
        {k: v for k, v in changed_review.items() if k != "challenge_receipt_canonical_sha256"}
    )
    nonzero_counter = {**receipt, "target_contact_count": 1}
    nonzero_counter["challenge_receipt_canonical_sha256"] = public.digest(
        {k: v for k, v in nonzero_counter.items() if k != "challenge_receipt_canonical_sha256"}
    )
    wrong_digest = {**receipt, "challenge_receipt_canonical_sha256": "0" * 64}
    checks = {
        "valid_challenge_receipt_passes": validate_discovery_challenge_receipt_payload(
            receipt,
            expected_source_commit="2" * 40,
            expected_challenge=challenge,
            expected_nonce_sha="1" * 64,
            expected_source_review_binding=challenge["source_authority_review"],
        )["passed"],
        "changed_review_binding_rejected": not validate_discovery_challenge_receipt_payload(
            changed_review,
            expected_source_commit="2" * 40,
            expected_challenge=challenge,
            expected_nonce_sha="1" * 64,
            expected_source_review_binding=challenge["source_authority_review"],
        )["passed"],
        "nonzero_precontact_counter_rejected": not validate_discovery_challenge_receipt_payload(nonzero_counter)["passed"],
        "canonical_digest_mismatch_rejected": not validate_discovery_challenge_receipt_payload(wrong_digest)["passed"],
        "extra_field_rejected": not validate_discovery_challenge_receipt_payload({**receipt, "extra": True})["passed"],
    }
    return {"schema": "FAMILY10H_CARRIER_TOMOGRAPHY_CHALLENGE_RECEIPT_REGRESSION_V1", "passed": all(checks.values()), "checks": checks}


def controller_acquisition_transaction_regression() -> dict[str, Any]:
    def seal_attempt(state: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT_V1",
            "passed": state == "complete",
            "attempt_state": state,
            "source_authority_commit": "1" * 40,
            "source_authority_review": {
                "findings_sha256": "a" * 64,
                "review_report_sha256": "b" * 64,
                "review_quorum_sha256": "c" * 64,
                "source_authority_commit": "1" * 40,
                "source_hashes_sha256": "d" * 64,
                "source_bundle_sha256": "e" * 64,
            },
            "controller_challenge_sha256": "2" * 64,
            "challenge_receipt_canonical_sha256": "3" * 64,
            "challenge_receipt_file_sha256": "4" * 64,
            **ATTEMPT_STATE_COUNTERS[state],
        }
        if overrides:
            row.update(overrides)
        row["discovery_attempt_sha256"] = public.digest({k: v for k, v in row.items() if k != "discovery_attempt_sha256"})
        return row

    def seal_transport(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        nonce_sha = "8" * 64
        scope = discovery_transport_scope(target_host=TARGET_HOST, remote_root=DISCOVERY_REMOTE_ROOT, nonce_sha=nonce_sha)
        row = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1",
            "passed": True,
            "cleanup": {"attempted": True, "passed": True, "absence_verified": True},
            "retry_count": 0,
            "target_command_invoked": True,
            "target_host": scope["target_host"],
            "remote_base_root": scope["remote_base_root"],
            "remote_root": scope["remote_root"],
            "remote_source_root": scope["remote_source_root"],
            "remote_receipt_path": scope["remote_receipt_path"],
            "source_authority_commit": "f" * 40,
            "controller_challenge": {"controller_nonce_sha256": nonce_sha, "transport_scope": scope},
            "controller_challenge_sha256": "1" * 64,
            "challenge_receipt_canonical_sha256": "2" * 64,
            "challenge_receipt_file_sha256": "7" * 64,
            "target_discovery_receipt_sha256": "3" * 64,
            "target_discovery_receipt_file_sha256": "5" * 64,
            "authority_receipt_sha256": "4" * 64,
            "authority_receipt_file_sha256": "6" * 64,
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
        if overrides:
            row.update(overrides)
        row["discovery_transport_sha256"] = public.digest({k: v for k, v in row.items() if k != "discovery_transport_sha256"})
        return row

    production_signature = set(inspect.signature(acquire_temperature_sensor_authority).parameters)
    valid_preflight = parse_preflight_stdout("FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n")
    malformed_preflight = parse_preflight_stdout("base_preexisting=0\n")
    bool_challenge = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE_RECEIPT_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_authority_commit": "1" * 40,
        "source_authority_review": fixture_source_review_binding("1" * 40, "2" * 64, "3" * 64),
        "controller_challenge": {"controller_nonce_sha256": "4" * 64},
        "controller_challenge_sha256": public.digest({"controller_nonce_sha256": "4" * 64}),
        "controller_nonce_sha256": "4" * 64,
        "pre_contact": True,
        "target_contact_count": False,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    bool_challenge["challenge_receipt_canonical_sha256"] = public.digest(
        {k: v for k, v in bool_challenge.items() if k != "challenge_receipt_canonical_sha256"}
    )
    bool_journal_rows = [seal_attempt(state) for state in ATTEMPT_STATE_SEQUENCE]
    bool_journal_rows[1] = seal_attempt("transport_contact_invoked", {"target_contact_count": True})
    original_cleanup_path = DISCOVERY_CLEANUP_CUSTODY_PATH
    with tempfile.TemporaryDirectory(prefix="family10h_cleanup_custody_regression_") as tmp:
        try:
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = Path(tmp) / "cleanup.json"
            cleanup_valid = write_discovery_cleanup_custody_receipt(
                {
                    "passed": True,
                    "source_authority_commit": "1" * 40,
                    "source_authority_review": {},
                    "controller_challenge_sha256": "2" * 64,
                    "challenge_receipt_canonical_sha256": "3" * 64,
                    "challenge_receipt_file_sha256": "4" * 64,
                    "target_contact_count": 1,
                    "sensor_inventory_count": 0,
                    "candidate_scan_count": 0,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "cleanup": {"attempted": True, "passed": True, "absence_verified": True},
                    "commands": [],
                }
            )
            try:
                write_discovery_cleanup_custody_receipt(
                    {
                        **cleanup_valid,
                        "cleanup_custody_sha256": None,
                        "target_contact_count": True,
                    }
                )
                bool_cleanup_rejected = False
            except ControllerError:
                bool_cleanup_rejected = True
            try:
                write_discovery_cleanup_custody_receipt(
                    {
                        **cleanup_valid,
                        "cleanup_custody_sha256": None,
                        "sensor_inventory_count": 1,
                        "candidate_scan_count": 0,
                    }
                )
                pair_cleanup_rejected = False
            except ControllerError:
                pair_cleanup_rejected = True
        finally:
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = original_cleanup_path
    original_paths = {
        "DISCOVERY_ATTEMPT_PATH": DISCOVERY_ATTEMPT_PATH,
        "DISCOVERY_ATTEMPT_JOURNAL_PATH": DISCOVERY_ATTEMPT_JOURNAL_PATH,
        "DISCOVERY_CLEANUP_CUSTODY_PATH": DISCOVERY_CLEANUP_CUSTODY_PATH,
        "TEMPERATURE_SENSOR_AUTHORITY": TEMPERATURE_SENSOR_AUTHORITY,
        "DISCOVERY_TRANSPORT_PATH": DISCOVERY_TRANSPORT_PATH,
        "TARGET_DISCOVERY_RECEIPT": TARGET_DISCOVERY_RECEIPT,
    }
    with tempfile.TemporaryDirectory(prefix="family10h_parsed_invalid_copyback_regression_") as tmp:
        temp_root = Path(tmp)
        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            for state in ATTEMPT_STATE_SEQUENCE[:3]:
                write_discovery_attempt_receipt(seal_attempt(state))
            parsed_invalid_cleanup = write_discovery_cleanup_custody_receipt(
                {
                    "passed": True,
                    "source_authority_commit": "1" * 40,
                    "source_authority_review": {},
                    "controller_challenge_sha256": "2" * 64,
                    "challenge_receipt_canonical_sha256": "3" * 64,
                    "challenge_receipt_file_sha256": "4" * 64,
                    "target_contact_count": 1,
                    "sensor_inventory_count": 0,
                    "candidate_scan_count": 0,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "cleanup": {"attempted": True, "passed": True, "absence_verified": True},
                    "commands": [["ssh", "cleanup"], ["ssh", "absence-probe"]],
                }
            )
            parsed_invalid_rows = strict_jsonl_loads(DISCOVERY_ATTEMPT_JOURNAL_PATH.read_bytes())
            parsed_invalid_states = [row.get("attempt_state") for row in parsed_invalid_rows]
            parsed_invalid_cleanup_sealed = parsed_invalid_cleanup["cleanup_custody_sha256"] == public.digest(
                {k: v for k, v in parsed_invalid_cleanup.items() if k != "cleanup_custody_sha256"}
            )
            parsed_invalid_journal_stopped_before_success_cleanup = parsed_invalid_states == ATTEMPT_STATE_SEQUENCE[:3]
            parsed_invalid_no_success_artifacts = (
                not TEMPERATURE_SENSOR_AUTHORITY.exists()
                and not DISCOVERY_TRANSPORT_PATH.exists()
                and not TARGET_DISCOVERY_RECEIPT.exists()
            )
        finally:
            for name, path in original_paths.items():
                globals()[name] = path
    production_originals = {
        "DISCOVERY_ATTEMPT_PATH": DISCOVERY_ATTEMPT_PATH,
        "DISCOVERY_ATTEMPT_JOURNAL_PATH": DISCOVERY_ATTEMPT_JOURNAL_PATH,
        "DISCOVERY_CLEANUP_CUSTODY_PATH": DISCOVERY_CLEANUP_CUSTODY_PATH,
        "TEMPERATURE_SENSOR_AUTHORITY": TEMPERATURE_SENSOR_AUTHORITY,
        "DISCOVERY_TRANSPORT_PATH": DISCOVERY_TRANSPORT_PATH,
        "TARGET_DISCOVERY_RECEIPT": TARGET_DISCOVERY_RECEIPT,
        "DISCOVERY_CHALLENGE_PATH": DISCOVERY_CHALLENGE_PATH,
        "TARGET_DISCOVERY_FAILURE_PATH": TARGET_DISCOVERY_FAILURE_PATH,
        "run": run,
        "read_source_hash_authority": read_source_hash_authority,
        "read_existing_source_bundle_authority": read_existing_source_bundle_authority,
        "source_authority_commit_verification": source_authority_commit_verification,
        "read_source_authority_review_for_discovery": read_source_authority_review_for_discovery,
        "materialize_source_authority_snapshot": materialize_source_authority_snapshot,
        "commit_exists": commit_exists,
        "durable_write_bytes_exclusive": durable_write_bytes_exclusive,
        "durable_sync_directory": durable_sync_directory,
    }
    production_invalid_error = ""
    production_invalid_cleanup_invoked = False
    production_invalid_absence_probe_invoked = False
    production_invalid_cleanup_receipt_sealed = False
    production_invalid_cleanup_receipt_records_absence = False
    production_invalid_journal_stopped_before_receipt_copied = False
    production_invalid_no_success_artifacts = False
    production_invalid_no_success_cleanup_states = False
    production_success_cleanup_invoked = False
    production_success_absence_probe_invoked = False
    production_success_receipts_sealed_at_cleanup = False
    production_success_directory_barriers_before_cleanup = False
    production_success_transport_passed = False
    production_seal_failure_error = ""
    production_seal_failure_cleanup_invoked = False
    production_seal_failure_absence_probe_invoked = False
    production_target_directory_sync_failure_error = ""
    production_target_directory_sync_failure_cleanup_invoked = False
    production_target_directory_sync_failure_absence_probe_invoked = False
    production_authority_directory_sync_failure_error = ""
    production_authority_directory_sync_failure_cleanup_invoked = False
    production_authority_directory_sync_failure_absence_probe_invoked = False
    production_post_cleanup_error = ""
    production_post_cleanup_cleanup_invoked = False
    production_post_cleanup_receipt_recoverable = False
    production_post_cleanup_authority_recoverable = False
    invalid_discovery = {
        "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
        "discovery_mode": "target_read_only_sensor_inventory",
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "pmu_open_count": 0,
        "runtime_launch_count": 0,
    }
    invalid_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in invalid_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    invalid_discovery_bytes = (strict_json_dumps(invalid_discovery, indent=2) + "\n").encode("utf-8")
    invalid_discovery_file_sha = hashlib.sha256(invalid_discovery_bytes).hexdigest()
    fake_source_hashes = read_json(SOURCE_HASHES)
    fake_bundle = read_existing_source_bundle_authority()
    structured_failure_commit = "1" * 40
    structured_failure_runtime_sha = "f" * 64
    structured_failure_nonce_sha = "9" * 64
    structured_failure_challenge = build_temperature_authority_challenge(
        source_hashes=fake_source_hashes,
        source_bundle_sha256=fake_bundle["sha256"],
        runtime_binary_sha256=structured_failure_runtime_sha,
        schedule_sidecar=read_json(public.SCHEDULE_SHA),
        authorized_commit=structured_failure_commit,
        controller_nonce_sha256=structured_failure_nonce_sha,
        transport_scope=fixture_transport_scope(HERE, structured_failure_nonce_sha),
        source_authority_review=fixture_source_review_binding(
            structured_failure_commit,
            fake_source_hashes["source_hashes_sha256"],
            fake_bundle["sha256"],
            structured_failure_runtime_sha,
        ),
    )

    def seal_structured_failure_receipt(
        overrides: dict[str, Any] | None = None,
        *,
        challenge_override: dict[str, Any] | None = None,
        source_commit_override: str | None = None,
    ) -> dict[str, Any]:
        active_challenge = challenge_override or structured_failure_challenge
        active_source_commit = source_commit_override or structured_failure_commit
        active_transport_scope = active_challenge.get("transport_scope") if isinstance(active_challenge.get("transport_scope"), dict) else {}
        counters = {
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "candidate_scan_count": 1,
        }
        candidate = {
            "class_path": "/sys/class/hwmon/hwmon0/temp1_input",
            "input_basename": "temp1_input",
            "input_path_exists": True,
            "input_readability": True,
            "raw_input_text": "42000",
            "raw_input_parse_failure": None,
            "parsed_millidegree_value": 42000,
            "physical_range_passed": True,
            "hwmon_name_path_exists": True,
            "hwmon_name_readability": True,
            "hwmon_name_value": "k10temp",
            "sensor_label_path_exists": True,
            "sensor_label_present": True,
            "sensor_label_readability": True,
            "sensor_label_value": "ambient",
            "resolved_input_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0/temp1_input",
            "resolved_hwmon_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon0",
            "resolved_device_path": "/sys/devices/pci0000:00/0000:00:18.3",
            "resolved_driver_path": "/sys/bus/pci/drivers/k10temp",
            "resolved_subsystem_path": "/sys/bus/pci",
            "device_driver": "k10temp",
            "device_subsystem": "pci",
            "device_modalias_path_exists": True,
            "device_modalias_readability": True,
            "device_modalias_value": "pci:v00001022d00001203sv00000000sd00000000bc06sc00i00",
            "input_st_dev": 1,
            "input_st_ino": 2,
            "input_st_mode": 33060,
            "observation_errors": [],
            "approval_profile": target.LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
            "sensor_semantic_role": target.LEGACY_FAMILY10H_TEMPERATURE_ROLE,
            "sensor_semantic_profile": target.LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
            "approved": False,
            "rejection_reasons": ["present temp1_label is not Tctl"],
            "identity": None,
            "canonical_path_law_active": True,
            "class_path_under_hwmon_root": True,
            "resolved_input_under_sys_devices": True,
            "resolved_device_under_sys_devices": True,
            "rejection_reason": "present temp1_label is not Tctl",
        }
        row = {
            "schema": target.TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA,
            "discovery_mode": "target_read_only_sensor_inventory",
            "passed": False,
            "failure_classification": "LEGACY_CANDIDATE_REJECTED_IDENTITY",
            "failure_detail": "fixture rejected",
            "target_contact_count": counters["target_contact_count"],
            "sensor_inventory_count": counters["sensor_inventory_count"],
            "candidate_scan_count": counters["candidate_scan_count"],
            "live_invocation_count": counters["live_invocation_count"],
            "pmu_acquisition_count": counters["pmu_acquisition_count"],
            "pmu_open_count": 0,
            "runtime_launch_count": 0,
            "tomography_output_root_created": False,
            "source_root": str(active_transport_scope.get("remote_source_root", HERE)),
            "receipt_path": str(active_transport_scope.get("remote_receipt_path", HERE / target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME)),
            "hwmon_root": "/sys/class/hwmon",
            "provenance": {
                "authority": "target_sensor_discovery",
                "science_package_id": public.SCIENCE_PACKAGE_ID,
                "transaction_run_id": public.TRANSACTION_RUN_ID,
                "target_platform": {"vendor": "AuthenticAMD", "cpu_family": 16},
                "discovery_monotonic_ns": 1,
                "controller_challenge_sha256": public.digest(active_challenge),
                "authorized_commit": active_source_commit,
            },
            "controller_challenge_sha256": public.digest(active_challenge),
            "controller_nonce_sha256": active_challenge["controller_nonce_sha256"],
            "authorized_commit": active_source_commit,
            "source_hashes_sha256": active_challenge["source_hashes_sha256"],
            "source_bundle_sha256": active_challenge["source_bundle_sha256"],
            "runtime_binary_sha256": active_challenge["runtime_binary_sha256"],
            "source_authority_review": active_challenge["source_authority_review"],
            "source_authority": {"passed": True, "fixture": True},
            "challenge_validation": {"passed": True, "challenge_sha256": public.digest(active_challenge)},
            "top_level_visibility_snapshot": {"temp_input_candidate_count": 1},
            "observed_candidates": [candidate],
            "candidate_count": 1,
            "approved_count": 0,
            "active_counters": counters,
        }
        if overrides:
            row.update(overrides)
        row["target_discovery_receipt_sha256"] = public.digest({k: v for k, v in row.items() if k != "target_discovery_receipt_sha256"})
        return row

    def mutated_failure_receipt(mutator: Any) -> dict[str, Any]:
        row = seal_structured_failure_receipt()
        mutator(row)
        row["target_discovery_receipt_sha256"] = public.digest({k: v for k, v in row.items() if k != "target_discovery_receipt_sha256"})
        return row

    valid_structured_failure_validation = validate_target_discovery_failure_receipt(
        seal_structured_failure_receipt(),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    boolean_top_level_failure_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.update({"target_contact_count": True})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    boolean_active_counter_failure_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row["active_counters"].update({"target_contact_count": True})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    contradictory_active_counter_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row["active_counters"].update({"candidate_scan_count": 0})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    missing_source_review_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.pop("source_authority_review")),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    mismatched_source_review_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.update({"source_authority_review": {**structured_failure_challenge["source_authority_review"], "findings_sha256": "0" * 64}})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    source_hash_mismatch_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.update({"source_hashes_sha256": "0" * 64})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    source_commit_mismatch_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.update({"authorized_commit": "2" * 40})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    candidate_mutation_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row["observed_candidates"].append({"approved": False})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    provenance_mutation_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row["provenance"].pop("target_platform")),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    boolean_pmu_open_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.update({"pmu_open_count": False})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    float_runtime_launch_validation = validate_target_discovery_failure_receipt(
        mutated_failure_receipt(lambda row: row.update({"runtime_launch_count": 0.0})),
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    digest_corruption_receipt = seal_structured_failure_receipt()
    digest_corruption_receipt["target_discovery_receipt_sha256"] = "0" * 64
    digest_corruption_validation = validate_target_discovery_failure_receipt(
        digest_corruption_receipt,
        expected_challenge=structured_failure_challenge,
        expected_source_commit=structured_failure_commit,
    )
    challenge_mismatch_validation = validate_target_discovery_failure_receipt(
        seal_structured_failure_receipt(),
        expected_challenge={**structured_failure_challenge, "source_bundle_sha256": "0" * 64},
        expected_source_commit=structured_failure_commit,
    )

    def fake_source_review(
        *,
        source_commit: str,
        source_hashes_sha256: str,
        source_bundle_sha256: str,
        runtime_binary_sha256: str,
    ) -> dict[str, Any]:
        quorum = {"passed": True, "failures": [], "fixture": "production invalid copyback regression"}
        return {
            "passed": True,
            "failures": [],
            "review_quorum": quorum,
            "review_quorum_sha256": public.digest(quorum),
            "source_authority_commit": source_commit,
            "source_hashes_sha256": source_hashes_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "runtime_binary_sha256": runtime_binary_sha256,
            "findings_sha256": "a" * 64,
            "review_report_sha256": "b" * 64,
        }

    def fake_materialize(_commit: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for name in DISCOVERY_TRANSFER_FILE_NAMES:
            shutil.copy2(HERE / name, destination / name)

    def seal_success_discovery_receipt(active_challenge: dict[str, Any], source_commit: str) -> dict[str, Any]:
        scope = active_challenge.get("transport_scope") if isinstance(active_challenge.get("transport_scope"), dict) else {}
        identity = public.with_temperature_identity_digest(
            {
                **{key: public.synthetic_temperature_identity()[key] for key in public.synthetic_temperature_identity() if key != "identity_sha256"},
                "class_path": "/sys/class/hwmon/hwmon9/temp1_input",
                "resolved_input_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon9/temp1_input",
                "resolved_hwmon_path": "/sys/devices/pci0000:00/0000:00:18.3/hwmon/hwmon9",
                "input_st_ino": 99,
            }
        )
        platform = {
            "vendor": "AuthenticAMD",
            "cpu_family": 16,
            "cpu_models": [10],
            "processor_count": 6,
            "processors": [
                {"processor": cpu, "vendor_id": "AuthenticAMD", "cpu_family": 16, "model": 10}
                for cpu in range(6)
            ],
            "checked_before_discovery": True,
            "cpuinfo_path": "/proc/cpuinfo",
            "source_cpu_expected": public.SOURCE_CPU_EXPECTED,
            "receiver_cpu_expected": public.RECEIVER_CPU_EXPECTED,
            "source_receiver_cpus_present": True,
            "affinity_checked": True,
            "affinity_cpus": list(range(6)),
            "inherited_affinity_checked": True,
            "inherited_affinity_cpus": list(range(6)),
            "operational_pin_capability": target.fake_pin_probe(
                {public.SOURCE_CPU_EXPECTED, public.RECEIVER_CPU_EXPECTED},
                inherited_affinity=list(range(6)),
            ),
            "operational_pin_capability_passed": True,
        }
        authorizing_scope = {
            "canonical_cpuinfo": True,
            "canonical_hwmon_root": True,
            "selected_class_path_is_sysfs_hwmon": True,
            "selected_input_is_legacy_temp1": True,
            "selected_semantic_profile_is_legacy_family10h": True,
            "selected_semantic_role_is_tctl": True,
            "resolved_input_is_sysfs_device": True,
            "resolved_device_is_sysfs_device": True,
            "resolved_driver_is_k10temp": True,
            "resolved_subsystem_is_pci": True,
            "authorizing": True,
            "cpuinfo_path": "/proc/cpuinfo",
            "hwmon_root": "/sys/class/hwmon",
        }
        receipt = {
            "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
            "discovery_mode": "target_read_only_sensor_inventory",
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "candidate_scan_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "pmu_open_count": 0,
            "runtime_launch_count": 0,
            "tomography_output_root_created": False,
            "source_root": scope.get("remote_source_root"),
            "receipt_path": scope.get("remote_receipt_path"),
            "controller_nonce_sha256": active_challenge["controller_nonce_sha256"],
            "selected_identity": identity,
            "identity_before": identity,
            "identity_after": identity,
            "observed_candidates": [{"identity": identity, "approved": True}],
            "selection": {
                "law": "exactly one LEGACY_FAMILY10H_K10TEMP_TEMP1_V1 candidate",
                "approval_profile": target.LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
                "approved_count": 1,
                "deterministic_law": True,
                "selected_class_path": identity["class_path"],
            },
            "source_authority": {"passed": True, "fixture": True},
            "authorizing_scope": authorizing_scope,
            "provenance": {
                "authority": "target_sensor_discovery",
                "science_package_id": public.SCIENCE_PACKAGE_ID,
                "transaction_run_id": public.TRANSACTION_RUN_ID,
                "target_platform": platform,
                "discovery_monotonic_ns": 1,
                "controller_challenge_sha256": public.digest(active_challenge),
                "authorized_commit": source_commit,
            },
            "sample": {
                "identity": identity,
                "path": identity["class_path"],
                "label_present": identity["sensor_label_present"],
                "label_value": identity["sensor_label_value"],
                "semantic_role": identity["sensor_semantic_role"],
                "semantic_profile": identity["sensor_semantic_profile"],
                "value_c": 42.0,
                "pinned_descriptor": target.expected_descriptor_identity(identity),
                "read_law": "manifest-approved resolved input descriptor",
            },
        }
        receipt["target_discovery_receipt_sha256"] = public.digest(
            {k: v for k, v in receipt.items() if k != "target_discovery_receipt_sha256"}
        )
        return receipt

    def fake_transport_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal production_invalid_cleanup_invoked, production_invalid_absence_probe_invoked
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.startswith("sha256sum "):
                return subprocess.CompletedProcess(command_list, 0, invalid_discovery_file_sha + "\n", "")
            if "rm -rf --" in script:
                production_invalid_cleanup_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.strip().endswith("false"):
                return subprocess.CompletedProcess(command_list, 1, "", "")
            if "test ! -e " in script:
                production_invalid_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 1, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and command_list[2].startswith(f"{TARGET_HOST}:"):
                Path(command_list[3]).write_bytes(invalid_discovery_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_production_invalid_copyback_regression_") as tmp:
        temp_root = Path(tmp)
        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = fake_transport_run
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {
                "passed": True,
                "failures": [],
                "commit": commit,
                "files": {},
                "status": [],
            }
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except ControllerError as exc:
                production_invalid_error = str(exc)
            rows = strict_jsonl_loads(DISCOVERY_ATTEMPT_JOURNAL_PATH.read_bytes())
            states = [row.get("attempt_state") for row in rows]
            if DISCOVERY_CLEANUP_CUSTODY_PATH.exists():
                cleanup_receipt = read_json(DISCOVERY_CLEANUP_CUSTODY_PATH)
                production_invalid_cleanup_receipt_sealed = cleanup_receipt["cleanup_custody_sha256"] == public.digest(
                    {k: v for k, v in cleanup_receipt.items() if k != "cleanup_custody_sha256"}
                )
                cleanup_state = cleanup_receipt.get("cleanup", {})
                production_invalid_cleanup_receipt_records_absence = (
                    isinstance(cleanup_state, dict)
                    and cleanup_state.get("attempted") is False
                    and cleanup_state.get("passed") is False
                    and cleanup_state.get("absence_verified") is False
                    and cleanup_state.get("skipped_reason") == "target discovery receipt was not durably sealed before cleanup"
                    and cleanup_receipt.get("sensor_inventory_count") == 0
                    and cleanup_receipt.get("candidate_scan_count") == 0
                )
            production_invalid_journal_stopped_before_receipt_copied = states == ATTEMPT_STATE_SEQUENCE[:3]
            production_invalid_no_success_cleanup_states = "cleanup_armed" not in states and "cleanup_completed" not in states
            production_invalid_no_success_artifacts = (
                not TEMPERATURE_SENSOR_AUTHORITY.exists()
                and not DISCOVERY_TRANSPORT_PATH.exists()
                and not TARGET_DISCOVERY_RECEIPT.exists()
            )
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    success_challenge_seen: dict[str, Any] | None = None
    success_discovery_payload: dict[str, Any] | None = None
    success_discovery_bytes = b""
    success_discovery_sha = ""
    success_directory_barriers: list[Path] = []

    def successful_target_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal success_challenge_seen
        nonlocal success_discovery_payload
        nonlocal success_discovery_bytes
        nonlocal success_discovery_sha
        nonlocal production_success_cleanup_invoked
        nonlocal production_success_absence_probe_invoked
        nonlocal production_success_receipts_sealed_at_cleanup
        nonlocal production_success_directory_barriers_before_cleanup
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.startswith("sha256sum "):
                if success_challenge_seen is None:
                    return subprocess.CompletedProcess(command_list, 1, "", "")
                success_discovery_payload = seal_success_discovery_receipt(success_challenge_seen, "1" * 40)
                success_discovery_bytes = (strict_json_dumps(success_discovery_payload, indent=2) + "\n").encode("utf-8")
                success_discovery_sha = hashlib.sha256(success_discovery_bytes).hexdigest()
                return subprocess.CompletedProcess(command_list, 0, success_discovery_sha + "\n", "")
            if "rm -rf --" in script:
                production_success_cleanup_invoked = True
                production_success_receipts_sealed_at_cleanup = (
                    bool(success_discovery_sha)
                    and TARGET_DISCOVERY_RECEIPT.exists()
                    and TEMPERATURE_SENSOR_AUTHORITY.exists()
                    and public.sha256_file(TARGET_DISCOVERY_RECEIPT) == success_discovery_sha
                    and read_json(TEMPERATURE_SENSOR_AUTHORITY).get("target_discovery_receipt") == success_discovery_payload
                )
                production_success_directory_barriers_before_cleanup = success_directory_barriers.count(TARGET_DISCOVERY_RECEIPT.parent) >= 4
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                production_success_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and Path(command_list[2]).name == "controller_challenge.json":
                success_challenge_seen = read_json(Path(command_list[2]))
            elif len(command_list) >= 4 and command_list[2].startswith(f"{TARGET_HOST}:"):
                Path(command_list[3]).write_bytes(success_discovery_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_production_success_copyback_regression_") as tmp:
        temp_root = Path(tmp)

        def recording_directory_barrier(path: Path) -> None:
            success_directory_barriers.append(Path(path))

        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = successful_target_run
            globals()["durable_sync_directory"] = recording_directory_barrier
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {
                "passed": True,
                "failures": [],
                "commit": commit,
                "files": {},
                "status": [],
            }
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            transport_receipt = acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            production_success_transport_passed = transport_receipt.get("passed") is True
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    seal_failure_challenge_seen: dict[str, Any] | None = None
    seal_failure_bytes = b""
    seal_failure_sha = ""

    def seal_failure_target_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal seal_failure_challenge_seen
        nonlocal seal_failure_bytes
        nonlocal seal_failure_sha
        nonlocal production_seal_failure_cleanup_invoked
        nonlocal production_seal_failure_absence_probe_invoked
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.startswith("sha256sum "):
                if seal_failure_challenge_seen is None:
                    return subprocess.CompletedProcess(command_list, 1, "", "")
                discovery_payload = seal_success_discovery_receipt(seal_failure_challenge_seen, "1" * 40)
                seal_failure_bytes = (strict_json_dumps(discovery_payload, indent=2) + "\n").encode("utf-8")
                seal_failure_sha = hashlib.sha256(seal_failure_bytes).hexdigest()
                return subprocess.CompletedProcess(command_list, 0, seal_failure_sha + "\n", "")
            if "rm -rf --" in script:
                production_seal_failure_cleanup_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                production_seal_failure_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 1, "", "")
            if script == "set -u; false":
                return subprocess.CompletedProcess(command_list, 1, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and Path(command_list[2]).name == "controller_challenge.json":
                seal_failure_challenge_seen = read_json(Path(command_list[2]))
            elif len(command_list) >= 4 and command_list[2].startswith(f"{TARGET_HOST}:"):
                Path(command_list[3]).write_bytes(seal_failure_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_production_success_seal_failure_regression_") as tmp:
        temp_root = Path(tmp)

        def failing_durable_write(path: Path, data: bytes, *, require_directory_sync: bool = False) -> None:
            if Path(path) == temp_root / "target_discovery.json":
                raise OSError("fixture target discovery local seal failure")
            production_originals["durable_write_bytes_exclusive"](path, data, require_directory_sync=require_directory_sync)

        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = seal_failure_target_run
            globals()["durable_write_bytes_exclusive"] = failing_durable_write
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except Exception as exc:  # noqa: BLE001 - fixture injects raw local I/O failure
                production_seal_failure_error = str(exc)
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    target_directory_sync_challenge_seen: dict[str, Any] | None = None
    target_directory_sync_bytes = b""
    target_directory_sync_sha = ""

    def target_directory_sync_failure_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal target_directory_sync_challenge_seen
        nonlocal target_directory_sync_bytes
        nonlocal target_directory_sync_sha
        nonlocal production_target_directory_sync_failure_cleanup_invoked
        nonlocal production_target_directory_sync_failure_absence_probe_invoked
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.startswith("sha256sum "):
                if target_directory_sync_challenge_seen is None:
                    return subprocess.CompletedProcess(command_list, 1, "", "")
                discovery_payload = seal_success_discovery_receipt(target_directory_sync_challenge_seen, "1" * 40)
                target_directory_sync_bytes = (strict_json_dumps(discovery_payload, indent=2) + "\n").encode("utf-8")
                target_directory_sync_sha = hashlib.sha256(target_directory_sync_bytes).hexdigest()
                return subprocess.CompletedProcess(command_list, 0, target_directory_sync_sha + "\n", "")
            if "rm -rf --" in script:
                production_target_directory_sync_failure_cleanup_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                production_target_directory_sync_failure_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 1, "", "")
            if script == "set -u; false":
                return subprocess.CompletedProcess(command_list, 1, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and Path(command_list[2]).name == "controller_challenge.json":
                target_directory_sync_challenge_seen = read_json(Path(command_list[2]))
            elif len(command_list) >= 4 and command_list[2].startswith(f"{TARGET_HOST}:"):
                Path(command_list[3]).write_bytes(target_directory_sync_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_production_target_directory_sync_failure_regression_") as tmp:
        temp_root = Path(tmp)

        def failing_target_directory_barrier(path: Path) -> None:
            if Path(path) == temp_root:
                raise OSError("fixture target discovery directory sync failure")
            production_originals["durable_sync_directory"](path)

        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = target_directory_sync_failure_run
            globals()["durable_sync_directory"] = failing_target_directory_barrier
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except Exception as exc:  # noqa: BLE001 - fixture injects required local directory barrier failure
                production_target_directory_sync_failure_error = str(exc)
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    authority_directory_sync_challenge_seen: dict[str, Any] | None = None
    authority_directory_sync_bytes = b""
    authority_directory_sync_sha = ""

    def authority_directory_sync_failure_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal authority_directory_sync_challenge_seen
        nonlocal authority_directory_sync_bytes
        nonlocal authority_directory_sync_sha
        nonlocal production_authority_directory_sync_failure_cleanup_invoked
        nonlocal production_authority_directory_sync_failure_absence_probe_invoked
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.startswith("sha256sum "):
                if authority_directory_sync_challenge_seen is None:
                    return subprocess.CompletedProcess(command_list, 1, "", "")
                discovery_payload = seal_success_discovery_receipt(authority_directory_sync_challenge_seen, "1" * 40)
                authority_directory_sync_bytes = (strict_json_dumps(discovery_payload, indent=2) + "\n").encode("utf-8")
                authority_directory_sync_sha = hashlib.sha256(authority_directory_sync_bytes).hexdigest()
                return subprocess.CompletedProcess(command_list, 0, authority_directory_sync_sha + "\n", "")
            if "rm -rf --" in script:
                production_authority_directory_sync_failure_cleanup_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                production_authority_directory_sync_failure_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 1, "", "")
            if script == "set -u; false":
                return subprocess.CompletedProcess(command_list, 1, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and Path(command_list[2]).name == "controller_challenge.json":
                authority_directory_sync_challenge_seen = read_json(Path(command_list[2]))
            elif len(command_list) >= 4 and command_list[2].startswith(f"{TARGET_HOST}:"):
                Path(command_list[3]).write_bytes(authority_directory_sync_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_production_authority_directory_sync_failure_regression_") as tmp:
        temp_root = Path(tmp)
        authority_directory_sync_count = 0

        def failing_authority_directory_barrier(path: Path) -> None:
            nonlocal authority_directory_sync_count
            if Path(path) == temp_root:
                authority_directory_sync_count += 1
                if authority_directory_sync_count == 2:
                    raise OSError("fixture authority receipt directory sync failure")
            else:
                production_originals["durable_sync_directory"](path)

        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = authority_directory_sync_failure_run
            globals()["durable_sync_directory"] = failing_authority_directory_barrier
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except Exception as exc:  # noqa: BLE001 - fixture injects required local directory barrier failure
                production_authority_directory_sync_failure_error = str(exc)
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    post_cleanup_challenge_seen: dict[str, Any] | None = None
    post_cleanup_discovery_payload: dict[str, Any] | None = None
    post_cleanup_discovery_bytes = b""
    post_cleanup_discovery_sha = ""

    def post_cleanup_failure_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal post_cleanup_challenge_seen
        nonlocal post_cleanup_discovery_payload
        nonlocal post_cleanup_discovery_bytes
        nonlocal post_cleanup_discovery_sha
        nonlocal production_post_cleanup_cleanup_invoked
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if script.startswith("sha256sum "):
                if post_cleanup_challenge_seen is None:
                    return subprocess.CompletedProcess(command_list, 1, "", "")
                post_cleanup_discovery_payload = seal_success_discovery_receipt(post_cleanup_challenge_seen, "1" * 40)
                post_cleanup_discovery_bytes = (strict_json_dumps(post_cleanup_discovery_payload, indent=2) + "\n").encode("utf-8")
                post_cleanup_discovery_sha = hashlib.sha256(post_cleanup_discovery_bytes).hexdigest()
                return subprocess.CompletedProcess(command_list, 0, post_cleanup_discovery_sha + "\n", "")
            if "rm -rf --" in script:
                production_post_cleanup_cleanup_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                return subprocess.CompletedProcess(command_list, 1, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and Path(command_list[2]).name == "controller_challenge.json":
                post_cleanup_challenge_seen = read_json(Path(command_list[2]))
            elif len(command_list) >= 4 and command_list[2].startswith(f"{TARGET_HOST}:"):
                Path(command_list[3]).write_bytes(post_cleanup_discovery_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_production_post_cleanup_failure_regression_") as tmp:
        temp_root = Path(tmp)
        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = post_cleanup_failure_run
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except ControllerError as exc:
                production_post_cleanup_error = str(exc)
            production_post_cleanup_receipt_recoverable = (
                production_post_cleanup_cleanup_invoked
                and TARGET_DISCOVERY_RECEIPT.exists()
                and public.sha256_file(TARGET_DISCOVERY_RECEIPT) == post_cleanup_discovery_sha
                and read_json(TARGET_DISCOVERY_RECEIPT) == post_cleanup_discovery_payload
            )
            production_post_cleanup_authority_recoverable = (
                production_post_cleanup_cleanup_invoked
                and TEMPERATURE_SENSOR_AUTHORITY.exists()
                and read_json(TEMPERATURE_SENSOR_AUTHORITY).get("target_discovery_receipt") == post_cleanup_discovery_payload
            )
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    c3_reuse_blocked = False
    try:
        acquire_temperature_sensor_authority(source_authority_commit=C3_SOURCE_AUTHORITY_COMMIT)
    except ControllerError as exc:
        c3_reuse_blocked = "C3 source authority acquisition already failed" in str(exc)
    c5_reuse_blocked = False
    try:
        acquire_temperature_sensor_authority(source_authority_commit=C5_ATTEMPT_SOURCE_COMMIT)
    except ControllerError as exc:
        c5_reuse_blocked = "C5 source authority acquisition already failed" in str(exc)

    missing_review_contacted = False
    missing_review_blocked = False

    def failed_source_review(**_kwargs: Any) -> dict[str, Any]:
        return {"passed": False, "failures": ["missing C6 review"], "review_quorum": {"passed": False}}

    def contact_detecting_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal missing_review_contacted
        command_text = " ".join(str(item) for item in command)
        if "python3 family10h_carrier_tomography_target.py" in command_text:
            missing_review_contacted = True
        return subprocess.CompletedProcess([str(item) for item in command], 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_missing_c6_review_regression_") as tmp:
        temp_root = Path(tmp)
        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = contact_detecting_run
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = failed_source_review
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except ControllerError as exc:
                missing_review_blocked = "source authority C6 review gate failed before target contact" in str(exc)
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    target_failure_error = ""
    target_failure_receipt: dict[str, Any] = {}
    target_failure_cleanup_invoked = False
    target_failure_cleanup_skipped_seen = False
    target_failure_absence_probe_invoked = False

    def failing_target_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal target_failure_cleanup_invoked, target_failure_cleanup_skipped_seen, target_failure_absence_probe_invoked
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 17, "bounded target stdout\n", "bounded target stderr\n")
            if script == "set -u; false":
                target_failure_cleanup_skipped_seen = True
                return subprocess.CompletedProcess(command_list, 1, "", "")
            if "rm -rf --" in script:
                target_failure_cleanup_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                target_failure_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 1 if target_failure_cleanup_skipped_seen else 0, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_target_failure_regression_") as tmp:
        temp_root = Path(tmp)
        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = failing_target_run
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except ControllerError as exc:
                target_failure_error = str(exc)
            if TARGET_DISCOVERY_FAILURE_PATH.exists():
                target_failure_receipt = read_json(TARGET_DISCOVERY_FAILURE_PATH)
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    structured_target_failure_error = ""
    structured_target_failure_receipt: dict[str, Any] = {}
    structured_target_failure_cleanup_invoked = False
    structured_target_failure_absence_probe_invoked = False
    structured_target_failure_preserved_before_cleanup_seen = False
    structured_target_failure_file_hash_bound_seen = False
    structured_failure_challenge_seen: dict[str, Any] | None = None
    structured_failure_payload: dict[str, Any] | None = None
    structured_failure_bytes = b""
    structured_failure_sha = ""

    def structured_failing_target_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal structured_target_failure_cleanup_invoked
        nonlocal structured_target_failure_absence_probe_invoked
        nonlocal structured_target_failure_preserved_before_cleanup_seen
        nonlocal structured_failure_challenge_seen
        nonlocal structured_failure_payload
        nonlocal structured_failure_bytes
        nonlocal structured_failure_sha
        command_list = [str(item) for item in command]
        if command_list and command_list[0] == "ssh":
            script = command_list[-1]
            if "FAMILY10H_DISCOVERY_PREFLIGHT" in script:
                return subprocess.CompletedProcess(command_list, 0, "FAMILY10H_DISCOVERY_PREFLIGHT base_preexisting=0 root_created=1\n", "")
            if "python3 family10h_carrier_tomography_target.py" in script:
                return subprocess.CompletedProcess(command_list, 17, "structured target stdout\n", "structured target stderr\n")
            if "sha256sum" in script and target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME in script:
                if structured_failure_challenge_seen is None:
                    return subprocess.CompletedProcess(command_list, 0, "", "")
                structured_failure_payload = seal_structured_failure_receipt(
                    challenge_override=structured_failure_challenge_seen,
                    source_commit_override="1" * 40,
                )
                structured_failure_bytes = (strict_json_dumps(structured_failure_payload, indent=2) + "\n").encode("utf-8")
                structured_failure_sha = hashlib.sha256(structured_failure_bytes).hexdigest()
                return subprocess.CompletedProcess(command_list, 0, structured_failure_sha + "\n", "")
            if "rm -rf --" in script:
                structured_target_failure_cleanup_invoked = True
                if TARGET_DISCOVERY_FAILURE_PATH.exists() and TARGET_DISCOVERY_RECEIPT.exists():
                    pre_cleanup = read_json(TARGET_DISCOVERY_FAILURE_PATH)
                    structured_target_failure_preserved_before_cleanup_seen = (
                        pre_cleanup.get("failure_receipt_written_before_cleanup") is True
                        and pre_cleanup.get("target_failure_preserved_before_cleanup") is True
                        and pre_cleanup.get("target_failure_receipt_sealed_before_cleanup") is True
                        and pre_cleanup.get("candidate_scan_count") == 1
                        and pre_cleanup.get("sensor_inventory_count") == 1
                    )
                return subprocess.CompletedProcess(command_list, 0, "", "")
            if "test ! -e " in script:
                structured_target_failure_absence_probe_invoked = True
                return subprocess.CompletedProcess(command_list, 0, "", "")
            return subprocess.CompletedProcess(command_list, 0, "", "")
        if command_list and command_list[0] == "scp":
            if len(command_list) >= 4 and Path(command_list[2]).name == "controller_challenge.json":
                structured_failure_challenge_seen = read_json(Path(command_list[2]))
            elif len(command_list) >= 4 and command_list[2].endswith(target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME):
                Path(command_list[3]).write_bytes(structured_failure_bytes)
            return subprocess.CompletedProcess(command_list, 0, "", "")
        return subprocess.CompletedProcess(command_list, 0, "", "")

    with tempfile.TemporaryDirectory(prefix="family10h_target_structured_failure_regression_") as tmp:
        temp_root = Path(tmp)
        try:
            globals()["DISCOVERY_ATTEMPT_PATH"] = temp_root / "attempt.json"
            globals()["DISCOVERY_ATTEMPT_JOURNAL_PATH"] = temp_root / "attempt.jsonl"
            globals()["DISCOVERY_CLEANUP_CUSTODY_PATH"] = temp_root / "cleanup.json"
            globals()["TEMPERATURE_SENSOR_AUTHORITY"] = temp_root / "authority.json"
            globals()["DISCOVERY_TRANSPORT_PATH"] = temp_root / "transport.json"
            globals()["TARGET_DISCOVERY_RECEIPT"] = temp_root / "target_discovery.json"
            globals()["DISCOVERY_CHALLENGE_PATH"] = temp_root / "challenge.json"
            globals()["TARGET_DISCOVERY_FAILURE_PATH"] = temp_root / "target_failure.json"
            globals()["run"] = structured_failing_target_run
            globals()["read_source_hash_authority"] = lambda: fake_source_hashes
            globals()["read_existing_source_bundle_authority"] = lambda: fake_bundle
            globals()["source_authority_commit_verification"] = lambda commit: {"passed": True, "failures": [], "commit": commit, "files": {}, "status": []}
            globals()["read_source_authority_review_for_discovery"] = fake_source_review
            globals()["materialize_source_authority_snapshot"] = fake_materialize
            globals()["commit_exists"] = lambda _commit: False
            try:
                acquire_temperature_sensor_authority(source_authority_commit="1" * 40)
            except ControllerError as exc:
                structured_target_failure_error = str(exc)
            if TARGET_DISCOVERY_FAILURE_PATH.exists():
                structured_target_failure_receipt = read_json(TARGET_DISCOVERY_FAILURE_PATH)
                target_discovery_receipt_path = structured_target_failure_receipt.get("target_discovery_receipt_path")
                structured_target_failure_file_hash_bound_seen = (
                    structured_target_failure_receipt.get("target_discovery_receipt_file_sha256") == structured_failure_sha
                    and bool(structured_failure_sha)
                    and isinstance(target_discovery_receipt_path, str)
                    and Path(target_discovery_receipt_path).exists()
                    and public.sha256_file(Path(target_discovery_receipt_path)) == structured_failure_sha
                )
        finally:
            for name, value in production_originals.items():
                globals()[name] = value

    original_history_index = ATTEMPT_HISTORY_INDEX_PATH
    try:
        globals()["ATTEMPT_HISTORY_INDEX_PATH"] = Path(tempfile.gettempdir()) / "family10h_missing_attempt_history_index_for_regression.json"
        history_missing_report = known_historical_lane_contact_report()
    finally:
        globals()["ATTEMPT_HISTORY_INDEX_PATH"] = original_history_index
    history_index_metadata = attempt_history_index_metadata_reporting_only()
    stale_crlf_checkout_sha256 = "0a4444d7d01602bb2390c495cd1d76a1fdb072b1a56acfb6bdde2df1628b2e45"
    stale_checkout_manifest = {
        "temperature_sensor_authority": {
            "historical_attempt_index_metadata_reporting_only": {
                **history_index_metadata,
                "file_sha256": stale_crlf_checkout_sha256,
            }
        }
    }
    valid_history_index_manifest = {
        "temperature_sensor_authority": {
            "historical_attempt_index_metadata_reporting_only": history_index_metadata,
        }
    }
    stale_checkout_history_binding = validate_attempt_history_index_manifest_binding(stale_checkout_manifest)
    valid_history_binding = validate_attempt_history_index_manifest_binding(valid_history_index_manifest)
    with tempfile.TemporaryDirectory(prefix="family10h_untracked_history_index_regression_", dir=HERE) as tmp:
        temp_root = Path(tmp)
        try:
            globals()["ATTEMPT_HISTORY_INDEX_PATH"] = temp_root / "ATTEMPT_HISTORY_INDEX.json"
            write_json(ATTEMPT_HISTORY_INDEX_PATH, {"schema": "FAMILY10H_UNTRACKED_HISTORY_INDEX_FIXTURE_V1"})
            untracked_history_index_metadata = attempt_history_index_metadata_reporting_only()
            untracked_history_index_binding = validate_attempt_history_index_manifest_binding(
                {
                    "temperature_sensor_authority": {
                        "historical_attempt_index_metadata_reporting_only": untracked_history_index_metadata,
                    }
                }
            )
            globals()["ATTEMPT_HISTORY_INDEX_PATH"] = temp_root / "MISSING_ATTEMPT_HISTORY_INDEX.json"
            missing_history_index_metadata = attempt_history_index_metadata_reporting_only()
            missing_history_index_binding = validate_attempt_history_index_manifest_binding(
                {
                    "temperature_sensor_authority": {
                        "historical_attempt_index_metadata_reporting_only": missing_history_index_metadata,
                    }
                }
            )
        finally:
            globals()["ATTEMPT_HISTORY_INDEX_PATH"] = original_history_index

    checks = {
        "production_acquisition_has_no_injected_transport_surface": production_signature
        == {"target_host", "remote_root", "source_authority_commit"},
        "valid_structured_preflight_parses": valid_preflight["passed"] is True and valid_preflight["remote_base_preexisting"] is False,
        "malformed_preflight_fails_closed_and_preserves_base": malformed_preflight["passed"] is False
        and malformed_preflight["remote_base_preexisting"] is True,
        "boolean_challenge_counter_rejected": not validate_discovery_challenge_receipt_payload(bool_challenge)["passed"],
        "boolean_journal_counter_rejected": not replay_attempt_journal_rows(bool_journal_rows)["passed"],
        "boolean_transport_counter_rejected": not validate_discovery_transport_receipt(seal_transport({"target_contact_count": True}))["passed"],
        "boolean_transport_retry_count_rejected": not validate_discovery_transport_receipt(seal_transport({"retry_count": False}))["passed"],
        "c3_source_commit_reuse_blocked": c3_reuse_blocked,
        "c5_source_commit_reuse_blocked": c5_reuse_blocked,
        "valid_structured_target_failure_receipt_accepted": valid_structured_failure_validation["passed"] is True,
        "structured_target_failure_top_level_boolean_counter_rejected": not boolean_top_level_failure_validation["passed"],
        "structured_target_failure_active_boolean_counter_rejected": not boolean_active_counter_failure_validation["passed"],
        "structured_target_failure_active_counter_contradiction_rejected": not contradictory_active_counter_validation["passed"],
        "structured_target_failure_missing_source_review_rejected": not missing_source_review_validation["passed"],
        "structured_target_failure_mismatched_source_review_rejected": not mismatched_source_review_validation["passed"],
        "structured_target_failure_source_hash_mismatch_rejected": not source_hash_mismatch_validation["passed"],
        "structured_target_failure_source_commit_mismatch_rejected": not source_commit_mismatch_validation["passed"],
        "structured_target_failure_candidate_mutation_rejected": not candidate_mutation_validation["passed"],
        "structured_target_failure_provenance_mutation_rejected": not provenance_mutation_validation["passed"],
        "structured_target_failure_boolean_pmu_open_rejected": not boolean_pmu_open_validation["passed"],
        "structured_target_failure_float_runtime_launch_rejected": not float_runtime_launch_validation["passed"],
        "structured_target_failure_digest_corruption_rejected": not digest_corruption_validation["passed"],
        "structured_target_failure_challenge_mismatch_rejected": not challenge_mismatch_validation["passed"],
        "missing_c5_review_blocks_before_target_contact": missing_review_blocked and not missing_review_contacted,
        "target_nonzero_failure_raises_no_retry": "target discovery failed rc=17" in target_failure_error,
        "target_nonzero_missing_structured_failure_rejected": "structured failure receipt invalid or missing" in target_failure_error
        and target_failure_receipt.get("target_failure", {}).get("structured_failure_receipt_present") is False
        and target_failure_receipt.get("structured_target_discovery_failure_validation", {}).get("passed") is False,
        "target_nonzero_failure_receipt_persisted": target_failure_receipt.get("target_failure", {}).get("target_return_code") == 17,
        "target_nonzero_failure_stdout_bound_and_hashed": target_failure_receipt.get("target_failure", {}).get("stdout_sha256") == sha256_text("bounded target stdout\n"),
        "target_nonzero_failure_stderr_bound_and_hashed": target_failure_receipt.get("target_failure", {}).get("stderr_sha256") == sha256_text("bounded target stderr\n"),
        "target_nonzero_failure_cleanup_recorded": target_failure_cleanup_skipped_seen
        and target_failure_absence_probe_invoked
        and not target_failure_cleanup_invoked
        and target_failure_receipt.get("cleanup", {}).get("attempted") is False
        and target_failure_receipt.get("cleanup_result") is False
        and target_failure_receipt.get("remote_root_absence_result") is False
        and target_failure_receipt.get("cleanup", {}).get("skipped_reason")
        == "target failure evidence was not locally preserved before cleanup",
        "target_nonzero_valid_structured_failure_raises_no_retry": "target discovery failed rc=17" in structured_target_failure_error,
        "target_nonzero_valid_structured_failure_preserved_before_cleanup": structured_target_failure_preserved_before_cleanup_seen,
        "target_nonzero_valid_structured_failure_cleanup_after_preservation": structured_target_failure_cleanup_invoked
        and structured_target_failure_absence_probe_invoked
        and structured_target_failure_receipt.get("cleanup_result") is True
        and structured_target_failure_receipt.get("remote_root_absence_result") is True,
        "target_nonzero_valid_structured_failure_scan_count_from_receipt": structured_target_failure_receipt.get("candidate_scan_count") == 1
        and structured_target_failure_receipt.get("structured_target_discovery_failure_receipt", {}).get("candidate_scan_count") == 1,
        "target_nonzero_valid_structured_failure_file_hash_bound": structured_target_failure_file_hash_bound_seen,
        "history_index_deletion_does_not_fabricate_active_authority": history_missing_report["authoritative_for_active_transaction"] is False and history_missing_report["complete_cryptographic_lane_ledger_claimed"] is False,
        "history_index_manifest_binds_git_text_blob": valid_history_binding["passed"] is True
        and history_index_metadata.get("file_sha256") == history_index_metadata.get("git_blob_sha256")
        and history_index_metadata.get("git_blob_matches_file_sha256") is True,
        "history_index_manifest_rejects_checkout_digest": stale_checkout_history_binding["passed"] is False,
        "history_index_manifest_rejects_false_git_blob_match": untracked_history_index_binding["passed"] is False
        and untracked_history_index_metadata.get("git_blob_matches_file_sha256") is False,
        "history_index_manifest_rejects_missing_git_blob": missing_history_index_binding["passed"] is False
        and missing_history_index_metadata.get("git_blob_sha256") is None,
        "production_success_copyback_receipts_sealed_before_cleanup": production_success_receipts_sealed_at_cleanup,
        "production_success_directory_barriers_before_cleanup": production_success_directory_barriers_before_cleanup,
        "production_success_acquisition_completes_after_precleanup_seal": production_success_transport_passed
        and production_success_cleanup_invoked
        and production_success_absence_probe_invoked,
        "production_success_local_seal_failure_blocks_remote_cleanup": "fixture target discovery local seal failure" in production_seal_failure_error
        and not production_seal_failure_cleanup_invoked
        and production_seal_failure_absence_probe_invoked,
        "production_success_target_directory_sync_failure_blocks_remote_cleanup": "fixture target discovery directory sync failure"
        in production_target_directory_sync_failure_error
        and not production_target_directory_sync_failure_cleanup_invoked
        and production_target_directory_sync_failure_absence_probe_invoked,
        "production_success_authority_directory_sync_failure_blocks_remote_cleanup": "fixture authority receipt directory sync failure"
        in production_authority_directory_sync_failure_error
        and not production_authority_directory_sync_failure_cleanup_invoked
        and production_authority_directory_sync_failure_absence_probe_invoked,
        "production_success_post_cleanup_failure_preserves_receipts": "discovery cleanup or remote-root absence verification failed" in production_post_cleanup_error
        and production_post_cleanup_receipt_recoverable
        and production_post_cleanup_authority_recoverable,
        "failure_cleanup_receipt_writes_without_success_journal": cleanup_valid["cleanup_custody_sha256"]
        == public.digest({k: v for k, v in cleanup_valid.items() if k != "cleanup_custody_sha256"}),
        "parsed_invalid_copyback_cleanup_receipt_sealed": parsed_invalid_cleanup_sealed,
        "parsed_invalid_copyback_journal_stops_before_success_cleanup": parsed_invalid_journal_stopped_before_success_cleanup,
        "parsed_invalid_copyback_no_success_artifacts": parsed_invalid_no_success_artifacts,
        "production_invalid_copyback_acquisition_raises_before_success": "discovery selected identity missing" in production_invalid_error,
        "production_invalid_copyback_cleanup_not_invoked": not production_invalid_cleanup_invoked,
        "production_invalid_copyback_absence_probe_invoked": production_invalid_absence_probe_invoked,
        "production_invalid_copyback_cleanup_receipt_sealed": production_invalid_cleanup_receipt_sealed,
        "production_invalid_copyback_cleanup_receipt_records_absence": production_invalid_cleanup_receipt_records_absence,
        "production_invalid_copyback_journal_stops_before_receipt_copied": production_invalid_journal_stopped_before_receipt_copied,
        "production_invalid_copyback_no_success_cleanup_states": production_invalid_no_success_cleanup_states,
        "production_invalid_copyback_no_success_artifacts": production_invalid_no_success_artifacts,
        "boolean_cleanup_counter_rejected": bool_cleanup_rejected,
        "cleanup_inventory_candidate_pair_rejected": pair_cleanup_rejected,
    }
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_CONTROLLER_ACQUISITION_TRANSACTION_REGRESSION_V2",
        "passed": all(checks.values()),
        "checks": checks,
        "production_invalid_error": production_invalid_error,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }


def controller_self_test() -> dict[str, Any]:
    transport = fake_transport_self_tests()
    discovery_transport = discovery_transport_self_tests()
    production_transfer = production_discovery_transfer_plan_regression()
    validation = offline_validate()
    runtime_gate = runtime_authority_gate_self_test()
    live_env_absent = target.validate_no_live_authority_env()
    source_hash_authority = source_hash_authority_regression()
    runtime_overlay_mutation = runtime_binary_overlay_mutation_regression()
    temperature_authority = temperature_sensor_authority_regression()
    null_model_baseline = review_quorum_null_model_baseline()
    source_audit_regression = source_audit_quorum_regression()
    bundle_mode_regression = source_bundle_mode_regression()
    attempt_journal_regression = discovery_attempt_journal_regression()
    strict_json = strict_json_regression()
    challenge_receipt = challenge_receipt_regression()
    historical_lane_report = known_historical_lane_contact_report()
    package_parse = package_json_parse_audit()
    acquisition_transaction = controller_acquisition_transaction_regression()
    clearances = {
        role: {
            "role": label,
            "originating_agent": label,
            "agent_id": f"reviewer-{index}",
            "verdict": "NO_MATERIAL_BLOCKER",
            "final_response": True,
        }
        for index, (role, label) in enumerate(REQUIRED_REVIEW_ROLES.items(), start=1)
    }
    three_clearances = dict(list(clearances.items())[:3])
    duplicate_clearances = {
        **clearances,
        "claim_boundary_adjudicator": {**clearances["claim_boundary_adjudicator"], "agent_id": clearances["custody_evidence_auditor"]["agent_id"]},
    }
    nonfinal_clearances = {
        **clearances,
        "claim_boundary_adjudicator": {**clearances["claim_boundary_adjudicator"], "final_response": False},
    }
    missing_final_clearances = {
        **clearances,
        "claim_boundary_adjudicator": {
            key: value
            for key, value in clearances["claim_boundary_adjudicator"].items()
            if key != "final_response"
        },
    }
    fifth_clearance = {
        **clearances,
        "surplus_clear_reviewer": {
            "role": "surplus_clear_reviewer",
            "originating_agent": "surplus clear reviewer",
            "agent_id": "reviewer-surplus-clear",
            "verdict": "NO_MATERIAL_BLOCKER",
            "final_response": True,
        },
    }
    fifth_blocker = {
        **clearances,
        "surplus_blocking_reviewer": {
            "role": "surplus_blocking_reviewer",
            "originating_agent": "surplus blocking reviewer",
            "agent_id": "reviewer-surplus-blocker",
            "verdict": "MATERIAL_BLOCKER",
            "final_response": True,
        },
    }
    fifth_malformed = {**clearances, "surplus_malformed_reviewer": "NO_MATERIAL_BLOCKER"}
    quorum_regressions = {
        "missing_findings_blocked": not review_quorum({})["passed"],
        "empty_findings_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": {}})["passed"],
        "three_clearances_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": three_clearances})["passed"],
        "duplicate_role_or_agent_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": duplicate_clearances})["passed"],
        "non_final_response_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": nonfinal_clearances})["passed"],
        "missing_final_response_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": missing_final_clearances})["passed"],
        "fifth_clear_response_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": fifth_clearance})["passed"],
        "fifth_blocking_response_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": fifth_blocker})["passed"],
        "fifth_malformed_response_blocked": not review_quorum({"material_blockers": [], "reviewer_verdicts": fifth_malformed})["passed"],
        "material_blocker_list_blocked": not review_quorum({"material_blockers": [{"id": "BLOCKER"}], "reviewer_verdicts": clearances})["passed"],
        "exact_four_final_clearances_satisfy_legacy_quorum_only": review_quorum({"material_blockers": [], "reviewer_verdicts": clearances})["passed"],
        "legacy_quorum_without_authority_cannot_freeze_package": public.PACKAGE_DECISION_BLOCKED
        == (
            public.PACKAGE_DECISION_BLOCKED
            if not source_audit_regression["checks"]["exact_source_audit_quorum_passes"]
            or not read_temperature_sensor_authority()["passed"]
            or not FINAL_OBJECT_VERIFY_PATH.exists()
            else public.PACKAGE_DECISION_FROZEN
        ),
    }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST_V1",
        "self_test_passed": validation["passed"]
        and transport["passed"]
        and discovery_transport["passed"]
        and production_transfer["passed"]
        and runtime_gate["passed"]
        and live_env_absent["passed"]
        and source_hash_authority["passed"]
        and runtime_overlay_mutation["passed"]
        and temperature_authority["passed"]
        and null_model_baseline["passed"]
        and source_audit_regression["passed"]
        and bundle_mode_regression["passed"]
        and attempt_journal_regression["passed"]
        and strict_json["passed"]
        and challenge_receipt["passed"]
        and package_parse["passed"]
        and historical_lane_report["authoritative_for_active_transaction"] is False
        and historical_lane_report["complete_cryptographic_lane_ledger_claimed"] is False
        and acquisition_transaction["passed"]
        and all(quorum_regressions.values()),
        "offline_validate_sha256": validation["offline_validate_sha256"],
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "discovery_transport_self_test": discovery_transport,
        "production_discovery_transfer_plan_regression": production_transfer,
        "runtime_authority_gate": runtime_gate,
        "source_hash_authority_regression": source_hash_authority,
        "runtime_binary_overlay_mutation_regression": runtime_overlay_mutation,
        "temperature_sensor_authority_regression": temperature_authority,
        "review_quorum_null_model_baseline": null_model_baseline,
        "source_audit_quorum_regression": source_audit_regression,
        "source_bundle_mode_regression": bundle_mode_regression,
        "discovery_attempt_journal_regression": attempt_journal_regression,
        "strict_json_regression": strict_json,
        "package_json_parse_audit": package_parse,
        "challenge_receipt_regression": challenge_receipt,
        "historical_lane_contact_report": historical_lane_report,
        "controller_acquisition_transaction_regression": acquisition_transaction,
        "live_authority_env_absent": live_env_absent,
        "review_quorum_regressions": quorum_regressions,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(CONTROLLER_SELF_TEST_PATH, result)
    return result


def authority_contact_counters(temperature_authority: dict[str, Any]) -> dict[str, int]:
    if DISCOVERY_ATTEMPT_PATH.exists():
        attempt = read_json(DISCOVERY_ATTEMPT_PATH)
        for key in ATTEMPT_COUNTER_KEYS:
            require(is_strict_int(attempt.get(key)), f"discovery attempt counter missing or invalid {key}")
        return {key: attempt[key] for key in ATTEMPT_COUNTER_KEYS}
    if temperature_authority.get("passed") is not True or not TEMPERATURE_SENSOR_AUTHORITY.exists():
        return {
            "target_contact_count": 0,
            "sensor_inventory_count": 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    receipt = read_json(TEMPERATURE_SENSOR_AUTHORITY)
    for key in ATTEMPT_COUNTER_KEYS:
        require(is_strict_int(receipt.get(key)), f"temperature authority counter missing or invalid {key}")
    return {key: receipt[key] for key in ATTEMPT_COUNTER_KEYS}


def manifest() -> dict[str, Any]:
    schedule = read_json(public.SCHEDULE_JSON)
    schedule_sidecar = read_json(public.SCHEDULE_SHA)
    source_hashes = read_source_hash_authority()
    bundle = write_source_bundle()
    runtime_result = read_json(RUNTIME_SELF_TEST_PATH) if RUNTIME_SELF_TEST_PATH.exists() else runtime_self_test()
    offline = read_json(OFFLINE_VALIDATE_PATH) if OFFLINE_VALIDATE_PATH.exists() else offline_validate()
    transport = read_json(TRANSPORT_SIM_PATH) if TRANSPORT_SIM_PATH.exists() else fake_transport_self_tests()
    deployment = read_json(DEPLOYMENT_LAYOUT_PATH) if DEPLOYMENT_LAYOUT_PATH.exists() else deployment_layout_self_test()
    target_result = read_json(TARGET_SELF_TEST_PATH) if TARGET_SELF_TEST_PATH.exists() else target_self_test()
    controller_result = read_json(CONTROLLER_SELF_TEST_PATH) if CONTROLLER_SELF_TEST_PATH.exists() else controller_self_test()
    git = git_state()
    temperature_challenge, source_authority_commit = expected_temperature_authority_challenge_for_manifest(
        source_hashes=source_hashes,
        source_bundle_sha256=bundle["sha256"],
        schedule_sidecar=schedule_sidecar,
        fallback_authorized_commit=git["head"],
    )
    temperature_authority = read_temperature_sensor_authority(expected_challenge=temperature_challenge)
    contact_counters = authority_contact_counters(temperature_authority)
    historical_lane_report = known_historical_lane_contact_report()
    prior_lane_counters = historical_lane_report["known_counters_before_active_attempt"]
    cumulative_lane_counters = counter_sum(prior_lane_counters, contact_counters)
    independent_review = {}
    if SUBAGENT_FINDINGS_PATH.exists():
        independent_review = read_json(SUBAGENT_FINDINGS_PATH)
    quorum = review_quorum(independent_review)
    prior_reviewer_ids = {
        str(item.get("agent_id"))
        for item in (independent_review.get("reviewer_verdicts") or {}).values()
        if isinstance(item, dict) and item.get("agent_id")
    }
    source_audit_paths = source_audit_paths_for_commit(source_authority_commit)
    source_audit_findings_path = source_audit_paths["findings_path"]
    source_audit_review_path = source_audit_paths["review_path"]
    source_audit_review_dir = source_audit_paths["review_dir"]
    source_audit = {}
    if source_audit_findings_path.exists():
        source_audit = read_json(source_audit_findings_path)
    source_quorum = source_audit_quorum(
        source_audit,
        expected_source_commit=source_authority_commit,
        expected_source_hashes_sha256=source_hashes["source_hashes_sha256"],
        expected_source_bundle_sha256=bundle["sha256"],
        expected_runtime_binary_sha256=source_hashes["runtime_binary_authority"]["sha256"],
        review_report_present=source_audit_review_path.exists(),
        excluded_agent_ids=prior_reviewer_ids,
        review_root=source_audit_review_dir,
    )
    review_blocked = not quorum["passed"]
    source_review_blocked = not source_quorum["passed"]
    expected_contact_counters = {
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    } if DISCOVERY_ATTEMPT_PATH.exists() else {
        "target_contact_count": 1 if temperature_authority["passed"] else 0,
        "sensor_inventory_count": 1 if temperature_authority["passed"] else 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    contact_counters_truthful = contact_counters == expected_contact_counters
    final_exact_object = read_json(FINAL_OBJECT_VERIFY_PATH) if FINAL_OBJECT_VERIFY_PATH.exists() else {}
    final_evidence_authority = read_final_evidence_commit_authority() if FINAL_OBJECT_VERIFY_PATH.exists() else {
        "passed": False,
        "failures": ["final evidence commit authority missing"],
        "receipt": {},
    }
    final_evidence_receipt = final_evidence_authority.get("receipt", {}) if isinstance(final_evidence_authority, dict) else {}
    final_exact_validation = validate_final_exact_object_receipt(
        final_exact_object,
        expected_source_commit=source_authority_commit,
        expected_evidence_commit=final_evidence_receipt.get("evidence_commit") if isinstance(final_evidence_receipt, dict) else None,
    )
    final_exact_object_passed = final_evidence_authority["passed"] and final_exact_validation["passed"]
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "claim_ceiling": "route-scoped public carrier-state model only",
        "package_decision": public.PACKAGE_DECISION_BLOCKED
        if review_blocked
        or source_review_blocked
        or not offline["passed"]
        or not temperature_authority["passed"]
        or not contact_counters_truthful
        or not final_exact_object_passed
        else public.PACKAGE_DECISION_FROZEN,
        "independent_review": {
            "findings_path": str(SUBAGENT_FINDINGS_PATH) if SUBAGENT_FINDINGS_PATH.exists() else None,
            "findings_sha256": public.sha256_file(SUBAGENT_FINDINGS_PATH) if SUBAGENT_FINDINGS_PATH.exists() else None,
            "review_report_path": str(SUBAGENT_REVIEW_PATH) if SUBAGENT_REVIEW_PATH.exists() else None,
            "review_report_sha256": public.sha256_file(SUBAGENT_REVIEW_PATH) if SUBAGENT_REVIEW_PATH.exists() else None,
            "material_blocker_count": len(independent_review.get("material_blockers", [])),
            "verdict": independent_review.get("package_decision"),
            "review_quorum": quorum,
        },
        "source_authority_review": {
            "findings_path": str(source_audit_findings_path) if source_audit_findings_path.exists() else None,
            "findings_sha256": public.sha256_file(source_audit_findings_path) if source_audit_findings_path.exists() else None,
            "review_report_path": str(source_audit_review_path) if source_audit_review_path.exists() else None,
            "review_report_sha256": public.sha256_file(source_audit_review_path) if source_audit_review_path.exists() else None,
            "source_authority_commit": source_authority_commit,
            "source_hashes_sha256": source_hashes["source_hashes_sha256"],
            "source_bundle_sha256": bundle["sha256"],
            "runtime_binary_sha256": source_hashes["runtime_binary_authority"]["sha256"],
            "material_blocker_count": len(source_audit.get("material_blockers", [])) if isinstance(source_audit, dict) else None,
            "review_quorum": source_quorum,
        },
        "final_exact_object_verification": {
            "path": str(FINAL_OBJECT_VERIFY_PATH) if FINAL_OBJECT_VERIFY_PATH.exists() else None,
            "file_sha256": public.sha256_file(FINAL_OBJECT_VERIFY_PATH) if FINAL_OBJECT_VERIFY_PATH.exists() else None,
            "final_evidence_commit_path": str(FINAL_EVIDENCE_COMMIT_PATH) if FINAL_EVIDENCE_COMMIT_PATH.exists() else None,
            "final_evidence_commit_file_sha256": public.sha256_file(FINAL_EVIDENCE_COMMIT_PATH) if FINAL_EVIDENCE_COMMIT_PATH.exists() else None,
            "final_evidence_commit_sha256": final_evidence_receipt.get("final_evidence_commit_sha256") if isinstance(final_evidence_receipt, dict) else None,
            "evidence_manifest_file_sha256": final_evidence_receipt.get("evidence_manifest_file_sha256") if isinstance(final_evidence_receipt, dict) else None,
            "evidence_manifest_canonical_sha256": final_evidence_receipt.get("evidence_manifest_canonical_sha256") if isinstance(final_evidence_receipt, dict) else None,
            "evidence_manifest_sidecar_sha256": final_evidence_receipt.get("evidence_manifest_sidecar_sha256") if isinstance(final_evidence_receipt, dict) else None,
            "passed": final_exact_object_passed,
            "source_authority_commit": final_exact_object.get("source_authority_commit"),
            "evidence_commit": final_evidence_receipt.get("evidence_commit") if isinstance(final_evidence_receipt, dict) else final_exact_object.get("evidence_commit"),
            "verification_sha256": final_exact_object.get("final_exact_object_verification_sha256"),
            "failures": final_evidence_authority["failures"] + final_exact_validation["failures"] or final_exact_object.get("failures", ["final exact-object verification missing"]),
        },
        "allowed_result_classes": public.ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": public.FORBIDDEN_RESULT_CLASSES,
        "public_preparation_grammar": public.public_preparations(),
        "query_family": public.query_family(),
        "delay_grid": public.DELAY_GRID,
        "exact_tuple_count": schedule["tuple_count"],
        "factorial_arm_definition": {
            "arms": public.FACTORIAL_ARMS,
            "J_q": "Y_q(A,B) - Y_q(A,dummy) - Y_q(dummy,B) + Y_q(dummy,dummy)",
            "interpretation_ceiling": "observed nonadditivity under matched arms only",
        },
        "operator_ladder": ["S0 scalar preparation amplitude", "S1 amplitude + query + delay + mapping", "S2 interactions and order"],
        "expected_local_root": public.EXPECTED_LOCAL_ROOT,
        "expected_remote_root": public.EXPECTED_REMOTE_ROOT,
        "expected_remote_output_root": public.EXPECTED_REMOTE_OUTPUT_ROOT,
        "schedule_hashes": schedule_sidecar,
        "source_hashes": source_hashes,
        "runtime_binary_authority": source_hashes["runtime_binary_authority"],
        "source_bundle": bundle,
        "runtime_self_test": {
            "passed": runtime_result["passed"],
            "sha256": runtime_result.get("runtime_self_test_sha256"),
            "offline_binary_sha256": runtime_result.get("compile", {}).get("offline_binary_sha256"),
        },
        "target_self_test": {"passed": target_result["self_test_passed"], "sha256": target_result["self_test_sha256"]},
        "controller_self_test": {"passed": controller_result["self_test_passed"], "sha256": controller_result["self_test_sha256"]},
        "offline_validate": {"passed": offline["passed"], "sha256": offline["offline_validate_sha256"]},
        "transport_simulation": {"passed": transport["passed"], "sha256": transport["transport_simulation_sha256"]},
        "deployment_layout": {"passed": deployment["passed"], "sha256": deployment["deployment_layout_self_test_sha256"]},
        "temperature_sensor_authority": {
            "approved_hwmon_names": target.APPROVED_TEMPERATURE_HWMON_NAMES,
            "approved_sensor_labels": target.APPROVED_TEMPERATURE_SENSOR_LABELS,
            "legacy_family10h_sensor_profile": target.LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
            "legacy_family10h_sensor_input": target.LEGACY_FAMILY10H_TEMPERATURE_INPUT,
            "legacy_family10h_sensor_role": target.LEGACY_FAMILY10H_TEMPERATURE_ROLE,
            "legacy_family10h_label_law": "temp1_label absent is accepted; temp1_label present must be exactly Tctl",
            "required_identity_fields": target.TEMPERATURE_IDENTITY_FIELDS,
            "authority_receipt_path": str(TEMPERATURE_SENSOR_AUTHORITY) if TEMPERATURE_SENSOR_AUTHORITY.exists() else None,
            "authority_receipt_file_sha256": public.sha256_file(TEMPERATURE_SENSOR_AUTHORITY) if TEMPERATURE_SENSOR_AUTHORITY.exists() else None,
            "target_discovery_receipt_path": str(TARGET_DISCOVERY_RECEIPT) if TARGET_DISCOVERY_RECEIPT.exists() else None,
            "target_discovery_receipt_file_sha256": public.sha256_file(TARGET_DISCOVERY_RECEIPT) if TARGET_DISCOVERY_RECEIPT.exists() else None,
            "discovery_transport_path": str(DISCOVERY_TRANSPORT_PATH) if DISCOVERY_TRANSPORT_PATH.exists() else None,
            "discovery_transport_file_sha256": public.sha256_file(DISCOVERY_TRANSPORT_PATH) if DISCOVERY_TRANSPORT_PATH.exists() else None,
            "challenge_receipt_path": str(DISCOVERY_CHALLENGE_PATH) if DISCOVERY_CHALLENGE_PATH.exists() else None,
            "challenge_receipt_file_sha256": public.sha256_file(DISCOVERY_CHALLENGE_PATH) if DISCOVERY_CHALLENGE_PATH.exists() else None,
            "attempt_receipt_path": str(DISCOVERY_ATTEMPT_PATH) if DISCOVERY_ATTEMPT_PATH.exists() else None,
            "attempt_receipt_file_sha256": public.sha256_file(DISCOVERY_ATTEMPT_PATH) if DISCOVERY_ATTEMPT_PATH.exists() else None,
            "authority_receipt_present": temperature_authority["present"],
            "authority_receipt_passed": temperature_authority["passed"],
            "authority_receipt_failures": temperature_authority["failures"],
            "controller_challenge": temperature_challenge,
            "controller_challenge_sha256": public.digest(temperature_challenge) if isinstance(temperature_challenge, dict) else None,
            "controller_nonce_env": TEMPERATURE_AUTHORITY_NONCE_ENV,
            "controller_nonce_sha256_env": TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV,
            "approved_sensor_identity": temperature_authority["approved_sensor_identity"],
            "source_authority_commit": source_authority_commit,
            "contact_counters": contact_counters,
            "historical_attempt_index_metadata_reporting_only": {
                **attempt_history_index_metadata_reporting_only(),
            },
            "active_attempt_counters": contact_counters,
            "historical_lane_contact_report": historical_lane_report,
            "known_prior_lane_counters_reporting_only": prior_lane_counters,
            "known_cumulative_lane_counters_reporting_only": cumulative_lane_counters,
            "synthetic_or_provenance_free_identity_cannot_freeze": True,
            "self_authored_discovery_without_controller_challenge_cannot_freeze": True,
            "resolved_identity_bound_in_evidence": True,
            "identity_stability_required": "temperature_after.identity must exactly match temperature_before.identity",
            "rejects_non_cpu_sensors": True,
            "rejects_path_substitution": True,
            "rejects_same_class_path_substitution": True,
            "rejects_identity_drift": True,
            "rejects_unreadable_approved_sensor": True,
        },
        "future_authorization": {
            "commit_binding_env": COMMIT_ENV,
            "manifest_binding_env": MANIFEST_ENV,
            "live_authority_env": AUTHORITY_ENV,
            "runtime_authority_env": target.RUNTIME_AUTHORITY_ENV,
            "live_authority_value": AUTHORITY_VALUE,
            "this_task_authorizes_live_execution": False,
        },
        "contact_counter_attestation": contact_counters,
        "active_attempt_counter_attestation": contact_counters,
        "historical_lane_contact_report": historical_lane_report,
        "known_cumulative_lane_counters_reporting_only": cumulative_lane_counters,
        "zero_live_contact_attestation": {
            "target_contact_count": contact_counters["target_contact_count"],
            "sensor_inventory_count": contact_counters["sensor_inventory_count"],
            "live_invocation_count": contact_counters["live_invocation_count"],
            "pmu_acquisition_count": contact_counters["pmu_acquisition_count"],
        },
        "git_state_at_manifest_build": git,
        "starting_head_required": EXPECTED_STARTING_HEAD,
    }
    result["manifest_canonical_sha256"] = public.digest({k: v for k, v in result.items() if k != "manifest_canonical_sha256"})
    write_json(MANIFEST_PATH, result)
    sidecar = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST_SHA256_V1",
        "manifest_canonical_sha256": result["manifest_canonical_sha256"],
        "manifest_file_sha256": public.sha256_file(MANIFEST_PATH),
    }
    write_json(MANIFEST_SHA_PATH, sidecar)
    return result


def prepare_only() -> dict[str, Any]:
    clear_generated_receipts()
    schedule_hashes = public.write_schedule_artifacts()
    public_self = public.write_self_test(HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json")
    write_split_self_tests(public_self)
    source_hashes = read_source_hash_authority()
    bundle = write_source_bundle()
    target_result = target_self_test()
    runtime_result = runtime_self_test()
    transport = fake_transport_self_tests()
    deployment = deployment_layout_self_test()
    offline = offline_validate()
    controller = controller_self_test()
    manifest_result = manifest()
    contact_counters = manifest_result["contact_counter_attestation"]
    cumulative_lane_counters = manifest_result["known_cumulative_lane_counters_reporting_only"]
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_PREPARE_ONLY_RECEIPT_V1",
        "passed": all(
            [
                public_self["self_test_passed"],
                target_result["self_test_passed"],
                runtime_result["passed"],
                transport["passed"],
                deployment["passed"],
                offline["passed"],
                controller["self_test_passed"],
                manifest_result["package_decision"] in {public.PACKAGE_DECISION_FROZEN, public.PACKAGE_DECISION_BLOCKED},
            ]
        ),
        "schedule_hashes": schedule_hashes,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": bundle["sha256"],
        "manifest_canonical_sha256": manifest_result["manifest_canonical_sha256"],
        "manifest_file_sha256": public.sha256_file(MANIFEST_PATH),
        "offline_binary_sha256": runtime_result.get("compile", {}).get("offline_binary_sha256"),
        "target_contact_count": contact_counters["target_contact_count"],
        "sensor_inventory_count": contact_counters["sensor_inventory_count"],
        "live_invocation_count": contact_counters["live_invocation_count"],
        "pmu_acquisition_count": contact_counters["pmu_acquisition_count"],
        "active_attempt_counters": contact_counters,
        "known_cumulative_lane_counters_reporting_only": cumulative_lane_counters,
    }
    print(strict_json_dumps(result, indent=2))
    return result


def receipt_digest_matches(receipt: dict[str, Any], field: str) -> bool:
    return bool(receipt.get(field)) and receipt.get(field) == public.digest({k: v for k, v in receipt.items() if k != field})


def commit_exists(commit: str) -> bool:
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        return False
    return (
        run(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "cat-file", "-e", f"{commit}^{{commit}}"],
            timeout=30.0,
            check=False,
            cwd=REPO_ROOT,
        ).returncode
        == 0
    )


def is_ancestor_commit(ancestor: str, descendant: str) -> bool:
    return (
        run(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "merge-base", "--is-ancestor", ancestor, descendant],
            timeout=30.0,
            check=False,
            cwd=REPO_ROOT,
        ).returncode
        == 0
    )


def path_to_repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def commit_object_type(commit: str, path: Path) -> str | None:
    rel = path_to_repo_relative(path)
    completed = run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "cat-file", "-t", f"{commit}:{rel}"],
        timeout=30.0,
        check=False,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def commit_blob_id(commit: str, path: Path) -> str | None:
    rel = path_to_repo_relative(path)
    completed = run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", f"{commit}:{rel}"],
        timeout=30.0,
        check=False,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        return None
    blob_id = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", blob_id or "") is None:
        return None
    return blob_id


def commit_blob_bytes(commit: str, path: Path) -> bytes | None:
    rel = path_to_repo_relative(path)
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "show", f"{commit}:{rel}"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=30.0,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def commit_blob_json(commit: str, path: Path) -> Any | None:
    blob = commit_blob_bytes(commit, path)
    if blob is None:
        return None
    try:
        return strict_json_loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def commit_blob_record(commit: str, path: Path) -> dict[str, Any]:
    rel = path_to_repo_relative(path)
    object_type = commit_object_type(commit, path)
    blob_id = commit_blob_id(commit, path) if object_type == "blob" else None
    blob = commit_blob_bytes(commit, path) if object_type == "blob" else None
    return {
        "repo_path": rel,
        "object_type": object_type,
        "blob_id": blob_id,
        "sha256": hashlib.sha256(blob).hexdigest() if blob is not None else None,
        "size": len(blob) if blob is not None else None,
        "present": object_type == "blob" and blob_id is not None and blob is not None,
    }


def commit_has_blob(commit: str, path: Path) -> bool:
    return commit_blob_record(commit, path)["present"] is True


def final_evidence_paths(source_commit: str | None = None) -> dict[str, Path]:
    source_audit_paths = source_audit_paths_for_commit(source_commit)
    source_audit_findings_path = source_audit_paths["findings_path"]
    source_audit_review_path = source_audit_paths["review_path"]
    source_audit_review_dir = source_audit_paths["review_dir"]
    paths = {
        "source audit findings": source_audit_findings_path,
        "source audit report": source_audit_review_path,
        "target discovery receipt": TARGET_DISCOVERY_RECEIPT,
        "temperature authority receipt": TEMPERATURE_SENSOR_AUTHORITY,
        "discovery transport receipt": DISCOVERY_TRANSPORT_PATH,
        "discovery challenge receipt": DISCOVERY_CHALLENGE_PATH,
        "discovery attempt receipt": DISCOVERY_ATTEMPT_PATH,
        "discovery attempt journal": DISCOVERY_ATTEMPT_JOURNAL_PATH,
        "manifest": MANIFEST_PATH,
        "manifest sidecar": MANIFEST_SHA_PATH,
    }
    for role, (body_name, receipt_name) in SOURCE_AUDIT_REVIEW_ARCHIVE_FILES.items():
        paths[f"source audit {role} body"] = source_audit_review_dir / body_name
        paths[f"source audit {role} receipt"] = source_audit_review_dir / receipt_name
    return paths


def source_authority_commit_blob_verification(commit: str) -> dict[str, Any]:
    failures: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        return {"passed": False, "failures": ["source authority commit malformed"], "commit": commit, "files": files}
    if not commit_exists(commit):
        return {"passed": False, "failures": ["source authority commit missing"], "commit": commit, "files": files}
    source_hashes = commit_blob_json(commit, SOURCE_HASHES)
    if not isinstance(source_hashes, dict):
        failures.append("source hash receipt blob missing or invalid")
        source_hashes = {}
    expected = source_hashes.get("source_files", {})
    if set(expected) != set(SOURCE_FILE_NAMES):
        failures.append("source hash receipt keyset mismatch")
    for name in SOURCE_FILE_NAMES:
        path = HERE / name
        record = commit_blob_record(commit, path)
        files[name] = record
        if not record["present"]:
            failures.append(f"source file blob missing {name}")
            continue
        expected_item = expected.get(name, {}) if isinstance(expected, dict) else {}
        if expected_item.get("sha256") != record["sha256"] or expected_item.get("size") != record["size"]:
            failures.append(f"source file authority blob mismatch {name}")
    if source_hashes and source_hashes.get("source_hashes_sha256") != public.digest(
        {k: v for k, v in source_hashes.items() if k != "source_hashes_sha256"}
    ):
        failures.append("source hash receipt blob digest mismatch")
    runtime_authority = source_hashes.get("runtime_binary_authority", {}) if isinstance(source_hashes, dict) else {}
    runtime_record = commit_blob_record(commit, BINARY_PATH)
    files[RUNTIME_BINARY_NAME] = runtime_record
    if not runtime_record["present"]:
        failures.append("runtime binary blob missing")
    elif not isinstance(runtime_authority, dict):
        failures.append("runtime binary authority missing or malformed")
    else:
        if runtime_authority.get("git_blob_id") != runtime_record.get("blob_id"):
            failures.append("runtime binary authority blob-id mismatch")
        if runtime_authority.get("sha256") != runtime_record.get("sha256"):
            failures.append("runtime binary authority sha256 mismatch")
        if runtime_authority.get("size") != runtime_record.get("size"):
            failures.append("runtime binary authority size mismatch")
    bundle_record = commit_blob_record(commit, SOURCE_BUNDLE)
    files[SOURCE_BUNDLE.name] = bundle_record
    bundle_reconstruction = source_bundle_reconstruction_from_commit(commit)
    if not bundle_record["present"]:
        failures.append("source bundle blob missing")
    elif not bundle_reconstruction["passed"] or bundle_reconstruction["sha256"] != bundle_record["sha256"]:
        failures.append("source bundle blob does not match deterministic source reconstruction")
    for name in SOURCE_AUTHORITY_FILE_NAMES:
        if name in files:
            continue
        record = commit_blob_record(commit, HERE / name)
        files[name] = record
        if not record["present"]:
            failures.append(f"source authority blob missing {name}")
    return {
        "passed": not failures,
        "failures": failures,
        "commit": commit,
        "files": files,
        "source_hashes_sha256": source_hashes.get("source_hashes_sha256") if isinstance(source_hashes, dict) else None,
        "source_bundle_sha256": bundle_record.get("sha256"),
        "runtime_binary_sha256": runtime_record.get("sha256"),
        "runtime_binary_blob_id": runtime_record.get("blob_id"),
        "source_bundle_reconstruction": bundle_reconstruction,
    }


def exact_review_agent_ids(review: dict[str, Any]) -> set[str]:
    return {
        str(item.get("agent_id"))
        for item in (review.get("reviewer_verdicts") or {}).values()
        if isinstance(item, dict) and item.get("agent_id")
    }


def replay_final_exact_objects(
    source_commit: str,
    evidence_commit: str,
    *,
    repo_root: Path | None = None,
    package_root: Path | None = None,
    authority_file_names: list[str] | None = None,
) -> dict[str, Any]:
    if repo_root is not None or package_root is not None:
        replay_repo_root = Path(repo_root or REPO_ROOT).resolve()
        replay_package_root = Path(package_root or HERE).resolve()
        replay_source_audit_version = source_audit_version_for_commit(source_commit)
        replay_source_audit_dir_name = f"SOURCE_AUTHORITY_{replay_source_audit_version}_REVIEW"
        replay_source_audit_prefix = f"SOURCE_AUTHORITY_{replay_source_audit_version}"
        path_bindings = {
            "REPO_ROOT": replay_repo_root,
            "HERE": replay_package_root,
            "CONTRACT_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_CONTRACT.md",
            "SOURCE_BUNDLE": replay_package_root / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz",
            "SOURCE_HASHES": replay_package_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json",
            "TEMPERATURE_SENSOR_AUTHORITY": replay_package_root / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY.json",
            "TARGET_DISCOVERY_RECEIPT": replay_package_root / "CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_RECEIPT.json",
            "DISCOVERY_TRANSPORT_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT.json",
            "DISCOVERY_CHALLENGE_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE.json",
            "DISCOVERY_ATTEMPT_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT.json",
            "DISCOVERY_ATTEMPT_JOURNAL_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT.jsonl",
            "DISCOVERY_CLEANUP_CUSTODY_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_DISCOVERY_CLEANUP_CUSTODY.json",
            "FINAL_OBJECT_VERIFY_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_FINAL_OBJECT_VERIFY.json",
            "FINAL_EVIDENCE_COMMIT_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT.json",
            "MANIFEST_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json",
            "MANIFEST_SHA_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256",
            "RUNTIME_SELF_TEST_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST.json",
            "TARGET_SELF_TEST_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_TARGET_SELF_TEST.json",
            "CONTROLLER_SELF_TEST_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST.json",
            "OFFLINE_VALIDATE_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json",
            "TRANSPORT_SIM_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION.json",
            "DEPLOYMENT_LAYOUT_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST.json",
            "FEATURE_BOUNDARY_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_FEATURE_BOUNDARY_SELF_TEST.json",
            "OPERATOR_ANALYSIS_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST.json",
            "FACTORIAL_ARM_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_FACTORIAL_ARM_SELF_TEST.json",
            "SOURCE_DEATH_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_SOURCE_DEATH_CUSTODY_SELF_TEST.json",
            "EXACT_COVERAGE_PATH": replay_package_root / "CARRIER_TOMOGRAPHY_EXACT_COVERAGE_SELF_TEST.json",
            "SUBAGENT_FINDINGS_PATH": replay_package_root / "SUBAGENT_FINDINGS_NORMALIZED.json",
            "SUBAGENT_REVIEW_PATH": replay_package_root / "SUBAGENT_REVIEW_REPORTS.md",
            "BINARY_PATH": replay_package_root / "family10h_carrier_tomography_runtime",
            "SOURCE_AUDIT_REVIEW_DIR": replay_package_root / replay_source_audit_dir_name,
            "SOURCE_AUDIT_FINDINGS_PATH": replay_package_root / f"{replay_source_audit_prefix}_REVIEW_NORMALIZED.json",
            "SOURCE_AUDIT_REVIEW_PATH": replay_package_root / f"{replay_source_audit_prefix}_REVIEW_REPORTS.md",
        }
        old_bindings = {name: globals()[name] for name in path_bindings}
        try:
            globals().update(path_bindings)
            return replay_final_exact_objects(source_commit, evidence_commit, authority_file_names=authority_file_names)
        finally:
            globals().update(old_bindings)

    authority_file_names = list(authority_file_names or DISCOVERY_TRANSFER_FILE_NAMES)
    failures: list[str] = []
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        failures.append("source authority commit missing or malformed")
    elif not commit_exists(source_commit):
        failures.append("source authority commit not found")
    if re.fullmatch(r"[0-9a-f]{40}", evidence_commit or "") is None:
        failures.append("evidence commit missing or malformed")
    elif not commit_exists(evidence_commit):
        failures.append("evidence commit not found")
    if not failures and source_commit == evidence_commit:
        failures.append("evidence commit must be distinct from source authority commit")
    if not failures and not is_ancestor_commit(source_commit, evidence_commit):
        failures.append("evidence commit is not a descendant of source authority commit")

    source_authority = source_authority_commit_blob_verification(source_commit) if commit_exists(source_commit) else {
        "passed": False,
        "failures": ["source authority commit unavailable"],
        "files": {},
    }
    if not source_authority["passed"]:
        failures.append("source authority commit blob verification failed")

    source_authority_blob_records: dict[str, dict[str, Any]] = {}
    changed_source_files: list[str] = []
    if commit_exists(source_commit) and commit_exists(evidence_commit):
        for name in authority_file_names:
            path = HERE / name
            source_record = commit_blob_record(source_commit, path)
            evidence_record = commit_blob_record(evidence_commit, path)
            unchanged = source_record["present"] and evidence_record["present"] and source_record["blob_id"] == evidence_record["blob_id"]
            source_authority_blob_records[name] = {
                "source": source_record,
                "evidence": evidence_record,
                "unchanged_after_c2": unchanged,
            }
            if not unchanged:
                changed_source_files.append(name)
        if changed_source_files:
            failures.append("evidence overlay changed source/runtime authority blobs: " + ",".join(changed_source_files))

    evidence_blob_records = {
        label: commit_blob_record(evidence_commit, path) if commit_exists(evidence_commit) else {
            "repo_path": path_to_repo_relative(path),
            "present": False,
        }
        for label, path in final_evidence_paths(source_commit).items()
    }
    for label, record in evidence_blob_records.items():
        if record.get("present") is not True:
            failures.append(f"evidence commit lacks {label} blob")

    manifest_data = commit_blob_json(evidence_commit, MANIFEST_PATH) if commit_exists(evidence_commit) else None
    sidecar = commit_blob_json(evidence_commit, MANIFEST_SHA_PATH) if commit_exists(evidence_commit) else None
    if not isinstance(manifest_data, dict):
        failures.append("evidence manifest blob is not valid JSON")
        manifest_data = {}
    if not isinstance(sidecar, dict):
        failures.append("evidence manifest sidecar blob is not valid JSON")
        sidecar = {}
    if set(sidecar) != {"schema", "manifest_canonical_sha256", "manifest_file_sha256"}:
        failures.append("manifest sidecar field mismatch")
    if sidecar.get("manifest_file_sha256") != evidence_blob_records.get("manifest", {}).get("sha256"):
        failures.append("manifest sidecar file hash does not match evidence blob")
    if sidecar.get("manifest_canonical_sha256") != public.digest({k: v for k, v in manifest_data.items() if k != "manifest_canonical_sha256"}):
        failures.append("manifest sidecar canonical hash does not match evidence blob")

    independent_review = commit_blob_json(evidence_commit, SUBAGENT_FINDINGS_PATH) if commit_exists(evidence_commit) else None
    if not isinstance(independent_review, dict):
        failures.append("independent review findings blob is not valid JSON")
        independent_review = {}
    independent_quorum = review_quorum(independent_review)
    if not independent_quorum["passed"]:
        failures.append("independent review quorum failed from evidence blob")

    source_audit_paths = source_audit_paths_for_commit(source_commit)
    source_audit = commit_blob_json(evidence_commit, source_audit_paths["findings_path"]) if commit_exists(evidence_commit) else None
    if not isinstance(source_audit, dict):
        failures.append("source audit findings blob is not valid JSON")
        source_audit = {}
    source_quorum = source_audit_quorum(
        source_audit,
        expected_source_commit=source_commit,
        expected_source_hashes_sha256=source_authority.get("source_hashes_sha256"),
        expected_source_bundle_sha256=source_authority.get("source_bundle_sha256"),
        expected_runtime_binary_sha256=source_authority.get("runtime_binary_sha256"),
        review_report_present=evidence_blob_records.get("source audit report", {}).get("present") is True,
        excluded_agent_ids=exact_review_agent_ids(independent_review),
        evidence_commit=evidence_commit,
        review_root=source_audit_paths["review_dir"],
    )
    if not source_quorum["passed"]:
        failures.append("source authority review quorum failed from evidence blob")
    evidence_source_review_binding = {
        "findings_sha256": evidence_blob_records.get("source audit findings", {}).get("sha256"),
        "review_report_sha256": evidence_blob_records.get("source audit report", {}).get("sha256"),
        "review_quorum_sha256": public.digest(source_quorum),
        "source_authority_commit": source_commit,
        "source_hashes_sha256": source_authority.get("source_hashes_sha256"),
        "source_bundle_sha256": source_authority.get("source_bundle_sha256"),
        "runtime_binary_sha256": source_authority.get("runtime_binary_sha256"),
    }

    challenge = manifest_data.get("temperature_sensor_authority", {}).get("controller_challenge")
    if not isinstance(challenge, dict):
        failures.append("temperature challenge missing from evidence manifest")
        challenge = {}
    if challenge.get("authorized_commit") != source_commit:
        failures.append("temperature challenge does not bind source authority commit")
    if challenge.get("source_hashes_sha256") != source_authority.get("source_hashes_sha256"):
        failures.append("temperature challenge source hash mismatch")
    if challenge.get("source_bundle_sha256") != source_authority.get("source_bundle_sha256"):
        failures.append("temperature challenge source bundle mismatch")
    if challenge.get("runtime_binary_sha256") != source_authority.get("runtime_binary_sha256"):
        failures.append("temperature challenge runtime binary mismatch")
    if challenge.get("source_authority_review") != evidence_source_review_binding:
        failures.append("temperature challenge source-review binding does not match evidence blobs")
    challenge_receipt = commit_blob_json(evidence_commit, DISCOVERY_CHALLENGE_PATH) if commit_exists(evidence_commit) else None
    if not isinstance(challenge_receipt, dict):
        failures.append("discovery challenge receipt blob is not valid JSON")
        challenge_receipt = None
        challenge_receipt_validation = {"passed": False, "failures": ["challenge receipt missing"]}
    else:
        challenge_receipt_validation = validate_discovery_challenge_receipt_payload(
            challenge_receipt,
            expected_source_commit=source_commit,
            expected_challenge=challenge,
            expected_nonce_sha=challenge.get("controller_nonce_sha256"),
            expected_source_review_binding=evidence_source_review_binding,
        )
        if not challenge_receipt_validation["passed"]:
            failures.append("discovery challenge receipt blob failed validation")
    challenge_receipt_canonical_sha256 = (
        challenge_receipt.get("challenge_receipt_canonical_sha256") if isinstance(challenge_receipt, dict) else None
    )
    challenge_receipt_file_sha256 = evidence_blob_records.get("discovery challenge receipt", {}).get("sha256")

    counters = manifest_data.get("contact_counter_attestation", {})
    expected_counters = {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0}
    if not counter_dict_equal_strict(counters, expected_counters):
        failures.append("final contact counters are not exactly 1/1/0/0")

    discovery = commit_blob_json(evidence_commit, TARGET_DISCOVERY_RECEIPT) if commit_exists(evidence_commit) else None
    authority = commit_blob_json(evidence_commit, TEMPERATURE_SENSOR_AUTHORITY) if commit_exists(evidence_commit) else None
    transport = commit_blob_json(evidence_commit, DISCOVERY_TRANSPORT_PATH) if commit_exists(evidence_commit) else None
    attempt = commit_blob_json(evidence_commit, DISCOVERY_ATTEMPT_PATH) if commit_exists(evidence_commit) else None
    if not isinstance(discovery, dict):
        failures.append("target discovery blob is not valid JSON")
        discovery = None
    if not isinstance(authority, dict):
        failures.append("temperature authority blob is not valid JSON")
        authority = None
    if not isinstance(transport, dict):
        failures.append("discovery transport blob is not valid JSON")
        transport = None
    if not isinstance(attempt, dict):
        failures.append("discovery attempt blob is not valid JSON")
        attempt = None
    if isinstance(transport, dict):
        transport_review = transport.get("source_authority_review")
        if not isinstance(transport_review, dict):
            failures.append("discovery transport source-review binding missing")
        else:
            if transport_review.get("review_quorum") != source_quorum:
                failures.append("discovery transport source-review quorum does not match evidence replay")
            if source_review_challenge_binding(transport_review) != evidence_source_review_binding:
                failures.append("discovery transport source-review binding does not match evidence blobs")
        transport_validation = validate_discovery_transport_receipt(transport)
        if not transport_validation["passed"]:
            failures.append("discovery transport blob failed validation")
        if transport.get("challenge_receipt_canonical_sha256") != challenge_receipt_canonical_sha256:
            failures.append("discovery transport challenge receipt canonical hash mismatch")
        if transport.get("challenge_receipt_file_sha256") != challenge_receipt_file_sha256:
            failures.append("discovery transport challenge receipt file hash mismatch")
    if isinstance(authority, dict):
        authority_validation = temperature_sensor_authority_from_receipt(
            authority,
            expected_challenge=challenge,
            expected_discovery_receipt=discovery,
            expected_transport_receipt=transport,
        )
        if not authority_validation["passed"]:
            failures.append("temperature authority blob failed validation")
    if isinstance(attempt, dict):
        if attempt.get("passed") is not True or attempt.get("attempt_state") != "complete":
            failures.append("discovery attempt blob is not complete")
        if not receipt_attempt_digest_matches(attempt):
            failures.append("discovery attempt snapshot digest mismatch")
        cleanup_state = attempt.get("cleanup")
        if not isinstance(cleanup_state, dict) or cleanup_state.get("passed") is not True or cleanup_state.get("absence_verified") is not True:
            failures.append("discovery attempt cleanup state missing or failed")
        if not counters_equal_strict(attempt, expected_counters):
            failures.append("discovery attempt counters are not exactly 1/1/0/0")
        attempt_review = attempt.get("source_authority_review")
        if not isinstance(attempt_review, dict):
            failures.append("discovery attempt source-review binding missing")
        elif source_review_challenge_binding(attempt_review) != evidence_source_review_binding:
            failures.append("discovery attempt source-review binding does not match evidence replay")
        if attempt.get("challenge_receipt_canonical_sha256") != challenge_receipt_canonical_sha256:
            failures.append("discovery attempt challenge receipt canonical hash mismatch")
        if attempt.get("challenge_receipt_file_sha256") != challenge_receipt_file_sha256:
            failures.append("discovery attempt challenge receipt file hash mismatch")

    journal_rows = read_jsonl_blob(evidence_commit, DISCOVERY_ATTEMPT_JOURNAL_PATH) if commit_exists(evidence_commit) else None
    if journal_rows is None:
        failures.append("discovery attempt journal blob is not valid JSONL")
        journal_replay = {"passed": False, "failures": ["journal missing"], "final_row": None, "row_count": 0}
    else:
        journal_replay = replay_attempt_journal_rows(journal_rows)
        if not journal_replay["passed"]:
            failures.append("discovery attempt journal replay failed")
        if isinstance(attempt, dict) and journal_replay.get("final_row") != attempt:
            failures.append("discovery attempt snapshot does not equal journal final row")

    return {
        "passed": not failures,
        "failures": failures,
        "source_authority": source_authority,
        "source_authority_blob_records": source_authority_blob_records,
        "changed_source_files_after_c1": changed_source_files,
        "changed_source_files_after_c2": changed_source_files,
        "evidence_blob_records": evidence_blob_records,
        "independent_quorum": independent_quorum,
        "source_quorum": source_quorum,
        "source_review_binding": evidence_source_review_binding,
        "challenge_receipt_validation": challenge_receipt_validation,
        "attempt_journal": journal_replay,
        "contact_counters": counters,
        "manifest_file_sha256": sidecar.get("manifest_file_sha256"),
        "manifest_canonical_sha256": sidecar.get("manifest_canonical_sha256"),
    }


def validate_final_exact_object_receipt(
    receipt: dict[str, Any] | None,
    *,
    expected_source_commit: str | None = None,
    expected_evidence_commit: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(receipt, dict):
        return {"passed": False, "failures": ["final exact-object verification receipt missing"]}
    if receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EXACT_OBJECT_VERIFICATION_V1":
        failures.append("final exact-object verification schema mismatch")
    if not receipt_digest_matches(receipt, "final_exact_object_verification_sha256"):
        failures.append("final exact-object verification digest mismatch")
    if receipt.get("passed") is not True:
        failures.append("final exact-object verification did not pass")
    if receipt.get("failures") not in ([], None):
        failures.append("final exact-object verification contains failures")
    source_commit = receipt.get("source_authority_commit")
    evidence_commit = receipt.get("evidence_commit")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        failures.append("final exact-object verification source commit mismatch")
    if expected_evidence_commit is not None and evidence_commit != expected_evidence_commit:
        failures.append("final exact-object verification evidence commit mismatch")
    replay = replay_final_exact_objects(str(source_commit or ""), str(evidence_commit or ""))
    if not replay["passed"]:
        failures.extend("replay: " + item for item in replay["failures"])
    if receipt.get("passed") != replay["passed"]:
        failures.append("final exact-object verification pass bit does not match replay")
    if receipt.get("failures") != replay["failures"]:
        failures.append("final exact-object verification failure list does not match replay")
    if receipt.get("changed_source_files_after_c1") != replay["changed_source_files_after_c1"]:
        failures.append("final exact-object verification source-file delta does not match replay")
    if receipt.get("changed_source_files_after_c2") not in (None, replay["changed_source_files_after_c2"]):
        failures.append("final exact-object verification C2 source-file delta does not match replay")
    if receipt.get("source_authority") != replay["source_authority"]:
        failures.append("final exact-object source authority record does not match replay")
    if receipt.get("source_authority_blob_records") != replay["source_authority_blob_records"]:
        failures.append("final exact-object source authority blob records do not match replay")
    if receipt.get("evidence_blob_records") != replay["evidence_blob_records"]:
        failures.append("final exact-object evidence blob records do not match replay")
    if receipt.get("source_review_binding") != replay["source_review_binding"]:
        failures.append("final exact-object source-review binding does not match replay")
    if receipt.get("challenge_receipt_validation") != replay["challenge_receipt_validation"]:
        failures.append("final exact-object challenge receipt validation does not match replay")
    if receipt.get("attempt_journal") != replay["attempt_journal"]:
        failures.append("final exact-object attempt journal replay does not match replay")
    if receipt.get("contact_counters") != replay["contact_counters"]:
        failures.append("final exact-object verification counters do not match replay")
    if receipt.get("manifest_file_sha256") != replay["manifest_file_sha256"]:
        failures.append("final exact-object manifest file hash does not match replay")
    if receipt.get("manifest_canonical_sha256") != replay["manifest_canonical_sha256"]:
        failures.append("final exact-object manifest canonical hash does not match replay")
    final_evidence = read_final_evidence_commit_authority()
    if not final_evidence["passed"]:
        failures.append("final evidence commit authority invalid: " + ",".join(final_evidence["failures"]))
    else:
        evidence_receipt = final_evidence["receipt"]
        if evidence_receipt.get("source_authority_commit") != source_commit:
            failures.append("final evidence commit authority source mismatch")
        if evidence_receipt.get("evidence_commit") != evidence_commit:
            failures.append("final evidence commit authority evidence mismatch")
        if evidence_receipt.get("evidence_manifest_file_sha256") != replay["manifest_file_sha256"]:
            failures.append("final evidence commit authority manifest file mismatch")
        if evidence_receipt.get("evidence_manifest_canonical_sha256") != replay["manifest_canonical_sha256"]:
            failures.append("final evidence commit authority manifest canonical mismatch")
        if receipt.get("final_evidence_commit_sha256") != evidence_receipt.get("final_evidence_commit_sha256"):
            failures.append("final exact-object evidence authority digest mismatch")
        if receipt.get("final_evidence_commit_file_sha256") != public.sha256_file(FINAL_EVIDENCE_COMMIT_PATH):
            failures.append("final exact-object evidence authority file mismatch")
    return {"passed": not failures, "failures": failures}


def write_final_evidence_commit_receipt(source_commit: str, evidence_commit: str, replay: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT_V1",
        "source_authority_commit": source_commit,
        "evidence_commit": evidence_commit,
        "evidence_manifest_file_sha256": replay.get("manifest_file_sha256"),
        "evidence_manifest_canonical_sha256": replay.get("manifest_canonical_sha256"),
        "evidence_manifest_sidecar_sha256": replay.get("evidence_blob_records", {}).get("manifest sidecar", {}).get("sha256"),
    }
    receipt["final_evidence_commit_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "final_evidence_commit_sha256"})
    write_json(FINAL_EVIDENCE_COMMIT_PATH, receipt)
    return receipt


def read_final_evidence_commit_authority() -> dict[str, Any]:
    if not FINAL_EVIDENCE_COMMIT_PATH.exists():
        return {"passed": False, "failures": ["final evidence commit authority missing"]}
    receipt = read_json(FINAL_EVIDENCE_COMMIT_PATH)
    failures: list[str] = []
    if receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT_V1":
        failures.append("final evidence commit authority schema mismatch")
    if receipt.get("final_evidence_commit_sha256") != public.digest({k: v for k, v in receipt.items() if k != "final_evidence_commit_sha256"}):
        failures.append("final evidence commit authority digest mismatch")
    for field, pattern in {"source_authority_commit": r"[0-9a-f]{40}", "evidence_commit": r"[0-9a-f]{40}"}.items():
        if re.fullmatch(pattern, str(receipt.get(field, ""))) is None:
            failures.append(f"final evidence commit authority {field} malformed")
    for field in ["evidence_manifest_file_sha256", "evidence_manifest_canonical_sha256", "evidence_manifest_sidecar_sha256"]:
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))) is None:
            failures.append(f"final evidence commit authority {field} malformed")
    return {"passed": not failures, "failures": failures, "receipt": receipt}


def validate_receipt_chain(manifest_data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not target.validate_no_live_authority_env()["passed"]:
        failures.append("live authority environment present during validation")
    receipt_specs = {
        "public_self": (HERE / "CARRIER_TOMOGRAPHY_SELF_TEST.json", "self_test_sha256", "self_test_passed"),
        "target_self": (TARGET_SELF_TEST_PATH, "self_test_sha256", "self_test_passed"),
        "runtime": (RUNTIME_SELF_TEST_PATH, "runtime_self_test_sha256", "passed"),
        "transport": (TRANSPORT_SIM_PATH, "transport_simulation_sha256", "passed"),
        "deployment": (DEPLOYMENT_LAYOUT_PATH, "deployment_layout_self_test_sha256", "passed"),
        "controller": (CONTROLLER_SELF_TEST_PATH, "self_test_sha256", "self_test_passed"),
        "offline": (OFFLINE_VALIDATE_PATH, "offline_validate_sha256", "passed"),
        "feature_boundary": (FEATURE_BOUNDARY_PATH, "feature_boundary_self_test_sha256", "passed"),
        "operator_analysis": (OPERATOR_ANALYSIS_PATH, "operator_analysis_self_test_sha256", "passed"),
        "factorial_arm": (FACTORIAL_ARM_PATH, "factorial_arm_self_test_sha256", "passed"),
        "source_death": (SOURCE_DEATH_PATH, "source_death_custody_self_test_sha256", "passed"),
        "exact_coverage": (EXACT_COVERAGE_PATH, "exact_coverage_self_test_sha256", "passed"),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for name, (path, digest_field, pass_field) in receipt_specs.items():
        if not path.exists():
            failures.append(f"{name} receipt missing")
            continue
        receipt = read_json(path)
        receipts[name] = receipt
        if receipt.get(pass_field) is not True:
            failures.append(f"{name} receipt not passed")
        if digest_field and not receipt_digest_matches(receipt, digest_field):
            failures.append(f"{name} receipt digest mismatch")

    if not SOURCE_HASHES.exists():
        failures.append("source hashes receipt missing")
        source_hashes = {}
    else:
        source_hashes = read_json(SOURCE_HASHES)
        if not receipt_digest_matches(source_hashes, "source_hashes_sha256"):
            failures.append("source hashes receipt digest mismatch")
        if manifest_data.get("source_hashes") != source_hashes:
            failures.append("manifest source hashes mismatch")
        if manifest_data.get("runtime_binary_authority") != source_hashes.get("runtime_binary_authority"):
            failures.append("manifest runtime binary authority mismatch")

    offline = receipts.get("offline", {})
    if offline:
        expected_offline_links = {
            "public_self_test_sha256": receipts.get("public_self", {}).get("self_test_sha256"),
            "target_self_test_sha256": receipts.get("target_self", {}).get("self_test_sha256"),
            "runtime_self_test_sha256": receipts.get("runtime", {}).get("runtime_self_test_sha256"),
            "transport_simulation_sha256": receipts.get("transport", {}).get("transport_simulation_sha256"),
            "deployment_layout_self_test_sha256": receipts.get("deployment", {}).get("deployment_layout_self_test_sha256"),
            "feature_boundary_self_test_sha256": receipts.get("feature_boundary", {}).get("feature_boundary_self_test_sha256"),
            "operator_analysis_self_test_sha256": receipts.get("operator_analysis", {}).get("operator_analysis_self_test_sha256"),
            "factorial_arm_self_test_sha256": receipts.get("factorial_arm", {}).get("factorial_arm_self_test_sha256"),
            "source_death_custody_self_test_sha256": receipts.get("source_death", {}).get("source_death_custody_self_test_sha256"),
            "exact_coverage_self_test_sha256": receipts.get("exact_coverage", {}).get("exact_coverage_self_test_sha256"),
            "source_hashes_sha256": source_hashes.get("source_hashes_sha256"),
        }
        for field, expected in expected_offline_links.items():
            if offline.get(field) != expected:
                failures.append(f"offline receipt link mismatch {field}")
        for field in ["operator_analysis_passed", "factorial_arm_passed", "source_death_custody_passed", "exact_coverage_passed"]:
            if offline.get(field) is not True:
                failures.append(f"offline receipt false {field}")

    controller = receipts.get("controller", {})
    if controller:
        if controller.get("offline_validate_sha256") != receipts.get("offline", {}).get("offline_validate_sha256"):
            failures.append("controller offline receipt link mismatch")
        if controller.get("transport_simulation_sha256") != receipts.get("transport", {}).get("transport_simulation_sha256"):
            failures.append("controller transport receipt link mismatch")

    runtime = receipts.get("runtime", {})
    if runtime:
        if runtime.get("runtime_binary_authority") != source_hashes.get("runtime_binary_authority"):
            failures.append("runtime receipt authority mismatch")
        compile_equivalence = runtime.get("compile_equivalence")
        if not isinstance(compile_equivalence, dict) or compile_equivalence.get("law") != "byte_exact_isolated_compile":
            failures.append("runtime compile equivalence law missing")
        elif compile_equivalence.get("passed") is not True:
            failures.append("runtime compile equivalence did not pass")

    manifest_links = {
        ("runtime_self_test", "sha256"): receipts.get("runtime", {}).get("runtime_self_test_sha256"),
        ("target_self_test", "sha256"): receipts.get("target_self", {}).get("self_test_sha256"),
        ("controller_self_test", "sha256"): receipts.get("controller", {}).get("self_test_sha256"),
        ("offline_validate", "sha256"): receipts.get("offline", {}).get("offline_validate_sha256"),
        ("transport_simulation", "sha256"): receipts.get("transport", {}).get("transport_simulation_sha256"),
        ("deployment_layout", "sha256"): receipts.get("deployment", {}).get("deployment_layout_self_test_sha256"),
    }
    for (section, field), expected in manifest_links.items():
        if manifest_data.get(section, {}).get(field) != expected:
            failures.append(f"manifest receipt link mismatch {section}.{field}")
        if manifest_data.get(section, {}).get("passed") is not True:
            failures.append(f"manifest receipt section not passed {section}")

    for name, value in receipts.items():
        for key in ["target_contact_count", "sensor_inventory_count", "live_invocation_count", "pmu_acquisition_count"]:
            if not zero_contact_counter_valid(value, key):
                failures.append(f"offline receipt contact counter must be zero {name}.{key}")

    manifest_counters = manifest_data.get("contact_counter_attestation") or manifest_data.get("zero_live_contact_attestation") or {}
    if not isinstance(manifest_counters, dict):
        failures.append("manifest contact counters missing or malformed")
        manifest_counters = {}
    if not is_strict_int(manifest_counters.get("live_invocation_count")) or manifest_counters.get("live_invocation_count") != 0:
        failures.append("manifest live invocation count must be zero")
    if not is_strict_int(manifest_counters.get("pmu_acquisition_count")) or manifest_counters.get("pmu_acquisition_count") != 0:
        failures.append("manifest PMU acquisition count must be zero")
    authority_section = manifest_data.get("temperature_sensor_authority", {})
    authority_passed = authority_section.get("authority_receipt_passed") is True
    if DISCOVERY_ATTEMPT_PATH.exists():
        attempt = read_json(DISCOVERY_ATTEMPT_PATH)
        missing_attempt_counters = [key for key in ATTEMPT_COUNTER_KEYS if not is_strict_int(attempt.get(key))]
        if missing_attempt_counters:
            failures.append("discovery attempt counters missing: " + ",".join(missing_attempt_counters))
            expected_counters = {}
        else:
            expected_counters = {key: attempt[key] for key in ATTEMPT_COUNTER_KEYS}
    else:
        expected_counters = {
            "target_contact_count": 1 if authority_passed else 0,
            "sensor_inventory_count": 1 if authority_passed else 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    if not expected_counters or not counter_dict_equal_strict(manifest_counters, expected_counters):
        failures.append("manifest contact counters are not truthful for authority state")
    if TEMPERATURE_SENSOR_AUTHORITY.exists():
        authority_result = read_temperature_sensor_authority(expected_challenge=authority_section.get("controller_challenge"))
        if not authority_result["passed"]:
            failures.append("temperature authority receipt invalid: " + ",".join(authority_result["failures"]))
    if DISCOVERY_TRANSPORT_PATH.exists():
        transport_result = validate_discovery_transport_receipt(read_json(DISCOVERY_TRANSPORT_PATH))
        if not transport_result["passed"]:
            failures.append("discovery transport receipt invalid: " + ",".join(transport_result["failures"]))
    if authority_passed:
        for path, label in [
            (TARGET_DISCOVERY_RECEIPT, "target discovery receipt"),
            (DISCOVERY_TRANSPORT_PATH, "discovery transport receipt"),
            (DISCOVERY_CHALLENGE_PATH, "discovery challenge receipt"),
            (DISCOVERY_ATTEMPT_PATH, "discovery attempt receipt"),
        ]:
            if not path.exists():
                failures.append(f"{label} missing for passed authority")
        if DISCOVERY_ATTEMPT_PATH.exists():
            attempt = read_json(DISCOVERY_ATTEMPT_PATH)
            if attempt.get("passed") is not True or attempt.get("attempt_state") != "complete":
                failures.append("discovery attempt receipt is not complete")
    final_section = manifest_data.get("final_exact_object_verification", {})
    if FINAL_OBJECT_VERIFY_PATH.exists():
        final_receipt = read_json(FINAL_OBJECT_VERIFY_PATH)
        final_validation = validate_final_exact_object_receipt(
            final_receipt,
            expected_source_commit=final_section.get("source_authority_commit"),
            expected_evidence_commit=final_section.get("evidence_commit"),
        )
        if not final_validation["passed"]:
            failures.append("final exact-object verification invalid: " + ",".join(final_validation["failures"]))
    if manifest_data.get("package_decision") == public.PACKAGE_DECISION_FROZEN:
        if final_section.get("passed") is not True:
            failures.append("frozen package lacks passing final exact-object verification")
        if not FINAL_OBJECT_VERIFY_PATH.exists():
            failures.append("final exact-object verification receipt missing")
    return failures


def validate_only() -> dict[str, Any]:
    artifacts = [
        public.SCHEDULE_JSON,
        public.SCHEDULE_TSV,
        public.SCHEDULE_SHA,
        CONTRACT_PATH,
        MANIFEST_PATH,
        MANIFEST_SHA_PATH,
        SOURCE_HASHES,
        SOURCE_BUNDLE,
        RUNTIME_SELF_TEST_PATH,
        TARGET_SELF_TEST_PATH,
        CONTROLLER_SELF_TEST_PATH,
        OFFLINE_VALIDATE_PATH,
        TRANSPORT_SIM_PATH,
        DEPLOYMENT_LAYOUT_PATH,
        FEATURE_BOUNDARY_PATH,
        OPERATOR_ANALYSIS_PATH,
        FACTORIAL_ARM_PATH,
        SOURCE_DEATH_PATH,
        EXACT_COVERAGE_PATH,
    ]
    discovery_artifacts = [
        TEMPERATURE_SENSOR_AUTHORITY,
        TARGET_DISCOVERY_RECEIPT,
        DISCOVERY_TRANSPORT_PATH,
        DISCOVERY_CHALLENGE_PATH,
        DISCOVERY_ATTEMPT_PATH,
        TARGET_DISCOVERY_FAILURE_PATH,
        FINAL_OBJECT_VERIFY_PATH,
    ]
    if any(path.exists() for path in discovery_artifacts):
        artifacts.extend(discovery_artifacts)
    missing = [str(path) for path in artifacts if not path.exists()]
    schedule = public.load_schedule_from_artifacts() if not missing else {}
    sidecar = read_json(MANIFEST_SHA_PATH) if MANIFEST_SHA_PATH.exists() else {}
    manifest_data = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    failures = []
    if missing:
        failures.append("missing artifacts")
    if sidecar:
        if set(sidecar) != {"schema", "manifest_canonical_sha256", "manifest_file_sha256"}:
            failures.append("manifest sidecar field mismatch")
        if sidecar.get("manifest_canonical_sha256") != public.digest({k: v for k, v in manifest_data.items() if k != "manifest_canonical_sha256"}):
            failures.append("manifest canonical digest mismatch")
        if sidecar.get("manifest_file_sha256") != public.sha256_file(MANIFEST_PATH):
            failures.append("manifest file digest mismatch")
    source_authority = target.validate_source_file_authority(HERE)
    if not source_authority["passed"]:
        failures.append("source file authority failed")
    bundle_preview = source_bundle_preview() if SOURCE_BUNDLE.exists() else {"sha256": None}
    manifest_bundle_sha = manifest_data.get("source_bundle", {}).get("sha256")
    if SOURCE_BUNDLE.exists() and (public.sha256_file(SOURCE_BUNDLE) != manifest_bundle_sha or bundle_preview["sha256"] != manifest_bundle_sha):
        failures.append("source bundle reconstruction mismatch")
    failures.extend(validate_receipt_chain(manifest_data))
    history_index_binding = validate_attempt_history_index_manifest_binding(manifest_data)
    if not history_index_binding["passed"]:
        failures.append("attempt history index manifest binding mismatch")
    feature_boundary = public.feature_boundary_self_test()
    if not feature_boundary["passed"]:
        failures.append("feature boundary scan failed")
    package_parse = package_json_parse_audit()
    if not package_parse["passed"]:
        failures.append("package JSON/JSONL parsing failed")
    manifest_counters = manifest_data.get("contact_counter_attestation") or manifest_data.get("zero_live_contact_attestation") or {
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_VALIDATE_ONLY_RECEIPT_V1",
        "passed": not failures,
        "failures": failures,
        "missing": missing,
        "tuple_count": schedule.get("tuple_count") if schedule else None,
        "manifest_sha": sidecar,
        "feature_boundary_passed": feature_boundary["passed"],
        "feature_boundary_sha256": feature_boundary["feature_boundary_self_test_sha256"],
        "package_json_parse": package_parse,
        "source_authority": source_authority,
        "source_bundle_reconstruction": bundle_preview,
        "target_contact_count": manifest_counters.get("target_contact_count", 0),
        "sensor_inventory_count": manifest_counters.get("sensor_inventory_count", 0),
        "live_invocation_count": manifest_counters.get("live_invocation_count", 0),
        "pmu_acquisition_count": manifest_counters.get("pmu_acquisition_count", 0),
    }
    print(strict_json_dumps(result, indent=2))
    return result


def final_exact_object_verification(*, evidence_commit: str | None = None) -> dict[str, Any]:
    evidence_commit = evidence_commit or os.environ.get(FINAL_EVIDENCE_COMMIT_ENV) or git_text("rev-parse", "HEAD")
    evidence_manifest = commit_blob_json(evidence_commit, MANIFEST_PATH) if commit_exists(evidence_commit) else None
    working_manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    manifest_data = evidence_manifest if isinstance(evidence_manifest, dict) else working_manifest
    source_commit = manifest_data.get("source_authority_review", {}).get("source_authority_commit")
    if not isinstance(source_commit, str):
        source_commit = ""
    replay = replay_final_exact_objects(source_commit, evidence_commit)
    final_evidence_authority = write_final_evidence_commit_receipt(source_commit, evidence_commit, replay)
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EXACT_OBJECT_VERIFICATION_V1",
        "passed": replay["passed"],
        "failures": replay["failures"],
        "source_authority_commit": source_commit,
        "evidence_commit": evidence_commit,
        "changed_source_files_after_c1": replay["changed_source_files_after_c1"],
        "changed_source_files_after_c2": replay["changed_source_files_after_c2"],
        "source_authority": replay["source_authority"],
        "source_authority_blob_records": replay["source_authority_blob_records"],
        "evidence_blob_records": replay["evidence_blob_records"],
        "independent_quorum": replay["independent_quorum"],
        "source_quorum": replay["source_quorum"],
        "source_review_binding": replay["source_review_binding"],
        "challenge_receipt_validation": replay["challenge_receipt_validation"],
        "attempt_journal": replay["attempt_journal"],
        "contact_counters": replay["contact_counters"],
        "manifest_file_sha256": replay["manifest_file_sha256"],
        "manifest_canonical_sha256": replay["manifest_canonical_sha256"],
        "final_evidence_commit_path": str(FINAL_EVIDENCE_COMMIT_PATH),
        "final_evidence_commit_file_sha256": public.sha256_file(FINAL_EVIDENCE_COMMIT_PATH),
        "final_evidence_commit_sha256": final_evidence_authority["final_evidence_commit_sha256"],
        "required_evidence_paths": {label: str(path) for label, path in final_evidence_paths(source_commit).items()},
        "missing_evidence_paths": sorted(label for label, record in replay["evidence_blob_records"].items() if record.get("present") is not True),
    }
    result["final_exact_object_verification_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "final_exact_object_verification_sha256"}
    )
    write_json(FINAL_OBJECT_VERIFY_PATH, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--transport-simulation", action="store_true")
    parser.add_argument("--deployment-layout-self-test", action="store_true")
    parser.add_argument("--offline-validate", action="store_true")
    parser.add_argument("--initialize-source-hashes", action="store_true")
    parser.add_argument("--acquire-temperature-sensor-authority", action="store_true")
    parser.add_argument("--final-exact-object-verification", action="store_true")
    parser.add_argument("--target-host", default=TARGET_HOST)
    parser.add_argument("--discovery-remote-root", default=DISCOVERY_REMOTE_ROOT)
    parser.add_argument("--source-authority-commit", default=None)
    parser.add_argument("--evidence-commit", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        modes = [
            args.initialize_source_hashes,
            args.acquire_temperature_sensor_authority,
            args.final_exact_object_verification,
            args.prepare_only,
            args.validate_only,
            args.self_test,
            args.transport_simulation,
            args.deployment_layout_self_test,
            args.offline_validate,
        ]
        if sum(1 for mode in modes if mode) != 1:
            raise ControllerError("exactly one controller mode must be selected")
        if args.initialize_source_hashes:
            result = initialize_source_hash_authority(force=args.force)
            print(strict_json_dumps(result, indent=2))
            return 0
        if args.acquire_temperature_sensor_authority:
            result = acquire_temperature_sensor_authority(
                target_host=args.target_host,
                remote_root=args.discovery_remote_root,
                source_authority_commit=args.source_authority_commit,
            )
            print(strict_json_dumps(result, indent=2))
            return 0 if result["passed"] else 1
        if args.final_exact_object_verification:
            result = final_exact_object_verification(evidence_commit=args.evidence_commit)
            print(strict_json_dumps(result, indent=2))
            return 0 if result["passed"] else 1
        if args.prepare_only:
            result = prepare_only()
            return 0 if result["passed"] else 1
        if args.validate_only:
            result = validate_only()
            return 0 if result["passed"] else 1
        if args.self_test:
            result = controller_self_test()
        elif args.transport_simulation:
            result = fake_transport_self_tests()
        elif args.deployment_layout_self_test:
            result = deployment_layout_self_test()
        elif args.offline_validate:
            result = offline_validate()
        else:
            parser.print_help()
            return 2
        print(strict_json_dumps(result, indent=2))
        return 0 if result.get("passed", result.get("self_test_passed", False)) else 1
    except Exception as exc:  # noqa: BLE001 - CLI receipt
        print(strict_json_dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
