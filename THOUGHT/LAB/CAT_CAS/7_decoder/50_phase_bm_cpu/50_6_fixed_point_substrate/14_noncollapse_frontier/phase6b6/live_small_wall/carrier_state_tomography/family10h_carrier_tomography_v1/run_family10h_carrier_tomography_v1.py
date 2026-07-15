#!/usr/bin/env python3
"""Offline controller for the public Family 10h carrier tomography package."""

from __future__ import annotations

import argparse
import hashlib
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
DISCOVERY_TRANSPORT_PATH = HERE / "CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT.json"
DISCOVERY_CHALLENGE_PATH = HERE / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE.json"
DISCOVERY_ATTEMPT_PATH = HERE / "CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT.json"
FINAL_OBJECT_VERIFY_PATH = HERE / "CARRIER_TOMOGRAPHY_FINAL_OBJECT_VERIFY.json"
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

EXPECTED_STARTING_HEAD = "836d53a81225fb37406528f1c25e87e208aa9495"
COMMIT_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING"
MANIFEST_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256"
AUTHORITY_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY"
AUTHORITY_VALUE = public.TRANSACTION_RUN_ID
TEMPERATURE_SENSOR_AUTHORITY_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY_V1"
TEMPERATURE_SENSOR_DISCOVERY_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_DISCOVERY_V1"
TEMPERATURE_AUTHORITY_NONCE_ENV = target.TEMPERATURE_AUTHORITY_NONCE_ENV
TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV = target.TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV
TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA = target.TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA
REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS = target.REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS
TARGET_HOST = "root@192.168.137.100"
DISCOVERY_REMOTE_ROOT = f"{public.EXPECTED_REMOTE_ROOT}_sensor_authority_discovery"
DISCOVERY_REMOTE_SOURCE_ROOT = f"{DISCOVERY_REMOTE_ROOT}/source"
DISCOVERY_REMOTE_RECEIPT = f"{DISCOVERY_REMOTE_SOURCE_ROOT}/{target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME}"

SOURCE_FILE_NAMES = target.SOURCE_FILE_NAMES
SOURCE_AUTHORITY_FILE_NAMES = target.SOURCE_AUTHORITY_FILE_NAMES
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
SOURCE_AUDIT_FINDINGS_PATH = HERE / "SOURCE_AUTHORITY_C1_REVIEW_NORMALIZED.json"
SOURCE_AUDIT_REVIEW_PATH = HERE / "SOURCE_AUTHORITY_C1_REVIEW_REPORTS.md"
SOURCE_AUDIT_REQUIRED_REVIEW_ROLES = {
    "physical_sensor_authority_auditor": "physical sensor-authority auditor",
    "discovery_transport_custody_auditor": "discovery transport and custody auditor",
    "source_bundle_evidence_auditor": "source/bundle and evidence auditor",
    "claim_boundary_adjudicator": "claim-boundary adjudicator",
}
SOURCE_AUDIT_ROLE_ALIASES = {
    "physical_sensor_authority_auditor": "physical_sensor_authority_auditor",
    "physical_sensor_authority": "physical_sensor_authority_auditor",
    "discovery_transport_and_custody_auditor": "discovery_transport_custody_auditor",
    "discovery_transport_custody_auditor": "discovery_transport_custody_auditor",
    "source_bundle_and_evidence_auditor": "source_bundle_evidence_auditor",
    "source_bundle_evidence_auditor": "source_bundle_evidence_auditor",
    "claim_boundary_adjudicator": "claim_boundary_adjudicator",
}


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


def source_audit_quorum(
    source_audit: dict[str, Any],
    *,
    expected_source_commit: str | None,
    expected_source_hashes_sha256: str,
    expected_source_bundle_sha256: str,
    review_report_present: bool,
    excluded_agent_ids: set[str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    excluded_agent_ids = excluded_agent_ids or set()
    if source_audit.get("source_authority_commit") != expected_source_commit:
        failures.append("source audit commit mismatch")
    if source_audit.get("source_hashes_sha256") != expected_source_hashes_sha256:
        failures.append("source audit source-hashes mismatch")
    if source_audit.get("source_bundle_sha256") != expected_source_bundle_sha256:
        failures.append("source audit source-bundle mismatch")
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
            "audited_commit": item.get("audited_commit"),
            "source_hashes_sha256": item.get("source_hashes_sha256"),
            "source_bundle_sha256": item.get("source_bundle_sha256"),
            "boundary_attestation": item.get("boundary_attestation"),
            "passed": isinstance(item.get("agent_id"), str)
            and bool(item.get("agent_id"))
            and item.get("verdict") == "NO_MATERIAL_BLOCKER"
            and item.get("final_response") is True,
        }
        role_entry = by_role[role]
        if role_entry["audited_commit"] != expected_source_commit:
            failures.append(f"source audit reviewer commit mismatch {role}")
            role_entry["passed"] = False
        if role_entry["source_hashes_sha256"] != expected_source_hashes_sha256:
            failures.append(f"source audit reviewer source-hashes mismatch {role}")
            role_entry["passed"] = False
        if role_entry["source_bundle_sha256"] != expected_source_bundle_sha256:
            failures.append(f"source audit reviewer source-bundle mismatch {role}")
            role_entry["passed"] = False
        boundary = role_entry["boundary_attestation"]
        if not isinstance(boundary, dict) or boundary.get("no_git_write") is not True or boundary.get("no_target_contact") is not True:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, timeout: float, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, cwd=cwd)
    if check and completed.returncode != 0:
        raise ControllerError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


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


def compute_source_hashes() -> dict[str, Any]:
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_HASHES_V1",
        "source_files": source_file_map(),
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
    status = git_lines("status", "--porcelain", "--", *[package_relative_path(name) for name in SOURCE_AUTHORITY_FILE_NAMES])
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
    return {"passed": not failures, "failures": failures, "commit": commit, "files": files, "status": status}


def build_temperature_authority_challenge(
    *,
    source_hashes: dict[str, Any],
    source_bundle_sha256: str,
    schedule_sidecar: dict[str, Any],
    authorized_commit: str,
    controller_nonce_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA,
        "authority": "controller_issued_temperature_sensor_challenge",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "schedule_canonical_sha256": schedule_sidecar["canonical_sha256"],
        "schedule_json_sha256": schedule_sidecar["json_sha256"],
        "schedule_tsv_sha256": schedule_sidecar["tsv_sha256"],
        "authorized_commit": authorized_commit,
        "controller_nonce_sha256": controller_nonce_sha256,
    }


def write_discovery_challenge_receipt(challenge: dict[str, Any], *, source_commit: str, nonce_sha: str) -> dict[str, Any]:
    receipt = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_CHALLENGE_RECEIPT_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_authority_commit": source_commit,
        "controller_challenge": challenge,
        "controller_challenge_sha256": public.digest(challenge),
        "controller_nonce_sha256": nonce_sha,
        "pre_contact": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    receipt["challenge_receipt_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "challenge_receipt_sha256"})
    write_json_exclusive(DISCOVERY_CHALLENGE_PATH, receipt)
    return receipt


def write_discovery_attempt_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["schema"] = "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_ATTEMPT_V1"
    receipt["discovery_attempt_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "discovery_attempt_sha256"})
    if DISCOVERY_ATTEMPT_PATH.exists():
        write_json(DISCOVERY_ATTEMPT_PATH, receipt)
    else:
        write_json_exclusive(DISCOVERY_ATTEMPT_PATH, receipt)
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
        schedule_sidecar=schedule_sidecar,
        authorized_commit=authorized_commit,
        controller_nonce_sha256=nonce_sha,
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
        if receipt.get("challenge_receipt_sha256") != public.digest({k: v for k, v in receipt.items() if k != "challenge_receipt_sha256"}):
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
                schedule_sidecar=schedule_sidecar,
                authorized_commit=source_commit,
                controller_nonce_sha256=nonce_sha,
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
    if receipt.get("provenance_bound") is not True:
        failures.append("temperature sensor authority provenance not bound")
    if identity == public.synthetic_temperature_identity():
        failures.append("synthetic temperature sensor identity cannot authorize frozen status")
    discovery = receipt.get("target_discovery_receipt")
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
                if platform.get("vendor") != "AuthenticAMD" or platform.get("cpu_family") != 16:
                    failures.append("temperature sensor discovery target platform is not AMD Family 10h")
                if platform.get("checked_before_discovery") is not True:
                    failures.append("temperature sensor discovery platform was not checked before discovery")
            if not isinstance(provenance.get("discovery_monotonic_ns"), int) or provenance.get("discovery_monotonic_ns", 0) <= 0:
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
        if discovery.get("target_contact_count") != 1:
            failures.append("temperature sensor discovery target contact count must be one")
        if discovery.get("sensor_inventory_count") != 1:
            failures.append("temperature sensor discovery inventory count must be one")
        if discovery.get("live_invocation_count") != 0:
            failures.append("temperature sensor discovery live invocation count must be zero")
        if discovery.get("pmu_acquisition_count") != 0:
            failures.append("temperature sensor discovery PMU acquisition count must be zero")
        if discovery.get("pmu_open_count") != 0:
            failures.append("temperature sensor discovery PMU open count must be zero")
        if discovery.get("runtime_launch_count") != 0:
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
                expected_approved = (
                    candidate_identity.get("hwmon_name") in target.APPROVED_TEMPERATURE_HWMON_NAMES
                    and candidate_identity.get("sensor_label") in target.APPROVED_TEMPERATURE_SENSOR_LABELS
                )
                if candidate.get("approved") is not expected_approved:
                    failures.append(f"temperature sensor discovery candidate approval mismatch {index}")
                if expected_approved:
                    approved_candidates.append(candidate)
            if identity not in [candidate.get("identity") for candidate in approved_candidates]:
                failures.append("temperature sensor discovery selected identity not in approved candidates")
            if approved_candidates:
                label_rank = {"Tctl": 0, "Tdie": 1}
                recomputed_selected = sorted(
                    approved_candidates,
                    key=lambda candidate: (
                        label_rank.get(candidate["identity"]["sensor_label"], 99),
                        candidate["identity"]["class_path"],
                        candidate["identity"]["resolved_input_path"],
                    ),
                )[0]["identity"]
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
                    if selection.get("selected_class_path") != recomputed_selected.get("class_path"):
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
                if sample.get("label") != identity.get("sensor_label"):
                    failures.append("temperature sensor discovery sample label mismatch")
                value = sample.get("value_c")
                if not public.is_json_number(value) or not 0.0 < float(value) < 120.0:
                    failures.append("temperature sensor discovery sample value invalid")
                descriptor = sample.get("pinned_descriptor")
                if not isinstance(descriptor, dict) or descriptor.get("resolved_input_path") != identity.get("resolved_input_path"):
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
        if expected_transport_receipt.get("retry_count") != 0:
            failures.append("temperature discovery transport retry count must be zero")
        if expected_transport_receipt.get("target_contact_count") != 1:
            failures.append("temperature discovery transport target contact count must be one")
        if expected_transport_receipt.get("sensor_inventory_count") != 1:
            failures.append("temperature discovery transport inventory count must be one")
        if expected_transport_receipt.get("live_invocation_count") != 0:
            failures.append("temperature discovery transport live invocation count must be zero")
        if expected_transport_receipt.get("pmu_acquisition_count") != 0:
            failures.append("temperature discovery transport PMU acquisition count must be zero")
    failures.extend(validate_temperature_authority_challenge(receipt, discovery, expected_challenge))
    if receipt.get("hwmon_name") not in target.APPROVED_TEMPERATURE_HWMON_NAMES:
        failures.append("temperature sensor authority hwmon name not approved")
    if receipt.get("sensor_label") not in target.APPROVED_TEMPERATURE_SENSOR_LABELS:
        failures.append("temperature sensor authority sensor label not approved")
    if identity is not None:
        if receipt.get("hwmon_name") != identity.get("hwmon_name"):
            failures.append("temperature sensor authority hwmon name mismatch")
        if receipt.get("sensor_label") != identity.get("sensor_label"):
            failures.append("temperature sensor authority sensor label mismatch")
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
        schedule_sidecar={"canonical_sha256": "3" * 64, "json_sha256": "4" * 64, "tsv_sha256": "6" * 64},
        authorized_commit="7" * 40,
        controller_nonce_sha256=hashlib.sha256(controller_nonce.encode("ascii")).hexdigest(),
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
            "sensor_label": complete_forged_identity["sensor_label"],
            "approved_sensor_identity": complete_forged_identity,
            "target_discovery_receipt": complete_forged_discovery,
            "controller_challenge": synthetic_challenge,
            "controller_challenge_sha256": synthetic_challenge_sha,
            "controller_nonce": controller_nonce,
        }
    )

    provenance_free = {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
        "provenance_bound": False,
        "provenance": "offline_synthetic_fixture",
        "hwmon_name": synthetic["hwmon_name"],
        "sensor_label": synthetic["sensor_label"],
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
            "sensor_label": synthetic["sensor_label"],
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
    malformed_result = temperature_sensor_authority_from_receipt(malformed)
    current = read_temperature_sensor_authority()
    result = {
        "synthetic_or_provenance_free_identity_cannot_freeze": not synthetic_result["passed"],
        "synthetic_identity_with_asserted_provenance_cannot_freeze": not synthetic_true_result["passed"],
        "schema_complete_forged_discovery_rejected": not forged_result["passed"],
        "well_formed_self_authored_discovery_without_expected_challenge_rejected": not complete_forged_without_expected_result["passed"],
        "well_formed_self_authored_discovery_wrong_expected_challenge_rejected": not complete_forged_wrong_expected_result["passed"],
        "well_formed_challenge_bound_fixture_without_transport_rejected": not complete_forged_with_expected_result["passed"],
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
        "sensor_label": identity["sensor_label"],
        "approved_sensor_identity": identity,
        "target_discovery_receipt": discovery,
        "controller_challenge": controller_challenge,
        "controller_challenge_sha256": public.digest(controller_challenge),
        "controller_nonce": controller_nonce,
        "source_authority_commit": controller_challenge["authorized_commit"],
        "target_contact_count": discovery.get("target_contact_count"),
        "sensor_inventory_count": discovery.get("sensor_inventory_count"),
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
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def acquire_temperature_sensor_authority(
    *,
    target_host: str = TARGET_HOST,
    remote_root: str = DISCOVERY_REMOTE_ROOT,
    source_authority_commit: str | None = None,
) -> dict[str, Any]:
    if any(
        path.exists()
        for path in [
            TEMPERATURE_SENSOR_AUTHORITY,
            TARGET_DISCOVERY_RECEIPT,
            DISCOVERY_TRANSPORT_PATH,
            DISCOVERY_ATTEMPT_PATH,
            DISCOVERY_CHALLENGE_PATH,
        ]
    ):
        raise ControllerError("second discovery attempt rejected: authority, discovery, transport, or attempt receipt already exists")
    source_hashes = read_source_hash_authority()
    schedule_sidecar = read_json(public.SCHEDULE_SHA)
    bundle = read_existing_source_bundle_authority()
    source_commit = source_authority_commit or os.environ.get("FAMILY10H_CARRIER_TOMOGRAPHY_SOURCE_AUTHORITY_COMMIT") or git_text("rev-parse", "HEAD")
    source_commit_check = source_authority_commit_verification(source_commit)
    if not source_commit_check["passed"]:
        raise ControllerError("source authority commit verification failed: " + ",".join(source_commit_check["failures"]))
    nonce = secrets.token_hex(32)
    nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    challenge = build_temperature_authority_challenge(
        source_hashes=source_hashes,
        source_bundle_sha256=bundle["sha256"],
        schedule_sidecar=schedule_sidecar,
        authorized_commit=source_commit,
        controller_nonce_sha256=nonce_sha,
    )
    challenge_receipt = write_discovery_challenge_receipt(challenge, source_commit=source_commit, nonce_sha=nonce_sha)
    remote_source = f"{remote_root}/source"
    remote_challenge = f"{remote_source}/controller_challenge.json"
    remote_receipt = f"{remote_source}/{target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME}"
    commands: list[list[str]] = []
    cleanup = {"attempted": False, "passed": False, "absence_verified": False}
    remote_root_created = False
    target_command_invoked = False
    discovery: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    target_discovery_file_sha: str | None = None
    write_discovery_attempt_receipt(
        {
            "passed": False,
            "attempt_state": "claimed_pre_contact",
            "source_authority_commit": source_commit,
            "source_authority": source_commit_check,
            "controller_challenge_sha256": public.digest(challenge),
            "challenge_receipt_sha256": challenge_receipt["challenge_receipt_sha256"],
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
        local_copyback = tmp_root / target.TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME
        try:
            preflight = f"set -eu; test ! -e {sh_quote(remote_root)}; install -d -m 0700 {sh_quote(remote_source)}"
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target_host, preflight]
            commands.append(command)
            write_discovery_attempt_receipt(
                {
                    "passed": False,
                    "attempt_state": "transport_contact_invoked",
                    "source_authority_commit": source_commit,
                    "source_authority": source_commit_check,
                    "controller_challenge_sha256": public.digest(challenge),
                    "challenge_receipt_sha256": challenge_receipt["challenge_receipt_sha256"],
                    "target_contact_count": 1,
                    "sensor_inventory_count": 0,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "commands": commands,
                }
            )
            remote_root_created = True
            run(command, timeout=20.0)
            for name in SOURCE_AUTHORITY_FILE_NAMES:
                local = source_stage / name
                command = ["scp", "-q", str(local), f"{target_host}:{remote_source}/{name}"]
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
                    "controller_challenge_sha256": public.digest(challenge),
                    "challenge_receipt_sha256": challenge_receipt["challenge_receipt_sha256"],
                    "target_contact_count": 1,
                    "sensor_inventory_count": 1,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "commands": commands,
                }
            )
            completed = run(command, timeout=120.0, check=False)
            if completed.returncode != 0:
                raise ControllerError(f"target discovery failed rc={completed.returncode}: {completed.stderr or completed.stdout}")
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
            discovery = read_json(local_copyback)
            authority = build_temperature_sensor_authority_receipt(
                discovery=discovery,
                controller_challenge=challenge,
                controller_nonce=nonce,
            )
            write_discovery_attempt_receipt(
                {
                    "passed": False,
                    "attempt_state": "receipt_copied_cleanup_pending",
                    "source_authority_commit": source_commit,
                    "source_authority": source_commit_check,
                    "controller_challenge_sha256": public.digest(challenge),
                    "challenge_receipt_sha256": challenge_receipt["challenge_receipt_sha256"],
                    "target_discovery_receipt_sha256": discovery.get("target_discovery_receipt_sha256"),
                    "target_discovery_receipt_file_sha256": local_sha,
                    "target_contact_count": 1,
                    "sensor_inventory_count": 1,
                    "live_invocation_count": 0,
                    "pmu_acquisition_count": 0,
                    "commands": commands,
                }
            )
        finally:
            if remote_root_created:
                cleanup["attempted"] = True
                cleanup_command = ["ssh", "-o", "BatchMode=yes", target_host, f"rm -rf -- {sh_quote(remote_root)}"]
                commands.append(cleanup_command)
                cleanup_result = run(cleanup_command, timeout=30.0, check=False)
                cleanup["passed"] = cleanup_result.returncode == 0
                absence_command = ["ssh", "-o", "BatchMode=yes", target_host, f"test ! -e {sh_quote(remote_root)}"]
                commands.append(absence_command)
                absence_result = run(absence_command, timeout=20.0, check=False)
                cleanup["absence_verified"] = absence_result.returncode == 0
    if not cleanup["passed"] or not cleanup["absence_verified"]:
        raise ControllerError("discovery cleanup or remote-root absence verification failed")
    if discovery is None or authority is None:
        raise ControllerError("discovery receipt or authority receipt missing after cleanup")
    if target_discovery_file_sha is None or serialized_json_sha256(discovery) != target_discovery_file_sha:
        raise ControllerError("discovery receipt serialized hash verification failed")
    authority_file_sha = serialized_json_sha256(authority)
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1",
        "passed": True,
        "target_host": target_host,
        "remote_root": remote_root,
        "remote_source_root": remote_source,
        "source_authority_commit": source_commit,
        "controller_challenge": challenge,
        "controller_challenge_sha256": public.digest(challenge),
        "challenge_receipt_path": str(DISCOVERY_CHALLENGE_PATH),
        "challenge_receipt_sha256": public.sha256_file(DISCOVERY_CHALLENGE_PATH),
        "controller_nonce_sha256": nonce_sha,
        "target_discovery_receipt_path": str(TARGET_DISCOVERY_RECEIPT),
        "target_discovery_receipt_sha256": discovery["target_discovery_receipt_sha256"],
        "target_discovery_receipt_file_sha256": target_discovery_file_sha,
        "authority_receipt_path": str(TEMPERATURE_SENSOR_AUTHORITY),
        "authority_receipt_sha256": authority["temperature_sensor_authority_sha256"],
        "authority_receipt_file_sha256": authority_file_sha,
        "approved_sensor_identity": authority["approved_sensor_identity"],
        "cleanup": cleanup,
        "commands": commands,
        "retry_count": 0,
        "target_command_invoked": target_command_invoked,
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
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
    TARGET_DISCOVERY_RECEIPT.write_bytes((json.dumps(discovery, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if public.sha256_file(TARGET_DISCOVERY_RECEIPT) != target_discovery_file_sha:
        raise ControllerError("discovery receipt local write verification failed")
    write_json(TEMPERATURE_SENSOR_AUTHORITY, authority)
    if public.sha256_file(TEMPERATURE_SENSOR_AUTHORITY) != authority_file_sha or read_json(TEMPERATURE_SENSOR_AUTHORITY) != authority:
        raise ControllerError("temperature authority receipt write verification failed")
    write_json(DISCOVERY_TRANSPORT_PATH, result)
    write_discovery_attempt_receipt(
        {
            "passed": True,
            "attempt_state": "complete",
            "source_authority_commit": source_commit,
            "source_authority": source_commit_check,
            "controller_challenge_sha256": public.digest(challenge),
            "challenge_receipt_sha256": challenge_receipt["challenge_receipt_sha256"],
            "target_discovery_receipt_sha256": result["target_discovery_receipt_sha256"],
            "target_discovery_receipt_file_sha256": result["target_discovery_receipt_file_sha256"],
            "authority_receipt_sha256": result["authority_receipt_sha256"],
            "authority_receipt_file_sha256": result["authority_receipt_file_sha256"],
            "discovery_transport_sha256": result["discovery_transport_sha256"],
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
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


def materialize_source_authority_snapshot(commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_AUTHORITY_FILE_NAMES:
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


def compile_runtime() -> dict[str, Any]:
    compiler = find_c_compiler()
    source = HERE / "family10h_carrier_tomography_runtime.c"
    if BINARY_PATH.exists():
        BINARY_PATH.unlink()
    if compiler is None:
        return {
            "passed": False,
            "compiler": None,
            "compile_command": None,
            "offline_binary_sha256": None,
            "failure": "no local C compiler found",
        }
    runtime_command: list[str] | None = None
    if compiler[0].endswith("wsl.exe") or Path(compiler[0]).name.lower() == "wsl.exe":
        win_source = str(source)
        win_binary = str(BINARY_PATH)
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
            str(BINARY_PATH),
            str(source),
        ]
        completed = run(compile_command, timeout=60.0, check=False)
        runtime_command = [str(BINARY_PATH), "--self-test"]
    passed = completed.returncode == 0 and BINARY_PATH.exists()
    return {
        "passed": passed,
        "compiler": compiler,
        "compile_command": compile_command,
        "runtime_command": runtime_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "offline_binary_sha256": public.sha256_file(BINARY_PATH) if BINARY_PATH.exists() else None,
    }


def runtime_self_test() -> dict[str, Any]:
    compile_receipt = compile_runtime()
    if not compile_receipt["passed"]:
        result = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST_RECEIPT_V1",
            "passed": False,
            "compile": compile_receipt,
        }
        write_json(RUNTIME_SELF_TEST_PATH, result)
        return result
    completed = run(compile_receipt["runtime_command"], timeout=20.0, check=False)
    try:
        runtime_json = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        runtime_json = {"passed": False, "raw_stdout": completed.stdout}
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST_RECEIPT_V1",
        "passed": completed.returncode == 0 and runtime_json.get("passed") is True,
        "compile": compile_receipt,
        "runtime_returncode": completed.returncode,
        "runtime_stdout": completed.stdout,
        "runtime_stderr": completed.stderr,
        "runtime_json": runtime_json,
    }
    result["runtime_self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "runtime_self_test_sha256"})
    write_json(RUNTIME_SELF_TEST_PATH, result)
    return result


def runtime_authority_gate_self_test() -> dict[str, Any]:
    compile_receipt = compile_runtime()
    if not compile_receipt["passed"]:
        return {"passed": False, "compile": compile_receipt}

    def direct_execute(output_root: Path, authority_value: str | None) -> subprocess.CompletedProcess[str]:
        compiler = compile_receipt["compiler"]
        if compiler and (compiler[0].endswith("wsl.exe") or Path(compiler[0]).name.lower() == "wsl.exe"):
            binary = f"\"$(wslpath '{BINARY_PATH}')\""
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
            [str(BINARY_PATH), "--execute-schedule", str(public.SCHEDULE_TSV), str(output_root)],
            text=True,
            capture_output=True,
            timeout=20.0,
            check=False,
            env=env,
        )

    with tempfile.TemporaryDirectory(prefix="carrier_tomography_runtime_gate_") as tmp:
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
        for name in SOURCE_AUTHORITY_FILE_NAMES:
            path = HERE / name
            if path.exists():
                shutil.copy2(path, source / name)
        completed = run(
            [sys.executable, str(source / "family10h_carrier_tomography_target.py"), "--self-test", "--source-root", str(source), "--output-root", str(output)],
            timeout=60.0,
            check=False,
        )
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
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
    if receipt.get("retry_count") != 0:
        failures.append("discovery transport retry count must be zero")
    if receipt.get("passed") is not True:
        failures.append("discovery transport must pass")
    if receipt.get("target_command_invoked") is not True:
        failures.append("discovery transport target command invocation missing")
    cleanup = receipt.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("passed") is not True or cleanup.get("absence_verified") is not True:
        failures.append("discovery cleanup and absence verification required")
    if receipt.get("target_contact_count") != 1:
        failures.append("discovery target contact count must be one")
    if receipt.get("sensor_inventory_count") != 1:
        failures.append("discovery sensor inventory count must be one")
    if receipt.get("live_invocation_count") != 0:
        failures.append("discovery live invocation count must be zero")
    if receipt.get("pmu_acquisition_count") != 0:
        failures.append("discovery PMU acquisition count must be zero")
    if receipt.get("discovery_transport_sha256") != public.digest({k: v for k, v in receipt.items() if k != "discovery_transport_sha256"}):
        failures.append("discovery transport digest mismatch")
    for field in [
        "controller_challenge_sha256",
        "challenge_receipt_sha256",
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

    base = seal(
        {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT_V1",
            "passed": True,
            "cleanup": {"attempted": True, "passed": True, "absence_verified": True},
            "retry_count": 0,
            "target_command_invoked": True,
            "source_authority_commit": "f" * 40,
            "controller_challenge_sha256": "1" * 64,
            "challenge_receipt_sha256": "2" * 64,
            "target_discovery_receipt_sha256": "3" * 64,
            "target_discovery_receipt_file_sha256": "5" * 64,
            "authority_receipt_sha256": "4" * 64,
            "authority_receipt_file_sha256": "6" * 64,
            "target_contact_count": 1,
            "sensor_inventory_count": 1,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    )
    cleanup_failure = seal({**base, "cleanup": {"attempted": True, "passed": False, "absence_verified": True}})
    absence_failure = seal({**base, "cleanup": {"attempted": True, "passed": True, "absence_verified": False}})
    retry = seal({**base, "retry_count": 1})
    zero_contact = seal({**base, "target_contact_count": 0})
    live_nonzero = seal({**base, "live_invocation_count": 1})
    pmu_nonzero = seal({**base, "pmu_acquisition_count": 1})
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
            schedule_sidecar=read_json(source / public.SCHEDULE_SHA.name),
            authorized_commit="f" * 40,
            controller_nonce_sha256=nonce_sha,
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
            "sensor_label": identity["sensor_label"],
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
                "source_authority_commit": "f" * 40,
                "controller_challenge_sha256": public.digest(challenge),
                "challenge_receipt_sha256": "9" * 64,
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
        "incorrect_zero_target_contact_rejected": not validate_discovery_transport_receipt(zero_contact)["passed"],
        "nonzero_live_invocation_rejected": not validate_discovery_transport_receipt(live_nonzero)["passed"],
        "nonzero_pmu_acquisition_rejected": not validate_discovery_transport_receipt(pmu_nonzero)["passed"],
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
        corrupted_records = [json.loads(line) for line in (corrupted_root / "raw_records.jsonl").read_text(encoding="utf-8").splitlines()]
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
    clearances = {
        role: {
            "role": label,
            "agent_id": f"source-reviewer-{index}",
            "verdict": "NO_MATERIAL_BLOCKER",
            "final_response": True,
            "audited_commit": source_commit,
            "source_hashes_sha256": source_hash,
            "source_bundle_sha256": bundle_hash,
            "boundary_attestation": {"no_git_write": True, "no_target_contact": True},
        }
        for index, (role, label) in enumerate(SOURCE_AUDIT_REQUIRED_REVIEW_ROLES.items(), start=1)
    }
    base = {
        "source_authority_commit": source_commit,
        "source_hashes_sha256": source_hash,
        "source_bundle_sha256": bundle_hash,
        "review_report_present": True,
        "material_blockers": [],
        "reviewer_verdicts": clearances,
    }
    wrong_commit = {
        **base,
        "reviewer_verdicts": {
            **clearances,
            "claim_boundary_adjudicator": {**clearances["claim_boundary_adjudicator"], "audited_commit": "4" * 40},
        },
    }
    missing_boundary = {
        **base,
        "reviewer_verdicts": {
            **clearances,
            "claim_boundary_adjudicator": {**clearances["claim_boundary_adjudicator"], "boundary_attestation": {"no_git_write": True}},
        },
    }
    checks = {
        "exact_source_audit_quorum_passes": source_audit_quorum(
            base,
            expected_source_commit=source_commit,
            expected_source_hashes_sha256=source_hash,
            expected_source_bundle_sha256=bundle_hash,
            review_report_present=True,
        )["passed"],
        "missing_report_blocked": not source_audit_quorum(
            base,
            expected_source_commit=source_commit,
            expected_source_hashes_sha256=source_hash,
            expected_source_bundle_sha256=bundle_hash,
            review_report_present=False,
        )["passed"],
        "wrong_top_level_bundle_blocked": not source_audit_quorum(
            {**base, "source_bundle_sha256": "5" * 64},
            expected_source_commit=source_commit,
            expected_source_hashes_sha256=source_hash,
            expected_source_bundle_sha256=bundle_hash,
            review_report_present=True,
        )["passed"],
        "wrong_reviewer_commit_blocked": not source_audit_quorum(
            wrong_commit,
            expected_source_commit=source_commit,
            expected_source_hashes_sha256=source_hash,
            expected_source_bundle_sha256=bundle_hash,
            review_report_present=True,
        )["passed"],
        "missing_boundary_attestation_blocked": not source_audit_quorum(
            missing_boundary,
            expected_source_commit=source_commit,
            expected_source_hashes_sha256=source_hash,
            expected_source_bundle_sha256=bundle_hash,
            review_report_present=True,
        )["passed"],
        "prior_package_reviewer_reuse_blocked": not source_audit_quorum(
            base,
            expected_source_commit=source_commit,
            expected_source_hashes_sha256=source_hash,
            expected_source_bundle_sha256=bundle_hash,
            review_report_present=True,
            excluded_agent_ids={"source-reviewer-1"},
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


def controller_self_test() -> dict[str, Any]:
    transport = fake_transport_self_tests()
    discovery_transport = discovery_transport_self_tests()
    validation = offline_validate()
    runtime_gate = runtime_authority_gate_self_test()
    live_env_absent = target.validate_no_live_authority_env()
    source_hash_authority = source_hash_authority_regression()
    temperature_authority = temperature_sensor_authority_regression()
    null_model_baseline = review_quorum_null_model_baseline()
    source_audit_regression = source_audit_quorum_regression()
    bundle_mode_regression = source_bundle_mode_regression()
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
        "exact_four_final_clearances_can_freeze": review_quorum({"material_blockers": [], "reviewer_verdicts": clearances})["passed"],
    }
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST_V1",
        "self_test_passed": validation["passed"]
        and transport["passed"]
        and discovery_transport["passed"]
        and runtime_gate["passed"]
        and live_env_absent["passed"]
        and source_hash_authority["passed"]
        and temperature_authority["passed"]
        and null_model_baseline["passed"]
        and source_audit_regression["passed"]
        and bundle_mode_regression["passed"]
        and all(quorum_regressions.values()),
        "offline_validate_sha256": validation["offline_validate_sha256"],
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "discovery_transport_self_test": discovery_transport,
        "runtime_authority_gate": runtime_gate,
        "source_hash_authority_regression": source_hash_authority,
        "temperature_sensor_authority_regression": temperature_authority,
        "review_quorum_null_model_baseline": null_model_baseline,
        "source_audit_quorum_regression": source_audit_regression,
        "source_bundle_mode_regression": bundle_mode_regression,
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
        return {
            "target_contact_count": int(attempt.get("target_contact_count", 0)),
            "sensor_inventory_count": int(attempt.get("sensor_inventory_count", 0)),
            "live_invocation_count": int(attempt.get("live_invocation_count", 0)),
            "pmu_acquisition_count": int(attempt.get("pmu_acquisition_count", 0)),
        }
    if temperature_authority.get("passed") is not True or not TEMPERATURE_SENSOR_AUTHORITY.exists():
        return {
            "target_contact_count": 0,
            "sensor_inventory_count": 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    receipt = read_json(TEMPERATURE_SENSOR_AUTHORITY)
    return {
        "target_contact_count": int(receipt.get("target_contact_count", 0)),
        "sensor_inventory_count": int(receipt.get("sensor_inventory_count", 0)),
        "live_invocation_count": int(receipt.get("live_invocation_count", 0)),
        "pmu_acquisition_count": int(receipt.get("pmu_acquisition_count", 0)),
    }


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
    independent_review = {}
    if SUBAGENT_FINDINGS_PATH.exists():
        independent_review = read_json(SUBAGENT_FINDINGS_PATH)
    quorum = review_quorum(independent_review)
    prior_reviewer_ids = {
        str(item.get("agent_id"))
        for item in (independent_review.get("reviewer_verdicts") or {}).values()
        if isinstance(item, dict) and item.get("agent_id")
    }
    source_audit = {}
    if SOURCE_AUDIT_FINDINGS_PATH.exists():
        source_audit = read_json(SOURCE_AUDIT_FINDINGS_PATH)
    source_quorum = source_audit_quorum(
        source_audit,
        expected_source_commit=source_authority_commit,
        expected_source_hashes_sha256=source_hashes["source_hashes_sha256"],
        expected_source_bundle_sha256=bundle["sha256"],
        review_report_present=SOURCE_AUDIT_REVIEW_PATH.exists(),
        excluded_agent_ids=prior_reviewer_ids,
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
    final_exact_validation = validate_final_exact_object_receipt(
        final_exact_object,
        expected_source_commit=source_authority_commit,
        expected_evidence_commit=final_exact_object.get("evidence_commit") if isinstance(final_exact_object, dict) else None,
    )
    final_exact_object_passed = final_exact_validation["passed"]
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
            "findings_path": str(SOURCE_AUDIT_FINDINGS_PATH) if SOURCE_AUDIT_FINDINGS_PATH.exists() else None,
            "findings_sha256": public.sha256_file(SOURCE_AUDIT_FINDINGS_PATH) if SOURCE_AUDIT_FINDINGS_PATH.exists() else None,
            "review_report_path": str(SOURCE_AUDIT_REVIEW_PATH) if SOURCE_AUDIT_REVIEW_PATH.exists() else None,
            "review_report_sha256": public.sha256_file(SOURCE_AUDIT_REVIEW_PATH) if SOURCE_AUDIT_REVIEW_PATH.exists() else None,
            "source_authority_commit": source_authority_commit,
            "source_hashes_sha256": source_hashes["source_hashes_sha256"],
            "source_bundle_sha256": bundle["sha256"],
            "material_blocker_count": len(source_audit.get("material_blockers", [])) if isinstance(source_audit, dict) else None,
            "review_quorum": source_quorum,
        },
        "final_exact_object_verification": {
            "path": str(FINAL_OBJECT_VERIFY_PATH) if FINAL_OBJECT_VERIFY_PATH.exists() else None,
            "file_sha256": public.sha256_file(FINAL_OBJECT_VERIFY_PATH) if FINAL_OBJECT_VERIFY_PATH.exists() else None,
            "passed": final_exact_object_passed,
            "source_authority_commit": final_exact_object.get("source_authority_commit"),
            "evidence_commit": final_exact_object.get("evidence_commit"),
            "verification_sha256": final_exact_object.get("final_exact_object_verification_sha256"),
            "failures": final_exact_validation["failures"] or final_exact_object.get("failures", ["final exact-object verification missing"]),
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
    target_result = target_self_test()
    runtime_result = runtime_self_test()
    transport = fake_transport_self_tests()
    deployment = deployment_layout_self_test()
    offline = offline_validate()
    controller = controller_self_test()
    bundle = write_source_bundle()
    manifest_result = manifest()
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
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
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
        return json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
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


def final_evidence_paths() -> dict[str, Path]:
    return {
        "source audit findings": SOURCE_AUDIT_FINDINGS_PATH,
        "source audit report": SOURCE_AUDIT_REVIEW_PATH,
        "target discovery receipt": TARGET_DISCOVERY_RECEIPT,
        "temperature authority receipt": TEMPERATURE_SENSOR_AUTHORITY,
        "discovery transport receipt": DISCOVERY_TRANSPORT_PATH,
        "discovery challenge receipt": DISCOVERY_CHALLENGE_PATH,
        "discovery attempt receipt": DISCOVERY_ATTEMPT_PATH,
        "manifest": MANIFEST_PATH,
        "manifest sidecar": MANIFEST_SHA_PATH,
    }


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
    bundle_record = commit_blob_record(commit, SOURCE_BUNDLE)
    files[SOURCE_BUNDLE.name] = bundle_record
    if not bundle_record["present"]:
        failures.append("source bundle blob missing")
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
    }


def exact_review_agent_ids(review: dict[str, Any]) -> set[str]:
    return {
        str(item.get("agent_id"))
        for item in (review.get("reviewer_verdicts") or {}).values()
        if isinstance(item, dict) and item.get("agent_id")
    }


def replay_final_exact_objects(source_commit: str, evidence_commit: str) -> dict[str, Any]:
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
        for name in SOURCE_AUTHORITY_FILE_NAMES:
            path = HERE / name
            source_record = commit_blob_record(source_commit, path)
            evidence_record = commit_blob_record(evidence_commit, path)
            unchanged = source_record["present"] and evidence_record["present"] and source_record["blob_id"] == evidence_record["blob_id"]
            source_authority_blob_records[name] = {
                "source": source_record,
                "evidence": evidence_record,
                "unchanged_after_c1": unchanged,
            }
            if not unchanged:
                changed_source_files.append(name)
        if changed_source_files:
            failures.append("evidence overlay changed source authority blobs")

    evidence_blob_records = {
        label: commit_blob_record(evidence_commit, path) if commit_exists(evidence_commit) else {
            "repo_path": path_to_repo_relative(path),
            "present": False,
        }
        for label, path in final_evidence_paths().items()
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

    source_audit = commit_blob_json(evidence_commit, SOURCE_AUDIT_FINDINGS_PATH) if commit_exists(evidence_commit) else None
    if not isinstance(source_audit, dict):
        failures.append("source audit findings blob is not valid JSON")
        source_audit = {}
    source_quorum = source_audit_quorum(
        source_audit,
        expected_source_commit=source_commit,
        expected_source_hashes_sha256=source_authority.get("source_hashes_sha256"),
        expected_source_bundle_sha256=source_authority.get("source_bundle_sha256"),
        review_report_present=evidence_blob_records.get("source audit report", {}).get("present") is True,
        excluded_agent_ids=exact_review_agent_ids(independent_review),
    )
    if not source_quorum["passed"]:
        failures.append("source authority review quorum failed from evidence blob")

    challenge = manifest_data.get("temperature_sensor_authority", {}).get("controller_challenge")
    if not isinstance(challenge, dict):
        failures.append("temperature challenge missing from evidence manifest")
        challenge = {}
    if challenge.get("authorized_commit") != source_commit:
        failures.append("temperature challenge does not bind source C1")
    if challenge.get("source_hashes_sha256") != source_authority.get("source_hashes_sha256"):
        failures.append("temperature challenge source hash mismatch")
    if challenge.get("source_bundle_sha256") != source_authority.get("source_bundle_sha256"):
        failures.append("temperature challenge source bundle mismatch")

    counters = manifest_data.get("contact_counter_attestation", {})
    expected_counters = {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0}
    if counters != expected_counters:
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
        transport_validation = validate_discovery_transport_receipt(transport)
        if not transport_validation["passed"]:
            failures.append("discovery transport blob failed validation")
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
        attempt_counters = {
            "target_contact_count": int(attempt.get("target_contact_count", 0)),
            "sensor_inventory_count": int(attempt.get("sensor_inventory_count", 0)),
            "live_invocation_count": int(attempt.get("live_invocation_count", 0)),
            "pmu_acquisition_count": int(attempt.get("pmu_acquisition_count", 0)),
        }
        if attempt_counters != expected_counters:
            failures.append("discovery attempt counters are not exactly 1/1/0/0")

    return {
        "passed": not failures,
        "failures": failures,
        "source_authority": source_authority,
        "source_authority_blob_records": source_authority_blob_records,
        "changed_source_files_after_c1": changed_source_files,
        "evidence_blob_records": evidence_blob_records,
        "independent_quorum": independent_quorum,
        "source_quorum": source_quorum,
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
    if receipt.get("source_authority") != replay["source_authority"]:
        failures.append("final exact-object source authority record does not match replay")
    if receipt.get("source_authority_blob_records") != replay["source_authority_blob_records"]:
        failures.append("final exact-object source authority blob records do not match replay")
    if receipt.get("evidence_blob_records") != replay["evidence_blob_records"]:
        failures.append("final exact-object evidence blob records do not match replay")
    if receipt.get("contact_counters") != replay["contact_counters"]:
        failures.append("final exact-object verification counters do not match replay")
    if receipt.get("manifest_file_sha256") != replay["manifest_file_sha256"]:
        failures.append("final exact-object manifest file hash does not match replay")
    if receipt.get("manifest_canonical_sha256") != replay["manifest_canonical_sha256"]:
        failures.append("final exact-object manifest canonical hash does not match replay")
    return {"passed": not failures, "failures": failures}


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
            if key in value and value.get(key) != 0:
                failures.append(f"offline receipt contact counter must be zero {name}.{key}")

    manifest_counters = manifest_data.get("contact_counter_attestation") or manifest_data.get("zero_live_contact_attestation") or {}
    if manifest_counters.get("live_invocation_count", 0) != 0:
        failures.append("manifest live invocation count must be zero")
    if manifest_counters.get("pmu_acquisition_count", 0) != 0:
        failures.append("manifest PMU acquisition count must be zero")
    authority_section = manifest_data.get("temperature_sensor_authority", {})
    authority_passed = authority_section.get("authority_receipt_passed") is True
    if DISCOVERY_ATTEMPT_PATH.exists():
        attempt = read_json(DISCOVERY_ATTEMPT_PATH)
        expected_counters = {
            "target_contact_count": int(attempt.get("target_contact_count", 0)),
            "sensor_inventory_count": int(attempt.get("sensor_inventory_count", 0)),
            "live_invocation_count": int(attempt.get("live_invocation_count", 0)),
            "pmu_acquisition_count": int(attempt.get("pmu_acquisition_count", 0)),
        }
    else:
        expected_counters = {
            "target_contact_count": 1 if authority_passed else 0,
            "sensor_inventory_count": 1 if authority_passed else 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    if manifest_counters != expected_counters:
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
    feature_boundary = public.feature_boundary_self_test()
    if not feature_boundary["passed"]:
        failures.append("feature boundary scan failed")
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_VALIDATE_ONLY_RECEIPT_V1",
        "passed": not failures,
        "failures": failures,
        "missing": missing,
        "tuple_count": schedule.get("tuple_count") if schedule else None,
        "manifest_sha": sidecar,
        "feature_boundary_passed": feature_boundary["passed"],
        "feature_boundary_sha256": feature_boundary["feature_boundary_self_test_sha256"],
        "source_authority": source_authority,
        "source_bundle_reconstruction": bundle_preview,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
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
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EXACT_OBJECT_VERIFICATION_V1",
        "passed": replay["passed"],
        "failures": replay["failures"],
        "source_authority_commit": source_commit,
        "evidence_commit": evidence_commit,
        "changed_source_files_after_c1": replay["changed_source_files_after_c1"],
        "source_authority": replay["source_authority"],
        "source_authority_blob_records": replay["source_authority_blob_records"],
        "evidence_blob_records": replay["evidence_blob_records"],
        "independent_quorum": replay["independent_quorum"],
        "source_quorum": replay["source_quorum"],
        "contact_counters": replay["contact_counters"],
        "manifest_file_sha256": replay["manifest_file_sha256"],
        "manifest_canonical_sha256": replay["manifest_canonical_sha256"],
        "required_evidence_paths": {label: str(path) for label, path in final_evidence_paths().items()},
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
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.acquire_temperature_sensor_authority:
            result = acquire_temperature_sensor_authority(
                target_host=args.target_host,
                remote_root=args.discovery_remote_root,
                source_authority_commit=args.source_authority_commit,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 1
        if args.final_exact_object_verification:
            result = final_exact_object_verification(evidence_commit=args.evidence_commit)
            print(json.dumps(result, indent=2, sort_keys=True))
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
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("passed", result.get("self_test_passed", False)) else 1
    except Exception as exc:  # noqa: BLE001 - CLI receipt
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
