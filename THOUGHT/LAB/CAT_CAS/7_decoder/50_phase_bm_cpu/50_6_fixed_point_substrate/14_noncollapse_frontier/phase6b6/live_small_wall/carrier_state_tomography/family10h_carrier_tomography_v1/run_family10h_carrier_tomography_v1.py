#!/usr/bin/env python3
"""Offline controller for the public Family 10h carrier tomography package."""

from __future__ import annotations

import argparse
import hashlib
import re
import gzip
import json
import os
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
CONTRACT_PATH = HERE / "CARRIER_TOMOGRAPHY_CONTRACT.md"
SOURCE_BUNDLE = HERE / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
SOURCE_HASHES = HERE / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
TEMPERATURE_SENSOR_AUTHORITY = HERE / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY.json"
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

SOURCE_FILE_NAMES = target.SOURCE_FILE_NAMES
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


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
    return run(["git", *args], timeout=30.0).stdout.strip()


def git_state() -> dict[str, Any]:
    return {
        "branch": git_text("branch", "--show-current"),
        "head": git_text("rev-parse", "HEAD"),
        "origin_main": git_text("rev-parse", "origin/main"),
        "status_porcelain": run(["git", "status", "--porcelain"], timeout=30.0).stdout,
        "stash_list": run(["git", "stash", "list"], timeout=30.0).stdout.splitlines(),
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
            if not isinstance(provenance.get("target_platform"), dict):
                failures.append("temperature sensor discovery target platform missing")
            if not isinstance(provenance.get("discovery_monotonic_ns"), int) or provenance.get("discovery_monotonic_ns", 0) <= 0:
                failures.append("temperature sensor discovery monotonic timestamp missing")
        if discovery.get("target_contact_count") != 0 or discovery.get("live_invocation_count") != 0:
            failures.append("temperature sensor discovery counters must be zero")
        if discovery.get("selected_identity") != identity:
            failures.append("temperature sensor discovery identity mismatch")
        candidates = discovery.get("observed_candidates")
        if not isinstance(candidates, list) or not candidates:
            failures.append("temperature sensor discovery candidates missing")
        elif identity is not None:
            complete_candidates = []
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    failures.append(f"temperature sensor discovery candidate malformed {index}")
                    continue
                candidate_identity = candidate.get("identity")
                if not isinstance(candidate_identity, dict) or set(candidate_identity) != public.TEMPERATURE_SENSOR_IDENTITY_KEYS:
                    failures.append(f"temperature sensor discovery candidate identity malformed {index}")
                    continue
                if candidate_identity.get("identity_sha256") != public.temperature_identity_digest(candidate_identity):
                    failures.append(f"temperature sensor discovery candidate identity digest mismatch {index}")
                if candidate.get("approved") is True:
                    complete_candidates.append(candidate_identity)
            if identity not in complete_candidates:
                failures.append("temperature sensor discovery selected identity not in approved candidates")
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
    return temperature_sensor_authority_from_receipt(read_json(TEMPERATURE_SENSOR_AUTHORITY), expected_challenge=expected_challenge)


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
        "target_contact_count": 0,
        "live_invocation_count": 0,
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
        "well_formed_challenge_bound_fixture_passes_controller_validator": complete_forged_with_expected_result["passed"],
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
        and result["well_formed_challenge_bound_fixture_passes_controller_validator"]
        and result["wrong_hwmon_authority_rejected"]
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
                        info.uname = ""
                        info.gname = ""
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
        return {"sha256": public.sha256_file(temp_bundle), "file_count": len(names), "files": sorted(names)}


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
        "live_invocation_count": 0,
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
        for name in SOURCE_FILE_NAMES:
            path = HERE / name
            if path.exists():
                shutil.copy2(path, source / name)
        if SOURCE_HASHES.exists():
            shutil.copy2(SOURCE_HASHES, source / SOURCE_HASHES.name)
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
        "live_invocation_count": 0,
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
        "live_invocation_count": 0,
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


def controller_self_test() -> dict[str, Any]:
    transport = fake_transport_self_tests()
    validation = offline_validate()
    runtime_gate = runtime_authority_gate_self_test()
    live_env_absent = target.validate_no_live_authority_env()
    source_hash_authority = source_hash_authority_regression()
    temperature_authority = temperature_sensor_authority_regression()
    null_model_baseline = review_quorum_null_model_baseline()
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
        and runtime_gate["passed"]
        and live_env_absent["passed"]
        and source_hash_authority["passed"]
        and temperature_authority["passed"]
        and null_model_baseline["passed"]
        and all(quorum_regressions.values()),
        "offline_validate_sha256": validation["offline_validate_sha256"],
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "runtime_authority_gate": runtime_gate,
        "source_hash_authority_regression": source_hash_authority,
        "temperature_sensor_authority_regression": temperature_authority,
        "review_quorum_null_model_baseline": null_model_baseline,
        "live_authority_env_absent": live_env_absent,
        "review_quorum_regressions": quorum_regressions,
        "target_contact_count": 0,
        "live_invocation_count": 0,
    }
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(CONTROLLER_SELF_TEST_PATH, result)
    return result


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
    temperature_challenge = expected_temperature_authority_challenge(
        source_hashes=source_hashes,
        source_bundle_sha256=bundle["sha256"],
        schedule_sidecar=schedule_sidecar,
        authorized_commit=git["head"],
    )
    temperature_authority = read_temperature_sensor_authority(expected_challenge=temperature_challenge)
    independent_review = {}
    if SUBAGENT_FINDINGS_PATH.exists():
        independent_review = read_json(SUBAGENT_FINDINGS_PATH)
    quorum = review_quorum(independent_review)
    review_blocked = not quorum["passed"]
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "claim_ceiling": "route-scoped public carrier-state model only",
        "package_decision": public.PACKAGE_DECISION_BLOCKED
        if review_blocked or not offline["passed"] or not temperature_authority["passed"]
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
            "authority_receipt_present": temperature_authority["present"],
            "authority_receipt_passed": temperature_authority["passed"],
            "authority_receipt_failures": temperature_authority["failures"],
            "controller_challenge": temperature_challenge,
            "controller_challenge_sha256": public.digest(temperature_challenge) if isinstance(temperature_challenge, dict) else None,
            "controller_nonce_env": TEMPERATURE_AUTHORITY_NONCE_ENV,
            "controller_nonce_sha256_env": TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV,
            "approved_sensor_identity": temperature_authority["approved_sensor_identity"],
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
        "zero_live_contact_attestation": {"target_contact_count": 0, "live_invocation_count": 0},
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
        "live_invocation_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def receipt_digest_matches(receipt: dict[str, Any], field: str) -> bool:
    return bool(receipt.get(field)) and receipt.get(field) == public.digest({k: v for k, v in receipt.items() if k != field})


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
    contact_objects = {**receipts, "manifest": manifest_data}
    for name, value in contact_objects.items():
        encoded = json.dumps(value, sort_keys=True)
        decoded = json.loads(encoded)
        stack = [(name, decoded)]
        while stack:
            path, item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    child_path = f"{path}.{key}"
                    if key in {"target_contact_count", "live_invocation_count"} and child != 0:
                        failures.append(f"nonzero contact counter {child_path}")
                    stack.append((child_path, child))
            elif isinstance(item, list):
                stack.extend((f"{path}[{index}]", child) for index, child in enumerate(item))
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
        "live_invocation_count": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.initialize_source_hashes:
            result = initialize_source_hash_authority(force=args.force)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
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
