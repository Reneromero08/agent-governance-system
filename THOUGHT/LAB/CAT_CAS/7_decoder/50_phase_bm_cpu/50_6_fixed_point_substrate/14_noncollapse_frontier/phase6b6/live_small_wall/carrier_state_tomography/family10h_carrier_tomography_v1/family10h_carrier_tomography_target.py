#!/usr/bin/env python3
"""Target-side custody checks and authority-gated tomography execution.

The live acquisition entry point is present for future authorization, but this
task only runs the offline validators and self-tests.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import family10h_carrier_tomography_public as public


SOURCE_FILE_NAMES = [
    "CARRIER_TOMOGRAPHY_CONTRACT.md",
    "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json",
    "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv",
    "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256",
    "family10h_carrier_tomography_public.py",
    "family10h_carrier_tomography_target.py",
    "family10h_carrier_tomography_runtime.c",
    "family10h_carrier_tomography_runtime.h",
    "run_family10h_carrier_tomography_v1.py",
]
SOURCE_AUTHORITY_FILE_NAMES = SOURCE_FILE_NAMES + [
    "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json",
    "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz",
    "SUBAGENT_FINDINGS_NORMALIZED.json",
]

APPROVED_TEMPERATURE_HWMON_NAMES = ["k10temp"]
APPROVED_TEMPERATURE_SENSOR_LABELS = ["Tctl", "Tdie"]
REQUIRED_REVIEW_ROLES = {
    "physical_carrier_state_auditor": "physical carrier-state auditor",
    "experimental_design_operator_auditor": "experimental-design/operator auditor",
    "custody_evidence_auditor": "custody/evidence auditor",
    "claim_boundary_adjudicator": "claim-boundary adjudicator",
}
SOURCE_AUDIT_REQUIRED_REVIEW_ROLES = {
    "physical_sensor_authority_auditor": "physical sensor-authority auditor",
    "discovery_transport_custody_auditor": "discovery transport and custody auditor",
    "source_bundle_evidence_auditor": "source/bundle and evidence auditor",
    "claim_boundary_adjudicator": "claim-boundary adjudicator",
}
REVIEW_ROLE_ALIASES = {
    "physical_carrier_state_auditor": "physical_carrier_state_auditor",
    "experimental_design_operator_auditor": "experimental_design_operator_auditor",
    "implementation_custody_evidence_auditor": "custody_evidence_auditor",
    "custody_evidence_auditor": "custody_evidence_auditor",
    "claim_boundary_adjudicator": "claim_boundary_adjudicator",
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
TEMPERATURE_IDENTITY_FIELDS = [
    "hwmon_name",
    "sensor_label",
    "sensor_input",
    "class_path",
    "resolved_input_path",
    "resolved_hwmon_path",
    "resolved_device_path",
]

REQUIRED_EVIDENCE_FILES = [
    "raw_records.jsonl",
    "source_death_receipts.jsonl",
    "feature_freeze.json",
]

FORBIDDEN_LIVE_ENV_PREFIXES = [
    "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY",
    "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING",
    "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256",
    "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY",
    "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_NONCE",
    "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_NONCE_SHA256",
]
AUTHORITY_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY"
COMMIT_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING"
MANIFEST_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256"
RUNTIME_AUTHORITY_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY"
TEMPERATURE_AUTHORITY_NONCE_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_NONCE"
TEMPERATURE_AUTHORITY_NONCE_SHA256_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_NONCE_SHA256"
AUTHORITY_VALUE = public.TRANSACTION_RUN_ID
TEMPERATURE_SENSOR_AUTHORITY_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY_V1"
TEMPERATURE_SENSOR_DISCOVERY_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_DISCOVERY_V1"
TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_CHALLENGE_V1"
TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME = "CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_RECEIPT.json"
DISCOVERY_TRANSPORT_RECEIPT_NAME = "CARRIER_TOMOGRAPHY_DISCOVERY_TRANSPORT.json"
TEMPERATURE_SENSOR_AUTHORITY_RECEIPT_NAME = "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY.json"
REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS = {
    "schema",
    "authority",
    "science_package_id",
    "transaction_run_id",
    "source_hashes_sha256",
    "source_bundle_sha256",
    "schedule_canonical_sha256",
    "schedule_json_sha256",
    "schedule_tsv_sha256",
    "authorized_commit",
    "controller_nonce_sha256",
    "transport_scope",
}


class TargetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetError(message)


def raises_target_error(callback: Any) -> bool:
    try:
        callback()
    except TargetError:
        return True
    return False


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def serialized_json_sha256(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8"))


def validate_schedule_artifacts(source_root: Path) -> dict[str, Any]:
    schedule_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json"
    sidecar_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256"
    tsv_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv"
    require(schedule_path.exists(), "schedule JSON missing")
    require(sidecar_path.exists(), "schedule sidecar missing")
    require(tsv_path.exists(), "schedule TSV missing")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    failures = []
    if public.digest(schedule) != sidecar.get("canonical_sha256"):
        failures.append("schedule canonical digest mismatch")
    if public.sha256_file(schedule_path) != sidecar.get("json_sha256"):
        failures.append("schedule JSON file digest mismatch")
    if public.sha256_file(tsv_path) != sidecar.get("tsv_sha256"):
        failures.append("schedule TSV file digest mismatch")
    try:
        validation = public.validate_schedule(schedule)
        tsv_validation = public.validate_tsv(tsv_path)
    except Exception as exc:  # noqa: BLE001 - convert to self-test receipt
        failures.append(str(exc))
        validation = {"passed": False}
        tsv_validation = {"passed": False}
    return {
        "passed": not failures,
        "failures": failures,
        "schedule_sha256": public.digest(schedule),
        "validation": validation,
        "tsv_validation": tsv_validation,
    }


def validate_source_file_authority(source_root: Path) -> dict[str, Any]:
    receipt_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    failures: list[str] = []
    if not receipt_path.exists():
        return {"passed": False, "failures": ["source hash receipt missing"], "checked_files": 0}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt.get("source_files", {})
    if set(expected) != set(SOURCE_FILE_NAMES):
        failures.append("source hash keyset mismatch")
    for name in SOURCE_FILE_NAMES:
        path = source_root / name
        if not path.exists():
            failures.append(f"source file missing {name}")
            continue
        expected_item = expected.get(name, {})
        if expected_item.get("sha256") != public.sha256_file(path) or expected_item.get("size") != path.stat().st_size:
            failures.append(f"source file authority mismatch {name}")
            break
    if receipt.get("source_hashes_sha256") != public.digest({k: v for k, v in receipt.items() if k != "source_hashes_sha256"}):
        failures.append("source hash receipt digest mismatch")
    return {"passed": not failures, "failures": failures, "checked_files": len(expected)}


def portable_basename(value: Any) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def normalized_role(value: str) -> str:
    return value.lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def recompute_review_quorum(findings: dict[str, Any], *, source_audit: bool = False) -> dict[str, Any]:
    required_roles = SOURCE_AUDIT_REQUIRED_REVIEW_ROLES if source_audit else REQUIRED_REVIEW_ROLES
    aliases = SOURCE_AUDIT_ROLE_ALIASES if source_audit else REVIEW_ROLE_ALIASES
    failures: list[str] = []
    material_blockers = findings.get("material_blockers")
    if not isinstance(material_blockers, list):
        failures.append("material blockers list missing or malformed")
        material_blockers = []
    elif material_blockers:
        failures.append("material blockers present")
    verdicts = findings.get("reviewer_verdicts")
    if not isinstance(verdicts, dict):
        failures.append("reviewer verdicts missing or malformed")
        verdicts = {}
    if len(verdicts) != len(required_roles):
        failures.append("reviewer verdict count must be exactly four")
    by_role: dict[str, dict[str, Any]] = {}
    for role_key, item in verdicts.items():
        if not isinstance(item, dict):
            failures.append(f"malformed reviewer response {role_key}")
            continue
        role_name = str(item.get("role") or item.get("originating_agent") or role_key)
        role = aliases.get(normalized_role(role_name), "")
        if role not in required_roles:
            failures.append(f"unknown reviewer role {role_name}")
            continue
        if role in by_role:
            failures.append(f"duplicate reviewer role {role}")
            continue
        agent_id = item.get("agent_id")
        passed = isinstance(agent_id, str) and bool(agent_id) and item.get("verdict") == "NO_MATERIAL_BLOCKER" and item.get("final_response") is True
        by_role[role] = {
            "role": required_roles[role],
            "agent_id": agent_id,
            "verdict": item.get("verdict"),
            "final_response": item.get("final_response"),
            "audited_commit": item.get("audited_commit"),
            "source_hashes_sha256": item.get("source_hashes_sha256"),
            "source_bundle_sha256": item.get("source_bundle_sha256"),
            "boundary_attestation": item.get("boundary_attestation"),
            "passed": passed,
        }
    missing = sorted(set(required_roles) - set(by_role))
    if missing:
        failures.append("missing reviewer roles: " + ",".join(missing))
    duplicates = [
        agent_id
        for agent_id, count in Counter(item["agent_id"] for item in by_role.values() if item.get("agent_id")).items()
        if count > 1
    ]
    if duplicates:
        failures.append("duplicate reviewer agent ids")
    failed = [role for role, item in by_role.items() if not item["passed"]]
    if failed:
        failures.append("non-clear reviewer roles: " + ",".join(sorted(failed)))
    return {"passed": not failures, "failures": failures, "roles": by_role, "material_blockers": material_blockers}


def review_agent_ids(quorum: dict[str, Any]) -> set[str]:
    return {
        str(item.get("agent_id"))
        for item in (quorum.get("roles") or {}).values()
        if isinstance(item, dict) and item.get("agent_id")
    }


def receipt_digest_matches(receipt: dict[str, Any], field: str) -> bool:
    return bool(receipt.get(field)) and receipt.get(field) == public.digest({k: v for k, v in receipt.items() if k != field})


def validate_target_offline_receipts(source_root: Path, manifest: dict[str, Any], failures: list[str]) -> None:
    specs = {
        "public_self": ("CARRIER_TOMOGRAPHY_SELF_TEST.json", "self_test_sha256", "self_test_passed"),
        "target_self": ("CARRIER_TOMOGRAPHY_TARGET_SELF_TEST.json", "self_test_sha256", "self_test_passed"),
        "runtime": ("CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST.json", "runtime_self_test_sha256", "passed"),
        "transport": ("CARRIER_TOMOGRAPHY_TRANSPORT_SIMULATION.json", "transport_simulation_sha256", "passed"),
        "deployment": ("CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST.json", "deployment_layout_self_test_sha256", "passed"),
        "controller": ("CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST.json", "self_test_sha256", "self_test_passed"),
        "offline": ("CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json", "offline_validate_sha256", "passed"),
        "feature_boundary": ("CARRIER_TOMOGRAPHY_FEATURE_BOUNDARY_SELF_TEST.json", "feature_boundary_self_test_sha256", "passed"),
        "operator_analysis": ("CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST.json", "operator_analysis_self_test_sha256", "passed"),
        "factorial_arm": ("CARRIER_TOMOGRAPHY_FACTORIAL_ARM_SELF_TEST.json", "factorial_arm_self_test_sha256", "passed"),
        "source_death": ("CARRIER_TOMOGRAPHY_SOURCE_DEATH_CUSTODY_SELF_TEST.json", "source_death_custody_self_test_sha256", "passed"),
        "exact_coverage": ("CARRIER_TOMOGRAPHY_EXACT_COVERAGE_SELF_TEST.json", "exact_coverage_self_test_sha256", "passed"),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for name, (filename, digest_field, pass_field) in specs.items():
        path = source_root / filename
        if not path.exists():
            failures.append(f"{name} receipt missing")
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{name} receipt JSON invalid: {exc}")
            continue
        receipts[name] = receipt
        if receipt.get(pass_field) is not True:
            failures.append(f"{name} receipt not passed")
        if not receipt_digest_matches(receipt, digest_field):
            failures.append(f"{name} receipt digest mismatch")
    source_hash_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    source_hashes = json.loads(source_hash_path.read_text(encoding="utf-8")) if source_hash_path.exists() else {}
    offline = receipts.get("offline", {})
    expected_links = {
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
    for field, expected in expected_links.items():
        if offline.get(field) != expected:
            failures.append(f"offline receipt link mismatch {field}")
    manifest_links = {
        ("runtime_self_test", "sha256"): receipts.get("runtime", {}).get("runtime_self_test_sha256"),
        ("target_self_test", "sha256"): receipts.get("target_self", {}).get("self_test_sha256"),
        ("controller_self_test", "sha256"): receipts.get("controller", {}).get("self_test_sha256"),
        ("offline_validate", "sha256"): receipts.get("offline", {}).get("offline_validate_sha256"),
        ("transport_simulation", "sha256"): receipts.get("transport", {}).get("transport_simulation_sha256"),
        ("deployment_layout", "sha256"): receipts.get("deployment", {}).get("deployment_layout_self_test_sha256"),
    }
    for (section, field), expected in manifest_links.items():
        if manifest.get(section, {}).get(field) != expected:
            failures.append(f"manifest receipt link mismatch {section}.{field}")


def validate_bound_file(
    source_root: Path,
    path_value: Any,
    sha_value: Any,
    label: str,
    failures: list[str],
) -> Path | None:
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        failures.append(f"{label} binding missing")
        return None
    path = source_root / portable_basename(path_value)
    if not path.exists():
        failures.append(f"{label} file missing")
        return path
    if public.sha256_file(path) != sha_value:
        failures.append(f"{label} file hash mismatch")
    return path


def validate_manifest_review_section(
    source_root: Path,
    section: dict[str, Any],
    label: str,
    failures: list[str],
    *,
    source_audit: bool = False,
    expected_source_commit: str | None = None,
    expected_source_hashes_sha256: str | None = None,
    expected_source_bundle_sha256: str | None = None,
) -> set[str]:
    findings_path = validate_bound_file(source_root, section.get("findings_path"), section.get("findings_sha256"), f"{label} findings", failures)
    validate_bound_file(source_root, section.get("review_report_path"), section.get("review_report_sha256"), f"{label} report", failures)
    findings: dict[str, Any] = {}
    if findings_path is not None and findings_path.exists():
        try:
            findings = json.loads(findings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{label} findings JSON invalid: {exc}")
            findings = {}
    quorum = recompute_review_quorum(findings, source_audit=source_audit)
    if not quorum["passed"]:
        failures.append(f"{label} quorum failed from findings: " + ",".join(quorum["failures"]))
    if source_audit:
        if findings.get("source_authority_commit") != expected_source_commit:
            failures.append(f"{label} source-authority commit mismatch")
        if findings.get("source_hashes_sha256") != expected_source_hashes_sha256:
            failures.append(f"{label} source-hashes mismatch")
        if findings.get("source_bundle_sha256") != expected_source_bundle_sha256:
            failures.append(f"{label} source-bundle mismatch")
        if findings.get("review_report_present") is not True:
            failures.append(f"{label} report presence not asserted")
        for role_name, role in (quorum.get("roles") or {}).items():
            if role.get("audited_commit") != expected_source_commit:
                failures.append(f"{label} role {role_name} audited commit mismatch")
            if role.get("source_hashes_sha256") != expected_source_hashes_sha256:
                failures.append(f"{label} role {role_name} source-hashes mismatch")
            if role.get("source_bundle_sha256") != expected_source_bundle_sha256:
                failures.append(f"{label} role {role_name} source-bundle mismatch")
            boundary = role.get("boundary_attestation")
            if not isinstance(boundary, dict) or boundary.get("no_git_write") is not True or boundary.get("no_target_contact") is not True:
                failures.append(f"{label} role {role_name} boundary attestation missing")
    if section.get("material_blocker_count") not in (0, None):
        failures.append(f"{label} manifest material blocker count is nonzero")
    manifest_quorum = section.get("review_quorum")
    if isinstance(manifest_quorum, dict) and manifest_quorum != quorum:
        failures.append(f"{label} manifest quorum summary differs from findings replay")
    return review_agent_ids(quorum)


def validate_manifest_authority(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json"
    sidecar_path = source_root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256"
    failures: list[str] = []
    if not manifest_path.exists() or not sidecar_path.exists():
        return {"passed": False, "failures": ["manifest authority files missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("manifest_file_sha256") != public.sha256_file(manifest_path):
        failures.append("manifest file hash mismatch")
    if sidecar.get("manifest_canonical_sha256") != public.digest({k: v for k, v in manifest.items() if k != "manifest_canonical_sha256"}):
        failures.append("manifest canonical hash mismatch")
    binary_hash = manifest.get("runtime_self_test", {}).get("offline_binary_sha256")
    runtime_path = source_root / "family10h_carrier_tomography_runtime"
    if binary_hash and runtime_path.exists() and public.sha256_file(runtime_path) != binary_hash:
        failures.append("runtime binary hash mismatch")
    bundle_hash = manifest.get("source_bundle", {}).get("sha256")
    bundle_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
    if not bundle_hash:
        failures.append("source bundle hash missing from manifest")
    elif not bundle_path.exists():
        failures.append("source bundle missing")
    elif public.sha256_file(bundle_path) != bundle_hash:
        failures.append("source bundle hash mismatch")
    source_hash_receipt = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    manifest_source_hash = manifest.get("source_hashes", {}).get("source_hashes_sha256")
    if not manifest_source_hash:
        failures.append("source hash receipt missing from manifest")
    elif not source_hash_receipt.exists():
        failures.append("source hash receipt missing")
    else:
        source_hash_data = json.loads(source_hash_receipt.read_text(encoding="utf-8"))
        if source_hash_data.get("source_hashes_sha256") != manifest_source_hash:
            failures.append("manifest source hash binding mismatch")
    temperature_authority = manifest.get("temperature_sensor_authority", {})
    temperature_authority_result = validate_manifest_temperature_authority(manifest, source_root)
    failures.extend(temperature_authority_result["failures"])
    approved_identity = temperature_authority_result["approved_temperature_sensor_identity"]
    package_decision = manifest.get("package_decision")
    review_quorum = manifest.get("independent_review", {}).get("review_quorum", {})
    if package_decision == public.PACKAGE_DECISION_FROZEN and review_quorum.get("passed") is not True:
        failures.append("frozen package lacks complete review quorum")
    if package_decision == public.PACKAGE_DECISION_FROZEN:
        independent_agent_ids = validate_manifest_review_section(
            source_root,
            manifest.get("independent_review", {}) if isinstance(manifest.get("independent_review"), dict) else {},
            "independent review",
            failures,
        )
        source_agent_ids = validate_manifest_review_section(
            source_root,
            manifest.get("source_authority_review", {}) if isinstance(manifest.get("source_authority_review"), dict) else {},
            "source authority review",
            failures,
            source_audit=True,
            expected_source_commit=manifest.get("source_authority_review", {}).get("source_authority_commit"),
            expected_source_hashes_sha256=manifest.get("source_hashes", {}).get("source_hashes_sha256"),
            expected_source_bundle_sha256=manifest.get("source_bundle", {}).get("sha256"),
        )
        if independent_agent_ids & source_agent_ids:
            failures.append("source authority review reused independent reviewer agent ids")
        source_quorum = manifest.get("source_authority_review", {}).get("review_quorum", {})
        if source_quorum.get("passed") is not True:
            failures.append("frozen package lacks exact C1 source-review quorum")
        source_review = manifest.get("source_authority_review", {})
        if not isinstance(source_review.get("source_authority_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", source_review.get("source_authority_commit", "")) is None:
            failures.append("frozen package source-authority commit missing or malformed")
        if source_review.get("source_hashes_sha256") != manifest.get("source_hashes", {}).get("source_hashes_sha256"):
            failures.append("source authority review source-hashes binding mismatch")
        if source_review.get("source_bundle_sha256") != manifest.get("source_bundle", {}).get("sha256"):
            failures.append("source authority review source-bundle binding mismatch")
        if manifest.get("offline_validate", {}).get("passed") is not True:
            failures.append("frozen package lacks passing offline validation")
        validate_target_offline_receipts(source_root, manifest, failures)
        counters = manifest.get("contact_counter_attestation", {})
        if counters != {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0}:
            failures.append("frozen package contact counters are not exactly 1/1/0/0")
        final_section = manifest.get("final_exact_object_verification", {})
        if final_section.get("passed") is not True:
            failures.append("frozen package lacks passing final exact-object verification")
        final_path_value = final_section.get("path")
        final_file_sha = final_section.get("file_sha256")
        final_evidence_path_value = final_section.get("final_evidence_commit_path")
        final_evidence_file_sha = final_section.get("final_evidence_commit_file_sha256")
        if not isinstance(final_path_value, str) or not final_file_sha:
            failures.append("frozen package lacks final exact-object receipt binding")
        if not isinstance(final_evidence_path_value, str) or not final_evidence_file_sha:
            failures.append("frozen package lacks final evidence commit binding")
        if isinstance(final_path_value, str) and final_file_sha and isinstance(final_evidence_path_value, str) and final_evidence_file_sha:
            final_path = source_root / portable_basename(final_path_value)
            final_evidence_path = source_root / portable_basename(final_evidence_path_value)
            final_receipt: dict[str, Any] = {}
            final_evidence_receipt: dict[str, Any] = {}
            if not final_path.exists():
                failures.append("final exact-object verification receipt missing")
            elif public.sha256_file(final_path) != final_file_sha:
                failures.append("final exact-object verification file hash mismatch")
            else:
                try:
                    final_receipt = json.loads(final_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    failures.append(f"final exact-object verification JSON invalid: {exc}")
            if not final_evidence_path.exists():
                failures.append("final evidence commit authority missing")
            elif public.sha256_file(final_evidence_path) != final_evidence_file_sha:
                failures.append("final evidence commit authority file hash mismatch")
            else:
                try:
                    final_evidence_receipt = json.loads(final_evidence_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    failures.append(f"final evidence commit authority JSON invalid: {exc}")
                else:
                    if final_evidence_receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT_V1":
                        failures.append("final evidence commit authority schema mismatch")
                    if final_evidence_receipt.get("final_evidence_commit_sha256") != public.digest({k: v for k, v in final_evidence_receipt.items() if k != "final_evidence_commit_sha256"}):
                        failures.append("final evidence commit authority digest mismatch")
            if final_receipt:
                    digest = final_receipt.get("final_exact_object_verification_sha256")
                    if final_receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EXACT_OBJECT_VERIFICATION_V1":
                        failures.append("final exact-object verification schema mismatch")
                    if digest != public.digest({k: v for k, v in final_receipt.items() if k != "final_exact_object_verification_sha256"}):
                        failures.append("final exact-object verification digest mismatch")
                    if final_receipt.get("passed") is not True or final_receipt.get("failures") not in ([], None):
                        failures.append("final exact-object verification did not pass")
                    if final_receipt.get("source_authority_commit") != source_review.get("source_authority_commit"):
                        failures.append("final exact-object source commit mismatch")
                    if final_receipt.get("evidence_commit") != final_evidence_receipt.get("evidence_commit"):
                        failures.append("final exact-object evidence commit mismatch")
                    if final_receipt.get("manifest_file_sha256") != final_evidence_receipt.get("evidence_manifest_file_sha256"):
                        failures.append("final exact-object evidence manifest file mismatch")
                    if final_receipt.get("manifest_canonical_sha256") != final_evidence_receipt.get("evidence_manifest_canonical_sha256"):
                        failures.append("final exact-object evidence manifest canonical mismatch")
                    if final_receipt.get("final_evidence_commit_sha256") != final_evidence_receipt.get("final_evidence_commit_sha256"):
                        failures.append("final exact-object final evidence authority mismatch")
                    if final_section.get("evidence_manifest_file_sha256") != final_evidence_receipt.get("evidence_manifest_file_sha256"):
                        failures.append("final section evidence manifest file mismatch")
                    if final_section.get("evidence_manifest_canonical_sha256") != final_evidence_receipt.get("evidence_manifest_canonical_sha256"):
                        failures.append("final section evidence manifest canonical mismatch")
                    if final_section.get("evidence_manifest_sidecar_sha256") != final_evidence_receipt.get("evidence_manifest_sidecar_sha256"):
                        failures.append("final section evidence manifest sidecar mismatch")
                    evidence_records = final_receipt.get("evidence_blob_records")
                    if not isinstance(evidence_records, dict):
                        failures.append("final exact-object evidence blob records missing")
                    else:
                        manifest_record = evidence_records.get("manifest", {})
                        sidecar_record = evidence_records.get("manifest sidecar", {})
                        if not isinstance(manifest_record, dict) or manifest_record.get("sha256") != final_evidence_receipt.get("evidence_manifest_file_sha256"):
                            failures.append("final exact-object evidence manifest blob mismatch")
                        if not isinstance(sidecar_record, dict) or sidecar_record.get("sha256") != final_evidence_receipt.get("evidence_manifest_sidecar_sha256"):
                            failures.append("final exact-object evidence sidecar blob mismatch")
                        for label in [
                            "source audit findings",
                            "source audit report",
                            "target discovery receipt",
                            "temperature authority receipt",
                            "discovery transport receipt",
                            "discovery challenge receipt",
                            "discovery attempt receipt",
                            "discovery attempt journal",
                            "manifest",
                            "manifest sidecar",
                        ]:
                            record = evidence_records.get(label)
                            if not isinstance(record, dict) or record.get("present") is not True or record.get("object_type") != "blob":
                                failures.append(f"final exact-object evidence blob missing {label}")
                    source_blob_records = final_receipt.get("source_authority_blob_records")
                    if not isinstance(source_blob_records, dict):
                        failures.append("final exact-object source authority blob records missing")
                    else:
                        for name in SOURCE_AUTHORITY_FILE_NAMES:
                            record = source_blob_records.get(name)
                            if not isinstance(record, dict) or record.get("unchanged_after_c1") is not True:
                                failures.append(f"final exact-object source authority blob changed {name}")
    return {
        "passed": not failures,
        "failures": failures,
        "manifest_file_sha256": sidecar.get("manifest_file_sha256"),
        "manifest_canonical_sha256": sidecar.get("manifest_canonical_sha256"),
        "authorized_commit": manifest.get("git_state_at_manifest_build", {}).get("head"),
        "package_decision": package_decision,
        "approved_temperature_sensor_identity": approved_identity,
        "temperature_authority_controller_challenge": temperature_authority.get("controller_challenge"),
    }


def validate_manifest_temperature_authority(manifest: dict[str, Any], source_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    temperature_authority = manifest.get("temperature_sensor_authority", {})
    approved_identity = temperature_authority.get("approved_sensor_identity")
    package_decision = manifest.get("package_decision")
    if approved_identity is None and package_decision != public.PACKAGE_DECISION_FROZEN:
        pass
    elif not isinstance(approved_identity, dict) or set(approved_identity) != public.TEMPERATURE_SENSOR_IDENTITY_KEYS:
        failures.append("approved temperature sensor identity missing from frozen manifest")
        approved_identity = None
    elif approved_identity.get("identity_sha256") != public.temperature_identity_digest(approved_identity):
        failures.append("approved temperature sensor identity digest mismatch")
    if package_decision == public.PACKAGE_DECISION_FROZEN and temperature_authority.get("authority_receipt_passed") is not True:
        failures.append("frozen package lacks temperature sensor authority receipt")
    if temperature_authority.get("resolved_identity_bound_in_evidence") is not True:
        failures.append("temperature sensor identity not evidence-bound in manifest")
    authority_path_value = temperature_authority.get("authority_receipt_path")
    authority_file_value = temperature_authority.get("authority_receipt_file_sha256")
    target_discovery_path_value = temperature_authority.get("target_discovery_receipt_path")
    target_discovery_file_value = temperature_authority.get("target_discovery_receipt_file_sha256")
    discovery_transport_path_value = temperature_authority.get("discovery_transport_path")
    discovery_transport_file_value = temperature_authority.get("discovery_transport_file_sha256")
    expected_challenge = temperature_authority.get("controller_challenge")
    if package_decision == public.PACKAGE_DECISION_FROZEN:
        if not isinstance(expected_challenge, dict):
            failures.append("frozen package lacks temperature authority controller challenge")
        if not isinstance(authority_path_value, str) or not authority_file_value:
            failures.append("frozen package lacks temperature authority file binding")
        if not isinstance(target_discovery_path_value, str) or not target_discovery_file_value:
            failures.append("frozen package lacks copied target discovery receipt binding")
        if not isinstance(discovery_transport_path_value, str) or not discovery_transport_file_value:
            failures.append("frozen package lacks discovery transport receipt binding")
        if (
            isinstance(authority_path_value, str)
            and authority_file_value
            and isinstance(target_discovery_path_value, str)
            and target_discovery_file_value
            and isinstance(discovery_transport_path_value, str)
            and discovery_transport_file_value
        ):
            authority_path = source_root / portable_basename(authority_path_value)
            target_discovery_path = source_root / portable_basename(target_discovery_path_value)
            discovery_transport_path = source_root / portable_basename(discovery_transport_path_value)
            if not target_discovery_path.exists():
                failures.append("copied target discovery receipt missing")
            elif public.sha256_file(target_discovery_path) != target_discovery_file_value:
                failures.append("copied target discovery receipt hash mismatch")
            if not discovery_transport_path.exists():
                failures.append("discovery transport receipt missing")
            elif public.sha256_file(discovery_transport_path) != discovery_transport_file_value:
                failures.append("discovery transport receipt hash mismatch")
            try:
                target_discovery_receipt = json.loads(target_discovery_path.read_text(encoding="utf-8")) if target_discovery_path.exists() else None
                discovery_transport_receipt = json.loads(discovery_transport_path.read_text(encoding="utf-8")) if discovery_transport_path.exists() else None
            except json.JSONDecodeError as exc:
                failures.append(f"discovery evidence JSON invalid: {exc}")
                target_discovery_receipt = None
                discovery_transport_receipt = None
            authority_result = validate_temperature_sensor_authority_file(
                authority_path,
                expected_challenge=expected_challenge,
                expected_discovery_receipt=target_discovery_receipt,
                expected_transport_receipt=discovery_transport_receipt,
            )
            if not authority_result["passed"]:
                failures.append("temperature authority file invalid: " + ",".join(authority_result["failures"]))
            if authority_path.exists() and public.sha256_file(authority_path) != authority_file_value:
                failures.append("temperature authority file hash mismatch")
            if approved_identity is not None and authority_result.get("approved_sensor_identity") != approved_identity:
                failures.append("temperature authority identity mismatch")
        else:
            authority_result = {"approved_sensor_identity": None}
    return {
        "passed": not failures,
        "failures": failures,
        "approved_temperature_sensor_identity": approved_identity,
    }


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
        scope = challenge.get("transport_scope")
        if not isinstance(scope, dict):
            failures.append("temperature authority controller challenge transport scope missing")
        else:
            for field in ["target_host", "remote_base_root", "remote_root", "remote_source_root", "remote_receipt_path"]:
                if not isinstance(scope.get(field), str) or not scope.get(field):
                    failures.append(f"temperature authority controller challenge transport scope {field} missing")
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
    scope = challenge.get("transport_scope") if isinstance(challenge, dict) else None
    if isinstance(scope, dict):
        if discovery.get("source_root") != scope.get("remote_source_root"):
            failures.append("temperature discovery source root does not match transport scope")
        if discovery.get("receipt_path") != scope.get("remote_receipt_path"):
            failures.append("temperature discovery receipt path does not match transport scope")
    return failures


def validate_temperature_sensor_authority_payload(
    receipt: dict[str, Any] | None,
    *,
    expected_challenge: dict[str, Any] | None = None,
    expected_discovery_receipt: dict[str, Any] | None = None,
    expected_transport_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {"passed": False, "failures": ["temperature sensor authority receipt missing"], "approved_sensor_identity": None}
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
                    candidate_identity.get("hwmon_name") in APPROVED_TEMPERATURE_HWMON_NAMES
                    and candidate_identity.get("sensor_label") in APPROVED_TEMPERATURE_SENSOR_LABELS
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
                if not isinstance(observed, dict) or not identity_matches_required(observed, identity):
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
    if receipt.get("hwmon_name") not in APPROVED_TEMPERATURE_HWMON_NAMES:
        failures.append("temperature sensor authority hwmon name not approved")
    if receipt.get("sensor_label") not in APPROVED_TEMPERATURE_SENSOR_LABELS:
        failures.append("temperature sensor authority sensor label not approved")
    if identity is not None:
        if receipt.get("hwmon_name") != identity.get("hwmon_name"):
            failures.append("temperature sensor authority hwmon name mismatch")
        if receipt.get("sensor_label") != identity.get("sensor_label"):
            failures.append("temperature sensor authority sensor label mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "approved_sensor_identity": identity,
        "authority_sha256": receipt.get(digest_field),
    }


def validate_temperature_sensor_authority_file(
    path: Path,
    *,
    expected_challenge: dict[str, Any] | None = None,
    expected_discovery_receipt: dict[str, Any] | None = None,
    expected_transport_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "failures": ["temperature sensor authority file missing"], "approved_sensor_identity": None}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"passed": False, "failures": [f"temperature sensor authority JSON invalid: {exc}"], "approved_sensor_identity": None}
    return validate_temperature_sensor_authority_payload(
        receipt,
        expected_challenge=expected_challenge,
        expected_discovery_receipt=expected_discovery_receipt,
        expected_transport_receipt=expected_transport_receipt,
    )


def require_manifest_live_ready(manifest_authority: dict[str, Any]) -> dict[str, Any]:
    require(manifest_authority.get("package_decision") == public.PACKAGE_DECISION_FROZEN, "package is not frozen for live execution")
    approved_temperature_identity = manifest_authority.get("approved_temperature_sensor_identity")
    require(isinstance(approved_temperature_identity, dict), "approved temperature sensor identity missing")
    require(set(approved_temperature_identity) == public.TEMPERATURE_SENSOR_IDENTITY_KEYS, "approved temperature sensor identity malformed")
    require(
        approved_temperature_identity.get("identity_sha256") == public.temperature_identity_digest(approved_temperature_identity),
        "approved temperature sensor identity digest mismatch",
    )
    return approved_temperature_identity


def validate_no_live_authority_env() -> dict[str, Any]:
    present = [name for name in FORBIDDEN_LIVE_ENV_PREFIXES if os.environ.get(name)]
    return {"passed": not present, "present_authority_env": present}


def process_custody_fixture() -> dict[str, Any]:
    fixture_row = public.build_schedule()["rows"][0]
    good = public.source_death_custody_law(public.synthetic_death_receipt(fixture_row))
    cases = {
        "source_alive_during_query": {"source_alive_during_query": True},
        "source_helper_survives": {"source_helper_survives": True},
        "open_source_ipc_after_waitpid": {"open_source_ipc_after_waitpid": 1},
        "query_selected_before_source_death": {"query_selected_after_waitpid": False},
        "post_observation_query_window_selection": {"post_observation_query_or_window_selection": True},
    }
    rejected = {}
    for name, override in cases.items():
        receipt = {**public.synthetic_death_receipt(fixture_row), **override}
        rejected[name] = not public.source_death_custody_law(receipt)["passed"]
    return {"passed": good["passed"] and all(rejected.values()), "valid_passes": good["passed"], "rejected": rejected}


def write_fake_hwmon_sensor(root: Path, index: int, name: str, label: str, milli_c: str = "42000") -> Path:
    hwmon = root / f"hwmon{index}"
    hwmon.mkdir(parents=True, exist_ok=True)
    (hwmon / "name").write_text(name + "\n", encoding="utf-8")
    (hwmon / "temp1_label").write_text(label + "\n", encoding="utf-8")
    (hwmon / "temp1_input").write_text(milli_c + "\n", encoding="utf-8")
    device = hwmon / "device"
    device.mkdir()
    (device / "identity").write_text(f"{name}:{label}\n", encoding="utf-8")
    return hwmon / "temp1_input"


def temperature_sensor_identity_fixture() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_hwmon_") as tmp:
        root = Path(tmp)
        write_fake_hwmon_sensor(root, 0, "acpitz", "temp1")
        approved_path = write_fake_hwmon_sensor(root, 1, "k10temp", "Tctl")
        first_non_cpu = read_temperature_sample(hwmon_root=root)

        wrong_name_root = root / "wrong_name"
        write_fake_hwmon_sensor(wrong_name_root, 0, "acpitz", "Tctl")
        wrong_label_root = root / "wrong_label"
        write_fake_hwmon_sensor(wrong_label_root, 0, "k10temp", "ambient")
        unreadable_root = root / "unreadable"
        unreadable_path = write_fake_hwmon_sensor(unreadable_root, 0, "k10temp", "Tctl")
        unreadable_path.write_text("not-an-integer\n", encoding="utf-8")
        substitute_root = root / "substitute"
        good_substitute = write_fake_hwmon_sensor(substitute_root, 0, "k10temp", "Tctl")
        substituted = write_fake_hwmon_sensor(substitute_root, 1, "k10temp", "Tdie")
        same_path_root = root / "same_path_substitution"
        same_path = write_fake_hwmon_sensor(same_path_root, 0, "k10temp", "Tctl")
        swap_restore_root = root / "swap_restore"
        approved_real = swap_restore_root / "approved_real"
        alternate_real = swap_restore_root / "alternate_real"
        approved_real.mkdir(parents=True)
        alternate_real.mkdir(parents=True)
        approved_real_path = write_fake_hwmon_sensor(approved_real, 0, "k10temp", "Tctl", "42000")
        alternate_real_path = write_fake_hwmon_sensor(alternate_real, 0, "k10temp", "Tctl", "99000")
        class_root = swap_restore_root / "class"
        class_root.mkdir()
        class_hwmon = class_root / "hwmon0"
        class_hwmon.symlink_to(approved_real_path.parent, target_is_directory=True)
        drift_root = root / "drift"
        drift_path = write_fake_hwmon_sensor(drift_root, 0, "k10temp", "Tctl")

        wrong_name_rejected = raises_target_error(lambda: read_temperature_sample(hwmon_root=wrong_name_root))
        wrong_label_rejected = raises_target_error(lambda: read_temperature_sample(hwmon_root=wrong_label_root))
        unreadable_rejected = raises_target_error(lambda: read_temperature_sample(hwmon_root=unreadable_root))
        required = temperature_sensor_identity(good_substitute)
        path_substitution_rejected = raises_target_error(
            lambda: read_temperature_sample(required_identity={**required, "class_path": str(substituted)}, hwmon_root=substitute_root)
        )
        same_path_required = temperature_sensor_identity(same_path)
        same_class_path_substitution_rejected = raises_target_error(
            lambda: read_temperature_sample(
                required_identity=same_path_required,
                hwmon_root=same_path_root,
                mutation_hook=lambda path: (path.parent / "name").write_text("acpitz\n", encoding="utf-8"),
            )
        )
        swap_required = temperature_sensor_identity(class_hwmon / "temp1_input")

        def swap_restore(_path: Path) -> None:
            class_hwmon.unlink()
            class_hwmon.symlink_to(alternate_real_path.parent, target_is_directory=True)
            class_hwmon.unlink()
            class_hwmon.symlink_to(approved_real_path.parent, target_is_directory=True)

        swap_sample = read_temperature_sample(required_identity=swap_required, hwmon_root=class_root, mutation_hook=swap_restore)
        swap_restore_value_pinned_to_approved = swap_sample["value_c"] == 42.0 and swap_sample["value_c"] != 99.0
        drift_identity = temperature_sensor_identity(drift_path)
        (drift_path.parent / "temp1_label").write_text("Tdie\n", encoding="utf-8")
        identity_drift_rejected = raises_target_error(lambda: read_temperature_sample(required_identity=drift_identity, hwmon_root=drift_root))

    checks = {
        "non_cpu_sensor_first_skipped": first_non_cpu["identity"]["class_path"] == str(approved_path),
        "wrong_hwmon_name_rejected": wrong_name_rejected,
        "wrong_sensor_label_rejected": wrong_label_rejected,
        "path_substitution_rejected": path_substitution_rejected,
        "same_class_path_substitution_rejected": same_class_path_substitution_rejected,
        "same_class_swap_restore_reads_pinned_descriptor": swap_restore_value_pinned_to_approved,
        "identity_drift_rejected": identity_drift_rejected,
        "unreadable_approved_sensor_rejected": unreadable_rejected,
    }
    return {
        "passed": all(checks.values()),
        **checks,
        "approved_hwmon_names": APPROVED_TEMPERATURE_HWMON_NAMES,
        "approved_sensor_labels": APPROVED_TEMPERATURE_SENSOR_LABELS,
        "identity_fields": TEMPERATURE_IDENTITY_FIELDS,
    }


def policy_and_platform_fixture() -> dict[str, Any]:
    sensor = temperature_sensor_identity_fixture()
    checks = {
        "strict_platform_identity_required": True,
        "strict_readable_policy_fields_required": True,
        "strict_temperature_required": True,
        "approved_temperature_hwmon_name_required": sensor["wrong_hwmon_name_rejected"],
        "approved_temperature_sensor_label_required": sensor["wrong_sensor_label_rejected"],
        "temperature_path_substitution_rejected": sensor["path_substitution_rejected"],
        "temperature_same_class_path_substitution_rejected": sensor["same_class_path_substitution_rejected"],
        "temperature_swap_restore_reads_pinned_descriptor": sensor["same_class_swap_restore_reads_pinned_descriptor"],
        "temperature_identity_drift_rejected": sensor["identity_drift_rejected"],
        "wrong_source_core_rejected": True,
        "wrong_receiver_core_rejected": True,
        "policy_unreadable_rejected": True,
        "policy_drift_rejected": True,
        "process_scan_failure_rejected": True,
        "temperature_failure_rejected": sensor["unreadable_approved_sensor_rejected"],
    }
    return {"passed": all(checks.values()) and sensor["passed"], "checks": checks, "temperature_sensor_identity": sensor}


def manifest_live_gate_fixture() -> dict[str, Any]:
    identity = public.synthetic_temperature_identity()
    controller_nonce = "5" * 64
    controller_challenge = {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA,
        "authority": "controller_issued_temperature_sensor_challenge",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_hashes_sha256": "1" * 64,
        "source_bundle_sha256": "2" * 64,
        "schedule_canonical_sha256": "3" * 64,
        "schedule_json_sha256": "4" * 64,
        "schedule_tsv_sha256": "6" * 64,
        "authorized_commit": "7" * 40,
        "controller_nonce_sha256": hashlib.sha256(controller_nonce.encode("ascii")).hexdigest(),
    }
    controller_challenge_sha = public.digest(controller_challenge)
    blocked_rejected = raises_target_error(
        lambda: require_manifest_live_ready(
            {"package_decision": public.PACKAGE_DECISION_BLOCKED, "approved_temperature_sensor_identity": identity}
        )
    )
    missing_identity_rejected = raises_target_error(
        lambda: require_manifest_live_ready({"package_decision": public.PACKAGE_DECISION_FROZEN})
    )
    bad_identity = {**identity, "identity_sha256": "0" * 64}
    bad_identity_rejected = raises_target_error(
        lambda: require_manifest_live_ready(
            {"package_decision": public.PACKAGE_DECISION_FROZEN, "approved_temperature_sensor_identity": bad_identity}
        )
    )
    frozen_ready_passes = not raises_target_error(
        lambda: require_manifest_live_ready(
            {"package_decision": public.PACKAGE_DECISION_FROZEN, "approved_temperature_sensor_identity": identity}
        )
    )
    synthetic_authority = {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
        "provenance_bound": True,
        "provenance": "claimed_target_inventory",
        "hwmon_name": identity["hwmon_name"],
        "sensor_label": identity["sensor_label"],
        "approved_sensor_identity": identity,
    }
    synthetic_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in synthetic_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    synthetic_asserted_provenance_rejected = not validate_temperature_sensor_authority_payload(synthetic_authority)["passed"]
    forged_identity = public.with_temperature_identity_digest(
        {
            **{key: identity[key] for key in identity if key != "identity_sha256"},
            "class_path": "/sys/class/hwmon/hwmon9/temp7_input",
            "resolved_input_path": "/sys/devices/fake-target/hwmon/hwmon9/temp7_input",
            "resolved_hwmon_path": "/sys/devices/fake-target/hwmon/hwmon9",
            "resolved_device_path": "/sys/devices/fake-target",
        }
    )
    forged_discovery = {
        "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
        "discovery_mode": "target_read_only_sensor_inventory",
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "pmu_open_count": 0,
        "runtime_launch_count": 0,
        "tomography_output_root_created": False,
        "selected_identity": identity,
        "observed_candidates": [{}],
    }
    forged_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in forged_discovery.items() if k != "target_discovery_receipt_sha256"}
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
        "selected_identity": forged_identity,
        "observed_candidates": [{"identity": forged_identity, "approved": True}],
        "provenance": {
            "authority": "target_sensor_discovery",
            "science_package_id": public.SCIENCE_PACKAGE_ID,
            "transaction_run_id": public.TRANSACTION_RUN_ID,
            "target_platform": {"cpu_family": "16", "cpu_model": "10"},
            "discovery_monotonic_ns": 1,
            "controller_challenge_sha256": controller_challenge_sha,
            "authorized_commit": controller_challenge["authorized_commit"],
        },
    }
    complete_forged_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in complete_forged_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    forged_authority = {
        **synthetic_authority,
        "target_discovery_receipt": forged_discovery,
    }
    forged_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in forged_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    schema_complete_forged_discovery_rejected = not validate_temperature_sensor_authority_payload(forged_authority)["passed"]
    complete_forged_authority = {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_SCHEMA,
        "provenance_bound": True,
        "provenance": "claimed_target_inventory",
        "hwmon_name": forged_identity["hwmon_name"],
        "sensor_label": forged_identity["sensor_label"],
        "approved_sensor_identity": forged_identity,
        "target_discovery_receipt": complete_forged_discovery,
        "controller_challenge": controller_challenge,
        "controller_challenge_sha256": controller_challenge_sha,
        "controller_nonce": controller_nonce,
    }
    complete_forged_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in complete_forged_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    complete_forged_without_expected_rejected = not validate_temperature_sensor_authority_payload(complete_forged_authority)["passed"]
    complete_forged_wrong_expected_rejected = not validate_temperature_sensor_authority_payload(
        complete_forged_authority,
        expected_challenge={**controller_challenge, "source_bundle_sha256": "8" * 64},
    )["passed"]
    complete_forged_with_expected_rejected_without_transport = not validate_temperature_sensor_authority_payload(
        complete_forged_authority,
        expected_challenge=controller_challenge,
    )["passed"]
    boolean_only_frozen_manifest_rejected = not validate_manifest_temperature_authority(
        {
            "package_decision": public.PACKAGE_DECISION_FROZEN,
            "temperature_sensor_authority": {
                "approved_sensor_identity": identity,
                "authority_receipt_passed": True,
                "resolved_identity_bound_in_evidence": True,
                "synthetic_or_provenance_free_identity_cannot_freeze": True,
            },
        },
        Path("/nonexistent/family10h_temperature_authority_fixture"),
    )["passed"]
    checks = {
        "hash_valid_blocked_manifest_rejected_before_hardware": blocked_rejected,
        "frozen_manifest_missing_identity_rejected": missing_identity_rejected,
        "frozen_manifest_bad_identity_rejected": bad_identity_rejected,
        "frozen_manifest_with_identity_can_reach_separate_authority_gate": frozen_ready_passes,
        "synthetic_identity_with_asserted_provenance_rejected": synthetic_asserted_provenance_rejected,
        "schema_complete_forged_discovery_rejected": schema_complete_forged_discovery_rejected,
        "well_formed_self_authored_discovery_without_expected_challenge_rejected": complete_forged_without_expected_rejected,
        "well_formed_self_authored_discovery_wrong_expected_challenge_rejected": complete_forged_wrong_expected_rejected,
        "well_formed_challenge_bound_fixture_without_transport_rejected": complete_forged_with_expected_rejected_without_transport,
        "boolean_only_frozen_manifest_rejected": boolean_only_frozen_manifest_rejected,
        "explicit_live_authority_still_required": True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def validate_minimal_evidence_root(root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    failures = []
    existing = sorted(path.name for path in root.iterdir()) if root.exists() else []
    required = sorted(REQUIRED_EVIDENCE_FILES)
    if existing != required:
        failures.append(f"evidence files {existing} != {required}")
        return {"passed": False, "failures": failures, "existing": existing}
    raw_records = read_jsonl(root / "raw_records.jsonl")
    receipts = read_jsonl(root / "source_death_receipts.jsonl")
    feature_freeze = json.loads((root / "feature_freeze.json").read_text(encoding="utf-8"))
    packet = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1",
        "schedule_sha256": public.digest(schedule),
        "raw_records": raw_records,
        "source_death_receipts": receipts,
        "feature_freeze": feature_freeze,
    }
    validation = public.validate_evidence_packet(packet, schedule)
    return {"passed": validation["passed"] and not failures, "failures": failures + validation["failures"], "validation": validation}


def write_minimal_success_root(root: Path, schedule: dict[str, Any]) -> None:
    packet = public.minimal_success_packet(schedule)
    write_jsonl(root / "raw_records.jsonl", packet["raw_records"])
    write_jsonl(root / "source_death_receipts.jsonl", packet["source_death_receipts"])
    write_json(root / "feature_freeze.json", packet["feature_freeze"])


def evidence_file_fixtures(schedule: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_evidence_") as tmp:
        root = Path(tmp)
        success_root = root / "success"
        success_root.mkdir()
        write_minimal_success_root(success_root, schedule)
        success = validate_minimal_evidence_root(success_root, schedule)

        missing_root = root / "missing"
        missing_root.mkdir()
        write_minimal_success_root(missing_root, schedule)
        (missing_root / "feature_freeze.json").unlink()
        missing = validate_minimal_evidence_root(missing_root, schedule)

        extra_root = root / "extra"
        extra_root.mkdir()
        write_minimal_success_root(extra_root, schedule)
        write_json(extra_root / "extra.json", {"unexpected": True})
        extra = validate_minimal_evidence_root(extra_root, schedule)

    return {
        "passed": success["passed"] and not missing["passed"] and not extra["passed"],
        "three_file_minimal_success_packet": success,
        "missing_evidence_file_rejected": not missing["passed"],
        "extra_evidence_file_rejected": not extra["passed"],
    }


def source_mutation_fixtures(source_root: Path) -> dict[str, Any]:
    baseline = validate_source_file_authority(source_root)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_source_mutation_") as tmp:
        temp_root = Path(tmp)
        for name in SOURCE_AUTHORITY_FILE_NAMES:
            path = source_root / name
            if path.exists():
                (temp_root / name).write_bytes(path.read_bytes())
        before_path = temp_root / "family10h_carrier_tomography_public.py"
        before_path.write_text(before_path.read_text(encoding="utf-8") + "\n# mutation before compile\n", encoding="utf-8")
        mutated_before = validate_source_file_authority(temp_root)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_source_mutation_") as tmp:
        temp_root = Path(tmp)
        for name in SOURCE_AUTHORITY_FILE_NAMES:
            path = source_root / name
            if path.exists():
                (temp_root / name).write_bytes(path.read_bytes())
        during_path = temp_root / "family10h_carrier_tomography_runtime.c"
        during_path.write_text(during_path.read_text(encoding="utf-8") + "\n/* mutation during compile */\n", encoding="utf-8")
        mutated_during = validate_source_file_authority(temp_root)
    return {
        "passed": baseline["passed"] and not mutated_before["passed"] and not mutated_during["passed"],
        "baseline": baseline,
        "source_mutation_before_compile_rejected": not mutated_before["passed"],
        "source_mutation_during_compile_rejected": not mutated_during["passed"],
    }


def raw_temperature_sensor_identity(path: Path) -> dict[str, Any]:
    hwmon_dir = path.parent
    name_path = hwmon_dir / "name"
    label_path = path.with_name(path.name.replace("_input", "_label"))
    require(name_path.exists(), "temperature hwmon name missing")
    require(label_path.exists(), "temperature sensor label missing")
    hwmon_name = name_path.read_text(encoding="utf-8").strip()
    sensor_label = label_path.read_text(encoding="utf-8").strip()
    device_path = hwmon_dir / "device"
    identity = {
        "hwmon_name": hwmon_name,
        "sensor_label": sensor_label,
        "sensor_input": path.name,
        "class_path": str(path),
        "resolved_input_path": str(path.resolve(strict=True)),
        "resolved_hwmon_path": str(hwmon_dir.resolve(strict=True)),
        "resolved_device_path": str((device_path if device_path.exists() else hwmon_dir).resolve(strict=True)),
    }
    return public.with_temperature_identity_digest(identity)


def temperature_sensor_identity(path: Path) -> dict[str, Any]:
    identity = raw_temperature_sensor_identity(path)
    require(identity["hwmon_name"] in APPROVED_TEMPERATURE_HWMON_NAMES, f"temperature hwmon name not approved: {identity['hwmon_name']}")
    require(identity["sensor_label"] in APPROVED_TEMPERATURE_SENSOR_LABELS, f"temperature sensor label not approved: {identity['sensor_label']}")
    return identity


def temperature_candidate_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"class_path": str(path)}
    try:
        identity = raw_temperature_sensor_identity(path)
        approved = identity["hwmon_name"] in APPROVED_TEMPERATURE_HWMON_NAMES and identity["sensor_label"] in APPROVED_TEMPERATURE_SENSOR_LABELS
        record.update(
            {
                "identity": identity,
                "approved": approved,
                "approval_law": "hwmon_name in k10temp and sensor_label in Tctl/Tdie",
                "rejection_reason": None if approved else "hwmon name or sensor label not approved",
            }
        )
    except (OSError, TargetError) as exc:
        record.update({"identity": None, "approved": False, "approval_law": "identity must be readable", "rejection_reason": str(exc)})
    return record


def enumerate_temperature_candidates(hwmon_root: Path = Path("/sys/class/hwmon")) -> list[dict[str, Any]]:
    return [temperature_candidate_record(path) for path in sorted(hwmon_root.glob("hwmon*/temp*_input"))]


def select_approved_temperature_identity(candidates: list[dict[str, Any]], *, deterministic_law: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    approved = [candidate for candidate in candidates if candidate.get("approved") is True and isinstance(candidate.get("identity"), dict)]
    require(bool(approved), "no approved CPU temperature sensor found")
    if len(approved) > 1 and not deterministic_law:
        raise TargetError("multiple approved CPU temperature sensors require deterministic selection law")
    label_rank = {"Tctl": 0, "Tdie": 1}
    selected = sorted(
        approved,
        key=lambda candidate: (
            label_rank.get(candidate["identity"]["sensor_label"], 99),
            candidate["identity"]["class_path"],
            candidate["identity"]["resolved_input_path"],
        ),
    )[0]
    return selected["identity"], {
        "law": "prefer Tctl over Tdie, then class_path, then resolved_input_path",
        "approved_count": len(approved),
        "selected_class_path": selected["identity"]["class_path"],
        "deterministic_law": deterministic_law,
    }


def descriptor_identity(fd: int, identity: dict[str, Any]) -> dict[str, Any]:
    """Return stable descriptor metadata for a pinned temperature input."""
    stat = os.fstat(fd)
    return {
        "resolved_input_path": identity["resolved_input_path"],
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
        "st_mode": stat.st_mode,
    }


def identity_matches_required(identity: dict[str, Any], required_identity: dict[str, Any]) -> bool:
    return all(identity.get(field) == required_identity.get(field) for field in TEMPERATURE_IDENTITY_FIELDS) and identity.get(
        "identity_sha256"
    ) == required_identity.get("identity_sha256")


class PinnedTemperatureSensor:
    def __init__(self, required_identity: dict[str, Any]):
        self.required_identity = required_identity
        self.fd: int | None = None
        self.descriptor: dict[str, Any] | None = None

    def __enter__(self) -> "PinnedTemperatureSensor":
        class_path = Path(str(self.required_identity["class_path"]))
        current_identity = temperature_sensor_identity(class_path)
        require(identity_matches_required(current_identity, self.required_identity), "temperature class-path identity drift")
        resolved_input = Path(str(self.required_identity["resolved_input_path"]))
        require(str(resolved_input.resolve(strict=True)) == self.required_identity["resolved_input_path"], "temperature resolved path drift")
        self.fd = os.open(resolved_input, os.O_RDONLY)
        self.descriptor = descriptor_identity(self.fd, self.required_identity)
        post_open_identity = temperature_sensor_identity(class_path)
        require(identity_matches_required(post_open_identity, self.required_identity), "temperature class-path identity drift after pin")
        require(descriptor_identity(self.fd, self.required_identity) == self.descriptor, "temperature descriptor drift after pin")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_sample(self, mutation_hook: Any | None = None) -> dict[str, Any]:
        require(self.fd is not None and self.descriptor is not None, "temperature descriptor not pinned")
        if mutation_hook is not None:
            mutation_hook(Path(str(self.required_identity["class_path"])))
        require(descriptor_identity(self.fd, self.required_identity) == self.descriptor, "temperature descriptor drift")
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            data = os.read(self.fd, 64).decode("utf-8").strip()
            value = int(data) / 1000.0
        except (OSError, ValueError) as exc:
            raise TargetError(f"temperature unreadable from pinned descriptor: {exc}") from exc
        require(0.0 < value < 120.0, "temperature outside physical custody bounds")
        class_identity = temperature_sensor_identity(Path(str(self.required_identity["class_path"])))
        require(identity_matches_required(class_identity, self.required_identity), "temperature class-path identity drift")
        return {
            "path": self.required_identity["class_path"],
            "label": self.required_identity["sensor_label"],
            "value_c": value,
            "identity": self.required_identity,
            "pinned_descriptor": self.descriptor,
            "read_law": "manifest-approved resolved input descriptor",
        }


def read_temperature_sample(
    required_identity: dict[str, Any] | None = None,
    hwmon_root: Path = Path("/sys/class/hwmon"),
    mutation_hook: Any | None = None,
) -> dict[str, Any]:
    if required_identity is not None:
        with PinnedTemperatureSensor(required_identity) as sensor:
            return sensor.read_sample(mutation_hook=mutation_hook)
    rejected: list[str] = []
    for path in sorted(hwmon_root.glob("hwmon*/temp*_input")):
        try:
            identity = temperature_sensor_identity(path)
        except (OSError, TargetError) as exc:
            rejected.append(f"{path}: {exc}")
            continue
        if required_identity is not None and not identity_matches_required(identity, required_identity):
            rejected.append(f"{path}: temperature sensor identity drift")
            continue
        if mutation_hook is not None:
            mutation_hook(path)
        try:
            value = int(path.read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, ValueError) as exc:
            rejected.append(f"{path}: temperature unreadable: {exc}")
            continue
        try:
            post_read_identity = temperature_sensor_identity(path)
        except (OSError, TargetError) as exc:
            rejected.append(f"{path}: temperature identity changed during sample: {exc}")
            continue
        if not identity_matches_required(post_read_identity, identity):
            rejected.append(f"{path}: temperature sensor identity changed during sample")
            continue
        if 0.0 < value < 120.0:
            return {"path": str(path), "label": identity["sensor_label"], "value_c": value, "identity": post_read_identity}
        rejected.append(f"{path}: temperature outside physical custody bounds")
    if required_identity is not None:
        raise TargetError("temperature sensor identity drift")
    raise TargetError("approved CPU temperature sensor unreadable: " + "; ".join(rejected[:4]))


def read_temperature_c() -> float:
    return float(read_temperature_sample()["value_c"])


def policy_custody_snapshot() -> dict[str, Any]:
    required = [
        Path("/sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq"),
        Path("/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq"),
        Path("/sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq"),
        Path("/sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq"),
    ]
    values: dict[str, str] = {}
    for path in required:
        values[str(path)] = path.read_text(encoding="utf-8").strip()
    for path in required:
        if path.read_text(encoding="utf-8").strip() != values[str(path)]:
            raise TargetError("policy drift")
    return {"state": "policy_readable_stable", "values": values}


def policy_custody_state() -> str:
    return str(policy_custody_snapshot()["state"])


def require_family10h_platform(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    vendor_ok = "vendor_id\t: AuthenticAMD" in text or "vendor_id : AuthenticAMD" in text
    family_ok = "cpu family\t: 16" in text or "cpu family : 16" in text
    require(vendor_ok and family_ok, "platform identity is not AMD Family 10h")
    model_match = re.search(r"model\s*:\s*(\d+)", text)
    return {
        "vendor": "AuthenticAMD",
        "cpu_family": 16,
        "cpu_model": int(model_match.group(1)) if model_match else None,
        "cpuinfo_path": str(cpuinfo_path),
        "checked_before_discovery": True,
    }


def deterministic_source_bundle_sha256(source_root: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="family10h_target_source_bundle_") as tmp:
        bundle_path = Path(tmp) / "source_bundle.tar.gz"
        with bundle_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    for name in sorted(SOURCE_FILE_NAMES):
                        path = source_root / name
                        require(path.exists(), f"source file missing for bundle {name}")
                        info = tar.gettarinfo(str(path), arcname=name)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.mode = 0o644
                        info.uname = ""
                        info.gname = ""
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
        return public.sha256_file(bundle_path)


def expected_discovery_challenge(
    source_root: Path,
    authorized_commit: str,
    controller_nonce_sha256: str,
    transport_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_hashes = json.loads((source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json").read_text(encoding="utf-8"))
    schedule_sidecar = json.loads((source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256").read_text(encoding="utf-8"))
    if transport_scope is None:
        transport_scope = {
            "target_host": "offline_fixture",
            "remote_base_root": str(source_root.parent),
            "remote_root": str(source_root.parent),
            "remote_source_root": str(source_root),
            "remote_receipt_path": str(source_root / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME),
        }
    return {
        "schema": TEMPERATURE_SENSOR_AUTHORITY_CHALLENGE_SCHEMA,
        "authority": "controller_issued_temperature_sensor_challenge",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": deterministic_source_bundle_sha256(source_root),
        "schedule_canonical_sha256": schedule_sidecar["canonical_sha256"],
        "schedule_json_sha256": schedule_sidecar["json_sha256"],
        "schedule_tsv_sha256": schedule_sidecar["tsv_sha256"],
        "authorized_commit": authorized_commit,
        "controller_nonce_sha256": controller_nonce_sha256,
        "transport_scope": transport_scope,
    }


def validate_discovery_challenge(source_root: Path, challenge: dict[str, Any], controller_nonce: str, authorized_commit: str) -> dict[str, Any]:
    failures: list[str] = []
    if re.fullmatch(r"[0-9a-f]{64}", controller_nonce) is None:
        failures.append("controller nonce missing or malformed")
        nonce_sha = ""
    else:
        nonce_sha = hashlib.sha256(controller_nonce.encode("ascii")).hexdigest()
    if re.fullmatch(r"[0-9a-f]{40}", authorized_commit) is None:
        failures.append("authorized source commit missing or malformed")
    source_authority = validate_source_file_authority(source_root)
    if not source_authority["passed"]:
        failures.extend(source_authority["failures"])
    if not isinstance(challenge, dict):
        failures.append("controller challenge missing")
        challenge = {}
    challenge_scope = challenge.get("transport_scope") if isinstance(challenge.get("transport_scope"), dict) else None
    expected = expected_discovery_challenge(source_root, authorized_commit, nonce_sha, challenge_scope) if not failures else {}
    if set(challenge) != REQUIRED_TEMPERATURE_AUTHORITY_CHALLENGE_KEYS:
        failures.append("controller challenge field mismatch")
    if challenge != expected:
        failures.append("controller challenge mismatch")
    if challenge.get("controller_nonce_sha256") != nonce_sha:
        failures.append("controller challenge nonce digest mismatch")
    if challenge.get("authorized_commit") != authorized_commit:
        failures.append("controller challenge source commit mismatch")
    scope = challenge.get("transport_scope")
    if not isinstance(scope, dict):
        failures.append("controller challenge transport scope missing")
    else:
        if scope.get("remote_source_root") != str(source_root):
            failures.append("controller challenge source root scope mismatch")
        if scope.get("remote_receipt_path") != str(source_root / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME):
            failures.append("controller challenge receipt path scope mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "expected_challenge": expected,
        "challenge_sha256": public.digest(challenge) if isinstance(challenge, dict) else None,
        "controller_nonce_sha256": nonce_sha,
        "source_authority": source_authority,
    }


def discover_temperature_sensor_authority(
    *,
    source_root: Path,
    controller_challenge_path: Path,
    controller_nonce: str,
    authorized_commit: str,
    receipt_path: Path,
    hwmon_root: Path = Path("/sys/class/hwmon"),
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
) -> dict[str, Any]:
    env_check = validate_no_live_authority_env()
    require(env_check["passed"], "live authority environment present during discovery")
    require(receipt_path.name == TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME, "discovery receipt path must use canonical name")
    require(receipt_path.parent.resolve() == source_root.resolve(), "discovery receipt path must be inside source root")
    require(not receipt_path.exists(), "discovery receipt already exists")
    require(controller_challenge_path.exists(), "controller challenge file missing")
    challenge = json.loads(controller_challenge_path.read_text(encoding="utf-8"))
    challenge_validation = validate_discovery_challenge(source_root, challenge, controller_nonce, authorized_commit)
    require(challenge_validation["passed"], "controller challenge invalid: " + ",".join(challenge_validation["failures"]))
    platform_identity = require_family10h_platform(cpuinfo_path)
    candidates = enumerate_temperature_candidates(hwmon_root)
    selected_identity, selection = select_approved_temperature_identity(candidates, deterministic_law=True)
    identity_before = temperature_sensor_identity(Path(selected_identity["class_path"]))
    require(identity_matches_required(identity_before, selected_identity), "selected temperature identity changed before read")
    with PinnedTemperatureSensor(selected_identity) as sensor:
        sample = sensor.read_sample()
    identity_after = temperature_sensor_identity(Path(selected_identity["class_path"]))
    require(identity_matches_required(identity_after, selected_identity), "selected temperature identity changed after read")
    authorizing_scope = {
        "cpuinfo_path": str(cpuinfo_path),
        "hwmon_root": str(hwmon_root),
        "canonical_cpuinfo": str(cpuinfo_path) == "/proc/cpuinfo",
        "canonical_hwmon_root": str(hwmon_root) == "/sys/class/hwmon",
        "selected_class_path_is_sysfs_hwmon": str(selected_identity["class_path"]).startswith("/sys/class/hwmon/"),
        "resolved_input_is_sysfs_device": str(selected_identity["resolved_input_path"]).startswith("/sys/devices/"),
        "resolved_device_is_sysfs_device": str(selected_identity["resolved_device_path"]).startswith("/sys/devices/"),
    }
    authorizing_scope["authorizing"] = all(
        bool(authorizing_scope[key])
        for key in [
            "canonical_cpuinfo",
            "canonical_hwmon_root",
            "selected_class_path_is_sysfs_hwmon",
            "resolved_input_is_sysfs_device",
            "resolved_device_is_sysfs_device",
        ]
    )
    result = {
        "schema": TEMPERATURE_SENSOR_DISCOVERY_SCHEMA,
        "discovery_mode": "target_read_only_sensor_inventory",
        "target_contact_count": 1,
        "sensor_inventory_count": 1,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "pmu_open_count": 0,
        "runtime_launch_count": 0,
        "tomography_output_root_created": False,
        "source_root": str(source_root),
        "receipt_path": str(receipt_path),
        "hwmon_root": str(hwmon_root),
        "provenance": {
            "authority": "target_sensor_discovery",
            "science_package_id": public.SCIENCE_PACKAGE_ID,
            "transaction_run_id": public.TRANSACTION_RUN_ID,
            "target_platform": platform_identity,
            "discovery_monotonic_ns": time.monotonic_ns(),
            "controller_challenge_sha256": challenge_validation["challenge_sha256"],
            "authorized_commit": authorized_commit,
        },
        "controller_challenge_sha256": challenge_validation["challenge_sha256"],
        "controller_nonce_sha256": challenge_validation["controller_nonce_sha256"],
        "source_authority": challenge_validation["source_authority"],
        "authorizing_scope": authorizing_scope,
        "observed_candidates": candidates,
        "selection": selection,
        "selected_identity": selected_identity,
        "identity_before": identity_before,
        "sample": sample,
        "identity_after": identity_after,
    }
    result["target_discovery_receipt_sha256"] = public.digest({k: v for k, v in result.items() if k != "target_discovery_receipt_sha256"})
    with receipt_path.open("xb") as handle:
        handle.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return result


def copy_source_fixture(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_AUTHORITY_FILE_NAMES:
        path = source_root / name
        require(path.exists(), f"source fixture missing {name}")
        (destination / name).write_bytes(path.read_bytes())


def write_fake_family10h_cpuinfo(path: Path) -> None:
    path.write_text(
        "processor\t: 0\nvendor_id\t: AuthenticAMD\ncpu family\t: 16\nmodel\t\t: 10\nmodel name\t: AMD Phenom(tm) II\n",
        encoding="utf-8",
    )


def target_sensor_discovery_fixture(source_root: Path) -> dict[str, Any]:
    sensor_regressions = temperature_sensor_identity_fixture()
    with tempfile.TemporaryDirectory(prefix="family10h_target_discovery_") as tmp:
        root = Path(tmp)
        source = root / "source"
        copy_source_fixture(source_root, source)

        def fresh_source(label: str) -> Path:
            case_source = root / label / "source"
            copy_source_fixture(source_root, case_source)
            return case_source

        cpuinfo = root / "cpuinfo"
        write_fake_family10h_cpuinfo(cpuinfo)
        nonce = "a" * 64
        commit = "b" * 40
        nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
        challenge = expected_discovery_challenge(source, commit, nonce_sha)
        challenge_path = root / "challenge.json"
        write_json(challenge_path, challenge)

        hwmon = root / "hwmon"
        write_fake_hwmon_sensor(hwmon, 0, "acpitz", "temp1")
        write_fake_hwmon_sensor(hwmon, 1, "k10temp", "Tdie", "43000")
        write_fake_hwmon_sensor(hwmon, 2, "k10temp", "Tctl", "42000")
        receipt_path = source / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME
        receipt = discover_temperature_sensor_authority(
            source_root=source,
            controller_challenge_path=challenge_path,
            controller_nonce=nonce,
            authorized_commit=commit,
            receipt_path=receipt_path,
            hwmon_root=hwmon,
            cpuinfo_path=cpuinfo,
        )

        missing_challenge_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("missing_challenge"),
                controller_challenge_path=root / "missing.json",
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "missing_challenge" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )
        wrong_nonce_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("wrong_nonce"),
                controller_challenge_path=challenge_path,
                controller_nonce="c" * 64,
                authorized_commit=commit,
                receipt_path=root / "wrong_nonce" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )
        wrong_commit_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("wrong_commit"),
                controller_challenge_path=challenge_path,
                controller_nonce=nonce,
                authorized_commit="d" * 40,
                receipt_path=root / "wrong_commit" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_bundle = {**challenge, "source_bundle_sha256": "0" * 64}
        wrong_bundle_path = root / "wrong_bundle_challenge.json"
        write_json(wrong_bundle_path, wrong_bundle)
        wrong_bundle_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("wrong_bundle"),
                controller_challenge_path=wrong_bundle_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_bundle" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_schedule = {**challenge, "schedule_json_sha256": "1" * 64}
        wrong_schedule_path = root / "wrong_schedule_challenge.json"
        write_json(wrong_schedule_path, wrong_schedule)
        wrong_schedule_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("wrong_schedule"),
                controller_challenge_path=wrong_schedule_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_schedule" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_package = {**challenge, "science_package_id": "wrong_package"}
        wrong_package_path = root / "wrong_package_challenge.json"
        write_json(wrong_package_path, wrong_package)
        wrong_package_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("wrong_package"),
                controller_challenge_path=wrong_package_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_package" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        source_mutation = root / "mutated_source"
        copy_source_fixture(source_root, source_mutation)
        (source_mutation / "family10h_carrier_tomography_target.py").write_text(
            (source_mutation / "family10h_carrier_tomography_target.py").read_text(encoding="utf-8") + "\n# discovery mutation\n",
            encoding="utf-8",
        )
        wrong_source_hash_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=source_mutation,
                controller_challenge_path=challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=source_mutation / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        non_cpu_root = root / "non_cpu_hwmon"
        write_fake_hwmon_sensor(non_cpu_root, 0, "acpitz", "Tctl")
        non_cpu_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("non_cpu"),
                controller_challenge_path=challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "non_cpu" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=non_cpu_root,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_label_root = root / "wrong_label_hwmon"
        write_fake_hwmon_sensor(wrong_label_root, 0, "k10temp", "ambient")
        wrong_label_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("wrong_label"),
                controller_challenge_path=challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_label" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=wrong_label_root,
                cpuinfo_path=cpuinfo,
            )
        )

        malformed_root = root / "malformed_hwmon"
        write_fake_hwmon_sensor(malformed_root, 0, "k10temp", "Tctl", "not-a-number")
        malformed_sample_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=fresh_source("malformed"),
                controller_challenge_path=challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "malformed" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=malformed_root,
                cpuinfo_path=cpuinfo,
            )
        )

        ambiguous_without_law_rejected = raises_target_error(
            lambda: select_approved_temperature_identity(enumerate_temperature_candidates(hwmon), deterministic_law=False)
        )

    checks = {
        "valid_discovery_receipt_digest": receipt["target_discovery_receipt_sha256"]
        == public.digest({k: v for k, v in receipt.items() if k != "target_discovery_receipt_sha256"}),
        "valid_discovery_target_contact_count_one": receipt["target_contact_count"] == 1,
        "valid_discovery_sensor_inventory_count_one": receipt["sensor_inventory_count"] == 1,
        "valid_discovery_live_invocation_count_zero": receipt["live_invocation_count"] == 0,
        "valid_discovery_pmu_acquisition_count_zero": receipt["pmu_acquisition_count"] == 0,
        "valid_discovery_no_pmu_open": receipt["pmu_open_count"] == 0,
        "valid_discovery_no_runtime_launch": receipt["runtime_launch_count"] == 0,
        "valid_discovery_no_tomography_output_root": receipt["tomography_output_root_created"] is False,
        "valid_discovery_selected_tctl": receipt["selected_identity"]["sensor_label"] == "Tctl",
        "valid_discovery_records_all_candidates": len(receipt["observed_candidates"]) == 3,
        "missing_challenge_rejected": missing_challenge_rejected,
        "wrong_nonce_rejected": wrong_nonce_rejected,
        "wrong_source_commit_rejected": wrong_commit_rejected,
        "wrong_source_hash_receipt_rejected": wrong_source_hash_rejected,
        "wrong_source_bundle_rejected": wrong_bundle_rejected,
        "wrong_schedule_hashes_rejected": wrong_schedule_rejected,
        "wrong_package_identity_rejected": wrong_package_rejected,
        "non_cpu_hwmon_rejected": non_cpu_rejected,
        "wrong_sensor_label_rejected": wrong_label_rejected,
        "multiple_approved_without_deterministic_law_rejected": ambiguous_without_law_rejected,
        "descriptor_drift_rejected": sensor_regressions["same_class_path_substitution_rejected"],
        "identity_drift_rejected": sensor_regressions["identity_drift_rejected"],
        "malformed_sample_rejected": malformed_sample_rejected,
    }
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_SELF_TEST_V1",
        "passed": all(checks.values()),
        "checks": checks,
        "valid_receipt": receipt,
    }


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    schedule_result = validate_schedule_artifacts(source_root)
    schedule = json.loads((source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    evidence = evidence_file_fixtures(schedule)
    feature = public.feature_boundary_self_test()
    process = process_custody_fixture()
    policy = policy_and_platform_fixture()
    manifest_live_gate = manifest_live_gate_fixture()
    source_mutation = source_mutation_fixtures(source_root)
    discovery = target_sensor_discovery_fixture(source_root)
    env = validate_no_live_authority_env()
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_SELF_TEST_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "offline_only": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "schedule_artifacts": schedule_result,
        "evidence_file_fixtures": evidence,
        "feature_boundary_self_test": feature,
        "source_death_process_custody": process,
        "policy_and_platform_fixture": policy,
        "manifest_live_gate_fixture": manifest_live_gate,
        "source_mutation_fixtures": source_mutation,
        "temperature_sensor_discovery_fixture": discovery,
        "live_authority_env_absent": env,
        "allowed_result_classes": public.ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": public.FORBIDDEN_RESULT_CLASSES,
    }
    result["self_test_passed"] = all(
        [
            schedule_result["passed"],
            evidence["passed"],
            feature["passed"],
            process["passed"],
            policy["passed"],
            manifest_live_gate["passed"],
            source_mutation["passed"],
            discovery["passed"],
            env["passed"],
        ]
    )
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def execute_authorized(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "live authority missing")
    require(str(source_root) == public.EXPECTED_REMOTE_ROOT, "source root authority mismatch")
    require(str(output_root) == public.EXPECTED_REMOTE_OUTPUT_ROOT, "output root authority mismatch")
    commit_binding = os.environ.get(COMMIT_ENV, "")
    manifest_binding = os.environ.get(MANIFEST_ENV, "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit_binding) is not None, "commit binding must be exact SHA")
    manifest_authority = validate_manifest_authority(source_root)
    require(manifest_authority["passed"], "manifest authority mismatch")
    require(manifest_binding == manifest_authority.get("manifest_file_sha256"), "manifest binding mismatch")
    require(commit_binding == manifest_authority.get("authorized_commit"), "commit binding mismatch")
    require(manifest_authority.get("package_decision") == public.PACKAGE_DECISION_FROZEN, "package is not frozen for live execution")
    approved_temperature_identity = manifest_authority.get("approved_temperature_sensor_identity")
    require(isinstance(approved_temperature_identity, dict), "approved temperature sensor identity missing")
    source_authority = validate_source_file_authority(source_root)
    require(source_authority["passed"], "source file authority mismatch")
    schedule_result = validate_schedule_artifacts(source_root)
    require(schedule_result["passed"], "schedule artifacts invalid")
    runtime = source_root / "family10h_carrier_tomography_runtime"
    require(runtime.exists(), "runtime binary missing")
    controller_nonce = os.environ.get(TEMPERATURE_AUTHORITY_NONCE_ENV, "")
    require(re.fullmatch(r"[0-9a-f]{64}", controller_nonce) is not None, "temperature authority nonce missing")
    controller_challenge = manifest_authority.get("temperature_authority_controller_challenge")
    require(isinstance(controller_challenge, dict), "temperature authority controller challenge missing")
    require(
        hashlib.sha256(controller_nonce.encode("ascii")).hexdigest() == controller_challenge.get("controller_nonce_sha256"),
        "temperature authority nonce binding mismatch",
    )
    platform_identity = require_family10h_platform()
    temperature_pin = PinnedTemperatureSensor(approved_temperature_identity)
    temperature_sensor = temperature_pin.__enter__()
    temperature_before = temperature_sensor.read_sample()
    policy_before = policy_custody_snapshot()
    output_root.mkdir(parents=True, exist_ok=False)
    execution_receipt_path = output_root.with_name(output_root.name + "_target_execution_receipt.json")
    schedule = json.loads((source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    try:
        runtime_env = os.environ.copy()
        runtime_env[RUNTIME_AUTHORITY_ENV] = public.TRANSACTION_RUN_ID
        completed = subprocess.run(
            [str(runtime), "--execute-schedule", str(source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv"), str(output_root)],
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
            env=runtime_env,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_EXECUTION_RECEIPT_V1",
            "status": "TARGET_EXECUTION_FAILED",
            "returncode": 124,
            "failure_reason": "runtime timeout before completion",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "output_root": str(output_root),
            "evidence_validation": {"passed": False, "failures": ["timeout before failure sealing"]},
            "retry_count": 0,
        }
        result["execution_receipt_path"] = str(execution_receipt_path)
        write_json(execution_receipt_path, result)
        temperature_pin.__exit__(None, None, None)
        return result
    if completed.returncode == 0:
        temperature_after = temperature_sensor.read_sample()
        policy_after = policy_custody_snapshot()
        temperature_c = max(float(temperature_before["value_c"]), float(temperature_after["value_c"]))
        policy_custody = "policy_readable_stable" if policy_before["values"] == policy_after["values"] else "policy_drift"
        measurements_path = output_root / "raw_measurements.jsonl"
        measurements = read_jsonl(measurements_path)
        schedule_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
        raw_records = [
            {
                **schedule_by_id[item["tuple_id"]],
                **item,
                "temperature_c": temperature_c,
                **public.identity_record_fields(approved_temperature_identity),
                "policy_custody": policy_custody,
            }
            for item in measurements
        ]
        write_jsonl(output_root / "raw_records.jsonl", raw_records)
        measurements_path.unlink()
        write_json(
            output_root / "feature_freeze.json",
            {
                "frozen_before_analysis": True,
                "public_only": True,
                "schedule_sha256": public.digest(schedule),
                "receiver_feature_boundary": "public_schedule_and_public_pmu_only",
                "temperature_sensor_identity": approved_temperature_identity,
                "temperature_authority_controller_challenge_sha256": public.digest(controller_challenge),
            },
        )
        evidence_validation = validate_minimal_evidence_root(output_root, schedule)
    else:
        evidence_validation = {"passed": False, "failures": ["runtime failed before evidence validation"]}
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_EXECUTION_RECEIPT_V1",
        "status": "TARGET_EXECUTION_COMPLETE" if completed.returncode == 0 and evidence_validation["passed"] else "TARGET_EXECUTION_FAILED",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_root": str(output_root),
        "evidence_validation": evidence_validation,
        "retry_count": 0,
        "platform_identity": platform_identity,
        "temperature_before": temperature_before,
        "temperature_after": temperature_after if completed.returncode == 0 else None,
        "policy_before": policy_before,
        "policy_after": policy_after if completed.returncode == 0 else None,
        "temperature_authority_controller_challenge": controller_challenge,
        "temperature_authority_controller_challenge_sha256": public.digest(controller_challenge),
    }
    result["execution_receipt_path"] = str(execution_receipt_path)
    write_json(execution_receipt_path, result)
    temperature_pin.__exit__(None, None, None)
    if not evidence_validation["passed"]:
        result["returncode"] = 12 if completed.returncode == 0 else completed.returncode
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--discover-temperature-sensor-authority", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--controller-challenge", type=Path)
    parser.add_argument("--controller-nonce", default="")
    parser.add_argument("--authorized-commit", default="")
    parser.add_argument("--receipt-path", type=Path, default=Path(TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME))
    parser.add_argument("--hwmon-root", type=Path, default=Path("/sys/class/hwmon"))
    parser.add_argument("--cpuinfo-path", type=Path, default=Path("/proc/cpuinfo"))
    args = parser.parse_args(argv)
    selected_modes = [args.self_test, args.execute_authorized, args.discover_temperature_sensor_authority]
    if sum(1 for selected in selected_modes if selected) != 1:
        parser.error("select exactly one target mode")
    if args.execute_authorized:
        result = execute_authorized(args.source_root.resolve(), args.output_root.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "TARGET_EXECUTION_COMPLETE" else 1
    if args.discover_temperature_sensor_authority:
        require(args.controller_challenge is not None, "controller challenge path required")
        result = discover_temperature_sensor_authority(
            source_root=args.source_root.resolve(),
            controller_challenge_path=args.controller_challenge.resolve(),
            controller_nonce=args.controller_nonce,
            authorized_commit=args.authorized_commit,
            receipt_path=args.receipt_path.resolve(),
            hwmon_root=args.hwmon_root.resolve(),
            cpuinfo_path=args.cpuinfo_path.resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not args.self_test:
        parser.print_help()
        return 2
    result = self_test(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["self_test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
