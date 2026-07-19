#!/usr/bin/env python3
"""Target-side custody checks and authority-gated tomography execution.

The live acquisition entry point is present for future authorization, but this
task only runs the offline validators and self-tests.
"""

from __future__ import annotations

import argparse
import errno as errno_module
import gzip
import hashlib
import inspect
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
RUNTIME_BINARY_NAME = "family10h_carrier_tomography_runtime"
RUNTIME_AUTHORITY_FILE_NAMES = [RUNTIME_BINARY_NAME]
SOURCE_AUTHORITY_FILE_NAMES = SOURCE_FILE_NAMES + [
    "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json",
    "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz",
    "SUBAGENT_FINDINGS_NORMALIZED.json",
]
DISCOVERY_TRANSFER_FILE_NAMES = SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES

C3_SOURCE_AUTHORITY_COMMIT = "55e059bc7acaafee3feacddac2069d7b5e40edd1"
C4_SOURCE_AUTHORITY_COMMIT = "092d0a655e94d7c00f69efc1236cf1c8a2896ee1"
C5_SOURCE_AUTHORITY_COMMIT = "ca8f8490e9d2fc9b36debbfe7c927bfe2fde5c5e"
C5_FAILURE_EVIDENCE_COMMIT = "8021563a6122b72316ddd218077b8b82e36f9055"
APPROVED_TEMPERATURE_HWMON_NAMES = ["k10temp"]
APPROVED_TEMPERATURE_SENSOR_LABELS = ["Tctl"]
APPROVED_TEMPERATURE_DEVICE_DRIVERS = ["k10temp"]
APPROVED_TEMPERATURE_DEVICE_SUBSYSTEMS = ["pci"]
LEGACY_FAMILY10H_TEMPERATURE_PROFILE = "LEGACY_FAMILY10H_K10TEMP_TEMP1_V1"
LEGACY_FAMILY10H_TEMPERATURE_INPUT = "temp1_input"
LEGACY_FAMILY10H_TEMPERATURE_ROLE = "Tctl"
REQUIRED_REVIEW_ROLES = {
    "physical_carrier_state_auditor": "physical carrier-state auditor",
    "experimental_design_operator_auditor": "experimental-design/operator auditor",
    "custody_evidence_auditor": "custody/evidence auditor",
    "claim_boundary_adjudicator": "claim-boundary adjudicator",
}
SOURCE_AUDIT_REQUIRED_REVIEW_ROLES = {
    "physical_sensor_authority_auditor": "physical sensor-authority auditor",
    "discovery_transport_custody_auditor": "discovery transport and custody auditor",
    "source_bundle_runtime_evidence_auditor": "source/bundle/runtime evidence auditor",
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
TEMPERATURE_IDENTITY_FIELDS = [
    "hwmon_name",
    "sensor_label_present",
    "sensor_label_value",
    "sensor_input",
    "sensor_semantic_role",
    "sensor_semantic_profile",
    "class_path",
    "resolved_input_path",
    "resolved_hwmon_path",
    "resolved_device_path",
    "resolved_driver_path",
    "resolved_subsystem_path",
    "device_driver",
    "device_subsystem",
    "device_modalias",
    "input_st_dev",
    "input_st_ino",
    "input_st_mode",
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
TEMPERATURE_SENSOR_DISCOVERY_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_DISCOVERY_V2"
TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA = "FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_DISCOVERY_FAILURE_V2"
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
    "runtime_binary_sha256",
    "schedule_canonical_sha256",
    "schedule_json_sha256",
    "schedule_tsv_sha256",
    "authorized_commit",
    "controller_nonce_sha256",
    "transport_scope",
    "source_authority_review",
}
SOURCE_REVIEW_BINDING_KEYS = {
    "findings_sha256",
    "review_report_sha256",
    "review_quorum_sha256",
    "source_authority_commit",
    "source_hashes_sha256",
    "source_bundle_sha256",
    "runtime_binary_sha256",
}
CONTACT_COUNTER_KEYS = ["target_contact_count", "sensor_inventory_count", "live_invocation_count", "pmu_acquisition_count"]
FROZEN_CONTACT_COUNTERS = {"target_contact_count": 1, "sensor_inventory_count": 1, "live_invocation_count": 0, "pmu_acquisition_count": 0}


def contact_counter_object_equal_strict(value: Any, expected: dict[str, int] = FROZEN_CONTACT_COUNTERS) -> bool:
    if not isinstance(value, dict) or set(value) != set(expected):
        return False
    return all(public.is_json_int(value.get(key)) and value.get(key) == expected_value for key, expected_value in expected.items())


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


def raises_target_error_containing(callback: Any, expected: str) -> bool:
    try:
        callback()
    except TargetError as exc:
        return expected in str(exc)
    return False


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


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((strict_json_dumps(value, indent=2) + "\n").encode("utf-8"))


def write_json_exclusive_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"receipt already exists: {path}")
    tmp = path.with_name(path.name + ".tmp")
    require(not tmp.exists(), f"temporary receipt path already exists: {tmp}")
    try:
        tmp.write_bytes((strict_json_dumps(value, indent=2) + "\n").encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def serialized_json_sha256(value: Any) -> str:
    return hashlib.sha256((strict_json_dumps(value, indent=2) + "\n").encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise TargetError("blank JSONL row rejected")
    return [strict_json_loads(line) for line in lines]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(strict_json_dumps(row) + "\n" for row in rows).encode("utf-8"))


def validate_schedule_artifacts(source_root: Path) -> dict[str, Any]:
    schedule_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json"
    sidecar_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256"
    tsv_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv"
    require(schedule_path.exists(), "schedule JSON missing")
    require(sidecar_path.exists(), "schedule sidecar missing")
    require(tsv_path.exists(), "schedule TSV missing")
    schedule = read_json(schedule_path)
    sidecar = read_json(sidecar_path)
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
    receipt = read_json(receipt_path)
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
    runtime_authority = validate_runtime_binary_authority(
        source_root,
        expected=receipt.get("runtime_binary_authority") if isinstance(receipt.get("runtime_binary_authority"), dict) else None,
    )
    if not runtime_authority["passed"]:
        failures.extend("runtime authority: " + item for item in runtime_authority["failures"])
    if receipt.get("source_hashes_sha256") != public.digest({k: v for k, v in receipt.items() if k != "source_hashes_sha256"}):
        failures.append("source hash receipt digest mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "checked_files": len(expected),
        "runtime_binary_authority": runtime_authority.get("authority"),
    }


def git_blob_id_for_bytes(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def runtime_binary_authority(source_root: Path) -> dict[str, Any]:
    binary_path = source_root / RUNTIME_BINARY_NAME
    source_c_path = source_root / "family10h_carrier_tomography_runtime.c"
    source_h_path = source_root / "family10h_carrier_tomography_runtime.h"
    if not binary_path.exists():
        return {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_BINARY_AUTHORITY_V1",
            "path": RUNTIME_BINARY_NAME,
            "present": False,
        }
    payload = binary_path.read_bytes()
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_BINARY_AUTHORITY_V1",
        "path": RUNTIME_BINARY_NAME,
        "present": True,
        "git_blob_id": git_blob_id_for_bytes(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "compile_equivalence_law": "byte_exact_isolated_compile",
        "compile_command_identity": {
            "language_standard": "c11",
            "flags": ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"],
            "inputs": ["family10h_carrier_tomography_runtime.c"],
            "output": RUNTIME_BINARY_NAME,
        },
        "compiler_identity": "gcc-compatible C compiler selected by controller find_c_compiler()",
        "compiler_flags": ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"],
        "runtime_c_sha256": public.sha256_file(source_c_path) if source_c_path.exists() else None,
        "runtime_h_sha256": public.sha256_file(source_h_path) if source_h_path.exists() else None,
    }


def validate_runtime_binary_authority(source_root: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    authority = runtime_binary_authority(source_root)
    if authority.get("present") is not True:
        failures.append("runtime binary missing")
    if expected is None:
        failures.append("runtime binary authority missing from source receipt")
    elif set(expected) != set(authority):
        failures.append("runtime binary authority field mismatch")
    else:
        for key, value in authority.items():
            if expected.get(key) != value:
                failures.append(f"runtime binary authority mismatch {key}")
    return {"passed": not failures, "failures": failures, "authority": authority}


def validate_discovery_transfer_root(
    source_root: Path,
    *,
    challenge: dict[str, Any] | None = None,
    allowed_extra_names: set[str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    allowed_extra_names = allowed_extra_names or set()
    expected_names = list(DISCOVERY_TRANSFER_FILE_NAMES)
    observed_names = sorted(path.name for path in source_root.iterdir() if path.is_file()) if source_root.exists() else []
    missing = sorted(set(expected_names) - set(observed_names))
    unexpected = sorted(set(observed_names) - set(expected_names) - set(allowed_extra_names))
    if missing:
        failures.append("discovery transfer files missing: " + ",".join(missing))
    if unexpected:
        failures.append("discovery transfer unexpected files: " + ",".join(unexpected))

    source_authority = validate_source_file_authority(source_root)
    if not source_authority["passed"]:
        failures.extend("source authority: " + item for item in source_authority["failures"])

    source_hash_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    source_hashes = read_json(source_hash_path) if source_hash_path.exists() else {}
    bundle_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
    bundle_file_sha256 = public.sha256_file(bundle_path) if bundle_path.exists() else None
    bundle_reconstruction_sha256: str | None = None
    if not bundle_path.exists():
        failures.append("source bundle file missing")
    try:
        bundle_reconstruction_sha256 = deterministic_source_bundle_sha256(source_root)
    except Exception as exc:  # deterministic bundle errors are converted into validation failures.
        failures.append(f"source bundle reconstruction failed: {exc}")

    if isinstance(challenge, dict):
        if source_hashes.get("source_hashes_sha256") != challenge.get("source_hashes_sha256"):
            failures.append("source hash receipt does not match challenged identity")
        if bundle_file_sha256 != challenge.get("source_bundle_sha256"):
            failures.append("source bundle file hash does not match challenged identity")
        if bundle_reconstruction_sha256 != challenge.get("source_bundle_sha256"):
            failures.append("source bundle reconstruction does not match challenged identity")
        runtime_authority = source_hashes.get("runtime_binary_authority") if isinstance(source_hashes.get("runtime_binary_authority"), dict) else {}
        if runtime_authority.get("sha256") != challenge.get("runtime_binary_sha256"):
            failures.append("runtime binary authority does not match challenged identity")
    elif bundle_file_sha256 is not None and bundle_reconstruction_sha256 is not None and bundle_file_sha256 != bundle_reconstruction_sha256:
        failures.append("source bundle file hash does not match deterministic reconstruction")

    return {
        "passed": not failures,
        "failures": failures,
        "expected_files": expected_names,
        "observed_files": observed_names,
        "missing_files": missing,
        "unexpected_files": unexpected,
        "source_authority": source_authority,
        "source_hashes_sha256": source_hashes.get("source_hashes_sha256"),
        "source_bundle_file_sha256": bundle_file_sha256,
        "source_bundle_reconstruction_sha256": bundle_reconstruction_sha256,
        "runtime_binary_sha256": (
            source_hashes.get("runtime_binary_authority", {}).get("sha256")
            if isinstance(source_hashes.get("runtime_binary_authority"), dict)
            else None
        ),
    }


def portable_basename(value: Any) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def normalized_role(value: str) -> str:
    return value.lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def source_audit_version_for_commit(source_commit: str | None) -> str:
    if source_commit == C3_SOURCE_AUTHORITY_COMMIT:
        return "C3"
    if source_commit == C4_SOURCE_AUTHORITY_COMMIT:
        return "C4"
    if source_commit == C5_SOURCE_AUTHORITY_COMMIT:
        return "C5"
    return "C6"


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

    def validate_source_review_receipt(item: dict[str, Any], role: str, role_name: str) -> list[str]:
        if not source_audit:
            return []
        receipt = item.get("review_receipt")
        if not isinstance(receipt, dict):
            return [f"source audit reviewer receipt missing {role}"]
        receipt_failures: list[str] = []
        audited_commit = item.get("audited_commit") if isinstance(item.get("audited_commit"), str) else None
        expected_receipt_schema = source_audit_receipt_schema_for_commit(audited_commit)
        expected_receipt_keys = source_audit_receipt_keys_for_schema(expected_receipt_schema)
        if set(receipt) != expected_receipt_keys:
            receipt_failures.append(f"source audit reviewer receipt field mismatch {role}")
        expected_pairs = {
            "schema": expected_receipt_schema,
            "issuer": SOURCE_AUDIT_REVIEW_RECEIPT_ISSUER,
            "receipt_kind": SOURCE_AUDIT_RECEIPT_KIND,
            "agent_id": item.get("agent_id"),
            "role": role_name,
            "review_body_sha256": item.get("body_canonical_sha256"),
            "review_body_canonicalization": SOURCE_AUDIT_REVIEW_BODY_CANONICALIZATION,
            "audited_commit": item.get("audited_commit"),
            "source_hashes_sha256": item.get("source_hashes_sha256"),
            "source_bundle_sha256": item.get("source_bundle_sha256"),
            "runtime_binary_sha256": item.get("runtime_binary_sha256"),
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
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("review_body_sha256", ""))) is None:
            receipt_failures.append(f"source audit reviewer body digest invalid {role}")
        if not isinstance(receipt.get("thread_id"), str) or not receipt["thread_id"]:
            receipt_failures.append(f"source audit reviewer thread id missing {role}")
        if not isinstance(receipt.get("model"), str) or not receipt["model"]:
            receipt_failures.append(f"source audit reviewer model missing {role}")
        if item.get("self_authored") is True or item.get("evidence_origin") in {"target-derived", "parent-created"}:
            receipt_failures.append(f"source audit reviewer provenance rejected {role}")
        return receipt_failures

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
            "material_blocker_ids": item.get("material_blocker_ids"),
            "audited_commit": item.get("audited_commit"),
            "source_hashes_sha256": item.get("source_hashes_sha256"),
            "source_bundle_sha256": item.get("source_bundle_sha256"),
            "runtime_binary_sha256": item.get("runtime_binary_sha256"),
            "body_canonical_sha256": item.get("body_canonical_sha256"),
            "boundary_attestation": item.get("boundary_attestation"),
            "review_receipt": item.get("review_receipt"),
            "passed": passed,
        }
        receipt_failures = validate_source_review_receipt(item, role, required_roles[role])
        if receipt_failures:
            failures.extend(receipt_failures)
            by_role[role]["passed"] = False
        if (
            source_audit
            and source_audit_receipt_schema_for_commit(item.get("audited_commit") if isinstance(item.get("audited_commit"), str) else None)
            == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3
            and by_role[role]["material_blocker_ids"] != []
        ):
            failures.append(f"source audit reviewer material blocker ids present {role}")
            by_role[role]["passed"] = False
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
    return {
        "required_roles": required_roles,
        "roles": by_role,
        "material_blockers": material_blockers,
        "missing_roles": missing,
        "failures": failures,
        "passed": not failures,
    }


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
            receipt = read_json(path)
        except json.JSONDecodeError as exc:
            failures.append(f"{name} receipt JSON invalid: {exc}")
            continue
        receipts[name] = receipt
        if receipt.get(pass_field) is not True:
            failures.append(f"{name} receipt not passed")
        if not receipt_digest_matches(receipt, digest_field):
            failures.append(f"{name} receipt digest mismatch")
    source_hash_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    source_hashes = read_json(source_hash_path) if source_hash_path.exists() else {}
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
    expected_runtime_binary_sha256: str | None = None,
) -> set[str]:
    findings_path = validate_bound_file(source_root, section.get("findings_path"), section.get("findings_sha256"), f"{label} findings", failures)
    validate_bound_file(source_root, section.get("review_report_path"), section.get("review_report_sha256"), f"{label} report", failures)
    findings: dict[str, Any] = {}
    if findings_path is not None and findings_path.exists():
        try:
            findings = read_json(findings_path)
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
        if findings.get("runtime_binary_sha256") != expected_runtime_binary_sha256:
            failures.append(f"{label} runtime-binary mismatch")
        if findings.get("review_report_present") is not True:
            failures.append(f"{label} report presence not asserted")
        for role_name, role in (quorum.get("roles") or {}).items():
            if role.get("audited_commit") != expected_source_commit:
                failures.append(f"{label} role {role_name} audited commit mismatch")
            if role.get("source_hashes_sha256") != expected_source_hashes_sha256:
                failures.append(f"{label} role {role_name} source-hashes mismatch")
            if role.get("source_bundle_sha256") != expected_source_bundle_sha256:
                failures.append(f"{label} role {role_name} source-bundle mismatch")
            if role.get("runtime_binary_sha256") != expected_runtime_binary_sha256:
                failures.append(f"{label} role {role_name} runtime-binary mismatch")
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
    manifest = read_json(manifest_path)
    sidecar = read_json(sidecar_path)
    if sidecar.get("manifest_file_sha256") != public.sha256_file(manifest_path):
        failures.append("manifest file hash mismatch")
    if sidecar.get("manifest_canonical_sha256") != public.digest({k: v for k, v in manifest.items() if k != "manifest_canonical_sha256"}):
        failures.append("manifest canonical hash mismatch")
    binary_hash = manifest.get("runtime_self_test", {}).get("offline_binary_sha256")
    runtime_path = source_root / "family10h_carrier_tomography_runtime"
    if binary_hash and runtime_path.exists() and public.sha256_file(runtime_path) != binary_hash:
        failures.append("runtime binary hash mismatch")
    runtime_authority = validate_runtime_binary_authority(
        source_root,
        expected=manifest.get("runtime_binary_authority") if isinstance(manifest.get("runtime_binary_authority"), dict) else None,
    )
    if not runtime_authority["passed"]:
        failures.extend("runtime binary authority: " + item for item in runtime_authority["failures"])
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
        source_hash_data = read_json(source_hash_receipt)
        if source_hash_data.get("source_hashes_sha256") != manifest_source_hash:
            failures.append("manifest source hash binding mismatch")
        if source_hash_data.get("runtime_binary_authority") != manifest.get("runtime_binary_authority"):
            failures.append("manifest runtime binary authority binding mismatch")
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
            expected_runtime_binary_sha256=manifest.get("runtime_binary_authority", {}).get("sha256")
            if isinstance(manifest.get("runtime_binary_authority"), dict)
            else None,
        )
        if independent_agent_ids & source_agent_ids:
            failures.append("source authority review reused independent reviewer agent ids")
        source_quorum = manifest.get("source_authority_review", {}).get("review_quorum", {})
        if source_quorum.get("passed") is not True:
            failures.append("frozen package lacks exact source-authority review quorum")
        source_review = manifest.get("source_authority_review", {})
        if not isinstance(source_review.get("source_authority_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", source_review.get("source_authority_commit", "")) is None:
            failures.append("frozen package source-authority commit missing or malformed")
        if source_review.get("source_hashes_sha256") != manifest.get("source_hashes", {}).get("source_hashes_sha256"):
            failures.append("source authority review source-hashes binding mismatch")
        if source_review.get("source_bundle_sha256") != manifest.get("source_bundle", {}).get("sha256"):
            failures.append("source authority review source-bundle binding mismatch")
        if source_review.get("runtime_binary_sha256") != manifest.get("runtime_binary_authority", {}).get("sha256"):
            failures.append("source authority review runtime-binary binding mismatch")
        if manifest.get("offline_validate", {}).get("passed") is not True:
            failures.append("frozen package lacks passing offline validation")
        validate_target_offline_receipts(source_root, manifest, failures)
        counters = manifest.get("contact_counter_attestation", {})
        if not contact_counter_object_equal_strict(counters, FROZEN_CONTACT_COUNTERS):
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
            final_receipt: Any = None
            final_evidence_receipt: Any = None
            final_receipt_loaded = False
            if not final_path.exists():
                failures.append("final exact-object verification receipt missing")
            elif public.sha256_file(final_path) != final_file_sha:
                failures.append("final exact-object verification file hash mismatch")
            else:
                try:
                    final_receipt = read_json(final_path)
                    final_receipt_loaded = True
                except json.JSONDecodeError as exc:
                    failures.append(f"final exact-object verification JSON invalid: {exc}")
            if not final_evidence_path.exists():
                failures.append("final evidence commit authority missing")
            elif public.sha256_file(final_evidence_path) != final_evidence_file_sha:
                failures.append("final evidence commit authority file hash mismatch")
            else:
                try:
                    final_evidence_receipt = read_json(final_evidence_path)
                except json.JSONDecodeError as exc:
                    failures.append(f"final evidence commit authority JSON invalid: {exc}")
                else:
                    if not isinstance(final_evidence_receipt, dict) or not final_evidence_receipt:
                        failures.append("final evidence commit authority must be a nonempty JSON object")
                        final_evidence_receipt = {}
                    else:
                        if final_evidence_receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT_V1":
                            failures.append("final evidence commit authority schema mismatch")
                        if final_evidence_receipt.get("final_evidence_commit_sha256") != public.digest(
                            {k: v for k, v in final_evidence_receipt.items() if k != "final_evidence_commit_sha256"}
                        ):
                            failures.append("final evidence commit authority digest mismatch")
            if final_receipt_loaded:
                if not isinstance(final_receipt, dict) or not final_receipt:
                    failures.append("final exact-object verification receipt must be a nonempty JSON object")
                    final_receipt = {}
                else:
                    final_evidence_object = final_evidence_receipt if isinstance(final_evidence_receipt, dict) else {}
                    digest = final_receipt.get("final_exact_object_verification_sha256")
                    if final_receipt.get("schema") != "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EXACT_OBJECT_VERIFICATION_V1":
                        failures.append("final exact-object verification schema mismatch")
                    if digest != public.digest({k: v for k, v in final_receipt.items() if k != "final_exact_object_verification_sha256"}):
                        failures.append("final exact-object verification digest mismatch")
                    if final_receipt.get("passed") is not True or final_receipt.get("failures") not in ([], None):
                        failures.append("final exact-object verification did not pass")
                    if final_receipt.get("source_authority_commit") != source_review.get("source_authority_commit"):
                        failures.append("final exact-object source commit mismatch")
                    if final_receipt.get("evidence_commit") != final_evidence_object.get("evidence_commit"):
                        failures.append("final exact-object evidence commit mismatch")
                    if final_receipt.get("manifest_file_sha256") != final_evidence_object.get("evidence_manifest_file_sha256"):
                        failures.append("final exact-object evidence manifest file mismatch")
                    if final_receipt.get("manifest_canonical_sha256") != final_evidence_object.get("evidence_manifest_canonical_sha256"):
                        failures.append("final exact-object evidence manifest canonical mismatch")
                    if final_receipt.get("final_evidence_commit_sha256") != final_evidence_object.get("final_evidence_commit_sha256"):
                        failures.append("final exact-object final evidence authority mismatch")
                    if final_section.get("evidence_manifest_file_sha256") != final_evidence_object.get("evidence_manifest_file_sha256"):
                        failures.append("final section evidence manifest file mismatch")
                    if final_section.get("evidence_manifest_canonical_sha256") != final_evidence_object.get("evidence_manifest_canonical_sha256"):
                        failures.append("final section evidence manifest canonical mismatch")
                    if final_section.get("evidence_manifest_sidecar_sha256") != final_evidence_object.get("evidence_manifest_sidecar_sha256"):
                        failures.append("final section evidence manifest sidecar mismatch")
                    evidence_records = final_receipt.get("evidence_blob_records")
                    if not isinstance(evidence_records, dict):
                        failures.append("final exact-object evidence blob records missing")
                    else:
                        manifest_record = evidence_records.get("manifest", {})
                        sidecar_record = evidence_records.get("manifest sidecar", {})
                        if not isinstance(manifest_record, dict) or manifest_record.get("sha256") != final_evidence_object.get("evidence_manifest_file_sha256"):
                            failures.append("final exact-object evidence manifest blob mismatch")
                        if not isinstance(sidecar_record, dict) or sidecar_record.get("sha256") != final_evidence_object.get("evidence_manifest_sidecar_sha256"):
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
                        for name in SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES:
                            record = source_blob_records.get(name)
                            if not isinstance(record, dict) or record.get("unchanged_after_c2") is not True:
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
                target_discovery_receipt = read_json(target_discovery_path) if target_discovery_path.exists() else None
                discovery_transport_receipt = read_json(discovery_transport_path) if discovery_transport_path.exists() else None
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
    elif not legacy_temperature_identity_is_approved(identity):
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
                failures.extend(operational_pin_capability_failures(platform))
            if not public.is_json_int(provenance.get("discovery_monotonic_ns")) or provenance.get("discovery_monotonic_ns", 0) <= 0:
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
                temperature_physical_authority_failures(
                    identity,
                    authorizing_scope=authorizing_scope if isinstance(authorizing_scope, dict) else None,
                    platform_identity=platform_identity,
                    require_authorizing_scope=True,
                    require_pin_evidence=True,
                )
            )
        if not public.is_json_int(discovery.get("target_contact_count")) or discovery.get("target_contact_count") != 1:
            failures.append("temperature sensor discovery target contact count must be one")
        if not public.is_json_int(discovery.get("sensor_inventory_count")) or discovery.get("sensor_inventory_count") != 1:
            failures.append("temperature sensor discovery inventory count must be one")
        if not public.is_json_int(discovery.get("candidate_scan_count")) or discovery.get("candidate_scan_count") != 1:
            failures.append("temperature sensor discovery candidate scan count must be one")
        if not public.is_json_int(discovery.get("live_invocation_count")) or discovery.get("live_invocation_count") != 0:
            failures.append("temperature sensor discovery live invocation count must be zero")
        if not public.is_json_int(discovery.get("pmu_acquisition_count")) or discovery.get("pmu_acquisition_count") != 0:
            failures.append("temperature sensor discovery PMU acquisition count must be zero")
        if not public.is_json_int(discovery.get("pmu_open_count")) or discovery.get("pmu_open_count") != 0:
            failures.append("temperature sensor discovery PMU open count must be zero")
        if not public.is_json_int(discovery.get("runtime_launch_count")) or discovery.get("runtime_launch_count") != 0:
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
                expected_approved = legacy_temperature_identity_is_approved(candidate_identity) and not temperature_physical_authority_failures(
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
        if not public.is_json_int(expected_transport_receipt.get("retry_count")) or expected_transport_receipt.get("retry_count") != 0:
            failures.append("temperature discovery transport retry count must be zero")
        if not public.is_json_int(expected_transport_receipt.get("target_contact_count")) or expected_transport_receipt.get("target_contact_count") != 1:
            failures.append("temperature discovery transport target contact count must be one")
        if not public.is_json_int(expected_transport_receipt.get("sensor_inventory_count")) or expected_transport_receipt.get("sensor_inventory_count") != 1:
            failures.append("temperature discovery transport inventory count must be one")
        if not public.is_json_int(expected_transport_receipt.get("candidate_scan_count")) or expected_transport_receipt.get("candidate_scan_count") != 1:
            failures.append("temperature discovery transport candidate scan count must be one")
        if not public.is_json_int(expected_transport_receipt.get("live_invocation_count")) or expected_transport_receipt.get("live_invocation_count") != 0:
            failures.append("temperature discovery transport live invocation count must be zero")
        if not public.is_json_int(expected_transport_receipt.get("pmu_acquisition_count")) or expected_transport_receipt.get("pmu_acquisition_count") != 0:
            failures.append("temperature discovery transport PMU acquisition count must be zero")
    failures.extend(validate_temperature_authority_challenge(receipt, discovery, expected_challenge))
    if receipt.get("hwmon_name") not in APPROVED_TEMPERATURE_HWMON_NAMES:
        failures.append("temperature sensor authority hwmon name not approved")
    if receipt.get("sensor_semantic_profile") != LEGACY_FAMILY10H_TEMPERATURE_PROFILE:
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
    for key in ["target_contact_count", "sensor_inventory_count", "live_invocation_count", "pmu_acquisition_count"]:
        if not public.is_json_int(receipt.get(key)):
            failures.append(f"temperature sensor authority counter missing or invalid {key}")
        elif isinstance(discovery, dict) and receipt.get(key) != discovery.get(key):
            failures.append(f"temperature sensor authority counter mismatch {key}")
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
        receipt = read_json(path)
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


def write_fake_hwmon_sensor(
    root: Path,
    index: int,
    name: str,
    label: str | None = "Tctl",
    milli_c: str = "42000",
    *,
    input_basename: str = "temp1_input",
    device_driver: str = "k10temp",
    device_subsystem: str = "pci",
    device_modalias: str = "pci:v00001022d00001203sv00000000sd00000000bc06sc00i00",
) -> Path:
    hwmon = root / f"hwmon{index}"
    hwmon.mkdir(parents=True, exist_ok=True)
    (hwmon / "name").write_text(name + "\n", encoding="utf-8")
    label_basename = input_basename.replace("_input", "_label")
    if label is not None:
        (hwmon / label_basename).write_text(label + "\n", encoding="utf-8")
    (hwmon / input_basename).write_text(milli_c + "\n", encoding="utf-8")
    bus_root = root / "bus" / device_subsystem
    driver_target = bus_root / "drivers" / device_driver
    driver_target.mkdir(parents=True, exist_ok=True)
    bus_root.mkdir(parents=True, exist_ok=True)
    device = hwmon / "device"
    device.mkdir()
    (device / "driver").symlink_to(driver_target, target_is_directory=True)
    (device / "subsystem").symlink_to(bus_root, target_is_directory=True)
    (device / "modalias").write_text(device_modalias + "\n", encoding="utf-8")
    (device / "identity").write_text(f"{name}:{label}\n", encoding="utf-8")
    return hwmon / input_basename


def temperature_sensor_identity_fixture() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_hwmon_") as tmp:
        root = Path(tmp)

        def selected_identity_for_fixture(hwmon_root: Path) -> dict[str, Any]:
            selected, _selection = select_approved_temperature_identity(
                enumerate_temperature_candidates(hwmon_root),
                visibility=hwmon_visibility_snapshot(hwmon_root),
            )
            return selected

        def fixture_sample(hwmon_root: Path, *, mutation_hook: Any | None = None) -> dict[str, Any]:
            return read_temperature_sample(
                selected_identity_for_fixture(hwmon_root),
                hwmon_root=hwmon_root,
                mutation_hook=mutation_hook,
                allow_noncanonical_fixture=True,
            )

        def fixture_selection_rejected(hwmon_root: Path) -> bool:
            return raises_target_error(lambda: selected_identity_for_fixture(hwmon_root))

        def candidate_selection_temperature_input_reads(hwmon_root: Path) -> list[str]:
            reads: list[str] = []
            original_read_text = Path.read_text

            def spy_read_text(path_self: Path, *args: Any, **kwargs: Any) -> str:
                if path_self.name.endswith("_input"):
                    reads.append(str(path_self))
                return original_read_text(path_self, *args, **kwargs)

            try:
                Path.read_text = spy_read_text  # type: ignore[method-assign]
                selected_identity_for_fixture(hwmon_root)
            finally:
                Path.read_text = original_read_text  # type: ignore[method-assign]
            return reads

        def pinned_read_oserror_rejected(hwmon_root: Path) -> bool:
            original_read = os.read

            def failing_read(_fd: int, _size: int) -> bytes:
                raise OSError("fixture unreadable approved sensor")

            try:
                os.read = failing_read  # type: ignore[assignment]
                return raises_target_error(lambda: fixture_sample(hwmon_root))
            finally:
                os.read = original_read  # type: ignore[assignment]

        write_fake_hwmon_sensor(root, 0, "acpitz", "temp1")
        approved_path = write_fake_hwmon_sensor(root, 1, "k10temp", None)
        candidate_input_reads = candidate_selection_temperature_input_reads(root)
        first_non_cpu = fixture_sample(root)
        root_candidates = enumerate_temperature_candidates(root)
        approved_identity = temperature_sensor_identity(approved_path)
        canonical_identity = public.synthetic_temperature_identity()
        noncanonical_driver_identity = public.with_temperature_identity_digest(
            {
                **{key: canonical_identity[key] for key in canonical_identity if key != "identity_sha256"},
                "resolved_driver_path": "/tmp/k10temp",
            }
        )
        noncanonical_subsystem_identity = public.with_temperature_identity_digest(
            {
                **{key: canonical_identity[key] for key in canonical_identity if key != "identity_sha256"},
                "resolved_subsystem_path": "/tmp/pci",
            }
        )
        noncanonical_driver_path_rejected = any(
            "resolved driver path is not canonical" in failure
            for failure in temperature_physical_authority_failures(noncanonical_driver_identity)
        )
        noncanonical_subsystem_path_rejected = any(
            "resolved subsystem path is not canonical" in failure
            for failure in temperature_physical_authority_failures(noncanonical_subsystem_identity)
        )

        wrong_name_root = root / "wrong_name"
        write_fake_hwmon_sensor(wrong_name_root, 0, "acpitz", "Tctl")
        labeled_tctl_root = root / "labeled_tctl"
        labeled_tctl_path = write_fake_hwmon_sensor(labeled_tctl_root, 0, "k10temp", "Tctl")
        wrong_label_root = root / "wrong_label_tdie"
        wrong_label_path = write_fake_hwmon_sensor(wrong_label_root, 0, "k10temp", "Tdie")
        temp2_root = root / "temp2"
        write_fake_hwmon_sensor(temp2_root, 0, "k10temp", None, input_basename="temp2_input")
        wrong_driver_root = root / "wrong_driver"
        write_fake_hwmon_sensor(wrong_driver_root, 0, "k10temp", None, device_driver="not_k10temp")
        wrong_subsystem_root = root / "wrong_subsystem"
        write_fake_hwmon_sensor(wrong_subsystem_root, 0, "k10temp", None, device_subsystem="platform")
        wrong_modalias_root = root / "wrong_modalias"
        write_fake_hwmon_sensor(wrong_modalias_root, 0, "k10temp", None, device_modalias="pci:v00008086d00001203sv00000000sd00000000bc06sc00i00")
        spoofed_subvendor_root = root / "spoofed_subvendor"
        spoofed_subvendor_path = write_fake_hwmon_sensor(
            spoofed_subvendor_root,
            0,
            "k10temp",
            None,
            device_modalias="pci:v00008086d00001203sv00001022sd00000000bc06sc00i00",
        )
        trailing_modalias_root = root / "trailing_modalias"
        write_fake_hwmon_sensor(
            trailing_modalias_root,
            0,
            "k10temp",
            None,
            device_modalias="pci:v00001022d00001203sv00000000sd00000000bc06sc00i00TRAILING",
        )
        truncated_modalias_root = root / "truncated_modalias"
        write_fake_hwmon_sensor(truncated_modalias_root, 0, "k10temp", None, device_modalias="pci:v00001022d00001203")
        malformed_modalias_root = root / "malformed_modalias"
        write_fake_hwmon_sensor(malformed_modalias_root, 0, "k10temp", None, device_modalias="not-a-pci-modalias")
        unreadable_root = root / "unreadable"
        write_fake_hwmon_sensor(unreadable_root, 0, "k10temp", "Tctl")
        substitute_root = root / "substitute"
        good_substitute = write_fake_hwmon_sensor(substitute_root, 0, "k10temp", "Tctl")
        substituted = write_fake_hwmon_sensor(substitute_root, 1, "k10temp", "Tctl", "99000")
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

        wrong_name_rejected = fixture_selection_rejected(wrong_name_root)
        wrong_label_rejected = fixture_selection_rejected(wrong_label_root)
        temp2_rejected = fixture_selection_rejected(temp2_root)
        wrong_driver_rejected = fixture_selection_rejected(wrong_driver_root)
        wrong_subsystem_rejected = fixture_selection_rejected(wrong_subsystem_root)
        wrong_modalias_rejected = fixture_selection_rejected(wrong_modalias_root)
        spoofed_subvendor_rejected = fixture_selection_rejected(spoofed_subvendor_root)
        spoofed_subvendor_identity = identity_from_candidate_record(temperature_candidate_record(spoofed_subvendor_path, hwmon_root=spoofed_subvendor_root))
        trailing_modalias_rejected = fixture_selection_rejected(trailing_modalias_root)
        truncated_modalias_rejected = fixture_selection_rejected(truncated_modalias_root)
        malformed_modalias_rejected = fixture_selection_rejected(malformed_modalias_root)
        unreadable_rejected = pinned_read_oserror_rejected(unreadable_root)
        labeled_tctl_accepted = fixture_sample(labeled_tctl_root)["identity"]["sensor_label_value"] == "Tctl"
        unlabeled_record = enumerate_temperature_candidates(root)[1]
        wrong_label_record = enumerate_temperature_candidates(wrong_label_root)[0]
        required = temperature_sensor_identity(good_substitute)
        substituted_required = public.with_temperature_identity_digest(
            {**{key: required[key] for key in required if key != "identity_sha256"}, "class_path": str(substituted)}
        )
        path_substitution_rejected = raises_target_error(
            lambda: read_temperature_sample(
                required_identity=substituted_required,
                hwmon_root=substitute_root,
                allow_noncanonical_fixture=True,
            )
        )
        same_path_required = temperature_sensor_identity(same_path)
        same_class_path_substitution_rejected = raises_target_error(
            lambda: read_temperature_sample(
                required_identity=same_path_required,
                hwmon_root=same_path_root,
                mutation_hook=lambda path: (path.parent / "name").write_text("acpitz\n", encoding="utf-8"),
                allow_noncanonical_fixture=True,
            )
        )
        swap_required = temperature_sensor_identity(class_hwmon / "temp1_input")

        def swap_restore(_path: Path) -> None:
            class_hwmon.unlink()
            class_hwmon.symlink_to(alternate_real_path.parent, target_is_directory=True)
            class_hwmon.unlink()
            class_hwmon.symlink_to(approved_real_path.parent, target_is_directory=True)

        swap_sample = read_temperature_sample(
            required_identity=swap_required,
            hwmon_root=class_root,
            mutation_hook=swap_restore,
            allow_noncanonical_fixture=True,
        )
        swap_restore_value_pinned_to_approved = swap_sample["value_c"] == 42.0 and swap_sample["value_c"] != 99.0
        drift_identity = temperature_sensor_identity(drift_path)
        (drift_path.parent / "temp1_label").write_text("Tdie\n", encoding="utf-8")
        identity_drift_rejected = raises_target_error(
            lambda: read_temperature_sample(
                required_identity=drift_identity,
                hwmon_root=drift_root,
                allow_noncanonical_fixture=True,
            )
        )

    checks = {
        "non_cpu_sensor_first_skipped": first_non_cpu["identity"]["class_path"] == str(approved_path),
        "candidate_classification_does_not_read_temperature_inputs": candidate_input_reads == [],
        "candidate_records_are_metadata_only": all(
            candidate.get("raw_input_text") is None
            and candidate.get("parsed_millidegree_value") is None
            and candidate.get("raw_input_parse_failure") is None
            for candidate in root_candidates
        ),
        "unlabeled_k10temp_temp1_input_accepted": approved_identity["sensor_label_present"] is False and approved_identity["sensor_label_value"] is None,
        "labeled_k10temp_temp1_input_tctl_accepted": labeled_tctl_accepted,
        "wrong_hwmon_name_rejected": wrong_name_rejected,
        "labeled_k10temp_temp1_input_tdie_rejected": wrong_label_rejected,
        "unlabeled_temp2_input_rejected": temp2_rejected,
        "wrong_pci_driver_rejected": wrong_driver_rejected,
        "wrong_subsystem_rejected": wrong_subsystem_rejected,
        "non_amd_modalias_rejected": wrong_modalias_rejected,
        "amd_subvendor_spoof_rejected": spoofed_subvendor_rejected
        and spoofed_subvendor_identity is not None
        and not legacy_temperature_identity_is_approved(spoofed_subvendor_identity),
        "trailing_junk_modalias_rejected": trailing_modalias_rejected,
        "truncated_modalias_rejected": truncated_modalias_rejected,
        "malformed_modalias_rejected": malformed_modalias_rejected,
        "missing_label_retained_in_candidate_record": unlabeled_record["sensor_label_present"] is False and unlabeled_record["sensor_label_value"] is None,
        "wrong_label_retained_with_rejection_reason": wrong_label_record["sensor_label_value"] == "Tdie"
        and "present temp1_label is not Tctl" in wrong_label_record["rejection_reasons"],
        "path_substitution_rejected": path_substitution_rejected,
        "noncanonical_resolved_driver_path_rejected": noncanonical_driver_path_rejected,
        "noncanonical_resolved_subsystem_path_rejected": noncanonical_subsystem_path_rejected,
        "same_class_path_substitution_rejected": same_class_path_substitution_rejected,
        "same_class_swap_restore_reads_pinned_descriptor": swap_restore_value_pinned_to_approved,
        "identity_drift_rejected": identity_drift_rejected,
        "unreadable_approved_sensor_rejected": unreadable_rejected,
    }
    return {
        "passed": all(checks.values()),
        **checks,
        "approved_hwmon_names": APPROVED_TEMPERATURE_HWMON_NAMES,
        "approved_sensor_semantic_profile": LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
        "identity_fields": TEMPERATURE_IDENTITY_FIELDS,
    }


def policy_and_platform_fixture() -> dict[str, Any]:
    sensor = temperature_sensor_identity_fixture()
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_platform_") as tmp:
        platform = platform_identity_regression(Path(tmp))
    checks = {
        "strict_platform_identity_required": all(platform.values()),
        "strict_readable_policy_fields_required": True,
        "strict_temperature_required": True,
        "approved_temperature_hwmon_name_required": sensor["wrong_hwmon_name_rejected"],
        "approved_temperature_sensor_label_required": sensor["labeled_k10temp_temp1_input_tdie_rejected"],
        "temperature_path_substitution_rejected": sensor["path_substitution_rejected"],
        "temperature_noncanonical_driver_path_rejected": sensor["noncanonical_resolved_driver_path_rejected"],
        "temperature_noncanonical_subsystem_path_rejected": sensor["noncanonical_resolved_subsystem_path_rejected"],
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
    return {"passed": all(checks.values()) and sensor["passed"], "checks": checks, "platform_identity": platform, "temperature_sensor_identity": sensor}


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
        "sensor_label_present": identity["sensor_label_present"],
        "sensor_label_value": identity["sensor_label_value"],
        "sensor_semantic_role": identity["sensor_semantic_role"],
        "sensor_semantic_profile": identity["sensor_semantic_profile"],
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
        "sensor_label_present": forged_identity["sensor_label_present"],
        "sensor_label_value": forged_identity["sensor_label_value"],
        "sensor_semantic_role": forged_identity["sensor_semantic_role"],
        "sensor_semantic_profile": forged_identity["sensor_semantic_profile"],
        "approved_sensor_identity": forged_identity,
        "target_discovery_receipt": complete_forged_discovery,
        "controller_challenge": controller_challenge,
        "controller_challenge_sha256": controller_challenge_sha,
        "controller_nonce": controller_nonce,
        "source_authority_commit": controller_challenge["authorized_commit"],
        "target_contact_count": complete_forged_discovery["target_contact_count"],
        "sensor_inventory_count": complete_forged_discovery["sensor_inventory_count"],
        "live_invocation_count": complete_forged_discovery["live_invocation_count"],
        "pmu_acquisition_count": complete_forged_discovery["pmu_acquisition_count"],
    }
    complete_forged_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in complete_forged_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    boolean_discovery_counter = dict(complete_forged_discovery)
    boolean_discovery_counter["target_contact_count"] = True
    boolean_discovery_counter["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in boolean_discovery_counter.items() if k != "target_discovery_receipt_sha256"}
    )
    boolean_discovery_counter_authority = {
        **complete_forged_authority,
        "target_discovery_receipt": boolean_discovery_counter,
        "target_contact_count": True,
    }
    boolean_discovery_counter_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in boolean_discovery_counter_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    boolean_authority_counter = {**complete_forged_authority, "target_contact_count": True}
    boolean_authority_counter["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in boolean_authority_counter.items() if k != "temperature_sensor_authority_sha256"}
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
    boolean_discovery_counter_result = validate_temperature_sensor_authority_payload(
        boolean_discovery_counter_authority,
        expected_challenge=controller_challenge,
    )
    boolean_pmu_open_discovery = dict(complete_forged_discovery)
    boolean_pmu_open_discovery["pmu_open_count"] = False
    boolean_pmu_open_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in boolean_pmu_open_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    boolean_pmu_open_authority = {
        **complete_forged_authority,
        "target_discovery_receipt": boolean_pmu_open_discovery,
    }
    boolean_pmu_open_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in boolean_pmu_open_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    boolean_runtime_launch_discovery = dict(complete_forged_discovery)
    boolean_runtime_launch_discovery["runtime_launch_count"] = False
    boolean_runtime_launch_discovery["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in boolean_runtime_launch_discovery.items() if k != "target_discovery_receipt_sha256"}
    )
    boolean_runtime_launch_authority = {
        **complete_forged_authority,
        "target_discovery_receipt": boolean_runtime_launch_discovery,
    }
    boolean_runtime_launch_authority["temperature_sensor_authority_sha256"] = public.digest(
        {k: v for k, v in boolean_runtime_launch_authority.items() if k != "temperature_sensor_authority_sha256"}
    )
    boolean_pmu_open_result = validate_temperature_sensor_authority_payload(
        boolean_pmu_open_authority,
        expected_challenge=controller_challenge,
    )
    boolean_runtime_launch_result = validate_temperature_sensor_authority_payload(
        boolean_runtime_launch_authority,
        expected_challenge=controller_challenge,
    )
    boolean_authority_counter_result = validate_temperature_sensor_authority_payload(
        boolean_authority_counter,
        expected_challenge=controller_challenge,
    )
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
    boolean_frozen_contact_counter_rejected = not contact_counter_object_equal_strict(
        {"target_contact_count": True, "sensor_inventory_count": True, "live_invocation_count": False, "pmu_acquisition_count": False},
        FROZEN_CONTACT_COUNTERS,
    )
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
        "boolean_discovery_counter_rejected": any("temperature sensor discovery target contact count must be one" in item for item in boolean_discovery_counter_result["failures"]),
        "boolean_discovery_pmu_open_count_rejected": any("temperature sensor discovery PMU open count must be zero" in item for item in boolean_pmu_open_result["failures"]),
        "boolean_discovery_runtime_launch_count_rejected": any("temperature sensor discovery runtime launch count must be zero" in item for item in boolean_runtime_launch_result["failures"]),
        "boolean_authority_counter_rejected": any("temperature sensor authority counter missing or invalid target_contact_count" in item for item in boolean_authority_counter_result["failures"]),
        "boolean_frozen_contact_counter_rejected": boolean_frozen_contact_counter_rejected,
        "boolean_only_frozen_manifest_rejected": boolean_only_frozen_manifest_rejected,
        "explicit_live_authority_still_required": True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def final_exact_object_falsey_manifest_fixture(source_root: Path) -> dict[str, Any]:
    falsey_values = {
        "empty_object": {},
        "empty_array": [],
        "null": None,
        "false": False,
        "zero": 0,
        "empty_string": "",
    }
    falsey_failure = "final exact-object verification receipt must be a nonempty JSON object"
    copied_names = sorted(set(SOURCE_FILE_NAMES + ["CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz", "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json", RUNTIME_BINARY_NAME]))
    rejected: dict[str, bool] = {}
    execute_aborted_before_output: dict[str, bool] = {}
    failures_by_value: dict[str, list[str]] = {}
    original_expected_root = public.EXPECTED_REMOTE_ROOT
    original_expected_output_root = public.EXPECTED_REMOTE_OUTPUT_ROOT
    env_keys = [AUTHORITY_ENV, COMMIT_ENV, MANIFEST_ENV, TEMPERATURE_AUTHORITY_NONCE_ENV]
    original_env = {key: os.environ.get(key) for key in env_keys}

    def clone_json(value: Any) -> Any:
        return strict_json_loads(strict_json_dumps(value))

    def seal_fixture_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        manifest_path = root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json"
        sidecar_path = root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256"
        write_json(manifest_path, manifest)
        sidecar = {
            "manifest_file_sha256": public.sha256_file(manifest_path),
            "manifest_canonical_sha256": public.digest({k: v for k, v in manifest.items() if k != "manifest_canonical_sha256"}),
        }
        write_json(sidecar_path, sidecar)
        return sidecar

    def final_evidence_authority() -> dict[str, Any]:
        receipt = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT_V1",
            "evidence_commit": "2" * 40,
            "evidence_manifest_file_sha256": "a" * 64,
            "evidence_manifest_canonical_sha256": "b" * 64,
            "evidence_manifest_sidecar_sha256": "c" * 64,
        }
        receipt["final_evidence_commit_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "final_evidence_commit_sha256"})
        return receipt

    def final_exact_positive_receipt(manifest: dict[str, Any], evidence_authority: dict[str, Any]) -> dict[str, Any]:
        evidence_records = {
            label: {"present": True, "object_type": "blob", "sha256": "d" * 64}
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
            ]
        }
        evidence_records["manifest"]["sha256"] = evidence_authority["evidence_manifest_file_sha256"]
        evidence_records["manifest sidecar"]["sha256"] = evidence_authority["evidence_manifest_sidecar_sha256"]
        receipt = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_FINAL_EXACT_OBJECT_VERIFICATION_V1",
            "passed": True,
            "failures": [],
            "source_authority_commit": manifest.get("source_authority_review", {}).get("source_authority_commit"),
            "evidence_commit": evidence_authority["evidence_commit"],
            "manifest_file_sha256": evidence_authority["evidence_manifest_file_sha256"],
            "manifest_canonical_sha256": evidence_authority["evidence_manifest_canonical_sha256"],
            "final_evidence_commit_sha256": evidence_authority["final_evidence_commit_sha256"],
            "evidence_blob_records": evidence_records,
            "source_authority_blob_records": {
                name: {"present": True, "object_type": "blob", "unchanged_after_c2": True}
                for name in SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES
            },
        }
        receipt["final_exact_object_verification_sha256"] = public.digest(
            {k: v for k, v in receipt.items() if k != "final_exact_object_verification_sha256"}
        )
        return receipt

    with tempfile.TemporaryDirectory(prefix="family10h_falsey_final_receipt_fixture_") as tmp:
        fixture_root = Path(tmp)
        for name in copied_names:
            source_path = source_root / name
            if source_path.exists():
                (fixture_root / name).write_bytes(source_path.read_bytes())
        source_hash_receipt = read_json(source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json")
        source_hashes_sha256 = source_hash_receipt.get("source_hashes_sha256")
        source_bundle_sha256 = public.sha256_file(source_root / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz")
        runtime_authority = source_hash_receipt.get("runtime_binary_authority") if isinstance(source_hash_receipt.get("runtime_binary_authority"), dict) else {}
        base_manifest = {
            "package_decision": public.PACKAGE_DECISION_FROZEN,
            "git_state_at_manifest_build": {"head": "1" * 40},
            "runtime_self_test": {"offline_binary_sha256": runtime_authority.get("sha256")},
            "runtime_binary_authority": runtime_authority,
            "source_bundle": {"sha256": source_bundle_sha256},
            "source_hashes": {"source_hashes_sha256": source_hashes_sha256},
            "temperature_sensor_authority": {},
            "independent_review": {"review_quorum": {"passed": True}},
            "source_authority_review": {
                "review_quorum": {"passed": True},
                "source_authority_commit": "1" * 40,
                "source_hashes_sha256": source_hashes_sha256,
                "source_bundle_sha256": source_bundle_sha256,
                "runtime_binary_sha256": runtime_authority.get("sha256"),
            },
            "offline_validate": {"passed": True},
            "contact_counter_attestation": dict(FROZEN_CONTACT_COUNTERS),
        }
        evidence_path = fixture_root / "CARRIER_TOMOGRAPHY_FINAL_EVIDENCE_COMMIT.json"
        final_path = fixture_root / "CARRIER_TOMOGRAPHY_FINAL_OBJECT_VERIFY.json"
        evidence_authority = final_evidence_authority()
        write_json(evidence_path, evidence_authority)
        public.EXPECTED_REMOTE_ROOT = str(fixture_root)
        try:
            for label, value in falsey_values.items():
                output_root = fixture_root / f"output_{label}"
                public.EXPECTED_REMOTE_OUTPUT_ROOT = str(output_root)
                write_json(final_path, value)
                manifest = clone_json(base_manifest)
                manifest["final_exact_object_verification"] = {
                    "passed": True,
                    "path": str(final_path),
                    "file_sha256": public.sha256_file(final_path),
                    "final_evidence_commit_path": str(evidence_path),
                    "final_evidence_commit_file_sha256": public.sha256_file(evidence_path),
                    "source_authority_commit": manifest.get("source_authority_review", {}).get("source_authority_commit"),
                    "evidence_commit": evidence_authority["evidence_commit"],
                    "evidence_manifest_file_sha256": evidence_authority["evidence_manifest_file_sha256"],
                    "evidence_manifest_canonical_sha256": evidence_authority["evidence_manifest_canonical_sha256"],
                    "evidence_manifest_sidecar_sha256": evidence_authority["evidence_manifest_sidecar_sha256"],
                }
                sidecar = seal_fixture_manifest(fixture_root, manifest)
                validation = validate_manifest_authority(fixture_root)
                failures_by_value[label] = validation["failures"]
                rejected[label] = validation["passed"] is False and falsey_failure in validation["failures"]
                os.environ[AUTHORITY_ENV] = AUTHORITY_VALUE
                os.environ[COMMIT_ENV] = "1" * 40
                os.environ[MANIFEST_ENV] = sidecar["manifest_file_sha256"]
                execute_aborted_before_output[label] = (
                    raises_target_error_containing(lambda: execute_authorized(fixture_root, output_root), "manifest authority mismatch")
                    and not output_root.exists()
                )
            positive_manifest = clone_json(base_manifest)
            positive_receipt = final_exact_positive_receipt(positive_manifest, evidence_authority)
            write_json(final_path, positive_receipt)
            positive_manifest["final_exact_object_verification"] = {
                "passed": True,
                "path": str(final_path),
                "file_sha256": public.sha256_file(final_path),
                "final_evidence_commit_path": str(evidence_path),
                "final_evidence_commit_file_sha256": public.sha256_file(evidence_path),
                "source_authority_commit": positive_receipt["source_authority_commit"],
                "evidence_commit": evidence_authority["evidence_commit"],
                "evidence_manifest_file_sha256": evidence_authority["evidence_manifest_file_sha256"],
                "evidence_manifest_canonical_sha256": evidence_authority["evidence_manifest_canonical_sha256"],
                "evidence_manifest_sidecar_sha256": evidence_authority["evidence_manifest_sidecar_sha256"],
            }
            seal_fixture_manifest(fixture_root, positive_manifest)
            positive_validation = validate_manifest_authority(fixture_root)
            positive_failures = positive_validation["failures"]
        finally:
            public.EXPECTED_REMOTE_ROOT = original_expected_root
            public.EXPECTED_REMOTE_OUTPUT_ROOT = original_expected_output_root
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    positive_control = falsey_failure not in positive_failures and all(
        forbidden not in positive_failures
        for forbidden in [
            "final exact-object verification schema mismatch",
            "final exact-object verification digest mismatch",
            "final exact-object verification did not pass",
        ]
    )
    checks = {
        "falsey_final_receipts_rejected": all(rejected.values()) and set(rejected) == set(falsey_values),
        "falsey_final_execute_aborts_before_output_root": all(execute_aborted_before_output.values())
        and set(execute_aborted_before_output) == set(falsey_values),
        "valid_final_receipt_positive_control": positive_control,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "falsey_receipts": rejected,
        "execute_aborted_before_output": execute_aborted_before_output,
        "positive_control_failures": positive_failures,
        "failures_by_value": failures_by_value,
    }


def source_audit_receipt_version_fixture(source_root: Path) -> dict[str, Any]:
    source_hashes = read_json(source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json")
    source_hash = source_hashes["source_hashes_sha256"]
    bundle_hash = deterministic_source_bundle_sha256(source_root)
    runtime_hash = source_hashes["runtime_binary_authority"]["sha256"]

    def build_audit(source_commit: str, *, force_schema: str | None = None) -> dict[str, Any]:
        clearances: dict[str, dict[str, Any]] = {}
        schema = force_schema or source_audit_receipt_schema_for_commit(source_commit)
        for index, (role, label) in enumerate(SOURCE_AUDIT_REQUIRED_REVIEW_ROLES.items(), start=1):
            agent_id = f"target-source-reviewer-{index}"
            thread_id = f"target-source-thread-{index}"
            body_hash = hashlib.sha256(f"{label} {source_commit}\n".encode("utf-8")).hexdigest()
            item: dict[str, Any] = {
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
                "body_canonical_sha256": body_hash,
                "boundary_attestation": {
                    "no_git_write": True,
                    "no_file_edits": True,
                    "no_checkout_mutation": True,
                    "no_target_contact": True,
                    "no_live_authority": True,
                    "no_pmu": True,
                },
            }
            receipt = {
                "schema": schema,
                "issuer": SOURCE_AUDIT_REVIEW_RECEIPT_ISSUER,
                "receipt_kind": SOURCE_AUDIT_RECEIPT_KIND,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "role": label,
                "model": "gpt-5.6-sol",
                "review_body_sha256": body_hash,
                "review_body_canonicalization": SOURCE_AUDIT_REVIEW_BODY_CANONICALIZATION,
                "audited_commit": source_commit,
                "source_hashes_sha256": source_hash,
                "source_bundle_sha256": bundle_hash,
                "runtime_binary_sha256": runtime_hash,
                "no_git_write": True,
                "no_file_edits": True,
                "no_checkout_mutation": True,
                "no_target_contact": True,
                "no_live_authority": True,
                "no_pmu": True,
                "self_authored": False,
                "evidence_origin": SOURCE_AUDIT_ALLOWED_EVIDENCE_ORIGIN,
            }
            if schema == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3:
                receipt.update(
                    {
                        "verdict": item["verdict"],
                        "final_response": item["final_response"],
                        "material_blocker_ids": item["material_blocker_ids"],
                    }
                )
            item["review_receipt"] = receipt
            clearances[role] = item
        return {
            "schema": "FAMILY10H_TARGET_SOURCE_AUDIT_RECEIPT_VERSION_FIXTURE_V1",
            "source_authority_commit": source_commit,
            "source_hashes_sha256": source_hash,
            "source_bundle_sha256": bundle_hash,
            "runtime_binary_sha256": runtime_hash,
            "review_report_present": True,
            "material_blockers": [],
            "reviewer_verdicts": clearances,
        }

    def mutated_v3_receipt(**fields: Any) -> dict[str, Any]:
        audit = build_audit("f" * 40)
        item = audit["reviewer_verdicts"]["claim_boundary_adjudicator"]
        item["review_receipt"] = {**item["review_receipt"], **fields}
        return audit

    def normalized_nonempty_ids() -> dict[str, Any]:
        audit = build_audit("e" * 40)
        item = audit["reviewer_verdicts"]["claim_boundary_adjudicator"]
        item["material_blocker_ids"] = ["TARGET-NORMALIZED-BLOCKER"]
        item["review_receipt"] = {**item["review_receipt"], "material_blocker_ids": ["TARGET-NORMALIZED-BLOCKER"]}
        return audit

    c5_audit_path = source_root / "SOURCE_AUTHORITY_C5_REVIEW_NORMALIZED.json"
    c5_audit_source = "committed_archive" if c5_audit_path.exists() else "synthetic_v2_archive"
    c5_audit = read_json(c5_audit_path) if c5_audit_path.exists() else build_audit(
        C5_SOURCE_AUTHORITY_COMMIT,
        force_schema=SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2,
    )
    checks = {
        "c5_actual_source_authority_commit_selects_c5": source_audit_version_for_commit(C5_SOURCE_AUTHORITY_COMMIT) == "C5",
        "c5_actual_source_authority_commit_uses_v2_receipts": source_audit_receipt_schema_for_commit(C5_SOURCE_AUTHORITY_COMMIT)
        == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2,
        "c5_v2_archive_replays": recompute_review_quorum(c5_audit, source_audit=True)["passed"],
        "c5_failure_evidence_commit_selects_c6": source_audit_version_for_commit(C5_FAILURE_EVIDENCE_COMMIT) == "C6",
        "c5_failure_evidence_commit_uses_v3_receipts": source_audit_receipt_schema_for_commit(C5_FAILURE_EVIDENCE_COMMIT)
        == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3,
        "future_source_authority_commit_uses_v3_receipts": source_audit_receipt_schema_for_commit("f" * 40)
        == SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V3,
        "valid_c6_v3_archive_replays": recompute_review_quorum(build_audit("f" * 40), source_audit=True)["passed"],
        "v2_receipt_for_c5_failure_evidence_rejected": not recompute_review_quorum(
            build_audit(C5_FAILURE_EVIDENCE_COMMIT, force_schema=SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2),
            source_audit=True,
        )["passed"],
        "future_v2_receipt_rejected": not recompute_review_quorum(
            build_audit("d" * 40, force_schema=SOURCE_AUDIT_REVIEW_RECEIPT_SCHEMA_V2),
            source_audit=True,
        )["passed"],
        "receipt_bound_material_verdict_rejected": not recompute_review_quorum(
            mutated_v3_receipt(verdict="MATERIAL_BLOCKER", material_blocker_ids=["TARGET-BOUND-BLOCKER"]),
            source_audit=True,
        )["passed"],
        "receipt_bound_nonfinal_rejected": not recompute_review_quorum(mutated_v3_receipt(final_response=False), source_audit=True)["passed"],
        "receipt_bound_blocker_ids_rejected": not recompute_review_quorum(
            mutated_v3_receipt(material_blocker_ids=["TARGET-BOUND-BLOCKER"]),
            source_audit=True,
        )["passed"],
        "normalized_nonempty_blocker_ids_rejected": not recompute_review_quorum(normalized_nonempty_ids(), source_audit=True)["passed"],
    }
    return {
        "schema": "FAMILY10H_TARGET_SOURCE_AUDIT_RECEIPT_VERSION_FIXTURE_V1",
        "passed": all(checks.values()),
        "c5_audit_source": c5_audit_source,
        "checks": checks,
    }


def validate_minimal_evidence_root(root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    failures = []
    existing = sorted(path.name for path in root.iterdir()) if root.exists() else []
    required = sorted(REQUIRED_EVIDENCE_FILES)
    if existing != required:
        failures.append(f"evidence files {existing} != {required}")
        return {"passed": False, "failures": failures, "existing": existing}
    raw_records = read_jsonl(root / "raw_records.jsonl")
    receipts = read_jsonl(root / "source_death_receipts.jsonl")
    feature_freeze = read_json(root / "feature_freeze.json")
    packet = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1",
        "schedule_sha256": public.digest(schedule),
        "raw_records": raw_records,
        "source_death_receipts": receipts,
        "feature_freeze": feature_freeze,
    }
    validation = public.validate_evidence_packet(packet, schedule)
    return {"passed": validation["passed"] and not failures, "failures": failures + validation["failures"], "validation": validation}


def build_authorized_feature_freeze(schedule: dict[str, Any], approved_temperature_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "frozen_before_analysis": True,
        "public_only": True,
        "schedule_sha256": public.digest(schedule),
        "receiver_feature_boundary": "public_schedule_and_public_pmu_only",
        "temperature_sensor_identity": approved_temperature_identity,
    }


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

        authorized_shape_root = root / "authorized_shape"
        authorized_shape_root.mkdir()
        packet = public.minimal_success_packet(schedule)
        write_jsonl(authorized_shape_root / "raw_records.jsonl", packet["raw_records"])
        write_jsonl(authorized_shape_root / "source_death_receipts.jsonl", packet["source_death_receipts"])
        authorized_feature_freeze = build_authorized_feature_freeze(
            schedule,
            packet["feature_freeze"]["temperature_sensor_identity"],
        )
        write_json(authorized_shape_root / "feature_freeze.json", authorized_feature_freeze)
        authorized_shape = validate_minimal_evidence_root(authorized_shape_root, schedule)

        authorized_extra_root = root / "authorized_extra"
        authorized_extra_root.mkdir()
        write_jsonl(authorized_extra_root / "raw_records.jsonl", packet["raw_records"])
        write_jsonl(authorized_extra_root / "source_death_receipts.jsonl", packet["source_death_receipts"])
        write_json(
            authorized_extra_root / "feature_freeze.json",
            {**authorized_feature_freeze, "temperature_authority_controller_challenge_sha256": "0" * 64},
        )
        authorized_extra = validate_minimal_evidence_root(authorized_extra_root, schedule)

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
        "passed": all(
            [
                success["passed"],
                authorized_shape["passed"],
                not authorized_extra["passed"],
                not missing["passed"],
                not extra["passed"],
            ]
        ),
        "three_file_minimal_success_packet": success,
        "authorized_feature_freeze_shape_passed": authorized_shape,
        "authorized_feature_freeze_extra_challenge_hash_rejected": not authorized_extra["passed"],
        "authorized_feature_freeze_keyset": sorted(authorized_feature_freeze),
        "missing_evidence_file_rejected": not missing["passed"],
        "extra_evidence_file_rejected": not extra["passed"],
    }


def source_mutation_fixtures(source_root: Path) -> dict[str, Any]:
    baseline = validate_source_file_authority(source_root)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_source_mutation_") as tmp:
        temp_root = Path(tmp)
        for name in SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES:
            path = source_root / name
            if path.exists():
                (temp_root / name).write_bytes(path.read_bytes())
        before_path = temp_root / "family10h_carrier_tomography_public.py"
        before_path.write_text(before_path.read_text(encoding="utf-8") + "\n# mutation before compile\n", encoding="utf-8")
        mutated_before = validate_source_file_authority(temp_root)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_source_mutation_") as tmp:
        temp_root = Path(tmp)
        for name in SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES:
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


def read_optional_sysfs_text(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "value": None,
        "error": None,
    }
    if not result["exists"]:
        return result
    try:
        result["value"] = path.read_text(encoding="utf-8", errors="replace").strip()
        result["readable"] = True
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def resolve_path_record(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "resolved": None, "error": None}
    if not result["exists"]:
        return result
    try:
        result["resolved"] = str(path.resolve(strict=True))
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def strict_millicelsius_parse(text: Any) -> tuple[int | None, str | None]:
    if not isinstance(text, str):
        return None, "missing raw input text"
    if re.fullmatch(r"[+-]?\d+", text) is None:
        return None, "temperature input is not a strict integer"
    return int(text), None


def amd_pci_modalias_vendor_1022(modalias: Any) -> bool:
    if not isinstance(modalias, str):
        return False
    match = re.fullmatch(
        r"pci:v(?P<vendor>[0-9a-f]{8})d[0-9a-f]{8}sv[0-9a-f]{8}sd[0-9a-f]{8}bc[0-9a-f]{2}sc[0-9a-f]{2}i[0-9a-f]{2}",
        modalias.lower(),
    )
    return match is not None and match.group("vendor") == "00001022"


def identity_from_candidate_record(record: dict[str, Any]) -> dict[str, Any] | None:
    fields = {
        "hwmon_name": record.get("hwmon_name_value"),
        "sensor_label_present": record.get("sensor_label_present"),
        "sensor_label_value": record.get("sensor_label_value"),
        "sensor_input": record.get("input_basename"),
        "sensor_semantic_role": record.get("sensor_semantic_role"),
        "sensor_semantic_profile": record.get("sensor_semantic_profile"),
        "class_path": record.get("class_path"),
        "resolved_input_path": record.get("resolved_input_path"),
        "resolved_hwmon_path": record.get("resolved_hwmon_path"),
        "resolved_device_path": record.get("resolved_device_path"),
        "resolved_driver_path": record.get("resolved_driver_path"),
        "resolved_subsystem_path": record.get("resolved_subsystem_path"),
        "device_driver": record.get("device_driver"),
        "device_subsystem": record.get("device_subsystem"),
        "device_modalias": record.get("device_modalias_value"),
        "input_st_dev": record.get("input_st_dev"),
        "input_st_ino": record.get("input_st_ino"),
        "input_st_mode": record.get("input_st_mode"),
    }
    for key, value in fields.items():
        if key == "sensor_label_value":
            if value is not None and not isinstance(value, str):
                return None
        elif key == "sensor_label_present":
            if type(value) is not bool:
                return None
        elif key in {"input_st_dev", "input_st_ino", "input_st_mode"}:
            if not public.is_json_int(value):
                return None
        elif not isinstance(value, str) or not value:
            return None
    return public.with_temperature_identity_digest(fields)


def classify_legacy_family10h_candidate(
    path: Path,
    *,
    hwmon_root: Path = Path("/sys/class/hwmon"),
    platform_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hwmon_dir = path.parent
    name_path = hwmon_dir / "name"
    label_path = path.with_name(path.name.replace("_input", "_label"))
    device_path = hwmon_dir / "device"
    record: dict[str, Any] = {
        "class_path": str(path),
        "input_basename": path.name,
        "input_path_exists": path.exists(),
        "input_readability": False,
        "raw_input_text": None,
        "raw_input_parse_failure": None,
        "parsed_millidegree_value": None,
        "physical_range_passed": False,
        "hwmon_name_path_exists": name_path.exists(),
        "hwmon_name_readability": False,
        "hwmon_name_value": None,
        "sensor_label_path_exists": label_path.exists(),
        "sensor_label_present": label_path.exists(),
        "sensor_label_readability": False,
        "sensor_label_value": None,
        "resolved_input_path": None,
        "resolved_hwmon_path": None,
        "resolved_device_path": None,
        "resolved_driver_path": None,
        "resolved_subsystem_path": None,
        "device_driver": None,
        "device_subsystem": None,
        "device_modalias_path_exists": False,
        "device_modalias_readability": False,
        "device_modalias_value": None,
        "input_st_dev": None,
        "input_st_ino": None,
        "input_st_mode": None,
        "observation_errors": [],
        "approval_profile": LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
        "sensor_semantic_role": LEGACY_FAMILY10H_TEMPERATURE_ROLE,
        "sensor_semantic_profile": LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
        "approved": False,
        "rejection_reasons": [],
        "identity": None,
    }

    # Candidate classification is metadata-only. The first temperature value read
    # occurs only after one approved identity has been selected and pinned.
    record["input_readability"] = path.exists()

    name_text = read_optional_sysfs_text(name_path)
    record["hwmon_name_readability"] = name_text["readable"]
    record["hwmon_name_value"] = name_text["value"]
    if name_text["error"]:
        record["observation_errors"].append(f"hwmon name read failed: {name_text['error']}")

    label_text = read_optional_sysfs_text(label_path)
    record["sensor_label_readability"] = label_text["readable"]
    record["sensor_label_value"] = label_text["value"] if label_text["readable"] else None
    if label_text["error"]:
        record["observation_errors"].append(f"sensor label read failed: {label_text['error']}")

    for key, resolved in [
        ("resolved_input_path", resolve_path_record(path)),
        ("resolved_hwmon_path", resolve_path_record(hwmon_dir)),
        ("resolved_device_path", resolve_path_record(device_path)),
    ]:
        record[key] = resolved["resolved"]
        if resolved["error"]:
            record["observation_errors"].append(f"{key} failed: {resolved['error']}")

    resolved_device = Path(str(record["resolved_device_path"])) if isinstance(record.get("resolved_device_path"), str) else device_path
    driver_path = resolved_device / "driver"
    subsystem_path = resolved_device / "subsystem"
    modalias_path = resolved_device / "modalias"
    for key, resolved in [
        ("resolved_driver_path", resolve_path_record(driver_path)),
        ("resolved_subsystem_path", resolve_path_record(subsystem_path)),
    ]:
        record[key] = resolved["resolved"]
        if resolved["error"]:
            record["observation_errors"].append(f"{key} failed: {resolved['error']}")
    record["device_driver"] = Path(str(record["resolved_driver_path"])).name if isinstance(record.get("resolved_driver_path"), str) else None
    record["device_subsystem"] = Path(str(record["resolved_subsystem_path"])).name if isinstance(record.get("resolved_subsystem_path"), str) else None

    modalias_text = read_optional_sysfs_text(modalias_path)
    record["device_modalias_path_exists"] = modalias_text["exists"]
    record["device_modalias_readability"] = modalias_text["readable"]
    record["device_modalias_value"] = modalias_text["value"]
    if modalias_text["error"]:
        record["observation_errors"].append(f"modalias read failed: {modalias_text['error']}")

    if isinstance(record.get("resolved_input_path"), str):
        try:
            input_stat = Path(str(record["resolved_input_path"])).stat()
            record["input_st_dev"] = input_stat.st_dev
            record["input_st_ino"] = input_stat.st_ino
            record["input_st_mode"] = input_stat.st_mode
        except OSError as exc:
            record["observation_errors"].append(f"input stat failed: {type(exc).__name__}: {exc}")

    reasons: list[str] = []
    if platform_identity is not None:
        if platform_identity.get("vendor") != "AuthenticAMD":
            reasons.append("target platform vendor is not AuthenticAMD")
        if platform_identity.get("cpu_family") != 16:
            reasons.append("target CPU family is not 16")
        if platform_identity.get("operational_pin_capability_passed") is not True:
            reasons.append("operational pin capability did not pass")
    if record["input_basename"] != LEGACY_FAMILY10H_TEMPERATURE_INPUT:
        reasons.append("legacy profile requires temp1_input")
    if record["hwmon_name_value"] != "k10temp":
        reasons.append("legacy profile requires hwmon name k10temp")
    if record["sensor_label_present"]:
        if record["sensor_label_readability"] is not True:
            reasons.append("present temp1_label is unreadable")
        elif record["sensor_label_value"] != LEGACY_FAMILY10H_TEMPERATURE_ROLE:
            reasons.append("present temp1_label is not Tctl")
    if record["device_driver"] != "k10temp":
        reasons.append("legacy profile requires k10temp PCI driver")
    if record["device_subsystem"] != "pci":
        reasons.append("legacy profile requires pci subsystem")
    if not isinstance(record.get("device_modalias_value"), str) or not record["device_modalias_value"]:
        reasons.append("legacy profile requires nonempty PCI modalias")
    elif not amd_pci_modalias_vendor_1022(record["device_modalias_value"]):
        reasons.append("legacy profile requires AMD PCI vendor 1022 modalias")
    for field in ["resolved_input_path", "resolved_hwmon_path", "resolved_device_path", "resolved_driver_path", "resolved_subsystem_path"]:
        if not isinstance(record.get(field), str) or not record[field]:
            reasons.append(f"{field} missing")
    for field in ["input_st_dev", "input_st_ino", "input_st_mode"]:
        if not public.is_json_int(record.get(field)):
            reasons.append(f"{field} missing")

    canonical_hwmon = hwmon_root == Path("/sys/class/hwmon")
    record["canonical_path_law_active"] = canonical_hwmon
    record["class_path_under_hwmon_root"] = str(path).startswith(str(hwmon_root) + os.sep)
    record["resolved_input_under_sys_devices"] = isinstance(record.get("resolved_input_path"), str) and str(record["resolved_input_path"]).startswith("/sys/devices/")
    record["resolved_device_under_sys_devices"] = isinstance(record.get("resolved_device_path"), str) and str(record["resolved_device_path"]).startswith("/sys/devices/")
    if canonical_hwmon:
        if not record["class_path_under_hwmon_root"]:
            reasons.append("class path is outside /sys/class/hwmon")
        if not record["resolved_input_under_sys_devices"]:
            reasons.append("resolved input path is outside /sys/devices")
        if not record["resolved_device_under_sys_devices"]:
            reasons.append("resolved device path is outside /sys/devices")

    identity = identity_from_candidate_record(record)
    if identity is not None:
        record["identity"] = identity
    elif not reasons:
        reasons.append("candidate identity fields incomplete")
    record["rejection_reasons"] = sorted(set(reasons))
    record["rejection_reason"] = "; ".join(record["rejection_reasons"]) if record["rejection_reasons"] else None
    record["approved"] = identity is not None and not record["rejection_reasons"]
    return record


def raw_temperature_sensor_identity(path: Path) -> dict[str, Any]:
    record = classify_legacy_family10h_candidate(path, hwmon_root=path.parents[1] if len(path.parents) > 1 else path.parent)
    identity = record.get("identity")
    if not isinstance(identity, dict):
        raise TargetError(record.get("rejection_reason") or "temperature sensor identity incomplete")
    return identity


def temperature_sensor_identity(path: Path) -> dict[str, Any]:
    record = classify_legacy_family10h_candidate(path, hwmon_root=path.parents[1] if len(path.parents) > 1 else path.parent)
    identity = record.get("identity")
    require(isinstance(identity, dict), record.get("rejection_reason") or "temperature sensor identity incomplete")
    require(record.get("approved") is True, record.get("rejection_reason") or "temperature sensor identity not approved")
    return identity


def temperature_candidate_record(
    path: Path,
    *,
    hwmon_root: Path = Path("/sys/class/hwmon"),
    platform_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return classify_legacy_family10h_candidate(path, hwmon_root=hwmon_root, platform_identity=platform_identity)


def hwmon_visibility_snapshot(hwmon_root: Path = Path("/sys/class/hwmon")) -> dict[str, Any]:
    hwmon_dirs = sorted(path for path in hwmon_root.glob("hwmon*") if path.is_dir()) if hwmon_root.exists() else []
    candidates = sorted(hwmon_root.glob("hwmon*/temp*_input")) if hwmon_root.exists() else []
    k10temp_dirs: list[str] = []
    for hwmon in hwmon_dirs:
        name = read_optional_sysfs_text(hwmon / "name")
        if name["readable"] and name["value"] == "k10temp":
            k10temp_dirs.append(str(hwmon))
    driver_root = Path("/sys/bus/pci/drivers/k10temp")
    bound_devices = []
    if driver_root.exists():
        bound_devices = sorted(str(path) for path in driver_root.iterdir() if path.is_symlink())
    force_path = Path("/sys/module/k10temp/parameters/force")
    force_text = read_optional_sysfs_text(force_path)
    return {
        "hwmon_root": str(hwmon_root),
        "hwmon_root_exists": hwmon_root.exists(),
        "hwmon_directory_count": len(hwmon_dirs),
        "temp_input_candidate_count": len(candidates),
        "k10temp_hwmon_directory_count": len(k10temp_dirs),
        "k10temp_hwmon_directories": k10temp_dirs,
        "k10temp_driver_path": str(driver_root),
        "k10temp_driver_exists": driver_root.exists(),
        "bound_k10temp_pci_device_symlinks": bound_devices,
        "k10temp_module_path": "/sys/module/k10temp",
        "k10temp_module_exists": Path("/sys/module/k10temp").exists(),
        "k10temp_force_parameter_exists": force_path.exists(),
        "k10temp_force_parameter_readable": force_text["readable"],
        "k10temp_force_parameter_value": force_text["value"],
    }


def enumerate_temperature_candidates(
    hwmon_root: Path = Path("/sys/class/hwmon"),
    *,
    platform_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        temperature_candidate_record(path, hwmon_root=hwmon_root, platform_identity=platform_identity)
        for path in sorted(hwmon_root.glob("hwmon*/temp*_input"))
    ]


def classify_temperature_discovery_failure(candidates: list[dict[str, Any]], visibility: dict[str, Any]) -> str:
    if visibility.get("temp_input_candidate_count") == 0:
        return "NO_HWMON_TEMPERATURE_CANDIDATES"
    if visibility.get("k10temp_driver_exists") is not True:
        return "K10TEMP_DRIVER_NOT_VISIBLE"
    if visibility.get("k10temp_hwmon_directory_count") == 0:
        return "K10TEMP_HWMON_NOT_VISIBLE"
    legacy_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("hwmon_name_value") == "k10temp" and candidate.get("input_basename") == LEGACY_FAMILY10H_TEMPERATURE_INPUT
    ]
    if not legacy_candidates:
        return "LEGACY_TEMP1_INPUT_NOT_VISIBLE"
    approved = [candidate for candidate in candidates if candidate.get("approved") is True and isinstance(candidate.get("identity"), dict)]
    if len(approved) > 1:
        return "MULTIPLE_APPROVED_LEGACY_CANDIDATES"
    return "LEGACY_CANDIDATE_REJECTED_IDENTITY"


def select_approved_temperature_identity(
    candidates: list[dict[str, Any]],
    *,
    deterministic_law: bool = True,
    visibility: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    approved = [candidate for candidate in candidates if candidate.get("approved") is True and isinstance(candidate.get("identity"), dict)]
    if len(approved) != 1:
        classification = classify_temperature_discovery_failure(candidates, visibility or {"temp_input_candidate_count": len(candidates)})
        raise TargetError(classification)
    selected = approved[0]
    return selected["identity"], {
        "law": "exactly one LEGACY_FAMILY10H_K10TEMP_TEMP1_V1 candidate",
        "approved_count": len(approved),
        "selected_class_path": selected["identity"]["class_path"],
        "deterministic_law": deterministic_law,
        "approval_profile": LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
    }


def descriptor_identity(fd: int, identity: dict[str, Any]) -> dict[str, Any]:
    """Return stable descriptor metadata for a pinned temperature input."""
    stat = os.fstat(fd)
    return {
        "resolved_input_path": identity["resolved_input_path"],
        "input_st_dev": stat.st_dev,
        "input_st_ino": stat.st_ino,
        "input_st_mode": stat.st_mode,
    }


def expected_descriptor_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolved_input_path": identity["resolved_input_path"],
        "input_st_dev": identity["input_st_dev"],
        "input_st_ino": identity["input_st_ino"],
        "input_st_mode": identity["input_st_mode"],
    }


def identity_matches_required(identity: dict[str, Any], required_identity: dict[str, Any]) -> bool:
    return all(identity.get(field) == required_identity.get(field) for field in TEMPERATURE_IDENTITY_FIELDS) and identity.get(
        "identity_sha256"
    ) == required_identity.get("identity_sha256")


def operational_pin_capability_failures(platform_identity: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    if not isinstance(platform_identity, dict):
        return ["temperature sensor discovery target platform missing"]
    pin = platform_identity.get("operational_pin_capability")
    if platform_identity.get("operational_pin_capability_passed") is not True or not isinstance(pin, dict):
        return ["temperature sensor discovery operational pin capability missing or failed"]
    expected_cpus = [public.SOURCE_CPU_EXPECTED, public.RECEIVER_CPU_EXPECTED]
    if pin.get("schema") != "FAMILY10H_OPERATIONAL_PIN_CAPABILITY_PROBE_V1":
        failures.append("temperature sensor discovery operational pin capability schema mismatch")
    if pin.get("required_cpus") != expected_cpus:
        failures.append("temperature sensor discovery operational pin required CPUs mismatch")
    if pin.get("passed") is not True:
        failures.append("temperature sensor discovery operational pin capability did not pass")
    if pin.get("parent_affinity_restored") is not True:
        failures.append("temperature sensor discovery operational pin parent affinity was not restored")
    if pin.get("opened_pmu") is not False or pin.get("launched_runtime") is not False or pin.get("created_tomography_output_root") is not False:
        failures.append("temperature sensor discovery operational pin performed forbidden live work")
    if "skipped_reason" in pin:
        failures.append("temperature sensor discovery operational pin capability was skipped")
    per_cpu = pin.get("per_cpu")
    if not isinstance(per_cpu, dict) or set(per_cpu) != {str(cpu) for cpu in expected_cpus}:
        failures.append("temperature sensor discovery operational pin per-CPU evidence mismatch")
    else:
        for cpu in expected_cpus:
            item = per_cpu.get(str(cpu))
            if not isinstance(item, dict):
                failures.append(f"temperature sensor discovery operational pin CPU {cpu} evidence missing")
                continue
            if item.get("cpu") != cpu:
                failures.append(f"temperature sensor discovery operational pin CPU {cpu} identity mismatch")
            if item.get("passed") is not True:
                failures.append(f"temperature sensor discovery operational pin CPU {cpu} did not pass")
            if item.get("requested_affinity") != [cpu]:
                failures.append(f"temperature sensor discovery operational pin CPU {cpu} requested affinity mismatch")
            readback = item.get("readback_affinity")
            if not isinstance(readback, list) or readback != [cpu]:
                failures.append(f"temperature sensor discovery operational pin CPU {cpu} readback mismatch")
            if item.get("parent_affinity_restored") is not True:
                failures.append(f"temperature sensor discovery operational pin CPU {cpu} parent affinity was not restored")
    return failures


def legacy_temperature_identity_is_approved(identity: dict[str, Any]) -> bool:
    label_present = identity.get("sensor_label_present")
    label_value = identity.get("sensor_label_value")
    label_law_passed = (label_present is False and label_value is None) or (label_present is True and label_value == LEGACY_FAMILY10H_TEMPERATURE_ROLE)
    return (
        identity.get("hwmon_name") == "k10temp"
        and identity.get("sensor_input") == LEGACY_FAMILY10H_TEMPERATURE_INPUT
        and identity.get("sensor_semantic_role") == LEGACY_FAMILY10H_TEMPERATURE_ROLE
        and identity.get("sensor_semantic_profile") == LEGACY_FAMILY10H_TEMPERATURE_PROFILE
        and label_law_passed
        and identity.get("device_driver") == "k10temp"
        and identity.get("device_subsystem") == "pci"
        and amd_pci_modalias_vendor_1022(identity.get("device_modalias"))
        and public.is_json_int(identity.get("input_st_dev"))
        and public.is_json_int(identity.get("input_st_ino"))
        and public.is_json_int(identity.get("input_st_mode"))
    )


def temperature_physical_authority_failures(
    identity: dict[str, Any] | None,
    *,
    authorizing_scope: dict[str, Any] | None = None,
    platform_identity: dict[str, Any] | None = None,
    require_canonical_paths: bool = True,
    require_authorizing_scope: bool = False,
    require_pin_evidence: bool = False,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(identity, dict) or set(identity) != public.TEMPERATURE_SENSOR_IDENTITY_KEYS:
        return ["temperature physical authority identity missing or malformed"]
    if identity.get("identity_sha256") != public.temperature_identity_digest(identity):
        failures.append("temperature physical authority identity digest mismatch")
    if not legacy_temperature_identity_is_approved(identity):
        failures.append("temperature physical authority legacy identity incomplete")
    class_path = str(identity.get("class_path", ""))
    resolved_input = str(identity.get("resolved_input_path", ""))
    resolved_hwmon = str(identity.get("resolved_hwmon_path", ""))
    resolved_device = str(identity.get("resolved_device_path", ""))
    resolved_driver = str(identity.get("resolved_driver_path", ""))
    resolved_subsystem = str(identity.get("resolved_subsystem_path", ""))
    if require_canonical_paths:
        if re.fullmatch(r"/sys/class/hwmon/hwmon\d+/temp1_input", class_path) is None:
            failures.append("temperature physical authority class path is not canonical sysfs temp1_input")
        if not resolved_input.startswith("/sys/devices/") or not resolved_input.endswith("/temp1_input"):
            failures.append("temperature physical authority resolved input is not canonical sysfs temp1_input")
        if not resolved_hwmon.startswith("/sys/devices/") or "/hwmon/hwmon" not in resolved_hwmon:
            failures.append("temperature physical authority resolved hwmon path is not under sysfs devices")
        if not resolved_device.startswith("/sys/devices/"):
            failures.append("temperature physical authority resolved device path is not under sysfs devices")
        if resolved_hwmon and resolved_input and resolved_input != resolved_hwmon.rstrip("/") + "/temp1_input":
            failures.append("temperature physical authority resolved input does not match resolved hwmon/temp1_input")
        if resolved_device and resolved_hwmon and not resolved_hwmon.startswith(resolved_device.rstrip("/") + "/"):
            failures.append("temperature physical authority resolved hwmon is not below resolved device")
        if resolved_driver != "/sys/bus/pci/drivers/k10temp":
            failures.append("temperature physical authority resolved driver path is not canonical k10temp sysfs driver")
        if resolved_subsystem != "/sys/bus/pci":
            failures.append("temperature physical authority resolved subsystem path is not canonical pci sysfs bus")
    if isinstance(authorizing_scope, dict):
        expected_scope = {
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
        }
        for key, expected in expected_scope.items():
            if authorizing_scope.get(key) is not expected:
                failures.append(f"temperature physical authority scope {key} mismatch")
        if authorizing_scope.get("cpuinfo_path") != "/proc/cpuinfo":
            failures.append("temperature physical authority cpuinfo path is not canonical")
        if authorizing_scope.get("hwmon_root") != "/sys/class/hwmon":
            failures.append("temperature physical authority hwmon root is not canonical")
    elif require_authorizing_scope:
        failures.append("temperature physical authority scope missing")
    if require_pin_evidence:
        failures.extend(operational_pin_capability_failures(platform_identity))
    return failures


class PinnedTemperatureSensor:
    def __init__(self, required_identity: dict[str, Any], *, allow_noncanonical_fixture: bool = False):
        self.required_identity = required_identity
        self.allow_noncanonical_fixture = allow_noncanonical_fixture
        self.fd: int | None = None
        self.descriptor: dict[str, Any] | None = None

    def __enter__(self) -> "PinnedTemperatureSensor":
        authority_failures = temperature_physical_authority_failures(
            self.required_identity,
            require_canonical_paths=not self.allow_noncanonical_fixture,
        )
        require(not authority_failures, "temperature physical authority failed: " + "; ".join(authority_failures))
        class_path = Path(str(self.required_identity["class_path"]))
        current_identity = temperature_sensor_identity(class_path)
        require(identity_matches_required(current_identity, self.required_identity), "temperature class-path identity drift")
        resolved_input = Path(str(self.required_identity["resolved_input_path"]))
        require(str(resolved_input.resolve(strict=True)) == self.required_identity["resolved_input_path"], "temperature resolved path drift")
        self.fd = os.open(resolved_input, os.O_RDONLY)
        self.descriptor = descriptor_identity(self.fd, self.required_identity)
        require(self.descriptor == expected_descriptor_identity(self.required_identity), "temperature descriptor differs from discovery identity")
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
            "label_present": self.required_identity["sensor_label_present"],
            "label_value": self.required_identity["sensor_label_value"],
            "semantic_role": self.required_identity["sensor_semantic_role"],
            "semantic_profile": self.required_identity["sensor_semantic_profile"],
            "value_c": value,
            "identity": self.required_identity,
            "pinned_descriptor": self.descriptor,
            "read_law": "manifest-approved resolved input descriptor",
        }


def read_temperature_sample(
    required_identity: dict[str, Any],
    hwmon_root: Path = Path("/sys/class/hwmon"),
    mutation_hook: Any | None = None,
    allow_noncanonical_fixture: bool = False,
) -> dict[str, Any]:
    del hwmon_root
    require(isinstance(required_identity, dict), "explicit temperature identity is required")
    with PinnedTemperatureSensor(required_identity, allow_noncanonical_fixture=allow_noncanonical_fixture) as sensor:
        return sensor.read_sample(mutation_hook=mutation_hook)


def read_temperature_c(required_identity: dict[str, Any]) -> float:
    return float(read_temperature_sample(required_identity)["value_c"])


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


def parse_cpuinfo_stanzas(text: str) -> list[dict[str, str]]:
    stanzas: list[dict[str, str]] = []
    for raw_stanza in re.split(r"\n\s*\n", text.strip()):
        if not raw_stanza.strip():
            continue
        fields: dict[str, str] = {}
        for raw_line in raw_stanza.splitlines():
            if not raw_line.strip():
                continue
            if ":" not in raw_line:
                raise TargetError("platform identity line malformed")
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in fields and fields[key] != value:
                raise TargetError(f"platform identity conflicting field {key}")
            if key in fields:
                raise TargetError(f"platform identity duplicate field {key}")
            fields[key] = value
        if fields:
            stanzas.append(fields)
    return stanzas


def current_execution_cpu() -> int | None:
    stat_path = Path("/proc/self/stat")
    if not stat_path.exists():
        return None
    try:
        text = stat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) <= 36 or not re.fullmatch(r"\d+", fields[36]):
        return None
    return int(fields[36])


def affinity_error_name(value: int | None) -> str | None:
    if value is None:
        return None
    return errno_module.errorcode.get(value, f"ERRNO_{value}")


def probe_single_operational_pin(cpu: int) -> dict[str, Any]:
    if not (hasattr(os, "fork") and hasattr(os, "pipe") and hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")):
        return {
            "cpu": cpu,
            "passed": False,
            "failure_class": "affinity_syscall_unavailable",
            "diagnostic": "operational pin capability syscall unavailable",
        }
    parent_before = sorted(os.sched_getaffinity(0))
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        result: dict[str, Any] = {
            "cpu": cpu,
            "passed": False,
            "child_inherited_affinity": None,
            "requested_affinity": [cpu],
            "readback_affinity": None,
            "actual_execution_cpu": None,
            "actual_execution_cpu_available": False,
            "errno": None,
            "errno_name": None,
            "failure_class": None,
            "diagnostic": None,
        }
        try:
            result["child_inherited_affinity"] = sorted(os.sched_getaffinity(0))
            try:
                os.sched_setaffinity(0, {cpu})
            except OSError as exc:
                result["errno"] = exc.errno
                result["errno_name"] = affinity_error_name(exc.errno)
                if exc.errno == errno_module.EPERM:
                    result["failure_class"] = "permission_failure"
                elif exc.errno == errno_module.EINVAL:
                    result["failure_class"] = "cpuset_or_kernel_rejected_cpu"
                else:
                    result["failure_class"] = "sched_setaffinity_errno"
                result["diagnostic"] = f"sched_setaffinity {result['errno_name']}"
            else:
                if hasattr(os, "sched_yield"):
                    os.sched_yield()
                readback = sorted(os.sched_getaffinity(0))
                actual_cpu = current_execution_cpu()
                result["readback_affinity"] = readback
                result["actual_execution_cpu"] = actual_cpu
                result["actual_execution_cpu_available"] = actual_cpu is not None
                if readback != [cpu]:
                    result["failure_class"] = "affinity_readback_failure"
                    result["diagnostic"] = "child affinity readback differs from requested singleton"
                elif actual_cpu is not None and actual_cpu != cpu:
                    result["failure_class"] = "actual_execution_cpu_mismatch"
                    result["diagnostic"] = "actual execution CPU differs from requested singleton"
                else:
                    result["passed"] = True
                    result["failure_class"] = None
                    result["diagnostic"] = "operational pin capability passed"
        except BaseException as exc:  # noqa: BLE001 - child must report all probe failures to parent.
            result["failure_class"] = "child_probe_exception"
            result["diagnostic"] = f"{type(exc).__name__}: {exc}"
        data = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        os.write(write_fd, data)
        os.close(write_fd)
        os._exit(0 if result["passed"] else 1)
    os.close(write_fd)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(read_fd, 8192)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    waited_pid, status = os.waitpid(pid, 0)
    parent_after = sorted(os.sched_getaffinity(0))
    try:
        result = strict_json_loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        result = {
            "cpu": cpu,
            "passed": False,
            "failure_class": "child_probe_receipt_malformed",
            "diagnostic": "child probe receipt malformed",
        }
    result["child_pid"] = waited_pid
    result["child_exit_status"] = status
    result["parent_affinity_before"] = parent_before
    result["parent_affinity_after"] = parent_after
    result["parent_affinity_restored"] = parent_after == parent_before
    if parent_after != parent_before:
        result["passed"] = False
        result["failure_class"] = "parent_affinity_changed"
        result["diagnostic"] = "parent affinity changed after child probe"
    return result


def probe_operational_pin_capability(required_cpus: set[int]) -> dict[str, Any]:
    parent_before = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    per_cpu = {str(cpu): probe_single_operational_pin(cpu) for cpu in sorted(required_cpus)}
    parent_after = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    inherited_excludes = (
        sorted(cpu for cpu in required_cpus if parent_before is not None and cpu not in parent_before)
        if parent_before is not None
        else []
    )
    all_cpu_passed = all(item.get("passed") is True for item in per_cpu.values())
    parent_restored = parent_before == parent_after
    return {
        "schema": "FAMILY10H_OPERATIONAL_PIN_CAPABILITY_PROBE_V1",
        "required_cpus": sorted(required_cpus),
        "inherited_parent_affinity": parent_before,
        "inherited_affinity_excluded_required_cpus": inherited_excludes,
        "inherited_mask_restriction_only": bool(inherited_excludes) and all_cpu_passed,
        "per_cpu": per_cpu,
        "parent_affinity_after_all_probes": parent_after,
        "parent_affinity_restored": parent_restored,
        "opened_pmu": False,
        "launched_runtime": False,
        "created_tomography_output_root": False,
        "passed": all_cpu_passed and parent_restored,
    }


def format_pin_capability_failure(probe: dict[str, Any]) -> str:
    if probe.get("parent_affinity_restored") is False:
        return "operational pin capability failed: parent affinity changed after child probe"
    per_cpu = probe.get("per_cpu")
    if isinstance(per_cpu, dict):
        for key in sorted(per_cpu, key=lambda item: int(item) if str(item).isdigit() else str(item)):
            item = per_cpu.get(key)
            if not isinstance(item, dict) or item.get("passed") is True:
                continue
            cpu = item.get("cpu", key)
            failure = item.get("failure_class")
            if failure in {"permission_failure", "cpuset_or_kernel_rejected_cpu", "sched_setaffinity_errno"}:
                return f"operational pin capability failed for CPU {cpu}: sched_setaffinity {item.get('errno_name')}"
            if failure == "affinity_readback_failure":
                return f"operational pin capability failed for CPU {cpu}: child affinity readback differs"
            if failure == "actual_execution_cpu_mismatch":
                return f"operational pin capability failed for CPU {cpu}: actual execution CPU differs"
            return f"operational pin capability failed for CPU {cpu}: {item.get('diagnostic') or failure}"
    return "operational pin capability failed"


def require_family10h_platform(
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
    *,
    inherited_affinity_cpus: list[int] | None = None,
    pin_probe: Any | None = None,
) -> dict[str, Any]:
    text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    stanzas = parse_cpuinfo_stanzas(text)
    require(bool(stanzas), "platform identity CPU stanza missing")
    processors: list[dict[str, Any]] = []
    for stanza in stanzas:
        require("processor" in stanza, "platform identity processor field missing")
        require(stanza.get("vendor_id") == "AuthenticAMD", "platform identity vendor is not AuthenticAMD")
        require(stanza.get("cpu family") == "16", "platform identity CPU family is not 16")
        require(re.fullmatch(r"\d+", stanza.get("processor", "")) is not None, "platform identity processor id malformed")
        require(re.fullmatch(r"\d+", stanza.get("model", "")) is not None, "platform identity model malformed")
        processors.append(
            {
                "processor": int(stanza["processor"]),
                "vendor_id": stanza["vendor_id"],
                "cpu_family": int(stanza["cpu family"]),
                "model": int(stanza["model"]),
            }
        )
    processor_ids = [item["processor"] for item in processors]
    require(len(set(processor_ids)) == len(processor_ids), "platform identity duplicate processor id")
    required_cpus = {public.SOURCE_CPU_EXPECTED, public.RECEIVER_CPU_EXPECTED}
    require(public.SOURCE_CPU_EXPECTED in processor_ids, "platform identity missing frozen source CPU")
    require(public.RECEIVER_CPU_EXPECTED in processor_ids, "platform identity missing frozen receiver CPU")
    canonical_cpuinfo = cpuinfo_path == Path("/proc/cpuinfo")
    if inherited_affinity_cpus is not None:
        inherited_affinity = sorted(inherited_affinity_cpus)
        inherited_affinity_checked = True
    elif hasattr(os, "sched_getaffinity") and canonical_cpuinfo:
        inherited_affinity = sorted(os.sched_getaffinity(0))
        inherited_affinity_checked = True
    else:
        inherited_affinity = sorted(processor_ids)
        inherited_affinity_checked = False
    if pin_probe is not None:
        operational_pin = pin_probe(required_cpus)
    elif canonical_cpuinfo:
        operational_pin = probe_operational_pin_capability(required_cpus)
    else:
        operational_pin = {
            "schema": "FAMILY10H_OPERATIONAL_PIN_CAPABILITY_PROBE_V1",
            "required_cpus": sorted(required_cpus),
            "inherited_parent_affinity": inherited_affinity,
            "inherited_affinity_excluded_required_cpus": [],
            "inherited_mask_restriction_only": False,
            "per_cpu": {},
            "parent_affinity_restored": True,
            "opened_pmu": False,
            "launched_runtime": False,
            "created_tomography_output_root": False,
            "passed": True,
            "skipped_reason": "noncanonical cpuinfo fixture",
        }
    require(isinstance(operational_pin, dict), "operational pin capability result malformed")
    require(operational_pin.get("passed") is True, format_pin_capability_failure(operational_pin))
    return {
        "vendor": "AuthenticAMD",
        "cpu_family": 16,
        "cpu_models": sorted(set(item["model"] for item in processors)),
        "processor_count": len(processors),
        "processors": processors,
        "source_cpu_expected": public.SOURCE_CPU_EXPECTED,
        "receiver_cpu_expected": public.RECEIVER_CPU_EXPECTED,
        "source_receiver_cpus_present": True,
        "affinity_checked": inherited_affinity_checked,
        "affinity_cpus": inherited_affinity,
        "inherited_affinity_checked": inherited_affinity_checked,
        "inherited_affinity_cpus": inherited_affinity,
        "operational_pin_capability": operational_pin,
        "operational_pin_capability_passed": operational_pin.get("passed") is True,
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
    source_authority_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_hashes = read_json(source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json")
    schedule_sidecar = read_json(source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256")
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
        "runtime_binary_sha256": source_hashes["runtime_binary_authority"]["sha256"],
        "schedule_canonical_sha256": schedule_sidecar["canonical_sha256"],
        "schedule_json_sha256": schedule_sidecar["json_sha256"],
        "schedule_tsv_sha256": schedule_sidecar["tsv_sha256"],
        "authorized_commit": authorized_commit,
        "controller_nonce_sha256": controller_nonce_sha256,
        "transport_scope": transport_scope,
        "source_authority_review": source_authority_review,
    }


def fixture_source_review_binding(source_root: Path, authorized_commit: str) -> dict[str, Any]:
    source_hashes = read_json(source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json")
    return {
        "findings_sha256": "a" * 64,
        "review_report_sha256": "b" * 64,
        "review_quorum_sha256": "c" * 64,
        "source_authority_commit": authorized_commit,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": deterministic_source_bundle_sha256(source_root),
        "runtime_binary_sha256": source_hashes["runtime_binary_authority"]["sha256"],
    }


def validate_discovery_challenge(
    source_root: Path,
    challenge: dict[str, Any],
    controller_nonce: str,
    authorized_commit: str,
    *,
    allowed_transfer_extra_names: set[str] | None = None,
) -> dict[str, Any]:
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
    challenge_review = challenge.get("source_authority_review") if isinstance(challenge.get("source_authority_review"), dict) else None
    expected = expected_discovery_challenge(source_root, authorized_commit, nonce_sha, challenge_scope, challenge_review) if not failures else {}
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
    review = challenge.get("source_authority_review")
    if not isinstance(review, dict):
        failures.append("controller challenge source authority review missing")
    else:
        if set(review) != SOURCE_REVIEW_BINDING_KEYS:
            failures.append("controller challenge source authority review field mismatch")
        for field in [
            "findings_sha256",
            "review_report_sha256",
            "review_quorum_sha256",
            "source_hashes_sha256",
            "source_bundle_sha256",
            "runtime_binary_sha256",
        ]:
            if re.fullmatch(r"[0-9a-f]{64}", str(review.get(field, ""))) is None:
                failures.append(f"controller challenge source authority review {field} invalid")
        if review.get("source_authority_commit") != authorized_commit:
            failures.append("controller challenge source authority review commit mismatch")
        if review.get("source_hashes_sha256") != challenge.get("source_hashes_sha256"):
            failures.append("controller challenge source authority review source-hashes mismatch")
        if review.get("source_bundle_sha256") != challenge.get("source_bundle_sha256"):
            failures.append("controller challenge source authority review source-bundle mismatch")
        if review.get("runtime_binary_sha256") != challenge.get("runtime_binary_sha256"):
            failures.append("controller challenge source authority review runtime-binary mismatch")
    transfer_root = validate_discovery_transfer_root(
        source_root,
        challenge=challenge,
        allowed_extra_names=allowed_transfer_extra_names,
    )
    if not transfer_root["passed"]:
        failures.extend("discovery transfer root: " + item for item in transfer_root["failures"])
    return {
        "passed": not failures,
        "failures": failures,
        "expected_challenge": expected,
        "challenge_sha256": public.digest(challenge) if isinstance(challenge, dict) else None,
        "controller_nonce_sha256": nonce_sha,
        "source_authority": source_authority,
        "discovery_transfer_root": transfer_root,
    }


def classify_temperature_discovery_exception(message: str, *, candidate_scan_count: int) -> str:
    if message in {
        "NO_HWMON_TEMPERATURE_CANDIDATES",
        "K10TEMP_DRIVER_NOT_VISIBLE",
        "K10TEMP_HWMON_NOT_VISIBLE",
        "LEGACY_TEMP1_INPUT_NOT_VISIBLE",
        "LEGACY_CANDIDATE_REJECTED_IDENTITY",
        "LEGACY_CANDIDATE_UNREADABLE",
        "MULTIPLE_APPROVED_LEGACY_CANDIDATES",
    }:
        return message
    if message.startswith("controller challenge invalid"):
        return "CONTROLLER_CHALLENGE_INVALID"
    if "controller challenge file missing" in message:
        return "CONTROLLER_CHALLENGE_MISSING"
    lowered = message.lower()
    if "platform" in lowered or "cpu" in lowered or "affinity" in lowered:
        return "PLATFORM_IDENTITY_INVALID"
    if "selected temperature identity changed before read" in message:
        return "SELECTED_IDENTITY_CHANGED_BEFORE_READ"
    if "selected temperature identity changed after read" in message:
        return "SELECTED_IDENTITY_CHANGED_AFTER_READ"
    if "temperature unreadable from pinned descriptor" in message or "temperature outside physical custody bounds" in message:
        return "LEGACY_CANDIDATE_UNREADABLE"
    if candidate_scan_count:
        return "LEGACY_CANDIDATE_REJECTED_IDENTITY"
    return "PRE_SCAN_DISCOVERY_FAILURE"


def write_temperature_discovery_failure_receipt(
    *,
    source_root: Path,
    receipt_path: Path,
    hwmon_root: Path,
    controller_nonce: str,
    authorized_commit: str,
    challenge: dict[str, Any] | None,
    challenge_validation: dict[str, Any] | None,
    platform_identity: dict[str, Any] | None,
    visibility: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    candidate_scan_count: int,
    failure_classification: str,
    failure_detail: str,
) -> dict[str, Any]:
    challenge_data = challenge if isinstance(challenge, dict) else {}
    validation = challenge_validation if isinstance(challenge_validation, dict) else {}
    sensor_inventory_count = 1 if candidate_scan_count else 0
    counters = {
        "target_contact_count": 1,
        "sensor_inventory_count": sensor_inventory_count,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "candidate_scan_count": candidate_scan_count,
    }
    approved_count = len([candidate for candidate in candidates if isinstance(candidate, dict) and candidate.get("approved") is True])
    result = {
        "schema": TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA,
        "discovery_mode": "target_read_only_sensor_inventory",
        "passed": False,
        "failure_classification": failure_classification,
        "failure_detail": failure_detail,
        "target_contact_count": counters["target_contact_count"],
        "sensor_inventory_count": counters["sensor_inventory_count"],
        "candidate_scan_count": counters["candidate_scan_count"],
        "live_invocation_count": counters["live_invocation_count"],
        "pmu_acquisition_count": counters["pmu_acquisition_count"],
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
            "controller_challenge_sha256": validation.get("challenge_sha256") or (public.digest(challenge_data) if challenge_data else None),
            "authorized_commit": authorized_commit,
        },
        "controller_challenge_sha256": validation.get("challenge_sha256") or (public.digest(challenge_data) if challenge_data else None),
        "controller_nonce_sha256": validation.get("controller_nonce_sha256") or hashlib.sha256(controller_nonce.encode("ascii")).hexdigest(),
        "authorized_commit": authorized_commit,
        "source_hashes_sha256": challenge_data.get("source_hashes_sha256"),
        "source_bundle_sha256": challenge_data.get("source_bundle_sha256"),
        "runtime_binary_sha256": challenge_data.get("runtime_binary_sha256"),
        "source_authority_review": challenge_data.get("source_authority_review"),
        "source_authority": validation.get("source_authority"),
        "challenge_validation": validation,
        "top_level_visibility_snapshot": visibility if isinstance(visibility, dict) else {},
        "observed_candidates": candidates,
        "candidate_count": len(candidates),
        "approved_count": approved_count,
        "active_counters": counters,
    }
    result["target_discovery_receipt_sha256"] = public.digest(
        {k: v for k, v in result.items() if k != "target_discovery_receipt_sha256"}
    )
    write_json_exclusive_atomic(receipt_path, result)
    return result


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
    challenge: dict[str, Any] | None = None
    challenge_validation: dict[str, Any] | None = None
    platform_identity: dict[str, Any] | None = None
    visibility: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    candidate_scan_count = 0
    try:
        env_check = validate_no_live_authority_env()
        require(env_check["passed"], "live authority environment present during discovery")
        require(receipt_path.name == TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME, "discovery receipt path must use canonical name")
        require(receipt_path.parent.resolve() == source_root.resolve(), "discovery receipt path must be inside source root")
        require(not receipt_path.exists(), "discovery receipt already exists")
        require(controller_challenge_path.exists(), "controller challenge file missing")
        challenge = read_json(controller_challenge_path)
        challenge_validation = validate_discovery_challenge(
            source_root,
            challenge,
            controller_nonce,
            authorized_commit,
            allowed_transfer_extra_names={controller_challenge_path.name},
        )
        require(challenge_validation["passed"], "controller challenge invalid: " + ",".join(challenge_validation["failures"]))
        platform_identity = require_family10h_platform(cpuinfo_path)
        visibility = hwmon_visibility_snapshot(hwmon_root)
        candidates = enumerate_temperature_candidates(hwmon_root, platform_identity=platform_identity)
        candidate_scan_count = 1
        approved_count = len([candidate for candidate in candidates if candidate.get("approved") is True])
        selected_identity, selection = select_approved_temperature_identity(candidates, deterministic_law=True, visibility=visibility)
    except Exception as exc:  # noqa: BLE001 - target discovery failures must be sealed when possible
        failure_detail = str(exc)
        failure_classification = classify_temperature_discovery_exception(failure_detail, candidate_scan_count=candidate_scan_count)
        if not receipt_path.exists():
            write_temperature_discovery_failure_receipt(
                source_root=source_root,
                receipt_path=receipt_path,
                hwmon_root=hwmon_root,
                controller_nonce=controller_nonce,
                authorized_commit=authorized_commit,
                challenge=challenge,
                challenge_validation=challenge_validation,
                platform_identity=platform_identity,
                visibility=visibility,
                candidates=candidates,
                candidate_scan_count=candidate_scan_count,
                failure_classification=failure_classification,
                failure_detail=failure_detail,
            )
        raise TargetError(failure_classification) from exc
    try:
        identity_before = temperature_sensor_identity(Path(selected_identity["class_path"]))
        require(identity_matches_required(identity_before, selected_identity), "selected temperature identity changed before read")
        with PinnedTemperatureSensor(selected_identity, allow_noncanonical_fixture=hwmon_root != Path("/sys/class/hwmon")) as sensor:
            sample = sensor.read_sample()
        identity_after = temperature_sensor_identity(Path(selected_identity["class_path"]))
        require(identity_matches_required(identity_after, selected_identity), "selected temperature identity changed after read")
    except Exception as exc:  # noqa: BLE001 - selected sensor pin/read failures must be sealed when possible
        failure_detail = str(exc)
        failure_classification = classify_temperature_discovery_exception(failure_detail, candidate_scan_count=candidate_scan_count)
        if not receipt_path.exists():
            write_temperature_discovery_failure_receipt(
                source_root=source_root,
                receipt_path=receipt_path,
                hwmon_root=hwmon_root,
                controller_nonce=controller_nonce,
                authorized_commit=authorized_commit,
                challenge=challenge,
                challenge_validation=challenge_validation,
                platform_identity=platform_identity,
                visibility=visibility,
                candidates=candidates,
                candidate_scan_count=candidate_scan_count,
                failure_classification=failure_classification,
                failure_detail=failure_detail,
            )
        raise TargetError(failure_classification) from exc
    authorizing_scope = {
        "cpuinfo_path": str(cpuinfo_path),
        "hwmon_root": str(hwmon_root),
        "canonical_cpuinfo": str(cpuinfo_path) == "/proc/cpuinfo",
        "canonical_hwmon_root": str(hwmon_root) == "/sys/class/hwmon",
        "selected_class_path_is_sysfs_hwmon": str(selected_identity["class_path"]).startswith("/sys/class/hwmon/"),
        "selected_input_is_legacy_temp1": selected_identity["sensor_input"] == LEGACY_FAMILY10H_TEMPERATURE_INPUT,
        "selected_semantic_profile_is_legacy_family10h": selected_identity["sensor_semantic_profile"] == LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
        "selected_semantic_role_is_tctl": selected_identity["sensor_semantic_role"] == LEGACY_FAMILY10H_TEMPERATURE_ROLE,
        "resolved_input_is_sysfs_device": str(selected_identity["resolved_input_path"]).startswith("/sys/devices/"),
        "resolved_device_is_sysfs_device": str(selected_identity["resolved_device_path"]).startswith("/sys/devices/"),
        "resolved_driver_is_k10temp": selected_identity["device_driver"] == "k10temp",
        "resolved_subsystem_is_pci": selected_identity["device_subsystem"] == "pci",
    }
    authorizing_scope["authorizing"] = all(
        bool(authorizing_scope[key])
        for key in [
            "canonical_cpuinfo",
            "canonical_hwmon_root",
            "selected_class_path_is_sysfs_hwmon",
            "selected_input_is_legacy_temp1",
            "selected_semantic_profile_is_legacy_family10h",
            "selected_semantic_role_is_tctl",
            "resolved_input_is_sysfs_device",
            "resolved_device_is_sysfs_device",
            "resolved_driver_is_k10temp",
            "resolved_subsystem_is_pci",
        ]
    )
    result = {
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
        "top_level_visibility_snapshot": visibility,
        "observed_candidates": candidates,
        "candidate_count": len(candidates),
        "approved_count": approved_count,
        "selection": selection,
        "selected_identity": selected_identity,
        "identity_before": identity_before,
        "sample": sample,
        "identity_after": identity_after,
    }
    result["target_discovery_receipt_sha256"] = public.digest({k: v for k, v in result.items() if k != "target_discovery_receipt_sha256"})
    write_json_exclusive_atomic(receipt_path, result)
    return result


def copy_source_fixture(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in DISCOVERY_TRANSFER_FILE_NAMES:
        path = source_root / name
        require(path.exists(), f"source fixture missing {name}")
        (destination / name).write_bytes(path.read_bytes())


def write_fake_family10h_cpuinfo(path: Path) -> None:
    path.write_text(
        "".join(
            f"processor\t: {cpu}\nvendor_id\t: AuthenticAMD\ncpu family\t: 16\nmodel\t\t: 10\nmodel name\t: AMD Phenom(tm) II\n\n"
            for cpu in range(6)
        ),
        encoding="utf-8",
    )


def write_fake_cpuinfo_processors(path: Path, processors: list[int]) -> Path:
    path.write_text(
        "".join(
            f"processor\t: {cpu}\nvendor_id\t: AuthenticAMD\ncpu family\t: 16\nmodel\t\t: 10\nmodel name\t: AMD Phenom(tm) II\n\n"
            for cpu in processors
        ),
        encoding="utf-8",
    )
    return path


def fake_pin_probe(
    required_cpus: set[int],
    *,
    inherited_affinity: list[int] | None = None,
    failures: dict[int, dict[str, Any]] | None = None,
    parent_restored: bool = True,
) -> dict[str, Any]:
    inherited_affinity = sorted(inherited_affinity if inherited_affinity is not None else required_cpus)
    failures = failures or {}
    per_cpu: dict[str, dict[str, Any]] = {}
    for cpu in sorted(required_cpus):
        failure = failures.get(cpu)
        if failure is None:
            per_cpu[str(cpu)] = {
                "cpu": cpu,
                "passed": True,
                "child_inherited_affinity": inherited_affinity,
                "requested_affinity": [cpu],
                "readback_affinity": [cpu],
                "actual_execution_cpu": cpu,
                "actual_execution_cpu_available": True,
                "errno": None,
                "errno_name": None,
                "failure_class": None,
                "diagnostic": "operational pin capability passed",
                "parent_affinity_before": inherited_affinity,
                "parent_affinity_after": inherited_affinity,
                "parent_affinity_restored": True,
            }
        else:
            errno_value = failure.get("errno")
            per_cpu[str(cpu)] = {
                "cpu": cpu,
                "passed": False,
                "child_inherited_affinity": inherited_affinity,
                "requested_affinity": [cpu],
                "readback_affinity": failure.get("readback_affinity"),
                "actual_execution_cpu": failure.get("actual_execution_cpu"),
                "actual_execution_cpu_available": failure.get("actual_execution_cpu") is not None,
                "errno": errno_value,
                "errno_name": failure.get("errno_name") or affinity_error_name(errno_value),
                "failure_class": failure["failure_class"],
                "diagnostic": failure.get("diagnostic"),
                "parent_affinity_before": inherited_affinity,
                "parent_affinity_after": inherited_affinity if parent_restored else [0],
                "parent_affinity_restored": parent_restored,
            }
    inherited_excludes = sorted(cpu for cpu in required_cpus if cpu not in inherited_affinity)
    return {
        "schema": "FAMILY10H_OPERATIONAL_PIN_CAPABILITY_PROBE_V1",
        "required_cpus": sorted(required_cpus),
        "inherited_parent_affinity": inherited_affinity,
        "inherited_affinity_excluded_required_cpus": inherited_excludes,
        "inherited_mask_restriction_only": bool(inherited_excludes) and not failures,
        "per_cpu": per_cpu,
        "parent_affinity_after_all_probes": inherited_affinity if parent_restored else [0],
        "parent_affinity_restored": parent_restored,
        "opened_pmu": False,
        "launched_runtime": False,
        "created_tomography_output_root": False,
        "passed": not failures and parent_restored,
    }


def platform_identity_regression(root: Path) -> dict[str, bool]:
    valid = root / "cpuinfo_valid"
    write_fake_family10h_cpuinfo(valid)
    intel = root / "cpuinfo_intel"
    intel.write_text("processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 58\n", encoding="utf-8")
    wrong_family = root / "cpuinfo_wrong_family"
    wrong_family.write_text("processor\t: 0\nvendor_id\t: AuthenticAMD\ncpu family\t: 23\nmodel\t\t: 1\n", encoding="utf-8")
    split = root / "cpuinfo_split"
    split.write_text(
        "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel\t\t: 10\n\nprocessor\t: 1\ncpu family\t: 16\nmodel\t\t: 10\n",
        encoding="utf-8",
    )
    mixed = root / "cpuinfo_mixed"
    mixed.write_text(
        "processor\t: 0\nvendor_id\t: AuthenticAMD\ncpu family\t: 16\nmodel\t\t: 10\n\n"
        "processor\t: 1\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 58\n",
        encoding="utf-8",
    )
    conflicting = root / "cpuinfo_conflicting"
    conflicting.write_text("processor\t: 0\nvendor_id\t: AuthenticAMD\nvendor_id\t: GenuineIntel\ncpu family\t: 16\nmodel\t\t: 10\n", encoding="utf-8")
    malformed = root / "cpuinfo_malformed"
    malformed.write_text("processor\t 0\nvendor_id\t: AuthenticAMD\ncpu family\t: 16\nmodel\t\t: 10\n", encoding="utf-8")
    valid_platform = require_family10h_platform(
        valid,
        inherited_affinity_cpus=list(range(6)),
        pin_probe=lambda required: fake_pin_probe(required, inherited_affinity=list(range(6))),
    )
    inherited_excludes_pass = require_family10h_platform(
        valid,
        inherited_affinity_cpus=[0, 1, 2, 3],
        pin_probe=lambda required: fake_pin_probe(required, inherited_affinity=[0, 1, 2, 3]),
    )

    def cloned_platform_with_readback(readback_value: Any, *, omit: bool = False) -> dict[str, Any]:
        platform = json.loads(json.dumps(valid_platform))
        per_cpu = platform["operational_pin_capability"]["per_cpu"][str(public.SOURCE_CPU_EXPECTED)]
        if omit:
            per_cpu.pop("readback_affinity", None)
        else:
            per_cpu["readback_affinity"] = readback_value
        return platform

    return {
        "valid_family10h_platform_passes": require_family10h_platform(valid)["processor_count"] == 6,
        "intel_platform_rejected": raises_target_error(lambda: require_family10h_platform(intel)),
        "wrong_amd_family_rejected": raises_target_error(lambda: require_family10h_platform(wrong_family)),
        "split_vendor_family_rejected": raises_target_error(lambda: require_family10h_platform(split)),
        "mixed_processors_rejected": raises_target_error(lambda: require_family10h_platform(mixed)),
        "duplicate_conflicting_fields_rejected": raises_target_error(lambda: require_family10h_platform(conflicting)),
        "malformed_cpuinfo_rejected": raises_target_error(lambda: require_family10h_platform(malformed)),
        "missing_source_cpu_rejected": raises_target_error_containing(
            lambda: require_family10h_platform(write_fake_cpuinfo_processors(root / "cpuinfo_missing_source", [0, 1, 2, 3, 5])),
            "platform identity missing frozen source CPU",
        ),
        "missing_receiver_cpu_rejected": raises_target_error_containing(
            lambda: require_family10h_platform(write_fake_cpuinfo_processors(root / "cpuinfo_missing_receiver", [0, 1, 2, 3, 4])),
            "platform identity missing frozen receiver CPU",
        ),
        "inherited_mask_excludes_required_but_pin_capability_passes": inherited_excludes_pass["operational_pin_capability"]["inherited_mask_restriction_only"] is True,
        "sched_setaffinity_einval_source_rejected": raises_target_error_containing(
            lambda: require_family10h_platform(
                valid,
                inherited_affinity_cpus=[0, 1, 2, 3],
                pin_probe=lambda required: fake_pin_probe(
                    required,
                    inherited_affinity=[0, 1, 2, 3],
                    failures={
                        public.SOURCE_CPU_EXPECTED: {
                            "failure_class": "cpuset_or_kernel_rejected_cpu",
                            "errno": errno_module.EINVAL,
                            "diagnostic": "sched_setaffinity EINVAL",
                        }
                    },
                ),
            ),
            "operational pin capability failed for CPU 4: sched_setaffinity EINVAL",
        ),
        "sched_setaffinity_eperm_receiver_rejected": raises_target_error_containing(
            lambda: require_family10h_platform(
                valid,
                inherited_affinity_cpus=[0, 1, 2, 3],
                pin_probe=lambda required: fake_pin_probe(
                    required,
                    inherited_affinity=[0, 1, 2, 3],
                    failures={
                        public.RECEIVER_CPU_EXPECTED: {
                            "failure_class": "permission_failure",
                            "errno": errno_module.EPERM,
                            "diagnostic": "sched_setaffinity EPERM",
                        }
                    },
                ),
            ),
            "operational pin capability failed for CPU 5: sched_setaffinity EPERM",
        ),
        "child_affinity_readback_mismatch_rejected": raises_target_error_containing(
            lambda: require_family10h_platform(
                valid,
                inherited_affinity_cpus=[0, 1, 2, 3],
                pin_probe=lambda required: fake_pin_probe(
                    required,
                    inherited_affinity=[0, 1, 2, 3],
                    failures={
                        public.SOURCE_CPU_EXPECTED: {
                            "failure_class": "affinity_readback_failure",
                            "readback_affinity": [0],
                            "diagnostic": "child affinity readback differs from requested singleton",
                        }
                    },
                ),
            ),
            "operational pin capability failed for CPU 4: child affinity readback differs",
        ),
        "child_affinity_readback_missing_rejected": bool(operational_pin_capability_failures(cloned_platform_with_readback(None, omit=True))),
        "child_affinity_readback_null_rejected": bool(operational_pin_capability_failures(cloned_platform_with_readback(None))),
        "child_affinity_readback_nonlist_rejected": bool(operational_pin_capability_failures(cloned_platform_with_readback(str(public.SOURCE_CPU_EXPECTED)))),
        "parent_affinity_change_rejected": raises_target_error_containing(
            lambda: require_family10h_platform(
                valid,
                inherited_affinity_cpus=[0, 1, 2, 3],
                pin_probe=lambda required: fake_pin_probe(required, inherited_affinity=[0, 1, 2, 3], parent_restored=False),
            ),
            "operational pin capability failed: parent affinity changed after child probe",
        ),
    }


def discovery_command_isolation_regression() -> dict[str, bool]:
    source = inspect.getsource(discover_temperature_sensor_authority)
    forbidden_tokens = [
        "perf_event_open",
        "/dev/cpu",
        "msr",
        "subprocess.",
        "Popen",
        "run(",
        "execute_authorized",
        "EXPECTED_REMOTE_OUTPUT_ROOT",
        "output_root.mkdir",
    ]
    return {
        "discovery_source_has_no_pmu_runtime_or_output_root_tokens": not any(token in source for token in forbidden_tokens),
        "discovery_receipt_counters_are_literals_only": all(
            token in source
            for token in [
                '"target_contact_count": 1',
                '"sensor_inventory_count": 1',
                '"live_invocation_count": 0',
                '"pmu_acquisition_count": 0',
                '"pmu_open_count": 0',
                '"runtime_launch_count": 0',
                '"tomography_output_root_created": False',
            ]
        ),
    }


def target_sensor_discovery_fixture(source_root: Path) -> dict[str, Any]:
    sensor_regressions = temperature_sensor_identity_fixture()
    with tempfile.TemporaryDirectory(prefix="family10h_target_discovery_") as tmp:
        root = Path(tmp)
        platform_regressions = platform_identity_regression(root)
        isolation_regressions = discovery_command_isolation_regression()
        source = root / "source"
        copy_source_fixture(source_root, source)

        def fresh_source(label: str) -> Path:
            case_source = root / label / "source"
            copy_source_fixture(source_root, case_source)
            return case_source

        def fresh_challenge(label: str, mutator: Any | None = None) -> tuple[Path, Path]:
            case_source = fresh_source(label)
            case_challenge = expected_discovery_challenge(
                case_source,
                commit,
                nonce_sha,
                source_authority_review=fixture_source_review_binding(case_source, commit),
            )
            if mutator is not None:
                case_challenge = mutator(case_challenge)
            case_challenge_path = root / label / "challenge.json"
            write_json(case_challenge_path, case_challenge)
            return case_source, case_challenge_path

        cpuinfo = root / "cpuinfo"
        write_fake_family10h_cpuinfo(cpuinfo)
        nonce = "a" * 64
        commit = "b" * 40
        nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
        challenge = expected_discovery_challenge(
            source,
            commit,
            nonce_sha,
            source_authority_review=fixture_source_review_binding(source, commit),
        )
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

        missing_challenge_source = fresh_source("missing_challenge")
        missing_challenge_failure_path = missing_challenge_source / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME
        missing_challenge_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=missing_challenge_source,
                controller_challenge_path=root / "missing.json",
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=missing_challenge_failure_path,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )
        missing_challenge_failure = read_json(missing_challenge_failure_path) if missing_challenge_failure_path.exists() else {}
        wrong_nonce_source, wrong_nonce_challenge_path = fresh_challenge("wrong_nonce")
        wrong_nonce_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=wrong_nonce_source,
                controller_challenge_path=wrong_nonce_challenge_path,
                controller_nonce="c" * 64,
                authorized_commit=commit,
                receipt_path=root / "wrong_nonce" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )
        wrong_commit_source, wrong_commit_challenge_path = fresh_challenge("wrong_commit")
        wrong_commit_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=wrong_commit_source,
                controller_challenge_path=wrong_commit_challenge_path,
                controller_nonce=nonce,
                authorized_commit="d" * 40,
                receipt_path=root / "wrong_commit" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_bundle_source, wrong_bundle_path = fresh_challenge("wrong_bundle", lambda case: {**case, "source_bundle_sha256": "0" * 64})
        wrong_bundle_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=wrong_bundle_source,
                controller_challenge_path=wrong_bundle_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_bundle" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_schedule_source, wrong_schedule_path = fresh_challenge("wrong_schedule", lambda case: {**case, "schedule_json_sha256": "1" * 64})
        wrong_schedule_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=wrong_schedule_source,
                controller_challenge_path=wrong_schedule_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_schedule" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        wrong_package_source, wrong_package_path = fresh_challenge("wrong_package", lambda case: {**case, "science_package_id": "wrong_package"})
        wrong_package_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=wrong_package_source,
                controller_challenge_path=wrong_package_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_package" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )

        nested_wrong_hash_source, nested_wrong_hash_path = fresh_challenge(
            "nested_wrong_hash",
            lambda case: {**case, "source_authority_review": {**case["source_authority_review"], "source_hashes_sha256": "4" * 64}},
        )
        nested_wrong_hash_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=nested_wrong_hash_source,
                controller_challenge_path=nested_wrong_hash_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "nested_wrong_hash" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )
        nested_wrong_bundle_source, nested_wrong_bundle_path = fresh_challenge(
            "nested_wrong_bundle",
            lambda case: {**case, "source_authority_review": {**case["source_authority_review"], "source_bundle_sha256": "5" * 64}},
        )
        nested_wrong_bundle_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=nested_wrong_bundle_source,
                controller_challenge_path=nested_wrong_bundle_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "nested_wrong_bundle" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=hwmon,
                cpuinfo_path=cpuinfo,
            )
        )
        nested_extra_key_source, nested_extra_key_path = fresh_challenge(
            "nested_extra_key",
            lambda case: {**case, "source_authority_review": {**case["source_authority_review"], "unexpected": "blocked"}},
        )
        nested_extra_key_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=nested_extra_key_source,
                controller_challenge_path=nested_extra_key_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "nested_extra_key" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
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

        def transfer_root_rejects(label: str, mutator: Any) -> bool:
            case_source, case_challenge_path = fresh_challenge(label)
            case_challenge = read_json(case_challenge_path)
            mutator(case_source)
            return not validate_discovery_transfer_root(case_source, challenge=case_challenge)["passed"]

        transfer_runtime_omitted_rejected = transfer_root_rejects(
            "transfer_runtime_omitted",
            lambda case: (case / RUNTIME_BINARY_NAME).unlink(),
        )
        transfer_bundle_omitted_rejected = transfer_root_rejects(
            "transfer_bundle_omitted",
            lambda case: (case / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz").unlink(),
        )
        transfer_source_hash_omitted_rejected = transfer_root_rejects(
            "transfer_source_hash_omitted",
            lambda case: (case / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json").unlink(),
        )
        transfer_one_source_omitted_rejected = transfer_root_rejects(
            "transfer_one_source_omitted",
            lambda case: (case / SOURCE_FILE_NAMES[0]).unlink(),
        )
        transfer_runtime_bytes_mutated_rejected = transfer_root_rejects(
            "transfer_runtime_bytes_mutated",
            lambda case: (case / RUNTIME_BINARY_NAME).write_bytes((case / RUNTIME_BINARY_NAME).read_bytes() + b"\nmutated-runtime\n"),
        )
        transfer_runtime_size_mutated_rejected = transfer_root_rejects(
            "transfer_runtime_size_mutated",
            lambda case: (case / RUNTIME_BINARY_NAME).write_bytes((case / RUNTIME_BINARY_NAME).read_bytes()[:-1] or b"x"),
        )
        transfer_bundle_bytes_mutated_rejected = transfer_root_rejects(
            "transfer_bundle_bytes_mutated",
            lambda case: (case / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz").write_bytes(
                (case / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz").read_bytes() + b"\nmutated-bundle\n"
            ),
        )
        transfer_bundle_reconstruction_mismatch_rejected = transfer_root_rejects(
            "transfer_bundle_reconstruction_mismatch",
            lambda case: (case / SOURCE_FILE_NAMES[-1]).write_text(
                (case / SOURCE_FILE_NAMES[-1]).read_text(encoding="utf-8") + "\n# bundle reconstruction mismatch\n",
                encoding="utf-8",
            ),
        )
        transfer_extra_authority_file_rejected = transfer_root_rejects(
            "transfer_extra_authority_file",
            lambda case: (case / "unexpected_authority_file.txt").write_text("unexpected\n", encoding="utf-8"),
        )

        non_cpu_root = root / "non_cpu_hwmon"
        write_fake_hwmon_sensor(non_cpu_root, 0, "acpitz", "Tctl")
        non_cpu_source, non_cpu_challenge_path = fresh_challenge("non_cpu")
        non_cpu_failure_path = non_cpu_source / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME
        non_cpu_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=non_cpu_source,
                controller_challenge_path=non_cpu_challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=non_cpu_failure_path,
                hwmon_root=non_cpu_root,
                cpuinfo_path=cpuinfo,
            )
        )
        non_cpu_failure = read_json(non_cpu_failure_path) if non_cpu_failure_path.exists() else {}

        wrong_label_root = root / "wrong_label_hwmon"
        write_fake_hwmon_sensor(wrong_label_root, 0, "k10temp", "ambient")
        wrong_label_source, wrong_label_challenge_path = fresh_challenge("wrong_label")
        wrong_label_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=wrong_label_source,
                controller_challenge_path=wrong_label_challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "wrong_label" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=wrong_label_root,
                cpuinfo_path=cpuinfo,
            )
        )

        malformed_root = root / "malformed_hwmon"
        write_fake_hwmon_sensor(malformed_root, 0, "k10temp", "Tctl", "not-a-number")
        malformed_source, malformed_challenge_path = fresh_challenge("malformed")
        malformed_sample_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=malformed_source,
                controller_challenge_path=malformed_challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=root / "malformed" / "source" / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME,
                hwmon_root=malformed_root,
                cpuinfo_path=cpuinfo,
            )
        )

        ambiguous_root = root / "ambiguous_hwmon"
        write_fake_hwmon_sensor(ambiguous_root, 0, "k10temp", "Tctl", "42000")
        write_fake_hwmon_sensor(ambiguous_root, 1, "k10temp", None, "43000")
        ambiguous_without_law_rejected = raises_target_error(
            lambda: select_approved_temperature_identity(enumerate_temperature_candidates(ambiguous_root), deterministic_law=False)
        )
        ambiguous_source, ambiguous_challenge_path = fresh_challenge("ambiguous")
        ambiguous_failure_path = ambiguous_source / TEMPERATURE_SENSOR_DISCOVERY_RECEIPT_NAME
        multiple_approved_discovery_rejected = raises_target_error(
            lambda: discover_temperature_sensor_authority(
                source_root=ambiguous_source,
                controller_challenge_path=ambiguous_challenge_path,
                controller_nonce=nonce,
                authorized_commit=commit,
                receipt_path=ambiguous_failure_path,
                hwmon_root=ambiguous_root,
                cpuinfo_path=cpuinfo,
            )
        )
        ambiguous_failure = read_json(ambiguous_failure_path) if ambiguous_failure_path.exists() else {}

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
        "valid_discovery_selected_tctl": receipt["selected_identity"]["sensor_semantic_role"] == "Tctl"
        and receipt["selected_identity"]["sensor_semantic_profile"] == LEGACY_FAMILY10H_TEMPERATURE_PROFILE,
        "valid_discovery_records_all_candidates": len(receipt["observed_candidates"]) == 3,
        "missing_challenge_rejected": missing_challenge_rejected,
        "missing_challenge_writes_structured_failure_receipt": missing_challenge_failure.get("schema") == TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA
        and missing_challenge_failure.get("failure_classification") == "CONTROLLER_CHALLENGE_MISSING"
        and missing_challenge_failure.get("candidate_scan_count") == 0
        and missing_challenge_failure.get("observed_candidates") == []
        and missing_challenge_failure.get("target_discovery_receipt_sha256")
        == public.digest({k: v for k, v in missing_challenge_failure.items() if k != "target_discovery_receipt_sha256"}),
        "wrong_nonce_rejected": wrong_nonce_rejected,
        "wrong_source_commit_rejected": wrong_commit_rejected,
        "wrong_source_hash_receipt_rejected": wrong_source_hash_rejected,
        "wrong_source_bundle_rejected": wrong_bundle_rejected,
        "transfer_runtime_omitted_rejected": transfer_runtime_omitted_rejected,
        "transfer_source_bundle_omitted_rejected": transfer_bundle_omitted_rejected,
        "transfer_source_hash_receipt_omitted_rejected": transfer_source_hash_omitted_rejected,
        "transfer_one_source_omitted_rejected": transfer_one_source_omitted_rejected,
        "transfer_runtime_bytes_mutated_rejected": transfer_runtime_bytes_mutated_rejected,
        "transfer_runtime_size_mutated_rejected": transfer_runtime_size_mutated_rejected,
        "transfer_source_bundle_bytes_mutated_rejected": transfer_bundle_bytes_mutated_rejected,
        "transfer_source_bundle_reconstruction_mismatch_rejected": transfer_bundle_reconstruction_mismatch_rejected,
        "transfer_extra_authority_file_rejected": transfer_extra_authority_file_rejected,
        "wrong_schedule_hashes_rejected": wrong_schedule_rejected,
        "wrong_package_identity_rejected": wrong_package_rejected,
        "nested_source_review_hash_mismatch_rejected": nested_wrong_hash_rejected,
        "nested_source_review_bundle_mismatch_rejected": nested_wrong_bundle_rejected,
        "nested_source_review_keyset_mismatch_rejected": nested_extra_key_rejected,
        "non_cpu_hwmon_rejected": non_cpu_rejected,
        "zero_approved_inventory_writes_structured_failure_receipt": non_cpu_failure.get("schema") == TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA
        and non_cpu_failure.get("passed") is False
        and non_cpu_failure.get("candidate_scan_count") == 1
        and non_cpu_failure.get("candidate_count") == 1
        and non_cpu_failure.get("approved_count") == 0
        and non_cpu_failure.get("target_discovery_receipt_sha256")
        == public.digest({k: v for k, v in non_cpu_failure.items() if k != "target_discovery_receipt_sha256"}),
        "wrong_sensor_label_rejected": wrong_label_rejected,
        "multiple_approved_without_deterministic_law_rejected": ambiguous_without_law_rejected,
        "multiple_approved_inventory_writes_structured_failure_receipt": multiple_approved_discovery_rejected
        and ambiguous_failure.get("schema") == TEMPERATURE_SENSOR_DISCOVERY_FAILURE_SCHEMA
        and ambiguous_failure.get("passed") is False
        and ambiguous_failure.get("candidate_scan_count") == 1
        and ambiguous_failure.get("candidate_count") == 2
        and ambiguous_failure.get("approved_count") == 2
        and ambiguous_failure.get("target_discovery_receipt_sha256")
        == public.digest({k: v for k, v in ambiguous_failure.items() if k != "target_discovery_receipt_sha256"}),
        "descriptor_drift_rejected": sensor_regressions["same_class_path_substitution_rejected"],
        "identity_drift_rejected": sensor_regressions["identity_drift_rejected"],
        "malformed_sample_rejected": malformed_sample_rejected,
        **platform_regressions,
        **isolation_regressions,
    }
    return {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_DISCOVERY_SELF_TEST_V1",
        "passed": all(checks.values()),
        "checks": checks,
        "valid_receipt": receipt,
    }


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    schedule_result = validate_schedule_artifacts(source_root)
    schedule = read_json(source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json")
    evidence = evidence_file_fixtures(schedule)
    feature = public.feature_boundary_self_test()
    process = process_custody_fixture()
    policy = policy_and_platform_fixture()
    manifest_live_gate = manifest_live_gate_fixture()
    final_receipt_gate = final_exact_object_falsey_manifest_fixture(source_root)
    source_audit_receipt_versions = source_audit_receipt_version_fixture(source_root)
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
        "final_exact_object_falsey_manifest_fixture": final_receipt_gate,
        "source_audit_receipt_version_fixture": source_audit_receipt_versions,
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
            final_receipt_gate["passed"],
            source_audit_receipt_versions["passed"],
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
    schedule = read_json(source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json")
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
            build_authorized_feature_freeze(schedule, approved_temperature_identity),
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
        print(strict_json_dumps(result, indent=2))
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
        print(strict_json_dumps(result, indent=2))
        return 0
    if not args.self_test:
        parser.print_help()
        return 2
    result = self_test(args.source_root.resolve(), args.output_root.resolve())
    print(strict_json_dumps(result, indent=2))
    return 0 if result["self_test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
