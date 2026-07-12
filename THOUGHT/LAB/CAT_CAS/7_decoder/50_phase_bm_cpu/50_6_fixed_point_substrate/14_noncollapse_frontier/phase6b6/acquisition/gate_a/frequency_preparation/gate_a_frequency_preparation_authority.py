#!/usr/bin/env python3
"""Exact one-shot authority validation for Gate A frequency preparation.

This module is network-free and Git-free.  It validates immutable authority bytes
against an already validated preparation bundle manifest and returns an opaque
permit consumed by the target-side live transaction wrapper.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUTHORITY_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_AUTHORITY_V1"
MANIFEST_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_BUNDLE_MANIFEST_V1"
EXPECTED_TARGET = "root@192.168.137.100"
EXPECTED_HOSTNAME = "catcas"
EXPECTED_ARCHITECTURE = "x86_64"
EXPECTED_CPU_MODEL = "AMD Phenom(tm) II X6 1090T Processor"
EXPECTED_SYSFS_ROOT = "/sys"
REQUIRED_FREQUENCY_KHZ = 1_600_000
EXPECTED_BASELINE_MIN_KHZ = 800_000
EXPECTED_BASELINE_MAX_KHZ = 3_200_000
SAMPLE_COUNT = 200
SAMPLE_INTERVAL_MS = 10
MAXIMUM_TRANSACTION_COUNT = 1
MAXIMUM_WRITE_ATTEMPT_COUNT = 8

AUTHORITY_KEYS = {
    "schema_id",
    "authority_id",
    "reviewed_source_commit",
    "reviewed_source_tree_sha1",
    "independent_review_id",
    "bundle_sha256",
    "deterministic_archive_sha256",
    "manifest_sha256",
    "source_git_blobs",
    "target",
    "target_identity",
    "target_identity_sha256",
    "sysfs_root",
    "remote_execution_root",
    "remote_output_root",
    "remote_stage_archive",
    "remote_evidence_archive",
    "remote_claim_root",
    "required_frequency_khz",
    "expected_baseline_min_khz",
    "expected_baseline_max_khz",
    "sample_count",
    "sample_interval_ms",
    "maximum_transaction_count",
    "maximum_write_attempt_count",
    "consumed",
    "project_owner_approved",
    "authority_state",
}

SOURCE_BLOB_KEYS = {
    "host_adapter",
    "host_transport",
    "authority_validator",
    "live_transaction",
    "target_runner",
    "target_bundle",
    "reviewed_preparation_core",
}

TARGET_IDENTITY_KEYS = {"hostname", "architecture", "cpu_model"}

AUTHORITY_STATE_KEYS = {
    "authorization_artifact_created",
    "frequency_preparation_authorized",
    "restoration_authorized",
    "ssh_authorized",
    "scp_authorized",
    "target_filesystem_staging_authorized",
    "engineering_smoke_authorized",
    "hardware_execution_authorized",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
    "automatic_retry",
}

TARGET_PAYLOAD_ROLES = {
    "reviewed_preparation_core",
    "authority_validator",
    "live_transaction",
    "target_runner",
    "target_bundle",
}


class AuthorityError(RuntimeError):
    """Closed authority validation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(char in "0123456789abcdef" for char in value)


def target_identity_digest(value: dict[str, str]) -> str:
    require(set(value) == TARGET_IDENTITY_KEYS, "target identity key set mismatch")
    return sha256_bytes(canonical_bytes(value))


def expected_remote_paths(authority_id: str) -> dict[str, str]:
    require(re.fullmatch(r"gate_a_freqprep_[0-9a-f]{8}_[0-9]{2}", authority_id) is not None, "authority ID malformed")
    slug = authority_id
    return {
        "remote_execution_root": f"/root/catcas_phase6b6_gate_a_freqprep_{slug}",
        "remote_output_root": f"/root/catcas_phase6b6_gate_a_freqprep_{slug}/evidence",
        "remote_stage_archive": f"/root/catcas_phase6b6_gate_a_freqprep_{slug}.tar",
        "remote_evidence_archive": f"/root/catcas_phase6b6_gate_a_freqprep_{slug}.evidence.tar",
        "remote_claim_root": f"/root/.catcas_gate_a_freqprep_claim_{slug}",
    }


def validate_manifest_shape(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(
        set(manifest)
        == {
            "schema_id",
            "files",
            "bundle_sha256",
            "deterministic_archive_sha256",
            "authority_artifact_created",
            "live_frequency_preparation_authorized",
            "target_contact_authorized",
        },
        "preparation manifest key set mismatch",
    )
    require(manifest["schema_id"] == MANIFEST_SCHEMA_ID, "preparation manifest schema mismatch")
    require(manifest["authority_artifact_created"] is False, "source manifest must not contain authority")
    require(manifest["live_frequency_preparation_authorized"] is False, "source manifest must not authorize live preparation")
    require(manifest["target_contact_authorized"] is False, "source manifest must not authorize target contact")
    require(is_hex(manifest["bundle_sha256"], 64), "bundle digest malformed")
    require(is_hex(manifest["deterministic_archive_sha256"], 64), "archive digest malformed")
    files = manifest["files"]
    require(isinstance(files, list) and len(files) == 5, "preparation payload file count mismatch")
    roles: dict[str, dict[str, Any]] = {}
    previous = ""
    for entry in files:
        require(
            isinstance(entry, dict)
            and set(entry)
            == {
                "package_path",
                "source_repository_path",
                "git_mode",
                "git_blob_sha1",
                "byte_size",
                "sha256",
                "role",
            },
            "preparation manifest file entry mismatch",
        )
        package_path = entry["package_path"]
        require(isinstance(package_path, str) and package_path > previous, "manifest package paths not strictly sorted")
        require(not package_path.startswith("/") and ".." not in Path(package_path).parts, "unsafe package path")
        require(entry["git_mode"] == "100644", "unexpected package Git mode")
        require(is_hex(entry["git_blob_sha1"], 40), "package blob malformed")
        require(isinstance(entry["byte_size"], int) and entry["byte_size"] > 0, "package byte size malformed")
        require(is_hex(entry["sha256"], 64), "package SHA-256 malformed")
        role = entry["role"]
        require(role in TARGET_PAYLOAD_ROLES and role not in roles, "package role mismatch")
        roles[role] = entry
        previous = package_path
    require(set(roles) == TARGET_PAYLOAD_ROLES, "preparation manifest roles incomplete")
    core = {key: value for key, value in manifest.items() if key not in {"bundle_sha256", "deterministic_archive_sha256"}}
    require(sha256_bytes(canonical_bytes(core)) == manifest["bundle_sha256"], "preparation bundle digest mismatch")
    return roles


_PERMIT_TOKEN = object()


@dataclass(frozen=True)
class PreparationPermit:
    """Opaque validated authority permit.  Construction is module-private."""

    _token: object
    authority_id: str
    authority_sha256: str
    bundle_sha256: str
    reviewed_source_commit: str
    independent_review_id: int
    remote_execution_root: str
    remote_output_root: str
    remote_stage_archive: str
    remote_evidence_archive: str
    remote_claim_root: str
    sysfs_root: str
    required_frequency_khz: int
    sample_count: int
    sample_interval_ms: int
    maximum_write_attempt_count: int
    target: str
    target_identity: dict[str, str]


def require_permit(value: Any) -> PreparationPermit:
    require(isinstance(value, PreparationPermit), "validated preparation permit required")
    require(value._token is _PERMIT_TOKEN, "preparation permit token mismatch")
    return value


def validate_authority(
    authority: dict[str, Any],
    *,
    authority_bytes: bytes,
    authority_sha256: str,
    exact_manifest: dict[str, Any],
    expected_reviewed_source_commit: str,
    expected_independent_review_id: int,
) -> PreparationPermit:
    require(sha256_bytes(authority_bytes) == authority_sha256, "authority SHA-256 mismatch")
    require(set(authority) == AUTHORITY_KEYS, "preparation authority key set mismatch")
    require(authority["schema_id"] == AUTHORITY_SCHEMA_ID, "preparation authority schema mismatch")
    authority_id = authority["authority_id"]
    paths = expected_remote_paths(authority_id)
    for field, expected in paths.items():
        require(authority[field] == expected, f"preparation authority {field} mismatch")

    require(is_hex(authority["reviewed_source_commit"], 40), "reviewed source commit malformed")
    require(authority["reviewed_source_commit"] == expected_reviewed_source_commit, "reviewed source commit mismatch")
    require(is_hex(authority["reviewed_source_tree_sha1"], 40), "reviewed source tree malformed")
    require(
        isinstance(expected_independent_review_id, int) and expected_independent_review_id > 0,
        "expected independent review ID malformed",
    )
    require(authority["independent_review_id"] == expected_independent_review_id, "independent review ID mismatch")

    roles = validate_manifest_shape(exact_manifest)
    require(authority["bundle_sha256"] == exact_manifest["bundle_sha256"], "authority bundle digest mismatch")
    require(
        authority["deterministic_archive_sha256"] == exact_manifest["deterministic_archive_sha256"],
        "authority archive digest mismatch",
    )
    require(is_hex(authority["manifest_sha256"], 64), "authority manifest digest malformed")

    source_blobs = authority["source_git_blobs"]
    require(isinstance(source_blobs, dict) and set(source_blobs) == SOURCE_BLOB_KEYS, "source blob key set mismatch")
    require(all(is_hex(value, 40) for value in source_blobs.values()), "source blob binding malformed")
    for role in TARGET_PAYLOAD_ROLES:
        require(source_blobs[role] == roles[role]["git_blob_sha1"], f"source blob mismatch for {role}")

    require(authority["target"] == EXPECTED_TARGET, "target mismatch")
    identity = authority["target_identity"]
    require(identity == {
        "hostname": EXPECTED_HOSTNAME,
        "architecture": EXPECTED_ARCHITECTURE,
        "cpu_model": EXPECTED_CPU_MODEL,
    }, "target identity fields mismatch")
    require(authority["target_identity_sha256"] == target_identity_digest(identity), "target identity digest mismatch")
    require(authority["sysfs_root"] == EXPECTED_SYSFS_ROOT, "sysfs root mismatch")
    require(authority["required_frequency_khz"] == REQUIRED_FREQUENCY_KHZ, "required frequency mismatch")
    require(authority["expected_baseline_min_khz"] == EXPECTED_BASELINE_MIN_KHZ, "baseline minimum mismatch")
    require(authority["expected_baseline_max_khz"] == EXPECTED_BASELINE_MAX_KHZ, "baseline maximum mismatch")
    require(authority["sample_count"] == SAMPLE_COUNT, "sample count mismatch")
    require(authority["sample_interval_ms"] == SAMPLE_INTERVAL_MS, "sample interval mismatch")
    require(authority["maximum_transaction_count"] == MAXIMUM_TRANSACTION_COUNT, "transaction count mismatch")
    require(authority["maximum_write_attempt_count"] == MAXIMUM_WRITE_ATTEMPT_COUNT, "write attempt count mismatch")
    require(authority["consumed"] is False, "preparation authority already consumed")
    require(authority["project_owner_approved"] is True, "project owner approval missing")

    state = authority["authority_state"]
    require(isinstance(state, dict) and set(state) == AUTHORITY_STATE_KEYS, "authority state key set mismatch")
    true_fields = {
        "authorization_artifact_created",
        "frequency_preparation_authorized",
        "restoration_authorized",
        "ssh_authorized",
        "scp_authorized",
        "target_filesystem_staging_authorized",
    }
    false_fields = AUTHORITY_STATE_KEYS - true_fields
    require(all(state[field] is True for field in true_fields), "required preparation authority field missing")
    require(all(state[field] is False for field in false_fields), "downstream or retry authority must remain false")

    return PreparationPermit(
        _token=_PERMIT_TOKEN,
        authority_id=authority_id,
        authority_sha256=authority_sha256,
        bundle_sha256=authority["bundle_sha256"],
        reviewed_source_commit=authority["reviewed_source_commit"],
        independent_review_id=authority["independent_review_id"],
        remote_execution_root=authority["remote_execution_root"],
        remote_output_root=authority["remote_output_root"],
        remote_stage_archive=authority["remote_stage_archive"],
        remote_evidence_archive=authority["remote_evidence_archive"],
        remote_claim_root=authority["remote_claim_root"],
        sysfs_root=authority["sysfs_root"],
        required_frequency_khz=authority["required_frequency_khz"],
        sample_count=authority["sample_count"],
        sample_interval_ms=authority["sample_interval_ms"],
        maximum_write_attempt_count=authority["maximum_write_attempt_count"],
        target=authority["target"],
        target_identity=dict(identity),
    )


def load_and_validate_authority(
    path: Path,
    *,
    authority_sha256: str,
    exact_manifest: dict[str, Any],
    expected_reviewed_source_commit: str,
    expected_independent_review_id: int,
) -> tuple[dict[str, Any], PreparationPermit]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    require(isinstance(value, dict), "authority artifact must be a JSON object")
    permit = validate_authority(
        value,
        authority_bytes=data,
        authority_sha256=authority_sha256,
        exact_manifest=exact_manifest,
        expected_reviewed_source_commit=expected_reviewed_source_commit,
        expected_independent_review_id=expected_independent_review_id,
    )
    return value, permit
