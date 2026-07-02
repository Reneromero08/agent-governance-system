#!/usr/bin/env python3
"""Shared exact Gate A future-authority validator.

This validator is Git-free. It operates on the authority artifact bytes and an
already validated exact manifest object. The host supplies a manifest validated
by exact committed-tree Git reconstruction (build_gate_a_execution_bundle.py);
the target supplies a manifest validated locally against an extracted bundle
(gate_a_target_bundle.py). This module never requires .git, HEAD, git commands,
repository paths, or Git object reconstruction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gate_a_target_bundle as target_bundle

AUTHORITY_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_AUTHORITY_V1"
AUTHORITY_KEYS = {
    "schema_id",
    "reviewed_adapter_head",
    "independent_review_id",
    "execution_bundle_sha256",
    "host_adapter_git_blob_sha1",
    "target_runner_git_blob_sha1",
    "target_worker_git_blob_sha1",
    "schedule_sha256",
    "target_namespace_sha256",
    "target_identity_sha256",
    "target",
    "remote_execution_root",
    "remote_output_root",
    "maximum_execution_count",
    "consumed",
    "project_owner_approved",
    "authority_state",
}
AUTHORITY_STATE_KEYS = {
    "authorization_artifact_created",
    "engineering_smoke_authorized",
    "hardware_ran",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
    "automatic_retry",
}

SCHEDULE_SHA256 = target_bundle.SCHEDULE_SHA256
NAMESPACE_SHA256 = target_bundle.NAMESPACE_SHA256
TARGET_IDENTITY_SHA256 = target_bundle.TARGET_IDENTITY_SHA256
EXPECTED_TARGET = "root@192.168.137.100"
REMOTE_EXECUTION_ROOT = "/root/catcas_phase6b6_gate_a_smoke_9c416379"
REMOTE_OUTPUT_ROOT = "/root/catcas_phase6b6_gate_a_smoke_9c416379/evidence"


class AuthorityError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AuthorityError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json_object_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value, data


def entries_by_role(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        return target_bundle.entries_by_role(manifest)
    except target_bundle.TargetBundleError as exc:
        raise AuthorityError(str(exc)) from exc


def validate_execution_authority(
    authority: dict[str, Any],
    *,
    authority_sha256: str,
    authority_bytes: bytes,
    expected_reviewed_adapter_head: str,
    expected_independent_review_id: int,
    exact_manifest: dict[str, Any],
) -> dict[str, Any]:
    require(sha256_bytes(authority_bytes) == authority_sha256, "authority SHA-256 mismatch")
    require(set(authority) == AUTHORITY_KEYS, "authority top-level key set mismatch")
    require(authority["schema_id"] == AUTHORITY_SCHEMA_ID, "authority schema mismatch")
    require(authority["reviewed_adapter_head"] == expected_reviewed_adapter_head, "reviewed adapter head mismatch")
    require(isinstance(expected_independent_review_id, int) and expected_independent_review_id > 0, "expected review ID must be positive")
    require(authority["independent_review_id"] == expected_independent_review_id, "independent review ID mismatch")

    try:
        target_bundle.validate_manifest_shape(exact_manifest)
    except target_bundle.TargetBundleError as exc:
        raise AuthorityError(f"supplied manifest is not exact: {exc}") from exc
    roles = entries_by_role(exact_manifest)
    require(authority["execution_bundle_sha256"] == exact_manifest["execution_bundle_sha256"], "execution bundle digest mismatch")
    require(authority["host_adapter_git_blob_sha1"] == roles["host_adapter"]["git_blob_sha1"], "host adapter blob mismatch")
    require(authority["target_runner_git_blob_sha1"] == roles["target_runner"]["git_blob_sha1"], "target runner blob mismatch")
    require(authority["target_worker_git_blob_sha1"] == roles["target_worker"]["git_blob_sha1"], "target worker blob mismatch")
    require(authority["schedule_sha256"] == SCHEDULE_SHA256, "schedule digest mismatch")
    require(authority["target_namespace_sha256"] == NAMESPACE_SHA256, "target namespace digest mismatch")
    require(authority["target_identity_sha256"] == TARGET_IDENTITY_SHA256, "target identity digest mismatch")
    require(authority["target"] == EXPECTED_TARGET, "target mismatch")
    require(authority["remote_execution_root"] == REMOTE_EXECUTION_ROOT, "remote execution root mismatch")
    require(authority["remote_output_root"] == REMOTE_OUTPUT_ROOT, "remote output root mismatch")
    require(authority["maximum_execution_count"] == 1, "maximum execution count mismatch")
    require(authority["consumed"] is False, "authority already consumed")
    require(authority["project_owner_approved"] is True, "project owner approval missing")

    state = authority["authority_state"]
    require(isinstance(state, dict), "authority state must be an object")
    require(set(state) == AUTHORITY_STATE_KEYS, "authority-state key set mismatch")
    require(state["authorization_artifact_created"] is True, "future artifact must declare created")
    require(state["engineering_smoke_authorized"] is True, "future smoke authority missing")
    require(state["hardware_ran"] is False, "hardware-ran flag must be false")
    require(state["calibration_authorized"] is False, "calibration must remain unauthorized")
    require(state["scientific_acquisition_authorized"] is False, "scientific acquisition must remain unauthorized")
    require(state["restoration_authorized"] is False, "restoration must remain unauthorized")
    require(state["target_coupling_authorized"] is False, "target coupling must remain unauthorized")
    require(state["small_wall_authorized"] is False, "Small Wall work must remain unauthorized")
    require(state["automatic_retry"] is False, "retry must remain disabled")
    return {
        "status": "GATE_A_EXECUTION_AUTHORITY_EXACT",
        "reviewed_adapter_head": authority["reviewed_adapter_head"],
        "independent_review_id": authority["independent_review_id"],
        "execution_bundle_sha256": authority["execution_bundle_sha256"],
    }


def load_and_validate_execution_authority(
    path: Path,
    *,
    authority_sha256: str,
    expected_reviewed_adapter_head: str,
    expected_independent_review_id: int,
    exact_manifest: dict[str, Any],
) -> dict[str, Any]:
    authority, data = load_json_object_bytes(path)
    validate_execution_authority(
        authority,
        authority_sha256=authority_sha256,
        authority_bytes=data,
        expected_reviewed_adapter_head=expected_reviewed_adapter_head,
        expected_independent_review_id=expected_independent_review_id,
        exact_manifest=exact_manifest,
    )
    return authority
