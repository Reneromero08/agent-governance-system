#!/usr/bin/env python3
"""Independent protocol oracle for the hardware-absent V13 adapter gate.

This file imports no project module and performs no QEMU or hardware action.  It
models a frozen fail-closed protocol truth table and a deliberately separate
offline fixture domain.  Symbolic standard vectors can exercise parsing and
appraisal logic; they cannot establish production trust, a physical sample, a
campaign certificate, carrier custody, restoration, reuse, or advantage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn


SCHEMA = "M271_V13_HARDWARE_ADAPTER_GATE_SEPARATE_REFERENCE_V1"
PROTOCOL_VERSION = 13
MINIMUM_FIRMWARE_EPOCH = 7
PRODUCTION_DOMAIN = "PHASE_QEMU_V13_HARDWARE_GATE_PRODUCTION_V1"
FIXTURE_DOMAIN = "PHASE_QEMU_V13_OFFLINE_STANDARD_VECTOR_V1"
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
M257_GUARDRAIL = (
    "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_COUNTED_AS_A_PHASE_RESOURCE"
)

CLAIM = (
    "COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_GATE_"
    "REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_AND_TEST_FIXTURE_"
    "SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_CAMPAIGN_STATISTICAL_"
    "CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_COMPLETED_DISPATCHED_ATTEMPT_"
    "TERMINAL_ACK_THEN_SPENT"
)
CEILING = (
    "HARDWARE_ABSENT_PROTOCOL_CONFORMANCE_ONLY_NO_AUTHENTICATED_LIVE_DEVICE_SESSION_"
    "NO_PHYSICAL_SAMPLE_NO_CAMPAIGN_STATISTICAL_CERTIFICATE_NO_CUSTODY_RETURN_"
    "RESTORATION_REUSE_ADVANTAGE_OR_M257_ESCAPE"
)
SCOPE = (
    "COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_ABSENCE_AND_"
    "FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY"
)
DISPOSITION = (
    "V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_"
    "CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_FIXTURE_DOMAIN_"
    "SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_OUTPUT_DIRECT_EQUAL_ACCESS_"
    "PROTOCOL_COMPARATOR_CONTROLS_AND_M257_REMAINS_INTACT"
)
SUCCESSOR = (
    "USER_AUTHORIZED_PINNED_DEVICE_ENROLLMENT_FOLLOWED_BY_A_PREREGISTERED_BLINDED_"
    "DUAL_RAIL_DISPERSIVE_CAPTURE_CAMPAIGN_WITH_DEVICE_SIGNED_RAW_MANIFESTS_"
    "INDEPENDENT_MEASUREMENT_AND_FAMILYWISE_STATISTICAL_VALIDATION_BEHIND_THE_COMMON_"
    "PHASE_QEMU_BACKEND"
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FAIL_CLOSED {message}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def baseline_production_vector(vector_id: str) -> dict[str, Any]:
    return {
        "vector_id": vector_id,
        "attempt_id": f"symbolic-attempt::{vector_id}",
        "execution_domain": "PRODUCTION",
        "credential_domain": PRODUCTION_DOMAIN,
        "device_present": True,
        "device_id": "symbolic-enrolled-device-01",
        "enrollment_record_present": True,
        "attestation_present": True,
        "attestation_appraisal_valid": True,
        "session_nonce": 42,
        "last_accepted_nonce": 41,
        "replay_cache_contains_attempt": False,
        "protocol_version": PROTOCOL_VERSION,
        "firmware_epoch": MINIMUM_FIRMWARE_EPOCH,
        "sequence": 1,
        "expected_sequence": 1,
        "transcript_hash_matches": True,
        "device_signature_valid": True,
        "calibration_schema_valid": True,
        "resource_schema_valid": True,
        "transport_event": "NONE",
    }


TRUTH_TABLE_SPEC: tuple[dict[str, Any], ...] = (
    {
        "id": "absent",
        "failure": "ADAPTER_ABSENT",
        "dispatched": False,
        "override": {"device_present": False},
    },
    {
        "id": "unenrolled",
        "failure": "DEVICE_UNENROLLED",
        "dispatched": False,
        "override": {"enrollment_record_present": False},
    },
    {
        "id": "unattested",
        "failure": "ATTESTATION_MISSING_OR_INVALID",
        "dispatched": False,
        "override": {"attestation_present": False, "attestation_appraisal_valid": False},
    },
    {
        "id": "stale_nonce",
        "failure": "STALE_NONCE",
        "dispatched": False,
        "override": {"session_nonce": 41},
    },
    {
        "id": "replay",
        "failure": "REPLAY_DETECTED",
        "dispatched": False,
        "override": {"replay_cache_contains_attempt": True},
    },
    {
        "id": "downgraded_protocol",
        "failure": "PROTOCOL_DOWNGRADE",
        "dispatched": False,
        "override": {"protocol_version": PROTOCOL_VERSION - 1},
    },
    {
        "id": "downgraded_firmware",
        "failure": "FIRMWARE_DOWNGRADE",
        "dispatched": False,
        "override": {"firmware_epoch": MINIMUM_FIRMWARE_EPOCH - 1},
    },
    {
        "id": "fixture_credential_in_production",
        "failure": "FIXTURE_CREDENTIAL_FORBIDDEN_IN_PRODUCTION",
        "dispatched": False,
        "override": {"credential_domain": FIXTURE_DOMAIN},
    },
    {
        "id": "bad_signature",
        "failure": "BAD_DEVICE_SIGNATURE",
        "dispatched": False,
        "override": {"device_signature_valid": False},
    },
    {
        "id": "bad_hash",
        "failure": "BAD_TRANSCRIPT_HASH",
        "dispatched": False,
        "override": {"transcript_hash_matches": False},
    },
    {
        "id": "bad_sequence",
        "failure": "BAD_SEQUENCE",
        "dispatched": False,
        "override": {"sequence": 2},
    },
    {
        "id": "bad_calibration",
        "failure": "BAD_CALIBRATION_SCHEMA",
        "dispatched": False,
        "override": {"calibration_schema_valid": False},
    },
    {
        "id": "bad_resource_schema",
        "failure": "BAD_RESOURCE_SCHEMA",
        "dispatched": False,
        "override": {"resource_schema_valid": False},
    },
    {
        "id": "timeout",
        "failure": "TRANSPORT_TIMEOUT",
        "dispatched": True,
        "override": {"transport_event": "TIMEOUT"},
    },
    {
        "id": "reset",
        "failure": "RESET_DURING_ATTEMPT",
        "dispatched": True,
        "override": {"transport_event": "RESET"},
    },
    {
        "id": "migration",
        "failure": "MIGRATION_DURING_ATTEMPT",
        "dispatched": True,
        "override": {"transport_event": "MIGRATION"},
    },
    {
        "id": "disconnect",
        "failure": "DISCONNECT_DURING_ATTEMPT",
        "dispatched": True,
        "override": {"transport_event": "DISCONNECT"},
    },
)


def failure_receipt(
    vector: Mapping[str, Any], failure_code: str, dispatched: bool
) -> dict[str, Any]:
    destructive_sham = vector["transport_event"] in {"RESET", "MIGRATION", "DISCONNECT"}
    return {
        "receipt_schema": "M271_SYMBOLIC_FAILURE_RECEIPT_V1",
        "attempt_id": vector["attempt_id"],
        "device_id": vector["device_id"],
        "failure_code": failure_code,
        "dispatched": dispatched,
        "protocol_version": vector["protocol_version"],
        "firmware_epoch": vector["firmware_epoch"],
        "session_nonce": vector["session_nonce"],
        "sequence": vector["sequence"],
        "transcript_hash_status": (
            "MATCH" if vector["transcript_hash_matches"] else "MISMATCH"
        ),
        "signature_status": "VALID" if vector["device_signature_valid"] else "INVALID",
        "calibration_status": (
            "SCHEMA_VALID" if vector["calibration_schema_valid"] else "SCHEMA_INVALID"
        ),
        "resource_status": (
            "SCHEMA_VALID" if vector["resource_schema_valid"] else "SCHEMA_INVALID"
        ),
        "failure_state_committed": "SHAM" if destructive_sham else "FAILED",
        "destructive_sham": destructive_sham,
        "ack_required": not destructive_sham,
        "acknowledgement_action": None if destructive_sham else "ACK",
        "terminal_state_after_ack": None if destructive_sham else "SPENT",
        "terminal_state": "SHAM" if destructive_sham else "SPENT",
        "begin_reuse_authorized": False,
        "production_trust_established": False,
        "physical_output_present": False,
        "transaction_receipt_class": "NONE",
        "authenticated_physical_sample_present": False,
        "campaign_certificate_class": "NONE",
        "campaign_statistical_certificate_present": False,
        "physical_measurements": None,
        "physical_resources": None,
    }


def appraise_production_vector(vector: Mapping[str, Any]) -> dict[str, Any]:
    if vector["execution_domain"] != "PRODUCTION":
        fail("production_comparator_received_nonproduction_vector")

    predispatch_checks = (
        (not vector["device_present"], "ADAPTER_ABSENT"),
        (not vector["enrollment_record_present"], "DEVICE_UNENROLLED"),
        (
            not vector["attestation_present"] or not vector["attestation_appraisal_valid"],
            "ATTESTATION_MISSING_OR_INVALID",
        ),
        (vector["session_nonce"] <= vector["last_accepted_nonce"], "STALE_NONCE"),
        (vector["replay_cache_contains_attempt"], "REPLAY_DETECTED"),
        (vector["protocol_version"] < PROTOCOL_VERSION, "PROTOCOL_DOWNGRADE"),
        (vector["firmware_epoch"] < MINIMUM_FIRMWARE_EPOCH, "FIRMWARE_DOWNGRADE"),
        (
            vector["credential_domain"] != PRODUCTION_DOMAIN,
            "FIXTURE_CREDENTIAL_FORBIDDEN_IN_PRODUCTION",
        ),
        (not vector["device_signature_valid"], "BAD_DEVICE_SIGNATURE"),
        (not vector["transcript_hash_matches"], "BAD_TRANSCRIPT_HASH"),
        (vector["sequence"] != vector["expected_sequence"], "BAD_SEQUENCE"),
        (not vector["calibration_schema_valid"], "BAD_CALIBRATION_SCHEMA"),
        (not vector["resource_schema_valid"], "BAD_RESOURCE_SCHEMA"),
    )
    for rejected, failure_code in predispatch_checks:
        if rejected:
            return failure_receipt(vector, failure_code, dispatched=False)

    transport_failure = {
        "TIMEOUT": "TRANSPORT_TIMEOUT",
        "RESET": "RESET_DURING_ATTEMPT",
        "MIGRATION": "MIGRATION_DURING_ATTEMPT",
        "DISCONNECT": "DISCONNECT_DURING_ATTEMPT",
    }.get(vector["transport_event"])
    if transport_failure is not None:
        return failure_receipt(vector, transport_failure, dispatched=True)
    fail("unfrozen_production_accept_path")


def offline_standard_vectors() -> list[dict[str, Any]]:
    common = {
        "execution_domain": "OFFLINE_STANDARD_VECTOR",
        "credential_domain": FIXTURE_DOMAIN,
        "header_schema": "M271_SYMBOLIC_ADAPTER_HEADER_V1",
        "protocol_version": PROTOCOL_VERSION,
        "firmware_epoch": MINIMUM_FIRMWARE_EPOCH,
        "signature_encoding": "SYMBOLIC_STANDARD_VECTOR_SIGNATURE",
        "device_signature_valid": True,
        "transcript_hash_matches": True,
        "sequence_valid": True,
        "calibration_schema_valid": True,
        "resource_schema_valid": True,
    }
    specs = (
        ("offline_fixture_valid", {}, True),
        ("offline_fixture_bad_signature", {"device_signature_valid": False}, False),
        ("offline_fixture_bad_resource_schema", {"resource_schema_valid": False}, False),
    )
    vectors: list[dict[str, Any]] = []
    for vector_id, override, expected_appraisal in specs:
        vector = dict(common)
        vector.update(override)
        required_fields = set(common)
        parser_valid = required_fields.issubset(vector)
        appraisal_valid = (
            parser_valid
            and vector["execution_domain"] == "OFFLINE_STANDARD_VECTOR"
            and vector["credential_domain"] == FIXTURE_DOMAIN
            and vector["protocol_version"] == PROTOCOL_VERSION
            and vector["firmware_epoch"] >= MINIMUM_FIRMWARE_EPOCH
            and vector["device_signature_valid"]
            and vector["transcript_hash_matches"]
            and vector["sequence_valid"]
            and vector["calibration_schema_valid"]
            and vector["resource_schema_valid"]
        )
        vectors.append(
            {
                "vector_id": vector_id,
                "symbolic_fields": vector,
                "parser_valid": parser_valid,
                "appraisal_valid": appraisal_valid,
                "expected_appraisal_valid": expected_appraisal,
                "comparator_match": appraisal_valid == expected_appraisal,
                "production_trust_established": False,
                "eligible_for_production_dispatch": False,
                "authenticated_physical_sample_present": False,
                "campaign_statistical_certificate_present": False,
            }
        )
    return vectors


def build_payload() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def require(identifier: str, condition: bool) -> None:
        if not condition:
            fail(identifier)
        checks.append({"id": identifier, "pass": True})

    truth_table: list[dict[str, Any]] = []
    for spec in TRUTH_TABLE_SPEC:
        vector = baseline_production_vector(spec["id"])
        vector.update(spec["override"])
        receipt = appraise_production_vector(vector)
        truth_table.append(
            {
                "case": spec["id"],
                "symbolic_request_fields": vector,
                "expected_failure": spec["failure"],
                "expected_dispatched": spec["dispatched"],
                "direct_protocol_comparator_match": (
                    receipt["failure_code"] == spec["failure"]
                    and receipt["dispatched"] == spec["dispatched"]
                ),
                "symbolic_failure_receipt": receipt,
            }
        )

    require("truth_table_has_exactly_seventeen_cases", len(truth_table) == 17)
    require(
        "truth_table_case_order_is_frozen",
        [case["case"] for case in truth_table]
        == [spec["id"] for spec in TRUTH_TABLE_SPEC],
    )
    require(
        "direct_protocol_comparator_matches_every_case",
        all(case["direct_protocol_comparator_match"] for case in truth_table),
    )
    nondestructive_cases = [
        case
        for case in truth_table
        if not case["symbolic_failure_receipt"]["destructive_sham"]
    ]
    destructive_cases = [
        case
        for case in truth_table
        if case["symbolic_failure_receipt"]["destructive_sham"]
    ]
    require(
        "every_nondestructive_case_commits_failed_then_ack_then_spent",
        all(
            case["symbolic_failure_receipt"]["failure_state_committed"] == "FAILED"
            and case["symbolic_failure_receipt"]["ack_required"]
            and case["symbolic_failure_receipt"]["acknowledgement_action"] == "ACK"
            and case["symbolic_failure_receipt"]["terminal_state_after_ack"] == "SPENT"
            and case["symbolic_failure_receipt"]["terminal_state"] == "SPENT"
            for case in nondestructive_cases
        ),
    )
    require(
        "reset_migration_disconnect_are_destructive_sham_without_ack",
        [case["case"] for case in destructive_cases]
        == ["reset", "migration", "disconnect"]
        and all(
            case["symbolic_failure_receipt"]["failure_state_committed"] == "SHAM"
            and not case["symbolic_failure_receipt"]["ack_required"]
            and case["symbolic_failure_receipt"]["acknowledgement_action"] is None
            and case["symbolic_failure_receipt"]["terminal_state_after_ack"] is None
            and case["symbolic_failure_receipt"]["terminal_state"] == "SHAM"
            for case in destructive_cases
        ),
    )
    dispatched_cases = [
        case for case in truth_table if case["symbolic_failure_receipt"]["dispatched"]
    ]
    require(
        "only_timeout_reset_migration_disconnect_are_dispatched",
        [case["case"] for case in dispatched_cases]
        == ["timeout", "reset", "migration", "disconnect"],
    )
    nondestructively_completed_dispatched_cases = [
        case
        for case in dispatched_cases
        if not case["symbolic_failure_receipt"]["destructive_sham"]
    ]
    require(
        "every_nondestructively_completed_dispatched_attempt_is_acknowledged_then_spent",
        [case["case"] for case in nondestructively_completed_dispatched_cases]
        == ["timeout"]
        and all(
            case["symbolic_failure_receipt"]["failure_state_committed"] == "FAILED"
            and case["symbolic_failure_receipt"]["acknowledgement_action"] == "ACK"
            and case["symbolic_failure_receipt"]["terminal_state_after_ack"] == "SPENT"
            and case["symbolic_failure_receipt"]["terminal_state"] == "SPENT"
            for case in nondestructively_completed_dispatched_cases
        ),
    )
    require(
        "every_destructively_interrupted_dispatched_attempt_is_sham_without_ack",
        all(
            case["symbolic_failure_receipt"]["terminal_state"] == "SHAM"
            and not case["symbolic_failure_receipt"]["ack_required"]
            for case in destructive_cases
        ),
    )
    require(
        "no_case_publishes_physical_output_or_campaign_certificate",
        all(
            not case["symbolic_failure_receipt"]["physical_output_present"]
            and not case["symbolic_failure_receipt"][
                "authenticated_physical_sample_present"
            ]
            and not case["symbolic_failure_receipt"][
                "campaign_statistical_certificate_present"
            ]
            for case in truth_table
        ),
    )
    require(
        "no_failure_authorizes_begin_reuse",
        all(
            not case["symbolic_failure_receipt"]["begin_reuse_authorized"]
            for case in truth_table
        ),
    )

    offline_vectors = offline_standard_vectors()
    require("offline_fixture_vector_count_is_three", len(offline_vectors) == 3)
    require(
        "offline_fixture_comparator_matches_all_vectors",
        all(vector["comparator_match"] for vector in offline_vectors),
    )
    require(
        "only_valid_fixture_vector_passes_offline_appraisal",
        [vector["vector_id"] for vector in offline_vectors if vector["appraisal_valid"]]
        == ["offline_fixture_valid"],
    )
    require(
        "offline_vectors_never_establish_production_trust_or_physical_evidence",
        all(
            not vector["production_trust_established"]
            and not vector["eligible_for_production_dispatch"]
            and not vector["authenticated_physical_sample_present"]
            and not vector["campaign_statistical_certificate_present"]
            for vector in offline_vectors
        ),
    )

    evidence_layers = {
        "per_transaction_artifact": {
            "class": "AUTHENTICATED_PHYSICAL_SAMPLE",
            "present_in_m271": False,
            "required_properties_if_future": [
                "LIVE_ENROLLED_DEVICE_IDENTITY",
                "FRESH_ATTESTED_SESSION",
                "DEVICE_SIGNED_SAMPLE_MANIFEST",
                "BOUNDED_CALIBRATION_AND_RESOURCE_FIELDS",
            ],
            "does_not_by_itself_establish_campaign_statistics": True,
        },
        "independent_campaign_artifact": {
            "class": "STATISTICAL_ONLY",
            "present_in_m271": False,
            "required_properties_if_future": [
                "PREREGISTERED_CAMPAIGN",
                "BLINDED_CAPTURE",
                "DEVICE_SIGNED_RAW_MANIFESTS",
                "INDEPENDENT_MEASUREMENT",
                "FAMILYWISE_STATISTICAL_VALIDATION",
            ],
            "does_not_establish_exact_physical_return": True,
        },
        "artifacts_are_distinct": True,
        "neither_artifact_present": True,
    }
    require(
        "transaction_sample_and_campaign_certificate_are_distinct_and_absent",
        evidence_layers["artifacts_are_distinct"]
        and evidence_layers["neither_artifact_present"]
        and not evidence_layers["per_transaction_artifact"]["present_in_m271"]
        and not evidence_layers["independent_campaign_artifact"]["present_in_m271"],
    )

    direct_protocol_comparator = {
        "name": "DIRECT_PUBLIC_PROTOCOL_STATE_MACHINE_COMPARATOR",
        "input_fields": [
            "presence",
            "enrollment",
            "attestation",
            "nonce_freshness",
            "replay_state",
            "protocol_version",
            "firmware_epoch",
            "credential_domain",
            "signature_status",
            "transcript_hash_status",
            "sequence",
            "calibration_schema",
            "resource_schema",
            "transport_event",
        ],
        "production_hardware_access": False,
        "case_count": len(truth_table),
        "all_truth_table_outcomes_identical": all(
            case["direct_protocol_comparator_match"] for case in truth_table
        ),
        "physical_output_compared": False,
        "unique_protocol_advantage": False,
        "total_resource_advantage": "UNDETERMINED",
        "advantage_claim": False,
    }
    require(
        "direct_comparator_has_full_protocol_parity_without_advantage",
        direct_protocol_comparator["case_count"] == 17
        and direct_protocol_comparator["all_truth_table_outcomes_identical"]
        and not direct_protocol_comparator["production_hardware_access"]
        and not direct_protocol_comparator["unique_protocol_advantage"]
        and not direct_protocol_comparator["advantage_claim"],
    )

    receipt_field_contract = {
        "symbolic_failure_receipt_fields": sorted(
            truth_table[0]["symbolic_failure_receipt"].keys()
        ),
        "future_authenticated_physical_sample_fields": [
            "attested_device_id",
            "calibration_manifest_hash",
            "device_signature",
            "firmware_epoch",
            "physical_measurements",
            "protocol_version",
            "resource_manifest_hash",
            "sample_id",
            "sequence",
            "session_nonce",
            "transcript_hash",
        ],
        "future_campaign_statistical_certificate_fields": [
            "blinding_manifest_hash",
            "campaign_id",
            "device_signed_raw_manifest_set_hash",
            "familywise_method",
            "independent_measurement_dataset_hash",
            "preregistration_hash",
            "statistical_result",
        ],
        "future_fields_are_schema_only_not_present_evidence": True,
    }
    require(
        "future_sample_and_campaign_field_schemas_do_not_overlap_as_artifacts",
        set(receipt_field_contract["future_authenticated_physical_sample_fields"])
        != set(receipt_field_contract["future_campaign_statistical_certificate_fields"])
        and receipt_field_contract["future_fields_are_schema_only_not_present_evidence"],
    )

    unknown_physical_resources = {
        name: {
            "status": "UNKNOWN",
            "value": None,
            "unit": unit,
            "reason": "NO_AUTHENTICATED_LIVE_HARDWARE_SESSION_OR_PHYSICAL_SAMPLE",
        }
        for name, unit in (
            ("carrier_preparation_attempts", "count"),
            ("control_energy", "J"),
            ("device_energy", "J"),
            ("device_wall_time", "ns"),
            ("environmental_support_energy", "J"),
            ("physical_loss_probability", "dimensionless"),
        )
    }
    require(
        "unknown_physical_resources_are_null_not_zero",
        all(
            item["status"] == "UNKNOWN"
            and item["value"] is None
            and bool(item["reason"])
            for item in unknown_physical_resources.values()
        ),
    )

    classification = {
        "reference_layer": "INDEPENDENT_SYMBOLIC_FIXED_OFFLINE_SELECTOR_REFERENCE",
        "qemu_execution_performed": False,
        "hardware_execution_performed": False,
        "live_device_session_present": False,
        "production_authentication_established": False,
        "physical_output_present": False,
        "physical_evidence_class": "NONE",
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "physical_return_claim": False,
        "same_carrier_claim": False,
        "custody_claim": False,
        "reuse_claim": False,
        "architecture_promotion_claim_from_reference": False,
    }
    require(
        "reference_makes_no_qemu_hardware_or_restoration_claim",
        not classification["qemu_execution_performed"]
        and not classification["hardware_execution_performed"]
        and not classification["live_device_session_present"]
        and classification["restoration_classification"] == "NO_RESTORATION_CLAIM",
    )

    m257 = {
        "guardrail": M257_GUARDRAIL,
        "status": "INTACT",
        "escape_established": False,
    }
    require(
        "m257_remains_intact",
        m257["status"] == "INTACT" and not m257["escape_established"],
    )

    nonclaims = [
        "NO_ARCHITECTURE_PROMOTION_CLAIM_FROM_REFERENCE",
        "NO_AUTHENTICATED_LIVE_DEVICE_SESSION_CLAIM",
        "NO_AUTHENTICATED_PHYSICAL_SAMPLE_CLAIM",
        "NO_CAMPAIGN_STATISTICAL_CERTIFICATE_CLAIM",
        "NO_CARRIER_CUSTODY_CLAIM",
        "NO_CRYPTOGRAPHIC_SECURITY_CLAIM_FROM_SYMBOLIC_VECTORS",
        "NO_EXACT_PHYSICAL_RETURN_CLAIM",
        "NO_HARDWARE_EXECUTION_CLAIM",
        "NO_M257_ESCAPE_CLAIM",
        "NO_PHYSICAL_OUTPUT_CLAIM",
        "NO_PHYSICAL_RESOURCE_ADVANTAGE_CLAIM",
        "NO_PRODUCTION_TRUST_FROM_FIXTURE_VECTORS",
        "NO_QEMU_EXECUTION_CLAIM",
        "NO_RESTORATION_CLAIM",
        "NO_REUSE_CLAIM",
        "NO_SAME_CARRIER_CLAIM",
    ]
    require("nonclaims_are_unique_and_sorted", nonclaims == sorted(set(nonclaims)))

    authority = {
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "scope": SCOPE,
        "disposition": DISPOSITION,
        "successor": SUCCESSOR,
    }
    require(
        "authority_restoration_is_no_restoration_claim",
        authority["restoration_classification"] == "NO_RESTORATION_CLAIM",
    )

    payload = {
        "authority": authority,
        "frozen_protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "minimum_firmware_epoch": MINIMUM_FIRMWARE_EPOCH,
            "production_domain": PRODUCTION_DOMAIN,
            "fixture_domain": FIXTURE_DOMAIN,
            "production_and_fixture_domains_are_distinct": PRODUCTION_DOMAIN
            != FIXTURE_DOMAIN,
            "truth_table_case_count": len(truth_table),
        },
        "production_failure_truth_table": truth_table,
        "offline_standard_vectors": offline_vectors,
        "offline_vector_authority": (
            "PARSER_AND_APPRAISAL_LOGIC_ONLY_NEVER_PRODUCTION_TRUST_OR_PHYSICAL_EVIDENCE"
        ),
        "evidence_layers": evidence_layers,
        "receipt_field_contract": receipt_field_contract,
        "direct_protocol_comparator": direct_protocol_comparator,
        "resource_accounting": {
            "unknown_physical_resources": unknown_physical_resources,
            "unknown_value_encoding": "NULL_NEVER_NUMERIC_ZERO",
            "physical_resource_total": "UNKNOWN",
            "resource_advantage_claim": False,
        },
        "classification": classification,
        "m257": m257,
        "nonclaims": nonclaims,
        "determinism": {
            "canonical_encoding": "JSON_ASCII_SORTED_KEYS_COMPACT_NO_NAN",
            "self_run_count": 2,
            "two_self_runs_required_byte_identical": True,
        },
        "checks": checks,
        "check_count": len(checks),
        "all_checks_pass": True,
    }
    if payload["check_count"] != len(checks):
        fail("check_count_changed_before_output")
    return payload


def main() -> None:
    first_payload = build_payload()
    second_payload = build_payload()
    first_bytes = canonical_bytes(first_payload)
    second_bytes = canonical_bytes(second_payload)
    if first_bytes != second_bytes:
        fail("two_self_runs_are_not_byte_identical")

    document = {
        "schema": SCHEMA,
        "source_sha256": sha256_hex(Path(__file__).read_bytes()),
        "payload_sha256": sha256_hex(first_bytes),
        "two_self_runs_byte_identical": True,
        "payload": first_payload,
    }
    print(canonical_bytes(document).decode("ascii"))


if __name__ == "__main__":
    main()
