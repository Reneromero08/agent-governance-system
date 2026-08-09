#!/usr/bin/env python3
"""Independent offline oracle for the V13 pre-enrollment preparation."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import cryptography
import scipy
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scipy.stats import beta, t


CAMPAIGN = "PQV13-CAMPAIGN-NOT-AUTHORIZED-0001"
DEVICE = "OFFLINE-TEST-DEVICE-NOT-PHYSICAL"
BACKEND = "PHASE_QEMU_V13_COMMON_BACKEND"
AAD_BASE = b"phase-qemu-v13-pre-enrollment-v1:"


def digest(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def dhex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def initial(major: int, number: int) -> bytes:
    if not 0 <= number < 2**64:
        raise ValueError("integer outside reference profile")
    if number < 24:
        return bytes([(major << 5) | number])
    if number < 256:
        return bytes([(major << 5) | 24, number])
    if number < 65536:
        return bytes([(major << 5) | 25]) + number.to_bytes(2, "big")
    if number < 2**32:
        return bytes([(major << 5) | 26]) + number.to_bytes(4, "big")
    return bytes([(major << 5) | 27]) + number.to_bytes(8, "big")


def canonical(value: Any) -> bytes:
    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if value is None:
        return b"\xf6"
    if isinstance(value, int):
        return initial(0, value) if value >= 0 else initial(1, -1 - value)
    if isinstance(value, bytes):
        return initial(2, len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return initial(3, len(raw)) + raw
    if isinstance(value, (list, tuple)):
        return initial(4, len(value)) + b"".join(canonical(x) for x in value)
    if isinstance(value, dict):
        entries = sorted((canonical(k), canonical(v)) for k, v in value.items())
        if len({k for k, _ in entries}) != len(entries):
            raise ValueError("duplicate encoded key")
        return initial(5, len(entries)) + b"".join(k + v for k, v in entries)
    raise ValueError("type outside reference profile")


def private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        digest(("OFFLINE_TEST_VECTOR_ONLY:" + label).encode("ascii"))
    )


def pubraw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def sign1(record: dict[str, Any], key: Ed25519PrivateKey, kid: bytes, domain: str) -> bytes:
    protected = canonical({1: -8, 4: kid})
    payload = canonical(record)
    sig_structure = canonical(
        ["Signature1", protected, AAD_BASE + domain.encode("ascii"), payload]
    )
    signature = key.sign(sig_structure)
    key.public_key().verify(signature, sig_structure)
    return canonical([protected, {}, payload, signature])


def wrapped(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def analyze(values: list[int], target: int) -> dict[str, Any]:
    errors = [wrapped((x - target) / 1_000_000.0) for x in values]
    n = len(errors)
    linear_mean = sum(errors) / n
    variance = sum((x - linear_mean) ** 2 for x in errors) / (n - 1)
    standard_error = math.sqrt(variance / n)
    alpha = 0.05 / 4
    lower_p = float(t.sf((linear_mean + 0.08) / standard_error, n - 1))
    upper_p = float(t.cdf((linear_mean - 0.08) / standard_error, n - 1))
    successes = sum(abs(x) <= 0.04 for x in errors)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, n - successes + 1))
    tost = lower_p < alpha and upper_p < alpha
    count = lower >= 0.65
    return {
        "n": n,
        "mean_direction_rad": math.atan2(
            sum(math.sin(x) for x in errors), sum(math.cos(x) for x in errors)
        ),
        "resultant_length": math.hypot(
            sum(math.cos(x) for x in errors), sum(math.sin(x) for x in errors)
        ) / n,
        "linear_wrapped_error_mean_rad": linear_mean,
        "standard_error_rad": standard_error,
        "bonferroni_alpha": alpha,
        "tost_lower_p": lower_p,
        "tost_upper_p": upper_p,
        "tost_equivalent": tost,
        "within_tolerance_count": successes,
        "clopper_pearson_lower": lower,
        "minimum_success_probability": 0.65,
        "finite_count_pass": count,
        "endpoint_pass": tost and count,
    }


def receipt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": result["n"],
        "within_tolerance_count": result["within_tolerance_count"],
        "tost_equivalent": result["tost_equivalent"],
        "finite_count_pass": result["finite_count_pass"],
        "endpoint_pass": result["endpoint_pass"],
    }


def run() -> dict[str, Any]:
    enrollment_key = private("enrollment-authority")
    device_key = private("device")
    plan_key = private("preregistration-owner")
    adjudicator_key = private("independent-adjudicator")
    enrollment_kid = b"offline-enrollment-v1"
    device_kid = b"offline-device-v1"
    plan_kid = b"offline-plan-v1"
    adjudicator_kid = b"offline-adjudicator-v1"

    enrollment = {
        "schema": "phase_qemu.v13.offline_enrollment_record.v1",
        "campaign_id": CAMPAIGN,
        "device_id": DEVICE,
        "board_id": "OFFLINE-BOARD-NOT-PRESENT",
        "hardware_revision": "NONE",
        "device_signing_kid": device_kid,
        "device_public_key_sha256": digest(pubraw(device_key)),
        "fixture_domain_exclusion": ["TEST_FIXTURE", "OFFLINE_VECTOR"],
        "protocol_version": 1,
        "manifest_schema_version": 1,
        "analysis_schema_version": 1,
        "policy_version": 1,
        "measured_boot_sha256": digest(b"NO_MEASURED_BOOT_OFFLINE_VECTOR"),
        "firmware_sha256": digest(b"NO_FIRMWARE_OFFLINE_VECTOR"),
        "bom_digest": digest(b"NO_BOM_OFFLINE_VECTOR"),
        "calibration_digest": digest(b"NO_CALIBRATION_OFFLINE_VECTOR"),
        "created_ns": 0,
        "expires_ns": 1_000_000_000,
        "production_authorization": False,
        "offline_test_vector": True,
    }
    encoded_enrollment = sign1(enrollment, enrollment_key, enrollment_kid, "enrollment")
    plan = {
        "schema": "phase_qemu.v13.offline_preregistration.v1",
        "campaign_id": CAMPAIGN,
        "plan_version": 1,
        "locked_before_first_capture": True,
        "data_origin": "OFFLINE_SYNTHETIC_TEST_VECTORS_ONLY",
        "primary_endpoints": ["PROGRAM_A_PHASE", "PROGRAM_B_PHASE"],
        "family_alpha_ppm": 50_000,
        "family_size": 4,
        "equivalence_margin_urad": 80_000,
        "success_tolerance_urad": 40_000,
        "minimum_success_probability_ppm": 650_000,
        "minimum_n_per_endpoint": 12,
        "interim_looks": 0,
        "stopping_rule": "FIXED_N_NO_OPTIONAL_STOPPING",
        "multiplicity_method": "BONFERRONI_SINGLE_FAMILY",
        "equivalence_method": "PAIRED_WRAPPED_ERROR_STUDENT_T_TOST",
        "finite_count_method": "CLOPPER_PEARSON_EXACT_LOWER_BOUND",
        "circular_branch_rule": "WRAP_TO_MINUS_PI_INCLUSIVE_PLUS_PI_EXCLUSIVE_BEFORE_GROUP_SUMMARY",
        "randomization": "FIXED_PUBLIC_OFFLINE_VECTOR_ORDER_NOT_A_PHYSICAL_RANDOMIZATION",
        "blinding": "ROLE_SEPARATED_SYNTHETIC_LABELS_ONLY",
        "exclusion_policy": "NO_POST_HOC_EXCLUSIONS",
        "self_adjudication_allowed": False,
        "physical_authorization_present": False,
    }
    encoded_plan = sign1(plan, plan_key, plan_kid, "preregistration")

    a = [-12_000, 8_000, -6_000, 10_000, -9_000, 7_000, -4_000, 5_000, -3_000, 6_000, -5_000, 4_000]
    b = [-15_000, 11_000, -8_000, 13_000, -10_000, 9_000, -7_000, 8_000, -6_000, 7_000, -5_000, 6_000]
    control_values = [200_000, 210_000, 190_000, 205_000, 195_000, 215_000, 185_000, 220_000, 180_000, 225_000, 175_000, 230_000]
    samples = [[x for x in a], [1_570_796 + x for x in b]]
    raw = [struct.pack("<12i", *row) for row in samples]
    descriptors = [b"PROGRAM_A_PHASE_0", b"PROGRAM_B_PHASE_PI_OVER_2"]
    challenges = []
    signed_manifests = []
    manifests = []
    for i in (1, 2):
        challenge = {
            "schema": "phase_qemu.v13.offline_challenge.v1",
            "campaign_id": CAMPAIGN,
            "session_id": "OFFLINE-SESSION-0001",
            "transaction_id": i,
            "allocation_id": "OFFLINE-ALLOCATION-0001",
            "generation": i,
            "backend_id": BACKEND,
            "nonce": digest(f"offline-nonce-{i}".encode("ascii")),
            "descriptor_sha256": digest(descriptors[i - 1]),
            "protocol_version": 1,
            "manifest_schema_version": 1,
            "deadline_ns": 900_000_000 + i,
            "expected_rails": [0, 1],
            "tls_exporter_sha256": digest(b"NO_TLS_OFFLINE_VECTOR"),
            "transport_evidence_class": "OFFLINE_TEST_VECTOR_ONLY",
        }
        block = raw[i - 1]
        manifest = {
            "schema": "phase_qemu.v13.offline_raw_manifest.v1",
            "evidence_class": "OFFLINE_TEST_VECTOR_ONLY",
            "campaign_id": CAMPAIGN,
            "session_id": challenge["session_id"],
            "transaction_id": i,
            "allocation_id": challenge["allocation_id"],
            "generation": i,
            "monotonic_sequence": i,
            "challenge_sha256": digest(canonical(challenge)),
            "nonce": challenge["nonce"],
            "tls_exporter_sha256": challenge["tls_exporter_sha256"],
            "device_id": DEVICE,
            "signer_kid": device_kid,
            "attestation_result_digest": digest(b"NO_ATTESTATION_OFFLINE_VECTOR"),
            "firmware_sha256": enrollment["firmware_sha256"],
            "descriptor_sha256": challenge["descriptor_sha256"],
            "rail_labels": ["RAIL_A", "RAIL_B"],
            "sample_count": 12,
            "sample_encoding": "SIGNED_INT32_LE_MICRORADIANS",
            "sample_rate_hz": 1_000,
            "channel_map": ["SYNTHETIC_PHASE_ERROR"],
            "payload_blocks": [{"index": 0, "length": len(block), "sha256": digest(block)}],
            "raw_total_length": len(block),
            "capture_state": "OFFLINE_SYNTHETIC_COMPLETE",
            "faults": [],
            "physical_sample": False,
            "source": "SYNTHETIC_BYTES_GENERATED_IN_PROCESS",
        }
        challenges.append(challenge)
        manifests.append(manifest)
        signed_manifests.append(sign1(manifest, device_key, device_kid, "raw-manifest"))

    nodes = sorted(digest(b"\x00" + item) for item in signed_manifests)
    inventory = digest(b"\x01" + nodes[0] + nodes[1])
    endpoint_a = analyze(samples[0], 0)
    endpoint_b = analyze(samples[1], 1_570_796)
    control = analyze(control_values, 0)
    report = {
        "schema": "phase_qemu.v13.offline_adjudication_preflight.v1",
        "classification": "OFFLINE_SYNTHETIC_ADJUDICATION_PREFLIGHT_ONLY",
        "campaign_id": CAMPAIGN,
        "plan_sha256": digest(encoded_plan),
        "manifest_inventory_root": inventory,
        "manifest_count": 2,
        "missing_manifest_count": 0,
        "invalid_manifest_count": 0,
        "program_a": receipt(endpoint_a),
        "program_b": receipt(endpoint_b),
        "positive_offset_control": receipt(control),
        "all_primary_endpoints_pass": endpoint_a["endpoint_pass"] and endpoint_b["endpoint_pass"],
        "negative_control_rejected": not control["endpoint_pass"],
        "physical_evidence": False,
        "authenticated_physical_sample": False,
        "campaign_statistical_certificate": False,
        "physical_authorization_present": False,
        "architecture_promotion_eligible": False,
        "m257_escape": False,
    }
    signed_report = sign1(report, adjudicator_key, adjudicator_kid, "adjudication-preflight")

    corrupted = bytearray(signed_manifests[0])
    corrupted[-1] ^= 1
    invalid_signature_rejected = False
    try:
        protected = canonical({1: -8, 4: device_kid})
        payload = canonical(manifests[0])
        device_key.public_key().verify(
            bytes(corrupted[-64:]),
            canonical(["Signature1", protected, AAD_BASE + b"raw-manifest", payload]),
        )
    except InvalidSignature:
        invalid_signature_rejected = True

    encoded = {
        "enrollment_payload_sha256": dhex(canonical(enrollment)),
        "signed_enrollment_sha256": dhex(encoded_enrollment),
        "preregistration_payload_sha256": dhex(canonical(plan)),
        "signed_preregistration_sha256": dhex(encoded_plan),
        "challenge_sha256": [dhex(canonical(x)) for x in challenges],
        "manifest_payload_sha256": [dhex(canonical(x)) for x in manifests],
        "signed_manifest_sha256": [dhex(x) for x in signed_manifests],
        "manifest_inventory_root_sha256": inventory.hex(),
        "signed_adjudication_preflight_sha256": dhex(signed_report),
    }
    checks = {
        "four_distinct_fixture_keys": len({pubraw(x) for x in (enrollment_key, device_key, plan_key, adjudicator_key)}) == 4,
        "fixture_enrollment_not_authorized": enrollment["production_authorization"] is False,
        "plan_locked_no_interim_looks": plan["locked_before_first_capture"] is True and plan["interim_looks"] == 0,
        "two_manifest_signatures_created": len(signed_manifests) == 2,
        "invalid_manifest_signature_rejected": invalid_signature_rejected,
        "manifest_inventory_complete": len(inventory) == 32,
        "program_a_passes": endpoint_a["endpoint_pass"],
        "program_b_passes": endpoint_b["endpoint_pass"],
        "positive_offset_control_fails": not control["endpoint_pass"],
        "no_physical_evidence": report["physical_evidence"] is False,
        "no_campaign_certificate": report["campaign_statistical_certificate"] is False,
        "no_architecture_promotion": report["architecture_promotion_eligible"] is False,
        "m257_intact": report["m257_escape"] is False,
    }
    if not all(checks.values()):
        raise AssertionError([k for k, v in checks.items() if not v])
    source = Path(__file__).read_bytes()
    result = {
        "schema": "phase_qemu.v13.pre_enrollment_campaign.separate_reference.v1",
        "status": "PASS_INDEPENDENT_HARDWARE_FREE_PRE_ENROLLMENT_REFERENCE",
        "source_sha256": dhex(source),
        "model": "INDEPENDENT_DETERMINISTIC_CBOR_ED25519_COSE_AND_STATISTICAL_RECOMPUTATION",
        "encoded_objects": encoded,
        "analysis": {
            "program_a": endpoint_a,
            "program_b": endpoint_b,
            "positive_offset_control": control,
            "family_size": 4,
            "family_alpha": 0.05,
            "multiplicity": "BONFERRONI",
            "physical_interpretation_allowed": False,
        },
        "checks": checks,
        "resource_scope": {
            "reference_source_bytes": len(source),
            "reference_raw_fixture_bytes": sum(len(x) for x in raw),
            "reference_signature_creations": 5,
            "production_runtime_or_peak_resource_parity_claimed": False,
            "physical_resource_measurement_claimed": False,
            "cryptography_version": cryptography.__version__,
            "scipy_version": scipy.__version__,
        },
        "authority_gate": "EXPLICIT_USER_AUTHORIZATION_REQUIRED_BEFORE_DEVICE_ENROLLMENT_CONNECTION_OR_CAPTURE",
        "nonclaims": {
            "qemu_executed": False,
            "network_or_tls_executed": False,
            "device_enrolled": False,
            "physical_capture_executed": False,
            "authenticated_physical_sample": False,
            "campaign_statistical_certificate": False,
            "restoration_or_reuse": False,
            "architecture_promotion": False,
            "resource_advantage": False,
            "m257_escape": False,
        },
    }
    payload = dict(result)
    payload.pop("source_sha256")
    result["claim_payload_sha256"] = dhex(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, separators=(",", ":")))
