#!/usr/bin/env python3
"""Hardware-free preparation for the user-authorized V13 physical campaign.

This module deliberately stops before enrollment, transport, capture, or a
physical/statistical claim.  It exercises a narrow deterministic CBOR and
COSE_Sign1 profile, transaction binding, manifest inventory, preregistration
locking, and an independent-adjudicator-shaped synthetic analysis report.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable

import cryptography
import scipy
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from scipy.stats import beta, t


SCHEMA = "phase_qemu.v13.pre_enrollment_campaign.v1"
STATUS = "PASS_HARDWARE_FREE_PRE_ENROLLMENT_SOFTWARE_CONFORMANCE"
CLASSIFICATION = "PRE_ENROLLMENT_SOFTWARE_CONFORMANCE_PREPARATION_OUTSIDE_PHYSICAL_EVIDENCE"
AUTHORITY_GATE = "EXPLICIT_USER_AUTHORIZATION_REQUIRED_BEFORE_DEVICE_ENROLLMENT_CONNECTION_OR_CAPTURE"
COSE_ALG_EDDSA = -8
EXTERNAL_AAD_PREFIX = b"phase-qemu-v13-pre-enrollment-v1:"
CAMPAIGN_ID = "PQV13-CAMPAIGN-NOT-AUTHORIZED-0001"
DEVICE_ID = "OFFLINE-TEST-DEVICE-NOT-PHYSICAL"
BACKEND_ID = "PHASE_QEMU_V13_COMMON_BACKEND"
UNKNOWN = "UNKNOWN_NO_HARDWARE_OR_LIVE_SESSION"


class CBORError(ValueError):
    pass


class VerificationError(ValueError):
    pass


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _head(major: int, value: int) -> bytes:
    if value < 0 or value >= 1 << 64:
        raise CBORError("CBOR integer or length outside uint64 profile")
    prefix = major << 5
    if value < 24:
        return bytes([prefix | value])
    if value < 1 << 8:
        return bytes([prefix | 24, value])
    if value < 1 << 16:
        return bytes([prefix | 25]) + value.to_bytes(2, "big")
    if value < 1 << 32:
        return bytes([prefix | 26]) + value.to_bytes(4, "big")
    return bytes([prefix | 27]) + value.to_bytes(8, "big")


def cbor_encode(value: Any) -> bytes:
    """Encode the deliberately small RFC 8949 deterministic profile."""
    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if value is None:
        return b"\xf6"
    if isinstance(value, int):
        return _head(0, value) if value >= 0 else _head(1, -1 - value)
    if isinstance(value, bytes):
        return _head(2, len(value)) + value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return _head(3, len(encoded)) + encoded
    if isinstance(value, (list, tuple)):
        return _head(4, len(value)) + b"".join(cbor_encode(x) for x in value)
    if isinstance(value, dict):
        pairs: list[tuple[bytes, bytes]] = []
        seen: set[bytes] = set()
        for key, item in value.items():
            key_bytes = cbor_encode(key)
            if key_bytes in seen:
                raise CBORError("duplicate deterministic map-key encoding")
            seen.add(key_bytes)
            pairs.append((key_bytes, cbor_encode(item)))
        pairs.sort(key=lambda pair: pair[0])
        return _head(5, len(pairs)) + b"".join(k + v for k, v in pairs)
    raise CBORError(f"unsupported CBOR profile type: {type(value).__name__}")


class _Decoder:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def _take(self, count: int) -> bytes:
        end = self.offset + count
        if count < 0 or end > len(self.data):
            raise CBORError("truncated CBOR")
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def _argument(self, additional: int) -> int:
        if additional < 24:
            return additional
        widths = {24: 1, 25: 2, 26: 4, 27: 8}
        if additional not in widths:
            raise CBORError("indefinite or reserved CBOR argument rejected")
        width = widths[additional]
        value = int.from_bytes(self._take(width), "big")
        minima = {1: 24, 2: 1 << 8, 4: 1 << 16, 8: 1 << 32}
        if value < minima[width]:
            raise CBORError("non-shortest CBOR argument rejected")
        return value

    def item(self) -> Any:
        initial = self._take(1)[0]
        major, additional = initial >> 5, initial & 31
        if major == 7:
            if additional == 20:
                return False
            if additional == 21:
                return True
            if additional == 22:
                return None
            raise CBORError("floats and unrecognized simple values rejected")
        if major == 6:
            raise CBORError("CBOR tags are outside this profile")
        argument = self._argument(additional)
        if major == 0:
            return argument
        if major == 1:
            return -1 - argument
        if major == 2:
            return self._take(argument)
        if major == 3:
            try:
                return self._take(argument).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CBORError("invalid UTF-8") from exc
        if major == 4:
            return [self.item() for _ in range(argument)]
        if major == 5:
            result: dict[Any, Any] = {}
            previous: bytes | None = None
            for _ in range(argument):
                start = self.offset
                key = self.item()
                key_bytes = self.data[start:self.offset]
                if previous is not None and key_bytes <= previous:
                    raise CBORError("map keys not in strict deterministic order")
                previous = key_bytes
                try:
                    if key in result:
                        raise CBORError("duplicate map key")
                    result[key] = self.item()
                except TypeError as exc:
                    raise CBORError("unhashable map key outside profile") from exc
            return result
        raise CBORError("unknown CBOR major type")


def cbor_decode(data: bytes) -> Any:
    decoder = _Decoder(data)
    value = decoder.item()
    if decoder.offset != len(data):
        raise CBORError("trailing CBOR bytes")
    if cbor_encode(value) != data:
        raise CBORError("input is not the deterministic encoding")
    return value


def _fixture_private(label: str) -> Ed25519PrivateKey:
    seed = sha256(("OFFLINE_TEST_VECTOR_ONLY:" + label).encode("ascii"))
    return Ed25519PrivateKey.from_private_bytes(seed)


def _public_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _aad(kind: str) -> bytes:
    return EXTERNAL_AAD_PREFIX + kind.encode("ascii")


def cose_sign1(payload_object: dict[str, Any], key: Ed25519PrivateKey, kid: bytes, kind: str) -> bytes:
    payload = cbor_encode(payload_object)
    protected = cbor_encode({1: COSE_ALG_EDDSA, 4: kid})
    to_be_signed = cbor_encode(["Signature1", protected, _aad(kind), payload])
    signature = key.sign(to_be_signed)
    return cbor_encode([protected, {}, payload, signature])


def cose_verify1(
    encoded: bytes,
    key: Ed25519PublicKey,
    expected_kid: bytes,
    kind: str,
) -> dict[str, Any]:
    message = cbor_decode(encoded)
    if not isinstance(message, list) or len(message) != 4:
        raise VerificationError("COSE_Sign1 shape")
    protected_bytes, unprotected, payload, signature = message
    if not all(isinstance(x, bytes) for x in (protected_bytes, payload, signature)):
        raise VerificationError("COSE_Sign1 byte-string fields")
    if unprotected != {}:
        raise VerificationError("unprotected headers forbidden")
    protected = cbor_decode(protected_bytes)
    if protected != {1: COSE_ALG_EDDSA, 4: expected_kid}:
        raise VerificationError("protected algorithm or kid mismatch")
    to_be_signed = cbor_encode(["Signature1", protected_bytes, _aad(kind), payload])
    try:
        key.verify(signature, to_be_signed)
    except InvalidSignature as exc:
        raise VerificationError("Ed25519 signature rejected") from exc
    decoded = cbor_decode(payload)
    if not isinstance(decoded, dict):
        raise VerificationError("signed payload must be a map")
    return decoded


def _require_keys(record: dict[str, Any], required: set[str]) -> None:
    if set(record) != required:
        raise VerificationError(
            f"schema keys differ: missing={sorted(required-set(record))}, "
            f"extra={sorted(set(record)-required)}"
        )


ENROLLMENT_KEYS = {
    "schema",
    "campaign_id",
    "device_id",
    "board_id",
    "hardware_revision",
    "device_signing_kid",
    "device_public_key_sha256",
    "fixture_domain_exclusion",
    "protocol_version",
    "manifest_schema_version",
    "analysis_schema_version",
    "policy_version",
    "measured_boot_sha256",
    "firmware_sha256",
    "bom_digest",
    "calibration_digest",
    "created_ns",
    "expires_ns",
    "production_authorization",
    "offline_test_vector",
}


CHALLENGE_KEYS = {
    "schema",
    "campaign_id",
    "session_id",
    "transaction_id",
    "allocation_id",
    "generation",
    "backend_id",
    "nonce",
    "descriptor_sha256",
    "protocol_version",
    "manifest_schema_version",
    "deadline_ns",
    "expected_rails",
    "tls_exporter_sha256",
    "transport_evidence_class",
}


MANIFEST_KEYS = {
    "schema",
    "evidence_class",
    "campaign_id",
    "session_id",
    "transaction_id",
    "allocation_id",
    "generation",
    "monotonic_sequence",
    "challenge_sha256",
    "nonce",
    "tls_exporter_sha256",
    "device_id",
    "signer_kid",
    "attestation_result_digest",
    "firmware_sha256",
    "descriptor_sha256",
    "rail_labels",
    "sample_count",
    "sample_encoding",
    "sample_rate_hz",
    "channel_map",
    "payload_blocks",
    "raw_total_length",
    "capture_state",
    "faults",
    "physical_sample",
    "source",
}


PLAN_KEYS = {
    "schema",
    "campaign_id",
    "plan_version",
    "locked_before_first_capture",
    "data_origin",
    "primary_endpoints",
    "family_alpha_ppm",
    "family_size",
    "equivalence_margin_urad",
    "success_tolerance_urad",
    "minimum_success_probability_ppm",
    "minimum_n_per_endpoint",
    "interim_looks",
    "stopping_rule",
    "multiplicity_method",
    "equivalence_method",
    "finite_count_method",
    "circular_branch_rule",
    "randomization",
    "blinding",
    "exclusion_policy",
    "self_adjudication_allowed",
    "physical_authorization_present",
}


def build_enrollment(device_public: bytes, device_kid: bytes) -> dict[str, Any]:
    return {
        "schema": "phase_qemu.v13.offline_enrollment_record.v1",
        "campaign_id": CAMPAIGN_ID,
        "device_id": DEVICE_ID,
        "board_id": "OFFLINE-BOARD-NOT-PRESENT",
        "hardware_revision": "NONE",
        "device_signing_kid": device_kid,
        "device_public_key_sha256": sha256(device_public),
        "fixture_domain_exclusion": ["TEST_FIXTURE", "OFFLINE_VECTOR"],
        "protocol_version": 1,
        "manifest_schema_version": 1,
        "analysis_schema_version": 1,
        "policy_version": 1,
        "measured_boot_sha256": sha256(b"NO_MEASURED_BOOT_OFFLINE_VECTOR"),
        "firmware_sha256": sha256(b"NO_FIRMWARE_OFFLINE_VECTOR"),
        "bom_digest": sha256(b"NO_BOM_OFFLINE_VECTOR"),
        "calibration_digest": sha256(b"NO_CALIBRATION_OFFLINE_VECTOR"),
        "created_ns": 0,
        "expires_ns": 1_000_000_000,
        "production_authorization": False,
        "offline_test_vector": True,
    }


def verify_enrollment(record: dict[str, Any], device_public: bytes, device_kid: bytes) -> None:
    _require_keys(record, ENROLLMENT_KEYS)
    if record["schema"] != "phase_qemu.v13.offline_enrollment_record.v1":
        raise VerificationError("enrollment schema")
    if record["campaign_id"] != CAMPAIGN_ID or record["device_id"] != DEVICE_ID:
        raise VerificationError("enrollment identity")
    if record["device_signing_kid"] != device_kid:
        raise VerificationError("enrollment kid")
    if record["device_public_key_sha256"] != sha256(device_public):
        raise VerificationError("enrollment public-key digest")
    if record["production_authorization"] is not False or record["offline_test_vector"] is not True:
        raise VerificationError("offline enrollment cannot authorize production")
    if "TEST_FIXTURE" not in record["fixture_domain_exclusion"]:
        raise VerificationError("fixture-domain exclusion missing")


def build_challenge(transaction_id: int, generation: int, descriptor: bytes) -> dict[str, Any]:
    return {
        "schema": "phase_qemu.v13.offline_challenge.v1",
        "campaign_id": CAMPAIGN_ID,
        "session_id": "OFFLINE-SESSION-0001",
        "transaction_id": transaction_id,
        "allocation_id": "OFFLINE-ALLOCATION-0001",
        "generation": generation,
        "backend_id": BACKEND_ID,
        "nonce": sha256(f"offline-nonce-{transaction_id}".encode("ascii")),
        "descriptor_sha256": sha256(descriptor),
        "protocol_version": 1,
        "manifest_schema_version": 1,
        "deadline_ns": 900_000_000 + transaction_id,
        "expected_rails": [0, 1],
        "tls_exporter_sha256": sha256(b"NO_TLS_OFFLINE_VECTOR"),
        "transport_evidence_class": "OFFLINE_TEST_VECTOR_ONLY",
    }


def build_manifest(
    challenge: dict[str, Any],
    sequence: int,
    raw_blocks: list[bytes],
    sample_count: int,
    device_kid: bytes,
    enrollment: dict[str, Any],
) -> dict[str, Any]:
    block_receipts = [
        {"index": index, "length": len(block), "sha256": sha256(block)}
        for index, block in enumerate(raw_blocks)
    ]
    return {
        "schema": "phase_qemu.v13.offline_raw_manifest.v1",
        "evidence_class": "OFFLINE_TEST_VECTOR_ONLY",
        "campaign_id": challenge["campaign_id"],
        "session_id": challenge["session_id"],
        "transaction_id": challenge["transaction_id"],
        "allocation_id": challenge["allocation_id"],
        "generation": challenge["generation"],
        "monotonic_sequence": sequence,
        "challenge_sha256": sha256(cbor_encode(challenge)),
        "nonce": challenge["nonce"],
        "tls_exporter_sha256": challenge["tls_exporter_sha256"],
        "device_id": DEVICE_ID,
        "signer_kid": device_kid,
        "attestation_result_digest": sha256(b"NO_ATTESTATION_OFFLINE_VECTOR"),
        "firmware_sha256": enrollment["firmware_sha256"],
        "descriptor_sha256": challenge["descriptor_sha256"],
        "rail_labels": ["RAIL_A", "RAIL_B"],
        "sample_count": sample_count,
        "sample_encoding": "SIGNED_INT32_LE_MICRORADIANS",
        "sample_rate_hz": 1_000,
        "channel_map": ["SYNTHETIC_PHASE_ERROR"],
        "payload_blocks": block_receipts,
        "raw_total_length": sum(len(block) for block in raw_blocks),
        "capture_state": "OFFLINE_SYNTHETIC_COMPLETE",
        "faults": [],
        "physical_sample": False,
        "source": "SYNTHETIC_BYTES_GENERATED_IN_PROCESS",
    }


def verify_manifest(
    manifest: dict[str, Any],
    challenge: dict[str, Any],
    enrollment: dict[str, Any],
    raw_blocks: list[bytes],
    expected_sequence: int,
    replay_seen: set[tuple[str, int]],
) -> None:
    _require_keys(manifest, MANIFEST_KEYS)
    _require_keys(challenge, CHALLENGE_KEYS)
    if manifest["schema"] != "phase_qemu.v13.offline_raw_manifest.v1":
        raise VerificationError("manifest schema")
    if manifest["evidence_class"] != "OFFLINE_TEST_VECTOR_ONLY":
        raise VerificationError("manifest evidence class")
    if manifest["physical_sample"] is not False:
        raise VerificationError("offline manifest cannot claim a physical sample")
    for key in ("campaign_id", "session_id", "transaction_id", "allocation_id", "generation"):
        if manifest[key] != challenge[key]:
            raise VerificationError(f"challenge binding: {key}")
    if manifest["challenge_sha256"] != sha256(cbor_encode(challenge)):
        raise VerificationError("challenge digest")
    if manifest["nonce"] != challenge["nonce"]:
        raise VerificationError("nonce")
    if manifest["tls_exporter_sha256"] != challenge["tls_exporter_sha256"]:
        raise VerificationError("transport binding")
    if manifest["descriptor_sha256"] != challenge["descriptor_sha256"]:
        raise VerificationError("descriptor binding")
    if manifest["firmware_sha256"] != enrollment["firmware_sha256"]:
        raise VerificationError("firmware binding")
    if manifest["monotonic_sequence"] != expected_sequence:
        raise VerificationError("sequence")
    replay_key = (manifest["session_id"], manifest["monotonic_sequence"])
    if replay_key in replay_seen:
        raise VerificationError("replay")
    receipts = manifest["payload_blocks"]
    if len(receipts) != len(raw_blocks):
        raise VerificationError("payload block count")
    for index, (receipt, block) in enumerate(zip(receipts, raw_blocks, strict=True)):
        _require_keys(receipt, {"index", "length", "sha256"})
        if receipt != {"index": index, "length": len(block), "sha256": sha256(block)}:
            raise VerificationError("payload block receipt")
    if manifest["raw_total_length"] != sum(len(block) for block in raw_blocks):
        raise VerificationError("raw total length")
    replay_seen.add(replay_key)


def build_plan(plan_kid: bytes) -> dict[str, Any]:
    return {
        "schema": "phase_qemu.v13.offline_preregistration.v1",
        "campaign_id": CAMPAIGN_ID,
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


def verify_plan(plan: dict[str, Any]) -> None:
    _require_keys(plan, PLAN_KEYS)
    if plan["schema"] != "phase_qemu.v13.offline_preregistration.v1":
        raise VerificationError("plan schema")
    if plan["campaign_id"] != CAMPAIGN_ID:
        raise VerificationError("plan campaign")
    if plan["locked_before_first_capture"] is not True:
        raise VerificationError("plan not locked")
    if plan["physical_authorization_present"] is not False:
        raise VerificationError("offline plan cannot assert authorization")
    if plan["self_adjudication_allowed"] is not False:
        raise VerificationError("self-adjudication forbidden")
    if plan["interim_looks"] != 0 or plan["stopping_rule"] != "FIXED_N_NO_OPTIONAL_STOPPING":
        raise VerificationError("stopping rule")
    if plan["family_size"] != 4 or plan["multiplicity_method"] != "BONFERRONI_SINGLE_FAMILY":
        raise VerificationError("multiplicity family")


def wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def analyze_endpoint(samples_urad: list[int], target_urad: int, plan: dict[str, Any]) -> dict[str, Any]:
    if len(samples_urad) < plan["minimum_n_per_endpoint"]:
        raise VerificationError("endpoint below preregistered N")
    errors = [wrap_radians((sample - target_urad) / 1_000_000.0) for sample in samples_urad]
    n = len(errors)
    cosine = sum(math.cos(x) for x in errors)
    sine = sum(math.sin(x) for x in errors)
    resultant = math.hypot(cosine, sine) / n
    if resultant <= 1e-12:
        raise VerificationError("degenerate circular mean")
    mean_direction = math.atan2(sine, cosine)
    linear_mean = sum(errors) / n
    sample_variance = sum((x - linear_mean) ** 2 for x in errors) / (n - 1)
    standard_error = math.sqrt(sample_variance / n)
    margin = plan["equivalence_margin_urad"] / 1_000_000.0
    alpha = plan["family_alpha_ppm"] / 1_000_000.0 / plan["family_size"]
    if standard_error == 0.0:
        lower_p = 0.0 if linear_mean > -margin else 1.0
        upper_p = 0.0 if linear_mean < margin else 1.0
    else:
        lower_stat = (linear_mean + margin) / standard_error
        upper_stat = (linear_mean - margin) / standard_error
        lower_p = float(t.sf(lower_stat, n - 1))
        upper_p = float(t.cdf(upper_stat, n - 1))
    tolerance = plan["success_tolerance_urad"] / 1_000_000.0
    successes = sum(abs(error) <= tolerance for error in errors)
    cp_lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, n - successes + 1))
    minimum_success = plan["minimum_success_probability_ppm"] / 1_000_000.0
    tost_pass = lower_p < alpha and upper_p < alpha
    count_pass = cp_lower >= minimum_success
    return {
        "n": n,
        "mean_direction_rad": mean_direction,
        "resultant_length": resultant,
        "linear_wrapped_error_mean_rad": linear_mean,
        "standard_error_rad": standard_error,
        "bonferroni_alpha": alpha,
        "tost_lower_p": lower_p,
        "tost_upper_p": upper_p,
        "tost_equivalent": tost_pass,
        "within_tolerance_count": successes,
        "clopper_pearson_lower": cp_lower,
        "minimum_success_probability": minimum_success,
        "finite_count_pass": count_pass,
        "endpoint_pass": tost_pass and count_pass,
    }


def signed_endpoint_receipt(endpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the integer/Boolean subset admitted by the signed CBOR profile."""
    return {
        "n": endpoint["n"],
        "within_tolerance_count": endpoint["within_tolerance_count"],
        "tost_equivalent": endpoint["tost_equivalent"],
        "finite_count_pass": endpoint["finite_count_pass"],
        "endpoint_pass": endpoint["endpoint_pass"],
    }


def merkle_root(leaves: Iterable[bytes]) -> bytes:
    nodes = sorted(sha256(b"\x00" + leaf) for leaf in leaves)
    if not nodes:
        raise VerificationError("empty manifest inventory")
    if len(set(nodes)) != len(nodes):
        raise VerificationError("duplicate manifest inventory leaf")
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [sha256(b"\x01" + nodes[i] + nodes[i + 1]) for i in range(0, len(nodes), 2)]
    return nodes[0]


def _rejected(callable_object: Any) -> bool:
    try:
        callable_object()
    except (CBORError, VerificationError, InvalidSignature, ValueError):
        return True
    return False


def run() -> dict[str, Any]:
    enrollment_private = _fixture_private("enrollment-authority")
    device_private = _fixture_private("device")
    plan_private = _fixture_private("preregistration-owner")
    adjudicator_private = _fixture_private("independent-adjudicator")
    enrollment_public = enrollment_private.public_key()
    device_public = device_private.public_key()
    plan_public = plan_private.public_key()
    adjudicator_public = adjudicator_private.public_key()
    enrollment_kid = b"offline-enrollment-v1"
    device_kid = b"offline-device-v1"
    plan_kid = b"offline-plan-v1"
    adjudicator_kid = b"offline-adjudicator-v1"

    device_public_bytes = _public_bytes(device_public)
    enrollment = build_enrollment(device_public_bytes, device_kid)
    signed_enrollment = cose_sign1(enrollment, enrollment_private, enrollment_kid, "enrollment")
    verified_enrollment = cose_verify1(
        signed_enrollment, enrollment_public, enrollment_kid, "enrollment"
    )
    verify_enrollment(verified_enrollment, device_public_bytes, device_kid)

    plan = build_plan(plan_kid)
    signed_plan = cose_sign1(plan, plan_private, plan_kid, "preregistration")
    verified_plan = cose_verify1(signed_plan, plan_public, plan_kid, "preregistration")
    verify_plan(verified_plan)
    plan_hash = sha256(signed_plan)

    program_a_errors = [-12_000, 8_000, -6_000, 10_000, -9_000, 7_000, -4_000, 5_000, -3_000, 6_000, -5_000, 4_000]
    program_b_errors = [-15_000, 11_000, -8_000, 13_000, -10_000, 9_000, -7_000, 8_000, -6_000, 7_000, -5_000, 6_000]
    target_a = 0
    target_b = 1_570_796
    program_a_samples = [target_a + error for error in program_a_errors]
    program_b_samples = [target_b + error for error in program_b_errors]
    control_samples = [200_000, 210_000, 190_000, 205_000, 195_000, 215_000, 185_000, 220_000, 180_000, 225_000, 175_000, 230_000]
    raw_a = struct.pack("<12i", *program_a_samples)
    raw_b = struct.pack("<12i", *program_b_samples)
    descriptors = [b"PROGRAM_A_PHASE_0", b"PROGRAM_B_PHASE_PI_OVER_2"]
    challenges = [build_challenge(1, 1, descriptors[0]), build_challenge(2, 2, descriptors[1])]
    raw_sets = [[raw_a], [raw_b]]
    manifests: list[dict[str, Any]] = []
    signed_manifests: list[bytes] = []
    replay_seen: set[tuple[str, int]] = set()
    for index, (challenge, raw_set) in enumerate(zip(challenges, raw_sets, strict=True), start=1):
        manifest = build_manifest(
            challenge, index, raw_set, 12, device_kid, verified_enrollment
        )
        signed = cose_sign1(manifest, device_private, device_kid, "raw-manifest")
        verified = cose_verify1(signed, device_public, device_kid, "raw-manifest")
        verify_manifest(
            verified,
            challenge,
            verified_enrollment,
            raw_set,
            index,
            replay_seen,
        )
        manifests.append(verified)
        signed_manifests.append(signed)

    inventory_root = merkle_root(signed_manifests)
    endpoint_a = analyze_endpoint(program_a_samples, target_a, verified_plan)
    endpoint_b = analyze_endpoint(program_b_samples, target_b, verified_plan)
    control = analyze_endpoint(control_samples, 0, verified_plan)

    report = {
        "schema": "phase_qemu.v13.offline_adjudication_preflight.v1",
        "classification": "OFFLINE_SYNTHETIC_ADJUDICATION_PREFLIGHT_ONLY",
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": plan_hash,
        "manifest_inventory_root": inventory_root,
        "manifest_count": len(signed_manifests),
        "missing_manifest_count": 0,
        "invalid_manifest_count": 0,
        "program_a": signed_endpoint_receipt(endpoint_a),
        "program_b": signed_endpoint_receipt(endpoint_b),
        "positive_offset_control": signed_endpoint_receipt(control),
        "all_primary_endpoints_pass": endpoint_a["endpoint_pass"] and endpoint_b["endpoint_pass"],
        "negative_control_rejected": not control["endpoint_pass"],
        "physical_evidence": False,
        "authenticated_physical_sample": False,
        "campaign_statistical_certificate": False,
        "physical_authorization_present": False,
        "architecture_promotion_eligible": False,
        "m257_escape": False,
    }
    signed_report = cose_sign1(
        report, adjudicator_private, adjudicator_kid, "adjudication-preflight"
    )
    verified_report = cose_verify1(
        signed_report, adjudicator_public, adjudicator_kid, "adjudication-preflight"
    )

    bad_signature = bytearray(signed_manifests[0])
    bad_signature[-1] ^= 1
    wrong_challenge = dict(challenges[0])
    wrong_challenge["nonce"] = sha256(b"wrong-nonce")
    missing_block_manifest = dict(manifests[0])
    missing_block_manifest["payload_blocks"] = []
    replay_probe: set[tuple[str, int]] = set()
    verify_manifest(
        manifests[0],
        challenges[0],
        verified_enrollment,
        raw_sets[0],
        1,
        replay_probe,
    )
    mutated_plan = dict(verified_plan)
    mutated_plan["equivalence_margin_urad"] += 1

    checks = {
        "deterministic_cbor_roundtrip": cbor_decode(cbor_encode({"aa": 1, "z": 2, 10: 3})) == {"aa": 1, "z": 2, 10: 3},
        "nonshortest_integer_rejected": _rejected(lambda: cbor_decode(b"\x18\x17")),
        "indefinite_item_rejected": _rejected(lambda: cbor_decode(b"\x9f\xff")),
        "unsorted_map_rejected": _rejected(lambda: cbor_decode(b"\xa2\x61z\x01\x18\x64\x02")),
        "duplicate_map_key_rejected": _rejected(lambda: cbor_decode(b"\xa2\x01\x01\x01\x02")),
        "trailing_bytes_rejected": _rejected(lambda: cbor_decode(b"\x01\x00")),
        "enrollment_signature_verified": verified_enrollment == enrollment,
        "offline_enrollment_cannot_authorize_production": verified_enrollment["production_authorization"] is False,
        "fixture_domain_excluded": "TEST_FIXTURE" in verified_enrollment["fixture_domain_exclusion"],
        "preregistration_signature_verified": verified_plan == plan,
        "preregistration_locked_before_data": verified_plan["locked_before_first_capture"] is True,
        "post_lock_plan_mutation_changes_hash": sha256(cbor_encode(mutated_plan)) != sha256(cbor_encode(verified_plan)),
        "two_signed_manifests_verified": len(signed_manifests) == 2 and len(replay_seen) == 2,
        "bad_manifest_signature_rejected": _rejected(
            lambda: cose_verify1(bytes(bad_signature), device_public, device_kid, "raw-manifest")
        ),
        "wrong_signature_domain_rejected": _rejected(
            lambda: cose_verify1(signed_manifests[0], device_public, device_kid, "enrollment")
        ),
        "wrong_challenge_rejected": _rejected(
            lambda: verify_manifest(
                manifests[0], wrong_challenge, verified_enrollment, raw_sets[0], 1, set()
            )
        ),
        "missing_payload_block_rejected": _rejected(
            lambda: verify_manifest(
                missing_block_manifest,
                challenges[0],
                verified_enrollment,
                raw_sets[0],
                1,
                set(),
            )
        ),
        "replay_rejected": _rejected(
            lambda: verify_manifest(
                manifests[0],
                challenges[0],
                verified_enrollment,
                raw_sets[0],
                1,
                replay_probe,
            )
        ),
        "role_keys_are_distinct": len(
            {
                _public_bytes(enrollment_public),
                device_public_bytes,
                _public_bytes(plan_public),
                _public_bytes(adjudicator_public),
            }
        ) == 4,
        "manifest_inventory_complete": len(signed_manifests) == 2 and len(inventory_root) == 32,
        "primary_synthetic_endpoints_pass_preregistered_rules": verified_report["all_primary_endpoints_pass"] is True,
        "positive_offset_control_fails_preregistered_rules": verified_report["negative_control_rejected"] is True,
        "report_signed_by_distinct_adjudicator": adjudicator_kid not in {device_kid, enrollment_kid, plan_kid},
        "no_authenticated_physical_sample": verified_report["authenticated_physical_sample"] is False,
        "no_campaign_statistical_certificate": verified_report["campaign_statistical_certificate"] is False,
        "no_physical_or_architecture_promotion": verified_report["physical_evidence"] is False and verified_report["architecture_promotion_eligible"] is False,
        "m257_intact": verified_report["m257_escape"] is False,
    }
    if not all(checks.values()):
        raise AssertionError([key for key, value in checks.items() if not value])

    source_bytes = Path(__file__).read_bytes()
    evidence = {
        "schema": SCHEMA,
        "status": STATUS,
        "classification": CLASSIFICATION,
        "authority_gate": AUTHORITY_GATE,
        "source_sha256": sha256_hex(source_bytes),
        "standards_profile": {
            "cbor": "RFC8949_CORE_DETERMINISTIC_SUBSET_NO_TAGS_FLOATS_OR_INDEFINITE_ITEMS",
            "cose": "RFC9052_UNTAGGED_COSE_SIGN1_EDDSA_WITH_DOMAIN_SEPARATED_EXTERNAL_AAD",
            "attestation_roles": "RFC9334_SHAPED_ONLY_NO_EVIDENCE_OR_ATTESTATION_RESULT_EXECUTED",
            "eat": "RFC9711_NAMED_FOR_FUTURE_PROFILE_ONLY_NO_EAT_PARSED_OR_ISSUED",
        },
        "fixture_identity": {
            "campaign_id": CAMPAIGN_ID,
            "device_id": DEVICE_ID,
            "fixture_keys_are_embedded_nonproduction": True,
            "fixture_private_key_count": 4,
            "production_key_or_certificate_present": False,
        },
        "encoded_objects": {
            "enrollment_payload_sha256": sha256_hex(cbor_encode(enrollment)),
            "signed_enrollment_sha256": sha256_hex(signed_enrollment),
            "preregistration_payload_sha256": sha256_hex(cbor_encode(plan)),
            "signed_preregistration_sha256": sha256_hex(signed_plan),
            "challenge_sha256": [sha256_hex(cbor_encode(x)) for x in challenges],
            "manifest_payload_sha256": [sha256_hex(cbor_encode(x)) for x in manifests],
            "signed_manifest_sha256": [sha256_hex(x) for x in signed_manifests],
            "manifest_inventory_root_sha256": inventory_root.hex(),
            "signed_adjudication_preflight_sha256": sha256_hex(signed_report),
        },
        "analysis": {
            "program_a": endpoint_a,
            "program_b": endpoint_b,
            "positive_offset_control": control,
            "family_size": plan["family_size"],
            "family_alpha": plan["family_alpha_ppm"] / 1_000_000.0,
            "multiplicity": "BONFERRONI",
            "physical_interpretation_allowed": False,
        },
        "checks": checks,
        "resource_accounting": {
            "source_bytes": len(source_bytes),
            "raw_synthetic_payload_bytes": len(raw_a) + len(raw_b),
            "encoded_enrollment_bytes": len(signed_enrollment),
            "encoded_preregistration_bytes": len(signed_plan),
            "encoded_manifest_bytes": [len(x) for x in signed_manifests],
            "encoded_adjudication_preflight_bytes": len(signed_report),
            "signature_create_operations": 5,
            "signature_verify_operations_primary": 5,
            "sha256_digest_bytes": 32,
            "manifest_count": 2,
            "replay_state_entries": len(replay_seen),
            "retained_inverse_history_entries": 0,
            "rematerialized_physical_samples": 0,
            "compiler_or_interpreter": "CPYTHON",
            "cryptography_version": cryptography.__version__,
            "scipy_version": scipy.__version__,
            "numeric_precision": "PYTHON_BINARY64_FOR_SYNTHETIC_STATISTICS",
            "physical_energy_joules": UNKNOWN,
            "physical_bandwidth_hz": UNKNOWN,
            "physical_latency_seconds": UNKNOWN,
            "physical_controller_state": UNKNOWN,
            "physical_environment_history": UNKNOWN,
            "process_peak_rss_bytes": "NOT_INSTRUMENTED_NOT_USED_FOR_CLAIM",
        },
        "strongest_honest_comparators": {
            "direct_encoded_protocol_verifier": "SAME_PUBLIC_CBOR_COSE_SCHEMA_AND_KEYS",
            "direct_statistical_recomputation": "SAME_PUBLIC_RAW_SYNTHETIC_VALUES_AND_PREREGISTERED_RULES",
            "resource_advantage_established": False,
            "m257_escape_established": False,
        },
        "architecture_scope": {
            "existing_common_phase_qemu_v13_backend_targeted": True,
            "qemu_device_modified_by_this_preflight": False,
            "qemu_device_executed_by_this_preflight": False,
            "live_backend_connected": False,
            "device_enrolled": False,
            "physical_capture_executed": False,
            "authenticated_physical_sample_published": False,
            "campaign_statistical_certificate_published": False,
            "physical_restoration_or_reuse_claimed": False,
            "eligible_for_m272_promotion": False,
            "terminal": False,
        },
    }
    payload = dict(evidence)
    payload.pop("source_sha256")
    evidence["claim_payload_sha256"] = sha256_hex(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return evidence


def main() -> None:
    print(json.dumps(run(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
