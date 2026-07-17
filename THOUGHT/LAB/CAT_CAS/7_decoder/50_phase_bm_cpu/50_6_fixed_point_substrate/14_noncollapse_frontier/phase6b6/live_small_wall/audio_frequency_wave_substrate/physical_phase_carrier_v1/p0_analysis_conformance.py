"""Deterministic offline conformance for the prospective P0 analysis packet."""

from __future__ import annotations

import base64
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from jsonschema import Draft202012Validator

from p0_packet_validator import (
    NULL_BASELINE_CONTROL_IDS,
    PacketValidationError,
    canonical_bytes,
    sha256_bytes,
    sha256_object,
    validate_packet,
)


ROOT = Path(__file__).resolve().parent
VECTORS = ROOT / "P0_ANALYSIS_CONFORMANCE_VECTORS.json"
SCHEMAS = ROOT / "P0_EVIDENCE_SCHEMAS.json"
ZERO_SHA256 = "0" * 64
EXPECTED_DTYPE = "IEEE-754 binary64 round-to-nearest ties-to-even"
EXPECTED_FIXTURE_VERSION = "p0-analysis-conformance-v3"
EXPECTED_VECTOR_IDS = (
    "ideal_antipode",
    "weighted_projection_recovery",
    "drive_phase_error_not_gauged_away",
    "quarter_cycle_timing_control",
    "wrap_positive_pi_to_negative_pi",
    "flat_wave_reject",
    "near_singular_projection_reject",
    "matched_alignment",
    "noise_mad_maximum",
    "hac_regression",
    "block_jackknife",
    "schema_minimal_packet_accept",
    "schema_relation_metrics_accept",
    "schema_unknown_field_reject",
    "schema_path_traversal_reject",
    "schema_arbitrary_object_reject",
    "conformance_identity_missing_duplicate_reject",
    "packet_complete_accept",
    "packet_structural_preview_no_scientific_authority",
    "packet_replacement_complete_accept",
    "packet_replacement_missing_preserved_packet_reject",
    "packet_replacement_configuration_drift_reject",
    "packet_replacement_assignment_drift_reject",
    "packet_replacement_prior_count_reject",
    "packet_metrics_cross_hash_reject",
    "packet_raw_manifest_cross_hash_reject",
    "packet_receipt_chain_reject",
    "packet_early_reveal_reject",
    "packet_calibration_chronology_reject",
    "packet_early_acquisition_reject",
    "packet_late_acquisition_completion_reject",
    "packet_signature_reject",
    "packet_control_membership_reject",
    "packet_control_evidence_submanifest_reject",
    "packet_relation_arm_mapping_reject",
    "packet_relation_negative_metric_reject",
    "packet_contact_count_reject",
    "packet_attempt_replay_reject",
    "packet_duplicate_attempt_id_reject",
    "packet_pass_coherence_reject",
)


def wrap(value: float) -> float:
    return ((value + math.pi) % (2.0 * math.pi)) - math.pi


def norm2(values: list[complex]) -> float:
    return math.sqrt(sum(abs(value) ** 2 for value in values))


def circular_mean(values: list[float]) -> float:
    sine = sum(math.sin(value) for value in values)
    cosine = sum(math.cos(value) for value in values)
    if sine == 0.0 and cosine == 0.0:
        raise ValueError("zero circular resultant")
    return math.atan2(sine, cosine)


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def as_complex(values: list[list[str]]) -> list[complex]:
    return [complex(float(real), float(imag)) for real, imag in values]


def weighted_projection(
    signal: np.ndarray, phase: np.ndarray
) -> tuple[np.ndarray, float]:
    count = signal.size
    index = np.arange(count, dtype=np.float64)
    weight = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / np.float64(count - 1))
    design = np.column_stack((np.cos(phase), -np.sin(phase), np.ones(count)))
    gram = design.T @ (weight[:, None] * design)
    rhs = design.T @ (weight * signal)
    condition = float(np.linalg.cond(gram, 2))
    factor = np.linalg.cholesky(gram)
    beta = np.linalg.solve(factor, rhs)
    beta = np.linalg.solve(factor.T, beta)
    return beta, condition


def newey_west(
    design: np.ndarray, response: np.ndarray, lag: int = 7
) -> tuple[np.ndarray, np.ndarray, float]:
    xtx = design.T @ design
    beta = np.linalg.solve(xtx, design.T @ response)
    residual = response - design @ beta
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    for row in range(design.shape[0]):
        meat += residual[row] ** 2 * np.outer(design[row], design[row])
    for offset in range(1, lag + 1):
        gamma = np.zeros_like(meat)
        for row in range(offset, design.shape[0]):
            gamma += (
                residual[row]
                * residual[row - offset]
                * np.outer(design[row], design[row - offset])
            )
        meat += (1.0 - offset / (lag + 1.0)) * (gamma + gamma.T)
    inverse = np.linalg.inv(xtx)
    covariance = inverse @ meat @ inverse
    centered = response - np.mean(response)
    sst = float(centered @ centered)
    r_squared = 1.0 - float(residual @ residual) / sst
    return beta, covariance, r_squared


def block_jackknife_mean(values: np.ndarray, block_size: int = 8) -> float:
    block_count = values.size // block_size
    values = values[: block_count * block_size]
    estimates = []
    for block in range(block_count):
        keep = np.ones(values.size, dtype=bool)
        keep[block * block_size : (block + 1) * block_size] = False
        estimates.append(float(np.mean(values[keep])))
    estimate_array = np.asarray(estimates, dtype=np.float64)
    return math.sqrt(
        (block_count - 1.0)
        / block_count
        * float(np.sum((estimate_array - np.mean(estimate_array)) ** 2))
    )


def median_mad(values: list[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    median = float(np.median(ordered))
    return 1.4826 * float(np.median(np.abs(ordered - median)))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_hash(label: str) -> str:
    return sha256_bytes(label.encode("ascii"))


def conformance_identity_valid(payload: dict[str, Any]) -> bool:
    vectors = payload.get("vectors")
    if not isinstance(vectors, list):
        return False
    ids = [vector.get("id") for vector in vectors if isinstance(vector, dict)]
    return (
        payload.get("dtype") == EXPECTED_DTYPE
        and payload.get("fixture_version") == EXPECTED_FIXTURE_VERSION
        and len(vectors) == len(EXPECTED_VECTOR_IDS)
        and len(ids) == len(vectors)
        and len(set(ids)) == len(ids)
        and tuple(ids) == EXPECTED_VECTOR_IDS
    )


def signed_receipt(
    unsigned: dict[str, Any], private_key: Ed25519PrivateKey
) -> dict[str, Any]:
    receipt = dict(unsigned)
    receipt["signature_base64"] = base64.b64encode(
        private_key.sign(canonical_bytes(unsigned))
    ).decode("ascii")
    return receipt


def raw_descriptor(run_id: str, arm_id: str) -> dict[str, Any]:
    return {
        "acquisition_completed_utc": "2026-07-16T00:00:00.900000Z",
        "acquisition_started_utc": "2026-07-16T00:00:00.750000Z",
        "arm_id": arm_id,
        "bit_width": 16,
        "byte_order": "little",
        "channel_order": ["CH0", "CH1", "CH2", "CH3"],
        "export_version": "fixture-v1",
        "firmware": "fixture-fw",
        "frame_interleave": "frame-major",
        "gain": ["1", "1", "1", "1"],
        "header_bytes": 0,
        "instrument_model": "fixture-digitizer",
        "instrument_serial": "fixture-serial",
        "offset": ["0", "0", "0", "0"],
        "padding_bytes": 0,
        "parser_sha256": fixed_hash("parser"),
        "precommand_count": 1101000,
        "record_type": "raw_descriptor",
        "run_id": run_id,
        "sample_count": 3101000,
        "sample_rate_hz": 1000000,
        "schema_version": "p0-evidence-v1",
        "signedness": "signed",
        "units": ["V", "V", "V", "m_s2"],
    }


def arm_metrics(run_id: str, arm_id: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "f_hat_hz": "32768",
        "iq_sha256": fixed_hash(f"iq-{arm_id}"),
        "q_hat": "10294.370807283034",
        "quality_gate_pass": True,
        "record_type": "metrics",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
        "tau_a_s": "0.1",
        "u95_f_hz": "0.01",
        "u95_tau_s": "0.001",
        "usable_cycles": 300,
    }


def make_assignments(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    key = bytes(range(32, 64))
    salt = bytes(range(64, 96))
    nonce = bytes(range(12))
    salt_base64 = base64.b64encode(salt).decode("ascii")
    mapping = {"A": "0", "B": "pi"}
    plaintext = {**mapping, "salt_base64": salt_base64}
    aad = {
        "record_type": "arm_assignment_aad",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    ciphertext = ChaCha20Poly1305(key).encrypt(
        nonce, canonical_bytes(plaintext), canonical_bytes(aad)
    )
    commitment = sha256_object(plaintext)
    sealed = {
        "aad_sha256": sha256_object(aad),
        "cipher": "ChaCha20-Poly1305-IETF-RFC8439",
        "ciphertext_base64": base64.b64encode(ciphertext).decode("ascii"),
        "commitment_sha256": commitment,
        "custodian_id": "fixture-custodian",
        "key_sha256": sha256_bytes(key),
        "nonce_base64": base64.b64encode(nonce).decode("ascii"),
        "record_type": "arm_assignment_sealed",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    reveal = {
        "commitment_sha256": commitment,
        "custodian_id": "fixture-custodian",
        "key_base64": base64.b64encode(key).decode("ascii"),
        "mapping_sha256": sha256_object(mapping),
        "record_type": "arm_assignment_reveal",
        "released_utc": "2026-07-16T00:00:02.000000Z",
        "run_id": run_id,
        "salt_base64": salt_base64,
        "schema_version": "p0-evidence-v1",
    }
    return sealed, reveal


def reseal_final(bundle: dict[str, Any]) -> None:
    packet_root: Path = bundle["packet_root"]
    private_key: Ed25519PrivateKey = bundle["private_key"]
    raw_receipt_path: Path = bundle["raw_receipt"]
    final_receipt_path: Path = bundle["final_receipt"]
    files = {
        path.relative_to(packet_root).as_posix(): path
        for path in packet_root.rglob("*")
        if path.is_file() and path != packet_root / "manifest.sha256.json"
    }
    manifest = {
        "files": {
            relative: sha256_bytes(path.read_bytes())
            for relative, path in sorted(files.items())
        },
        "record_type": "manifest",
        "run_id": bundle["run_id"],
        "schema_version": "p0-evidence-v1",
    }
    manifest_path = packet_root / "manifest.sha256.json"
    write_json(manifest_path, manifest)
    raw_receipt_hash = sha256_bytes(raw_receipt_path.read_bytes())
    raw_receipt = read_json(raw_receipt_path)
    final_unsigned = {
        "created_utc": "2026-07-16T00:00:03.000000Z",
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "medium_id": "fixture-worm-001",
        "previous_receipt_sha256": raw_receipt_hash,
        "public_key_sha256": bundle["public_key_sha256"],
        "raw_root_receipt_sha256": raw_receipt_hash,
        "record_type": "final_root_receipt",
        "run_id": bundle["run_id"],
        "schema_version": "p0-evidence-v1",
        "sequence_number": int(raw_receipt["sequence_number"]) + 1,
        "signature_algorithm": "Ed25519",
        "witness_id": "fixture-witness",
    }
    write_json(final_receipt_path, signed_receipt(final_unsigned, private_key))


def build_packet(base: Path) -> dict[str, Any]:
    run_id = "p0test01"
    packet_root = base / "packet"
    packet_root.mkdir()
    calibration_receipt_path = base / "calibration_root.json"
    raw_receipt_path = base / "raw_root.json"
    final_receipt_path = base / "final_root.json"
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    public_key_sha256 = sha256_bytes(public_key_bytes)
    sealed, reveal = make_assignments(run_id)
    schema = json.loads(SCHEMAS.read_text(encoding="utf-8"))
    control_ids = schema["$defs"]["controls"]["properties"]["ordered_control_ids"]["const"]
    if not NULL_BASELINE_CONTROL_IDS <= set(control_ids):
        raise ValueError("required null-baseline controls are absent")
    offline_control_ids = {
        "offline_synthetic_conformance",
        "offline_wrong_guard_analysis",
        "offline_phase_reference_inversion",
    }
    acquisition_for_control = {
        control_id: (
            f"offline_{control_id}"
            if control_id in offline_control_ids
            else "A"
            if control_id == "opaque_matched_a"
            else "B"
            if control_id == "opaque_matched_b"
            else control_id
        )
        for control_id in control_ids
    }
    physical_arm_ids = [
        acquisition_for_control[control_id]
        for control_id in control_ids
        if control_id not in offline_control_ids
    ]

    write_json(packet_root / "arm_assignment.sealed.json", sealed)
    write_json(packet_root / "arm_assignment.reveal.json", reveal)
    instrument_configuration_sha256 = fixed_hash("instrument-configuration")
    calibration = {
        "analysis_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "campaign_index": 1,
        "conformance_sha256": sha256_bytes(VECTORS.read_bytes()),
        "created_utc": "2026-07-16T00:00:00.000000Z",
        "dependency_lock_sha256": fixed_hash("dependency-lock"),
        "ed25519_public_key_base64": base64.b64encode(public_key_bytes).decode("ascii"),
        "ed25519_public_key_sha256": public_key_sha256,
        "f_ref_hz": "32768",
        "instrument_configuration_sha256": instrument_configuration_sha256,
        "parser_sha256": fixed_hash("parser"),
        "record_type": "calibration",
        "residual_skew_s": "0.00000002",
        "run_id": run_id,
        "schema_sha256": sha256_bytes(SCHEMAS.read_bytes()),
        "schema_version": "p0-evidence-v1",
        "sealed_assignment_sha256": sha256_bytes(
            (packet_root / "arm_assignment.sealed.json").read_bytes()
        ),
        "thresholds": {
            "T_A": "0.05",
            "T_f": "0.000005",
            "T_feed": "0.1",
            "T_neg": "0.1",
            "T_phi_rad": "0.05",
            "T_tau": "0.05",
        },
        "worm_medium_id": "fixture-worm-001",
    }
    write_json(packet_root / "calibration.json", calibration)
    calibration_receipt_unsigned = {
        "calibration_sha256": sha256_bytes((packet_root / "calibration.json").read_bytes()),
        "created_utc": "2026-07-16T00:00:00.500000Z",
        "medium_id": "fixture-worm-001",
        "previous_receipt_sha256": ZERO_SHA256,
        "public_key_sha256": public_key_sha256,
        "record_type": "calibration_root_receipt",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
        "sequence_number": 1,
        "signature_algorithm": "Ed25519",
        "witness_id": "fixture-witness",
    }
    write_json(
        calibration_receipt_path,
        signed_receipt(calibration_receipt_unsigned, private_key),
    )

    fixed_records = {
        "packet.json": {
            "claim_ceiling": "PHYSICAL_PHASE_CARRIER_PI_RELATION_CHARACTERIZED",
            "created_utc": "2026-07-16T00:00:00.000000Z",
            "record_type": "packet",
            "run_id": run_id,
            "schema_version": "p0-evidence-v1",
            "status": "ADJUDICATED",
        },
        "hardware_identity.json": {
            "components": [
                {
                    "datasheet_sha256": fixed_hash("carrier-datasheet"),
                    "function": "mechanical carrier",
                    "manufacturer": "fixture",
                    "model": "fixture-quartz",
                    "serial_or_lot": "fixture-unit-1",
                }
            ],
            "record_type": "hardware_identity",
            "run_id": run_id,
            "schema_version": "p0-evidence-v1",
        },
        "carrier_identity.json": {
            "carrier_class": "hermetic_two_terminal_quartz_tuning_fork",
            "datasheet_sha256": fixed_hash("carrier-datasheet"),
            "electrode_a_net": "CARRIER_A",
            "electrode_b_net": "CARRIER_REF",
            "loaded_bvd_fit_sha256": fixed_hash("bvd-fit"),
            "manufacturer": "fixture",
            "part_number": "fixture-quartz",
            "record_type": "carrier_identity",
            "run_id": run_id,
            "schema_version": "p0-evidence-v1",
            "unit_id": "fixture-unit-1",
        },
        "topology.json": {
            "continuity_record_sha256": fixed_hash("continuity"),
            "edge_list": [{"from": "SOURCE", "kind": "series-open", "to": "CARRIER_A"}],
            "injection_scan_sha256": fixed_hash("injection"),
            "netlist_sha256": fixed_hash("netlist"),
            "record_type": "topology",
            "run_id": run_id,
            "schema_version": "p0-evidence-v1",
            "star_reference_node": "CARRIER_REF",
        },
        "instruments.json": {
            "channel_map": ["CH0_SOURCE", "CH1_CARRIER", "CH2_SWITCH", "CH3_ACCELERATION"],
            "configuration_sha256": instrument_configuration_sha256,
            "digitizer_firmware": "fixture-fw",
            "digitizer_model": "fixture-digitizer",
            "digitizer_serial": "fixture-serial",
            "record_type": "instruments",
            "run_id": run_id,
            "sample_count": 3101000,
            "sample_rate_hz": 1000000,
            "schema_version": "p0-evidence-v1",
        },
        "contact_counts.json": {
            "adc_or_dac_operations": len(physical_arm_ids),
            "audio_play_or_record_operations": 0,
            "hardware_operations": len(physical_arm_ids),
            "instrument_operations": len(physical_arm_ids),
            "physical_runs": len(physical_arm_ids),
            "record_type": "contact_counts",
            "remote_target_contacts": 0,
            "run_id": run_id,
            "schema_version": "p0-evidence-v1",
            "vendor_contacts": 0,
        },
    }
    for relative, record in fixed_records.items():
        write_json(packet_root / relative, record)

    for arm_id in physical_arm_ids:
        run_dir = packet_root / "runs" / arm_id
        derived_dir = packet_root / "derived" / arm_id
        write_json(run_dir / "raw_descriptor.json", raw_descriptor(run_id, arm_id))
        (run_dir / "raw_native.bin").write_bytes(f"raw-{arm_id}".encode("ascii"))
        (run_dir / "raw.f64le").write_bytes(f"parsed-{arm_id}".encode("ascii"))
        (run_dir / "reference.f64le").write_bytes(f"reference-{arm_id}".encode("ascii"))
        (run_dir / "switch_state.u8").write_bytes(b"\x08")
        (run_dir / "environment.csv").write_bytes(
            b"nearest_raw_sample_index,utc_timestamp,temperature_C,rh_percent\n"
            b"1111000,2026-07-16T00:00:01.111000Z,25,40\n"
        )
        derived_dir.mkdir(parents=True, exist_ok=True)
        (derived_dir / "iq.f64le").write_bytes(f"iq-{arm_id}".encode("ascii"))
        (derived_dir / "iq_start_index.u64le").write_bytes(b"\x00" * 8)
        write_json(derived_dir / "metrics.json", arm_metrics(run_id, arm_id))

    relation = {
        "antipode_in_u95": True,
        "arm_0_id": "A",
        "arm_pi_id": "B",
        "epsilon_A": "0.005",
        "epsilon_f": "0.000001",
        "epsilon_feed": "0.005",
        "epsilon_neg": "0.005",
        "epsilon_phi_rad": "0.005",
        "epsilon_tau": "0.005",
        "plus_minus_pi_over_2_excluded_u95": True,
        "record_type": "relation_metrics",
        "residual_zero_in_u95": True,
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
        "u95_A": "0.001",
        "u95_f": "0.000001",
        "u95_feed": "0.001",
        "u95_neg": "0.001",
        "u95_phi_rad": "0.001",
        "u95_tau": "0.001",
        "zero_excluded_u95": True,
    }
    write_json(packet_root / "derived" / "relation_metrics.json", relation)

    raw_paths = ["arm_assignment.sealed.json"]
    for arm_id in physical_arm_ids:
        raw_paths.extend(
            (
                f"runs/{arm_id}/environment.csv",
                f"runs/{arm_id}/raw_descriptor.json",
                f"runs/{arm_id}/raw_native.bin",
            )
        )
    raw_manifest = {
        "files": {
            relative: sha256_bytes((packet_root / relative).read_bytes())
            for relative in sorted(raw_paths)
        },
        "record_type": "raw_manifest",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    write_json(packet_root / "raw_manifest.json", raw_manifest)

    calibration_receipt_sha256 = sha256_bytes(calibration_receipt_path.read_bytes())
    raw_unsigned = {
        "acquisition_completed_utc": "2026-07-16T00:00:00.900000Z",
        "acquisition_started_utc": "2026-07-16T00:00:00.750000Z",
        "calibration_root_receipt_sha256": calibration_receipt_sha256,
        "created_utc": "2026-07-16T00:00:01.000000Z",
        "medium_id": "fixture-worm-001",
        "previous_receipt_sha256": calibration_receipt_sha256,
        "public_key_sha256": public_key_sha256,
        "raw_manifest_sha256": sha256_bytes((packet_root / "raw_manifest.json").read_bytes()),
        "record_type": "raw_root_receipt",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
        "sequence_number": 2,
        "signature_algorithm": "Ed25519",
        "witness_id": "fixture-witness",
    }
    write_json(raw_receipt_path, signed_receipt(raw_unsigned, private_key))

    raw_receipt_sha256 = sha256_bytes(raw_receipt_path.read_bytes())
    attempt_ledger = {
        "attempts": [
            {
                "acquisition_completed_utc": "2026-07-16T00:00:00.900000Z",
                "acquisition_started_utc": "2026-07-16T00:00:00.750000Z",
                "attempt_id": "attempt-1",
                "integrity_reason": "NONE",
                "outcome": "COMPLETE",
                "phase_content_opened": False,
                "preserved_invalid_packet_manifest_sha256": ZERO_SHA256,
                "raw_root_receipt_sha256": raw_receipt_sha256,
                "selected_for_adjudication": True,
                "sequence_index": 1,
            }
        ],
        "record_type": "attempt_ledger",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    write_json(packet_root / "attempt_ledger.json", attempt_ledger)

    outcomes = {}
    for ordinal, control_id in enumerate(control_ids):
        acquisition_id = acquisition_for_control[control_id]
        evidence_path = packet_root / "control_evidence" / f"{control_id}.json"
        is_physical = control_id not in offline_control_ids
        if is_physical:
            evidence_files = {
                f"runs/{acquisition_id}/raw_native.bin",
                f"runs/{acquisition_id}/raw_descriptor.json",
                f"runs/{acquisition_id}/raw.f64le",
                f"runs/{acquisition_id}/reference.f64le",
                f"runs/{acquisition_id}/switch_state.u8",
                f"runs/{acquisition_id}/environment.csv",
                f"derived/{acquisition_id}/iq.f64le",
                f"derived/{acquisition_id}/iq_start_index.u64le",
                f"derived/{acquisition_id}/metrics.json",
            }
        else:
            offline_path = packet_root / "offline" / f"{control_id}.bin"
            offline_path.parent.mkdir(parents=True, exist_ok=True)
            offline_path.write_bytes(f"offline-{control_id}".encode("ascii"))
            evidence_files = {f"offline/{control_id}.bin"}
        evidence = {
            "acquisition_id": acquisition_id,
            "control_id": control_id,
            "files": {
                relative: sha256_bytes((packet_root / relative).read_bytes())
                for relative in sorted(evidence_files)
            },
            "execution_class": "PHYSICAL" if is_physical else "OFFLINE",
            "physical_authority_consumed": is_physical,
            "record_type": "control_evidence",
            "run_id": run_id,
            "schema_version": "p0-evidence-v1",
            "sequence_ordinal": ordinal,
        }
        write_json(evidence_path, evidence)
        outcomes[control_id] = {
            "evidence_sha256": sha256_bytes(evidence_path.read_bytes()),
            "outcome": "PASS",
        }
    ledger_payload = {"ordered_control_ids": control_ids, "outcomes": outcomes}
    controls = {
        **ledger_payload,
        "ledger_sha256": sha256_object(ledger_payload),
        "record_type": "controls",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    write_json(packet_root / "controls.json", controls)

    metric_paths = [
        *(f"derived/{arm_id}/metrics.json" for arm_id in physical_arm_ids),
        "derived/relation_metrics.json",
    ]
    metric_bundle = {
        "files": {
            relative: sha256_bytes((packet_root / relative).read_bytes())
            for relative in sorted(metric_paths)
        }
    }
    adjudication = {
        "claim_ceiling": "PHYSICAL_PHASE_CARRIER_PI_RELATION_CHARACTERIZED",
        "control_ledger_sha256": controls["ledger_sha256"],
        "decision": "PASS",
        "kill_ids": [],
        "metrics_bundle_sha256": sha256_object(metric_bundle),
        "raw_root_receipt_sha256": raw_receipt_sha256,
        "record_type": "adjudication",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    write_json(packet_root / "adjudication.json", adjudication)

    bundle = {
        "calibration_receipt": calibration_receipt_path,
        "final_receipt": final_receipt_path,
        "packet_root": packet_root,
        "private_key": private_key,
        "public_key_sha256": public_key_sha256,
        "raw_receipt": raw_receipt_path,
        "run_id": run_id,
    }
    reseal_final(bundle)
    return bundle


def rebind_selected_raw(bundle: dict[str, Any]) -> None:
    packet_root: Path = bundle["packet_root"]
    raw_hash = sha256_bytes(bundle["raw_receipt"].read_bytes())
    attempts = read_json(packet_root / "attempt_ledger.json")
    attempts["attempts"][-1]["raw_root_receipt_sha256"] = raw_hash
    write_json(packet_root / "attempt_ledger.json", attempts)
    adjudication = read_json(packet_root / "adjudication.json")
    adjudication["raw_root_receipt_sha256"] = raw_hash
    write_json(packet_root / "adjudication.json", adjudication)


def rebind_metrics(packet_root: Path) -> None:
    paths = sorted(
        path.relative_to(packet_root).as_posix()
        for path in (packet_root / "derived").rglob("metrics.json")
    )
    bundle = {
        "files": {
            relative: sha256_bytes((packet_root / relative).read_bytes())
            for relative in paths
        }
    }
    adjudication = read_json(packet_root / "adjudication.json")
    adjudication["metrics_bundle_sha256"] = sha256_object(bundle)
    write_json(packet_root / "adjudication.json", adjudication)


def rebind_controls(packet_root: Path) -> None:
    controls = read_json(packet_root / "controls.json")
    payload = {
        "ordered_control_ids": controls["ordered_control_ids"],
        "outcomes": controls["outcomes"],
    }
    controls["ledger_sha256"] = sha256_object(payload)
    write_json(packet_root / "controls.json", controls)
    adjudication = read_json(packet_root / "adjudication.json")
    adjudication["control_ledger_sha256"] = controls["ledger_sha256"]
    write_json(packet_root / "adjudication.json", adjudication)


def make_replacement_attempt(bundle: dict[str, Any]) -> None:
    packet_root: Path = bundle["packet_root"]
    private_key: Ed25519PrivateKey = bundle["private_key"]
    prior_receipt_path = bundle["raw_receipt"].with_name("raw_root_attempt_1.json")
    prior_receipt_path.write_bytes(bundle["raw_receipt"].read_bytes())

    prior_prefix = packet_root / "prior_attempts" / "attempt-1"
    fixed_paths = (
        "packet.json",
        "hardware_identity.json",
        "carrier_identity.json",
        "topology.json",
        "instruments.json",
        "calibration.json",
        "arm_assignment.sealed.json",
        "contact_counts.json",
        "raw_manifest.json",
    )
    for relative in fixed_paths:
        target = prior_prefix / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packet_root / relative, target)
    prior_packet = read_json(prior_prefix / "packet.json")
    prior_packet["status"] = "INVALID"
    write_json(prior_prefix / "packet.json", prior_packet)

    raw_manifest = read_json(packet_root / "raw_manifest.json")
    for relative in raw_manifest["files"]:
        if relative == "arm_assignment.sealed.json":
            continue
        target = prior_prefix / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packet_root / relative, target)
    for relative in raw_manifest["files"]:
        if not relative.endswith("/raw_descriptor.json"):
            continue
        descriptor_path = packet_root / relative
        descriptor = read_json(descriptor_path)
        descriptor["acquisition_started_utc"] = "2026-07-16T00:00:01.100000Z"
        descriptor["acquisition_completed_utc"] = "2026-07-16T00:00:01.400000Z"
        write_json(descriptor_path, descriptor)
    selected_raw_manifest = read_json(packet_root / "raw_manifest.json")
    selected_raw_manifest["files"] = {
        relative: sha256_bytes((packet_root / relative).read_bytes())
        for relative in sorted(selected_raw_manifest["files"])
    }
    write_json(packet_root / "raw_manifest.json", selected_raw_manifest)
    controls = read_json(packet_root / "controls.json")
    for control_id in controls["ordered_control_ids"]:
        evidence_path = packet_root / "control_evidence" / f"{control_id}.json"
        evidence = read_json(evidence_path)
        evidence["files"] = {
            relative: sha256_bytes((packet_root / relative).read_bytes())
            for relative in sorted(evidence["files"])
        }
        write_json(evidence_path, evidence)
        controls["outcomes"][control_id]["evidence_sha256"] = sha256_bytes(
            evidence_path.read_bytes()
        )
    write_json(packet_root / "controls.json", controls)
    rebind_controls(packet_root)
    prior_files = {
        path.relative_to(prior_prefix).as_posix(): sha256_bytes(path.read_bytes())
        for path in prior_prefix.rglob("*")
        if path.is_file()
    }
    prior_manifest = {
        "files": dict(sorted(prior_files.items())),
        "record_type": "manifest",
        "run_id": bundle["run_id"],
        "schema_version": "p0-evidence-v1",
    }
    prior_manifest_path = prior_prefix / "manifest.sha256.json"
    write_json(prior_manifest_path, prior_manifest)

    calibration_receipt_hash = sha256_bytes(bundle["calibration_receipt"].read_bytes())
    prior_receipt_hash = sha256_bytes(prior_receipt_path.read_bytes())
    selected_unsigned = {
        "acquisition_completed_utc": "2026-07-16T00:00:01.400000Z",
        "acquisition_started_utc": "2026-07-16T00:00:01.100000Z",
        "calibration_root_receipt_sha256": calibration_receipt_hash,
        "created_utc": "2026-07-16T00:00:01.500000Z",
        "medium_id": "fixture-worm-001",
        "previous_receipt_sha256": prior_receipt_hash,
        "public_key_sha256": bundle["public_key_sha256"],
        "raw_manifest_sha256": sha256_bytes((packet_root / "raw_manifest.json").read_bytes()),
        "record_type": "raw_root_receipt",
        "run_id": bundle["run_id"],
        "schema_version": "p0-evidence-v1",
        "sequence_number": 3,
        "signature_algorithm": "Ed25519",
        "witness_id": "fixture-witness",
    }
    write_json(bundle["raw_receipt"], signed_receipt(selected_unsigned, private_key))
    selected_hash = sha256_bytes(bundle["raw_receipt"].read_bytes())
    attempt_ledger = {
        "attempts": [
            {
                "acquisition_completed_utc": "2026-07-16T00:00:00.900000Z",
                "acquisition_started_utc": "2026-07-16T00:00:00.750000Z",
                "attempt_id": "attempt-1",
                "integrity_reason": "BYTE_COUNT_OR_HASH",
                "outcome": "INTEGRITY_INVALID",
                "phase_content_opened": False,
                "preserved_invalid_packet_manifest_sha256": sha256_bytes(
                    prior_manifest_path.read_bytes()
                ),
                "raw_root_receipt_sha256": prior_receipt_hash,
                "selected_for_adjudication": False,
                "sequence_index": 1,
            },
            {
                "acquisition_completed_utc": "2026-07-16T00:00:01.400000Z",
                "acquisition_started_utc": "2026-07-16T00:00:01.100000Z",
                "attempt_id": "attempt-2",
                "integrity_reason": "NONE",
                "outcome": "COMPLETE",
                "phase_content_opened": False,
                "preserved_invalid_packet_manifest_sha256": ZERO_SHA256,
                "raw_root_receipt_sha256": selected_hash,
                "selected_for_adjudication": True,
                "sequence_index": 2,
            },
        ],
        "record_type": "attempt_ledger",
        "run_id": bundle["run_id"],
        "schema_version": "p0-evidence-v1",
    }
    write_json(packet_root / "attempt_ledger.json", attempt_ledger)
    counts = read_json(packet_root / "contact_counts.json")
    for field in (
        "physical_runs",
        "hardware_operations",
        "instrument_operations",
        "adc_or_dac_operations",
    ):
        counts[field] *= 2
    write_json(packet_root / "contact_counts.json", counts)
    adjudication = read_json(packet_root / "adjudication.json")
    adjudication["raw_root_receipt_sha256"] = selected_hash
    write_json(packet_root / "adjudication.json", adjudication)
    bundle["prior_raw_receipts"] = (prior_receipt_path,)
    reseal_final(bundle)


def rechain_replacement_attempt(bundle: dict[str, Any]) -> None:
    packet_root: Path = bundle["packet_root"]
    private_key: Ed25519PrivateKey = bundle["private_key"]
    prior_receipt_path: Path = bundle["prior_raw_receipts"][0]
    prior_prefix = packet_root / "prior_attempts" / "attempt-1"

    prior_raw_manifest_path = prior_prefix / "raw_manifest.json"
    prior_raw_manifest = read_json(prior_raw_manifest_path)
    prior_raw_manifest["files"] = {
        relative: sha256_bytes((prior_prefix / relative).read_bytes())
        for relative in sorted(prior_raw_manifest["files"])
    }
    write_json(prior_raw_manifest_path, prior_raw_manifest)
    prior_manifest_path = prior_prefix / "manifest.sha256.json"
    prior_manifest = read_json(prior_manifest_path)
    prior_manifest["files"] = {
        path.relative_to(prior_prefix).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(prior_prefix.rglob("*"))
        if path.is_file() and path != prior_manifest_path
    }
    write_json(prior_manifest_path, prior_manifest)

    prior_receipt = read_json(prior_receipt_path)
    prior_receipt["raw_manifest_sha256"] = sha256_bytes(prior_raw_manifest_path.read_bytes())
    prior_receipt.pop("signature_base64")
    write_json(prior_receipt_path, signed_receipt(prior_receipt, private_key))
    prior_receipt_hash = sha256_bytes(prior_receipt_path.read_bytes())

    selected_receipt = read_json(bundle["raw_receipt"])
    selected_receipt["previous_receipt_sha256"] = prior_receipt_hash
    selected_receipt.pop("signature_base64")
    write_json(bundle["raw_receipt"], signed_receipt(selected_receipt, private_key))
    selected_receipt_hash = sha256_bytes(bundle["raw_receipt"].read_bytes())

    ledger_path = packet_root / "attempt_ledger.json"
    ledger = read_json(ledger_path)
    ledger["attempts"][0]["raw_root_receipt_sha256"] = prior_receipt_hash
    ledger["attempts"][0]["preserved_invalid_packet_manifest_sha256"] = sha256_bytes(
        prior_manifest_path.read_bytes()
    )
    ledger["attempts"][1]["raw_root_receipt_sha256"] = selected_receipt_hash
    write_json(ledger_path, ledger)
    adjudication = read_json(packet_root / "adjudication.json")
    adjudication["raw_root_receipt_sha256"] = selected_receipt_hash
    write_json(packet_root / "adjudication.json", adjudication)
    reseal_final(bundle)


def packet_vector(mutation: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="p0_conformance_") as temporary:
        bundle = build_packet(Path(temporary))
        packet_root: Path = bundle["packet_root"]
        private_key: Ed25519PrivateKey = bundle["private_key"]
        if mutation in {
            "replacement",
            "replacement_missing_preserved_packet",
            "attempt_replay",
            "duplicate_attempt_id",
            "replacement_configuration_drift",
            "replacement_assignment_drift",
            "replacement_prior_count",
        }:
            make_replacement_attempt(bundle)
        if mutation == "metrics_cross_hash":
            adjudication = read_json(packet_root / "adjudication.json")
            adjudication["metrics_bundle_sha256"] = fixed_hash("wrong-metrics-bundle")
            write_json(packet_root / "adjudication.json", adjudication)
            reseal_final(bundle)
        elif mutation == "raw_manifest_cross_hash":
            raw_receipt = read_json(bundle["raw_receipt"])
            raw_receipt["raw_manifest_sha256"] = fixed_hash("wrong-raw-manifest")
            raw_receipt.pop("signature_base64")
            write_json(bundle["raw_receipt"], signed_receipt(raw_receipt, private_key))
            rebind_selected_raw(bundle)
            reseal_final(bundle)
        elif mutation == "receipt_chain":
            final_receipt = read_json(bundle["final_receipt"])
            final_receipt["previous_receipt_sha256"] = ZERO_SHA256
            final_receipt.pop("signature_base64")
            write_json(bundle["final_receipt"], signed_receipt(final_receipt, private_key))
        elif mutation == "early_reveal":
            reveal = read_json(packet_root / "arm_assignment.reveal.json")
            reveal["released_utc"] = "2026-07-15T00:00:00.000000Z"
            write_json(packet_root / "arm_assignment.reveal.json", reveal)
            reseal_final(bundle)
        elif mutation == "calibration_chronology":
            calibration_receipt = read_json(bundle["calibration_receipt"])
            calibration_receipt["created_utc"] = "2026-07-16T00:00:01.250000Z"
            calibration_receipt.pop("signature_base64")
            write_json(
                bundle["calibration_receipt"],
                signed_receipt(calibration_receipt, private_key),
            )
            calibration_receipt_hash = sha256_bytes(bundle["calibration_receipt"].read_bytes())
            raw_receipt = read_json(bundle["raw_receipt"])
            raw_receipt["calibration_root_receipt_sha256"] = calibration_receipt_hash
            raw_receipt["previous_receipt_sha256"] = calibration_receipt_hash
            raw_receipt.pop("signature_base64")
            write_json(bundle["raw_receipt"], signed_receipt(raw_receipt, private_key))
            rebind_selected_raw(bundle)
            reseal_final(bundle)
        elif mutation == "early_acquisition":
            ledger_path = packet_root / "attempt_ledger.json"
            ledger = read_json(ledger_path)
            early = "2026-07-16T00:00:00.250000Z"
            ledger["attempts"][0]["acquisition_started_utc"] = early
            write_json(ledger_path, ledger)
            raw_receipt = read_json(bundle["raw_receipt"])
            raw_receipt["acquisition_started_utc"] = early
            raw_receipt.pop("signature_base64")
            write_json(bundle["raw_receipt"], signed_receipt(raw_receipt, private_key))
            rebind_selected_raw(bundle)
            reseal_final(bundle)
        elif mutation == "late_acquisition_completion":
            ledger_path = packet_root / "attempt_ledger.json"
            ledger = read_json(ledger_path)
            late = "2026-07-16T00:00:01.250000Z"
            ledger["attempts"][0]["acquisition_completed_utc"] = late
            write_json(ledger_path, ledger)
            raw_receipt = read_json(bundle["raw_receipt"])
            raw_receipt["acquisition_completed_utc"] = late
            raw_receipt.pop("signature_base64")
            write_json(bundle["raw_receipt"], signed_receipt(raw_receipt, private_key))
            rebind_selected_raw(bundle)
            reseal_final(bundle)
        elif mutation == "signature":
            final_receipt = read_json(bundle["final_receipt"])
            signature = final_receipt["signature_base64"]
            final_receipt["signature_base64"] = ("A" if signature[0] != "A" else "B") + signature[1:]
            write_json(bundle["final_receipt"], final_receipt)
        elif mutation == "control_membership":
            controls = read_json(packet_root / "controls.json")
            controls["outcomes"].pop(next(iter(controls["outcomes"])))
            write_json(packet_root / "controls.json", controls)
            reseal_final(bundle)
        elif mutation == "control_evidence_submanifest":
            controls = read_json(packet_root / "controls.json")
            control_id = controls["ordered_control_ids"][0]
            evidence_path = packet_root / "control_evidence" / f"{control_id}.json"
            evidence = read_json(evidence_path)
            evidence["files"] = {
                "packet.json": sha256_bytes((packet_root / "packet.json").read_bytes())
            }
            write_json(evidence_path, evidence)
            controls["outcomes"][control_id]["evidence_sha256"] = sha256_bytes(
                evidence_path.read_bytes()
            )
            write_json(packet_root / "controls.json", controls)
            rebind_controls(packet_root)
            reseal_final(bundle)
        elif mutation == "relation_arm_mapping":
            relation_path = packet_root / "derived" / "relation_metrics.json"
            relation = read_json(relation_path)
            relation["arm_pi_id"] = relation["arm_0_id"]
            write_json(relation_path, relation)
            rebind_metrics(packet_root)
            reseal_final(bundle)
        elif mutation == "relation_negative_metric":
            relation_path = packet_root / "derived" / "relation_metrics.json"
            relation = read_json(relation_path)
            relation["epsilon_neg"] = "-0.001"
            write_json(relation_path, relation)
            rebind_metrics(packet_root)
            reseal_final(bundle)
        elif mutation == "contact_count":
            counts_path = packet_root / "contact_counts.json"
            counts = read_json(counts_path)
            counts["physical_runs"] -= 1
            write_json(counts_path, counts)
            reseal_final(bundle)
        elif mutation == "replacement_missing_preserved_packet":
            missing = packet_root / "prior_attempts" / "attempt-1" / "contact_counts.json"
            missing.unlink()
            reseal_final(bundle)
        elif mutation == "replacement_configuration_drift":
            topology_path = packet_root / "topology.json"
            topology = read_json(topology_path)
            topology["star_reference_node"] = "CARRIER_REF_DRIFT"
            write_json(topology_path, topology)
            reseal_final(bundle)
        elif mutation == "replacement_assignment_drift":
            assignment_path = (
                packet_root
                / "prior_attempts"
                / "attempt-1"
                / "arm_assignment.sealed.json"
            )
            assignment = read_json(assignment_path)
            assignment["custodian_id"] = "fixture-custodian-drift"
            write_json(assignment_path, assignment)
            rechain_replacement_attempt(bundle)
        elif mutation == "replacement_prior_count":
            counts_path = (
                packet_root / "prior_attempts" / "attempt-1" / "contact_counts.json"
            )
            counts = read_json(counts_path)
            counts["physical_runs"] -= 1
            write_json(counts_path, counts)
            rechain_replacement_attempt(bundle)
        elif mutation == "attempt_replay":
            ledger_path = packet_root / "attempt_ledger.json"
            ledger = read_json(ledger_path)
            ledger["attempts"][0]["phase_content_opened"] = True
            write_json(ledger_path, ledger)
            reseal_final(bundle)
        elif mutation == "duplicate_attempt_id":
            ledger_path = packet_root / "attempt_ledger.json"
            ledger = read_json(ledger_path)
            ledger["attempts"][1]["attempt_id"] = ledger["attempts"][0]["attempt_id"]
            write_json(ledger_path, ledger)
            reseal_final(bundle)
        elif mutation == "pass_coherence":
            controls = read_json(packet_root / "controls.json")
            first = controls["ordered_control_ids"][0]
            controls["outcomes"][first]["outcome"] = "FAIL"
            write_json(packet_root / "controls.json", controls)
            rebind_controls(packet_root)
            reseal_final(bundle)
        elif mutation not in {"none", "structural_scope", "replacement"}:
            raise ValueError(f"unknown packet mutation: {mutation}")
        try:
            result = validate_packet(
                packet_root,
                bundle["calibration_receipt"],
                bundle["raw_receipt"],
                bundle["final_receipt"],
                tuple(bundle.get("prior_raw_receipts", ())),
            )
        except PacketValidationError:
            return False
        if mutation == "structural_scope":
            return (
                result["structural_conformance"] is True
                and result["scientific_authority"] is False
                and result["scientific_pass"] is False
                and result["validation_scope"] == "STRUCTURAL_PREVIEW_ONLY"
            )
        return True


def check_vector(
    vector: dict[str, Any], validator: Draft202012Validator
) -> tuple[bool, str]:
    vector_id = str(vector["id"])
    inputs = dict(vector["input"])
    expected = dict(vector["expected"])

    if vector_id == "ideal_antipode":
        z_0 = as_complex(inputs["z_0"])
        z_pi = as_complex(inputs["z_pi"])
        theta_0 = [float(value) for value in inputs["theta_0"]]
        theta_pi = [float(value) for value in inputs["theta_pi"]]
        epsilon_neg = norm2([a + b for a, b in zip(z_pi, z_0, strict=True)]) / (
            0.5 * (norm2(z_pi) + norm2(z_0))
        )
        epsilon_a = norm2(
            [abs(a) - abs(b) for a, b in zip(z_pi, z_0, strict=True)]
        ) / (0.5 * (norm2([abs(a) for a in z_pi]) + norm2([abs(b) for b in z_0])))
        epsilon_phi = abs(
            circular_mean(
                [wrap(a - b - math.pi) for a, b in zip(theta_pi, theta_0, strict=True)]
            )
        )
        passed = all(
            (
                close(epsilon_neg, float(expected["epsilon_neg"])),
                close(epsilon_a, float(expected["epsilon_A"])),
                close(epsilon_phi, float(expected["epsilon_phi_rad"])),
            )
        )
    elif vector_id == "weighted_projection_recovery":
        count = int(inputs["sample_count"])
        phase = float(inputs["phase_start_rad"]) + float(inputs["phase_step_rad"]) * np.arange(count)
        signal = (
            float(inputs["I"]) * np.cos(phase)
            - float(inputs["Q"]) * np.sin(phase)
            + float(inputs["offset"])
        )
        beta, condition = weighted_projection(signal, phase)
        passed = (
            close(float(beta[0]), float(expected["I"]), 1e-13)
            and close(float(beta[1]), float(expected["Q"]), 1e-13)
            and close(float(beta[2]), float(expected["offset"]), 1e-13)
            and (condition <= 1e8) is bool(expected["condition_gate_pass"])
        )
    elif vector_id == "drive_phase_error_not_gauged_away":
        count = int(inputs["sample_count"])
        master = float(inputs["phase_step_rad"]) * np.arange(count)
        drive_phase = float(inputs["delta_command_rad"]) + float(inputs["phase_error_rad"])
        signal = np.cos(master + drive_phase)
        design = np.column_stack((np.cos(master), -np.sin(master), np.ones(count)))
        beta, covariance, _ = newey_west(design, signal)
        fitted = math.atan2(float(beta[1]), float(beta[0]))
        residual = wrap(fitted - float(inputs["delta_command_rad"]))
        denominator = float(beta[0] ** 2 + beta[1] ** 2)
        gradient = np.array([-beta[1] / denominator, beta[0] / denominator, 0.0])
        u_fit = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
        bound = abs(residual) + 1.96 * u_fit
        passed = close(residual, float(expected["e_drive_rad"]), 1e-12) and (
            bound <= 0.010
        ) is bool(expected["drive_fidelity_pass"])
    elif vector_id == "quarter_cycle_timing_control":
        frequency = float(inputs["frequency_hz"])
        delay = float(inputs["master_cycles_delayed"]) / frequency
        elapsed = float(inputs["elapsed_s"])
        omega = 2.0 * math.pi * frequency
        t0 = float(inputs["t_off_s"])
        delta = float(inputs["delta_rad"])
        first = complex(math.cos(omega * t0 + delta), math.sin(omega * t0 + delta)) * complex(
            math.cos(omega * elapsed), math.sin(omega * elapsed)
        ) / complex(math.cos(omega * (t0 + elapsed)), math.sin(omega * (t0 + elapsed)))
        second_t = t0 + delay
        second = complex(
            math.cos(omega * second_t + delta), math.sin(omega * second_t + delta)
        ) * complex(math.cos(omega * elapsed), math.sin(omega * elapsed)) / complex(
            math.cos(omega * (second_t + elapsed)), math.sin(omega * (second_t + elapsed))
        )
        change = wrap(math.atan2(second.imag, second.real) - math.atan2(first.imag, first.real))
        passed = close(change, float(expected["master_relative_phase_change_rad"]), 1e-11) and not bool(
            expected["matched_evidence_eligible"]
        )
    elif vector_id == "wrap_positive_pi_to_negative_pi":
        passed = close(wrap(float(inputs["x_rad"])), float(expected["output_rad"]))
    elif vector_id == "flat_wave_reject":
        accepted = float(inputs["amplitude"]) >= 10.0 * float(inputs["sigma_A"])
        passed = accepted is bool(expected["window_accept"])
    elif vector_id == "near_singular_projection_reject":
        count = int(inputs["sample_count"])
        phase = float(inputs["phase_step_rad"]) * np.arange(count)
        signal = np.cos(phase)
        try:
            _, condition = weighted_projection(signal, phase)
        except np.linalg.LinAlgError:
            condition = math.inf
        passed = (condition <= float(inputs["maximum_condition"])) is bool(
            expected["condition_gate_pass"]
        )
    elif vector_id == "matched_alignment":
        g_0 = int(inputs["n_admit_0"]) - int(inputs["n_gate_0"])
        g_pi = int(inputs["n_admit_pi"]) - int(inputs["n_gate_pi"])
        common = max(g_0, g_pi)
        start_0 = int(inputs["n_gate_0"]) + common
        start_pi = int(inputs["n_gate_pi"]) + common
        passed = (
            start_0 == int(expected["start_0"])
            and start_pi == int(expected["start_pi"])
            and start_0 - int(inputs["n_gate_0"]) == start_pi - int(inputs["n_gate_pi"])
        )
    elif vector_id == "noise_mad_maximum":
        sigma_i = max(
            *(median_mad(values) for values in inputs["controls_I"]),
            float(inputs["q_I"]) / math.sqrt(12.0),
        )
        sigma_q = max(
            *(median_mad(values) for values in inputs["controls_Q"]),
            float(inputs["q_Q"]) / math.sqrt(12.0),
        )
        passed = close(sigma_i, float(expected["sigma_I"])) and close(
            sigma_q, float(expected["sigma_Q"])
        )
    elif vector_id == "hac_regression":
        u = np.asarray(inputs["u"], dtype=np.float64)
        response = np.asarray(inputs["response"], dtype=np.float64)
        design = np.column_stack((np.ones(u.size), u))
        beta, covariance, r_squared = newey_west(design, response)
        passed = (
            close(float(beta[1]), float(expected["slope"]))
            and close(float(covariance[1, 1]), float(expected["cov_slope"]), 1e-14)
            and close(r_squared, float(expected["r_squared"]), 1e-12)
        )
    elif vector_id == "block_jackknife":
        values = np.asarray(inputs["values"], dtype=np.float64)
        se = block_jackknife_mean(values, int(inputs["block_size"]))
        passed = close(se, float(expected["se_jk"]), 1e-12)
    elif vector_id.startswith("schema_"):
        accepted = validator.is_valid(inputs["instance"])
        passed = accepted is bool(expected["schema_accept"])
    elif vector_id == "conformance_identity_missing_duplicate_reject":
        candidate_ids = list(EXPECTED_VECTOR_IDS[:-1]) + [EXPECTED_VECTOR_IDS[0]]
        candidate = {
            "dtype": EXPECTED_DTYPE,
            "fixture_version": EXPECTED_FIXTURE_VERSION,
            "vectors": [{"id": vector_id} for vector_id in candidate_ids],
        }
        passed = conformance_identity_valid(candidate) is bool(
            expected["identity_accept"]
        )
    elif vector_id.startswith("packet_"):
        accepted = packet_vector(str(inputs["mutation"]))
        passed = accepted is bool(expected["packet_accept"])
    else:
        raise ValueError(f"unknown conformance vector: {vector_id}")

    return passed, vector_id


def main() -> int:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    if not conformance_identity_valid(vectors):
        payload = {
            "check_count": 0,
            "checks": [],
            "error": "conformance fixture identity/order/count mismatch",
            "pass": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    schema = json.loads(SCHEMAS.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    checks = []
    for vector in vectors["vectors"]:
        passed, vector_id = check_vector(vector, validator)
        checks.append({"id": vector_id, "pass": passed})
    payload = {
        "check_count": len(checks),
        "checks": checks,
        "pass": all(check["pass"] for check in checks),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
