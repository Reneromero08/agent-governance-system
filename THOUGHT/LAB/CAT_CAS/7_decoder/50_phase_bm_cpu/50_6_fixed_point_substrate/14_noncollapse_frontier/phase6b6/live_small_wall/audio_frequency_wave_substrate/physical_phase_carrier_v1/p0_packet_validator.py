"""Strict structural preview validator for the prospective P0 packet format.

This module is offline-only.  It validates already-existing packet bytes and
external receipts; it contains no acquisition, control, hardware interface, or
scientific authority.  A future raw-derived analyzer is required before P0C.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "P0_EVIDENCE_SCHEMAS.json"
ZERO_SHA256 = "0" * 64
REPARSE_POINT = 0x400
NULL_BASELINE_CONTROL_IDS = frozenset({"zero_drive", "resonator_removed", "dummy_c0"})

FIXED_PACKET_FILES = {
    "packet.json",
    "hardware_identity.json",
    "carrier_identity.json",
    "topology.json",
    "instruments.json",
    "calibration.json",
    "arm_assignment.sealed.json",
    "arm_assignment.reveal.json",
    "attempt_ledger.json",
    "derived/relation_metrics.json",
    "controls.json",
    "adjudication.json",
    "contact_counts.json",
    "raw_manifest.json",
    "manifest.sha256.json",
}

THRESHOLD_FIELDS = {
    "epsilon_neg": ("u95_neg", "T_neg", Decimal("0.100")),
    "epsilon_A": ("u95_A", "T_A", Decimal("0.050")),
    "epsilon_f": ("u95_f", "T_f", Decimal("0.000005")),
    "epsilon_tau": ("u95_tau", "T_tau", Decimal("0.050")),
    "epsilon_phi_rad": ("u95_phi_rad", "T_phi_rad", Decimal("0.050")),
    "epsilon_feed": ("u95_feed", "T_feed", Decimal("0.100")),
}


class PacketValidationError(ValueError):
    """A deterministic packet-law violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    require(not raw.startswith(b"\xef\xbb\xbf"), f"{label}: UTF-8 BOM forbidden")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"{label}: duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PacketValidationError(f"{label}: nonfinite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PacketValidationError(f"{label}: invalid canonical JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label}: root must be an object")
    require(canonical_bytes(value) == raw, f"{label}: noncanonical JSON bytes")
    return value


def load_json(path: Path, label: str | None = None) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return strict_json_bytes(raw, label or str(path)), raw


def collect_packet_files(root: Path) -> dict[str, Path]:
    root = root.resolve(strict=True)
    require(root.is_dir(), "packet root is not a directory")
    found: dict[str, Path] = {}
    normalized: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                stat = entry.stat(follow_symlinks=False)
                attributes = int(getattr(stat, "st_file_attributes", 0))
                require(not entry.is_symlink(), f"symlink forbidden: {entry.path}")
                require(
                    not attributes & REPARSE_POINT,
                    f"junction/reparse point forbidden: {entry.path}",
                )
                path = Path(entry.path)
                resolved = path.resolve(strict=True)
                try:
                    relative = resolved.relative_to(root).as_posix()
                except ValueError as exc:
                    raise PacketValidationError(
                        f"resolved path escapes packet root: {entry.path}"
                    ) from exc
                identity = relative.casefold()
                require(identity not in normalized, f"duplicate normalized path: {relative}")
                normalized.add(identity)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(resolved)
                elif entry.is_file(follow_symlinks=False):
                    found[relative] = resolved
                else:
                    raise PacketValidationError(f"non-file packet entry: {relative}")
    return found


def validate_schema(
    value: dict[str, Any], validator: Draft202012Validator, label: str
) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = "/".join(str(item) for item in first.absolute_path) or "<root>"
        raise PacketValidationError(f"{label}: schema {location}: {first.message}")


def decode_canonical_base64(value: str, expected_bytes: int, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise PacketValidationError(f"{label}: invalid base64") from exc
    require(len(decoded) == expected_bytes, f"{label}: wrong decoded byte count")
    require(base64.b64encode(decoded).decode("ascii") == value, f"{label}: noncanonical base64")
    return decoded


def receipt_message(receipt: dict[str, Any]) -> bytes:
    unsigned = dict(receipt)
    unsigned.pop("signature_base64", None)
    return canonical_bytes(unsigned)


def verify_receipt_signature(
    receipt: dict[str, Any], public_key: Ed25519PublicKey, label: str
) -> None:
    signature = decode_canonical_base64(receipt["signature_base64"], 64, label)
    try:
        public_key.verify(signature, receipt_message(receipt))
    except InvalidSignature as exc:
        raise PacketValidationError(f"{label}: Ed25519 signature invalid") from exc


def metric_bundle(paths: list[str], packet_files: dict[str, Path]) -> dict[str, Any]:
    return {
        "files": {
            path: sha256_bytes(packet_files[path].read_bytes()) for path in sorted(paths)
        }
    }


def validate_packet(
    packet_root: Path,
    calibration_receipt_path: Path,
    raw_receipt_path: Path,
    final_receipt_path: Path,
    prior_raw_receipt_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    control_ids = schema["$defs"]["controls"]["properties"]["ordered_control_ids"][
        "const"
    ]
    offline_control_ids = {
        "offline_synthetic_conformance",
        "offline_wrong_guard_analysis",
        "offline_phase_reference_inversion",
    }
    require(
        NULL_BASELINE_CONTROL_IDS <= set(control_ids),
        "required null-baseline controls are absent",
    )

    packet_files = collect_packet_files(packet_root)
    require(FIXED_PACKET_FILES <= set(packet_files), "fixed packet file missing")
    require(
        {f"control_evidence/{control_id}.json" for control_id in control_ids}
        <= set(packet_files),
        "control evidence file missing",
    )

    records: dict[str, dict[str, Any]] = {}
    for relative, path in sorted(packet_files.items()):
        if path.suffix == ".json":
            value, _ = load_json(path, relative)
            validate_schema(value, validator, relative)
            records[relative] = value

    calibration_receipt, calibration_receipt_bytes = load_json(
        calibration_receipt_path, "calibration receipt"
    )
    raw_receipt, raw_receipt_bytes = load_json(raw_receipt_path, "raw receipt")
    final_receipt, _ = load_json(final_receipt_path, "final receipt")
    prior_raw_receipts = [
        load_json(path, f"prior raw receipt {index}")
        for index, path in enumerate(prior_raw_receipt_paths, start=1)
    ]
    validate_schema(calibration_receipt, validator, "calibration receipt")
    validate_schema(raw_receipt, validator, "raw receipt")
    validate_schema(final_receipt, validator, "final receipt")
    for index, (receipt, _) in enumerate(prior_raw_receipts, start=1):
        validate_schema(receipt, validator, f"prior raw receipt {index}")

    run_ids = {record["run_id"] for record in records.values()}
    run_ids.update(
        (
            calibration_receipt["run_id"],
            raw_receipt["run_id"],
            final_receipt["run_id"],
            *(receipt["run_id"] for receipt, _ in prior_raw_receipts),
        )
    )
    require(len(run_ids) == 1, "run_id mismatch")
    run_id = next(iter(run_ids))

    final_manifest = records["manifest.sha256.json"]
    expected_final_paths = set(packet_files) - {"manifest.sha256.json"}
    require(
        set(final_manifest["files"]) == expected_final_paths,
        "final manifest coverage mismatch",
    )
    for relative, expected_hash in final_manifest["files"].items():
        require(
            sha256_bytes(packet_files[relative].read_bytes()) == expected_hash,
            f"final manifest hash mismatch: {relative}",
        )

    descriptor_paths = sorted(
        path
        for path in records
        if path.startswith("runs/") and path.endswith("/raw_descriptor.json")
    )
    require(len(descriptor_paths) >= 2, "at least two arm descriptors required")
    arm_ids: list[str] = []
    expected_raw_paths = {"arm_assignment.sealed.json"}
    expected_arm_paths: set[str] = set()
    for path in descriptor_paths:
        _, arm_id, _ = path.split("/", 2)
        require(records[path]["arm_id"] == arm_id, f"descriptor arm mismatch: {arm_id}")
        arm_ids.append(arm_id)
        expected_raw_paths.update(
            {
                f"runs/{arm_id}/raw_native.bin",
                f"runs/{arm_id}/raw_descriptor.json",
                f"runs/{arm_id}/environment.csv",
            }
        )
        expected_arm_paths.update(
            {
                f"runs/{arm_id}/raw_native.bin",
                f"runs/{arm_id}/raw_descriptor.json",
                f"runs/{arm_id}/raw.f64le",
                f"runs/{arm_id}/reference.f64le",
                f"runs/{arm_id}/switch_state.u8",
                f"runs/{arm_id}/environment.csv",
                f"derived/{arm_id}/iq.f64le",
                f"derived/{arm_id}/iq_start_index.u64le",
                f"derived/{arm_id}/metrics.json",
            }
        )
    require(expected_arm_paths <= set(packet_files), "arm evidence file missing")

    raw_manifest = records["raw_manifest.json"]
    require(set(raw_manifest["files"]) == expected_raw_paths, "raw manifest coverage mismatch")
    for relative, expected_hash in raw_manifest["files"].items():
        require(
            sha256_bytes(packet_files[relative].read_bytes()) == expected_hash,
            f"raw manifest hash mismatch: {relative}",
        )

    calibration = records["calibration.json"]
    instruments = records["instruments.json"]
    assignment = records["arm_assignment.sealed.json"]
    reveal = records["arm_assignment.reveal.json"]
    require(
        calibration["instrument_configuration_sha256"]
        == instruments["configuration_sha256"],
        "instrument configuration binding mismatch",
    )
    require(
        calibration["sealed_assignment_sha256"]
        == sha256_bytes(packet_files["arm_assignment.sealed.json"].read_bytes()),
        "sealed assignment binding mismatch",
    )
    assignment_key = decode_canonical_base64(reveal["key_base64"], 32, "assignment key")
    assignment_salt = decode_canonical_base64(reveal["salt_base64"], 32, "assignment salt")
    assignment_nonce = decode_canonical_base64(
        assignment["nonce_base64"], 12, "assignment nonce"
    )
    assignment_ciphertext = base64.b64decode(
        assignment["ciphertext_base64"], validate=True
    )
    require(
        base64.b64encode(assignment_ciphertext).decode("ascii")
        == assignment["ciphertext_base64"],
        "assignment ciphertext base64 noncanonical",
    )
    require(sha256_bytes(assignment_key) == assignment["key_sha256"], "assignment key hash mismatch")
    aad = {
        "record_type": "arm_assignment_aad",
        "run_id": run_id,
        "schema_version": "p0-evidence-v1",
    }
    require(sha256_object(aad) == assignment["aad_sha256"], "assignment AAD hash mismatch")
    try:
        plaintext_bytes = ChaCha20Poly1305(assignment_key).decrypt(
            assignment_nonce, assignment_ciphertext, canonical_bytes(aad)
        )
    except InvalidTag as exc:
        raise PacketValidationError("assignment AEAD authentication failed") from exc
    plaintext = strict_json_bytes(plaintext_bytes, "assignment plaintext")
    require(
        sha256_bytes(plaintext_bytes) == assignment["commitment_sha256"],
        "assignment commitment mismatch",
    )
    require(
        reveal["commitment_sha256"] == assignment["commitment_sha256"],
        "assignment reveal commitment mismatch",
    )
    require(
        plaintext.get("salt_base64") == base64.b64encode(assignment_salt).decode("ascii"),
        "assignment salt mismatch",
    )
    mapping = {"A": plaintext.get("A"), "B": plaintext.get("B")}
    require(
        mapping in ({"A": "0", "B": "pi"}, {"A": "pi", "B": "0"}),
        "assignment mapping is not complementary",
    )
    require(sha256_object(mapping) == reveal["mapping_sha256"], "assignment mapping hash mismatch")
    require(
        reveal["custodian_id"] == assignment["custodian_id"],
        "assignment custodian mismatch",
    )
    public_key_bytes = decode_canonical_base64(
        calibration["ed25519_public_key_base64"], 32, "calibration public key"
    )
    public_key_hash = sha256_bytes(public_key_bytes)
    require(
        public_key_hash == calibration["ed25519_public_key_sha256"],
        "calibration public-key hash mismatch",
    )
    public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)

    calibration_hash = sha256_bytes(packet_files["calibration.json"].read_bytes())
    calibration_receipt_hash = sha256_bytes(calibration_receipt_bytes)
    require(calibration_receipt["sequence_number"] == 1, "calibration receipt sequence must be 1")
    require(
        calibration_receipt["previous_receipt_sha256"] == ZERO_SHA256,
        "calibration receipt must be genesis",
    )
    require(
        calibration_receipt["calibration_sha256"] == calibration_hash,
        "calibration receipt payload hash mismatch",
    )
    require(
        calibration["created_utc"] < calibration_receipt["created_utc"],
        "calibration must precede its external receipt",
    )
    raw_chain = [*prior_raw_receipts, (raw_receipt, raw_receipt_bytes)]
    require(len(raw_chain) in (1, 2), "raw receipt chain must contain one or two attempts")
    previous_hash = calibration_receipt_hash
    previous_time = calibration_receipt["created_utc"]
    raw_hashes = []
    for index, (receipt, receipt_bytes) in enumerate(raw_chain, start=2):
        label = f"raw receipt {index - 1}"
        require(receipt["sequence_number"] == index, f"{label}: sequence mismatch")
        require(receipt["previous_receipt_sha256"] == previous_hash, f"{label}: chain mismatch")
        require(
            receipt["calibration_root_receipt_sha256"] == calibration_receipt_hash,
            f"{label}: calibration-root binding mismatch",
        )
        require(previous_time < receipt["created_utc"], f"{label}: chronology mismatch")
        receipt_hash = sha256_bytes(receipt_bytes)
        raw_hashes.append(receipt_hash)
        previous_hash = receipt_hash
        previous_time = receipt["created_utc"]
    raw_receipt_hash = raw_hashes[-1]
    require(
        raw_receipt["created_utc"]
        < reveal["released_utc"]
        < final_receipt["created_utc"],
        "assignment reveal chronology must be selected raw receipt < reveal < final receipt",
    )
    require(
        final_receipt["sequence_number"] == len(raw_chain) + 2,
        "final receipt sequence mismatch",
    )
    require(
        final_receipt["previous_receipt_sha256"] == raw_receipt_hash,
        "final receipt previous hash mismatch",
    )
    require(
        final_receipt["raw_root_receipt_sha256"] == raw_receipt_hash,
        "final receipt raw-root binding mismatch",
    )
    all_receipts = [
        ("calibration receipt", calibration_receipt),
        *((f"raw receipt {index}", receipt) for index, (receipt, _) in enumerate(raw_chain, start=1)),
        ("final receipt", final_receipt),
    ]
    for label, receipt in all_receipts:
        require(receipt["public_key_sha256"] == public_key_hash, f"{label}: key hash mismatch")
        require(receipt["medium_id"] == calibration["worm_medium_id"], f"{label}: medium mismatch")
        require(receipt["witness_id"] == calibration_receipt["witness_id"], f"{label}: witness mismatch")
        verify_receipt_signature(receipt, public_key, label)
    require(
        raw_receipt["raw_manifest_sha256"]
        == sha256_bytes(packet_files["raw_manifest.json"].read_bytes()),
        "raw receipt raw-manifest hash mismatch",
    )
    require(
        final_receipt["manifest_sha256"]
        == sha256_bytes(packet_files["manifest.sha256.json"].read_bytes()),
        "final receipt manifest hash mismatch",
    )

    attempts = records["attempt_ledger.json"]["attempts"]
    require(len(attempts) == len(raw_chain), "attempt ledger/receipt count mismatch")
    require(
        len({attempt["attempt_id"] for attempt in attempts}) == len(attempts),
        "attempt IDs must be unique",
    )
    require(
        [attempt["sequence_index"] for attempt in attempts]
        == list(range(1, len(attempts) + 1)),
        "attempt sequence mismatch",
    )
    require(
        [attempt["raw_root_receipt_sha256"] for attempt in attempts] == raw_hashes,
        "attempt receipt hash mismatch",
    )
    for attempt, (receipt, _) in zip(attempts, raw_chain, strict=True):
        require(
            calibration_receipt["created_utc"]
            < receipt["acquisition_started_utc"]
            < receipt["acquisition_completed_utc"]
            < receipt["created_utc"],
            "attempt acquisition must occur after calibration receipt and before raw receipt",
        )
        require(
            attempt["acquisition_started_utc"] == receipt["acquisition_started_utc"]
            and attempt["acquisition_completed_utc"]
            == receipt["acquisition_completed_utc"],
            "attempt ledger/acquisition receipt interval mismatch",
        )
    selected_attempt = attempts[-1]
    for path in descriptor_paths:
        descriptor = records[path]
        require(
            selected_attempt["acquisition_started_utc"]
            <= descriptor["acquisition_started_utc"]
            < descriptor["acquisition_completed_utc"]
            <= selected_attempt["acquisition_completed_utc"],
            f"selected descriptor acquisition interval mismatch: {path}",
        )
    require(sum(bool(attempt["selected_for_adjudication"]) for attempt in attempts) == 1, "one selected attempt required")
    require(attempts[-1]["selected_for_adjudication"] is True, "last attempt must be selected")
    require(
        attempts[-1]["outcome"] == "COMPLETE"
        and attempts[-1]["integrity_reason"] == "NONE",
        "selected attempt must complete without replacement reason",
    )
    require(
        attempts[-1]["preserved_invalid_packet_manifest_sha256"] == ZERO_SHA256,
        "selected attempt cannot name an invalid-packet manifest",
    )
    if len(attempts) == 2:
        first = attempts[0]
        require(
            first["outcome"] == "INTEGRITY_INVALID"
            and first["integrity_reason"] != "NONE"
            and first["phase_content_opened"] is False
            and first["selected_for_adjudication"] is False,
            "replacement requires one unopened integrity-invalid predecessor",
        )
        require(
            first["preserved_invalid_packet_manifest_sha256"] != ZERO_SHA256,
            "replacement requires a preserved invalid-packet manifest",
        )
        prior_prefix = f"prior_attempts/{first['attempt_id']}/"
        prior_manifest_path = prior_prefix + "manifest.sha256.json"
        require(prior_manifest_path in records, "preserved invalid-packet manifest missing")
        require(
            sha256_bytes(packet_files[prior_manifest_path].read_bytes())
            == first["preserved_invalid_packet_manifest_sha256"],
            "preserved invalid-packet manifest hash mismatch",
        )
        prior_manifest = records[prior_manifest_path]
        prior_payload_paths = {
            path[len(prior_prefix) :]
            for path in packet_files
            if path.startswith(prior_prefix) and path != prior_manifest_path
        }
        prior_required_paths = {
            "packet.json",
            "hardware_identity.json",
            "carrier_identity.json",
            "topology.json",
            "instruments.json",
            "calibration.json",
            "arm_assignment.sealed.json",
            "contact_counts.json",
            "raw_manifest.json",
        } | {
            relative
            for arm_id in arm_ids
            for relative in (
                f"runs/{arm_id}/raw_native.bin",
                f"runs/{arm_id}/raw_descriptor.json",
                f"runs/{arm_id}/environment.csv",
            )
        }
        require(
            prior_payload_paths == prior_required_paths,
            "preserved invalid acquisition packet coverage mismatch",
        )
        require(
            set(prior_manifest["files"]) == prior_payload_paths,
            "preserved invalid-packet manifest coverage mismatch",
        )
        for relative, expected_hash in prior_manifest["files"].items():
            require(
                sha256_bytes(packet_files[prior_prefix + relative].read_bytes())
                == expected_hash,
                f"preserved invalid-packet hash mismatch: {relative}",
            )
        require(
            records[prior_prefix + "packet.json"]["status"] == "INVALID",
            "preserved predecessor packet status must be INVALID",
        )
        prior_raw_manifest = records[prior_prefix + "raw_manifest.json"]
        require(
            set(prior_raw_manifest["files"]) == expected_raw_paths,
            "preserved predecessor raw-manifest coverage mismatch",
        )
        for relative, expected_hash in prior_raw_manifest["files"].items():
            require(
                sha256_bytes(packet_files[prior_prefix + relative].read_bytes())
                == expected_hash,
                f"preserved predecessor raw hash mismatch: {relative}",
            )
        require(
            prior_raw_receipts[0][0]["raw_manifest_sha256"]
            == sha256_bytes(packet_files[prior_prefix + "raw_manifest.json"].read_bytes()),
            "prior receipt does not bind preserved predecessor raw manifest",
        )
        for relative in (
            "hardware_identity.json",
            "carrier_identity.json",
            "topology.json",
            "instruments.json",
            "calibration.json",
            "arm_assignment.sealed.json",
        ):
            require(
                packet_files[prior_prefix + relative].read_bytes()
                == packet_files[relative].read_bytes(),
                f"replacement changed frozen bytes: {relative}",
            )
        for arm_id in arm_ids:
            prior_descriptor_path = prior_prefix + f"runs/{arm_id}/raw_descriptor.json"
            prior_descriptor = records[prior_descriptor_path]
            require(
                first["acquisition_started_utc"]
                <= prior_descriptor["acquisition_started_utc"]
                < prior_descriptor["acquisition_completed_utc"]
                <= first["acquisition_completed_utc"],
                f"predecessor descriptor acquisition interval mismatch: {arm_id}",
            )
        prior_counts = records[prior_prefix + "contact_counts.json"]
        require(prior_counts["physical_runs"] == len(arm_ids), "predecessor physical run count mismatch")
        require(prior_counts["hardware_operations"] >= len(arm_ids), "predecessor hardware count too small")
        require(prior_counts["instrument_operations"] >= len(arm_ids), "predecessor instrument count too small")
        require(prior_counts["adc_or_dac_operations"] >= len(arm_ids), "predecessor digitizer count too small")
        require(prior_counts["audio_play_or_record_operations"] == 0, "predecessor audio operation forbidden")
        require(prior_counts["remote_target_contacts"] == 0, "predecessor remote target contact forbidden")
    else:
        require(
            attempts[0]["preserved_invalid_packet_manifest_sha256"] == ZERO_SHA256,
            "single complete attempt cannot name an invalid-packet manifest",
        )

    controls = records["controls.json"]
    require(controls["ordered_control_ids"] == control_ids, "control order mismatch")
    require(set(controls["outcomes"]) == set(control_ids), "control membership mismatch")
    acquisition_ids: set[str] = set()
    physical_acquisition_ids: set[str] = set()
    for ordinal, control_id in enumerate(control_ids):
        evidence_path = f"control_evidence/{control_id}.json"
        evidence = records[evidence_path]
        require(evidence["control_id"] == control_id, f"control evidence ID mismatch: {control_id}")
        require(evidence["sequence_ordinal"] == ordinal, f"control ordinal mismatch: {control_id}")
        acquisition_id = evidence["acquisition_id"]
        require(acquisition_id not in acquisition_ids, f"control acquisition reused: {control_id}")
        acquisition_ids.add(acquisition_id)
        is_physical = control_id not in offline_control_ids
        require(
            evidence["physical_authority_consumed"] is is_physical,
            f"control authority flag mismatch: {control_id}",
        )
        require(
            evidence["execution_class"] == ("PHYSICAL" if is_physical else "OFFLINE"),
            f"control execution class mismatch: {control_id}",
        )
        require(
            controls["outcomes"][control_id]["evidence_sha256"]
            == sha256_bytes(packet_files[evidence_path].read_bytes()),
            f"control evidence record hash mismatch: {control_id}",
        )
        if is_physical:
            physical_acquisition_ids.add(acquisition_id)
            required_evidence_paths = {
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
            required_evidence_paths = {f"offline/{control_id}.bin"}
        require(
            set(evidence["files"]) == required_evidence_paths,
            f"control-specific evidence membership mismatch: {control_id}",
        )
        for relative, expected_hash in evidence["files"].items():
            require(relative in final_manifest["files"], f"control evidence path missing: {control_id}")
            require(
                final_manifest["files"][relative] == expected_hash,
                f"control evidence payload hash mismatch: {control_id}",
            )
    require(set(arm_ids) == physical_acquisition_ids, "physical run/control identity mismatch")
    ledger_payload = {
        "ordered_control_ids": controls["ordered_control_ids"],
        "outcomes": controls["outcomes"],
    }
    ledger_hash = sha256_object(ledger_payload)
    require(controls["ledger_sha256"] == ledger_hash, "controls ledger hash mismatch")

    metrics_paths = sorted(
        path
        for path in records
        if (path.startswith("derived/") and path.endswith("/metrics.json"))
        or path == "derived/relation_metrics.json"
    )
    require(
        set(metrics_paths)
        == {f"derived/{arm_id}/metrics.json" for arm_id in arm_ids}
        | {"derived/relation_metrics.json"},
        "metrics bundle membership mismatch",
    )
    metrics_hash = sha256_object(metric_bundle(metrics_paths, packet_files))

    adjudication = records["adjudication.json"]
    require(
        adjudication["raw_root_receipt_sha256"] == raw_receipt_hash,
        "adjudication raw-root binding mismatch",
    )
    require(
        adjudication["control_ledger_sha256"] == ledger_hash,
        "adjudication controls binding mismatch",
    )
    require(
        adjudication["metrics_bundle_sha256"] == metrics_hash,
        "adjudication metrics binding mismatch",
    )

    if adjudication["decision"] == "PASS":
        require(records["packet.json"]["status"] == "ADJUDICATED", "PASS packet status mismatch")
        require(not adjudication["kill_ids"], "PASS cannot contain kill IDs")
        require(
            all(item["outcome"] == "PASS" for item in controls["outcomes"].values()),
            "PASS requires every control outcome PASS",
        )
        for arm_id in arm_ids:
            arm_metric = records[f"derived/{arm_id}/metrics.json"]
            require(arm_metric["quality_gate_pass"] is True, f"arm quality failed: {arm_id}")
            require(arm_metric["usable_cycles"] >= 256, f"arm usable cycles failed: {arm_id}")
            require(Decimal(arm_metric["f_hat_hz"]) > 0, f"arm frequency nonpositive: {arm_id}")
            require(Decimal(arm_metric["tau_a_s"]) > 0, f"arm decay nonpositive: {arm_id}")
            require(Decimal(arm_metric["q_hat"]) > 0, f"arm Q nonpositive: {arm_id}")
            require(Decimal(arm_metric["u95_f_hz"]) >= 0, f"arm frequency U95 negative: {arm_id}")
            require(Decimal(arm_metric["u95_tau_s"]) >= 0, f"arm decay U95 negative: {arm_id}")
        relation = records["derived/relation_metrics.json"]
        expected_arm_0 = "A" if mapping["A"] == "0" else "B"
        expected_arm_pi = "A" if mapping["A"] == "pi" else "B"
        require(
            relation["arm_0_id"] == expected_arm_0
            and relation["arm_pi_id"] == expected_arm_pi
            and relation["arm_0_id"] != relation["arm_pi_id"],
            "relation arms do not match the distinct revealed assignment",
        )
        for metric, (uncertainty, threshold, cap) in THRESHOLD_FIELDS.items():
            threshold_value = Decimal(calibration["thresholds"][threshold])
            require(Decimal("0") <= threshold_value <= cap, f"threshold cap failed: {threshold}")
            require(Decimal(relation[metric]) >= 0, f"relation metric negative: {metric}")
            require(Decimal(relation[uncertainty]) >= 0, f"relation U95 negative: {uncertainty}")
            require(
                Decimal(relation[metric]) + Decimal(relation[uncertainty])
                <= threshold_value,
                f"relation threshold failed: {metric}",
            )
        for field in (
            "antipode_in_u95",
            "zero_excluded_u95",
            "plus_minus_pi_over_2_excluded_u95",
            "residual_zero_in_u95",
        ):
            require(relation[field] is True, f"phase confidence failed: {field}")
        counts = records["contact_counts.json"]
        expected_physical_runs = len(physical_acquisition_ids) * len(attempts)
        require(
            counts["physical_runs"] == expected_physical_runs,
            "physical run count does not match control/attempt ledger",
        )
        require(counts["hardware_operations"] >= expected_physical_runs, "hardware operation count too small")
        require(counts["instrument_operations"] >= expected_physical_runs, "instrument operation count too small")
        require(counts["adc_or_dac_operations"] >= expected_physical_runs, "digitizer operation count too small")
        require(counts["audio_play_or_record_operations"] == 0, "audio operation forbidden")
        require(counts["remote_target_contacts"] == 0, "remote target contact forbidden")

    return {
        "arm_count": len(arm_ids),
        "control_count": len(control_ids),
        "file_count": len(packet_files),
        "manifest_sha256": final_receipt["manifest_sha256"],
        "scientific_authority": False,
        "scientific_pass": False,
        "structural_conformance": True,
        "raw_root_receipt_sha256": raw_receipt_hash,
        "run_id": run_id,
        "validation_scope": "STRUCTURAL_PREVIEW_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("calibration_receipt", type=Path)
    parser.add_argument("raw_receipt", type=Path)
    parser.add_argument("final_receipt", type=Path)
    parser.add_argument("--prior-raw-receipt", action="append", default=[], type=Path)
    args = parser.parse_args()
    try:
        result = validate_packet(
            args.packet_root,
            args.calibration_receipt,
            args.raw_receipt,
            args.final_receipt,
            tuple(args.prior_raw_receipt),
        )
    except PacketValidationError as exc:
        print(json.dumps({"error": str(exc), "pass": False}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
