"""Exact non-authorizing Phase 6 V2 spectral-calibration contract."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))
from waveform_reference import CODEBOOK  # noqa: E402

FALSE_AUTHORIZATIONS = {
    "acquisition_authorized": False,
    "restoration_authorized": False,
    "target_coupling_authorized": False,
    "small_wall_authorized": False,
}
ROUTE_CORES = {"v4s5": [4, 5], "v2s3": [2, 3]}
RUNTIME_PARAMETERS = {
    "pin_khz": 1600000,
    "slot_s": 0.5,
    "off_window_s": 0.5,
    "read_hz": 4000,
    "temperature_veto_c": 68.0,
    "automatic_retry": False,
}
THRESHOLDS = {
    "minimum_sender_on_off_separation_sigma": 5.0,
    "maximum_phase_error_radians": 0.39269908169872414,
    "maximum_sign_pi_error_radians": 0.39269908169872414,
    "amplitude_monotonicity_required": True,
    "maximum_off_bin_to_requested_ratio": 0.5,
    "maximum_frequency_deviation_fraction": 0.01,
    "maximum_temperature_c": 68.0,
    "minimum_repeated_session_complex_correlation": 0.8,
    "cross_route_pass_required": True,
    "final_verdict_rule": "ALL_GROUPS_AND_BOTH_REBOOT_PARTITIONS_AND_BOTH_ROUTES_PASS",
}


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _open_exclusive(path: Path, mode: str, **kwargs) -> BinaryIO | TextIO:
    return path.open(mode, **kwargs)


def write_exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_exclusive(path, "xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def write_immutable(path: Path, value: dict) -> str:
    payload = canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or sidecar.exists():
        raise FileExistsError("immutable artifact or checksum already exists")
    json_created = False
    try:
        with _open_exclusive(path, "xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        json_created = True
        with _open_exclusive(sidecar, "x", encoding="ascii", newline="\n") as output:
            output.write(f"{digest}  {path.name}\n")
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        if json_created:
            path.unlink(missing_ok=True)
        raise
    return digest


def _sources_for_sign(sign: int) -> list[int]:
    return [index for index, value in enumerate(CODEBOOK[0]) if int(value) == sign]


def calibration_windows(session_id: str) -> list[dict]:
    windows: list[dict] = []
    for tone in range(12):
        for amplitude in (1, 2, 3):
            for theta in range(8):
                for sign in (1, -1):
                    sources = _sources_for_sign(sign)
                    receiver_source = sources[(tone + theta) % len(sources)]
                    unshared = (tone + amplitude + theta + (sign < 0)) % 2 == 1
                    sender_source = sources[(tone + theta + (1 if unshared else 0)) % len(sources)]
                    sender_theta = (theta + (1 if unshared else 0)) % 8
                    index = len(windows)
                    scramble_digest = (
                        hashlib.sha256(f"{session_id}:{index}:scramble".encode()).hexdigest()
                        if unshared else "0" * 64
                    )
                    windows.append({
                        "window_index": index,
                        "session_id": session_id,
                        "stage": "V2_SPECTRAL_CALIBRATION_SENDER_ON",
                        "block_id": f"tone_{tone:02d}",
                        "family": "scramble" if unshared else "calibration",
                        "actual_mode": "basis",
                        "declared_mode": "basis",
                        "executed_tone_order": "ASC",
                        "declared_tone_order": "ASC",
                        "measurement_mode": "lockin_and_raw_ring",
                        "drive_on": True,
                        "sender_off_required": False,
                        "physical_tone_index": tone,
                        "receiver_codeword_source_index": receiver_source,
                        "sender_codeword_source_index": sender_source,
                        "receiver_theta_idx": theta,
                        "sender_theta_idx": sender_theta,
                        "shared_schedule": not unshared,
                        "scramble_key_digest": scramble_digest,
                        "amplitude_level": amplitude,
                        "expected_code_sign": sign,
                    })
        index = len(windows)
        windows.append({
            "window_index": index,
            "session_id": session_id,
            "stage": "V2_SPECTRAL_CALIBRATION_SENDER_OFF",
            "block_id": f"tone_{tone:02d}",
            "family": "silent",
            "actual_mode": "null",
            "declared_mode": "null",
            "executed_tone_order": "ASC",
            "declared_tone_order": "ASC",
            "measurement_mode": "raw_ring_sender_off",
            "drive_on": False,
            "sender_off_required": True,
            "physical_tone_index": None,
            "receiver_codeword_source_index": None,
            "sender_codeword_source_index": None,
            "receiver_theta_idx": None,
            "sender_theta_idx": None,
            "shared_schedule": True,
            "scramble_key_digest": "0" * 64,
            "amplitude_level": 0,
            "expected_code_sign": 0,
            "sender_off_control_for_tone_index": tone,
        })
    return windows


def build_plan() -> dict:
    sessions = []
    for route in ("v4s5", "v2s3"):
        for partition in ("PRE_REBOOT_REPETITION", "POST_REBOOT_REPETITION"):
            session_id = f"{route}_{partition.lower()}_0"
            windows = calibration_windows(session_id)
            sessions.append({
                "session_id": session_id,
                "route": route,
                "route_cores": ROUTE_CORES[route],
                "partition": partition,
                "frequency_settling_required": True,
                "window_count": len(windows),
                "windows": windows,
            })
    per_session = {session["window_count"] for session in sessions}
    if per_session != {588}:
        raise AssertionError(f"unexpected mechanically derived session counts: {per_session}")
    per_route = {
        route: sum(session["window_count"] for session in sessions if session["route"] == route)
        for route in ROUTE_CORES
    }
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_PLAN_V2",
        "execution_class": "ENGINEERING_QUALIFICATION_NOT_SCIENTIFIC_ACQUISITION",
        "sessions": sessions,
        "session_ids": [session["session_id"] for session in sessions],
        "session_count": len(sessions),
        "windows_per_session": 588,
        "windows_per_route": per_route,
        "total_window_count": sum(per_route.values()),
        "count_derivation": "12 tones * 3 amplitudes * 8 theta * 2 signs + 12 sender-off controls",
        "runtime_parameters": RUNTIME_PARAMETERS,
        "analysis_thresholds": THRESHOLDS,
        "calibration_authorized": False,
        **FALSE_AUTHORIZATIONS,
    }


def compile_sessions(plan: dict, output_root: Path) -> dict[str, str]:
    plan_digest = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    bindings: dict[str, str] = {}
    for session in plan["sessions"]:
        directory = output_root / session["session_id"]
        header = {
            "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V2",
            "campaign_source_commit": "0000000000000000000000000000000000000000",
            "campaign_plan_sha256": plan_digest,
            "session_id": session["session_id"],
            "route": session["route"],
            "partition": session["partition"],
            "window_count": session["window_count"],
            "frequency_settling_required": True,
            "restoration_authorized": False,
        }
        write_exclusive_bytes(directory / "session.json", canonical_bytes(header))
        windows_payload = b"".join(canonical_bytes(row) for row in session["windows"])
        write_exclusive_bytes(directory / "windows.jsonl", windows_payload)
        files = {}
        for name in ("session.json", "windows.jsonl"):
            path = directory / name
            files[name] = {"size": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {
            "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V2",
            "session_id": session["session_id"],
            "files": files,
        }
        write_exclusive_bytes(directory / "session_manifest.json", canonical_bytes(manifest))
        bindings[session["session_id"]] = hashlib.sha256(
            (directory / "session_manifest.json").read_bytes()
        ).hexdigest()
    return bindings


def build_source_bundle_manifest(bindings: dict[str, str]) -> dict:
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SOURCE_BUNDLE_MANIFEST_V1",
        "sessions": dict(sorted(bindings.items())),
    }


def validate_authorization(value: dict, expected_plan_sha256: str,
                           source_bundle: dict) -> None:
    required = {
        "schema_id", "calibration_authorized", *FALSE_AUTHORIZATIONS,
        "automatic_retry", "campaign_plan_sha256", "executor_commit",
        "executor_sha256", "source_bundle_sha256", "session_ids", "route_cores",
        "pin_khz", "slot_s", "off_window_s", "read_hz", "temperature_veto_c",
        "authorized_output_root", "authorized_by",
    }
    if set(value) != required:
        raise ValueError("authorization fields must exactly match schema")
    if value["schema_id"] != "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1":
        raise ValueError("wrong calibration authorization schema")
    if value["calibration_authorized"] is not True:
        raise ValueError("calibration is not authorized")
    for key, expected in FALSE_AUTHORIZATIONS.items():
        if value[key] is not expected:
            raise ValueError(f"{key} must remain false")
    if value["automatic_retry"] is not False:
        raise ValueError("automatic_retry must remain false")
    if value["campaign_plan_sha256"] != expected_plan_sha256:
        raise ValueError("campaign plan digest mismatch")
    source_bundle_digest = hashlib.sha256(canonical_bytes(source_bundle)).hexdigest()
    if value["source_bundle_sha256"] != source_bundle_digest:
        raise ValueError("source bundle digest mismatch")
    if value["route_cores"] != ROUTE_CORES:
        raise ValueError("route/core binding mismatch")
    if value["session_ids"] != list(source_bundle["sessions"]):
        raise ValueError("exact source-bundle session IDs required")
