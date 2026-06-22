#!/usr/bin/env python3
"""Fail-closed analyzer for Phase 6 V2 spectral calibration evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from calibration_contract import (
    FALSE_AUTHORIZATIONS,
    canonical_bytes,
    validate_authorization,
    write_immutable,
)

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
import sys
sys.path.insert(0, str(ANALYSIS))
from waveform_reference import intended_v2_gate, lockin, phase_index, tone_hz  # noqa: E402

RAW_DTYPE = np.dtype([("timestamp_tsc", "<u8"), ("ring_period", "<f8")])
RUN_FILES = {
    "run.json", "session.json", "windows.jsonl", "window_results.csv",
    "raw_samples.bin", "telemetry.csv", "stdout.log", "stderr.log",
    "orchestrator_stdout.log", "orchestrator_stderr.log",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_run_manifest(run_dir: Path) -> dict:
    manifest = load_json(run_dir / "run_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != RUN_FILES:
        raise ValueError("run manifest file set mismatch")
    actual = {path.name for path in run_dir.iterdir() if path.is_file()}
    if actual != RUN_FILES | {"run_manifest.json"}:
        raise ValueError("run directory file set mismatch")
    for name, binding in files.items():
        path = run_dir / name
        if binding != {"size": path.stat().st_size, "sha256": sha256(path)}:
            raise ValueError(f"run manifest binding mismatch: {name}")
    return manifest


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def angle_error(actual: float, expected: float) -> float:
    return abs(float(np.angle(np.exp(1j * (actual - expected)))))


def analyze_run(run_dir: Path, plan: dict, authorization: dict,
                source_bundle: dict) -> dict:
    verify_run_manifest(run_dir)
    run = load_json(run_dir / "run.json")
    session = load_json(run_dir / "session.json")
    schedule = read_rows(run_dir / "windows.jsonl")
    results = read_csv(run_dir / "window_results.csv")
    plan_digest = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    authorization = dict(authorization)
    authorization_digest = authorization.pop("artifact_sha256", None)
    validate_authorization(authorization, plan_digest, source_bundle)
    session_id = session.get("session_id")
    plan_sessions = {item["session_id"]: item for item in plan["sessions"]}
    if session_id not in plan_sessions:
        raise ValueError("session absent from exact calibration plan")
    planned = plan_sessions[session_id]
    if schedule != planned["windows"] or len(results) != len(schedule):
        raise ValueError("ordered window set differs from plan")
    if run.get("session_id") != session_id or run.get("route") != planned["route"]:
        raise ValueError("run/session/route binding mismatch")
    if run.get("campaign_plan_sha256") != plan_digest:
        raise ValueError("run campaign-plan binding mismatch")
    if run.get("executor_git_commit") != authorization["executor_commit"]:
        raise ValueError("executor commit binding mismatch")
    if run.get("authorization_artifact_sha256") != authorization_digest:
        raise ValueError("authorization digest binding mismatch")
    if run.get("execution_class") != "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION":
        raise ValueError("evidence is not V2 calibration")
    if run.get("hardware_executed") is not True:
        raise ValueError("real hardware calibration evidence required")
    for key, expected in FALSE_AUTHORIZATIONS.items():
        if run.get(key) is not expected:
            raise ValueError(f"calibration evidence cannot set {key}")

    raw_path = run_dir / "raw_samples.bin"
    counts = np.asarray([int(row["sample_count"]) for row in results], dtype=np.int64)
    if np.any(counts < 4):
        raise ValueError("every calibration window requires at least four samples")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if raw_path.stat().st_size != int(offsets[-1]) * RAW_DTYPE.itemsize:
        raise ValueError("raw binary size/record count mismatch")
    raw = np.memmap(raw_path, dtype=RAW_DTYPE, mode="r")
    if len(raw) != int(offsets[-1]):
        raise ValueError("raw trailing or missing records")

    tsc_hz = float(run["tsc_calibration_hz"])
    measurements = []
    off_by_tone: dict[int, list[float]] = {tone: [] for tone in range(12)}
    for index, (declared, result) in enumerate(zip(schedule, results)):
        if int(result["window_index"]) != index or result["session_id"] != session_id:
            raise ValueError("CSV slicing/order mismatch")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        timestamps = np.asarray(raw["timestamp_tsc"][start:stop])
        samples = np.asarray(raw["ring_period"][start:stop])
        if not np.isfinite(samples).all() or not np.all(np.diff(timestamps) > 0):
            raise ValueError("non-finite or non-monotonic raw window")
        if int(result["first_sample_tsc"]) != int(timestamps[0]) or \
                int(result["last_sample_tsc"]) != int(timestamps[-1]):
            raise ValueError("CSV raw slice boundary mismatch")
        if declared["sender_off_required"]:
            tone = int(declared["sender_off_control_for_tone_index"])
            response = lockin(timestamps, samples, origin_tsc=int(result["slot_start_tsc"]),
                              tsc_hz=tsc_hz, frequency_hz=tone_hz(tone))
            off_by_tone[tone].append(abs(response))
            continue
        tone = int(declared["physical_tone_index"])
        origin = int(result["slot_start_tsc"])
        sender_phase = phase_index(0, int(declared["sender_codeword_source_index"]),
                                   int(declared["sender_theta_idx"]))
        receiver_phase = phase_index(0, int(declared["receiver_codeword_source_index"]),
                                     int(declared["receiver_theta_idx"]))
        gate = intended_v2_gate(timestamps, origin_tsc=origin, tsc_hz=tsc_hz,
                                tone_index=tone, phase_index_value=sender_phase,
                                amplitude_level=int(declared["amplitude_level"]))
        receiver_gate = intended_v2_gate(
            timestamps, origin_tsc=origin, tsc_hz=tsc_hz, tone_index=tone,
            phase_index_value=receiver_phase,
            amplitude_level=int(declared["amplitude_level"]),
        )
        response = lockin(timestamps, samples, origin_tsc=origin, tsc_hz=tsc_hz,
                          frequency_hz=tone_hz(tone))
        reference = lockin(timestamps, gate, origin_tsc=origin, tsc_hz=tsc_hz,
                           frequency_hz=tone_hz(tone))
        off_bin = lockin(timestamps, samples, origin_tsc=origin, tsc_hz=tsc_hz,
                         frequency_hz=tone_hz(tone) * 1.37 + .071)
        sender_digest = hashlib.sha256(gate.astype(np.uint8).tobytes()).hexdigest()
        receiver_digest = hashlib.sha256(receiver_gate.astype(np.uint8).tobytes()).hexdigest()
        if bool(declared["shared_schedule"]) != (sender_digest == receiver_digest):
            raise ValueError("scramble physical gate digest contract violated")
        if gate.mean() != receiver_gate.mean():
            raise ValueError("scramble workload duty differs")
        measurements.append({
            "tone": tone,
            "amplitude": int(declared["amplitude_level"]),
            "theta": int(declared["sender_theta_idx"]),
            "sign": int(declared["expected_code_sign"]),
            "requested_magnitude": abs(response),
            "response_real": float(response.real),
            "response_imag": float(response.imag),
            "response_phase_radians": float(np.angle(response)),
            "reference_phase_radians": float(np.angle(reference)),
            "off_bin_ratio": abs(off_bin) / max(abs(response), 1e-300),
            "phase_error_radians": angle_error(np.angle(response), np.angle(reference)),
            "temperature_max_c": max(float(result["temp_before_c"]), float(result["temp_after_c"])),
            "frequency_deviation_fraction": max(
                abs(int(result["frequency_before_khz"]) - int(run["frequency_policy"])),
                abs(int(result["frequency_after_khz"]) - int(run["frequency_policy"])),
            ) / int(run["frequency_policy"]),
            "shared_schedule": bool(declared["shared_schedule"]),
            "sender_gate_sha256": sender_digest,
            "receiver_gate_sha256": receiver_digest,
        })

    thresholds = plan.get("analysis_thresholds")
    phase_progression_pass = True
    sign_pi_pass = True
    by_condition = {
        (item["tone"], item["amplitude"], item["theta"], item["sign"]): item
        for item in measurements
    }
    if thresholds:
        expected_conditions = {
            (tone, amplitude, theta, sign)
            for tone in range(12) for amplitude in (1, 2, 3)
            for theta in range(8) for sign in (1, -1)
        }
        if set(by_condition) != expected_conditions:
            raise ValueError("complete tone/amplitude/theta/sign grid required")
        for tone in range(12):
            for amplitude in (1, 2, 3):
                for sign in (1, -1):
                    for theta in range(7):
                        left = by_condition[(tone, amplitude, theta, sign)]
                        right = by_condition[(tone, amplitude, theta + 1, sign)]
                        observed = right["response_phase_radians"] - left["response_phase_radians"]
                        expected = right["reference_phase_radians"] - left["reference_phase_radians"]
                        phase_progression_pass &= angle_error(observed, expected) <= thresholds[
                            "maximum_phase_error_radians"
                        ]
                for theta in range(8):
                    positive = by_condition[(tone, amplitude, theta, 1)]
                    negative = by_condition[(tone, amplitude, theta, -1)]
                    observed = negative["response_phase_radians"] - positive["response_phase_radians"]
                    sign_pi_pass &= angle_error(observed, math.pi) <= thresholds[
                        "maximum_sign_pi_error_radians"
                    ]
    if not thresholds:
        verdict = "CALIBRATION_NOT_ADJUDICABLE_WITHOUT_FROZEN_THRESHOLDS"
    else:
        row_pass = []
        for item in measurements:
            off = np.asarray(off_by_tone[item["tone"]], dtype=float)
            boundary = float(off.mean() + thresholds["minimum_sender_on_off_separation_sigma"] *
                             (off.std() if len(off) > 1 else max(off.mean(), 1e-300)))
            row_pass.append(
                item["requested_magnitude"] > boundary and
                item["phase_error_radians"] <= thresholds["maximum_phase_error_radians"] and
                item["off_bin_ratio"] <= thresholds["maximum_off_bin_to_requested_ratio"] and
                item["temperature_max_c"] <= thresholds["maximum_temperature_c"] and
                item["frequency_deviation_fraction"] <= thresholds["maximum_frequency_deviation_fraction"]
            )
        amplitude_pass = True
        groups: dict[tuple[int, int, int], dict[int, list[float]]] = {}
        for item in measurements:
            group = groups.setdefault((item["tone"], item["theta"], item["sign"]), {})
            group.setdefault(item["amplitude"], []).append(item["requested_magnitude"])
        for group in groups.values():
            medians = [float(np.median(group[level])) for level in (1, 2, 3)]
            amplitude_pass &= medians[0] < medians[1] < medians[2]
        verdict = (
            "SESSION_CALIBRATION_PASS"
            if all(row_pass) and amplitude_pass and phase_progression_pass and sign_pi_pass
            else "SESSION_CALIBRATION_FAIL"
        )
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_ANALYSIS_V1",
        "session_id": session_id,
        "route": planned["route"],
        "record_count": int(offsets[-1]),
        "window_count": len(schedule),
        "measurement_count": len(measurements),
        "measurements": measurements,
        "theta_phase_progression_pass": phase_progression_pass,
        "sign_pi_relation_pass": sign_pi_pass,
        "verdict": verdict,
        "acquisition_authorized": False,
        "restoration_authorized": False,
        "target_coupling_authorized": False,
        "small_wall_authorized": False,
    }


def analyze_campaign(session_results: list[dict], plan: dict) -> dict:
    """Apply the frozen repetition and cross-route campaign verdict."""
    thresholds = plan.get("analysis_thresholds")
    if not thresholds:
        verdict = "CALIBRATION_NOT_ADJUDICABLE_WITHOUT_FROZEN_THRESHOLDS"
        repetition = {}
    else:
        expected_ids = set(plan["session_ids"])
        actual_ids = [result["session_id"] for result in session_results]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
            raise ValueError("complete exact calibration session set required")
        repetition = {}
        for route in ("v4s5", "v2s3"):
            pair = [result for result in session_results if result["route"] == route]
            if len(pair) != 2:
                raise ValueError("exact pre/post reboot pair required per route")
            vectors = [np.asarray([
                complex(item["response_real"], item["response_imag"])
                for item in result["measurements"]
            ]) for result in pair]
            if vectors[0].shape != vectors[1].shape or vectors[0].size == 0:
                raise ValueError("repeated-session measurement vectors differ")
            centered = [vector - vector.mean() for vector in vectors]
            denominator = float(np.linalg.norm(centered[0]) * np.linalg.norm(centered[1]))
            correlation = (
                float(abs(np.vdot(centered[0], centered[1])) / denominator)
                if denominator > 0 else float("nan")
            )
            repetition[route] = {
                "complex_response_correlation": correlation,
                "pass": bool(np.isfinite(correlation) and correlation >=
                             thresholds["minimum_repeated_session_complex_correlation"]),
            }
        sessions_pass = all(
            result["verdict"] == "SESSION_CALIBRATION_PASS" for result in session_results
        )
        routes_pass = all(item["pass"] for item in repetition.values())
        verdict = "CALIBRATION_PASS" if sessions_pass and routes_pass else "CALIBRATION_FAIL"
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_CAMPAIGN_ANALYSIS_V1",
        "session_results": session_results,
        "repeated_session_consistency": repetition,
        "cross_route_pass_required": bool(
            thresholds and thresholds["cross_route_pass_required"]
        ),
        "final_verdict_rule": thresholds.get("final_verdict_rule") if thresholds else None,
        "verdict": verdict,
        **FALSE_AUTHORIZATIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    authorization = load_json(args.authorization)
    authorization["artifact_sha256"] = sha256(args.authorization)
    plan = load_json(args.plan)
    bundle = load_json(args.source_bundle)
    sessions = [analyze_run(run_dir, plan, authorization, bundle)
                for run_dir in args.run_dirs]
    result = analyze_campaign(sessions, plan)
    write_immutable(args.output, result)
    print(result["verdict"])
    return 0 if result["verdict"] == "CALIBRATION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
