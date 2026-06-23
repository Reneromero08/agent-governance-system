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


def as_int(value: str, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field: {field}") from exc


def as_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float field: {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite float field: {field}")
    return result


def csv_value_matches(declared: Any, actual: str, field: str) -> bool:
    if declared is None:
        return actual in {"", "null", "-1"}
    if isinstance(declared, bool):
        return actual in {"1" if declared else "0", "true" if declared else "false"}
    if isinstance(declared, int):
        return as_int(actual, field) == declared
    return actual == str(declared)


ECHO_FIELDS = (
    "stage", "block_id", "family", "actual_mode", "declared_mode",
    "executed_tone_order", "declared_tone_order", "physical_tone_index",
    "receiver_codeword_source_index", "sender_codeword_source_index",
    "drive_on", "sender_off_required", "measurement_mode", "amplitude_level",
    "receiver_theta_idx", "sender_theta_idx", "shared_schedule",
    "scramble_key_digest",
)
NUMERIC_TOLERANCE = 1e-9
MAX_EPOCH_SKEW_SECONDS = 0.005


def angle_error(actual: float, expected: float) -> float:
    return abs(float(np.angle(np.exp(1j * (actual - expected)))))


def construct_complete_grid(schedule: list[dict]) -> dict[tuple[int, int, int, int], dict]:
    """Index the exact sender conditions and reject missing or duplicate rows."""
    expected = {
        (tone, amplitude, theta, sign)
        for tone in range(12) for amplitude in (1, 2, 3)
        for theta in range(8) for sign in (1, -1)
    }
    grid: dict[tuple[int, int, int, int], dict] = {}
    for row in schedule:
        if not row["drive_on"]:
            continue
        condition = (
            int(row["physical_tone_index"]),
            int(row["amplitude_level"]),
            int(row["sender_theta_idx"]),
            int(row["expected_code_sign"]),
        )
        if condition in grid:
            raise ValueError(f"duplicate sender calibration condition: {condition}")
        grid[condition] = row
    if set(grid) != expected:
        raise ValueError("complete tone/amplitude/sender-theta/sign grid required")
    return grid


def load_evidence_map(path: Path, plan: dict) -> list[tuple[Path, Path, Path]]:
    evidence = load_json(path)
    if set(evidence) != {"schema_id", "sessions"} or \
            evidence["schema_id"] != "CAT_CAS_PHASE6_V2_CALIBRATION_EVIDENCE_MAP_V1":
        raise ValueError("invalid evidence-map schema")
    sessions = evidence["sessions"]
    if not isinstance(sessions, dict):
        raise ValueError("evidence-map sessions must be an object")
    expected = set(plan["session_ids"])
    if set(sessions) != expected:
        raise ValueError("evidence-map session set must exactly match plan")
    seen_paths: dict[Path, str] = {}
    ordered = []
    for session_id in plan["session_ids"]:
        entry = sessions[session_id]
        if set(entry) != {"run_dir", "authorization", "source_bundle"}:
            raise ValueError("evidence-map entry fields mismatch")
        paths = tuple((path.parent / entry[key]).resolve()
                      for key in ("run_dir", "authorization", "source_bundle"))
        for item in paths:
            if item in seen_paths:
                raise ValueError("evidence-map path reused across sessions")
            seen_paths[item] = session_id
        ordered.append(paths)
    return ordered


def analyze_run(run_dir: Path, plan: dict, authorization: dict,
                source_bundle: dict,
                *,
                authorization_sha256: str | None = None,
                source_bundle_sha256: str | None = None) -> dict:
    manifest = verify_run_manifest(run_dir)
    run = load_json(run_dir / "run.json")
    session = load_json(run_dir / "session.json")
    schedule = read_rows(run_dir / "windows.jsonl")
    results = read_csv(run_dir / "window_results.csv")
    plan_digest = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    authorization = dict(authorization)
    authorization_digest = authorization_sha256 or authorization.pop("artifact_sha256", None)
    validate_authorization(authorization, plan_digest, source_bundle, source_bundle_sha256)
    session_id = session.get("session_id")
    plan_sessions = {item["session_id"]: item for item in plan["sessions"]}
    if session_id not in plan_sessions:
        raise ValueError("session absent from exact calibration plan")
    planned = plan_sessions[session_id]
    if authorization["session_ids"] != [session_id]:
        raise ValueError("authorization must bind exactly this one session")
    if set(source_bundle["sessions"]) != {session_id}:
        raise ValueError("source bundle must bind exactly this one session")
    if planned.get("window_count") != len(schedule):
        raise ValueError("planned window count mismatch")
    if manifest.get("schema_id") != "CAT_CAS_PHASE6_COMBINED_RUN_MANIFEST_V2" or \
            manifest.get("session_id") != session_id or manifest.get("status") != "COMPLETE":
        raise ValueError("run-manifest schema/session/COMPLETE binding mismatch")
    if schedule != planned["windows"] or len(results) != len(schedule):
        raise ValueError("ordered window set differs from plan")
    if run.get("session_id") != session_id or run.get("route") != planned["route"]:
        raise ValueError("run/session/route binding mismatch")
    if run.get("campaign_plan_sha256") != plan_digest:
        raise ValueError("run campaign-plan binding mismatch")
    if run.get("executor_git_commit") != authorization["executor_commit"]:
        raise ValueError("executor commit binding mismatch")
    if run.get("campaign_source_commit") != authorization["campaign_source_commit"] or \
            session.get("campaign_source_commit") != authorization["campaign_source_commit"]:
        raise ValueError("campaign source commit binding mismatch")
    if run.get("authorization_artifact_sha256") != authorization_digest:
        raise ValueError("authorization digest binding mismatch")
    if run.get("calibration_authorized") is not True:
        raise ValueError("run is not calibration-authorized")
    if run.get("exit_status") != "COMPLETE" or run.get("failure_reason") != "":
        raise ValueError("run completion binding mismatch")
    if run.get("host_control_state_restored") is not True:
        raise ValueError("host control state was not restored")
    if run.get("session_manifest_sha256") != source_bundle["sessions"].get(session_id):
        raise ValueError("session-manifest/source-bundle binding mismatch")
    route_cores = authorization["route_cores"][planned["route"]]
    if [run.get("victim_core"), run.get("sender_core")] != route_cores:
        raise ValueError("run route/core authorization mismatch")
    runtime_bindings = {
        "frequency_policy": "pin_khz",
        "read_rate_hz": "read_hz",
        "slot_duration_s": "slot_s",
        "sender_off_duration_s": "off_window_s",
        "temperature_veto_c": "temperature_veto_c",
    }
    for run_key, authorization_key in runtime_bindings.items():
        if run.get(run_key) != authorization[authorization_key]:
            raise ValueError(f"run runtime authorization mismatch: {run_key}")
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
    raw = np.fromfile(raw_path, dtype=RAW_DTYPE)
    if len(raw) != int(offsets[-1]):
        raise ValueError("raw trailing or missing records")

    tsc_hz = float(run["tsc_calibration_hz"])
    measurements = []
    off_by_tone: dict[int, list[tuple[int, float]]] = {tone: [] for tone in range(12)}
    for index, (declared, result) in enumerate(zip(schedule, results)):
        if int(result["window_index"]) != index or result["session_id"] != session_id:
            raise ValueError("CSV slicing/order mismatch")
        if result.get("window_status") != "OK":
            raise ValueError("window execution status is not OK")
        for field in ECHO_FIELDS:
            if field not in result or not csv_value_matches(declared[field], result[field], field):
                raise ValueError(f"window echo mismatch: {field}")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        timestamps = np.asarray(raw["timestamp_tsc"][start:stop])
        samples = np.asarray(raw["ring_period"][start:stop])
        if not np.isfinite(samples).all() or not np.all(np.diff(timestamps) > 0):
            raise ValueError("non-finite or non-monotonic raw window")
        if int(result["first_sample_tsc"]) != int(timestamps[0]) or \
                int(result["last_sample_tsc"]) != int(timestamps[-1]):
            raise ValueError("CSV raw slice boundary mismatch")
        slot_start = as_int(result["slot_start_tsc"], "slot_start_tsc")
        capture_deadline = as_int(result["capture_deadline_tsc"], "capture_deadline_tsc")
        if int(timestamps[0]) < slot_start or int(timestamps[-1]) > capture_deadline:
            raise ValueError("raw sample timestamp outside declared capture interval")
        if declared["sender_off_required"]:
            if as_int(result["sender_started"], "sender_started") != 0 or \
                    as_int(result["sender_alive_at_capture"], "sender_alive_at_capture") != 0 or \
                    as_int(result["first_drive_tsc"], "first_drive_tsc") != 0:
                raise ValueError("sender-off lifecycle mismatch")
            tone = int(declared["sender_off_control_for_tone_index"])
            response = lockin(timestamps, samples, origin_tsc=int(result["slot_start_tsc"]),
                              tsc_hz=tsc_hz, frequency_hz=tone_hz(tone))
            off_by_tone[tone].append((int(declared["sender_off_control_theta_idx"]),
                                      abs(response)))
            continue
        sender_ready = as_int(result["sender_ready_tsc"], "sender_ready_tsc")
        sender_epoch = as_int(result["sender_epoch_tsc"], "sender_epoch_tsc")
        first_drive = as_int(result["first_drive_tsc"], "first_drive_tsc")
        if as_int(result["sender_started"], "sender_started") != 1 or \
                as_int(result["sender_stopped"], "sender_stopped") != 1 or \
                as_int(result["sender_alive_at_capture"], "sender_alive_at_capture") != 1:
            raise ValueError("driven sender lifecycle mismatch")
        skew_limit = int(MAX_EPOCH_SKEW_SECONDS * tsc_hz)
        if sender_ready >= slot_start or sender_epoch < slot_start or \
                sender_epoch > slot_start + skew_limit or \
                first_drive < slot_start or first_drive > capture_deadline:
            raise ValueError("driven sender timing mismatch")
        tone = int(declared["physical_tone_index"])
        origin = slot_start
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
        computed = {
            "computed_I": response.real,
            "computed_Q": response.imag,
            "magnitude": abs(response),
            "floor": abs(off_bin),
        }
        for field, expected in computed.items():
            actual = result[field]
            if actual in {"", "null"} or \
                    not math.isclose(as_float(actual, field), expected,
                                     rel_tol=NUMERIC_TOLERANCE,
                                     abs_tol=NUMERIC_TOLERANCE):
                raise ValueError(f"executor spectral value mismatch: {field}")
        sender_digest = hashlib.sha256(gate.astype(np.uint8).tobytes()).hexdigest()
        receiver_digest = hashlib.sha256(receiver_gate.astype(np.uint8).tobytes()).hexdigest()
        if bool(declared["shared_schedule"]) != (sender_digest == receiver_digest):
            raise ValueError("logical sender-field gate digest contract violated")
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
                abs(as_int(result["victim_frequency_before_khz"],
                           "victim_frequency_before_khz") - int(run["frequency_policy"])),
                abs(as_int(result["victim_frequency_after_khz"],
                           "victim_frequency_after_khz") - int(run["frequency_policy"])),
                abs(as_int(result["sender_frequency_before_khz"],
                           "sender_frequency_before_khz") - int(run["frequency_policy"])),
                abs(as_int(result["sender_frequency_after_khz"],
                           "sender_frequency_after_khz") - int(run["frequency_policy"])),
            ) / int(run["frequency_policy"]),
            "shared_schedule": bool(declared["shared_schedule"]),
            "sender_gate_sha256": sender_digest,
            "receiver_gate_sha256": receiver_digest,
        })

    construct_complete_grid(schedule)
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
        sender_off_controls = {}
    else:
        sender_off_controls = {}
        for tone, values in off_by_tone.items():
            theta_indices = [theta for theta, _ in values]
            magnitudes = np.asarray([magnitude for _, magnitude in values], dtype=float)
            if len(values) != 8 or set(theta_indices) != set(range(8)):
                raise ValueError("exactly eight sender-off controls per tone required")
            if not np.isfinite(magnitudes).all():
                raise ValueError("non-finite sender-off control distribution")
            sample_std = float(np.std(magnitudes, ddof=1))
            if not math.isfinite(sample_std) or sample_std <= 0.0:
                raise ValueError("zero-variance sender-off control distribution")
            mean = float(np.mean(magnitudes))
            boundary = mean + thresholds["minimum_sender_on_off_separation_sigma"] * sample_std
            sender_off_controls[str(tone)] = {
                "count": 8,
                "mean": mean,
                "sample_std": sample_std,
                "boundary": boundary,
            }
        row_pass = []
        for item in measurements:
            boundary = sender_off_controls[str(item["tone"])]["boundary"]
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
        "sender_off_controls": sender_off_controls,
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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--evidence-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = load_json(args.plan)
    session_inputs = load_evidence_map(args.evidence_map.resolve(), plan)
    sessions = []
    for run_dir, authorization_path, source_bundle_path in session_inputs:
        authorization = load_json(authorization_path)
        bundle = load_json(source_bundle_path)
        sessions.append(
            analyze_run(
                run_dir, plan, authorization, bundle,
                authorization_sha256=sha256(authorization_path),
                source_bundle_sha256=sha256(source_bundle_path),
            )
        )
    result = analyze_campaign(sessions, plan)
    write_immutable(args.output, result)
    print(result["verdict"])
    return 0 if result["verdict"] == "CALIBRATION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
