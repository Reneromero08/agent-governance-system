#!/usr/bin/env python3
"""Fail-closed analyzer for Phase 6 V2 spectral calibration evidence."""

from __future__ import annotations

import argparse
import csv
import io
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np

from calibration_contract import (
    FALSE_AUTHORIZATIONS,
    ROUTE_CORES,
    RUNTIME_PARAMETERS,
    THRESHOLDS,
    canonical_bytes,
    validate_authorization,
    write_immutable,
)

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
import sys
sys.path.insert(0, str(ANALYSIS))
from waveform_reference import control_frequency_hz, intended_v2_gate, lockin, phase_index, tone_hz  # noqa: E402

RAW_DTYPE = np.dtype([("timestamp_tsc", "<u8"), ("ring_period", "<f8")])
RUN_FILES = {
    "run.json", "session.json", "windows.jsonl", "window_results.csv",
    "raw_samples.bin", "telemetry.csv", "stdout.log", "stderr.log",
    "orchestrator_stdout.log", "orchestrator_stderr.log",
}


PLAN_FIELDS = {
    "schema_id", "execution_class", "campaign_source_commit", "sessions",
    "session_ids", "session_count", "windows_per_session", "windows_per_route",
    "total_window_count", "count_derivation", "runtime_parameters",
    "analysis_thresholds", "calibration_authorized", *FALSE_AUTHORIZATIONS,
}
PLAN_SESSION_FIELDS = {
    "session_id", "route", "route_cores", "partition",
    "frequency_settling_required", "window_count", "windows",
}
WINDOW_FIELDS = {
    "window_index", "session_id", "stage", "block_id", "family",
    "actual_mode", "declared_mode", "executed_tone_order",
    "declared_tone_order", "measurement_mode", "drive_on",
    "sender_off_required", "physical_tone_index",
    "receiver_codeword_source_index", "sender_codeword_source_index",
    "receiver_theta_idx", "sender_theta_idx", "shared_schedule",
    "scramble_key_digest", "amplitude_level", "expected_code_sign",
    "sender_off_control_for_tone_index", "sender_off_control_theta_idx",
}
SESSION_FILE_FIELDS = {
    "schema_id", "campaign_source_commit", "campaign_plan_sha256", "session_id",
    "route", "partition", "window_count", "frequency_settling_required",
    "restoration_authorized",
}
RUN_MANIFEST_FIELDS = {"schema_id", "session_id", "status", "files"}
FILE_BINDING_FIELDS = {"size", "sha256"}
RUN_FIELDS = {
    "schema_id", "session_id", "campaign_source_commit", "campaign_plan_sha256",
    "session_manifest_sha256", "executor_git_commit", "host_identity",
    "kernel_identity", "cpu_model", "route", "execution_class",
    "authorization_artifact_sha256", "victim_core", "sender_core",
    "frequency_policy", "tsc_calibration_hz", "read_rate_hz",
    "slot_duration_s", "sender_off_duration_s", "temperature_veto_c",
    "start_timestamp", "end_timestamp", "exit_status", "failure_reason",
    "host_control_state_restored", "physical_carrier_restoration_claimed",
    "automatic_retry", "restoration_authorized", "calibration_authorized",
    "acquisition_authorized", "scientific_acquisition_authorized",
    "target_coupling_authorized", "small_wall_authorized", "hardware_executed",
    "original_cpufreq_state", "applied_cpufreq_state", "restored_cpufreq_state",
}
ORIGINAL_CPUFREQ_FIELDS = {"min_khz", "max_khz", "boost"}
APPLIED_CPUFREQ_FIELDS = {"min_khz", "max_khz", "boost"}
RESTORED_CPUFREQ_FIELDS = {"verified", "min_khz", "max_khz", "boost"}
RUNTIME_PARAMETER_FIELDS = {
    "pin_khz", "slot_s", "off_window_s", "read_hz", "temperature_veto_c",
    "automatic_retry",
}
THRESHOLD_FIELDS = {
    "minimum_sender_on_off_separation_sigma", "maximum_phase_error_radians",
    "maximum_sign_pi_error_radians", "amplitude_monotonicity_required",
    "maximum_off_bin_to_requested_ratio", "maximum_frequency_deviation_fraction",
    "maximum_temperature_c", "minimum_repeated_session_complex_correlation",
    "cross_route_pass_required", "final_verdict_rule", "capture_quality",
}
CAPTURE_QUALITY_FIELDS = {
    "minimum_capture_coverage_fraction", "minimum_empirical_sample_rate_fraction",
    "maximum_empirical_sample_rate_fraction", "minimum_empirical_nyquist_margin",
    "maximum_sample_gap_multiple",
}
PARTITIONS = {"PRE_REBOOT_REPETITION", "POST_REBOOT_REPETITION"}


def _require_exact_fields(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} fields mismatch: expected {sorted(expected)}, got {actual}")
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_plan_schema(plan: dict, *, require_full_campaign: bool) -> None:
    _require_exact_fields(plan, PLAN_FIELDS, "plan")
    if plan["schema_id"] != "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_PLAN_V2" or \
            plan["execution_class"] != "ENGINEERING_QUALIFICATION_NOT_SCIENTIFIC_ACQUISITION":
        raise ValueError("invalid V2 calibration plan schema or execution class")
    if plan["calibration_authorized"] is not False:
        raise ValueError("plan calibration_authorized must remain false")
    for key, expected in FALSE_AUTHORIZATIONS.items():
        if plan[key] is not expected:
            raise ValueError(f"plan {key} must remain false")
    source_commit = plan["campaign_source_commit"]
    if not isinstance(source_commit, str) or len(source_commit) != 40 or \
            source_commit == "0" * 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("invalid plan campaign source commit")
    runtime = _require_exact_fields(plan["runtime_parameters"], RUNTIME_PARAMETER_FIELDS,
                                    "plan runtime_parameters")
    if runtime != RUNTIME_PARAMETERS:
        raise ValueError("plan runtime parameter values mismatch")
    thresholds = _require_exact_fields(plan["analysis_thresholds"], THRESHOLD_FIELDS,
                                       "plan analysis_thresholds")
    capture = _require_exact_fields(thresholds["capture_quality"], CAPTURE_QUALITY_FIELDS,
                                    "plan capture_quality")
    if thresholds != THRESHOLDS or any(
            not _is_number(capture[key]) for key in CAPTURE_QUALITY_FIELDS
    ):
        raise ValueError("plan analysis threshold values mismatch")
    sessions = plan["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("plan sessions must be a nonempty list")
    ordered_ids: list[str] = []
    route_totals = {route: 0 for route in ROUTE_CORES}
    for session in sessions:
        _require_exact_fields(session, PLAN_SESSION_FIELDS, "plan session")
        session_id = session["session_id"]
        if not isinstance(session_id, str) or not session_id or session_id in ordered_ids:
            raise ValueError("plan session IDs must be unique nonempty strings")
        ordered_ids.append(session_id)
        route = session["route"]
        if route not in ROUTE_CORES or session["route_cores"] != ROUTE_CORES[route]:
            raise ValueError("plan route/core binding mismatch")
        if session["partition"] not in PARTITIONS or \
                session["frequency_settling_required"] is not True:
            raise ValueError("plan session partition/frequency-settling mismatch")
        windows = session["windows"]
        if not isinstance(windows, list) or session["window_count"] != len(windows):
            raise ValueError("plan session window count mismatch")
        for index, window in enumerate(windows):
            _require_exact_fields(window, WINDOW_FIELDS, "plan window")
            if window["window_index"] != index or window["session_id"] != session_id:
                raise ValueError("plan window index/session binding mismatch")
        route_totals[route] += len(windows)
    if plan["session_ids"] != ordered_ids or plan["session_count"] != len(sessions):
        raise ValueError("plan ordered session binding mismatch")
    if plan["windows_per_route"] != route_totals or \
            plan["total_window_count"] != sum(route_totals.values()):
        raise ValueError("plan aggregate window counts mismatch")
    counts = {session["window_count"] for session in sessions}
    if len(counts) != 1 or plan["windows_per_session"] != next(iter(counts)):
        raise ValueError("plan windows_per_session mismatch")
    if not isinstance(plan["count_derivation"], str) or not plan["count_derivation"]:
        raise ValueError("plan count derivation required")
    if require_full_campaign:
        expected_pairs = {(route, partition) for route in ROUTE_CORES for partition in PARTITIONS}
        actual_pairs = {(session["route"], session["partition"]) for session in sessions}
        if len(sessions) != 4 or actual_pairs != expected_pairs or \
                plan["windows_per_session"] != 672 or \
                plan["windows_per_route"] != {"v4s5": 1344, "v2s3": 1344} or \
                plan["total_window_count"] != 2688:
            raise ValueError("complete four-session V2 campaign plan required")


def validate_session_schema(session: dict, planned: dict) -> None:
    _require_exact_fields(session, SESSION_FILE_FIELDS, "session.json")
    expected = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V2",
        "campaign_source_commit": planned.get("campaign_source_commit"),
        "campaign_plan_sha256": planned.get("campaign_plan_sha256"),
        "session_id": planned["session_id"],
        "route": planned["route"],
        "partition": planned["partition"],
        "window_count": planned["window_count"],
        "frequency_settling_required": True,
        "restoration_authorized": False,
    }
    if session != expected:
        raise ValueError("session.json differs from planned exact session binding")


def validate_run_schema(run: dict) -> None:
    _require_exact_fields(run, RUN_FIELDS, "run.json")
    if run["schema_id"] != "CAT_CAS_PHASE6_COMBINED_RUN_V2":
        raise ValueError("run schema mismatch")
    _require_exact_fields(run["original_cpufreq_state"], ORIGINAL_CPUFREQ_FIELDS,
                          "run original_cpufreq_state")
    _require_exact_fields(run["applied_cpufreq_state"], APPLIED_CPUFREQ_FIELDS,
                          "run applied_cpufreq_state")
    _require_exact_fields(run["restored_cpufreq_state"], RESTORED_CPUFREQ_FIELDS,
                          "run restored_cpufreq_state")
    for state_name in ("original_cpufreq_state", "restored_cpufreq_state"):
        state = run[state_name]
        for list_name in ("min_khz", "max_khz"):
            values = state[list_name]
            if not isinstance(values, list) or not values or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in values
            ):
                raise ValueError(f"run {state_name}.{list_name} must be a nonempty integer list")
        if not isinstance(state["boost"], int) or isinstance(state["boost"], bool):
            raise ValueError(f"run {state_name}.boost must be an integer")
    applied = run["applied_cpufreq_state"]
    if any(not isinstance(applied[key], int) or isinstance(applied[key], bool)
           for key in ("min_khz", "max_khz", "boost")) or \
            run["restored_cpufreq_state"]["verified"] is not True:
        raise ValueError("run cpufreq restoration schema mismatch")
    for key in ("physical_carrier_restoration_claimed", "automatic_retry",
                "restoration_authorized", "acquisition_authorized",
                "scientific_acquisition_authorized", "target_coupling_authorized",
                "small_wall_authorized"):
        if run[key] is not False:
            raise ValueError(f"run {key} must remain false")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_path(path: Path, *, directory: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"immutable input missing or inaccessible: {path}") from exc
    if resolved != absolute:
        raise ValueError(f"immutable input symlink traversal rejected: {path}")
    if directory and not resolved.is_dir():
        raise ValueError(f"expected immutable directory: {path}")
    if not directory and not resolved.is_file():
        raise ValueError(f"immutable input must be a regular file: {path}")
    return resolved


def _canonical_regular_path(path: Path) -> Path:
    return _canonical_path(path, directory=False)


def _canonical_directory_path(path: Path) -> Path:
    return _canonical_path(path, directory=True)


def read_regular_bytes(path: Path) -> bytes:
    resolved = _canonical_regular_path(path)
    try:
        path_before = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"immutable input stat failed: {path}") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"immutable input open failed: {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"immutable input must be a regular file: {path}")
        if (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"immutable input changed before read: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        path_after = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"immutable input disappeared after read: {path}") from exc
    identity_before = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    path_identity_after = (path_after.st_dev, path_after.st_ino, path_after.st_size)
    if identity_before != identity_after or \
            path_identity_after != (after.st_dev, after.st_ino, after.st_size) or \
            len(payload) != after.st_size:
        raise ValueError(f"immutable input mutated during read: {path}")
    return payload


def sha256(path: Path) -> str:
    return sha256_bytes(read_regular_bytes(path))


def _decode_utf8(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {label}") from exc


def parse_json_bytes(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(_decode_utf8(payload, label))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def load_json(path: Path) -> dict:
    return parse_json_bytes(read_regular_bytes(path), str(path))


def parse_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    text = _decode_utf8(payload, label)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"empty JSONL row in {label}: {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row in {label}: {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object in {label}: {line_number}")
        rows.append(row)
    return rows


def parse_csv_bytes(payload: bytes, required_columns: tuple[str, ...], label: str) -> list[dict[str, str]]:
    source = io.StringIO(_decode_utf8(payload, label), newline="")
    reader = csv.DictReader(source)
    actual = tuple(reader.fieldnames) if reader.fieldnames else ()
    if actual != required_columns:
        raise ValueError(
            f"CSV column mismatch in {label}: expected {required_columns}, got {actual}"
        )
    rows = list(reader)
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row width mismatch in {label}: row {index}")
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    return parse_jsonl_bytes(read_regular_bytes(path), str(path))


def read_csv(path: Path) -> list[dict[str, str]]:
    payload = read_regular_bytes(path)
    source = io.StringIO(_decode_utf8(payload, str(path)), newline="")
    return list(csv.DictReader(source))


def read_csv_strict(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    return parse_csv_bytes(read_regular_bytes(path), required_columns, path.name)


def read_telemetry_bytes(payload: bytes, label: str) -> list[dict[str, str]]:
    rows = parse_csv_bytes(payload, TELEMETRY_COLUMNS, label)
    for i, row in enumerate(rows):
        if int(row["window_index"]) != i:
            raise ValueError(f"telemetry window_index not contiguous: row {i}")
    return rows


def read_telemetry(path: Path) -> list[dict[str, str]]:
    return read_telemetry_bytes(read_regular_bytes(path), path.name)


def verify_run_manifest(run_dir: Path) -> tuple[dict, bytes, dict[str, bytes]]:
    resolved_dir = _canonical_directory_path(run_dir)
    entries = list(resolved_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("run directory must contain regular files only")
    actual = {path.name for path in entries}
    if actual != RUN_FILES | {"run_manifest.json"}:
        raise ValueError("run directory file set mismatch")
    manifest_bytes = read_regular_bytes(resolved_dir / "run_manifest.json")
    manifest = parse_json_bytes(manifest_bytes, "run_manifest.json")
    _require_exact_fields(manifest, RUN_MANIFEST_FIELDS, "run_manifest.json")
    if manifest["schema_id"] != "CAT_CAS_PHASE6_COMBINED_RUN_MANIFEST_V2" or \
            manifest["status"] != "COMPLETE" or not isinstance(manifest["session_id"], str):
        raise ValueError("run-manifest schema/session/COMPLETE binding mismatch")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != RUN_FILES:
        raise ValueError("run manifest file set mismatch")
    captured: dict[str, bytes] = {}
    for name in sorted(RUN_FILES):
        binding = _require_exact_fields(files[name], FILE_BINDING_FIELDS,
                                        f"run manifest binding {name}")
        digest = binding["sha256"]
        if not isinstance(binding["size"], int) or isinstance(binding["size"], bool) or \
                binding["size"] < 0 or not isinstance(digest, str) or len(digest) != 64 or \
                any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid run manifest binding: {name}")
        payload = read_regular_bytes(resolved_dir / name)
        captured[name] = payload
        if binding != {"size": len(payload), "sha256": sha256_bytes(payload)}:
            raise ValueError(f"run manifest binding mismatch: {name}")
    return manifest, manifest_bytes, captured


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
    "sender_off_control_for_tone_index", "sender_off_control_theta_idx",
)
WINDOW_RESULTS_COLUMNS = (
    "window_index", "session_id", "stage", "block_id", "family",
    "actual_mode", "declared_mode", "executed_tone_order",
    "declared_tone_order", "physical_tone_index",
    "receiver_codeword_source_index", "sender_codeword_source_index",
    "drive_on", "sender_off_required", "measurement_mode",
    "amplitude_level", "receiver_theta_idx", "sender_theta_idx",
    "shared_schedule", "scramble_key_digest",
    "sender_off_control_for_tone_index", "sender_off_control_theta_idx",
    "slot_start_tsc", "capture_deadline_tsc", "sender_ready_tsc",
    "sender_epoch_tsc", "first_drive_tsc", "receiver_epoch_tsc",
    "first_sample_tsc", "last_sample_tsc", "sample_count",
    "temp_before_c", "temp_after_c",
    "victim_frequency_before_khz", "victim_frequency_after_khz",
    "sender_frequency_before_khz", "sender_frequency_after_khz",
    "aperf_before", "aperf_after", "mperf_before", "mperf_after",
    "cofvid_before", "cofvid_after",
    "computed_I", "computed_Q", "magnitude", "floor",
    "raw_mean", "raw_min", "raw_max",
    "sender_started", "sender_stopped", "sender_alive_at_capture",
    "window_status",
)
TELEMETRY_COLUMNS = (
    "window_index", "temp_before_c", "temp_after_c",
    "victim_frequency_before_khz", "victim_frequency_after_khz",
    "sender_frequency_before_khz", "sender_frequency_after_khz",
    "aperf_before", "aperf_after", "mperf_before", "mperf_after",
    "cofvid_before", "cofvid_after",
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


def load_evidence_map_bytes(payload: bytes, path: Path,
                            plan: dict) -> list[tuple[Path, ...]]:
    evidence = parse_json_bytes(payload, "evidence map")
    if set(evidence) != {"schema_id", "sessions"} or \
            evidence["schema_id"] != "CAT_CAS_PHASE6_V2_CALIBRATION_EVIDENCE_MAP_V2":
        raise ValueError("invalid evidence-map schema")
    sessions = evidence["sessions"]
    if not isinstance(sessions, dict):
        raise ValueError("evidence-map sessions must be an object")
    expected = set(plan["session_ids"])
    if set(sessions) != expected:
        raise ValueError("evidence-map session set must exactly match plan")
    seen_paths: dict[Path, str] = {}
    ordered = []
    base = path.parent
    for session_id in plan["session_ids"]:
        entry = sessions[session_id]
        expected_entry_keys = {"run_dir", "authorization", "source_bundle", "session_manifest"}
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys or any(not isinstance(entry[key], str) or not entry[key] for key in entry):
            raise ValueError("evidence-map entry fields mismatch")
        paths = (
            _canonical_directory_path(base / entry["run_dir"]),
            _canonical_regular_path(base / entry["authorization"]),
            _canonical_regular_path(base / entry["source_bundle"]),
            _canonical_regular_path(base / entry["session_manifest"]),
        )
        for item in paths:
            if item in seen_paths:
                raise ValueError("evidence-map path reused across sessions")
            seen_paths[item] = session_id
        ordered.append(paths)
    return ordered


def load_evidence_map(path: Path, plan: dict) -> list[tuple[Path, ...]]:
    return load_evidence_map_bytes(read_regular_bytes(path), path, plan)


def analyze_run(run_dir: Path, plan: dict, authorization: dict,
                source_bundle: dict,
                *,
                authorization_sha256: str | None = None,
                source_bundle_sha256: str | None = None,
                plan_sha256: str | None = None,
                evidence_map_sha256: str | None = None,
                run_manifest_sha256: str | None = None,
                session_manifest_sha256: str | None = None,
                run_json_sha256: str | None = None,
                raw_samples_sha256: str | None = None,
                window_results_sha256: str | None = None,
                telemetry_sha256: str | None = None,
                executor_commit: str | None = None,
                executor_sha256: str | None = None,
                campaign_source_commit: str | None = None,
                session_id_from_call: str | None = None,
                route: str | None = None,
                ) -> dict:
    validate_plan_schema(plan, require_full_campaign=False)
    manifest, manifest_bytes, captured = verify_run_manifest(run_dir)
    run = parse_json_bytes(captured["run.json"], "run.json")
    session = parse_json_bytes(captured["session.json"], "session.json")
    schedule = parse_jsonl_bytes(captured["windows.jsonl"], "windows.jsonl")
    results = parse_csv_bytes(
        captured["window_results.csv"], WINDOW_RESULTS_COLUMNS, "window_results.csv"
    )
    telemetry_rows = read_telemetry_bytes(captured["telemetry.csv"], "telemetry.csv")
    plan_digest = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    if plan_sha256 is not None and plan_sha256 != plan_digest:
        raise ValueError("provided plan digest differs from exact plan bytes")
    authorization = dict(authorization)
    authorization_digest = authorization_sha256 or authorization.pop("artifact_sha256", None)
    validate_authorization(authorization, plan_digest, source_bundle, source_bundle_sha256)
    if plan.get("campaign_source_commit") != authorization["campaign_source_commit"]:
        raise ValueError("plan/authorization campaign source commit mismatch")
    thresholds = plan.get("analysis_thresholds")
    if not isinstance(thresholds, dict) or not isinstance(thresholds.get("capture_quality"), dict):
        raise ValueError("frozen capture-quality thresholds required")
    capture_quality = thresholds["capture_quality"]
    required_capture_quality = {
        "minimum_capture_coverage_fraction",
        "minimum_empirical_sample_rate_fraction",
        "maximum_empirical_sample_rate_fraction",
        "minimum_empirical_nyquist_margin",
        "maximum_sample_gap_multiple",
    }
    if set(capture_quality) != required_capture_quality:
        raise ValueError("capture-quality threshold fields mismatch")
    session_id = session.get("session_id")
    plan_sessions = {item["session_id"]: item for item in plan["sessions"]}
    if session_id not in plan_sessions:
        raise ValueError("session absent from exact calibration plan")
    planned = plan_sessions[session_id]
    planned_file = {
        **planned,
        "campaign_source_commit": plan["campaign_source_commit"],
        "campaign_plan_sha256": plan_digest,
    }
    validate_session_schema(session, planned_file)
    validate_run_schema(run)
    if manifest["session_id"] != session_id:
        raise ValueError("run-manifest schema/session/COMPLETE binding mismatch")
    if authorization["session_ids"] != [session_id]:
        raise ValueError("authorization must bind exactly this one session")
    if set(source_bundle["sessions"]) != {session_id}:
        raise ValueError("source bundle must bind exactly this one session")
    if planned.get("window_count") != len(schedule):
        raise ValueError("planned window count mismatch")
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

    if len(telemetry_rows) != len(results):
        raise ValueError("telemetry row count must match window count")
    telemetry_fields = (
        "temp_before_c", "temp_after_c",
        "victim_frequency_before_khz", "victim_frequency_after_khz",
        "sender_frequency_before_khz", "sender_frequency_after_khz",
        "aperf_before", "aperf_after", "mperf_before", "mperf_after",
        "cofvid_before", "cofvid_after",
    )
    for i, (trow, wrow) in enumerate(zip(telemetry_rows, results)):
        for field in telemetry_fields:
            if trow.get(field) != wrow.get(field):
                raise ValueError(f"telemetry mismatch at window {i}, field {field}")

    raw_bytes = captured["raw_samples.bin"]
    counts = np.asarray([int(row["sample_count"]) for row in results], dtype=np.int64)
    if np.any(counts < 4):
        raise ValueError("every calibration window requires at least four samples")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if len(raw_bytes) != int(offsets[-1]) * RAW_DTYPE.itemsize:
        raise ValueError("raw binary size/record count mismatch")
    raw = np.frombuffer(raw_bytes, dtype=RAW_DTYPE)
    if len(raw) != int(offsets[-1]):
        raise ValueError("raw trailing or missing records")
    supplied_bindings = {
        "run_manifest.json": run_manifest_sha256,
        "run.json": run_json_sha256,
        "raw_samples.bin": raw_samples_sha256,
        "window_results.csv": window_results_sha256,
        "telemetry.csv": telemetry_sha256,
    }
    actual_bindings = {
        "run_manifest.json": sha256_bytes(manifest_bytes),
        "run.json": sha256_bytes(captured["run.json"]),
        "raw_samples.bin": sha256_bytes(captured["raw_samples.bin"]),
        "window_results.csv": sha256_bytes(captured["window_results.csv"]),
        "telemetry.csv": sha256_bytes(captured["telemetry.csv"]),
    }
    for name, supplied in supplied_bindings.items():
        if supplied is not None and supplied != actual_bindings[name]:
            raise ValueError(f"provided input digest mismatch: {name}")
    if session_manifest_sha256 is not None and \
            session_manifest_sha256 != run.get("session_manifest_sha256"):
        raise ValueError("provided session manifest digest mismatch")
    if executor_commit is not None and executor_commit != authorization["executor_commit"]:
        raise ValueError("provided executor commit mismatch")
    if executor_sha256 is not None and executor_sha256 != authorization["executor_sha256"]:
        raise ValueError("provided executor digest mismatch")
    if campaign_source_commit is not None and \
            campaign_source_commit != authorization["campaign_source_commit"]:
        raise ValueError("provided campaign source commit mismatch")
    if session_id_from_call is not None and session_id_from_call != session_id:
        raise ValueError("provided session ID mismatch")
    if route is not None and route != planned["route"]:
        raise ValueError("provided route mismatch")

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
        capture_span = int(timestamps[-1]) - int(timestamps[0])
        capture_window = capture_deadline - slot_start
        if capture_window <= 0:
            raise ValueError("invalid capture window")
        capture_coverage = capture_span / capture_window
        if capture_coverage < capture_quality["minimum_capture_coverage_fraction"]:
            raise ValueError("insufficient capture coverage")
        empirical_rate = (int(len(timestamps)) - 1) * tsc_hz / capture_span if capture_span > 0 else 0
        read_hz_auth = run.get("read_rate_hz")
        if not isinstance(read_hz_auth, (int, float)) or read_hz_auth <= 0:
            raise ValueError("invalid authorized read_hz")
        rate_fraction = empirical_rate / read_hz_auth
        if rate_fraction < capture_quality["minimum_empirical_sample_rate_fraction"] or \
                rate_fraction > capture_quality["maximum_empirical_sample_rate_fraction"]:
            raise ValueError("empirical sample rate out of bounds")
        gaps = np.diff(timestamps)
        max_gap = float(np.max(gaps))
        nominal_spacing = tsc_hz / read_hz_auth
        gap_multiple = max_gap / nominal_spacing if nominal_spacing > 0 else 0
        if gap_multiple > capture_quality["maximum_sample_gap_multiple"]:
            raise ValueError("pathological timestamp gap")
        analysis_tone = (
            int(declared["sender_off_control_for_tone_index"])
            if declared["sender_off_required"]
            else int(declared["physical_tone_index"])
        )
        max_analysis_frequency = max(
            tone_hz(analysis_tone),
            control_frequency_hz(tone_hz(analysis_tone)),
        )
        nyquist_margin = empirical_rate / (2.0 * max_analysis_frequency)
        if nyquist_margin < capture_quality["minimum_empirical_nyquist_margin"]:
            raise ValueError("empirical Nyquist margin insufficient")
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
                         frequency_hz=control_frequency_hz(tone_hz(tone)))
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
            "capture_coverage": capture_coverage,
            "empirical_sample_rate": empirical_rate,
            "sample_rate_fraction": rate_fraction,
            "max_sample_gap_multiple": gap_multiple,
            "empirical_nyquist_margin": nyquist_margin,
            "capture_quality_pass": True,
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
                item["frequency_deviation_fraction"] <= thresholds["maximum_frequency_deviation_fraction"] and
                item["capture_quality_pass"]
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
        "input_bindings": {
            "plan_sha256": plan_digest,
            "evidence_map_sha256": evidence_map_sha256,
            "authorization_sha256": authorization_digest,
            "source_bundle_sha256": source_bundle_sha256 or
                hashlib.sha256(canonical_bytes(source_bundle)).hexdigest(),
            "session_manifest_sha256": run.get("session_manifest_sha256"),
            "run_manifest_sha256": sha256_bytes(manifest_bytes),
            "run_json_sha256": sha256_bytes(captured["run.json"]),
            "session_json_sha256": sha256_bytes(captured["session.json"]),
            "windows_jsonl_sha256": sha256_bytes(captured["windows.jsonl"]),
            "raw_samples_sha256": sha256_bytes(captured["raw_samples.bin"]),
            "window_results_sha256": sha256_bytes(captured["window_results.csv"]),
            "telemetry_sha256": sha256_bytes(captured["telemetry.csv"]),
            "executor_commit": authorization["executor_commit"],
            "executor_sha256": authorization["executor_sha256"],
            "campaign_source_commit": authorization["campaign_source_commit"],
            "session_id": session_id,
            "route": planned["route"],
            "capture_quality_thresholds": thresholds.get("capture_quality") if thresholds else None,
        },
    }


def analyze_campaign(session_results: list[dict], plan: dict,
                     evidence_map_sha256: str | None = None) -> dict:
    """Apply the frozen repetition and cross-route campaign verdict."""
    thresholds = plan.get("analysis_thresholds")
    if not thresholds:
        verdict = "CALIBRATION_NOT_ADJUDICABLE_WITHOUT_FROZEN_THRESHOLDS"
        repetition = {}
    else:
        supported_rule = "ALL_GROUPS_AND_BOTH_REBOOT_PARTITIONS_AND_BOTH_ROUTES_PASS"
        if thresholds.get("amplitude_monotonicity_required") is not True:
            raise ValueError("amplitude_monotonicity_required must be true")
        if thresholds.get("cross_route_pass_required") is not True:
            raise ValueError("cross_route_pass_required must be true")
        if thresholds.get("final_verdict_rule") != supported_rule:
            raise ValueError(f"final_verdict_rule must be {supported_rule}")
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
        "input_bindings": {
            "plan_sha256": hashlib.sha256(canonical_bytes(plan)).hexdigest(),
            "evidence_map_sha256": evidence_map_sha256,
            "ordered_session_input_bindings": [
                result.get("input_bindings") for result in session_results
            ],
        },
        **FALSE_AUTHORIZATIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--evidence-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan_path = _canonical_regular_path(args.plan)
    plan_bytes = read_regular_bytes(plan_path)
    plan = parse_json_bytes(plan_bytes, "plan")
    validate_plan_schema(plan, require_full_campaign=True)
    canonical_plan_bytes = canonical_bytes(plan)
    if plan_bytes != canonical_plan_bytes:
        raise ValueError("plan must use exact canonical JSON bytes")
    plan_sha256 = sha256_bytes(plan_bytes)

    evidence_map_path = _canonical_regular_path(args.evidence_map)
    evidence_map_bytes = read_regular_bytes(evidence_map_path)
    evidence_map_sha256 = sha256_bytes(evidence_map_bytes)
    session_inputs = load_evidence_map_bytes(evidence_map_bytes, evidence_map_path, plan)
    sessions = []
    for run_dir, authorization_path, source_bundle_path, session_manifest_path in session_inputs:
        authorization_bytes = read_regular_bytes(authorization_path)
        source_bundle_bytes = read_regular_bytes(source_bundle_path)
        authorization = parse_json_bytes(authorization_bytes, str(authorization_path))
        bundle = parse_json_bytes(source_bundle_bytes, str(source_bundle_path))
        sessions.append(
            analyze_run(
                run_dir, plan, authorization, bundle,
                authorization_sha256=sha256_bytes(authorization_bytes),
                source_bundle_sha256=sha256_bytes(source_bundle_bytes),
                evidence_map_sha256=evidence_map_sha256,
                plan_sha256=plan_sha256,
            )
        )
    result = analyze_campaign(sessions, plan, evidence_map_sha256=evidence_map_sha256)
    result["input_bindings"]["evidence_map_sha256"] = evidence_map_sha256
    write_immutable(args.output, result)
    print(result["verdict"])
    return 0 if result["verdict"] == "CALIBRATION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
