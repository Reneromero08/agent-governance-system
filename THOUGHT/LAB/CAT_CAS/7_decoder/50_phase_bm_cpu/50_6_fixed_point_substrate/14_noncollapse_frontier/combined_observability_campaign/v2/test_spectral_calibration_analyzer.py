from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyze_spectral_calibration_v2 import (
    RAW_DTYPE,
    RUN_FILES,
    WINDOW_RESULTS_COLUMNS,
    analyze_campaign,
    analyze_run,
    construct_complete_grid,
    main as analyzer_main,
    validate_plan_schema,
)
from calibration_contract import (
    FALSE_AUTHORIZATIONS,
    ROUTE_CORES,
    build_plan,
    build_source_bundle_manifest,
    canonical_bytes,
)
from waveform_reference import control_frequency_hz, intended_v2_gate, lockin, phase_index, tone_hz

HERE = Path(__file__).resolve().parent
CONTRACTS = HERE / "contracts"
SOURCE_COMMIT = "b" * 40


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def cpufreq_states() -> dict:
    values = [1600000] * 6
    return {
        "original_cpufreq_state": {"min_khz": values, "max_khz": values, "boost": 0},
        "applied_cpufreq_state": {"min_khz": 1600000, "max_khz": 1600000, "boost": 0},
        "restored_cpufreq_state": {
            "verified": True, "min_khz": values, "max_khz": values, "boost": 0,
        },
    }


def update_run_manifest(run_dir: Path) -> None:
    files = {
        name: {"size": (run_dir / name).stat().st_size, "sha256": sha(run_dir / name)}
        for name in RUN_FILES
    }
    write_json(run_dir / "run_manifest.json", {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_RUN_MANIFEST_V2",
        "session_id": json.loads((run_dir / "session.json").read_text())["session_id"],
        "status": "COMPLETE",
        "files": files,
    })


def csv_echo(row: dict) -> dict:
    result = {}
    for key in (
        "stage", "block_id", "family", "actual_mode", "declared_mode",
        "executed_tone_order", "declared_tone_order", "physical_tone_index",
        "receiver_codeword_source_index", "sender_codeword_source_index",
        "drive_on", "sender_off_required", "measurement_mode",
        "amplitude_level", "receiver_theta_idx", "sender_theta_idx",
        "shared_schedule", "scramble_key_digest",
        "sender_off_control_for_tone_index", "sender_off_control_theta_idx",
    ):
        value = row[key]
        if value is None:
            result[key] = -1
        elif isinstance(value, bool):
            result[key] = 1 if value else 0
        else:
            result[key] = value
    return result


def authorization_for(plan: dict, session_id: str, bundle: dict,
                      bundle_path: Path) -> dict:
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1",
        "calibration_authorized": True, **FALSE_AUTHORIZATIONS,
        "automatic_retry": False,
        "campaign_plan_sha256": hashlib.sha256(canonical_bytes(plan)).hexdigest(),
        "executor_commit": "b" * 40,
        "executor_sha256": "c" * 64,
        "campaign_source_commit": SOURCE_COMMIT,
        "source_bundle_sha256": sha(bundle_path),
        "session_ids": [session_id],
        "route_cores": ROUTE_CORES,
        "pin_khz": 1600000, "slot_s": .5, "off_window_s": .5,
        "read_hz": 8000, "temperature_veto_c": 68.0,
        "authorized_output_root": "/tmp/calibration", "authorized_by": "TEST",
    }


def build_full_campaign_fixture(root: Path, *, mutate=None) -> tuple[Path, Path, dict]:
    plan = build_plan(SOURCE_COMMIT)
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    evidence = {
        "schema_id": "CAT_CAS_PHASE6_V2_CALIBRATION_EVIDENCE_MAP_V2",
        "sessions": {},
    }
    for session in plan["sessions"]:
        session_id = session["session_id"]
        base = root / session_id
        run_dir = base / "run"
        source_dir = base / "source"
        run_dir.mkdir(parents=True)
        source_dir.mkdir(parents=True)
        session_header = {
            "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V2",
            "campaign_source_commit": SOURCE_COMMIT,
            "campaign_plan_sha256": sha(plan_path),
            "session_id": session_id,
            "route": session["route"],
            "partition": session["partition"],
            "window_count": session["window_count"],
            "frequency_settling_required": True,
            "restoration_authorized": False,
        }
        write_json(source_dir / "session.json", session_header)
        (source_dir / "windows.jsonl").write_bytes(
            b"".join(canonical_bytes(row) for row in session["windows"])
        )
        source_files = {
            name: {"size": (source_dir / name).stat().st_size,
                   "sha256": sha(source_dir / name)}
            for name in ("session.json", "windows.jsonl")
        }
        write_json(source_dir / "session_manifest.json", {
            "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V2",
            "session_id": session_id,
            "files": source_files,
        })
        manifest_sha = sha(source_dir / "session_manifest.json")
        bundle = build_source_bundle_manifest({session_id: manifest_sha})
        bundle_path = base / "source_bundle.json"
        write_json(bundle_path, bundle)
        authorization = authorization_for(plan, session_id, bundle, bundle_path)
        authorization_path = base / "authorization.json"
        write_json(authorization_path, authorization)

        write_json(run_dir / "session.json", session_header)
        (run_dir / "windows.jsonl").write_bytes(
            (source_dir / "windows.jsonl").read_bytes()
        )
        raw_records = []
        rows = []
        tsc_hz = 1_000_000.0
        for index, declared in enumerate(session["windows"]):
            origin = 1_000_000 + index * 1_000_000
            spacing = int(tsc_hz / 8000)
            timestamps = origin + np.arange(1, 4097, dtype=np.uint64) * spacing
            span = int(timestamps[-1]) - int(timestamps[0])
            deadline_offset = int(span * 1.10) + 1
            if declared["drive_on"]:
                tone = int(declared["physical_tone_index"])
                sender_phase = phase_index(
                    0, int(declared["sender_codeword_source_index"]),
                    int(declared["sender_theta_idx"]),
                )
                samples = 100.0 * intended_v2_gate(
                    timestamps, origin_tsc=origin, tsc_hz=tsc_hz,
                    tone_index=tone, phase_index_value=sender_phase,
                    amplitude_level=int(declared["amplitude_level"]),
                )
                response = lockin(
                    timestamps, samples, origin_tsc=origin, tsc_hz=tsc_hz,
                    frequency_hz=tone_hz(tone),
                )
                off_bin = lockin(
                    timestamps, samples, origin_tsc=origin, tsc_hz=tsc_hz,
                    frequency_hz=control_frequency_hz(tone_hz(tone)),
                )
                row = {
                    "window_index": index, "session_id": session_id,
                    **csv_echo(declared),
                    "slot_start_tsc": origin,
                    "capture_deadline_tsc": origin + deadline_offset,
                    "sender_ready_tsc": origin - 1,
                    "sender_epoch_tsc": origin + 1,
                    "first_drive_tsc": origin + 1,
                    "receiver_epoch_tsc": origin + 2,
                    "first_sample_tsc": int(timestamps[0]),
                    "last_sample_tsc": int(timestamps[-1]),
                    "sample_count": len(timestamps),
                    "temp_before_c": 40, "temp_after_c": 41,
                    "victim_frequency_before_khz": 1600000,
                    "victim_frequency_after_khz": 1600000,
                    "sender_frequency_before_khz": 1600000,
                    "sender_frequency_after_khz": 1600000,
                    "computed_I": response.real,
                    "computed_Q": response.imag,
                    "magnitude": abs(response),
                    "floor": abs(off_bin),
                    "raw_mean": 50.0, "raw_min": 0.0, "raw_max": 100.0,
                    "aperf_before": 0, "aperf_after": 0,
                    "mperf_before": 0, "mperf_after": 0,
                    "cofvid_before": 0, "cofvid_after": 0,
                    "sender_started": 1,
                    "sender_stopped": 1,
                    "sender_alive_at_capture": 1,
                    "window_status": "OK",
                }
            else:
                tone = int(declared["sender_off_control_for_tone_index"])
                theta = int(declared["sender_off_control_theta_idx"])
                seconds = (timestamps.astype(np.float64) - float(origin)) / tsc_hz
                samples = (0.001 + 0.0002 * theta) * np.sin(
                    2.0 * np.pi * tone_hz(tone) * seconds
                )
                row = {
                    "window_index": index, "session_id": session_id,
                    **csv_echo(declared),
                    "slot_start_tsc": origin,
                    "capture_deadline_tsc": origin + deadline_offset,
                    "sender_ready_tsc": 0,
                    "sender_epoch_tsc": 0,
                    "first_drive_tsc": 0,
                    "receiver_epoch_tsc": origin + 2,
                    "first_sample_tsc": int(timestamps[0]),
                    "last_sample_tsc": int(timestamps[-1]),
                    "sample_count": len(timestamps),
                    "temp_before_c": 40, "temp_after_c": 41,
                    "victim_frequency_before_khz": 1600000,
                    "victim_frequency_after_khz": 1600000,
                    "sender_frequency_before_khz": 1600000,
                    "sender_frequency_after_khz": 1600000,
                    "computed_I": "null",
                    "computed_Q": "null",
                    "magnitude": "null",
                    "floor": "null",
                    "raw_mean": "null", "raw_min": "null", "raw_max": "null",
                    "aperf_before": 0, "aperf_after": 0,
                    "mperf_before": 0, "mperf_after": 0,
                    "cofvid_before": 0, "cofvid_after": 0,
                    "sender_started": 0,
                    "sender_stopped": 1,
                    "sender_alive_at_capture": 0,
                    "window_status": "OK",
                }
            rows.append(row)
            raw_records.extend(zip(timestamps, samples))
        raw = np.array(raw_records, dtype=RAW_DTYPE)
        raw.tofile(run_dir / "raw_samples.bin")
        with (run_dir / "window_results.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(WINDOW_RESULTS_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        (run_dir / "telemetry.csv").write_text(
            "window_index,temp_before_c,temp_after_c,"
            "victim_frequency_before_khz,victim_frequency_after_khz,"
            "sender_frequency_before_khz,sender_frequency_after_khz,"
            "aperf_before,aperf_after,mperf_before,mperf_after,"
            "cofvid_before,cofvid_after\n" +
            "".join(
                f"{i},{row['temp_before_c']},{row['temp_after_c']},"
                f"{row['victim_frequency_before_khz']},{row['victim_frequency_after_khz']},"
                f"{row['sender_frequency_before_khz']},{row['sender_frequency_after_khz']},"
                "0,0,0,0,0,0\n"
                for i, row in enumerate(rows)
            ),
            encoding="utf-8",
        )
        for name in ("stdout.log", "stderr.log", "orchestrator_stdout.log",
                     "orchestrator_stderr.log"):
            (run_dir / name).write_text("", encoding="utf-8")
        write_json(run_dir / "run.json", {
            "schema_id": "CAT_CAS_PHASE6_COMBINED_RUN_V2",
            "session_id": session_id, "route": session["route"],
            "campaign_source_commit": SOURCE_COMMIT,
            "campaign_plan_sha256": sha(plan_path),
            "session_manifest_sha256": manifest_sha,
            "executor_git_commit": "b" * 40,
            "host_identity": "test-host", "kernel_identity": "test-kernel",
            "cpu_model": "test-cpu",
            "authorization_artifact_sha256": sha(authorization_path),
            "execution_class": "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION",
            "victim_core": ROUTE_CORES[session["route"]][0],
            "sender_core": ROUTE_CORES[session["route"]][1],
            "tsc_calibration_hz": tsc_hz, "frequency_policy": 1600000,
            "read_rate_hz": 8000, "slot_duration_s": .5,
            "sender_off_duration_s": .5, "temperature_veto_c": 68.0,
            "start_timestamp": 1, "end_timestamp": 2,
            "exit_status": "COMPLETE", "failure_reason": "",
            "host_control_state_restored": True,
            "physical_carrier_restoration_claimed": False,
            "automatic_retry": False, "restoration_authorized": False,
            "calibration_authorized": True, "acquisition_authorized": False,
            "scientific_acquisition_authorized": False,
            "target_coupling_authorized": False, "small_wall_authorized": False,
            "hardware_executed": True, **cpufreq_states(),
        })
        update_run_manifest(run_dir)
        evidence["sessions"][session_id] = {
            "run_dir": str(run_dir),
            "authorization": str(authorization_path),
            "source_bundle": str(bundle_path),
            "session_manifest": str(source_dir / "session_manifest.json"),
        }
    if mutate:
        mutate(evidence)
    evidence_path = root / "evidence_map.json"
    write_json(evidence_path, evidence)
    return plan_path, evidence_path, plan


def build_fixture(root: Path):
    run_dir = root / "run"
    run_dir.mkdir()
    session_id = "v4s5_test"
    window = {
        "window_index": 0, "session_id": session_id,
        "stage": "V2_SPECTRAL_CALIBRATION_SENDER_OFF", "block_id": "tone_00",
        "family": "silent", "actual_mode": "null", "declared_mode": "null",
        "executed_tone_order": "ASC", "declared_tone_order": "ASC",
        "measurement_mode": "raw_ring_sender_off", "drive_on": False,
        "sender_off_required": True, "physical_tone_index": None,
        "receiver_codeword_source_index": None, "sender_codeword_source_index": None,
        "receiver_theta_idx": None, "sender_theta_idx": None, "shared_schedule": True,
        "scramble_key_digest": "0" * 64, "amplitude_level": 0,
        "expected_code_sign": 0, "sender_off_control_for_tone_index": 0,
        "sender_off_control_theta_idx": 0,
    }
    plan = build_plan(SOURCE_COMMIT)
    planned_session = {
        "session_id": session_id, "route": "v4s5", "route_cores": [4, 5],
        "partition": "PRE_REBOOT_REPETITION",
        "frequency_settling_required": True,
        "window_count": 1, "windows": [window],
    }
    plan["sessions"] = [planned_session]
    plan["session_ids"] = [session_id]
    plan["session_count"] = 1
    plan["windows_per_session"] = 1
    plan["windows_per_route"] = {"v4s5": 1, "v2s3": 0}
    plan["total_window_count"] = 1
    plan["count_derivation"] = "TEST_SINGLE_WINDOW"
    plan_digest = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    source_bundle = {
        "schema_id": "CAT_CAS_PHASE6_V2_SOURCE_BUNDLE_MANIFEST_V1",
        "sessions": {session_id: "a" * 64},
    }
    authorization = {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1",
        "calibration_authorized": True, **FALSE_AUTHORIZATIONS,
        "automatic_retry": False, "campaign_plan_sha256": plan_digest,
        "executor_commit": "b" * 40, "executor_sha256": "c" * 64,
        "campaign_source_commit": SOURCE_COMMIT,
        "source_bundle_sha256": hashlib.sha256(canonical_bytes(source_bundle)).hexdigest(),
        "session_ids": [session_id], "route_cores": ROUTE_CORES,
        "pin_khz": 1600000, "slot_s": .5, "off_window_s": .5,
        "read_hz": 8000, "temperature_veto_c": 68.0,
        "authorized_output_root": "/tmp/calibration", "authorized_by": "TEST",
    }
    authorization_bytes = canonical_bytes(authorization)
    authorization["artifact_sha256"] = hashlib.sha256(authorization_bytes).hexdigest()
    session = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V2",
        "campaign_source_commit": SOURCE_COMMIT,
        "campaign_plan_sha256": plan_digest,
        "session_id": session_id, "route": "v4s5",
        "partition": "PRE_REBOOT_REPETITION", "window_count": 1,
        "frequency_settling_required": True, "restoration_authorized": False,
    }
    (run_dir / "session.json").write_bytes(canonical_bytes(session))
    (run_dir / "windows.jsonl").write_bytes(canonical_bytes(window))
    spacing = 125
    raw = np.array([(100 + i * spacing, 1.0 + 0.1 * i) for i in range(64)], dtype=RAW_DTYPE)
    raw.tofile(run_dir / "raw_samples.bin")
    span = int(raw["timestamp_tsc"][-1]) - int(raw["timestamp_tsc"][0])
    deadline = max(int(raw["timestamp_tsc"][-1]) + 1, int(span * 1.09) + 1)
    result_row = {
        "window_index": 0, "session_id": session_id, "sample_count": 64,
        "first_sample_tsc": int(raw["timestamp_tsc"][0]),
        "last_sample_tsc": int(raw["timestamp_tsc"][-1]),
        "slot_start_tsc": 0,
        "capture_deadline_tsc": deadline,
        "sender_ready_tsc": 0, "sender_epoch_tsc": 0, "first_drive_tsc": 0,
        "receiver_epoch_tsc": 50,
        "temp_before_c": 40, "temp_after_c": 41,
        "victim_frequency_before_khz": 1600000,
        "victim_frequency_after_khz": 1600000,
        "sender_frequency_before_khz": 1600000,
        "sender_frequency_after_khz": 1600000,
        "sender_started": 0, "sender_stopped": 1,
        "sender_alive_at_capture": 0, "computed_I": "null",
        "computed_Q": "null", "magnitude": "null", "floor": "null",
        "raw_mean": "null", "raw_min": "null", "raw_max": "null",
        "aperf_before": 0, "aperf_after": 0,
        "mperf_before": 0, "mperf_after": 0,
        "cofvid_before": 0, "cofvid_after": 0,
        "window_status": "OK",
    }
    for key, value in window.items():
        if key in {
            "stage", "block_id", "family", "actual_mode", "declared_mode",
            "executed_tone_order", "declared_tone_order", "physical_tone_index",
            "receiver_codeword_source_index", "sender_codeword_source_index",
            "drive_on", "sender_off_required", "measurement_mode",
            "amplitude_level", "receiver_theta_idx", "sender_theta_idx",
            "shared_schedule", "scramble_key_digest",
            "sender_off_control_for_tone_index", "sender_off_control_theta_idx",
        }:
            if value is None:
                result_row[key] = -1
            elif isinstance(value, bool):
                result_row[key] = 1 if value else 0
            else:
                result_row[key] = value
    with (run_dir / "window_results.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(WINDOW_RESULTS_COLUMNS))
        writer.writeheader(); writer.writerow(result_row)
    run = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_RUN_V2",
        "session_id": session_id, "route": "v4s5", "campaign_plan_sha256": plan_digest,
        "campaign_source_commit": SOURCE_COMMIT, "session_manifest_sha256": "a" * 64,
        "executor_git_commit": "b" * 40,
        "host_identity": "test-host", "kernel_identity": "test-kernel",
        "cpu_model": "test-cpu",
        "authorization_artifact_sha256": authorization["artifact_sha256"],
        "execution_class": "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION",
        "victim_core": 4, "sender_core": 5,
        "tsc_calibration_hz": 1_000_000.0, "frequency_policy": 1600000,
        "read_rate_hz": 8000, "slot_duration_s": .5,
        "sender_off_duration_s": .5, "temperature_veto_c": 68.0,
        "start_timestamp": 1, "end_timestamp": 2,
        "exit_status": "COMPLETE", "failure_reason": "",
        "host_control_state_restored": True,
        "physical_carrier_restoration_claimed": False,
        "automatic_retry": False, "restoration_authorized": False,
        "calibration_authorized": True, "acquisition_authorized": False,
        "scientific_acquisition_authorized": False,
        "target_coupling_authorized": False, "small_wall_authorized": False,
        "hardware_executed": True, **cpufreq_states(),
    }
    (run_dir / "run.json").write_bytes(canonical_bytes(run))
    (run_dir / "telemetry.csv").write_text(
        "window_index,temp_before_c,temp_after_c,"
        "victim_frequency_before_khz,victim_frequency_after_khz,"
        "sender_frequency_before_khz,sender_frequency_after_khz,"
        "aperf_before,aperf_after,mperf_before,mperf_after,"
        "cofvid_before,cofvid_after\n"
        f"0,{result_row['temp_before_c']},{result_row['temp_after_c']},"
        f"{result_row['victim_frequency_before_khz']},{result_row['victim_frequency_after_khz']},"
        f"{result_row['sender_frequency_before_khz']},{result_row['sender_frequency_after_khz']},"
        "0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    for name in RUN_FILES - {
        "run.json", "session.json", "windows.jsonl", "window_results.csv",
        "raw_samples.bin", "telemetry.csv"
    }:
        (run_dir / name).write_bytes(b"")
    files = {
        name: {"size": (run_dir / name).stat().st_size, "sha256": sha(run_dir / name)}
        for name in RUN_FILES
    }
    (run_dir / "run_manifest.json").write_bytes(canonical_bytes({
        "schema_id": "CAT_CAS_PHASE6_COMBINED_RUN_MANIFEST_V2",
        "session_id": session_id,
        "status": "COMPLETE",
        "files": files,
    }))
    return run_dir, plan, authorization, source_bundle


class SpectralAnalyzerTests(unittest.TestCase):
    def assert_run_tamper_rejected(self, key: str, value: object, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, plan, authorization, bundle = build_fixture(Path(temp))
            run_path = run_dir / "run.json"
            run = json.loads(run_path.read_text())
            run[key] = value
            run_path.write_bytes(canonical_bytes(run))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["run.json"] = {
                "size": run_path.stat().st_size,
                "sha256": sha(run_path),
            }
            manifest_path.write_bytes(canonical_bytes(manifest))
            with self.assertRaisesRegex(ValueError, pattern):
                analyze_run(run_dir, plan, authorization, bundle)

    def test_incomplete_schedule_is_rejected_before_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = build_fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "complete tone/amplitude"):
                analyze_run(*args)
        run_cases = (
            ("calibration_authorized", False, "calibration-authorized"),
            ("exit_status", "FAILED", "completion binding"),
            ("failure_reason", "ERROR", "completion binding"),
            ("host_control_state_restored", False, "not restored"),
            ("session_manifest_sha256", "f" * 64, "session-manifest"),
            ("victim_core", 5, "route/core"),
            ("sender_core", 4, "route/core"),
            ("frequency_policy", 1, "frequency_policy"),
            ("read_rate_hz", 1, "read_rate_hz"),
            ("slot_duration_s", 1.0, "slot_duration_s"),
            ("sender_off_duration_s", 1.0, "sender_off_duration_s"),
            ("temperature_veto_c", 1.0, "temperature_veto_c"),
        )
        for key, value, pattern in run_cases:
            with self.subTest(key=key):
                self.assert_run_tamper_rejected(key, value, pattern)
        manifest_cases = (
            ("schema_id", "BAD"), ("session_id", "other"), ("status", "FAILED")
        )
        for key, value in manifest_cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                args = build_fixture(Path(temp))
                path = args[0] / "run_manifest.json"
                manifest = json.loads(path.read_text())
                manifest[key] = value
                path.write_bytes(canonical_bytes(manifest))
                with self.assertRaisesRegex(ValueError, "run-manifest"):
                    analyze_run(*args)

    def test_manifest_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, plan, authorization, bundle = build_fixture(Path(temp))
            (run_dir / "raw_samples.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "manifest binding mismatch"):
                analyze_run(run_dir, plan, authorization, bundle)

    def test_nonfinite_raw_sample_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, plan, authorization, bundle = build_fixture(Path(temp))
            raw = np.memmap(run_dir / "raw_samples.bin", dtype=RAW_DTYPE, mode="r+")
            raw[0]["ring_period"] = np.nan
            raw.flush()
            del raw
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            path = run_dir / "raw_samples.bin"
            manifest["files"]["raw_samples.bin"] = {"size": path.stat().st_size, "sha256": sha(path)}
            (run_dir / "run_manifest.json").write_bytes(canonical_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                analyze_run(run_dir, plan, authorization, bundle)

    def test_same_byte_custody_does_not_reopen_inputs_for_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = build_fixture(Path(temp))
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("path reopened")), \
                    mock.patch.object(Path, "read_text", side_effect=AssertionError("path reopened")), \
                    mock.patch.object(Path, "open", side_effect=AssertionError("path reopened")), \
                    mock.patch.object(np, "fromfile", side_effect=AssertionError("raw path reopened")):
                with self.assertRaisesRegex(ValueError, "complete tone/amplitude"):
                    analyze_run(*args)

    def test_run_manifest_binding_requires_exact_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, plan, authorization, bundle = build_fixture(Path(temp))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["run.json"]["unexpected"] = True
            manifest_path.write_bytes(canonical_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "binding run.json fields mismatch"):
                analyze_run(run_dir, plan, authorization, bundle)

    def test_run_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir, plan, authorization, bundle = build_fixture(root)
            linked = root / "linked-run"
            try:
                os.symlink(run_dir, linked, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(ValueError, "symlink traversal"):
                analyze_run(linked, plan, authorization, bundle)

    def test_raw_trailing_record_is_rejected_from_captured_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, plan, authorization, bundle = build_fixture(Path(temp))
            path = run_dir / "raw_samples.bin"
            with path.open("ab") as target:
                target.write(np.array([(999999, 1.0)], dtype=RAW_DTYPE).tobytes())
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["raw_samples.bin"] = {
                "size": path.stat().st_size, "sha256": sha(path),
            }
            manifest_path.write_bytes(canonical_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "raw binary size"):
                analyze_run(run_dir, plan, authorization, bundle)

    def test_plan_and_nested_window_schemas_are_exact(self) -> None:
        plan = build_plan(SOURCE_COMMIT)
        extra = json.loads(json.dumps(plan))
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "plan fields mismatch"):
            validate_plan_schema(extra, require_full_campaign=True)
        nested = json.loads(json.dumps(plan))
        nested["sessions"][0]["windows"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "plan window fields mismatch"):
            validate_plan_schema(nested, require_full_campaign=True)

    def test_campaign_requires_exact_reboot_and_route_session_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete exact calibration session set"):
            analyze_campaign([], build_plan())

    def test_exact_generated_schedule_constructs_complete_sender_grid(self) -> None:
        plan = json.loads((CONTRACTS / "CALIBRATION_PLAN_V2.json").read_text())
        for session in plan["sessions"]:
            schedule_path = CONTRACTS / "sessions" / session["session_id"] / "windows.jsonl"
            schedule = [json.loads(line) for line in schedule_path.read_text().splitlines()]
            self.assertEqual(schedule, session["windows"])
            self.assertEqual(len(schedule), 672)
            grid = construct_complete_grid(schedule)
            self.assertEqual(len(grid), 576)

    def test_campaign_applies_repetition_and_cross_route_rule(self) -> None:
        plan = build_plan()
        results = []
        for session in plan["sessions"]:
            results.append({
                "session_id": session["session_id"],
                "route": session["route"],
                "verdict": "SESSION_CALIBRATION_PASS",
                "measurements": [
                    {"response_real": 1.0, "response_imag": 0.0},
                    {"response_real": 2.0, "response_imag": 1.0},
                    {"response_real": 4.0, "response_imag": -1.0},
                ],
            })
        result = analyze_campaign(results, plan)
        self.assertEqual(result["verdict"], "CALIBRATION_PASS")
        self.assertTrue(all(item["pass"] for item in
                            result["repeated_session_consistency"].values()))
        self.assertFalse(result["acquisition_authorized"])

    def test_cli_validates_four_session_evidence_map_to_campaign_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path, evidence_path, _plan = build_full_campaign_fixture(root)
            output = root / "analysis.json"
            with mock.patch.object(sys, "argv", [
                "analyze_spectral_calibration_v2.py",
                "--plan", str(plan_path),
                "--evidence-map", str(evidence_path),
                "--output", str(output),
            ]):
                self.assertEqual(analyzer_main(), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["verdict"], "CALIBRATION_PASS")
            self.assertEqual(len(result["session_results"]), 4)

    def test_evidence_map_missing_extra_duplicate_and_swapped_paths_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path, evidence_path, plan = build_full_campaign_fixture(root)
            evidence = json.loads(evidence_path.read_text())
            first, second = plan["session_ids"][:2]
            cases = []
            missing = json.loads(json.dumps(evidence))
            missing["sessions"].pop(first)
            cases.append((missing, "session set"))
            extra = json.loads(json.dumps(evidence))
            extra["sessions"]["extra_session"] = extra["sessions"][first]
            cases.append((extra, "session set"))
            duplicate_path = json.loads(json.dumps(evidence))
            duplicate_path["sessions"][second]["authorization"] = \
                duplicate_path["sessions"][first]["authorization"]
            cases.append((duplicate_path, "path reused"))
            swapped_authorization = json.loads(json.dumps(evidence))
            swapped_authorization["sessions"][first]["authorization"], \
                swapped_authorization["sessions"][second]["authorization"] = (
                    swapped_authorization["sessions"][second]["authorization"],
                    swapped_authorization["sessions"][first]["authorization"],
                )
            cases.append((swapped_authorization, "source bundle digest"))
            swapped_bundle = json.loads(json.dumps(evidence))
            swapped_bundle["sessions"][first]["source_bundle"], \
                swapped_bundle["sessions"][second]["source_bundle"] = (
                    swapped_bundle["sessions"][second]["source_bundle"],
                    swapped_bundle["sessions"][first]["source_bundle"],
                )
            cases.append((swapped_bundle, "source bundle digest"))
            for index, (candidate, pattern) in enumerate(cases):
                with self.subTest(index=index):
                    candidate_path = root / f"bad_map_{index}.json"
                    write_json(candidate_path, candidate)
                    with mock.patch.object(sys, "argv", [
                        "analyze_spectral_calibration_v2.py",
                        "--plan", str(plan_path),
                        "--evidence-map", str(candidate_path),
                        "--output", str(root / f"bad_{index}.json"),
                    ]):
                        with self.assertRaisesRegex(ValueError, pattern):
                            analyzer_main()

    def test_sender_off_controls_fail_closed_for_count_variance_and_boundary(self) -> None:
        def run_with_mutation(mutator, pattern: str) -> None:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plan_path, evidence_path, plan = build_full_campaign_fixture(root)
                session_id = plan["session_ids"][0]
                run_dir = Path(json.loads(evidence_path.read_text())["sessions"][session_id]["run_dir"])
                mutator(run_dir)
                update_run_manifest(run_dir)
                with mock.patch.object(sys, "argv", [
                    "analyze_spectral_calibration_v2.py",
                    "--plan", str(plan_path),
                    "--evidence-map", str(evidence_path),
                    "--output", str(root / "bad_analysis.json"),
                ]):
                    with self.assertRaisesRegex(ValueError, pattern):
                        analyzer_main()

        def run_to_fail_verdict(mutator) -> None:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plan_path, evidence_path, plan = build_full_campaign_fixture(root)
                session_id = plan["session_ids"][0]
                run_dir = Path(json.loads(evidence_path.read_text())["sessions"][session_id]["run_dir"])
                mutator(run_dir)
                update_run_manifest(run_dir)
                output = root / "fail_analysis.json"
                with mock.patch.object(sys, "argv", [
                    "analyze_spectral_calibration_v2.py",
                    "--plan", str(plan_path),
                    "--evidence-map", str(evidence_path),
                    "--output", str(output),
                ]):
                    self.assertEqual(analyzer_main(), 2)
                self.assertEqual(json.loads(output.read_text())["verdict"], "CALIBRATION_FAIL")

        def missing_control(run_dir: Path) -> None:
            windows = (run_dir / "windows.jsonl").read_text().splitlines()
            rows = [json.loads(line) for line in windows]
            target = next(i for i, row in enumerate(rows)
                          if row["sender_off_required"] and
                          row["sender_off_control_for_tone_index"] == 0)
            rows.pop(target)
            (run_dir / "windows.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in rows),
                encoding="utf-8",
            )

        def one_control(run_dir: Path) -> None:
            rows = [json.loads(line) for line in (run_dir / "windows.jsonl").read_text().splitlines()]
            kept = False
            filtered = []
            for row in rows:
                if row["sender_off_required"] and row["sender_off_control_for_tone_index"] == 0:
                    if kept:
                        continue
                    kept = True
                filtered.append(row)
            (run_dir / "windows.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in filtered),
                encoding="utf-8",
            )

        def duplicate_control(run_dir: Path) -> None:
            rows = [json.loads(line) for line in (run_dir / "windows.jsonl").read_text().splitlines()]
            controls = [row for row in rows if row["sender_off_required"] and
                        row["sender_off_control_for_tone_index"] == 0]
            controls[-1]["sender_off_control_theta_idx"] = controls[0]["sender_off_control_theta_idx"]
            (run_dir / "windows.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in rows),
                encoding="utf-8",
            )

        def zero_variance(run_dir: Path) -> None:
            raw = np.memmap(run_dir / "raw_samples.bin", dtype=RAW_DTYPE, mode="r+")
            with (run_dir / "window_results.csv").open(newline="", encoding="utf-8") as source:
                results = list(csv.DictReader(source))
            for row in results:
                if row["sender_off_required"] == "1" and row["block_id"] == "tone_00":
                    start = sum(int(prev["sample_count"]) for prev in results[:int(row["window_index"])])
                    stop = start + int(row["sample_count"])
                    raw["ring_period"][start:stop] = 0.001
            raw.flush()
            del raw

        def below_boundary(run_dir: Path) -> None:
            raw = np.memmap(run_dir / "raw_samples.bin", dtype=RAW_DTYPE, mode="r+")
            with (run_dir / "window_results.csv").open(newline="", encoding="utf-8") as source:
                results = list(csv.DictReader(source))
            for row in results:
                if row["drive_on"] == "1" and row["block_id"] == "tone_00":
                    start = sum(int(prev["sample_count"]) for prev in results[:int(row["window_index"])])
                    stop = start + int(row["sample_count"])
                    raw["ring_period"][start:stop] *= 0.0
                    row["computed_I"] = "0"
                    row["computed_Q"] = "0"
                    row["magnitude"] = "0"
                    row["floor"] = "0"
                    break
            raw.flush()
            del raw
            with (run_dir / "window_results.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)

        run_with_mutation(one_control, "planned window count mismatch")
        run_with_mutation(missing_control, "planned window count mismatch")
        run_with_mutation(duplicate_control, "ordered window set differs")
        run_with_mutation(zero_variance, "zero-variance")
        run_to_fail_verdict(below_boundary)



    def test_valid_session_manifest_binding_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path, evidence_path, plan = build_full_campaign_fixture(root)
            output = root / "analysis.json"
            with mock.patch.object(sys, "argv", [
                "analyze_spectral_calibration_v2.py", "--plan", str(plan_path), "--evidence-map", str(evidence_path), "--output", str(output),
            ]):
                self.assertEqual(analyzer_main(), 0)


if __name__ == "__main__":
    unittest.main()
