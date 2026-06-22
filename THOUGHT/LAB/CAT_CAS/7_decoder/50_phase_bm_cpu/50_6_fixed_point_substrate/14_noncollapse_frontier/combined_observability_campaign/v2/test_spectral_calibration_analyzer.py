from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_spectral_calibration_v2 import (
    RAW_DTYPE,
    RUN_FILES,
    analyze_campaign,
    analyze_run,
)
from calibration_contract import (
    FALSE_AUTHORIZATIONS,
    ROUTE_CORES,
    build_plan,
    canonical_bytes,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    }
    plan = {
        "schema_id": "TEST_PLAN", "sessions": [{
            "session_id": session_id, "route": "v4s5", "windows": [window]
        }],
    }
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
        "source_bundle_sha256": hashlib.sha256(canonical_bytes(source_bundle)).hexdigest(),
        "session_ids": [session_id], "route_cores": ROUTE_CORES,
        "pin_khz": 1600000, "slot_s": .5, "off_window_s": .5,
        "read_hz": 4000, "temperature_veto_c": 68.0,
        "authorized_output_root": "/tmp/calibration", "authorized_by": "TEST",
    }
    authorization_bytes = canonical_bytes(authorization)
    authorization["artifact_sha256"] = hashlib.sha256(authorization_bytes).hexdigest()
    session = {"session_id": session_id, "route": "v4s5"}
    (run_dir / "session.json").write_bytes(canonical_bytes(session))
    (run_dir / "windows.jsonl").write_bytes(canonical_bytes(window))
    raw = np.array([(100, 1.0), (200, 2.0), (300, 1.5), (400, 2.5)], dtype=RAW_DTYPE)
    raw.tofile(run_dir / "raw_samples.bin")
    result_row = {
        "window_index": 0, "session_id": session_id, "sample_count": 4,
        "first_sample_tsc": 100, "last_sample_tsc": 400, "slot_start_tsc": 0,
        "temp_before_c": 40, "temp_after_c": 41,
        "frequency_before_khz": 1600000, "frequency_after_khz": 1600000,
    }
    with (run_dir / "window_results.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(result_row))
        writer.writeheader(); writer.writerow(result_row)
    run = {
        "session_id": session_id, "route": "v4s5", "campaign_plan_sha256": plan_digest,
        "executor_git_commit": "b" * 40,
        "authorization_artifact_sha256": authorization["artifact_sha256"],
        "execution_class": "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION",
        "hardware_executed": True, **FALSE_AUTHORIZATIONS,
        "tsc_calibration_hz": 1_000_000.0, "frequency_policy": 1600000,
    }
    (run_dir / "run.json").write_bytes(canonical_bytes(run))
    for name in RUN_FILES - {
        "run.json", "session.json", "windows.jsonl", "window_results.csv", "raw_samples.bin"
    }:
        (run_dir / name).write_bytes(b"")
    files = {
        name: {"size": (run_dir / name).stat().st_size, "sha256": sha(run_dir / name)}
        for name in RUN_FILES
    }
    (run_dir / "run_manifest.json").write_bytes(canonical_bytes({"files": files}))
    return run_dir, plan, authorization, source_bundle


class SpectralAnalyzerTests(unittest.TestCase):
    def test_structural_measurement_without_thresholds_is_not_adjudicable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = build_fixture(Path(temp))
            result = analyze_run(*args)
        self.assertEqual(result["verdict"], "CALIBRATION_NOT_ADJUDICABLE_WITHOUT_FROZEN_THRESHOLDS")
        self.assertEqual(result["record_count"], 4)
        self.assertFalse(result["acquisition_authorized"])

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

    def test_campaign_requires_exact_reboot_and_route_session_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete exact calibration session set"):
            analyze_campaign([], build_plan())

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


if __name__ == "__main__":
    unittest.main()
