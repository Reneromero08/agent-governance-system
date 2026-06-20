#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from catcas_preflight import (
    authorization_valid,
    session_bundles_valid,
    target_engineering_valid,
    validation_report_valid,
)
from collect_target_engineering_evidence import smoke_checks


WINDOW_FIELDS = [
    "drive_on",
    "sender_off_required",
    "slot_start_tsc",
    "capture_deadline_tsc",
    "sender_ready_tsc",
    "sender_epoch_tsc",
    "first_drive_tsc",
    "receiver_epoch_tsc",
    "first_sample_tsc",
    "last_sample_tsc",
    "sender_started",
    "sender_stopped",
    "sender_alive_at_capture",
    "computed_I",
    "computed_Q",
    "window_status",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(run: Path, names: tuple[str, ...]) -> None:
    files = {
        name: {"size": (run / name).stat().st_size, "sha256": sha256_file(run / name)}
        for name in names
    }
    write_json(run / "run_manifest.json", {"files": files})


def smoke_rows() -> list[dict[str, str]]:
    return [
        {
            "drive_on": "1",
            "sender_off_required": "0",
            "slot_start_tsc": "100000",
            "capture_deadline_tsc": "200000",
            "sender_ready_tsc": "90000",
            "sender_epoch_tsc": "100001",
            "first_drive_tsc": "100050",
            "receiver_epoch_tsc": "100002",
            "first_sample_tsc": "100003",
            "last_sample_tsc": "199000",
            "sender_started": "1",
            "sender_stopped": "1",
            "sender_alive_at_capture": "1",
            "computed_I": "1",
            "computed_Q": "2",
            "window_status": "OK",
        },
        {
            "drive_on": "1",
            "sender_off_required": "0",
            "slot_start_tsc": "300000",
            "capture_deadline_tsc": "400000",
            "sender_ready_tsc": "290000",
            "sender_epoch_tsc": "300001",
            "first_drive_tsc": "300050",
            "receiver_epoch_tsc": "300002",
            "first_sample_tsc": "300003",
            "last_sample_tsc": "399000",
            "sender_started": "1",
            "sender_stopped": "1",
            "sender_alive_at_capture": "1",
            "computed_I": "1",
            "computed_Q": "2",
            "window_status": "OK",
        },
        {
            "drive_on": "0",
            "sender_off_required": "1",
            "slot_start_tsc": "500000",
            "capture_deadline_tsc": "600000",
            "sender_ready_tsc": "0",
            "sender_epoch_tsc": "0",
            "first_drive_tsc": "0",
            "receiver_epoch_tsc": "500001",
            "first_sample_tsc": "500002",
            "last_sample_tsc": "599000",
            "sender_started": "0",
            "sender_stopped": "1",
            "sender_alive_at_capture": "0",
            "computed_I": "null",
            "computed_Q": "null",
            "window_status": "OK",
        },
    ]


def host_state() -> dict[str, object]:
    six_true = {str(index): True for index in range(6)}
    mins = {str(index): 800000 for index in range(6)}
    maxes = {str(index): 3200000 for index in range(6)}
    return {
        "host": "catcas",
        "effective_uid": 0,
        "cpu_count": 6,
        "cpu_flags": ["constant_tsc", "nonstop_tsc"],
        "k10temp_path": "/sys/class/hwmon/hwmon0/temp1_input",
        "msr_readable": six_true,
        "cpufreq_controls": six_true,
        "cpufreq_min_khz": mins,
        "cpufreq_max_khz": maxes,
        "boost": 1,
        "free_bytes": 100 * 1024**3,
        "runner_processes": [],
    }


def build_target_bundle(root: Path) -> dict[str, object]:
    commit = "a" * 40
    runner = root / "combined_pdn_runner"
    runner.write_bytes(b"runner")
    source_transfer = root / "source_transfer_bundle.json"
    source_transfer.write_text("{}\n", encoding="utf-8")
    source_digest = sha256_file(source_transfer)
    (root / "source_transfer_bundle.sha256").write_text(
        f"{source_digest}  source_transfer_bundle.json\n", encoding="utf-8"
    )

    evidence = root / "evidence"
    evidence.mkdir()
    before = host_state()
    write_json(
        evidence / "before_snapshot.json",
        {"schema_id": "CAT_CAS_PHASE6_TARGET_SNAPSHOT_V1", "host_state": before},
    )

    smoke = root / "smoke"
    smoke.mkdir()
    smoke_run = {
        "session_id": "ENGINEERING_SMOKE_TEST",
        "executor_git_commit": commit,
        "execution_class": "ENGINEERING_SMOKE_NOT_SCIENTIFIC_ACQUISITION",
        "authorization_artifact_sha256": None,
        "scientific_acquisition_authorized": False,
        "exit_status": "COMPLETE",
        "hardware_executed": True,
        "host_control_state_restored": True,
        "physical_carrier_restoration_claimed": False,
        "automatic_retry": False,
        "restoration_authorized": False,
        "tsc_calibration_hz": 1_000_000_000,
    }
    write_json(smoke / "run.json", smoke_run)
    rows = smoke_rows()
    with (smoke / "window_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_manifest(smoke, ("run.json", "window_results.csv"))
    _, _, smoke_result = smoke_checks(smoke)

    late = root / "late"
    late.mkdir()
    late_run = {
        "exit_status": "FAILED",
        "failure_reason": "SENDER_EPOCH_ALIGNMENT_FAILURE",
        "executor_git_commit": commit,
        "execution_class": "MOCK_HARDWARE_TEST",
        "authorization_artifact_sha256": None,
        "scientific_acquisition_authorized": False,
        "hardware_executed": False,
        "host_control_state_restored": True,
        "automatic_retry": False,
        "physical_carrier_restoration_claimed": False,
    }
    write_json(late / "run.json", late_run)
    write_manifest(late, ("run.json",))

    checks = {
        **{f"smoke_{name}": value for name, value in smoke_result.items()},
        "cleanup_cpufreq_min_restored": True,
        "cleanup_cpufreq_max_restored": True,
        "cleanup_boost_restored": True,
        "cleanup_no_runner_processes": True,
        "late_sender_failed_closed": True,
        "late_sender_executor_commit_recorded": True,
        "late_sender_correct_reason": True,
        "late_sender_mock_did_not_touch_hardware": True,
        "late_sender_mock_execution_class": True,
        "late_sender_scientific_acquisition_not_authorized": True,
        "late_sender_host_control_state_restored": True,
        "late_sender_automatic_retry_disabled": True,
        "late_sender_physical_carrier_restoration_not_claimed": True,
    }
    report = {
        "schema_id": "CAT_CAS_PHASE6_TARGET_ENGINEERING_EVIDENCE_V1",
        "executor_commit": commit,
        "source_transfer_bundle_sha256": source_digest,
        "executor_sha256": sha256_file(runner),
        "before_snapshot": "evidence/before_snapshot.json",
        "smoke_run_dir": "smoke",
        "late_sender_run_dir": "late",
        "before_host_state": before,
        "after_host_state": before,
        "smoke_run": smoke_run,
        "smoke_rows": rows,
        "late_sender_run": late_run,
        "checks": checks,
        "all_pass": True,
        "scientific_acquisition_started": False,
        "physical_carrier_restoration_claimed": False,
    }
    write_json(evidence / "target_engineering_report.json", report)
    return {
        "executor_commit": commit,
        "executor_sha256": sha256_file(runner),
        "source_transfer_bundle_sha256": source_digest,
        "evidence": {"target_engineering_report": "evidence/target_engineering_report.json"},
    }


class PreflightShapeTests(unittest.TestCase):
    def test_session_manifest_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            session.mkdir()
            (session / "session_manifest.json").write_text("[]", encoding="utf-8")
            ok, errors = session_bundles_valid(root)
            self.assertFalse(ok)
            self.assertTrue(any("JSON object" in error for error in errors))

    def test_validation_report_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.json"
            path.write_text("[]", encoding="utf-8")
            self.assertTrue(any("JSON object" in error for error in validation_report_valid(path)))

    def test_authorization_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            bundle = root / "source_bundle.json"
            authorization.write_text("[]", encoding="utf-8")
            bundle.write_text("{}", encoding="utf-8")
            ok, _, errors = authorization_valid(
                authorization, bundle_path=bundle, bundle={}, output_root=root / "out"
            )
            self.assertFalse(ok)
            self.assertTrue(any("JSON object" in error for error in errors))

    def test_target_evidence_is_mandatory_for_engineering_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = {"evidence": {}}
            ok, errors, checks, report = target_engineering_valid(root, bundle, 0.0)
            self.assertFalse(ok)
            self.assertEqual(checks, {})
            self.assertEqual(report, {})
            self.assertIn("missing target engineering report", errors)

    def test_target_evidence_is_recomputed_from_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_target_bundle(root)
            ok, errors, checks, _ = target_engineering_valid(root, bundle, 1.0)
            self.assertTrue(ok, errors)
            self.assertTrue(all(checks.values()), checks)

    def test_tampered_before_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_target_bundle(root)
            snapshot = root / "evidence" / "before_snapshot.json"
            value = json.loads(snapshot.read_text(encoding="utf-8"))
            value["host_state"]["boost"] = 0
            write_json(snapshot, value)
            ok, errors, checks, _ = target_engineering_valid(root, bundle, 1.0)
            self.assertFalse(ok)
            self.assertFalse(checks["before_snapshot_matches_report"])
            self.assertTrue(any("before_snapshot_matches_report" in error for error in errors))

    def test_tampered_source_transfer_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = build_target_bundle(root)
            (root / "source_transfer_bundle.sha256").write_text(
                f"{'0' * 64}  source_transfer_bundle.json\n", encoding="utf-8"
            )
            ok, errors, checks, _ = target_engineering_valid(root, bundle, 1.0)
            self.assertFalse(ok)
            self.assertFalse(checks["source_transfer_binding_present"])
            self.assertTrue(any("source_transfer_binding_present" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
