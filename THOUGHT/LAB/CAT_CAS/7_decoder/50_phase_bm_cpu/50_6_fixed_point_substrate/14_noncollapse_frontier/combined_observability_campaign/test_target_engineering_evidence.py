#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from collect_target_engineering_evidence import (
    is_runner_process,
    smoke_checks,
    validation_report_command,
)


FIELDS = [
    "drive_on", "sender_off_required", "slot_start_tsc", "capture_deadline_tsc",
    "sender_ready_tsc", "sender_epoch_tsc", "first_drive_tsc",
    "receiver_epoch_tsc", "first_sample_tsc", "last_sample_tsc", "sender_started", "sender_stopped",
    "sender_alive_at_capture", "computed_I", "computed_Q", "window_status",
]


def write_smoke(root: Path) -> None:
    (root / "run.json").write_text(
        json.dumps({
            "session_id": "ENGINEERING_SMOKE_TEST",
            "executor_git_commit": "a" * 40,
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
        }),
        encoding="utf-8",
    )
    rows = [
        {
            "drive_on": "1", "sender_off_required": "0",
            "slot_start_tsc": "100000", "capture_deadline_tsc": "200000",
            "sender_ready_tsc": "90000", "sender_epoch_tsc": "100001",
            "first_drive_tsc": "100050", "receiver_epoch_tsc": "100002",
            "first_sample_tsc": "100003", "last_sample_tsc": "199000", "sender_started": "1",
            "sender_stopped": "1", "sender_alive_at_capture": "1",
            "computed_I": "1", "computed_Q": "2", "window_status": "OK",
        },
        {
            "drive_on": "1", "sender_off_required": "0",
            "slot_start_tsc": "300000", "capture_deadline_tsc": "400000",
            "sender_ready_tsc": "290000", "sender_epoch_tsc": "300001",
            "first_drive_tsc": "300050", "receiver_epoch_tsc": "300002",
            "first_sample_tsc": "300003", "last_sample_tsc": "399000", "sender_started": "1",
            "sender_stopped": "1", "sender_alive_at_capture": "1",
            "computed_I": "1", "computed_Q": "2", "window_status": "OK",
        },
        {
            "drive_on": "0", "sender_off_required": "1",
            "slot_start_tsc": "500000", "capture_deadline_tsc": "600000",
            "sender_ready_tsc": "0", "sender_epoch_tsc": "0",
            "first_drive_tsc": "0", "receiver_epoch_tsc": "500001",
            "first_sample_tsc": "500002", "last_sample_tsc": "599000", "sender_started": "0",
            "sender_stopped": "1", "sender_alive_at_capture": "0",
            "computed_I": "null", "computed_Q": "null", "window_status": "OK",
        },
    ]
    with (root / "window_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_validation_run(root: Path, index: int, hardware: bool = False) -> None:
    run = root / f"session_{index:02d}"
    run.mkdir()
    run_json = run / "run.json"
    run_json.write_text(
        json.dumps({
            "status": "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED",
            "hardware_executed": hardware,
        }) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "files": {
            "run.json": {
                "size": run_json.stat().st_size,
                "sha256": sha256_file(run_json),
            }
        }
    }
    (run / "run_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


class TargetEngineeringEvidenceTests(unittest.TestCase):
    def test_runner_process_recognition_handles_linux_comm_truncation(self) -> None:
        self.assertTrue(is_runner_process("combined_pdn_ru", ""))
        self.assertTrue(is_runner_process("python3", "/tmp/combined_pdn_runner"))
        self.assertFalse(is_runner_process("python3", "/tmp/other"))

    def test_validation_report_is_derived_from_twelve_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            for index in range(12):
                write_validation_run(runs, index)
            output = root / "validation_report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = validation_report_command(
                    argparse.Namespace(evidence_root=root, runs_root=runs, output=output)
                )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["all_pass"])
            self.assertEqual(report["sessions_passed"], 12)
            self.assertFalse(report["hardware_touched"])

    def test_validation_report_rejects_hardware_touch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            for index in range(12):
                write_validation_run(runs, index, hardware=index == 4)
            output = root / "validation_report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = validation_report_command(
                    argparse.Namespace(evidence_root=root, runs_root=runs, output=output)
                )
            self.assertEqual(rc, 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["all_pass"])
            self.assertEqual(report["sessions_passed"], 11)
            self.assertTrue(report["hardware_touched"])

    def test_three_window_smoke_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_smoke(root)
            _, _, checks = smoke_checks(root)
            self.assertTrue(all(checks.values()), checks)

    def test_first_drive_need_not_precede_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_smoke(root)
            path = root / "window_results.csv"
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["first_drive_tsc"] = "100100"
            rows[0]["first_sample_tsc"] = "100003"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            _, _, checks = smoke_checks(root)
            self.assertTrue(checks["driven_timing_and_lifecycle"])

    def test_sender_off_drive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_smoke(root)
            path = root / "window_results.csv"
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[2]["first_drive_tsc"] = "500010"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            _, _, checks = smoke_checks(root)
            self.assertFalse(checks["sender_off_is_true_off"])


if __name__ == "__main__":
    unittest.main()
