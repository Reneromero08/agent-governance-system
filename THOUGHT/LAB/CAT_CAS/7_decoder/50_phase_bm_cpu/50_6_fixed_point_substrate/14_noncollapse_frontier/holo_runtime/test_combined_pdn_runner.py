#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "combined_pdn_runner"
VALID_COMMIT = "a" * 40


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def base_window(index: int, off: bool = False) -> dict:
    return {
        "window_index": index,
        "session_id": "v4s5_seed4",
        "stage": "C_PERSISTENCE_OFF" if off else "B_TONE_ORDER",
        "block_id": "block",
        "family": "silent" if off else "real",
        "actual_mode": "basis",
        "declared_mode": "basis",
        "executed_tone_order": "FWD",
        "declared_tone_order": "FWD",
        "measurement_mode": "raw_ring_sender_off" if off else "lockin_and_raw_ring",
        "drive_on": not off,
        "sender_off_required": off,
        "physical_tone_index": None if off else index,
        "codeword_source_index": None if off else index,
        "theta_idx": None if off else 0,
        "amplitude_level": 0 if off else 3,
    }


def write_session(root: Path, mutate=None, count: int = 3,
                  manifest_schema: str = "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1") -> Path:
    directory = root / "session"
    directory.mkdir()
    rows = [base_window(0), base_window(1), base_window(2, True)][:count]
    header = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1",
        "campaign_source_commit": "f5b6079a5748bb6138ab19d1c22d79c74734dddf",
        "campaign_plan_sha256": "e" * 64,
        "session_id": "v4s5_seed4",
        "route": "v4s5",
        "seed": 4,
        "partition": "stress",
        "window_count": len(rows),
        "restoration_authorized": False,
    }
    if mutate:
        mutate(header, rows)
    dump(directory / "session.json", header)
    (directory / "windows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_id": manifest_schema,
        "session_id": "v4s5_seed4",
        "files": {
            name: {"size": (directory / name).stat().st_size,
                   "sha256": sha(directory / name)}
            for name in ("session.json", "windows.jsonl")
        },
    }
    dump(directory / "session_manifest.json", manifest)
    return directory


def write_engineering_smoke(root: Path) -> Path:
    directory = root / "engineering_smoke"
    result = subprocess.run(
        [sys.executable, str(HERE / "make_engineering_smoke_schedule.py"), str(directory)],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return directory


class Tests(unittest.TestCase):
    def exec_runner(self, session: Path, output: Path, *args: str,
                    fail: str | None = None, commit: str = VALID_COMMIT):
        env = os.environ.copy()
        if fail:
            env["COMBINED_PDN_MOCK_FAIL"] = fail
        return subprocess.run(
            [str(RUNNER), "--session-dir", str(session), "--output-dir", str(output),
             "--victim", "4", "--sender", "5", "--executor-commit", commit, *args],
            text=True, capture_output=True, env=env,
        )

    def assert_reject(self, mutate, text, manifest_schema=None):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(
                Path(temp), mutate=mutate,
                manifest_schema=manifest_schema or "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1",
            )
            result = self.exec_runner(directory, Path(temp) / "out", "--validate-only")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(text, result.stderr)

    def test_validate_contract_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            output = Path(temp) / "out"
            result = self.exec_runner(directory, output, "--validate-only")
            self.assertEqual(result.returncode, 0, result.stderr)
            names = ("run.json", "session.json", "windows.jsonl", "window_results.csv",
                     "raw_samples.bin", "telemetry.csv", "stdout.log", "stderr.log",
                     "run_manifest.json")
            self.assertTrue(all((output / name).is_file() for name in names))
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertNotIn("run_manifest.json", manifest["files"])
            self.assertTrue(all(sha(output / name) == binding["sha256"]
                                for name, binding in manifest["files"].items()))

    def test_output_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            output = Path(temp) / "out"
            output.mkdir()
            self.assertIn("refusing existing output",
                          self.exec_runner(directory, output, "--validate-only").stderr)

    def test_manifest_schema(self):
        self.assert_reject(None, "unexpected session manifest schema", "BAD")

    def test_manifest_size(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            (directory / "windows.jsonl").write_text(
                (directory / "windows.jsonl").read_text() + " ")
            self.assertIn("size mismatch", self.exec_runner(
                directory, Path(temp) / "out", "--validate-only").stderr)

    def test_manifest_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            manifest = json.loads((directory / "session_manifest.json").read_text())
            manifest["files"]["windows.jsonl"]["sha256"] = "0" * 64
            dump(directory / "session_manifest.json", manifest)
            self.assertIn("sha256 mismatch", self.exec_runner(
                directory, Path(temp) / "out", "--validate-only").stderr)

    def test_noncontiguous_duplicate(self):
        self.assert_reject(lambda header, rows: rows[1].update(window_index=0),
                           "not contiguous")

    def test_session_id_mismatch(self):
        self.assert_reject(lambda header, rows: rows[0].update(session_id="bad"),
                           "session ID mismatch")

    def test_unsupported_mode(self):
        self.assert_reject(lambda header, rows: rows[0].update(measurement_mode="bad"),
                           "unsupported measurement")

    def test_unsupported_route(self):
        self.assert_reject(lambda header, rows: header.update(route="bad_route"),
                           "unsupported route")

    def test_sender_off_drive(self):
        self.assert_reject(lambda header, rows: rows[2].update(drive_on=True),
                           "sender_off_required + drive_on")

    def test_raw_ring_requires_off(self):
        self.assert_reject(lambda header, rows: rows[2].update(sender_off_required=False),
                           "raw_ring_sender_off requires")

    def test_driven_requires_tone(self):
        self.assert_reject(lambda header, rows: rows[0].update(physical_tone_index=None),
                           "missing physical tone")

    def test_driven_requires_codeword(self):
        self.assert_reject(lambda header, rows: rows[0].update(codeword_source_index=None),
                           "codeword source")

    def test_extra_rows(self):
        self.assert_reject(lambda header, rows: header.update(window_count=2),
                           "extra schedule rows")

    def test_short_count(self):
        self.assert_reject(lambda header, rows: header.update(window_count=4),
                           "short schedule row count")

    def test_unsafe_path(self):
        result = subprocess.run(
            [str(RUNNER), "--session-dir", "../x", "--output-dir", "/tmp/o",
             "--victim", "4", "--sender", "5", "--validate-only"],
            text=True, capture_output=True,
        )
        self.assertIn("unsafe path", result.stderr)

    def test_validation_never_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            result = self.exec_runner(directory, Path(temp) / "out",
                                      "--validate-only", fail="thermal")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_numeric_arguments_are_rejected(self):
        for option, value in (("--read-hz", "0"), ("--slot-s", "nan"),
                              ("--off-window-s", "0"), ("--pin-khz", "-1")):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temp:
                directory = write_session(Path(temp))
                result = self.exec_runner(
                    directory, Path(temp) / "out", "--mock-hardware", option, value
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid numeric arguments", result.stderr)

    def test_capture_capacity_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            output = Path(temp) / "out"
            result = self.exec_runner(
                directory, output, "--mock-hardware", "--read-hz", "1000000000"
            )
            self.assertEqual(result.returncode, 5)
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["failure_reason"], "CAPTURE_CAPACITY_INVALID")

    def test_real_hardware_requires_authorization_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            result = self.exec_runner(directory, Path(temp) / "out", "--hardware")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --authorization-artifact", result.stderr)

    def test_invalid_authorization_artifact_fails_before_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = write_session(root)
            authorization = root / "authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")
            result = self.exec_runner(
                directory,
                root / "out",
                "--hardware",
                "--authorization-artifact",
                str(authorization),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid acquisition authorization", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_valid_authorization_is_bound_and_recorded_before_real_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = write_session(root)
            authorized_root = root / "authorized"
            authorized_root.mkdir()
            output = authorized_root / "run"
            authorization = root / "authorization.json"
            dump(authorization, {
                "schema_id": "CAT_CAS_PHASE6_ACQUISITION_AUTHORIZATION_V1",
                "acquisition_authorized": True,
                "restoration_authorized": False,
                "executor_commit": VALID_COMMIT,
                "campaign_plan_sha256": "e" * 64,
                "source_bundle_sha256": "b" * 64,
                "authorized_output_root": str(authorized_root),
                "authorized_by": "PROJECT_OWNER_TEST",
            })
            result = self.exec_runner(
                directory,
                output,
                "--hardware",
                "--authorization-artifact",
                str(authorization),
                fail="thermal",
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["execution_class"], "AUTHORIZED_SCIENTIFIC_ACQUISITION")
            self.assertTrue(run["scientific_acquisition_authorized"])
            self.assertEqual(run["authorization_artifact_sha256"], sha(authorization))
            self.assertFalse(run["hardware_executed"])

    def test_engineering_smoke_gate_rejects_scientific_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            result = self.exec_runner(
                directory, Path(temp) / "out", "--engineering-smoke", "--mock-hardware"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("engineering smoke schedule mismatch", result.stderr)

    def test_exact_engineering_smoke_is_the_only_unauthorized_hardware_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = write_engineering_smoke(root)
            output = root / "out"
            result = self.exec_runner(
                directory, output, "--engineering-smoke", "--mock-hardware"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                run["execution_class"],
                "ENGINEERING_SMOKE_NOT_SCIENTIFIC_ACQUISITION",
            )
            self.assertFalse(run["scientific_acquisition_authorized"])
            self.assertIsNone(run["authorization_artifact_sha256"])

    def test_hardware_commit_validation(self):
        for commit in ("0" * 40, "A" * 40, "g" * 40, "abc"):
            with self.subTest(commit=commit), tempfile.TemporaryDirectory() as temp:
                directory = write_session(Path(temp))
                result = self.exec_runner(directory, Path(temp) / "out",
                                          "--mock-hardware", commit=commit)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("nonzero lowercase", result.stderr)

    def test_mock_sender_lifecycle_sender_off_and_shared_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            output = Path(temp) / "out"
            result = self.exec_runner(directory, output, "--mock-hardware")
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "window_results.csv").open() as file:
                rows = list(csv.DictReader(file))
            for row in rows[:2]:
                origin = int(row["slot_start_tsc"])
                self.assertLess(int(row["sender_ready_tsc"]), origin)
                self.assertGreaterEqual(int(row["sender_epoch_tsc"]), origin)
                self.assertGreaterEqual(int(row["receiver_epoch_tsc"]), origin)
                self.assertGreaterEqual(int(row["first_drive_tsc"]), origin)
                self.assertEqual(row["sender_started"], "1")
                self.assertEqual(row["sender_stopped"], "1")
            self.assertEqual(rows[2]["sender_alive_at_capture"], "0")
            self.assertEqual(rows[2]["first_drive_tsc"], "0")
            self.assertEqual(rows[2]["computed_I"], "null")

    def test_late_sender_epoch_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            output = Path(temp) / "out"
            result = self.exec_runner(directory, output, "--mock-hardware",
                                      fail="late_sender")
            self.assertEqual(result.returncode, 5)
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["failure_reason"], "SENDER_EPOCH_ALIGNMENT_FAILURE")
            self.assertFalse(run["hardware_executed"])

    def test_failure_cleanup_matrix(self):
        for failure in ("thermal", "cpufreq", "sender_create", "sender_stop",
                        "capture", "raw"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                directory = write_session(Path(temp))
                output = Path(temp) / "out"
                result = self.exec_runner(directory, output, "--mock-hardware",
                                          fail=failure)
                self.assertNotEqual(result.returncode, 0)
                run = json.loads((output / "run.json").read_text())
                self.assertTrue(run["host_control_state_restored"])
                self.assertFalse(run["physical_carrier_restoration_claimed"])
                self.assertEqual(run["exit_status"], "FAILED")
                self.assertTrue((output / "run_manifest.json").is_file())

    def test_restoration_failure_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            output = Path(temp) / "out"
            result = self.exec_runner(directory, output, "--mock-hardware",
                                      fail="restore")
            self.assertEqual(result.returncode, 6)
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["failure_reason"], "RESTORATION_FAILURE")
            self.assertFalse(run["host_control_state_restored"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
