#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER_C = HERE / "combined_pdn_runner.c"
RUNNER = HERE / "combined_pdn_runner"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_session(root: Path, *, invalid_off_drive: bool = False) -> Path:
    session = root / "session"
    session.mkdir()
    header = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1",
        "session_id": "v4s5_seed4",
        "route": "v4s5",
        "seed": 4,
        "partition": "stress",
        "window_count": 3,
        "restoration_authorized": False,
    }
    windows = [
        {"window_index": 0, "session_id": "v4s5_seed4", "measurement_mode": "lockin_and_raw_ring", "drive_on": True, "sender_off_required": False},
        {"window_index": 1, "session_id": "v4s5_seed4", "measurement_mode": "lockin_and_raw_ring", "drive_on": True, "sender_off_required": False},
        {"window_index": 2, "session_id": "v4s5_seed4", "measurement_mode": "raw_ring_sender_off", "drive_on": invalid_off_drive, "sender_off_required": True},
    ]
    write_json(session / "session.json", header)
    with (session / "windows.jsonl").open("w", encoding="utf-8") as handle:
        for window in windows:
            handle.write(json.dumps(window, sort_keys=True) + "\n")
    manifest = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1",
        "session_id": "v4s5_seed4",
        "files": {
            "session.json": {"size": (session / "session.json").stat().st_size, "sha256": sha256_file(session / "session.json")},
            "windows.jsonl": {"size": (session / "windows.jsonl").stat().st_size, "sha256": sha256_file(session / "windows.jsonl")},
        },
    }
    write_json(session / "session_manifest.json", manifest)
    return session


class CombinedPdnRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", str(RUNNER_C), "-o", str(RUNNER)], check=True)

    def run_runner(self, session: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(RUNNER), "--session-dir", str(session), "--output-dir", str(output), "--victim", "4", "--sender", "5", *extra],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_validate_only_writes_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = write_session(root)
            output = root / "run"
            proc = self.run_runner(session, output, "--validate-only")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for name in ("run.json", "session.json", "windows.jsonl", "window_results.csv", "raw_samples.bin", "telemetry.csv", "stdout.log", "stderr.log", "run_manifest.json"):
                self.assertTrue((output / name).is_file(), name)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertFalse(run["hardware_executed"])
            self.assertEqual(run["sender_off_windows"], 1)

    def test_refuses_output_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = write_session(root)
            output = root / "run"
            output.mkdir()
            proc = self.run_runner(session, output, "--validate-only")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing existing output", proc.stderr)

    def test_detects_manifest_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = write_session(root)
            (session / "windows.jsonl").write_text((session / "windows.jsonl").read_text(encoding="utf-8") + " ", encoding="utf-8")
            proc = self.run_runner(session, root / "run", "--validate-only")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("size mismatch", proc.stderr)

    def test_rejects_sender_off_drive_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = write_session(root, invalid_off_drive=True)
            proc = self.run_runner(session, root / "run", "--validate-only")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("sender-off window has drive_on=true", proc.stderr)

    def test_refuses_hardware_mode_until_backend_is_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = write_session(root)
            proc = self.run_runner(session, root / "run")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("hardware execution path is not implemented", proc.stderr)


if __name__ == "__main__":
    unittest.main()
