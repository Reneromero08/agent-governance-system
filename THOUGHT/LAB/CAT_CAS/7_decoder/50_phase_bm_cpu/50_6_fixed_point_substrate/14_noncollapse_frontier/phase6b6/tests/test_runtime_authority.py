from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.schedule import campaign_schedule  # noqa: E402
from runtime.explicit_slot_runtime import run_mock  # noqa: E402


class RuntimeAuthorityTests(unittest.TestCase):
    def test_validate_only_cli(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "runtime.explicit_slot_runtime", "--validate-only"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("PHASE6B6_VALIDATE_ONLY_OK", result.stdout)
        self.assertIn("slots=10368", result.stdout)

    def test_real_hardware_fails_closed(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "runtime.explicit_slot_runtime", "--hardware"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SOFTWARE_ENTRY_ONLY_AUTHORITY", result.stderr)

    def test_no_mode_fails_closed(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "runtime.explicit_slot_runtime"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SOFTWARE_ENTRY_ONLY_AUTHORITY", result.stderr)

    def test_mock_runtime_custody_is_deterministic(self) -> None:
        schedule = campaign_schedule()
        first = run_mock(schedule)
        second = run_mock(schedule)
        self.assertEqual(first["custody_sha256"], second["custody_sha256"])
        self.assertEqual(first["total_slots"], 10368)
        self.assertFalse(first["authority"]["hardware_ran"])

    def test_mock_runtime_writes_schedule_and_custody(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            schedule_path = Path(temp) / "schedule.json"
            custody_path = Path(temp) / "custody.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "runtime.explicit_slot_runtime",
                    "--mock-hardware",
                    "--schedule-out",
                    str(schedule_path),
                    "--custody-out",
                    str(custody_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("PHASE6B6_MOCK_RUNTIME_OK", result.stdout)
            self.assertEqual(json.loads(schedule_path.read_text(encoding="utf-8"))["total_slots"], 10368)
            self.assertEqual(json.loads(custody_path.read_text(encoding="utf-8"))["total_slots"], 10368)


if __name__ == "__main__":
    unittest.main()
