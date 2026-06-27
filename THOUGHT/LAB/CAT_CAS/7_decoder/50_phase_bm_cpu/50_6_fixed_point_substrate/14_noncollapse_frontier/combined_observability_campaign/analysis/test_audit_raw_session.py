from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from audit_raw_session import audit


class ValidationOnlyClassificationTests(unittest.TestCase):
    def test_each_validation_only_marker_rejects_before_physical_fields(self) -> None:
        cases = {
            "schema_id": {
                "schema_id": "CAT_CAS_PHASE6_V2_VALIDATION_ONLY_RUN_V1",
                "execution_class": "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION",
                "hardware_executed": True,
            },
            "execution_class": {
                "schema_id": "CAT_CAS_PHASE6_COMBINED_RUN_V2",
                "execution_class": "VALIDATION_ONLY",
                "hardware_executed": True,
            },
            "hardware_executed": {
                "schema_id": "CAT_CAS_PHASE6_COMBINED_RUN_V2",
                "execution_class": "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION",
                "hardware_executed": False,
            },
        }
        for marker, run in cases.items():
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                run_dir = root / "run"
                session_dir = root / "session"
                run_dir.mkdir()
                session_dir.mkdir()
                (run_dir / "run.json").write_text(
                    json.dumps(run) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "validation-only artifact is not physical raw-session evidence",
                ):
                    audit(
                        argparse.Namespace(
                            run_dir=run_dir,
                            session_dir=session_dir,
                            raw=root / "raw_samples.bin",
                        )
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
