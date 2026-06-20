#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import analyze_carrier_consolidation as consolidation

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results" / "phase6b5c_t48_d32b1bed_20260619"


class CarrierConsolidationTests(unittest.TestCase):
    def test_committed_packet_consolidates_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase6b5d"
            manifest = consolidation.build(RESULT_DIR, output)
            decision = manifest["decision"]

            self.assertFalse(decision["old_gate_scalar_calibration_can_change_result"])
            self.assertTrue(decision["cross_session_relational_generalization"])
            self.assertTrue(decision["gate_r_ready"])
            self.assertEqual(decision["carrier_claim_status"], "FROZEN_PENDING_GATE_R")
            self.assertIsNotNone(decision["seed4_localization"])

            old_gate = consolidation.load_json(output / "old_gate_chart_invariance.json")
            self.assertTrue(old_gate["all_selected_charts_scalar_identity"])
            self.assertTrue(old_gate["analytical_invariance"]["proven"])
            self.assertEqual(len(old_gate["runs"]), 12)

            residuals = consolidation.load_json(output / "residual_structure.json")
            self.assertEqual(residuals["record_count"], 576)
            self.assertIn("seed", residuals["eta_squared_by_factor"])
            self.assertIn("route", residuals["eta_squared_by_factor"])

            verification = consolidation.verify(output)
            self.assertTrue(verification["valid"], verification["errors"])

    def test_manifest_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase6b5d"
            consolidation.build(RESULT_DIR, output)
            target = output / "residual_structure.json"
            target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
            verification = consolidation.verify(output)
            self.assertFalse(verification["valid"])
            self.assertTrue(any("residual_structure.json" in item for item in verification["errors"]))

    def test_tampered_input_fails_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy_dir = Path(tmp) / "analysis"
            shutil.copytree(str(RESULT_DIR), str(copy_dir))
            output = Path(tmp) / "phase6b5d"
            chart_path = copy_dir / "chart_calibration.json"
            chart_path.write_text(chart_path.read_text() + " ", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                consolidation.build(copy_dir, output)
            self.assertIn("mismatch", str(ctx.exception))

    def test_missing_manifest_entry_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy_dir = Path(tmp) / "analysis"
            shutil.copytree(str(RESULT_DIR), str(copy_dir))
            output = Path(tmp) / "phase6b5d"
            manifest_path = copy_dir / "analysis_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["outputs"]["chart_calibration.json"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                consolidation.build(copy_dir, output)
            self.assertIn("missing required output entries", str(ctx.exception))

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy_dir = Path(tmp) / "analysis"
            shutil.copytree(str(RESULT_DIR), str(copy_dir))
            output = Path(tmp) / "phase6b5d"
            (copy_dir / "route_conjugacy.json").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                consolidation.build(copy_dir, output)
            self.assertIn("route_conjugacy.json", str(ctx.exception))

    def test_wrong_schema_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy_dir = Path(tmp) / "analysis"
            shutil.copytree(str(RESULT_DIR), str(copy_dir))
            output = Path(tmp) / "phase6b5d"
            manifest_path = copy_dir / "analysis_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_id"] = "WRONG_SCHEMA"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                consolidation.build(copy_dir, output)
            self.assertIn("unexpected manifest schema_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
