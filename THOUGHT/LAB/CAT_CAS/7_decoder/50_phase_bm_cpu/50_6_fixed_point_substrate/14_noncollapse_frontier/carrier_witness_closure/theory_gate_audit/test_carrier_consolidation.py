#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
