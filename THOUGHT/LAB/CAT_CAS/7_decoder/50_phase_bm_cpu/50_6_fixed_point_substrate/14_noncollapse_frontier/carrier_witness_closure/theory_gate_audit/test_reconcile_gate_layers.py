#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location(
    "gate_audit", HERE / "reconcile_gate_layers.py"
)
assert SPEC is not None and SPEC.loader is not None
gate_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate_audit)

CLOSURE = ROOT / "evidence" / "phase6b5_t48_d32b1bed_20260619" / "aggregate" / "closure_report.json"
DECOMP = ROOT / "replication_discrepancy" / "results" / "official_gate_decomposition.json"


class GateLayerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = gate_audit.build_audit(
            gate_audit.load_json(CLOSURE),
            gate_audit.load_json(DECOMP),
        )

    def test_two_gate_namespaces_are_reported(self) -> None:
        namespace = self.audit["gate_namespace"]
        self.assertEqual(namespace["contract_gate_count"], 7)
        self.assertEqual(namespace["analyzer_gate_count"], 9)
        self.assertEqual(
            namespace["analyzer_additional_keys"],
            ["pseudo_declared_match_le_0_35", "real_mode_floor_ge_0_45"],
        )

    def test_route_layer_counts(self) -> None:
        routes = self.audit["route_summary"]
        self.assertEqual(routes["v4s5"]["counts"]["contract_seven_gate_pass"], 1)
        self.assertEqual(routes["v4s5"]["counts"]["analyzer_nine_gate_pass"], 0)
        self.assertEqual(routes["v4s5"]["counts"]["core_carrier_transport"], 5)
        self.assertEqual(routes["v4s5"]["counts"]["phase_transport"], 6)
        self.assertEqual(routes["v2s3"]["counts"]["contract_seven_gate_pass"], 2)
        self.assertEqual(routes["v2s3"]["counts"]["analyzer_nine_gate_pass"], 2)
        self.assertEqual(routes["v2s3"]["counts"]["core_carrier_transport"], 6)
        self.assertEqual(routes["v2s3"]["counts"]["phase_transport"], 6)

    def test_seed_three_namespace_split(self) -> None:
        run = next(
            item for item in self.audit["run_reconciliation"]
            if item["run_id"] == "v4s5_matrix_seed3"
        )
        self.assertTrue(run["contract_seven_gate_pass"])
        self.assertFalse(run["analyzer_nine_gate_pass"])
        self.assertTrue(run["reported_scientific_pass"])
        self.assertEqual(run["reported_verdict"], "PHASE4B_PDN_PARTIAL")

    def test_small_groups_require_zero_errors_at_point_ninety_five(self) -> None:
        self.assertEqual(gate_audit.allowed_errors(10), 0)
        self.assertEqual(gate_audit.allowed_errors(19), 0)
        self.assertEqual(gate_audit.allowed_errors(20), 1)
        seed_zero = self.audit["finite_sample_geometry"]["runs"]["v4s5_matrix_seed0"]
        mini = seed_zero["groups"]["mini"]
        self.assertEqual(mini["combined_denominator"], 10)
        self.assertTrue(mini["zero_error_required"])
        self.assertEqual(mini["false_accepts"], 0)
        self.assertEqual(mini["real_false_rejects"], 3)


if __name__ == "__main__":
    unittest.main()
