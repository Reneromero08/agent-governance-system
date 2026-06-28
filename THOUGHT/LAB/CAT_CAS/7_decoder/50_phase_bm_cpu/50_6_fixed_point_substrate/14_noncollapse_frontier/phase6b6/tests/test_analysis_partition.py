from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.adjudication import adjudicate, validate_thresholds  # noqa: E402
from analysis.operators import (  # noqa: E402
    OPERATOR_LADDER,
    choose_simplest_within_two_percent,
    deterministic_seed,
    validate_operator_manifest,
)
from analysis.state import (  # noqa: E402
    assert_training_only_global_covariance,
    estimate_preamble_gauge,
    s0,
    s2_delayed,
    validate_measured_state_fields,
)
from contracts.contract import O4_FIXED_LIFTS, REGULARIZATION_LADDER  # noqa: E402


def row(i: int, *, stage: str = "preamble", split: str = "train") -> dict[str, object]:
    return {
        "stage": stage,
        "split": split,
        "r_t": {"lockin_I": float(i), "lockin_Q": float(i) / 10.0, "ring_osc_period": 100.0 + i},
        "u_t": {"drive_on": i % 2 == 0, "executed_mode": "MOCK", "physical_tone_index": i % 12},
    }


class AnalysisPartitionTests(unittest.TestCase):
    def test_measured_state_rejects_context_declared_and_future_leakage(self) -> None:
        for field in ("session_id", "route", "declared_order_family", "future_value", "target_label"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_measured_state_fields(["lockin_I", field])

    def test_s0_accepts_only_measured_response(self) -> None:
        self.assertEqual(s0(row(1)), (complex(1.0, 0.1), 101.0))

    def test_gauge_is_preamble_only(self) -> None:
        rows = [row(i) for i in range(4)]
        gauge = estimate_preamble_gauge(rows)
        self.assertEqual(gauge.amplitude_floor, 0.0)
        with self.assertRaises(ValueError):
            estimate_preamble_gauge(rows + [row(5, stage="trajectory")])

    def test_global_covariance_is_training_preambles_only(self) -> None:
        assert_training_only_global_covariance([row(i) for i in range(3)])
        with self.assertRaises(ValueError):
            assert_training_only_global_covariance([row(1), row(2, split="validation")])

    def test_s2_delay_uses_prior_executed_controls_only(self) -> None:
        rows = [row(i, stage="preamble") for i in range(20)]
        gauge = estimate_preamble_gauge(rows[:4])
        state = s2_delayed(rows, 10, 4, gauge)
        self.assertEqual(len(state["S1_history"]), 4)
        self.assertEqual(len(state["prior_executed_controls"]), 3)
        with self.assertRaises(ValueError):
            s2_delayed(rows, 10, 3, gauge)

    def test_operator_manifest_blocks_forbidden_terms(self) -> None:
        manifest = {
            "operator_ladder": OPERATOR_LADDER,
            "regularization_ladder": REGULARIZATION_LADDER,
            "o4_fixed_lifts": O4_FIXED_LIFTS,
        }
        validate_operator_manifest(manifest)
        with self.assertRaises(ValueError):
            validate_operator_manifest({**manifest, "model": "neural_backprop"})

    def test_deterministic_seed_and_simplest_selection(self) -> None:
        self.assertEqual(deterministic_seed("a" * 64, "bootstrap"), deterministic_seed("a" * 64, "bootstrap"))
        self.assertEqual(choose_simplest_within_two_percent([("O1", 1.009), ("O2", 1.0), ("O3", 0.99)]), "O1")

    def test_thresholds_and_verdicts(self) -> None:
        metrics = {
            "one_step_nrmse_gain": 0.11,
            "eight_step_nrmse_gain": 0.06,
            "one_step_bootstrap_lower": 0.01,
            "eight_step_bootstrap_lower": 0.01,
            "route_v4s5_complex_corr": 0.81,
            "route_v2s3_complex_corr": 0.82,
            "worst_session_delta_vs_baseline": -0.01,
            "session_lookup_gain_margin": 0.06,
        }
        self.assertTrue(validate_thresholds(metrics))
        self.assertEqual(
            adjudicate(
                shared_predictive_pass=True,
                drive_off_persistence_pass=False,
                within_route_pass=True,
                bidirectional_transfer_pass=True,
                confounded=False,
            ),
            ("SHARED_PREDICTIVE_OPERATOR_SUPPORTED", "DRIVEN_RELATIONAL_TRANSPORT_ONLY"),
        )
        self.assertEqual(
            adjudicate(
                shared_predictive_pass=False,
                drive_off_persistence_pass=False,
                within_route_pass=True,
                bidirectional_transfer_pass=False,
                confounded=False,
            ),
            ("ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY",),
        )
        self.assertEqual(
            adjudicate(
                shared_predictive_pass=False,
                drive_off_persistence_pass=False,
                within_route_pass=False,
                bidirectional_transfer_pass=False,
                confounded=True,
            ),
            ("CONFOUNDED_NO_OPERATOR_CLAIM",),
        )


if __name__ == "__main__":
    unittest.main()
