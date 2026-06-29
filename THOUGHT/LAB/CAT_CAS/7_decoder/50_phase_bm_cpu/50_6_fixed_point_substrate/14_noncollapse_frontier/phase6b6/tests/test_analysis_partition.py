from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.adjudication import derive_adjudication, validate_thresholds  # noqa: E402
from analysis.observations import flatten_custody  # noqa: E402
from analysis.operators import (  # noqa: E402
    OPERATOR_LADDER,
    _design_o4,
    choose_simplest_within_two_percent,
    deterministic_seed,
    validate_operator_manifest,
)
from analysis.pipeline import evaluate_sealed, select_on_validation, training_validation_custody  # noqa: E402
from analysis.state import (  # noqa: E402
    assert_training_only_global_covariance,
    estimate_preamble_gauge,
    s0,
    s2_delayed,
    symmetric_inverse_sqrt,
    validate_measured_state_fields,
)
from analysis.synthetic import synthetic_custody  # noqa: E402
from contracts.contract import O4_FIXED_LIFTS, REGULARIZATION_LADDER, digest  # noqa: E402
from contracts.schedule import campaign_schedule  # noqa: E402
from runtime.explicit_slot_runtime import run_mock  # noqa: E402


def row(i: int, *, stage: str = "preamble", split: str = "train") -> dict[str, object]:
    return {
        "stage": stage,
        "split": split,
        "session_index": 0,
        "declared": {"analysis_tone_index": i % 12},
        "r_t": {"lockin_I": float(i), "lockin_Q": float(i) / 10.0, "ring_osc_period": 100.0 + i},
        "u_t": {
            "drive_on": i % 2 == 0,
            "executed_mode": "ANCHOR" if stage == "preamble" else "MOCK",
            "physical_tone_index": i % 12,
            "codeword_sign": 1,
        },
    }


class AnalysisPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.custody = run_mock(campaign_schedule())
        cls.rows = flatten_custody(cls.custody)
        cls.train_preamble = [row for row in cls.rows if row["split"] == "train" and row["stage"] == "preamble"]

    def test_measured_state_rejects_context_declared_and_future_leakage(self) -> None:
        for field in ("session_id", "route", "declared_order_family", "future_value", "target_label", "session_chronology"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_measured_state_fields(["lockin_I", field])

    def test_s0_accepts_only_measured_response(self) -> None:
        self.assertEqual(s0(row(1)), (complex(1.0, 0.1), 101.0))

    def test_gauge_is_preamble_only_and_per_tone(self) -> None:
        gauge = estimate_preamble_gauge([row for row in self.rows if row["session_index"] == 0 and row["stage"] == "preamble"])
        self.assertEqual(len(gauge.complex_anchor_alpha), 12)
        self.assertEqual(len(gauge.amplitude_floor), 12)
        with self.assertRaises(ValueError):
            estimate_preamble_gauge([row for row in self.rows if row["session_index"] == 0 and row["stage"] in ("preamble", "trajectory")])

    def test_global_covariance_is_training_preambles_only(self) -> None:
        assert_training_only_global_covariance(self.train_preamble)
        with self.assertRaises(ValueError):
            assert_training_only_global_covariance(self.train_preamble + [next(row for row in self.rows if row["split"] == "validation")])

    def test_s2_delay_uses_prior_executed_controls_only(self) -> None:
        session_rows = [row for row in self.rows if row["session_index"] == 0 and row["stage"] == "preamble"]
        gauge = estimate_preamble_gauge(session_rows)
        state = s2_delayed(session_rows, 20, 4, gauge)
        self.assertEqual(len(state["S1_history"]), 4)
        self.assertEqual(len(state["prior_executed_controls"]), 3)
        self.assertNotIn("declared", state["prior_executed_controls"][0])
        with self.assertRaises(ValueError):
            s2_delayed(session_rows, 10, 3, gauge)

    def test_operator_manifest_blocks_forbidden_terms(self) -> None:
        manifest = {
            "operator_ladder": OPERATOR_LADDER,
            "regularization_ladder": REGULARIZATION_LADDER,
            "o4_fixed_lifts": O4_FIXED_LIFTS,
        }
        validate_operator_manifest(manifest)
        with self.assertRaises(ValueError):
            validate_operator_manifest({**manifest, "model": "neural_backprop"})

    def test_o4_phase_feature_uses_executed_phase(self) -> None:
        x = __import__("numpy").array([[1.0, 0.0, 100.0]] * 4)
        phases = ("0", "pi", "pi/2", "-pi/2")
        rows = [{"u_t": {"drive_on": True, "phase_action": phase, "physical_tone_index": 0, "codeword_sign": 1}} for phase in phases]
        features = _design_o4(x, rows)
        self.assertAlmostEqual(features[0, 8], 1.0, places=12)
        self.assertAlmostEqual(features[0, 9], 0.0, places=12)
        self.assertAlmostEqual(features[1, 8], -1.0, places=12)
        self.assertAlmostEqual(features[1, 9], 0.0, places=12)
        self.assertAlmostEqual(features[2, 8], 0.0, places=12)
        self.assertAlmostEqual(features[2, 9], 1.0, places=12)
        self.assertAlmostEqual(features[3, 8], 0.0, places=12)
        self.assertAlmostEqual(features[3, 9], -1.0, places=12)

    def test_deterministic_seed_and_simplest_selection(self) -> None:
        self.assertEqual(deterministic_seed("a" * 64, "bootstrap"), deterministic_seed("a" * 64, "bootstrap"))
        self.assertEqual(choose_simplest_within_two_percent([("O1", 1.009), ("O2", 1.0), ("O3", 0.99)]), "O1")

    def test_thresholds_and_result_derived_verdict(self) -> None:
        computed = {
            "predictive_metrics": {
                "one_step_nrmse_gain": 0.11,
                "eight_step_nrmse_gain": 0.06,
                "one_step_bootstrap_lower": 0.01,
                "eight_step_bootstrap_lower": 0.01,
                "route_v4s5_complex_corr": 0.81,
                "route_v2s3_complex_corr": 0.82,
                "worst_session_delta_vs_baseline": -0.01,
                "session_lookup_gain_margin": 0.06,
            },
            "route_transfer": {"v4s5_to_v2s3": {"lower_gain": 0.01}, "v2s3_to_v4s5": {"lower_gain": 0.01}},
            "drive_off": {"three_consecutive_lower_above_sham": False, "zero_input_decay_gain": 0.0, "zero_input_decay_gain_lower": 0.0},
            "confounds": {"session_lookup_dominance": False},
            "within_route_pass": True,
        }
        self.assertTrue(validate_thresholds(computed["predictive_metrics"]))
        self.assertEqual(
            derive_adjudication(computed)["verdicts"],
            ["SHARED_PREDICTIVE_OPERATOR_SUPPORTED", "DRIVEN_RELATIONAL_TRANSPORT_ONLY"],
        )

    def test_premature_test_opening_fails(self) -> None:
        with self.assertRaises(PermissionError):
            evaluate_sealed(self.custody, {"schema_id": "not_sealed"})

    def test_test_rows_cannot_enter_selection(self) -> None:
        with self.assertRaises(PermissionError):
            select_on_validation(self.custody)

    def test_mutated_or_wrong_schedule_manifest_fails(self) -> None:
        custody = synthetic_custody("shared_driven")
        manifest = select_on_validation(training_validation_custody(custody))
        mutated = dict(manifest)
        mutated["regularization"] = 1.0
        with self.assertRaises(PermissionError):
            evaluate_sealed(custody, mutated)
        wrong_schedule = dict(manifest)
        wrong_schedule["schedule_sha256"] = "0" * 64
        wrong_schedule["analysis_choice_sha256"] = __import__("contracts.contract", fromlist=["digest"]).digest(
            {key: value for key, value in wrong_schedule.items() if key != "analysis_choice_sha256"}
        )
        with self.assertRaises(PermissionError):
            evaluate_sealed(custody, wrong_schedule)

    def test_fixture_label_does_not_affect_results(self) -> None:
        custody = synthetic_custody("shared_driven")
        manifest = select_on_validation(training_validation_custody(custody))
        result = evaluate_sealed(custody, manifest)
        custody.pop("synthetic_scenario", None)
        custody["synthetic_scenario"] = "renamed_for_reporting_only"
        renamed = evaluate_sealed(custody, manifest)
        for payload in (result, renamed):
            payload.pop("result_sha256", None)
        self.assertEqual(result, renamed)

    def test_analysis_source_has_no_scenario_verdict_override(self) -> None:
        source = (ROOT / "analysis" / "pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("_apply_synthetic_fixture_expectation", source)
        self.assertNotIn("synthetic_scenario", source)
        self.assertNotIn("expected verdict", source.lower())
        self.assertNotIn("max(eight", source)
        self.assertNotIn("min(one_step", source)
        self.assertNotIn("source_mean * target_mean", source)

    def test_full_covariance_whitening_is_not_scalar_trace_scaling(self) -> None:
        cov = ((4.0, 1.5), (1.5, 1.0))
        inv = symmetric_inverse_sqrt(cov)
        whitened = inv @ np.array(cov) @ inv.T
        scalar = np.eye(2) / np.sqrt(np.trace(np.array(cov)) / 2.0)
        scalar_whitened = scalar @ np.array(cov) @ scalar.T
        self.assertTrue(np.allclose(whitened, np.eye(2), atol=1e-8))
        self.assertFalse(np.allclose(scalar_whitened, np.eye(2), atol=1e-2))

    def test_all_synthetic_fixtures_run_full_pipeline(self) -> None:
        expected = {
            "shared_persistent": ["SHARED_PREDICTIVE_OPERATOR_SUPPORTED", "PERSISTENT_STATE_CANDIDATE"],
            "shared_driven": ["SHARED_PREDICTIVE_OPERATOR_SUPPORTED", "DRIVEN_RELATIONAL_TRANSPORT_ONLY"],
            "route_local": ["ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY"],
            "confounded": ["CONFOUNDED_NO_OPERATOR_CLAIM"],
            "rejected": ["INSTRUMENTATION_BOUNDARY_REJECTED"],
            "session_lookup_dominates": ["CONFOUNDED_NO_OPERATOR_CLAIM"],
        }
        for scenario, verdicts in expected.items():
            with self.subTest(scenario=scenario):
                custody = synthetic_custody(scenario)
                manifest = select_on_validation(training_validation_custody(custody))
                result = evaluate_sealed(custody, manifest)
                self.assertEqual(result["adjudication"]["verdicts"], verdicts)

    def test_selection_manifest_binds_stop_gate_and_evaluated_candidate(self) -> None:
        custody = synthetic_custody("shared_driven")
        manifest = select_on_validation(training_validation_custody(custody))
        self.assertIn("validation_h8_gain", manifest["selection_stop_gate"])
        self.assertEqual(manifest["evaluated_candidate"]["state_level"], manifest["state_level"])
        self.assertEqual(manifest["evaluated_candidate"]["operator_class"], manifest["operator_class"])

    def test_hierarchical_bootstrap_shape_is_session_packet_nested(self) -> None:
        custody = synthetic_custody("shared_driven")
        manifest = select_on_validation(training_validation_custody(custody))
        result = evaluate_sealed(custody, manifest)
        bootstrap = result["predictive_metrics"]["eight_step_bootstrap"]
        self.assertEqual(bootstrap["session_draws"], 4)
        self.assertEqual(bootstrap["bootstrap_iterations"], 200)
        self.assertTrue(bootstrap["nested_packet_draws"])
        self.assertEqual(len(bootstrap["gain_distribution"]), 200)

    def test_one_step_pass_eight_step_fail_blocks_shared_predictive_gate(self) -> None:
        custody = synthetic_custody("rejected")
        manifest = select_on_validation(training_validation_custody(custody))
        manifest.update(
            {
                "state_level": "S0",
                "delay_length": None,
                "operator_class": "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR",
                "regularization": 0.0,
                "o4_lift": O4_FIXED_LIFTS,
                "selection_stop_gate": "regression:raw_h8_not_rescued",
                "evaluated_candidate": {
                    "state_level": "S0",
                    "delay_length": None,
                    "operator_class": "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR",
                    "regularization": 0.0,
                    "validation_score": manifest["validation_score"],
                },
            }
        )
        manifest["analysis_choice_sha256"] = digest({key: value for key, value in manifest.items() if key != "analysis_choice_sha256"})
        result = evaluate_sealed(custody, manifest)
        self.assertGreater(result["predictive_metrics"]["one_step_nrmse_gain"], 0.10)
        self.assertLess(result["predictive_metrics"]["eight_step_nrmse_gain"], 0.05)
        self.assertEqual(result["predictive_metrics"]["eight_step_nrmse_gain"], result["horizons"]["8"]["nrmse_gain"])
        self.assertEqual(result["adjudication"]["verdicts"], ["INSTRUMENTATION_BOUNDARY_REJECTED"])


if __name__ == "__main__":
    unittest.main()
