from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

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
from analysis import pipeline as analysis_pipeline  # noqa: E402
from analysis import state as analysis_state  # noqa: E402
from analysis.pipeline import _confounds, _is_order_label_sham, evaluate_sealed, select_on_validation, training_validation_custody  # noqa: E402
from analysis.state import (  # noqa: E402
    Gauge,
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


def diagnostic_row(
    i: int,
    *,
    split: str,
    session_index: int,
    executed_family: str,
    declared_family: str,
    position: int,
    tone: int = 0,
    order_control_family: str | None = None,
    declared_position: int | None = None,
) -> dict[str, object]:
    return {
        "stage": "prepared_order",
        "split": split,
        "packet_id": f"{split}:{session_index}:{i}",
        "session_index": session_index,
        "reboot_block": f"b{session_index}",
        "route": "v4s5",
        "sender_core": 4,
        "receiver_core": 5,
        "session_chronology": session_index,
        "slot_index": i,
        "declared": {
            "order_control_family": order_control_family if order_control_family is not None else declared_family,
            "declared_order_family": declared_family,
            "declared_order_position": declared_position if declared_position is not None else position,
            "analysis_tone_index": tone,
        },
        "u_t": {
            "drive_on": True,
            "executed_mode": "PREPARED_ORDER",
            "physical_tone_index": tone,
            "executed_order_family": executed_family,
            "executed_order_position": position,
            "codeword_sign": 1,
        },
        "r_t": {"lockin_I": 0.0, "lockin_Q": 0.0, "ring_osc_period": 100.0},
        "c_t": {},
    }


def y_from(values: list[float]) -> np.ndarray:
    return np.array([[value, 0.25 * value, 100.0 + value] for value in values], dtype=float)


class AnalysisPartitionTests(unittest.TestCase):
    _synthetic_baselines: dict[str, tuple[dict[str, object], dict[str, object], dict[str, object]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.custody = run_mock(campaign_schedule())
        cls.rows = flatten_custody(cls.custody)
        cls.train_preamble = [row for row in cls.rows if row["split"] == "train" and row["stage"] == "preamble"]

    @classmethod
    def _synthetic_baseline(
        cls, scenario: str
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        if scenario not in cls._synthetic_baselines:
            custody = synthetic_custody(scenario)
            manifest = select_on_validation(training_validation_custody(custody))
            result = evaluate_sealed(custody, manifest)
            cls._synthetic_baselines[scenario] = (custody, manifest, result)
        return copy.deepcopy(cls._synthetic_baselines[scenario])

    def test_measured_state_rejects_context_declared_and_future_leakage(self) -> None:
        for field in ("session_id", "route", "order_control_family", "declared_order_family", "future_value", "target_label", "session_chronology"):
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
        custody, manifest, result = self._synthetic_baseline("shared_driven")
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
        for forbidden in ("corrcoef", "route_gap", "sham_strength", "order_counts", "session_structure"):
            self.assertNotIn(forbidden, source)

    def test_confound_outputs_are_predictive_comparisons(self) -> None:
        _, _, result = self._synthetic_baseline("confounded")
        confounds = result["confounds"]
        self.assertIn("held-out predictor", confounds["physical_tone_indexed_performance"]["comparison_model"])
        self.assertIn("held-out predictor", confounds["execution_position_indexed_performance"]["comparison_model"])
        self.assertIn("O0_TIME_INDEX held-out NRMSE", confounds["time_index_within_five_percent"]["comparison_model"])
        self.assertIn("held_out_families", confounds["single_order_family_dependence"])
        self.assertEqual(set(confounds["single_order_family_dependence"]["held_out_families"]), set(("FWD", "REV", "RND1", "RND2", "ORDER_LABEL_SHAM")))
        self.assertIn("chronology_blocks", confounds["single_chronology_position_dependence"])
        self.assertFalse(confounds["session_lookup_dominance"]["confidence_interval"]["sealed_test_identity_substitution"])
        for payload in confounds.values():
            self.assertIn("metric", payload)
            self.assertIn("comparison_model", payload)
            self.assertIn("threshold", payload)
            self.assertIn("flag", payload)
        self.assertEqual(result["adjudication"]["verdicts"], ["CONFOUNDED_NO_OPERATOR_CLAIM"])

    def test_session_lookup_is_leave_one_known_session_out(self) -> None:
        custody, _, result = self._synthetic_baseline("session_lookup_dominates")
        diagnostic = result["confounds"]["session_lookup_dominance"]
        known_ids = set(diagnostic["held_out_session_ids"])
        test_ids = {session["session_index"] for session in custody["sessions"] if session["split"] == "test"}
        self.assertTrue(known_ids)
        self.assertTrue(known_ids.isdisjoint(test_ids))
        self.assertFalse(diagnostic["sealed_test_identity_substitution"])
        self.assertFalse(diagnostic["confidence_interval"]["sealed_test_identity_substitution"])
        for payload in diagnostic["per_session_nrmse"].values():
            self.assertFalse(payload["session_id_seen_during_fit"])
        self.assertIn("aggregate_gain", diagnostic)

    def test_order_label_sham_compares_declared_and_executed_predictors(self) -> None:
        _, _, result = self._synthetic_baseline("shared_driven")
        sham = result["confounds"]["order_label_sham_predicts_comparably"]
        self.assertIn("executed_order", sham)
        self.assertIn("declared_order", sham)
        self.assertIn("order_label_sham", sham)
        self.assertIn("declared-order predictor", sham["declared_order"]["comparison_model"])
        self.assertIn("executed-order prediction on ORDER_LABEL_SHAM rows", sham["executed_order"]["comparison_model"])
        self.assertIn("declared sham-label predictor NRMSE", sham["comparison_model"])
        self.assertGreater(sham["actual_sham_row_count"], 0)
        self.assertGreater(sham["ordinary_row_count"], 0)
        self.assertIn("executed_order_nrmse", sham)
        self.assertIn("order_label_sham_gain", sham)

    def test_generated_order_label_sham_rows_are_detected_truthfully(self) -> None:
        custody = run_mock(campaign_schedule())
        rows = flatten_custody(custody)
        sham_rows = [row for row in rows if _is_order_label_sham(row)]
        ordinary_order_rows = [
            row
            for row in rows
            if row["stage"] == "prepared_order" and row["declared"].get("order_control_family") != "ORDER_LABEL_SHAM"
        ]
        self.assertEqual(len(sham_rows), 12 * 72)
        self.assertEqual(len(ordinary_order_rows), 12 * 4 * 72)
        self.assertTrue(all(row["u_t"]["executed_order_family"] in ("RND1", "RND2") for row in sham_rows))
        self.assertTrue(all(row["declared"]["declared_order_family"] in ("RND1", "RND2") for row in sham_rows))
        self.assertTrue(all(row["declared"]["order_control_family"] == "ORDER_LABEL_SHAM" for row in sham_rows))
        self.assertFalse(any(_is_order_label_sham(row) for row in ordinary_order_rows))

    def test_physical_tone_diagnostic_alone_is_non_blocking(self) -> None:
        train_rows = [
            diagnostic_row(i, split="train", session_index=i // 2, executed_family="FWD", declared_family="FWD", position=0, tone=i % 2)
            for i in range(8)
        ]
        validation_rows = [
            diagnostic_row(i + 8, split="validation", session_index=4 + i // 2, executed_family="FWD", declared_family="FWD", position=0, tone=i % 2)
            for i in range(4)
        ]
        target_rows = [
            diagnostic_row(i + 12, split="test", session_index=8 + i // 2, executed_family="FWD", declared_family="FWD", position=0, tone=i % 2)
            for i in range(4)
        ]
        y_train = y_from([1.0 if row["u_t"]["physical_tone_index"] else -1.0 for row in train_rows])
        y_validation = y_from([1.0 if row["u_t"]["physical_tone_index"] else -1.0 for row in validation_rows])
        y_true = y_from([1.0 if row["u_t"]["physical_tone_index"] else -1.0 for row in target_rows])
        baseline = np.zeros_like(y_true)
        confounds = _confounds(
            train_rows + validation_rows + target_rows,
            train_rows,
            validation_rows,
            y_train,
            y_validation,
            np.zeros_like(y_validation),
            y_true,
            baseline,
            y_true,
            target_rows,
        )
        self.assertGreater(confounds["physical_tone_indexed_performance"]["metric"], 0.05)
        self.assertFalse(confounds["physical_tone_indexed_performance"]["flag"])
        self.assertFalse(confounds["tone_vs_execution_position_disagreement"]["flag"])

    def test_declared_order_prediction_does_not_substitute_for_order_label_sham(self) -> None:
        train_rows = [
            diagnostic_row(i, split="train", session_index=i // 2, executed_family="FWD" if i % 2 else "REV", declared_family="FWD" if i % 2 else "REV", position=0)
            for i in range(8)
        ]
        validation_rows = [
            diagnostic_row(i + 8, split="validation", session_index=4 + i // 2, executed_family="FWD" if i % 2 else "REV", declared_family="FWD" if i % 2 else "REV", position=0)
            for i in range(4)
        ]
        target_rows = [
            diagnostic_row(i + 12, split="test", session_index=8 + i // 2, executed_family="FWD" if i % 2 else "REV", declared_family="FWD" if i % 2 else "REV", position=0)
            for i in range(4)
        ]
        y_train = y_from([1.0 if row["declared"]["declared_order_family"] == "FWD" else -1.0 for row in train_rows])
        y_validation = y_from([1.0 if row["declared"]["declared_order_family"] == "FWD" else -1.0 for row in validation_rows])
        y_true = y_from([1.0 if row["declared"]["declared_order_family"] == "FWD" else -1.0 for row in target_rows])
        sham = _confounds(
            train_rows + validation_rows + target_rows,
            train_rows,
            validation_rows,
            y_train,
            y_validation,
            np.zeros_like(y_validation),
            y_true,
            np.zeros_like(y_true),
            y_true,
            target_rows,
        )["order_label_sham_predicts_comparably"]
        self.assertGreater(sham["declared_order"]["gain_vs_strongest_baseline"], 0.05)
        self.assertEqual(sham["actual_sham_row_count"], 0)
        self.assertEqual(sham["ordinary_row_count"], len(target_rows))
        self.assertFalse(sham["flag"])

    def test_order_label_sham_confound_fires_when_sham_matches_executed_order(self) -> None:
        train_rows = [
            diagnostic_row(i, split="train", session_index=i // 2, executed_family="RND2", declared_family="RND1", position=0, order_control_family="ORDER_LABEL_SHAM", declared_position=i % 2)
            for i in range(8)
        ]
        validation_rows = [
            diagnostic_row(i + 8, split="validation", session_index=4 + i // 2, executed_family="RND2", declared_family="RND1", position=0, order_control_family="ORDER_LABEL_SHAM", declared_position=i % 2)
            for i in range(4)
        ]
        target_rows = [
            diagnostic_row(i + 12, split="test", session_index=8 + i // 2, executed_family="RND2", declared_family="RND1", position=0, order_control_family="ORDER_LABEL_SHAM", declared_position=i % 2)
            for i in range(4)
        ]
        y_train = y_from([1.0 if row["declared"]["declared_order_position"] else -1.0 for row in train_rows])
        y_validation = y_from([1.0 if row["declared"]["declared_order_position"] else -1.0 for row in validation_rows])
        y_true = y_from([1.0 if row["declared"]["declared_order_position"] else -1.0 for row in target_rows])
        sham = _confounds(
            train_rows + validation_rows + target_rows,
            train_rows,
            validation_rows,
            y_train,
            y_validation,
            np.zeros_like(y_validation),
            y_true,
            np.zeros_like(y_true),
            y_true,
            target_rows,
        )["order_label_sham_predicts_comparably"]
        self.assertLessEqual(sham["performance_ratio"], sham["threshold"])
        self.assertTrue(sham["flag"])

    def test_declared_sham_label_mutation_changes_sham_diagnostic(self) -> None:
        train_rows = [
            diagnostic_row(i, split="train", session_index=i // 2, executed_family="RND2", declared_family="RND1" if i % 2 else "RND2", position=0, order_control_family="ORDER_LABEL_SHAM")
            for i in range(8)
        ]
        validation_rows = [
            diagnostic_row(i + 8, split="validation", session_index=4 + i // 2, executed_family="RND2", declared_family="RND1" if i % 2 else "RND2", position=0, order_control_family="ORDER_LABEL_SHAM")
            for i in range(4)
        ]
        target_rows = [
            diagnostic_row(i + 12, split="test", session_index=8 + i // 2, executed_family="RND2", declared_family="RND1" if i % 2 else "RND2", position=0, order_control_family="ORDER_LABEL_SHAM")
            for i in range(4)
        ]
        y_train = y_from([1.0 if row["declared"]["declared_order_family"] == "RND1" else -1.0 for row in train_rows])
        y_validation = y_from([1.0 if row["declared"]["declared_order_family"] == "RND1" else -1.0 for row in validation_rows])
        y_true = y_from([1.0 if row["declared"]["declared_order_family"] == "RND1" else -1.0 for row in target_rows])
        base = _confounds(train_rows + validation_rows + target_rows, train_rows, validation_rows, y_train, y_validation, np.zeros_like(y_validation), y_true, np.zeros_like(y_true), y_true, target_rows)["order_label_sham_predicts_comparably"]
        mutated_rows = [dict(row, declared={**row["declared"]}) for row in target_rows]
        mutated_rows[0]["declared"]["declared_order_family"] = "RND2" if mutated_rows[0]["declared"]["declared_order_family"] == "RND1" else "RND1"
        mutated = _confounds(train_rows + validation_rows + mutated_rows, train_rows, validation_rows, y_train, y_validation, np.zeros_like(y_validation), y_true, np.zeros_like(y_true), y_true, mutated_rows)["order_label_sham_predicts_comparably"]
        self.assertNotEqual(base["order_label_sham_nrmse"], mutated["order_label_sham_nrmse"])

    def test_executed_order_mutation_changes_executed_order_sham_prediction(self) -> None:
        train_rows = [
            diagnostic_row(i, split="train", session_index=i // 2, executed_family="RND1" if i % 2 else "RND2", declared_family="RND1", position=0, order_control_family="ORDER_LABEL_SHAM")
            for i in range(8)
        ]
        validation_rows = [
            diagnostic_row(i + 8, split="validation", session_index=4 + i // 2, executed_family="RND1" if i % 2 else "RND2", declared_family="RND1", position=0, order_control_family="ORDER_LABEL_SHAM")
            for i in range(4)
        ]
        target_rows = [
            diagnostic_row(i + 12, split="test", session_index=8 + i // 2, executed_family="RND1" if i % 2 else "RND2", declared_family="RND1", position=0, order_control_family="ORDER_LABEL_SHAM")
            for i in range(4)
        ]
        y_train = y_from([1.0 if row["u_t"]["executed_order_family"] == "RND1" else -1.0 for row in train_rows])
        y_validation = y_from([1.0 if row["u_t"]["executed_order_family"] == "RND1" else -1.0 for row in validation_rows])
        y_true = y_from([1.0 if row["u_t"]["executed_order_family"] == "RND1" else -1.0 for row in target_rows])
        base = _confounds(train_rows + validation_rows + target_rows, train_rows, validation_rows, y_train, y_validation, np.zeros_like(y_validation), y_true, np.zeros_like(y_true), y_true, target_rows)["order_label_sham_predicts_comparably"]
        mutated_rows = [dict(row, u_t={**row["u_t"]}) for row in target_rows]
        mutated_rows[0]["u_t"]["executed_order_family"] = "RND2" if mutated_rows[0]["u_t"]["executed_order_family"] == "RND1" else "RND1"
        mutated = _confounds(train_rows + validation_rows + mutated_rows, train_rows, validation_rows, y_train, y_validation, np.zeros_like(y_validation), y_true, np.zeros_like(y_true), y_true, mutated_rows)["order_label_sham_predicts_comparably"]
        self.assertNotEqual(base["executed_order_nrmse"], mutated["executed_order_nrmse"])

    def test_persistence_uses_test_only_matched_hierarchical_bounds(self) -> None:
        _, _, result = self._synthetic_baseline("shared_persistent")
        bounds = result["drive_off"]["position_bounds"]
        self.assertTrue(result["drive_off"]["three_consecutive_lower_above_sham"])
        for offset in ("1", "2", "3"):
            payload = bounds[offset]
            self.assertGreater(payload["post_drive_lower_95"], payload["matched_sham_upper_95"])
            self.assertEqual(payload["matched_strata"], ["route", "physical_tone", "sender_off_position", "reboot_block"])
            self.assertEqual(payload["unmatched_row_count"], 0)
            self.assertTrue(payload["matching_keys"])
            self.assertEqual(payload["post_drive_bootstrap"]["session_draws"], 4)
            self.assertEqual(payload["matched_sham_bootstrap"]["session_draws"], 4)
            self.assertTrue(payload["post_drive_bootstrap"]["mean_distribution"])
            self.assertTrue(payload["matched_sham_bootstrap"]["mean_distribution"])

    def test_persistence_sham_route_mutation_removes_exact_match(self) -> None:
        custody = synthetic_custody("shared_persistent")
        mutated = False
        for session in custody["sessions"]:
            if session["split"] != "test":
                continue
            for slot in session["slots"]:
                if slot["stage"] == "trajectory" and slot["u_t"].get("executed_mode") == "CARRIER_OFF_SHAM":
                    slot["route"] = "mutated_route"
                    mutated = True
                    break
            if mutated:
                break
        manifest = select_on_validation(training_validation_custody(custody))
        result = evaluate_sealed(custody, manifest)
        first = result["drive_off"]["position_bounds"]["1"]
        self.assertGreater(first["unmatched_row_count"], 0)
        self.assertFalse(first["pass"])

    def test_full_covariance_whitening_is_not_scalar_trace_scaling(self) -> None:
        cov = ((4.0, 1.5), (1.5, 1.0))
        inv = symmetric_inverse_sqrt(cov)
        whitened = inv @ np.array(cov) @ inv.T
        scalar = np.eye(2) / np.sqrt(np.trace(np.array(cov)) / 2.0)
        scalar_whitened = scalar @ np.array(cov) @ scalar.T
        self.assertTrue(np.allclose(whitened, np.eye(2), atol=1e-8))
        self.assertFalse(np.allclose(scalar_whitened, np.eye(2), atol=1e-2))

    def test_whitener_cache_preserves_independent_array_results(self) -> None:
        cov = ((3.75, 0.625), (0.625, 1.25))
        values, vectors = np.linalg.eigh(np.array(cov, dtype=float))
        expected = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-9))) @ vectors.T
        analysis_state._symmetric_inverse_sqrt_values.cache_clear()
        with mock.patch.object(
            analysis_state.np.linalg,
            "eigh",
            wraps=analysis_state.np.linalg.eigh,
        ) as eigh:
            first = symmetric_inverse_sqrt(cov)
            second = symmetric_inverse_sqrt(cov)
        self.assertEqual(eigh.call_count, 1)
        self.assertTrue(np.array_equal(first, expected))
        self.assertTrue(np.array_equal(first, second))
        first[0, 0] += 1.0
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.allclose(second @ np.array(cov) @ second.T, np.eye(2), atol=1e-8))

    def test_rollout_reuses_one_whitener_per_call(self) -> None:
        rows = []
        for index in range(10):
            item = diagnostic_row(
                index,
                split="test",
                session_index=0,
                executed_family="RND1",
                declared_family="RND1",
                position=index,
            )
            item["packet_id"] = "rollout-whitener-regression"
            item["r_t"] = {
                "lockin_I": float(index + 1),
                "lockin_Q": float(index + 1) / 4.0,
                "ring_osc_period": 100.0 + index,
            }
            rows.append(item)

        gauge = Gauge(
            complex_anchor_alpha=(0j,) * 12,
            amplitude_floor=(0.0,) * 12,
            preamble_drift_estimate=0j,
            local_idle_covariance=((1.0, 0.0), (0.0, 1.0)),
        )

        class ExactPredictor:
            def predict(self, _x: np.ndarray, target_rows: list[dict[str, object]]) -> np.ndarray:
                return np.array(
                    [
                        [
                            float(target["r_t"]["lockin_I"]),  # type: ignore[index]
                            float(target["r_t"]["lockin_Q"]),  # type: ignore[index]
                            float(target["r_t"]["ring_osc_period"]),  # type: ignore[index]
                        ]
                        for target in target_rows
                    ],
                    dtype=float,
                )

        predictor = ExactPredictor()
        sigma = ((2.0, 0.25), (0.25, 1.0))
        for state_level, expected_calls in (("S0", 0), ("S1", 1)):
            with self.subTest(state_level=state_level):
                with mock.patch.object(
                    analysis_pipeline,
                    "symmetric_inverse_sqrt",
                    wraps=analysis_pipeline.symmetric_inverse_sqrt,
                ) as inverse_sqrt:
                    metrics = analysis_pipeline._rollout_metrics(
                        {"state_level": state_level},
                        predictor,
                        predictor,
                        rows,
                        {0: gauge},
                        sigma,
                    )
                self.assertEqual(set(metrics), {1, 2, 4, 8})
                self.assertEqual(inverse_sqrt.call_count, expected_calls)

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
                _, _, result = self._synthetic_baseline(scenario)
                self.assertEqual(result["adjudication"]["verdicts"], verdicts)

    def test_route_local_requires_complete_within_route_gate(self) -> None:
        computed = {
            "predictive_metrics": {
                "one_step_nrmse_gain": 0.0,
                "eight_step_nrmse_gain": 0.0,
                "one_step_bootstrap_lower": -0.1,
                "eight_step_bootstrap_lower": -0.1,
                "route_v4s5_complex_corr": 0.99,
                "route_v2s3_complex_corr": 0.99,
                "worst_session_delta_vs_baseline": 0.0,
                "session_lookup_gain_margin": 0.0,
            },
            "route_transfer": {"v4s5_to_v2s3": {"lower_gain": -0.01}, "v2s3_to_v4s5": {"lower_gain": -0.01}},
            "drive_off": {"three_consecutive_lower_above_sham": False, "zero_input_decay_gain": 0.0, "zero_input_decay_gain_lower": 0.0},
            "confounds": {"session_lookup_dominance": {"flag": False}},
            "route_local_gates": {
                "v4s5": {
                    "one_step_gain": 0.0,
                    "eight_step_gain": 0.0,
                    "one_step_blocked_lower": -0.1,
                    "eight_step_blocked_lower": -0.1,
                    "complex_correlation": 0.99,
                    "worst_session_delta": 0.0,
                    "session_identity_gain": 0.20,
                    "session_identity_margin": -0.20,
                    "session_dominance_passed": False,
                    "pass": False,
                }
            },
            "within_route_pass": True,
        }
        self.assertEqual(derive_adjudication(computed)["verdicts"], ["INSTRUMENTATION_BOUNDARY_REJECTED"])
        computed["route_local_gates"]["v4s5"].update(
            {
                "one_step_gain": 0.10,
                "eight_step_gain": 0.05,
                "one_step_blocked_lower": 0.01,
                "eight_step_blocked_lower": 0.01,
                "session_identity_gain": 0.0,
                "session_identity_margin": 0.10,
                "session_dominance_passed": True,
                "pass": True,
            }
        )
        self.assertEqual(derive_adjudication(computed)["verdicts"], ["ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY"])

    def test_route_local_fixture_exposes_complete_gate_payload(self) -> None:
        _, _, result = self._synthetic_baseline("route_local")
        self.assertEqual(result["adjudication"]["verdicts"], ["ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY"])
        for route, gate in result["route_local_gates"].items():
            self.assertIn(route, ("v2s3", "v4s5"))
            self.assertTrue(gate["pass"])
            for key in (
                "one_step_gain",
                "eight_step_gain",
                "one_step_blocked_lower",
                "eight_step_blocked_lower",
                "complex_correlation",
                "worst_session_delta",
                "session_identity_gain",
                "session_identity_margin",
                "session_identity_diagnostic",
                "session_dominance_passed",
            ):
                self.assertIn(key, gate)
            self.assertGreater(gate["session_identity_margin"], 0.05)

    def test_selection_manifest_binds_stop_gate_and_evaluated_candidate(self) -> None:
        custody = synthetic_custody("shared_driven")
        manifest = select_on_validation(training_validation_custody(custody))
        self.assertIn("validation_h8_gain", manifest["selection_stop_gate"])
        self.assertEqual(manifest["evaluated_candidate"]["state_level"], manifest["state_level"])
        self.assertEqual(manifest["evaluated_candidate"]["operator_class"], manifest["operator_class"])

    def test_hierarchical_bootstrap_shape_is_session_packet_nested(self) -> None:
        _, _, result = self._synthetic_baseline("shared_driven")
        bootstrap = result["predictive_metrics"]["eight_step_bootstrap"]
        self.assertEqual(bootstrap["session_draws"], 4)
        self.assertEqual(bootstrap["bootstrap_iterations"], 200)
        self.assertTrue(bootstrap["nested_packet_draws"])
        self.assertEqual(len(bootstrap["gain_distribution"]), 200)

    def test_one_step_pass_eight_step_fail_blocks_shared_predictive_gate(self) -> None:
        custody = synthetic_custody("rejected")
        for session in custody["sessions"]:
            for slot in session["slots"]:
                state = 0.001 * ((slot["slot_index"] * 17 + session["session_index"] * 5) % 23 - 11)
                slot["r_t"]["lockin_I"] = round(state, 9)
                slot["r_t"]["lockin_Q"] = round(0.35 * state, 9)
                slot["r_t"]["ring_osc_period"] = round(100.0 + 0.25 * state, 9)
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
