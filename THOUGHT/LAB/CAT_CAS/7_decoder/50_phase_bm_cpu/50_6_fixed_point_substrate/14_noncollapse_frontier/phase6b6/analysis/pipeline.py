"""End-to-end Phase 6B.6 software analysis pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from analysis.adjudication import derive_adjudication
from analysis.metrics import complex_corr, hierarchical_bootstrap_bounds, hierarchical_bootstrap_gain, nrmse, packet_groups, summarize
from analysis.observations import assert_test_sealed, flatten_custody, split_rows
from analysis.operators import (
    OPERATOR_LADDER,
    analysis_contract_digest,
    deterministic_seed,
    fit_operator,
)
from analysis.state import estimate_session_gauges, executed_control_vector, state_vector, symmetric_inverse_sqrt, training_global_covariance
from contracts.contract import DELAY_CANDIDATES, HORIZONS, O4_FIXED_LIFTS, ORDER_FAMILIES, REGULARIZATION_LADDER, digest


SCIENCE_OPERATORS = tuple(name for name in OPERATOR_LADDER if not name.startswith("O0_"))
BASELINE_OPERATORS = ("O0_TRAINING_MEAN", "O0_LAST_VALUE", "O0_RETURN_TO_BASELINE", "O0_INPUT_ONLY", "O0_TIME_INDEX")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def training_validation_custody(custody: dict[str, Any]) -> dict[str, Any]:
    """Return a physical custody view whose sessions contain no test rows."""
    view = deepcopy(custody)
    view["sessions"] = []
    for session in custody["sessions"]:
        slots = [deepcopy(slot) for slot in session["slots"] if slot["split"] in ("train", "validation")]
        if slots:
            session_view = {key: deepcopy(value) for key, value in session.items() if key != "slots"}
            session_view["slots"] = slots
            view["sessions"].append(session_view)
    view["session_count"] = len(view["sessions"])
    view["total_slots"] = sum(len(session["slots"]) for session in view["sessions"])
    view["test_opened"] = False
    view["custody_view"] = "train_validation_only"
    view.pop("custody_sha256", None)
    return view


def _eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["stage"] in ("prepared_order", "trajectory")]


def _grouped(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in _eligible(rows):
        packet = row.get("packet_id")
        if packet is None and row["stage"] == "prepared_order":
            packet = "prepared_order"
        if packet is None:
            continue
        groups.setdefault((row["session_index"], str(packet)), []).append(row)
    return [sorted(value, key=lambda row: row["slot_index"]) for _, value in sorted(groups.items())]


def _dataset(
    rows: list[dict[str, Any]],
    state_level: str,
    delay: int | None,
    gauges: dict[int, Any],
    sigma_train: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[list[dict[str, Any]]]]:
    x_rows = []
    y_rows = []
    target_rows = []
    source_groups: list[list[dict[str, Any]]] = []
    for group in _grouped(rows):
        if len(group) < 2:
            continue
        source_groups.append(group)
        for i in range(len(group) - 1):
            if state_level == "S2" and delay is not None and i < delay - 1:
                continue
            gauge = gauges[group[i]["session_index"]]
            x_rows.append(state_vector(group, i, state_level, gauge, sigma_train, delay))
            y_rows.append(state_vector(group, i + 1, "S0", gauge, sigma_train)[:3])
            target_rows.append(group[i + 1])
    if not x_rows:
        raise ValueError("no usable trainable rows for selected state")
    return np.vstack(x_rows), np.vstack(y_rows), target_rows, source_groups


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(manifest)
    payload.pop("analysis_choice_sha256", None)
    return payload


def _validate_manifest(custody: dict[str, Any], manifest: dict[str, Any]) -> None:
    if "analysis_choice_sha256" not in manifest:
        raise PermissionError("TEST_SET_SEALED_UNTIL_ANALYSIS_CHOICE_MANIFEST")
    if digest(_manifest_payload(manifest)) != manifest["analysis_choice_sha256"]:
        raise PermissionError("ANALYSIS_CHOICE_MANIFEST_DIGEST_MISMATCH")
    if manifest["contract_sha256"] != custody["contract_sha256"]:
        raise PermissionError("ANALYSIS_CHOICE_CONTRACT_DIGEST_MISMATCH")
    if manifest["schedule_sha256"] != custody["schedule_sha256"]:
        raise PermissionError("ANALYSIS_CHOICE_SCHEDULE_DIGEST_MISMATCH")
    required = (
        "state_level",
        "delay_length",
        "operator_class",
        "regularization",
        "o4_lift",
        "validation_score",
        "thresholds",
        "bootstrap_seeds",
        "created_before_test_open",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise PermissionError("ANALYSIS_CHOICE_MANIFEST_INCOMPLETE:" + ",".join(missing))
    if not manifest["created_before_test_open"]:
        raise PermissionError("ANALYSIS_CHOICE_CREATED_AFTER_TEST_OPEN")


def create_analysis_choice_manifest(contract_sha256: str, schedule_sha256: str, choices: dict[str, Any]) -> dict[str, Any]:
    seeds = {
        "bootstrap": deterministic_seed(analysis_contract_digest(), "bootstrap"),
        "route_transfer": deterministic_seed(analysis_contract_digest(), "route_transfer"),
    }
    manifest = {
        "schema_id": "CAT_CAS_PHASE6B6_ANALYSIS_CHOICE_MANIFEST_V1",
        "contract_sha256": contract_sha256,
        "schedule_sha256": schedule_sha256,
        "state_level": choices["state_level"],
        "delay_length": choices.get("delay_length"),
        "operator_class": choices["operator_class"],
        "regularization": choices["regularization"],
        "o4_lift": choices.get("o4_lift"),
        "validation_score": choices["validation_score"],
        "selection_stop_gate": choices["selection_stop_gate"],
        "evaluated_candidate": {
            "state_level": choices["state_level"],
            "delay_length": choices.get("delay_length"),
            "operator_class": choices["operator_class"],
            "regularization": choices["regularization"],
            "validation_score": choices["validation_score"],
        },
        "selection_rule": "S0_THEN_S1_THEN_S2_AND_O1_THEN_O2_THEN_O3_THEN_O4_SIMPLEST_WITHIN_2_PERCENT",
        "thresholds": {
            "one_step_nrmse_gain": 0.10,
            "eight_step_nrmse_gain": 0.05,
            "route_complex_corr": 0.80,
            "session_lookup_margin": 0.05,
            "zero_input_decay_gain": 0.10,
        },
        "bootstrap_seeds": seeds,
        "created_before_test_open": True,
    }
    manifest["analysis_choice_sha256"] = digest(manifest)
    return manifest


def _state_candidates() -> Iterable[tuple[str, int | None]]:
    yield ("S0", None)
    yield ("S1", None)
    for delay in DELAY_CANDIDATES:
        yield ("S2", delay)


def _best_operator_for_state(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    state_level: str,
    delay: int | None,
    gauges: dict[int, Any],
    sigma: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    x_train, y_train, fit_rows, _ = _dataset(train_rows, state_level, delay, gauges, sigma)
    x_val, y_val, val_target_rows, _ = _dataset(validation_rows, state_level, delay, gauges, sigma)
    baseline_name, baseline_pred, _ = _strongest_baseline_predictions(x_train, y_train, fit_rows, x_val, y_val, val_target_rows)
    baseline_fit = fit_operator(baseline_name, x_train, y_train, fit_rows)
    best_candidate: dict[str, Any] | None = None
    for operator_class in SCIENCE_OPERATORS:
        regs = REGULARIZATION_LADDER if operator_class == "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR" else (0.0,)
        operator_candidates: list[dict[str, Any]] = []
        for reg in regs:
            fitted = fit_operator(operator_class, x_train, y_train, fit_rows, reg)
            pred = fitted.predict(x_val, val_target_rows)
            score = nrmse(y_val, pred)
            gain = _gain(y_val, pred, baseline_pred)
            temp_manifest = {"state_level": state_level, "delay_length": delay}
            validation_horizons = _rollout_metrics(temp_manifest, fitted, baseline_fit, validation_rows, gauges, sigma)
            validation_h8_gain = validation_horizons[8]["nrmse_gain"]
            validation_sufficiency_pass = gain >= 0.10 and validation_h8_gain >= 0.05
            operator_candidates.append(
                {
                    "state_level": state_level,
                    "delay_length": delay,
                    "operator_class": operator_class,
                    "regularization": reg,
                    "o4_lift": O4_FIXED_LIFTS if operator_class == "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR" else None,
                    "validation_score": score,
                    "validation_gain": gain,
                    "validation_h8_gain": validation_h8_gain,
                    "validation_sufficiency_pass": validation_sufficiency_pass,
                    "selection_stop_gate": f"{state_level}:{operator_class}:validation_gain>=0.10_and_validation_h8_gain>=0.05",
                }
            )
        selected = min(operator_candidates, key=lambda item: item["validation_score"])
        if best_candidate is None or selected["validation_score"] < best_candidate["validation_score"]:
            best_candidate = selected
        if selected["validation_sufficiency_pass"]:
            return selected
    if best_candidate is None:
        raise AssertionError("empty operator candidate set")
    best_candidate["selection_stop_gate"] = f"{state_level}:exhausted_operator_ladder"
    return best_candidate


def select_on_validation(custody: dict[str, Any]) -> dict[str, Any]:
    rows = flatten_custody(custody)
    assert_test_sealed(rows, allow_test=False)
    train_rows = split_rows(rows, "train")
    validation_rows = split_rows(rows, "validation")
    gauges = estimate_session_gauges(rows)
    sigma = training_global_covariance([row for row in train_rows if row["stage"] == "preamble"])
    selected = None
    for state_level in ("S0", "S1"):
        candidate = _best_operator_for_state(train_rows, validation_rows, state_level, None, gauges, sigma)
        selected = candidate
        if candidate["validation_sufficiency_pass"]:
            break
    if selected is None or not selected["validation_sufficiency_pass"]:
        s2_candidates = [_best_operator_for_state(train_rows, validation_rows, "S2", delay, gauges, sigma) for delay in DELAY_CANDIDATES]
        best_s2 = min(s2_candidates, key=lambda item: item["validation_score"])
        selected = min(
            [item for item in s2_candidates if item["validation_score"] <= best_s2["validation_score"] * 1.02],
            key=lambda item: item["delay_length"] or 0,
        )
        selected["selection_stop_gate"] = (
            selected["selection_stop_gate"] if selected["validation_sufficiency_pass"] else "S2:exhausted_state_and_operator_ladders"
        )
    return create_analysis_choice_manifest(custody["contract_sha256"], custody["schedule_sha256"], selected)


def _strongest_baseline_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    fit_rows: list[dict[str, Any]],
    x_target: np.ndarray,
    y_target: np.ndarray,
    target_rows: list[dict[str, Any]],
) -> tuple[str, np.ndarray, float]:
    scored = []
    for operator in BASELINE_OPERATORS:
        fitted = fit_operator(operator, x_train, y_train, fit_rows)
        pred = fitted.predict(x_target, target_rows)
        scored.append((operator, pred, nrmse(y_target, pred)))
    return min(scored, key=lambda item: item[2])


def _gain(y_true: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float:
    base = nrmse(y_true, baseline)
    mod = nrmse(y_true, model)
    return (base - mod) / max(base, 1e-9)


def _design_labels(labels: list[str], vocabulary: tuple[str, ...] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    vocab = vocabulary or tuple(sorted(set(labels)))
    index = {label: i for i, label in enumerate(vocab)}
    design = np.zeros((len(labels), len(vocab) + 1), dtype=float)
    design[:, 0] = 1.0
    for row_i, label in enumerate(labels):
        if label in index:
            design[row_i, index[label] + 1] = 1.0
    return design, vocab


def _fit_label_predictor(
    train_labels: list[str],
    y_train: np.ndarray,
    target_labels: list[str],
    vocabulary: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    train_design, vocab = _design_labels(train_labels, vocabulary)
    target_design, _ = _design_labels(target_labels, vocab)
    coefs = np.linalg.pinv(train_design.T @ train_design + 1e-9 * np.eye(train_design.shape[1])) @ train_design.T @ y_train
    return target_design @ coefs, vocab


def _diagnostic_summary(
    rows: list[dict[str, Any]],
    y_true: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    seed: int,
    comparison_model: str,
) -> dict[str, Any]:
    bootstrap = hierarchical_bootstrap_gain(rows, y_true, pred, baseline, seed)
    return {
        "comparison_model": comparison_model,
        "nrmse": nrmse(y_true, pred),
        "baseline_nrmse": nrmse(y_true, baseline),
        "gain_vs_strongest_baseline": _gain(y_true, pred, baseline),
        "blocked_confidence_interval": {
            "lower_95_gain": bootstrap["lower_95_bound"],
            "gain_distribution": bootstrap["gain_distribution"],
            "session_draws": bootstrap["session_draws"],
            "nested_packet_draws": bootstrap["nested_packet_draws"],
        },
        "per_route": summarize(rows, y_true, pred)["per_route"],
        "per_session": summarize(rows, y_true, pred)["per_session"],
    }


def _label_value(row: dict[str, Any], label: str) -> str:
    if label == "physical_tone":
        return f"tone:{row['u_t'].get('physical_tone_index')}"
    if label == "execution_position":
        return f"position:{row['u_t'].get('executed_order_position')}"
    if label == "executed_order":
        return f"{row['u_t'].get('executed_order_family')}:{row['u_t'].get('executed_order_position')}"
    if label == "declared_order":
        return f"{row['declared'].get('declared_order_family')}:{row['declared'].get('declared_order_position')}"
    if label == "order_label_sham":
        if _is_order_label_sham(row):
            return f"sham:{row['declared'].get('declared_order_family')}:{row['declared'].get('declared_order_position')}"
        return "ordinary-order-control"
    if label == "chronology":
        return f"chronology:{row['session_chronology']}"
    raise ValueError(f"unknown diagnostic label: {label}")


def _predictive_label_diagnostic(
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    test_rows: list[dict[str, Any]],
    y_test: np.ndarray,
    baseline: np.ndarray,
    label: str,
    seed: int,
    comparison_model: str,
) -> dict[str, Any]:
    train_labels = [_label_value(row, label) for row in train_rows]
    target_labels = [_label_value(row, label) for row in test_rows]
    pred, _ = _fit_label_predictor(train_labels, y_train, target_labels)
    return _diagnostic_summary(test_rows, y_test, pred, baseline, seed, comparison_model)


def _is_order_label_sham(row: dict[str, Any]) -> bool:
    return row["declared"].get("order_control_family") == "ORDER_LABEL_SHAM"


def _order_label_sham_diagnostic(
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    target_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    baseline: np.ndarray,
    declared_order: dict[str, Any],
) -> dict[str, Any]:
    train_sham_idx = [i for i, row in enumerate(train_rows) if _is_order_label_sham(row)]
    target_sham_idx = [i for i, row in enumerate(target_rows) if _is_order_label_sham(row)]
    ordinary_count = len(target_rows) - len(target_sham_idx)
    schedule_has_sham = bool(train_sham_idx or target_sham_idx)
    if schedule_has_sham and not target_sham_idx:
        return {
            **_confound(True, float("inf"), 1.05, "ORDER_LABEL_SHAM control rows were present but no sham rows were detected for analysis"),
            "actual_sham_row_count": 0,
            "ordinary_row_count": ordinary_count,
            "executed_order": {},
            "declared_order": declared_order,
            "order_label_sham": {},
            "executed_order_prediction_on_sham_rows": {},
            "declared_sham_label_prediction_on_sham_rows": {},
            "blocked_confidence_information": {},
            "executed_order_nrmse": float("inf"),
            "executed_order_gain": 0.0,
            "order_label_sham_nrmse": float("inf"),
            "order_label_sham_gain": 0.0,
            "performance_ratio": float("inf"),
            "performance_delta": 0.0,
        }
    if not target_sham_idx:
        return {
            **_confound(False, float("inf"), 1.05, "no ORDER_LABEL_SHAM control rows present in analysis stratum"),
            "actual_sham_row_count": 0,
            "ordinary_row_count": ordinary_count,
            "executed_order": {},
            "declared_order": declared_order,
            "order_label_sham": {},
            "executed_order_prediction_on_sham_rows": {},
            "declared_sham_label_prediction_on_sham_rows": {},
            "blocked_confidence_information": {},
            "executed_order_nrmse": float("inf"),
            "executed_order_gain": 0.0,
            "order_label_sham_nrmse": float("inf"),
            "order_label_sham_gain": 0.0,
            "performance_ratio": float("inf"),
            "performance_delta": 0.0,
        }
    if not train_sham_idx:
        return {
            **_confound(True, float("inf"), 1.05, "ORDER_LABEL_SHAM target rows lack training sham controls"),
            "actual_sham_row_count": len(target_sham_idx),
            "ordinary_row_count": ordinary_count,
            "executed_order": {},
            "declared_order": declared_order,
            "order_label_sham": {},
            "executed_order_prediction_on_sham_rows": {},
            "declared_sham_label_prediction_on_sham_rows": {},
            "blocked_confidence_information": {},
            "executed_order_nrmse": float("inf"),
            "executed_order_gain": 0.0,
            "order_label_sham_nrmse": float("inf"),
            "order_label_sham_gain": 0.0,
            "performance_ratio": float("inf"),
            "performance_delta": 0.0,
        }
    sham_rows = [target_rows[i] for i in target_sham_idx]
    y_sham = y_true[target_sham_idx]
    baseline_sham = baseline[target_sham_idx]
    executed_pred, _ = _fit_label_predictor(
        [_label_value(row, "executed_order") for row in train_rows],
        y_train,
        [_label_value(row, "executed_order") for row in sham_rows],
    )
    sham_pred, _ = _fit_label_predictor(
        [_label_value(train_rows[i], "order_label_sham") for i in train_sham_idx],
        y_train[train_sham_idx],
        [_label_value(row, "order_label_sham") for row in sham_rows],
    )
    executed_summary = _diagnostic_summary(sham_rows, y_sham, executed_pred, baseline_sham, 305, "executed-order prediction on ORDER_LABEL_SHAM rows")
    sham_summary = _diagnostic_summary(sham_rows, y_sham, sham_pred, baseline_sham, 306, "declared sham-label prediction on ORDER_LABEL_SHAM rows")
    direct_bootstrap = hierarchical_bootstrap_gain(sham_rows, y_sham, sham_pred, executed_pred, 307)
    ratio = sham_summary["nrmse"] / max(executed_summary["nrmse"], 1e-9)
    gain_delta = executed_summary["gain_vs_strongest_baseline"] - sham_summary["gain_vs_strongest_baseline"]
    confidence = {
        "executed_order_lower_95_gain": executed_summary["blocked_confidence_interval"]["lower_95_gain"],
        "order_label_sham_lower_95_gain": sham_summary["blocked_confidence_interval"]["lower_95_gain"],
        "sham_vs_executed_lower_95_gain": direct_bootstrap["lower_95_bound"],
        "sham_vs_executed_gain_distribution": direct_bootstrap["gain_distribution"],
        "blocked_gain_delta": gain_delta,
        "session_draws": direct_bootstrap["session_draws"],
        "nested_packet_draws": direct_bootstrap["nested_packet_draws"],
    }
    return {
        **_confound(
            ratio <= 1.05 and sham_summary["gain_vs_strongest_baseline"] > 0.05 and direct_bootstrap["lower_95_bound"] > 0.0,
            ratio,
            1.05,
            "declared sham-label predictor NRMSE / executed-order predictor NRMSE on actual ORDER_LABEL_SHAM rows",
            confidence,
        ),
        "actual_sham_row_count": len(target_sham_idx),
        "ordinary_row_count": ordinary_count,
        "executed_order": executed_summary,
        "declared_order": declared_order,
        "order_label_sham": sham_summary,
        "executed_order_prediction_on_sham_rows": executed_summary,
        "declared_sham_label_prediction_on_sham_rows": sham_summary,
        "blocked_confidence_information": confidence,
        "executed_order_nrmse": executed_summary["nrmse"],
        "executed_order_gain": executed_summary["gain_vs_strongest_baseline"],
        "order_label_sham_nrmse": sham_summary["nrmse"],
        "order_label_sham_gain": sham_summary["gain_vs_strongest_baseline"],
        "performance_ratio": ratio,
        "performance_delta": gain_delta,
    }


def _entity_gains(rows: list[dict[str, Any]], y_true: np.ndarray, pred: np.ndarray, base: np.ndarray, key: str) -> list[float]:
    gains = []
    values = sorted({row.get(key) for row in rows})
    for value in values:
        idx = [i for i, row in enumerate(rows) if row.get(key) == value]
        if idx:
            gains.append(_gain(y_true[idx], pred[idx], base[idx]))
    return gains


def _packet_gain_distribution(rows: list[dict[str, Any]], y_true: np.ndarray, pred: np.ndarray, base: np.ndarray) -> list[float]:
    gains = []
    for idx in packet_groups(rows).values():
        if idx:
            gains.append(_gain(y_true[idx], pred[idx], base[idx]))
    return gains


def _route_gate_payload(
    target_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    eight: dict[str, Any],
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    validation_rows: list[dict[str, Any]],
    y_validation: np.ndarray,
    validation_baseline: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for route in sorted({row["route"] for row in target_rows}):
        idx = [i for i, row in enumerate(target_rows) if row["route"] == route]
        eight_idx = [i for i, row in enumerate(eight["target_rows"]) if row["route"] == route]
        if not idx or not eight_idx:
            continue
        one_bootstrap = hierarchical_bootstrap_gain(
            [target_rows[i] for i in idx],
            y_true[idx],
            pred[idx],
            baseline[idx],
            seed + len(payload) * 10,
        )
        eight_bootstrap = hierarchical_bootstrap_gain(
            [eight["target_rows"][i] for i in eight_idx],
            eight["y_true"][eight_idx],
            eight["y_pred"][eight_idx],
            eight["y_baseline"][eight_idx],
            seed + len(payload) * 10 + 1,
        )
        route_train_idx = [i for i, row in enumerate(train_rows) if row["route"] == route]
        route_validation_idx = [i for i, row in enumerate(validation_rows) if row["route"] == route]
        session_gains = _entity_gains([target_rows[i] for i in idx], y_true[idx], pred[idx], baseline[idx], "session_index")
        route_identity = _session_identity_diagnostic(
            [train_rows[i] for i in route_train_idx],
            y_train[route_train_idx],
            [validation_rows[i] for i in route_validation_idx],
            y_validation[route_validation_idx],
            validation_baseline[route_validation_idx],
            _gain(y_true[idx], pred[idx], baseline[idx]),
        )
        session_identity_margin = _gain(y_true[idx], pred[idx], baseline[idx]) - route_identity["metric"]
        gate = {
            "one_step_gain": _gain(y_true[idx], pred[idx], baseline[idx]),
            "eight_step_gain": _gain(eight["y_true"][eight_idx], eight["y_pred"][eight_idx], eight["y_baseline"][eight_idx]),
            "one_step_blocked_lower": one_bootstrap["lower_95_bound"],
            "eight_step_blocked_lower": eight_bootstrap["lower_95_bound"],
            "complex_correlation": complex_corr(y_true[idx], pred[idx]),
            "worst_session_delta": min(session_gains) if session_gains else -1.0,
            "session_identity_gain": route_identity["metric"],
            "session_identity_margin": session_identity_margin,
            "session_identity_diagnostic": route_identity,
            "session_dominance_passed": session_identity_margin > 0.05,
        }
        gate["pass"] = (
            gate["one_step_gain"] >= 0.10
            and gate["eight_step_gain"] >= 0.05
            and gate["one_step_blocked_lower"] > 0.0
            and gate["eight_step_blocked_lower"] > 0.0
            and gate["complex_correlation"] >= 0.80
            and gate["worst_session_delta"] >= -0.05
            and gate["session_dominance_passed"]
        )
        payload[route] = gate
    return payload


def _rollout_metrics(
    manifest: dict[str, Any],
    fitted: Any,
    baseline_fit: Any,
    test_rows: list[dict[str, Any]],
    gauges: dict[int, Any],
    sigma: tuple[tuple[float, float], tuple[float, float]],
) -> dict[int, dict[str, Any]]:
    horizon_metrics: dict[int, dict[str, Any]] = {}
    state_level = manifest["state_level"]
    whitener = symmetric_inverse_sqrt(sigma) if state_level in ("S1", "S2") else None

    def normalized_prediction(row: dict[str, Any], gauge: Any, pred_state: np.ndarray) -> np.ndarray:
        if whitener is None:
            raise ValueError("normalized prediction requires a whitened state level")
        tone = row["u_t"].get("physical_tone_index")
        if tone is None:
            tone = row["declared"].get("analysis_tone_index") or 0
        z = complex(float(pred_state[0]), float(pred_state[1]))
        centered = z - gauge.complex_anchor_alpha[int(tone)]
        zn = whitener @ np.array([centered.real, centered.imag], dtype=float)
        return np.array([float(zn[0]), float(zn[1]), float(pred_state[2])], dtype=float)

    def next_x(
        group: list[dict[str, Any]],
        start_index: int,
        step: int,
        predicted_states: list[np.ndarray],
        gauge: Any,
    ) -> np.ndarray:
        if state_level == "S0":
            return predicted_states[-1].reshape(1, -1)
        if state_level == "S1":
            return normalized_prediction(group[start_index + step + 1], gauge, predicted_states[-1]).reshape(1, -1)
        delay = manifest.get("delay_length") or 2
        history: list[np.ndarray] = []
        for offset in range(delay):
            absolute_index = start_index + step + 1 - offset
            predicted_offset = step - offset
            if predicted_offset >= 0:
                history.append(normalized_prediction(group[absolute_index], gauge, predicted_states[predicted_offset]))
            else:
                history.append(state_vector(group, absolute_index, "S1", gauge, sigma)[:3])
        controls = []
        for offset in range(delay - 1):
            absolute_index = start_index + step - offset
            controls.append(executed_control_vector(group[absolute_index]))
        return np.concatenate(history + controls).reshape(1, -1)

    for horizon in HORIZONS:
        y_true = []
        y_pred = []
        y_base = []
        target_rows = []
        for group in _grouped(test_rows):
            for i in range(0, len(group) - horizon):
                if manifest["state_level"] == "S2" and manifest.get("delay_length") is not None and i < manifest["delay_length"] - 1:
                    continue
                gauge = gauges[group[i]["session_index"]]
                x = state_vector(group, i, manifest["state_level"], gauge, sigma, manifest.get("delay_length")).reshape(1, -1)
                x_base = x.copy()
                pred_state = None
                base_state = None
                predicted_states: list[np.ndarray] = []
                baseline_states: list[np.ndarray] = []
                for step in range(horizon):
                    target = group[i + step + 1]
                    pred_state = fitted.predict(x, [target])[0]
                    base_state = baseline_fit.predict(x_base, [target])[0]
                    predicted_states.append(pred_state)
                    baseline_states.append(base_state)
                    if step < horizon - 1:
                        x = next_x(group, i, step, predicted_states, gauge)
                        x_base = next_x(group, i, step, baseline_states, gauge)
                if pred_state is None or base_state is None:
                    continue
                y_true.append(state_vector(group, i + horizon, "S0", gauge, sigma)[:3])
                y_pred.append(pred_state)
                y_base.append(base_state)
                target_rows.append(group[i + horizon])
        yt = np.vstack(y_true)
        yp = np.vstack(y_pred)
        yb = np.vstack(y_base)
        horizon_metrics[horizon] = {
            "target_rows": target_rows,
            "y_true": yt,
            "y_pred": yp,
            "y_baseline": yb,
            "nrmse": nrmse(yt, yp),
            "baseline_nrmse": nrmse(yt, yb),
            "nrmse_gain": _gain(yt, yp, yb),
            "complex_correlation": complex_corr(yt, yp),
        }
    return horizon_metrics


def _route_transfer(rows: list[dict[str, Any]], manifest: dict[str, Any], gauges: dict[int, Any], sigma: tuple[tuple[float, float], tuple[float, float]]) -> dict[str, Any]:
    result = {}
    train_rows = [row for row in rows if row["split"] in ("train", "validation")]
    test_rows = split_rows(rows, "test")
    for source, target in (("v4s5", "v2s3"), ("v2s3", "v4s5")):
        source_train = [row for row in train_rows if row["route"] == source]
        target_train = [row for row in train_rows if row["route"] == target]
        target_test = [row for row in test_rows if row["route"] == target]
        x_train, y_train, fit_rows, _ = _dataset(source_train, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
        x_base_train, y_base_train, base_fit_rows, _ = _dataset(target_train, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
        x_test, y_test, target_rows, _ = _dataset(target_test, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
        fitted = fit_operator(manifest["operator_class"], x_train, y_train, fit_rows, manifest["regularization"])
        pred = fitted.predict(x_test, target_rows)
        baseline_name, baseline, _ = _strongest_baseline_predictions(x_base_train, y_base_train, base_fit_rows, x_test, y_test, target_rows)
        bootstrap = hierarchical_bootstrap_gain(target_rows, y_test, pred, baseline, manifest["bootstrap_seeds"]["route_transfer"])
        result[f"{source}_to_{target}"] = {
            "baseline_operator": baseline_name,
            "gain_distribution": _packet_gain_distribution(target_rows, y_test, pred, baseline),
            "bootstrap": bootstrap,
            "lower_gain": bootstrap["lower_95_bound"],
            "complex_correlation": complex_corr(y_test, pred),
        }
    return result


def _drive_off(rows: list[dict[str, Any]], manifest: dict[str, Any], pred: np.ndarray, base: np.ndarray, y_true: np.ndarray, target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_test_rows = [row for row in rows if row["split"] == "test" and row["stage"] == "trajectory"]
    raw_groups = {}
    for row in raw_test_rows:
        raw_groups.setdefault((row["session_index"], row.get("packet_id")), []).append(row)
    post_by_position: dict[int, list[dict[str, Any]]] = {}
    sham_by_position: dict[int, list[dict[str, Any]]] = {}

    def amplitude(row: dict[str, Any]) -> float:
        return float(np.hypot(row["r_t"]["lockin_I"], row["r_t"]["lockin_Q"]))

    def packet_type(row: dict[str, Any]) -> str:
        packet = str(row.get("packet_id") or "")
        return packet.rsplit(":", 1)[-1] if ":" in packet else packet

    def tone(row: dict[str, Any]) -> int | None:
        return row["u_t"].get("physical_tone_index") if row["u_t"].get("physical_tone_index") is not None else row["u_t"].get("analysis_tone_index")

    def matched_packet_type(row: dict[str, Any]) -> str | None:
        return row.get("matched_packet_type") or row.get("declared", {}).get("matched_packet_type")

    def matching_key(row: dict[str, Any], sender_off_position: int) -> tuple[tuple[str, Any], ...]:
        items: list[tuple[str, Any]] = [
            ("route", row["route"]),
            ("physical_tone", tone(row)),
            ("sender_off_position", sender_off_position),
            ("reboot_block", row["reboot_block"]),
        ]
        packet_match = matched_packet_type(row)
        if packet_match is not None:
            items.append(("packet_type", packet_match))
        return tuple(items)

    def key_payload(key: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
        return {name: value for name, value in key}

    def keyed_for_bootstrap(source_rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
        keyed_rows = []
        for row in source_rows:
            copy = dict(row)
            copy["packet_id"] = f"{row[key_name]}:{row.get('packet_id')}"
            keyed_rows.append(copy)
        return keyed_rows

    for row in raw_test_rows:
        if row["u_t"].get("executed_mode") == "CARRIER_OFF_SHAM":
            sham_offset = sum(
                1
                for other in raw_groups[(row["session_index"], row.get("packet_id"))]
                if other["slot_index"] <= row["slot_index"]
            )
            sham_row = dict(row)
            key = matching_key(sham_row, sham_offset)
            sham_row["matching_key"] = key_payload(key)
            sham_row["matching_key_id"] = repr(key)
            sham_by_position.setdefault(sham_offset, []).append(sham_row)

    for packet in raw_groups.values():
        packet = sorted(packet, key=lambda row: row["slot_index"])
        if not packet:
            continue
        driven = [i for i, row in enumerate(packet) if row["u_t"]["drive_on"]]
        if not driven:
            continue
        final_drive = max(driven)
        off_rows = [row for row in packet[final_drive + 1 :] if not row["u_t"]["drive_on"]]
        for offset, row in enumerate(off_rows[:7], start=1):
            post_row = dict(row)
            post_row["post_drive_packet_type"] = packet_type(row)
            key = matching_key(post_row, offset)
            post_row["matching_key"] = key_payload(key)
            post_row["matching_key_id"] = repr(key)
            post_by_position.setdefault(offset, []).append(post_row)

    def off_transition_dataset(source_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        x_parts = []
        y_parts = []
        out_rows = []
        for group in _grouped(source_rows):
            for i in range(len(group) - 1):
                if not group[i]["u_t"]["drive_on"] and not group[i + 1]["u_t"]["drive_on"]:
                    x_parts.append(np.array([[group[i]["r_t"]["lockin_I"], group[i]["r_t"]["lockin_Q"], group[i]["r_t"]["ring_osc_period"]]], dtype=float)[0])
                    y_parts.append(np.array([[group[i + 1]["r_t"]["lockin_I"], group[i + 1]["r_t"]["lockin_Q"], group[i + 1]["r_t"]["ring_osc_period"]]], dtype=float)[0])
                    out_rows.append(group[i + 1])
        if not x_parts:
            return np.empty((0, 3)), np.empty((0, 3)), []
        return np.vstack(x_parts), np.vstack(y_parts), out_rows

    train_x, train_y, train_decay_rows = off_transition_dataset([row for row in rows if row["split"] in ("train", "validation")])
    test_x, test_y, test_decay_rows = off_transition_dataset(raw_test_rows)
    if len(train_x) and len(test_x):
        decay = fit_operator("O1_SHARED_COMPLEX_AFFINE", train_x, train_y, train_decay_rows)
        decay_pred = decay.predict(test_x, test_decay_rows)
        baseline_scores = []
        for baseline_name in ("O0_TRAINING_MEAN", "O0_RETURN_TO_BASELINE", "O0_LAST_VALUE"):
            baseline_fit = fit_operator(baseline_name, train_x, train_y, train_decay_rows)
            baseline_pred = baseline_fit.predict(test_x, test_decay_rows)
            baseline_scores.append((baseline_name, baseline_pred, nrmse(test_y, baseline_pred)))
        baseline_name, decay_base, _ = min(baseline_scores, key=lambda item: item[2])
        decay_bootstrap = hierarchical_bootstrap_gain(test_decay_rows, test_y, decay_pred, decay_base, manifest["bootstrap_seeds"]["bootstrap"] + 9)
        decay_gain = _gain(test_y, decay_pred, decay_base)
    else:
        baseline_name = "NO_ZERO_INPUT_TEST_ROWS"
        decay_bootstrap = {
            "session_draws": 0,
            "nested_packet_draws": {},
            "bootstrap_iterations": 200,
            "gain_distribution": [],
            "lower_95_bound": 0.0,
        }
        decay_gain = 0.0
    position_payload: dict[str, Any] = {}
    consecutive = 0
    max_consecutive = 0
    for offset in sorted(post_by_position):
        post_rows = post_by_position[offset]
        sham_rows = sham_by_position.get(offset, [])
        sham_by_key: dict[str, list[dict[str, Any]]] = {}
        for row in sham_rows:
            sham_by_key.setdefault(row["matching_key_id"], []).append(row)
        matched_post_rows: list[dict[str, Any]] = []
        matched_sham_rows: list[dict[str, Any]] = []
        unmatched_rows = 0
        for row in post_rows:
            matching_sham = sham_by_key.get(row["matching_key_id"], [])
            if not matching_sham:
                unmatched_rows += 1
                continue
            matched_post_rows.append(row)
        for key in sorted({row["matching_key_id"] for row in matched_post_rows}):
            matched_sham_rows.extend(sham_by_key[key])
        post_values = np.array([amplitude(row) for row in matched_post_rows], dtype=float)
        sham_values = np.array([amplitude(row) for row in matched_sham_rows], dtype=float)
        post_bounds = hierarchical_bootstrap_bounds(
            keyed_for_bootstrap(matched_post_rows, "matching_key_id"),
            post_values,
            manifest["bootstrap_seeds"]["bootstrap"] + 100 + offset,
        )
        sham_bounds = hierarchical_bootstrap_bounds(
            keyed_for_bootstrap(matched_sham_rows, "matching_key_id"),
            sham_values,
            manifest["bootstrap_seeds"]["bootstrap"] + 200 + offset,
        )
        stratum_contrasts = []
        for key in sorted({row["matching_key_id"] for row in matched_post_rows}):
            key_post = [row for row in matched_post_rows if row["matching_key_id"] == key]
            key_sham = sham_by_key.get(key, [])
            post_mean = float(np.mean([amplitude(row) for row in key_post])) if key_post else 0.0
            sham_mean = float(np.mean([amplitude(row) for row in key_sham])) if key_sham else 0.0
            stratum_contrasts.append(
                {
                    "matching_key": key_post[0]["matching_key"] if key_post else {},
                    "post_drive_mean": post_mean,
                    "matched_sham_mean": sham_mean,
                    "contrast": post_mean - sham_mean,
                    "post_drive_count": len(key_post),
                    "matched_sham_count": len(key_sham),
                }
            )
        passes = bool(matched_post_rows and matched_sham_rows and unmatched_rows == 0 and post_bounds["lower_95_bound"] > sham_bounds["upper_95_bound"])
        position_payload[str(offset)] = {
            "matching_key_schema": ["route", "physical_tone", "sender_off_position", "reboot_block"],
            "matching_keys": [item["matching_key"] for item in stratum_contrasts],
            "stratum_contrasts": stratum_contrasts,
            "post_drive_session_ids": sorted({row["session_index"] for row in matched_post_rows}),
            "sham_session_ids": sorted({row["session_index"] for row in matched_sham_rows}),
            "packet_counts": {
                "post_drive": len({(row["session_index"], row.get("packet_id")) for row in matched_post_rows}),
                "matched_sham": len({(row["session_index"], row.get("packet_id")) for row in matched_sham_rows}),
            },
            "unmatched_row_count": unmatched_rows,
            "post_drive_distribution": post_values.tolist(),
            "matched_sham_distribution": sham_values.tolist(),
            "post_drive_bootstrap": post_bounds,
            "matched_sham_bootstrap": sham_bounds,
            "post_drive_lower_95": post_bounds["lower_95_bound"],
            "matched_sham_upper_95": sham_bounds["upper_95_bound"],
            "matched_strata": ["route", "physical_tone", "sender_off_position", "reboot_block"],
            "pass": passes,
        }
        if passes:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    return {
        "three_consecutive_lower_above_sham": max_consecutive >= 3,
        "position_bounds": position_payload,
        "post_drive_position_lowers": {offset: payload["post_drive_lower_95"] for offset, payload in position_payload.items()},
        "matched_sham_position_uppers": {offset: payload["matched_sham_upper_95"] for offset, payload in position_payload.items()},
        "zero_input_decay_operator": "O1_SHARED_COMPLEX_AFFINE_ON_SENDER_OFF_TRANSITIONS",
        "zero_input_decay_baseline": baseline_name,
        "zero_input_decay_gain": decay_gain,
        "zero_input_decay_gain_lower": decay_bootstrap["lower_95_bound"],
        "zero_input_decay_bootstrap": decay_bootstrap,
    }


def _confound(flag: bool, metric: float, threshold: float, comparison: str, confidence_interval: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "flag": bool(flag),
        "metric": float(metric),
        "threshold": float(threshold),
        "comparison_model": comparison,
        "comparison": comparison,
    }
    if confidence_interval is not None:
        payload["confidence_interval"] = confidence_interval
    return payload


def _order_family_diagnostic(
    y_true: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    target_rows: list[dict[str, Any]],
    full_gain: float,
) -> tuple[dict[str, Any], float]:
    families: dict[str, Any] = {}
    max_loss = 0.0
    for family in ORDER_FAMILIES:
        idx = [i for i, row in enumerate(target_rows) if row["u_t"].get("executed_order_family") != family]
        if not idx:
            continue
        gain_without = _gain(y_true[idx], pred[idx], baseline[idx])
        loss = full_gain - gain_without
        max_loss = max(max_loss, loss)
        families[family] = {
            "gain_with_family_held_out": gain_without,
            "performance_loss": loss,
            "held_out_complete_stratum": True,
        }
    return families, max_loss


def _chronology_diagnostic(y_true: np.ndarray, pred: np.ndarray, baseline: np.ndarray, target_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    blocks: dict[str, Any] = {}
    gains = []
    for chronology in sorted({row["session_chronology"] for row in target_rows}):
        idx = [i for i, row in enumerate(target_rows) if row["session_chronology"] == chronology]
        gain = _gain(y_true[idx], pred[idx], baseline[idx])
        gains.append(gain)
        blocks[str(chronology)] = {"gain": gain, "row_count": len(idx)}
    spread = max(gains) - min(gains) if gains else 0.0
    return blocks, spread


def _session_identity_diagnostic(
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    validation_rows: list[dict[str, Any]],
    y_validation: np.ndarray,
    validation_baseline: np.ndarray,
    model_gain: float,
) -> dict[str, Any]:
    known_rows = train_rows + validation_rows
    known_y = np.vstack([y_train, y_validation])
    known_baseline = np.vstack([np.repeat(np.mean(y_train, axis=0).reshape(1, -1), len(y_train), axis=0), validation_baseline])
    predictions = np.zeros_like(known_y)
    per_session: dict[str, Any] = {}
    held_out_sessions = sorted({int(row["session_index"]) for row in known_rows})
    for session in held_out_sessions:
        train_idx = [i for i, row in enumerate(known_rows) if int(row["session_index"]) != session]
        holdout_idx = [i for i, row in enumerate(known_rows) if int(row["session_index"]) == session]
        if not train_idx or not holdout_idx:
            continue
        labels = [f"session:{known_rows[i]['session_index']}" for i in train_idx]
        target_labels = [f"session:{known_rows[i]['session_index']}" for i in holdout_idx]
        holdout_pred, vocab = _fit_label_predictor(labels, known_y[train_idx], target_labels)
        predictions[holdout_idx] = holdout_pred
        per_session[str(session)] = {
            "nrmse": nrmse(known_y[holdout_idx], holdout_pred),
            "row_count": len(holdout_idx),
            "session_id_seen_during_fit": f"session:{session}" in vocab,
        }
    bootstrap = hierarchical_bootstrap_gain(known_rows, known_y, predictions, known_baseline, 409)
    gain = _gain(known_y, predictions, known_baseline)
    flag = gain >= model_gain - 0.01 and gain > 0.05
    payload = _confound(
        flag,
        gain,
        model_gain - 0.01,
        "leave-one-known-session-identity predictor on training/validation; unknown sealed-test identities not substituted",
        {
            "lower_95_gain": bootstrap["lower_95_bound"],
            "gain_distribution": bootstrap["gain_distribution"],
            "session_draws": bootstrap["session_draws"],
            "nested_packet_draws": bootstrap["nested_packet_draws"],
            "sealed_test_identity_substitution": False,
        },
    )
    payload.update(
        {
            "held_out_session_ids": held_out_sessions,
            "per_session_nrmse": per_session,
            "aggregate_gain": gain,
            "selected_dynamic_model_gain": model_gain,
            "sealed_test_identity_substitution": False,
        }
    )
    return payload


def _session_identity_gain(
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    validation_rows: list[dict[str, Any]],
    y_validation: np.ndarray,
    validation_baseline: np.ndarray,
) -> float:
    return _session_identity_diagnostic(train_rows, y_train, validation_rows, y_validation, validation_baseline, 0.0)["metric"]


def _confounds(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    validation_baseline: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    y_true: np.ndarray,
    target_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    model_gain = _gain(y_true, pred, baseline)
    physical = _predictive_label_diagnostic(
        train_rows,
        y_train,
        target_rows,
        y_true,
        baseline,
        "physical_tone",
        301,
        "physical-tone-indexed held-out predictor",
    )
    execution = _predictive_label_diagnostic(
        train_rows,
        y_train,
        target_rows,
        y_true,
        baseline,
        "execution_position",
        302,
        "execution-position-indexed held-out predictor",
    )
    executed_order = _predictive_label_diagnostic(train_rows, y_train, target_rows, y_true, baseline, "executed_order", 303, "executed-order predictor")
    declared_order = _predictive_label_diagnostic(train_rows, y_train, target_rows, y_true, baseline, "declared_order", 304, "declared-order predictor")
    order_sham = _order_label_sham_diagnostic(train_rows, y_train, target_rows, y_true, baseline, declared_order)
    time_fit = fit_operator("O0_TIME_INDEX", y_train, y_train, train_rows)
    time_pred = time_fit.predict(y_true, target_rows)
    time_ratio = nrmse(y_true, time_pred) / max(nrmse(y_true, pred), 1e-9)
    families, family_loss = _order_family_diagnostic(y_true, pred, baseline, target_rows, model_gain)
    chronology_blocks, chronology_spread = _chronology_diagnostic(y_true, pred, baseline, target_rows)
    tone_execution_delta = abs(physical["gain_vs_strongest_baseline"] - execution["gain_vs_strongest_baseline"])
    return {
        "physical_tone_indexed_performance": {**_confound(False, physical["gain_vs_strongest_baseline"], 0.05, physical["comparison_model"], physical["blocked_confidence_interval"]), "diagnostic": physical},
        "execution_position_indexed_performance": {**_confound(False, execution["gain_vs_strongest_baseline"], 0.05, execution["comparison_model"], execution["blocked_confidence_interval"]), "diagnostic": execution},
        "tone_vs_execution_position_disagreement": _confound(
            model_gain >= 0.10 and tone_execution_delta >= 10.0,
            tone_execution_delta,
            10.0,
            "material gain disagreement between physical-tone-indexed and execution-position-indexed predictors",
        ),
        "order_label_sham_predicts_comparably": order_sham,
        "time_index_within_five_percent": _confound(model_gain >= 0.10 and time_ratio <= 1.05, time_ratio, 1.05, "O0_TIME_INDEX held-out NRMSE / selected dynamic model held-out NRMSE"),
        "single_order_family_dependence": {**_confound(family_loss >= 0.10, family_loss, 0.10, "selected model performance loss with each complete order family held out"), "held_out_families": families},
        "single_chronology_position_dependence": {**_confound(chronology_spread >= 0.50, chronology_spread, 0.50, "selected model performance spread across frozen chronology positions"), "chronology_blocks": chronology_blocks},
        "session_lookup_dominance": _session_identity_diagnostic(train_rows, y_train, validation_rows, y_validation, validation_baseline, model_gain),
    }


def evaluate_sealed(custody: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(custody, manifest)
    rows = flatten_custody(custody)
    train_rows = split_rows(rows, "train")
    validation_rows = split_rows(rows, "validation")
    test_rows = split_rows(rows, "test")
    gauges = estimate_session_gauges(rows)
    sigma = training_global_covariance([row for row in train_rows if row["stage"] == "preamble"])
    x_train, y_train, fit_rows, _ = _dataset(train_rows, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
    x_validation, y_validation, validation_target_rows, _ = _dataset(validation_rows, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
    x_test, y_test, target_rows, _ = _dataset(test_rows, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
    fitted = fit_operator(manifest["operator_class"], x_train, y_train, fit_rows, manifest["regularization"])
    pred = fitted.predict(x_test, target_rows)
    baseline_name, baseline, _ = _strongest_baseline_predictions(x_train, y_train, fit_rows, x_test, y_test, target_rows)
    _, validation_baseline, _ = _strongest_baseline_predictions(x_train, y_train, fit_rows, x_validation, y_validation, validation_target_rows)
    horizon = _rollout_metrics(manifest, fitted, fit_operator(baseline_name, x_train, y_train, fit_rows), test_rows, gauges, sigma)
    one_gains = _entity_gains(target_rows, y_test, pred, baseline, "session_index")
    eight = horizon[8]
    eight_gains = _entity_gains(eight["target_rows"], eight["y_true"], eight["y_pred"], eight["y_baseline"], "session_index")
    route_summary = summarize(target_rows, y_test, pred)["per_route"]
    route_gates = _route_gate_payload(
        target_rows,
        y_test,
        pred,
        baseline,
        eight,
        fit_rows,
        y_train,
        validation_target_rows,
        y_validation,
        validation_baseline,
        manifest["bootstrap_seeds"]["bootstrap"] + 500,
    )
    transfer = _route_transfer(rows, manifest, gauges, sigma)
    one_step_gain = _gain(y_test, pred, baseline)
    one_bootstrap = hierarchical_bootstrap_gain(target_rows, y_test, pred, baseline, manifest["bootstrap_seeds"]["bootstrap"])
    eight_bootstrap = hierarchical_bootstrap_gain(
        eight["target_rows"],
        eight["y_true"],
        eight["y_pred"],
        eight["y_baseline"],
        manifest["bootstrap_seeds"]["bootstrap"] + 1,
    )
    session_identity_gain = _session_identity_gain(fit_rows, y_train, validation_target_rows, y_validation, validation_baseline)
    predictive_metrics = {
        "one_step_nrmse_gain": one_step_gain,
        "eight_step_nrmse_gain": eight["nrmse_gain"],
        "one_step_bootstrap_lower": one_bootstrap["lower_95_bound"],
        "eight_step_bootstrap_lower": eight_bootstrap["lower_95_bound"],
        "one_step_bootstrap": one_bootstrap,
        "eight_step_bootstrap": eight_bootstrap,
        "route_v4s5_complex_corr": route_summary["v4s5"]["complex_correlation"],
        "route_v2s3_complex_corr": route_summary["v2s3"]["complex_correlation"],
        "worst_session_delta_vs_baseline": min(_entity_gains(target_rows, y_test, pred, baseline, "session_index")),
        "session_lookup_gain_margin": one_step_gain - session_identity_gain,
    }
    result = {
        "schema_id": "CAT_CAS_PHASE6B6_ANALYSIS_RESULT_V1",
        "analysis_choice_sha256": manifest["analysis_choice_sha256"],
        "selected_model": {
            "state_level": manifest["state_level"],
            "delay_length": manifest["delay_length"],
            "operator_class": manifest["operator_class"],
            "regularization": manifest["regularization"],
            "o4_lift": manifest["o4_lift"],
            "validation_score": manifest["validation_score"],
        },
        "horizons": {
            str(h): {
                "nrmse": metrics["nrmse"],
                "baseline_nrmse": metrics["baseline_nrmse"],
                "nrmse_gain": metrics["nrmse_gain"],
                "complex_correlation": metrics["complex_correlation"],
            }
            for h, metrics in horizon.items()
        },
        "summary": summarize(target_rows, y_test, pred),
        "baseline_operator": baseline_name,
        "predictive_metrics": predictive_metrics,
        "route_local_gates": route_gates,
        "route_transfer": transfer,
        "drive_off": _drive_off(rows, manifest, pred, baseline, y_test, target_rows),
        "confounds": _confounds(
            rows,
            fit_rows,
            validation_target_rows,
            y_train,
            y_validation,
            validation_baseline,
            pred,
            baseline,
            y_test,
            target_rows,
        ),
        "within_route_pass": any(gate["pass"] for gate in route_gates.values()),
    }
    result["adjudication"] = derive_adjudication(result)
    result = _plain(result)
    result["result_sha256"] = digest(result)
    return result
