"""End-to-end Phase 6B.6 software analysis pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from analysis.adjudication import derive_adjudication
from analysis.metrics import bootstrap_gain_lower, complex_corr, nrmse, packet_groups, summarize
from analysis.observations import assert_test_sealed, flatten_custody, split_rows
from analysis.operators import (
    OPERATOR_LADDER,
    analysis_contract_digest,
    deterministic_seed,
    fit_operator,
)
from analysis.state import estimate_session_gauges, state_vector, training_global_covariance
from contracts.contract import DELAY_CANDIDATES, HORIZONS, O4_FIXED_LIFTS, REGULARIZATION_LADDER, digest


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
    candidates: list[dict[str, Any]] = []
    for operator_class in SCIENCE_OPERATORS:
        regs = REGULARIZATION_LADDER if operator_class == "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR" else (0.0,)
        for reg in regs:
            fitted = fit_operator(operator_class, x_train, y_train, fit_rows, reg)
            score = nrmse(y_val, fitted.predict(x_val, val_target_rows))
            candidates.append(
                {
                    "state_level": state_level,
                    "delay_length": delay,
                    "operator_class": operator_class,
                    "regularization": reg,
                    "o4_lift": O4_FIXED_LIFTS if operator_class == "O4_FIXED_PHASE_NATIVE_LIFT_REGULARIZED_LINEAR" else None,
                    "validation_score": score,
                }
            )
    best = min(candidates, key=lambda item: item["validation_score"])
    qualifying = [item for item in candidates if item["validation_score"] <= best["validation_score"] * 1.02]
    for operator_class in SCIENCE_OPERATORS:
        family = [item for item in qualifying if item["operator_class"] == operator_class]
        if family:
            return min(family, key=lambda item: item["regularization"])
    raise AssertionError("unreachable")


def select_on_validation(custody: dict[str, Any]) -> dict[str, Any]:
    rows = flatten_custody(custody)
    assert_test_sealed(rows, allow_test=False)
    train_rows = split_rows(rows, "train")
    validation_rows = split_rows(rows, "validation")
    gauges = estimate_session_gauges(rows)
    sigma = training_global_covariance([row for row in train_rows if row["stage"] == "preamble"])
    state_winners = []
    for state_level in ("S0", "S1", "S2"):
        candidates = [(level, delay) for level, delay in _state_candidates() if level == state_level]
        scored = [_best_operator_for_state(train_rows, validation_rows, level, delay, gauges, sigma) for level, delay in candidates]
        best = min(scored, key=lambda item: item["validation_score"])
        state_winners.append(best)
        if best["validation_score"] <= 1e-4:
            break
    selected = state_winners[-1]
    if selected["state_level"] == "S2":
        s2 = [item for item in state_winners if item["state_level"] == "S2"]
        best_s2 = min(s2, key=lambda item: item["validation_score"])
        selected = min(
            [item for item in s2 if item["validation_score"] <= best_s2["validation_score"] * 1.02],
            key=lambda item: item["delay_length"] or 0,
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


def _rollout_metrics(
    manifest: dict[str, Any],
    fitted: Any,
    baseline_fit: Any,
    test_rows: list[dict[str, Any]],
    gauges: dict[int, Any],
    sigma: tuple[tuple[float, float], tuple[float, float]],
) -> dict[int, dict[str, Any]]:
    horizon_metrics: dict[int, dict[str, Any]] = {}

    def normalized_prediction(row: dict[str, Any], gauge: Any, pred_state: np.ndarray) -> np.ndarray:
        tone = row["u_t"].get("physical_tone_index")
        if tone is None:
            tone = row["declared"].get("analysis_tone_index") or 0
        z = complex(float(pred_state[0]), float(pred_state[1]))
        centered = z - gauge.complex_anchor_alpha[int(tone)]
        cov = np.array(sigma, dtype=float)
        inv_sqrt = 1.0 / max(float(np.sqrt(np.trace(cov) / 2.0)), 1e-9)
        zn = centered * inv_sqrt
        return np.array([zn.real, zn.imag, float(pred_state[2])], dtype=float)

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
                pred_state = None
                base_state = None
                current_rows = [group[i + step] for step in range(horizon)]
                for step in range(horizon):
                    target = group[i + step + 1]
                    pred_state = fitted.predict(x, [target])[0]
                    base_state = baseline_fit.predict(x, [target])[0]
                    if step < horizon - 1:
                        if manifest["state_level"] == "S0":
                            x = pred_state.reshape(1, -1)
                        elif manifest["state_level"] == "S1":
                            x = pred_state.reshape(1, -1)
                        else:
                            history = [pred_state]
                            delay = manifest.get("delay_length") or 2
                            history = [normalized_prediction(target, gauge, pred_state)]
                            for j in range(1, delay):
                                prior_index = i + step + 1 - j
                                if prior_index < 0:
                                    break
                                history.append(state_vector(group, prior_index, "S0", gauge, sigma)[:3])
                            while len(history) < delay:
                                history.append(history[-1])
                            controls = []
                            for j in range(delay - 1):
                                row = current_rows[max(0, step - j)]
                                from analysis.state import executed_control_vector

                                controls.append(executed_control_vector(row))
                            x = np.concatenate(history[:delay] + controls[: delay - 1]).reshape(1, -1)
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
        target_test = [row for row in test_rows if row["route"] == target]
        x_train, y_train, fit_rows, _ = _dataset(source_train, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
        x_test, y_test, target_rows, _ = _dataset(target_test, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
        fitted = fit_operator(manifest["operator_class"], x_train, y_train, fit_rows, manifest["regularization"])
        pred = fitted.predict(x_test, target_rows)
        _, baseline, _ = _strongest_baseline_predictions(x_train, y_train, fit_rows, x_test, y_test, target_rows)
        gains = _packet_gain_distribution(target_rows, y_test, pred, baseline)
        source_mean = float(np.mean(y_train[:, 0]))
        target_mean = float(np.mean(y_test[:, 0]))
        if source_mean * target_mean < 0.0:
            gains = [-abs(gain) for gain in gains]
        result[f"{source}_to_{target}"] = {
            "gain_distribution": gains,
            "lower_gain": bootstrap_gain_lower(gains, manifest["bootstrap_seeds"]["route_transfer"]),
            "complex_correlation": complex_corr(y_test, pred),
        }
    return result


def _drive_off(rows: list[dict[str, Any]], manifest: dict[str, Any], pred: np.ndarray, base: np.ndarray, y_true: np.ndarray, target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    off_distances = []
    sham_distances = []
    packet_passes = []
    raw_test_rows = [row for row in rows if row["split"] == "test" and row["stage"] == "trajectory"]
    raw_groups = {}
    for row in raw_test_rows:
        raw_groups.setdefault((row["session_index"], row.get("packet_id")), []).append(row)
    for packet in raw_groups.values():
        packet = sorted(packet, key=lambda row: row["slot_index"])
        if not packet:
            continue
        first_drive = next((i for i, row in enumerate(packet) if row["u_t"]["drive_on"]), None)
        if first_drive is None:
            continue
        off_rows = [row for row in packet[first_drive + 1 :] if not row["u_t"]["drive_on"]][:3]
        if len(off_rows) >= 3:
            off = [float(np.hypot(row["r_t"]["lockin_I"], row["r_t"]["lockin_Q"])) for row in off_rows]
            off_distances.extend(off)
            packet_passes.append(min(off) > 0.05)
        if any("SHAM" in row["u_t"]["executed_mode"] for row in packet):
            sham_distances.extend(float(np.hypot(row["r_t"]["lockin_I"], row["r_t"]["lockin_Q"])) for row in packet)
    if off_distances:
        sham_upper = max(sham_distances) if sham_distances else 0.0
        distance_gain = (float(np.mean(off_distances)) - sham_upper) / max(float(np.mean(off_distances)), 1e-9)
        gains = [distance_gain for _ in off_distances]
    else:
        gains = []
    return {
        "three_consecutive_lower_above_sham": bool(packet_passes) and sum(packet_passes) >= max(3, len(packet_passes) // 4),
        "post_drive_distance_lower": min(off_distances) if off_distances else 0.0,
        "sham_upper": max(sham_distances) if sham_distances else 0.0,
        "zero_input_decay_gain": float(np.mean(gains)) if gains else 0.0,
        "zero_input_decay_gain_lower": bootstrap_gain_lower(gains, manifest["bootstrap_seeds"]["bootstrap"] + 9),
    }


def _confounds(rows: list[dict[str, Any]], pred: np.ndarray, baseline: np.ndarray, session_lookup: np.ndarray, y_true: np.ndarray, target_rows: list[dict[str, Any]]) -> dict[str, bool]:
    model_gain = _gain(y_true, pred, baseline)
    session_gain = _gain(y_true, session_lookup, baseline)
    time_corr = abs(np.corrcoef([row["slot_index"] for row in target_rows], y_true[:, 0])[0, 1]) if len(target_rows) > 2 else 0.0
    route_means = {route: float(np.mean([y_true[i, 0] for i, row in enumerate(target_rows) if row["route"] == route])) for route in sorted({row["route"] for row in target_rows})}
    route_gap = max(route_means.values()) - min(route_means.values()) if len(route_means) > 1 else 0.0
    order_families = sorted({row["u_t"].get("executed_order_family") for row in target_rows if row["u_t"].get("executed_order_family")})
    order_counts = [sum(1 for row in target_rows if row["u_t"].get("executed_order_family") == family) for family in order_families]
    session_means = [float(np.mean([y_true[i, 0] for i, row in enumerate(target_rows) if row["session_index"] == session])) for session in sorted({row["session_index"] for row in target_rows})]
    within_session = [
        float(np.std([y_true[i, 0] for i, row in enumerate(target_rows) if row["session_index"] == session]))
        for session in sorted({row["session_index"] for row in target_rows})
    ]
    session_structure = (float(np.std(session_means)) / max(float(np.mean(within_session)), 1e-9)) if session_means else 0.0
    return {
        "tone_vs_execution_position_disagreement": route_gap > 1.0 and model_gain < 0.05,
        "order_label_sham_predicts_comparably": model_gain < 0.05
        and any("SHAM" in str(row["u_t"].get("executed_mode")) and abs(y_true[i, 0]) > 0.25 for i, row in enumerate(target_rows)),
        "time_index_within_five_percent": time_corr > 0.98,
        "single_order_family_dependence": bool(order_counts) and max(order_counts) / max(sum(order_counts), 1) > 0.90,
        "single_chronology_position_dependence": time_corr > 0.995,
        "session_lookup_dominance": (
            (session_gain >= model_gain - 0.01 and session_gain > 0.05 and model_gain < 0.10)
            or (session_structure > 10.0 and model_gain < 0.10)
        ),
    }


def evaluate_sealed(custody: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(custody, manifest)
    rows = flatten_custody(custody)
    train_rows = split_rows(rows, "train")
    test_rows = split_rows(rows, "test")
    gauges = estimate_session_gauges(rows)
    sigma = training_global_covariance([row for row in train_rows if row["stage"] == "preamble"])
    x_train, y_train, fit_rows, _ = _dataset(train_rows, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
    x_test, y_test, target_rows, _ = _dataset(test_rows, manifest["state_level"], manifest.get("delay_length"), gauges, sigma)
    fitted = fit_operator(manifest["operator_class"], x_train, y_train, fit_rows, manifest["regularization"])
    pred = fitted.predict(x_test, target_rows)
    baseline_name, baseline, _ = _strongest_baseline_predictions(x_train, y_train, fit_rows, x_test, y_test, target_rows)
    session_lookup = fit_operator("O0_SESSION_LOOKUP_DIAGNOSTIC", x_train, y_train, fit_rows).predict(x_test, target_rows)
    horizon = _rollout_metrics(manifest, fitted, fit_operator(baseline_name, x_train, y_train, fit_rows), test_rows, gauges, sigma)
    one_gains = _entity_gains(target_rows, y_test, pred, baseline, "session_index") + _packet_gain_distribution(target_rows, y_test, pred, baseline)
    eight = horizon[8]
    eight_gains = _entity_gains(eight["target_rows"], eight["y_true"], eight["y_pred"], eight["y_baseline"], "session_index")
    eight_gains += _packet_gain_distribution(eight["target_rows"], eight["y_true"], eight["y_pred"], eight["y_baseline"])
    route_summary = summarize(target_rows, y_test, pred)["per_route"]
    transfer = _route_transfer(rows, manifest, gauges, sigma)
    predictive_metrics = {
        "one_step_nrmse_gain": _gain(y_test, pred, baseline),
        "eight_step_nrmse_gain": eight["nrmse_gain"],
        "one_step_bootstrap_lower": bootstrap_gain_lower(one_gains, manifest["bootstrap_seeds"]["bootstrap"]),
        "eight_step_bootstrap_lower": bootstrap_gain_lower(eight_gains, manifest["bootstrap_seeds"]["bootstrap"] + 1),
        "route_v4s5_complex_corr": route_summary["v4s5"]["complex_correlation"],
        "route_v2s3_complex_corr": route_summary["v2s3"]["complex_correlation"],
        "worst_session_delta_vs_baseline": min(_entity_gains(target_rows, y_test, pred, baseline, "session_index")),
        "session_lookup_gain_margin": (nrmse(y_test, session_lookup) - nrmse(y_test, pred)) / max(nrmse(y_test, session_lookup), 1e-9),
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
        "route_transfer": transfer,
        "drive_off": _drive_off(rows, manifest, pred, baseline, y_test, target_rows),
        "confounds": _confounds(rows, pred, baseline, session_lookup, y_test, target_rows),
        "within_route_pass": route_summary["v4s5"]["complex_correlation"] >= 0.80 or route_summary["v2s3"]["complex_correlation"] >= 0.80,
    }
    result["adjudication"] = derive_adjudication(result)
    result = _plain(result)
    result["result_sha256"] = digest(result)
    return result
