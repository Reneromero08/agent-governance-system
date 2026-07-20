from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BASE_AUDIT = HERE / "operator_dimension_audit.py"
spec = importlib.util.spec_from_file_location("operator_dimension_audit", BASE_AUDIT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not import base audit from {BASE_AUDIT}")
oda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oda)

RESULT_CLASS_UNRESOLVED = "FAMILY10H_SECOND_OPERATOR_DIMENSION_UNRESOLVED_RETROSPECTIVE"
RESULT_CLASS_NOT_OBSERVED = "FAMILY10H_SECOND_OPERATOR_DIMENSION_NOT_OBSERVED_RETROSPECTIVE"
RESULT_CLASS_CANDIDATE = "FAMILY10H_SECOND_OPERATOR_DIMENSION_CANDIDATE_RETROSPECTIVE"
CLAIM_CANDIDATE = "PUBLIC_POST_SOURCE_SECOND_OPERATOR_DIMENSION_CANDIDATE_RETROSPECTIVE"

CATEGORICAL_FACTORS = ["session", "replicate", "mapping", "delay_label", "source_order"]
HOLDOUT_FACTORS = CATEGORICAL_FACTORS + ["q"]
BASE_GEOMETRY_COLUMNS = [
    "Y_query_A",
    "Y_query_B",
    "Y_query_A_then_B",
    "Y_query_B_then_A",
    "Y_query_sham",
    "Y_carrier_off",
    "F_query_A_then_B_both_active",
    "F_query_A_then_B_A_active_B_dummy",
    "F_query_A_then_B_A_dummy_B_active",
    "F_query_A_then_B_both_dummy",
    "F_query_B_then_A_both_active",
    "F_query_B_then_A_A_active_B_dummy",
    "F_query_B_then_A_A_dummy_B_active",
    "F_query_B_then_A_both_dummy",
]


def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def canonical_digest(data: Any) -> str:
    return oda.canonical_digest(data)


def safe_corr(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float | None:
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    if len(aa) < 2 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.array(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "median": float(np.median(arr)),
    }


def sign_source_order(cell: dict[str, Any]) -> float:
    return 1.0 if cell["source_order"] == "A_then_B" else -1.0


def sign_mapping(cell: dict[str, Any]) -> float:
    return 1.0 if cell["mapping"] == "map0" else -1.0


def enrich_factorial_observables(raw_rows: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    factorial_by_cell: dict[tuple[Any, ...], dict[str, dict[str, dict[str, Any]]]] = {}
    for row in raw_rows:
        if row["matrix_block"] != "factorial_nonadditivity" or row["source_off_control"]:
            continue
        key = tuple(row[factor] for factor in oda.CELL_FACTORS)
        factorial_by_cell.setdefault(key, {}).setdefault(row["query"], {})[row["factorial_arm"]] = row

    missing: list[str] = []
    c2d_nonzero_cells = 0
    for cell in cells:
        key = tuple(cell[factor] for factor in oda.CELL_FACTORS)
        by_query = factorial_by_cell.get(key, {})
        for ordered_query in ["query_A_then_B", "query_B_then_A"]:
            arms = by_query.get(ordered_query, {})
            missing_arms = [arm for arm in oda.FACTORIAL_ARMS if arm not in arms]
            if missing_arms:
                missing.append(f"{key!r} {ordered_query} missing {missing_arms!r}")
                continue
            prefix = f"F_{ordered_query}"
            for arm, row in arms.items():
                cell[f"{prefix}_{arm}_c2d"] = float(row["change_to_dirty"])
                cell[f"{prefix}_{arm}_i_c2d"] = 1.0 if float(row["change_to_dirty"]) > 0.0 else 0.0
                cycles = max(float(row["cpu_cycles"]), 1.0)
                cell[f"{prefix}_{arm}_dirty_per_cycle"] = float(row["dirty_probe_response"]) / cycles
                cell[f"{prefix}_{arm}_c2d_per_cycle"] = float(row["change_to_dirty"]) / cycles
            for channel, suffix in [
                ("dirty", ""),
                ("c2d", "_c2d"),
                ("i_c2d", "_i_c2d"),
                ("dirty_per_cycle", "_dirty_per_cycle"),
                ("c2d_per_cycle", "_c2d_per_cycle"),
            ]:
                both = cell[f"{prefix}_both_active{suffix}"]
                a_active = cell[f"{prefix}_A_active_B_dummy{suffix}"]
                b_active = cell[f"{prefix}_A_dummy_B_active{suffix}"]
                both_dummy = cell[f"{prefix}_both_dummy{suffix}"]
                cell[f"L_{channel}_{ordered_query}"] = a_active - b_active
                cell[f"T_{channel}_{ordered_query}"] = 0.25 * (both + a_active + b_active + both_dummy)
                cell[f"J_{channel}_{ordered_query}"] = both - a_active - b_active + both_dummy

        if all(f"J_c2d_{query}" in cell for query in ["query_A_then_B", "query_B_then_A"]):
            cell["K_C2D"] = cell["J_c2d_query_A_then_B"] - cell["J_c2d_query_B_then_A"]
            cell["K_I_C2D"] = cell["J_i_c2d_query_A_then_B"] - cell["J_i_c2d_query_B_then_A"]
            cell["K_L_dirty"] = cell["L_dirty_query_A_then_B"] - cell["L_dirty_query_B_then_A"]
            cell["K_T_dirty"] = cell["T_dirty_query_A_then_B"] - cell["T_dirty_query_B_then_A"]
            cell["K_L_C2D"] = cell["L_c2d_query_A_then_B"] - cell["L_c2d_query_B_then_A"]
            cell["K_T_C2D"] = cell["T_c2d_query_A_then_B"] - cell["T_c2d_query_B_then_A"]
            cell["K_L_dirty_per_cycle"] = (
                cell["L_dirty_per_cycle_query_A_then_B"] - cell["L_dirty_per_cycle_query_B_then_A"]
            )
            cell["K_T_dirty_per_cycle"] = (
                cell["T_dirty_per_cycle_query_A_then_B"] - cell["T_dirty_per_cycle_query_B_then_A"]
            )
            cell["K_L_C2D_per_cycle"] = (
                cell["L_c2d_per_cycle_query_A_then_B"] - cell["L_c2d_per_cycle_query_B_then_A"]
            )
            cell["K_T_C2D_per_cycle"] = (
                cell["T_c2d_per_cycle_query_A_then_B"] - cell["T_c2d_per_cycle_query_B_then_A"]
            )
            if cell["K_C2D"] != 0.0:
                c2d_nonzero_cells += 1

    return {
        "complete": not missing,
        "missing": missing,
        "factorial_group_count": len(factorial_by_cell),
        "c2d_nonzero_K_C2D_cells": c2d_nonzero_cells,
        "interpretation": "C2D and Hadamard factorial coordinates are evaluated as diagnostic public observables, with no promotion unless they survive scalar leakage, held-out strata, and clean operator-law checks.",
    }


def augment_cells(cells: list[dict[str, Any]]) -> None:
    for cell in cells:
        eps = 1.0
        source_sign = sign_source_order(cell)
        map_sign = sign_mapping(cell)
        cell["NormOrder"] = cell["D_order"] / max(abs(cell["S_order"]), 1.0)
        cell["LogOrder"] = math.log((cell["Y_query_A_then_B"] + eps) / (cell["Y_query_B_then_A"] + eps))
        cell["SourceAlignedDOrder"] = source_sign * cell["D_order"]
        cell["MapAlignedDOrder"] = map_sign * cell["D_order"]
        cell["SourceMapAlignedDOrder"] = source_sign * map_sign * cell["D_order"]
        cell["SourceAlignedLogOrder"] = source_sign * cell["LogOrder"]
        cell["MapAlignedLogOrder"] = map_sign * cell["LogOrder"]
        cell["SourceMapAlignedLogOrder"] = source_sign * map_sign * cell["LogOrder"]
        cell["JSum"] = cell["J_query_A_then_B"] + cell["J_query_B_then_A"]
        cell["JMean"] = 0.5 * cell["JSum"]
        cell["JHalfOrder"] = 0.5 * cell["J_order"]
        cell["SourceAlignedJOrder"] = source_sign * cell["J_order"]
        cell["MapAlignedJOrder"] = map_sign * cell["J_order"]
        cell["SourceMapAlignedJOrder"] = source_sign * map_sign * cell["J_order"]
        cell["NormJOrder"] = cell["J_order"] / max(abs(cell["S_order"]), 1.0)
        cell["AminusSham"] = cell["Y_query_A"] - cell["Y_query_sham"]
        cell["BminusSham"] = cell["Y_query_B"] - cell["Y_query_sham"]
        cell["ShamScalarDifference"] = cell["AminusSham"] - cell["BminusSham"]
        cell["OrderedVsSingleExcess"] = cell["S_order"] - (cell["Y_query_A"] + cell["Y_query_B"])
        cell["OrderedVsShamExcess"] = cell["S_order"] - 2.0 * cell["Y_query_sham"]
        cell["CarrierSuppression"] = cell["Y_query_sham"] - cell["Y_carrier_off"]
        cell["OrderAminusA"] = cell["Y_query_A_then_B"] - cell["Y_query_A"]
        cell["OrderBminusB"] = cell["Y_query_B_then_A"] - cell["Y_query_B"]
        cell["CrossOrderResid"] = cell["OrderAminusA"] - cell["OrderBminusB"]
        cell["SourceAlignedCrossOrderResid"] = source_sign * cell["CrossOrderResid"]
        cell["MapAlignedCrossOrderResid"] = map_sign * cell["CrossOrderResid"]
        cell["SourceMapAlignedCrossOrderResid"] = source_sign * map_sign * cell["CrossOrderResid"]
        cell["CycleOrderNorm"] = cell["D_order_cycles"] / max(
            abs(cell["cycles_query_A_then_B"]) + abs(cell["cycles_query_B_then_A"]), 1.0
        )
        cell["DurationOrderNorm"] = cell["D_order_duration"] / max(
            abs(cell["duration_query_A_then_B"]) + abs(cell["duration_query_B_then_A"]), 1.0
        )
        cell["JOrderCyclesNorm"] = cell["J_order_cycles"] / max(
            abs(cell["J_query_A_then_B_cycles"]) + abs(cell["J_query_B_then_A_cycles"]), 1.0
        )
        cell["JOrderDurationNorm"] = cell["J_order_duration"] / max(
            abs(cell["J_query_A_then_B_duration"]) + abs(cell["J_query_B_then_A_duration"]), 1.0
        )


CANDIDATES: dict[str, dict[str, str]] = {
    "D_order": {
        "definition": "dirty_probe_response(query_A_then_B) - dirty_probe_response(query_B_then_A)",
        "family": "ordered_primary",
    },
    "NormOrder": {
        "definition": "D_order / abs(S_order)",
        "family": "ordered_primary",
    },
    "LogOrder": {
        "definition": "log((query_A_then_B + 1) / (query_B_then_A + 1))",
        "family": "ordered_primary",
    },
    "SourceAlignedDOrder": {
        "definition": "source_order_sign * D_order",
        "family": "aligned_ordered_primary",
    },
    "MapAlignedDOrder": {
        "definition": "mapping_sign * D_order",
        "family": "aligned_ordered_primary",
    },
    "SourceMapAlignedDOrder": {
        "definition": "source_order_sign * mapping_sign * D_order",
        "family": "aligned_ordered_primary",
    },
    "J_order": {
        "definition": "J_AB(query_A_then_B) - J_AB(query_B_then_A)",
        "family": "factorial",
    },
    "JSum": {
        "definition": "J_AB(query_A_then_B) + J_AB(query_B_then_A)",
        "family": "factorial",
    },
    "JMean": {
        "definition": "0.5 * (J_AB(query_A_then_B) + J_AB(query_B_then_A))",
        "family": "factorial",
    },
    "JHalfOrder": {
        "definition": "0.5 * J_order",
        "family": "factorial",
    },
    "SourceAlignedJOrder": {
        "definition": "source_order_sign * J_order",
        "family": "aligned_factorial",
    },
    "MapAlignedJOrder": {
        "definition": "mapping_sign * J_order",
        "family": "aligned_factorial",
    },
    "SourceMapAlignedJOrder": {
        "definition": "source_order_sign * mapping_sign * J_order",
        "family": "aligned_factorial",
    },
    "OrderedVsSingleExcess": {
        "definition": "query_A_then_B + query_B_then_A - query_A - query_B",
        "family": "ordered_total",
    },
    "OrderedVsShamExcess": {
        "definition": "query_A_then_B + query_B_then_A - 2 * query_sham",
        "family": "negative_control_referenced",
    },
    "CarrierSuppression": {
        "definition": "query_sham - carrier_off",
        "family": "negative_control_referenced",
    },
    "ShamScalarDifference": {
        "definition": "(query_A - query_sham) - (query_B - query_sham)",
        "family": "scalar_leakage_test",
    },
    "CrossOrderResid": {
        "definition": "(query_A_then_B - query_A) - (query_B_then_A - query_B)",
        "family": "scalar_leakage_test",
    },
    "SourceAlignedCrossOrderResid": {
        "definition": "source_order_sign * CrossOrderResid",
        "family": "aligned_scalar_leakage_test",
    },
    "MapAlignedCrossOrderResid": {
        "definition": "mapping_sign * CrossOrderResid",
        "family": "aligned_scalar_leakage_test",
    },
    "SourceMapAlignedCrossOrderResid": {
        "definition": "source_order_sign * mapping_sign * CrossOrderResid",
        "family": "aligned_scalar_leakage_test",
    },
    "CycleOrderNorm": {
        "definition": "D_order_cycles normalized by ordered-query cycle magnitude",
        "family": "secondary_timing",
    },
    "DurationOrderNorm": {
        "definition": "D_order_duration normalized by ordered-query duration magnitude",
        "family": "secondary_timing",
    },
    "JOrderCyclesNorm": {
        "definition": "J_order_cycles normalized by factorial cycle-interaction magnitude",
        "family": "secondary_timing",
    },
    "JOrderDurationNorm": {
        "definition": "J_order_duration normalized by factorial duration-interaction magnitude",
        "family": "secondary_timing",
    },
    "K_C2D": {
        "definition": "J_C2D(query_A_then_B) - J_C2D(query_B_then_A), using factorial change_to_dirty counts",
        "family": "factorial_c2d",
    },
    "K_I_C2D": {
        "definition": "J_1[change_to_dirty>0](query_A_then_B) - J_1[change_to_dirty>0](query_B_then_A)",
        "family": "factorial_c2d",
    },
    "K_L_dirty": {
        "definition": "Hadamard active-lane balance L_dirty(query_A_then_B) - L_dirty(query_B_then_A)",
        "family": "factorial_hadamard",
    },
    "K_T_dirty": {
        "definition": "Hadamard arm-total mean T_dirty(query_A_then_B) - T_dirty(query_B_then_A)",
        "family": "factorial_hadamard",
    },
    "K_L_C2D": {
        "definition": "Hadamard active-lane balance L_C2D(query_A_then_B) - L_C2D(query_B_then_A)",
        "family": "factorial_c2d",
    },
    "K_T_C2D": {
        "definition": "Hadamard arm-total mean T_C2D(query_A_then_B) - T_C2D(query_B_then_A)",
        "family": "factorial_c2d",
    },
    "K_L_dirty_per_cycle": {
        "definition": "K_L_dirty after per-arm dirty_probe_response/cpu_cycles exposure normalization",
        "family": "factorial_hadamard_exposure",
    },
    "K_T_dirty_per_cycle": {
        "definition": "K_T_dirty after per-arm dirty_probe_response/cpu_cycles exposure normalization",
        "family": "factorial_hadamard_exposure",
    },
    "K_L_C2D_per_cycle": {
        "definition": "K_L_C2D after per-arm change_to_dirty/cpu_cycles exposure normalization",
        "family": "factorial_c2d_exposure",
    },
    "K_T_C2D_per_cycle": {
        "definition": "K_T_C2D after per-arm change_to_dirty/cpu_cycles exposure normalization",
        "family": "factorial_c2d_exposure",
    },
}


def between_to_within(cells: list[dict[str, Any]], field: str, factor: str) -> dict[str, Any]:
    levels = sorted({cell[factor] for cell in cells}, key=lambda value: repr(value))
    means = []
    within = []
    by_level: dict[str, dict[str, Any]] = {}
    for level in levels:
        values = [float(cell[field]) for cell in cells if cell[factor] == level]
        entry = summarize(values)
        by_level[str(level)] = entry
        means.append(entry["mean"])
        within.append(entry["std"])
    pooled = float(np.mean([value for value in within if value > 0.0])) if any(value > 0.0 for value in within) else 0.0
    return {
        "factor": factor,
        "between_mean_std": float(np.std(means)),
        "pooled_within_std": pooled,
        "between_to_within": float(np.std(means) / pooled) if pooled else None,
        "by_level": by_level,
    }


def categories_from_train(train: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {factor: sorted({str(cell[factor]) for cell in train}) for factor in CATEGORICAL_FACTORS}


def scalar_adversary_features(
    rows: list[dict[str, Any]],
    categories: dict[str, list[str]],
    target: str,
) -> np.ndarray:
    target_family = CANDIDATES[target]["family"]
    include_ordered_primary = target_family not in {
        "ordered_primary",
        "aligned_ordered_primary",
        "ordered_total",
        "negative_control_referenced",
        "scalar_leakage_test",
        "aligned_scalar_leakage_test",
    }
    numeric = [
        "q_float",
        "D_single",
        "Y_query_A",
        "Y_query_B",
        "Y_query_sham",
        "Y_carrier_off",
        "C2D_query_A",
        "C2D_query_B",
        "temperature_c",
        "bank_A_work",
        "bank_B_work",
        "total_work",
        "cycles_query_A",
        "cycles_query_B",
        "duration_query_A",
        "duration_query_B",
    ]
    if include_ordered_primary:
        numeric.extend(
            [
                "Y_query_A_then_B",
                "Y_query_B_then_A",
                "S_order",
                "D_order",
                "D_order_cycles",
                "D_order_duration",
                "cycles_query_A_then_B",
                "cycles_query_B_then_A",
                "duration_query_A_then_B",
                "duration_query_B_then_A",
            ]
        )
    matrix: list[list[float]] = []
    for cell in rows:
        q = float(cell["q_float"]) / 1536.0
        d = float(cell["D_single"]) / 4096.0
        features = [1.0, q, q * q, q**3, q**4, q**5, d, d * d, q * d]
        for name in numeric:
            value = float(cell.get(name, 0.0))
            if name.startswith("cycles_"):
                value /= 1_000_000.0
            elif name.startswith("duration_"):
                value /= 1_000_000.0
            elif name in {"temperature_c"}:
                value /= 100.0
            elif name.startswith("C2D_"):
                value /= 16.0
            else:
                value /= 4096.0
            features.append(value)
        for factor in CATEGORICAL_FACTORS:
            factor_value = str(cell[factor])
            for level in categories[factor]:
                one_hot = 1.0 if factor_value == level else 0.0
                features.extend([one_hot, one_hot * q, one_hot * d])
        matrix.append(features)
    return np.array(matrix, dtype=float)


def ridge_predict(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    target: str,
    alpha: float = 1e-5,
) -> dict[str, Any]:
    categories = categories_from_train(train)
    x_train = scalar_adversary_features(train, categories, target)
    x_test = scalar_adversary_features(test, categories, target)
    y_train = np.array([float(cell[target]) for cell in train], dtype=float)
    y_test = np.array([float(cell[target]) for cell in test], dtype=float)
    coef = np.linalg.pinv(x_train.T @ x_train + alpha * np.eye(x_train.shape[1])) @ (x_train.T @ y_train)
    pred = x_test @ coef
    baseline = np.full_like(y_test, float(np.mean(y_train)))
    denom = float(np.std(y_test))
    if denom <= 1e-12:
        denom = max(float(np.mean(np.abs(y_test))), 1.0)
    return {
        "count": int(len(test)),
        "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
        "relative_rmse_vs_test_std": float(np.sqrt(np.mean((pred - y_test) ** 2)) / denom),
        "mean_baseline_relative_rmse_vs_test_std": float(np.sqrt(np.mean((baseline - y_test) ** 2)) / denom),
        "prediction_actual_corr": safe_corr(pred, y_test),
    }


def holdout_adversary(cells: list[dict[str, Any]], target: str) -> dict[str, Any]:
    by_factor: dict[str, list[dict[str, Any]]] = {}
    rels: list[float] = []
    corrs: list[float] = []
    for factor in HOLDOUT_FACTORS:
        rows = []
        levels = sorted({cell[factor] for cell in cells}, key=lambda value: repr(value))
        for level in levels:
            train = [cell for cell in cells if cell[factor] != level]
            test = [cell for cell in cells if cell[factor] == level]
            if len(test) < 2 or len(train) < 20:
                continue
            metrics = ridge_predict(train, test, target)
            metrics["held_out_level"] = level
            metrics["training_count"] = len(train)
            rows.append(metrics)
            rels.append(metrics["relative_rmse_vs_test_std"])
            if metrics["prediction_actual_corr"] is not None:
                corrs.append(float(metrics["prediction_actual_corr"]))
        by_factor[factor] = rows
    return {
        "holdouts": by_factor,
        "summary": {
            "mean_relative_rmse": float(np.mean(rels)) if rels else None,
            "worst_relative_rmse": float(np.max(rels)) if rels else None,
            "best_relative_rmse": float(np.min(rels)) if rels else None,
            "mean_prediction_actual_corr": float(np.mean(corrs)) if corrs else None,
        },
    }


def all_data_scalar_fit(cells: list[dict[str, Any]], target: str) -> dict[str, Any]:
    categories = categories_from_train(cells)
    x = scalar_adversary_features(cells, categories, target)
    y = np.array([float(cell[target]) for cell in cells], dtype=float)
    coef = np.linalg.pinv(x.T @ x + 1e-5 * np.eye(x.shape[1])) @ (x.T @ y)
    pred = x @ coef
    residual = y - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else 1.0,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "residual_std": float(np.std(residual)),
    }


def q_profile_stability(cells: list[dict[str, Any]], field: str) -> dict[str, Any]:
    q_levels = sorted({cell["q"] for cell in cells})
    profile_by_axis: dict[str, dict[str, list[float]]] = {}
    pair_corrs: list[float] = []
    for factor in CATEGORICAL_FACTORS:
        axis_profiles: dict[str, list[float]] = {}
        for level in sorted({cell[factor] for cell in cells}, key=lambda value: repr(value)):
            profile = []
            for q in q_levels:
                values = [float(cell[field]) for cell in cells if cell[factor] == level and cell["q"] == q]
                profile.append(float(np.mean(values)) if values else 0.0)
            axis_profiles[str(level)] = profile
        profile_by_axis[factor] = axis_profiles
        profiles = list(axis_profiles.values())
        for idx in range(len(profiles)):
            for jdx in range(idx + 1, len(profiles)):
                corr = safe_corr(profiles[idx], profiles[jdx])
                if corr is not None:
                    pair_corrs.append(corr)
    return {
        "q_levels": q_levels,
        "profiles": profile_by_axis,
        "min_pairwise_profile_corr": float(np.min(pair_corrs)) if pair_corrs else None,
        "median_pairwise_profile_corr": float(np.median(pair_corrs)) if pair_corrs else None,
    }


def score_candidate(cells: list[dict[str, Any]], target: str) -> dict[str, Any]:
    values = [float(cell[target]) for cell in cells]
    q_values = [float(cell["q_float"]) for cell in cells]
    d_values = [float(cell["D_single"]) for cell in cells]
    bws = {factor: between_to_within(cells, target, factor) for factor in HOLDOUT_FACTORS}
    adversary = holdout_adversary(cells, target)
    scalar_fit = all_data_scalar_fit(cells, target)
    profile = q_profile_stability(cells, target)
    corr_q = safe_corr(values, q_values)
    corr_d = safe_corr(values, d_values)
    max_non_q_bw = max(
        (entry["between_to_within"] or 0.0) for factor, entry in bws.items() if factor != "q"
    )
    q_bw = bws["q"]["between_to_within"] or 0.0
    scalar_leakage = bool(abs(corr_q or 0.0) >= 0.90 or abs(corr_d or 0.0) >= 0.90)
    if target == "CrossOrderResid":
        scalar_leakage = True
    if target == "ShamScalarDifference":
        scalar_leakage = True
    weak_signal = bool(max(q_bw, max_non_q_bw) < 0.50)
    stable_profile = bool((profile["min_pairwise_profile_corr"] or -1.0) >= 0.50)
    adversary_does_not_replay = bool((adversary["summary"]["mean_relative_rmse"] or 0.0) >= 0.90)
    clean_candidate = bool(
        not scalar_leakage
        and not weak_signal
        and stable_profile
        and adversary_does_not_replay
        and CANDIDATES[target]["family"] not in {"negative_control_referenced", "secondary_timing"}
    )
    rejection_reasons = []
    if scalar_leakage:
        rejection_reasons.append("scalar-q leakage or algebraic dependence on D_single")
    if weak_signal:
        rejection_reasons.append("no strong between-state structure after matched within-state noise")
    if not stable_profile:
        rejection_reasons.append("q-profile or transformation profile is not stable across required strata")
    if not adversary_does_not_replay:
        rejection_reasons.append("strong scalar/nuisance adversary can replay the candidate")
    if CANDIDATES[target]["family"] in {"negative_control_referenced", "secondary_timing"}:
        rejection_reasons.append("diagnostic negative-control/timing channel, not a primary carrier operator coordinate")
    return {
        "name": target,
        "definition": CANDIDATES[target]["definition"],
        "family": CANDIDATES[target]["family"],
        "summary": summarize(values),
        "correlation_with_q": corr_q,
        "correlation_with_D_single": corr_d,
        "between_to_within": bws,
        "scalar_adversary_holdout": adversary,
        "all_data_scalar_nuisance_fit": scalar_fit,
        "q_profile_stability": profile,
        "derived_flags": {
            "scalar_leakage": scalar_leakage,
            "weak_signal": weak_signal,
            "stable_profile": stable_profile,
            "adversary_does_not_replay": adversary_does_not_replay,
            "clean_second_operator_candidate": clean_candidate,
        },
        "rejection_reasons": rejection_reasons,
    }


def algebraic_identity_checks(cells: list[dict[str, Any]]) -> dict[str, Any]:
    cross_errors = [
        abs(float(cell["CrossOrderResid"]) - (float(cell["D_order"]) - float(cell["D_single"])))
        for cell in cells
    ]
    sham_errors = [
        abs(float(cell["ShamScalarDifference"]) - float(cell["D_single"]))
        for cell in cells
    ]
    order_sum_errors = [
        abs(float(cell["OrderedVsSingleExcess"]) - (float(cell["S_order"]) - float(cell["Y_query_A"]) - float(cell["Y_query_B"])))
        for cell in cells
    ]
    return {
        "CrossOrderResid_equals_D_order_minus_D_single": {
            "definition": "CrossOrderResid = (A_then_B - A) - (B_then_A - B) = D_order - D_single",
            "max_abs_error": float(np.max(cross_errors)),
            "passed": bool(float(np.max(cross_errors)) == 0.0),
            "interpretation": "Any strong q dependence in CrossOrderResid is scalar q leakage unless D_order supplies an independent stable residual.",
        },
        "ShamScalarDifference_equals_D_single": {
            "definition": "(query_A - query_sham) - (query_B - query_sham) = D_single",
            "max_abs_error": float(np.max(sham_errors)),
            "passed": bool(float(np.max(sham_errors)) == 0.0),
            "interpretation": "The sham-referenced difference cannot be a second dimension because the sham term cancels exactly.",
        },
        "OrderedVsSingleExcess_identity": {
            "definition": "OrderedVsSingleExcess = S_order - query_A - query_B",
            "max_abs_error": float(np.max(order_sum_errors)),
            "passed": bool(float(np.max(order_sum_errors)) == 0.0),
            "interpretation": "This is an ordinary ordered-query route/total offset unless it transforms independently under relation-only controls.",
        },
    }


def schedule_pairing_report(raw_rows: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    primary_rows = [
        row
        for row in raw_rows
        if row["matrix_block"] == "persistence_matrix"
        and row["factorial_arm"] == "primary_matrix"
        and not row["source_off_control"]
    ]
    primary_by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in primary_rows:
        key = tuple(row[factor] for factor in oda.CELL_FACTORS)
        primary_by_cell.setdefault(key, {})[row["query"]] = row

    ordered_gaps = []
    scalar_gaps = []
    signed_order_time_gaps = []
    d_orders = []
    for key, row_map in primary_by_cell.items():
        if "query_A_then_B" in row_map and "query_B_then_A" in row_map:
            first = row_map["query_A_then_B"]
            second = row_map["query_B_then_A"]
            ordered_gaps.append(
                {
                    "execution_ordinal_gap": abs(int(first["execution_ordinal"]) - int(second["execution_ordinal"])),
                    "start_time_gap_s": abs(
                        float(first["query_start_monotonic_ns"]) - float(second["query_start_monotonic_ns"])
                    )
                    / 1_000_000_000.0,
                }
            )
            signed_order_time_gaps.append(
                (
                    float(first["query_start_monotonic_ns"]) - float(second["query_start_monotonic_ns"])
                )
                / 1_000_000_000.0
            )
            d_orders.append(float(first["dirty_probe_response"]) - float(second["dirty_probe_response"]))
        if "query_A" in row_map and "query_B" in row_map:
            first = row_map["query_A"]
            second = row_map["query_B"]
            scalar_gaps.append(
                {
                    "execution_ordinal_gap": abs(int(first["execution_ordinal"]) - int(second["execution_ordinal"])),
                    "start_time_gap_s": abs(
                        float(first["query_start_monotonic_ns"]) - float(second["query_start_monotonic_ns"])
                    )
                    / 1_000_000_000.0,
                }
            )

    factorial_rows = [
        row
        for row in raw_rows
        if row["matrix_block"] == "factorial_nonadditivity" and not row["source_off_control"]
    ]
    factorial_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in factorial_rows:
        key = tuple(row[factor] for factor in oda.CELL_FACTORS) + (row["query"],)
        factorial_by_group.setdefault(key, []).append(row)
    factorial_spans = []
    for rows in factorial_by_group.values():
        if len(rows) != len(oda.FACTORIAL_ARMS):
            continue
        ordinals = [int(row["execution_ordinal"]) for row in rows]
        starts = [float(row["query_start_monotonic_ns"]) for row in rows]
        factorial_spans.append(
            {
                "execution_ordinal_span": max(ordinals) - min(ordinals),
                "start_time_span_s": (max(starts) - min(starts)) / 1_000_000_000.0,
            }
        )

    def gap_stats(entries: list[dict[str, Any]], ordinal_key: str, time_key: str) -> dict[str, Any]:
        ordinals = np.array([entry[ordinal_key] for entry in entries], dtype=float)
        times = np.array([entry[time_key] for entry in entries], dtype=float)
        return {
            "count": int(len(entries)),
            "execution_ordinal": {
                "median": float(np.median(ordinals)),
                "p95": float(np.percentile(ordinals, 95)),
                "max": float(np.max(ordinals)),
            },
            "time_s": {
                "median": float(np.median(times)),
                "p95": float(np.percentile(times, 95)),
                "max": float(np.max(times)),
            },
        }

    temperatures = [float(row["temperature_c"]) for row in raw_rows]
    return {
        "primary_ordered_AB_BA_gap": gap_stats(ordered_gaps, "execution_ordinal_gap", "start_time_gap_s"),
        "primary_scalar_A_B_gap": gap_stats(scalar_gaps, "execution_ordinal_gap", "start_time_gap_s"),
        "factorial_four_arm_span": gap_stats(factorial_spans, "execution_ordinal_span", "start_time_span_s"),
        "signed_time_gap_correlation_with_D_order": safe_corr(signed_order_time_gaps, d_orders),
        "temperature_report": {
            "unique_values": sorted({float(value) for value in temperatures}),
            "unique_value_count": len(set(temperatures)),
            "summary": summarize(temperatures),
            "statistically_controls_subdegree_thermal_drift": False,
        },
        "interpretation": (
            "The retained archive supports offline diagnostics, but it did not locally pair AB and BA or the factorial "
            "arms tightly enough to prove a relation coordinate against ordinary schedule, route, and gain drift."
        ),
    }


def source_off_control_cube(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in raw_rows
        if row["matrix_block"] == "persistence_matrix"
        and row["factorial_arm"] == "primary_matrix"
        and int(row["q"]) == 0
    ]
    by_state: dict[tuple[Any, ...], dict[bool, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        key = tuple(row[factor] for factor in ["session", "replicate", "mapping", "delay_label", "source_order"])
        by_state.setdefault(key, {}).setdefault(bool(row["source_off_control"]), {})[row["query"]] = row

    channels = ["dirty_probe_response", "change_to_dirty", "cpu_cycles", "duration_ns"]
    contrasts = {
        "D_order": ("query_A_then_B", "query_B_then_A"),
        "sham_minus_carrier_off": ("query_sham", "carrier_off"),
        "D_single": ("query_A", "query_B"),
    }
    residuals: dict[str, list[float]] = {f"{name}_{channel}": [] for name in contrasts for channel in channels}
    active_values: dict[str, list[float]] = {f"{name}_{channel}": [] for name in contrasts for channel in channels}
    source_off_values: dict[str, list[float]] = {f"{name}_{channel}": [] for name in contrasts for channel in channels}
    missing: list[str] = []
    for key, states in by_state.items():
        active = states.get(False, {})
        source_off = states.get(True, {})
        for contrast_name, (left_query, right_query) in contrasts.items():
            if left_query not in active or right_query not in active or left_query not in source_off or right_query not in source_off:
                missing.append(f"{key!r} {contrast_name} missing active/source-off query")
                continue
            for channel in channels:
                active_delta = float(active[left_query][channel]) - float(active[right_query][channel])
                source_off_delta = float(source_off[left_query][channel]) - float(source_off[right_query][channel])
                name = f"{contrast_name}_{channel}"
                active_values[name].append(active_delta)
                source_off_values[name].append(source_off_delta)
                residuals[name].append(active_delta - source_off_delta)

    report = {
        "complete": not missing,
        "missing": missing,
        "matched_q0_state_count": len(by_state),
        "contrasts": {},
        "interpretation": "Candidate survival requires exceeding matched source-off behavior without following sham/carrier receiver-route offsets.",
    }
    for name in residuals:
        source_values = source_off_values[name]
        active = active_values[name]
        resid = residuals[name]
        p95 = float(np.percentile(np.abs(source_values), 95)) if source_values else None
        report["contrasts"][name] = {
            "active_q0": summarize(active),
            "source_off_q0": summarize(source_values),
            "active_minus_source_off": summarize(resid),
            "source_off_abs_p95": p95,
            "active_mean_exceeds_source_off_abs_p95": bool(abs(float(np.mean(active))) > p95) if p95 is not None else False,
        }
    return report


def t_like(values: list[float]) -> float | None:
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return None
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return None
    return float(np.mean(arr) / (std / math.sqrt(len(arr))))


def sign_agreement_and_corr(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"count": 0}
    agreements = [
        (left == 0.0 and right == 0.0) or (left < 0.0 and right < 0.0) or (left > 0.0 and right > 0.0)
        for left, right in pairs
    ]
    return {
        "count": len(pairs),
        "sign_agreement_fraction": float(sum(agreements) / len(agreements)),
        "correlation": safe_corr([left for left, _ in pairs], [right for _, right in pairs]),
    }


def hadamard_parity_report(cells: list[dict[str, Any]]) -> dict[str, Any]:
    block_factors = ["session", "replicate", "delay_label"]
    q_abs_levels = [512, 1024, 1536]
    characters = {
        "H00_invariant": (0, 0),
        "H10_mapping_odd": (1, 0),
        "H01_source_order_odd": (0, 1),
        "H11_mapping_x_source_order": (1, 1),
    }

    cells_by_block_q: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for cell in cells:
        key = tuple(cell[factor] for factor in block_factors) + (int(cell["q"]),)
        cells_by_block_q.setdefault(key, []).append(cell)

    h_values: dict[tuple[Any, ...], dict[str, float]] = {}
    completeness_failures: list[str] = []
    for key, group in cells_by_block_q.items():
        if len(group) != 4:
            completeness_failures.append(f"{key!r} has {len(group)} mapping/source-order cells")
            continue
        seen = {(cell["mapping"], cell["source_order"]) for cell in group}
        if len(seen) != 4:
            completeness_failures.append(f"{key!r} missing mapping/source-order combination")
            continue
        char_values = {}
        for name, (a, b) in characters.items():
            total = 0.0
            for cell in group:
                m = 1 if cell["mapping"] == "map1" else 0
                s = 1 if cell["source_order"] == "B_then_A" else 0
                sign = -1.0 if ((a * m + b * s) % 2) else 1.0
                total += sign * float(cell["LogOrder"])
            char_values[name] = 0.25 * total
        h_values[key] = char_values

    even_values: dict[str, list[dict[str, Any]]] = {name: [] for name in characters}
    odd_values: dict[str, list[dict[str, Any]]] = {name: [] for name in characters}
    for session in sorted({cell["session"] for cell in cells}):
        for replicate in sorted({cell["replicate"] for cell in cells}):
            for delay in sorted({cell["delay_label"] for cell in cells}):
                base_key = (session, replicate, delay)
                if base_key + (0,) not in h_values:
                    completeness_failures.append(f"{base_key!r} missing q=0 Hadamard baseline")
                    continue
                for q_abs in q_abs_levels:
                    pos = h_values.get(base_key + (q_abs,))
                    neg = h_values.get(base_key + (-q_abs,))
                    zero = h_values.get(base_key + (0,))
                    if pos is None or neg is None or zero is None:
                        completeness_failures.append(f"{base_key!r} missing +/-{q_abs} Hadamard value")
                        continue
                    for name in characters:
                        even = 0.5 * (pos[name] + neg[name]) - zero[name]
                        odd = 0.5 * (pos[name] - neg[name])
                        payload = {
                            "session": session,
                            "replicate": replicate,
                            "delay_label": delay,
                            "abs_q": q_abs,
                            "value": float(even),
                        }
                        even_values[name].append(payload)
                        odd_values[name].append({**payload, "value": float(odd)})

    def component_report(entries: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(entry["value"]) for entry in entries]
        by_abs_q = {
            str(q_abs): summarize([entry["value"] for entry in entries if int(entry["abs_q"]) == q_abs])
            for q_abs in q_abs_levels
        }
        by_delay = {
            str(delay): summarize([entry["value"] for entry in entries if entry["delay_label"] == delay])
            for delay in sorted({entry["delay_label"] for entry in entries})
        }
        by_session = {
            str(session): summarize([entry["value"] for entry in entries if entry["session"] == session])
            for session in sorted({entry["session"] for entry in entries})
        }
        by_replicate = {
            str(replicate): summarize([entry["value"] for entry in entries if entry["replicate"] == replicate])
            for replicate in sorted({entry["replicate"] for entry in entries})
        }
        block_groups: dict[tuple[Any, ...], list[float]] = {}
        for entry in entries:
            key = (entry["session"], entry["replicate"], entry["delay_label"])
            block_groups.setdefault(key, []).append(float(entry["value"]))
        all_three_negative = sum(1 for values_in_block in block_groups.values() if len(values_in_block) == 3 and all(value < 0.0 for value in values_in_block))

        by_key = {
            (entry["session"], entry["replicate"], entry["delay_label"], int(entry["abs_q"])): float(entry["value"])
            for entry in entries
        }
        replicate_pairs = []
        for session in sorted({entry["session"] for entry in entries}):
            for delay in sorted({entry["delay_label"] for entry in entries}):
                for q_abs in q_abs_levels:
                    left = by_key.get((session, 0, delay, q_abs))
                    right = by_key.get((session, 1, delay, q_abs))
                    if left is not None and right is not None:
                        replicate_pairs.append((left, right))
        session_pairs = []
        for replicate in sorted({entry["replicate"] for entry in entries}):
            for delay in sorted({entry["delay_label"] for entry in entries}):
                for q_abs in q_abs_levels:
                    left = by_key.get(("session_0", replicate, delay, q_abs))
                    right = by_key.get(("session_1", replicate, delay, q_abs))
                    if left is not None and right is not None:
                        session_pairs.append((left, right))
        return {
            "summary": summarize(values),
            "t_like": t_like(values),
            "negative_count": int(sum(1 for value in values if value < 0.0)),
            "negative_fraction": float(sum(1 for value in values if value < 0.0) / len(values)) if values else None,
            "by_abs_q": by_abs_q,
            "by_delay": by_delay,
            "by_session": by_session,
            "by_replicate": by_replicate,
            "block_count": len(block_groups),
            "blocks_with_all_three_magnitudes_negative": int(all_three_negative),
            "replicate_pair_transport": sign_agreement_and_corr(replicate_pairs),
            "session_pair_transport": sign_agreement_and_corr(session_pairs),
        }

    even_report = {name: component_report(entries) for name, entries in even_values.items()}
    odd_report = {name: component_report(entries) for name, entries in odd_values.items()}
    c00 = even_report["H00_invariant"]
    c00_block_count = c00["block_count"]
    c00_all_negative = c00["blocks_with_all_three_magnitudes_negative"]
    c00_replicate = c00["replicate_pair_transport"]
    c00_session = c00["session_pair_transport"]
    clean = bool(
        c00_all_negative == c00_block_count
        and (c00_replicate.get("sign_agreement_fraction") or 0.0) >= 0.90
        and (c00_session.get("sign_agreement_fraction") or 0.0) >= 0.90
        and (abs(even_report["H10_mapping_odd"]["t_like"] or 0.0) < 2.0)
        and (abs(even_report["H01_source_order_odd"]["t_like"] or 0.0) < 2.0)
        and (abs(even_report["H11_mapping_x_source_order"]["t_like"] or 0.0) < 2.0)
    )
    return {
        "complete": not completeness_failures,
        "completeness_failures": completeness_failures,
        "observable": "LogOrder = log((query_A_then_B + 1) / (query_B_then_A + 1))",
        "characters": characters,
        "even_q_zero_referenced_components": even_report,
        "odd_q_components": odd_report,
        "clean_second_operator_candidate": clean,
        "interpretation": (
            "The invariant q-even curvature is an offline diagnostic lead, but it is q=0 referenced, scalar-replayable "
            "by |q| or q^2 response curvature, and does not transport cleanly across replicate/session blocks."
        ),
    }


def geometry_nuisance_features(cells: list[dict[str, Any]]) -> np.ndarray:
    categories = categories_from_train(cells)
    rows = []
    for cell in cells:
        q = float(cell["q_float"]) / 1536.0
        d = float(cell["D_single"]) / 4096.0
        features = [
            1.0,
            q,
            q * q,
            q**3,
            q**4,
            q**5,
            d,
            d * d,
            q * d,
            float(cell["bank_A_work"]) / 4096.0,
            float(cell["bank_B_work"]) / 4096.0,
            float(cell["total_work"]) / 4096.0,
            float(cell["temperature_c"]) / 100.0,
        ]
        for factor in CATEGORICAL_FACTORS:
            factor_value = str(cell[factor])
            for level in categories[factor]:
                one_hot = 1.0 if factor_value == level else 0.0
                features.extend([one_hot, one_hot * q, one_hot * d])
        rows.append(features)
    return np.array(rows, dtype=float)


def residual_matrix(cells: list[dict[str, Any]]) -> dict[str, Any]:
    columns = BASE_GEOMETRY_COLUMNS
    x = geometry_nuisance_features(cells)
    raw = np.array([[float(cell[column]) for column in columns] for cell in cells], dtype=float)
    residual = np.zeros_like(raw)
    for idx in range(raw.shape[1]):
        y = raw[:, idx]
        coef = np.linalg.pinv(x.T @ x + 1e-5 * np.eye(x.shape[1])) @ (x.T @ y)
        residual[:, idx] = y - x @ coef
    scale = np.std(residual, axis=0)
    scale[scale <= 1e-12] = 1.0
    z = residual / scale
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    spectrum = [float(value) for value in singular]
    normalized = singular / np.sum(singular)
    effective_rank = float(np.exp(-np.sum(normalized * np.log(normalized + 1e-15))))
    rng = np.random.default_rng(20260720)
    shuffle_second = []
    for _ in range(300):
        shuffled = z.copy()
        for col_idx in range(shuffled.shape[1]):
            shuffled[:, col_idx] = rng.permutation(shuffled[:, col_idx])
        _, s, _ = np.linalg.svd(shuffled, full_matrices=False)
        shuffle_second.append(float(s[1]))
    full_second_vector = vt[1, :]
    stratum_alignment: dict[str, dict[str, float]] = {}
    for factor in CATEGORICAL_FACTORS:
        alignments = {}
        for level in sorted({cell[factor] for cell in cells}, key=lambda value: repr(value)):
            idxs = [idx for idx, cell in enumerate(cells) if cell[factor] == level]
            if len(idxs) < len(columns) + 2:
                continue
            _, _, sub_vt = np.linalg.svd(z[idxs, :], full_matrices=False)
            alignments[str(level)] = float(abs(np.dot(full_second_vector, sub_vt[1, :])))
        stratum_alignment[factor] = alignments
    min_alignment = min(
        (value for axis in stratum_alignment.values() for value in axis.values()),
        default=None,
    )
    return {
        "algebra_preserving_independent_base_columns_only": True,
        "columns": columns,
        "singular_values": spectrum,
        "effective_rank_after_scalar_nuisance_residualization": effective_rank,
        "second_singular_value": spectrum[1],
        "second_singular_shuffle_p95": float(np.percentile(shuffle_second, 95)),
        "second_singular_shuffle_p99": float(np.percentile(shuffle_second, 99)),
        "second_above_column_shuffle_p95": bool(spectrum[1] > float(np.percentile(shuffle_second, 95))),
        "second_above_column_shuffle_p99": bool(spectrum[1] > float(np.percentile(shuffle_second, 99))),
        "second_axis_loadings": {column: float(full_second_vector[idx]) for idx, column in enumerate(columns)},
        "second_axis_stratum_alignment": stratum_alignment,
        "minimum_second_axis_alignment_across_required_strata": min_alignment,
        "stable_second_axis_across_required_strata": bool(min_alignment is not None and min_alignment >= 0.50),
    }


def decision_from_candidates(candidates: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    clean = [name for name, entry in candidates.items() if entry["derived_flags"]["clean_second_operator_candidate"]]
    if clean and geometry["stable_second_axis_across_required_strata"]:
        return {
            "result_class": RESULT_CLASS_CANDIDATE,
            "scientific_claim": CLAIM_CANDIDATE,
            "candidate_found": True,
            "candidate_observables": clean,
            "rationale": "At least one candidate survived scalar leakage, scalar replay, profile stability, and residual-axis stability checks.",
        }
    if geometry["second_above_column_shuffle_p95"] or any(
        entry["scalar_adversary_holdout"]["summary"]["mean_relative_rmse"] and entry["scalar_adversary_holdout"]["summary"]["mean_relative_rmse"] >= 0.90
        for entry in candidates.values()
    ):
        result = RESULT_CLASS_UNRESOLVED
    else:
        result = RESULT_CLASS_NOT_OBSERVED
    return {
        "result_class": result,
        "scientific_claim": None,
        "candidate_found": False,
        "candidate_observables": [],
        "rationale": (
            "No clean second operator dimension survived. The strongest apparent signals are either algebraic scalar-q "
            "leakage, negative-control route offsets, or residual axes that do not remain stable across the required "
            "session/replicate/mapping/delay/source-order strata."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    candidates = report["candidate_ranking"]
    decision = report["decision"]
    top = candidates[:8]
    lines = [
        "# Family 10h Deep Operator-Dimension Hunt",
        "",
        f"Result: `{decision['result_class']}`",
        "",
        "Scope: offline-only exploratory follow-up over the sealed v1.1 attempt-1 archive. No target contact, PMU acquisition, runtime execution, SSH, SCP, deployment, or cleanup was performed.",
        "",
        "## What Changed",
        "",
        "This pass explicitly tests the easy-to-miss transforms: source-order aligned order contrasts, mapping-aligned order contrasts, log/ratio order contrasts, factorial symmetric and antisymmetric terms, factorial `change_to_dirty` interactions, Hadamard arm-balance coordinates, sham/carrier negative controls, timing-normalized terms, and the tempting cross-order residual.",
        "",
        "## Main Findings",
        "",
        "- The strongest new-looking coordinate, `CrossOrderResid`, is rejected because `(query_A_then_B - query_A) - (query_B_then_A - query_B) = D_order - D_single`; it is dominated by the already-confirmed scalar q readout.",
        "- The sham-referenced difference also cancels to exactly `D_single`, so it is now a mechanical scalar-leakage rejection.",
        "- Source-order and mapping-aligned order transforms reduce the scalar contamination but do not produce a stable held-out operator law.",
        "- Sham/carrier terms expose a large ordinary carrier/query route offset, not a relation-specific coordinate.",
        "- The q=0 source-off cube separates true active response from source-off null for `D_order`, but the sham/carrier route offset is nearly the same active and source-off, so it is not a carrier-state dimension.",
        "- Factorial `change_to_dirty` and Hadamard arm coordinates are present but sparse/noisy; they do not supply a clean held-out operator law.",
        "- Hadamard parity over mapping and source-order shows an invariant q-even curvature lead, but it is q=0 referenced, scalar-replayable, and fails session/replicate transport.",
        "- Timing and cycle transforms are diagnostic only and do not survive as primary carrier coordinates.",
        "- Algebra-preserving residual geometry now uses only independent measured base columns, not derived columns shuffled as if they were independent evidence.",
        "",
        "## Top Candidate Ranking",
        "",
    ]
    for entry in top:
        flags = entry["derived_flags"]
        lines.append(
            f"- `{entry['name']}`: family `{entry['family']}`, corr(q) `{entry['correlation_with_q']}`, "
            f"mean held-out rel RMSE `{entry['scalar_adversary_holdout']['summary']['mean_relative_rmse']}`, "
            f"clean candidate `{flags['clean_second_operator_candidate']}`"
        )
        if entry["rejection_reasons"]:
            lines.append(f"  Rejected: {'; '.join(entry['rejection_reasons'])}.")
    geometry = report["residual_geometry"]
    lines.extend(
        [
            "",
            "## Residual Geometry",
            "",
            f"- residual effective rank: `{geometry['effective_rank_after_scalar_nuisance_residualization']}`",
            f"- second singular value above column-shuffle p95: `{geometry['second_above_column_shuffle_p95']}`",
            f"- second singular value above column-shuffle p99: `{geometry['second_above_column_shuffle_p99']}`",
            f"- minimum second-axis alignment across required strata: `{geometry['minimum_second_axis_alignment_across_required_strata']}`",
            f"- stable second axis across required strata: `{geometry['stable_second_axis_across_required_strata']}`",
            "",
            "## Decision",
            "",
            decision["rationale"],
            "",
            "This does not establish full carrier-state tomography, relational carrier, physical relational memory, catalytic borrowing, or `SMALL_WALL_CROSSED`.",
            "",
            "## Smallest Next Grammar",
            "",
            "The next useful experiment should create a relation-only contrast: hold `query_A`, `query_B`, `D_single`, total work, route pressure, query count, source order, and delay fixed while transforming only the A/B address relation. Include relation-preserving, relation-swapped, relation-distance-shifted, and relation-sham layouts plus scalar replay and route-pressure controls.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    root = oda.repo_root()
    evidence, raw_rows = oda.validate_evidence(root)
    cells, completeness = oda.build_cells(raw_rows)
    if not evidence["passed"] or not completeness["passed"]:
        raise SystemExit("evidence validation or matched-cell completeness failed")
    factorial_enrichment = enrich_factorial_observables(raw_rows, cells)
    if not factorial_enrichment["complete"]:
        raise SystemExit("factorial C2D/Hadamard enrichment failed")
    augment_cells(cells)
    scalar = oda.scalar_baseline(cells)
    candidate_reports = {name: score_candidate(cells, name) for name in CANDIDATES}
    ranking = sorted(
        candidate_reports.values(),
        key=lambda entry: (
            entry["derived_flags"]["clean_second_operator_candidate"],
            not entry["derived_flags"]["scalar_leakage"],
            not entry["derived_flags"]["weak_signal"],
            entry["scalar_adversary_holdout"]["summary"]["mean_relative_rmse"] or 0.0,
        ),
        reverse=True,
    )
    geometry = residual_matrix(cells)
    algebra = algebraic_identity_checks(cells)
    pairing = schedule_pairing_report(raw_rows, cells)
    source_off_cube = source_off_control_cube(raw_rows)
    hadamard = hadamard_parity_report(cells)
    decision = decision_from_candidates(candidate_reports, geometry)
    core = {
        "schema": "FAMILY10H_OPERATOR_DIMENSION_DEEP_HUNT_V1",
        "source_authority_commit": oda.SOURCE_AUTHORITY_COMMIT,
        "manifest_freeze_commit": oda.MANIFEST_FREEZE_COMMIT,
        "postrun_seal_commit": oda.POSTRUN_SEAL_COMMIT,
        "archive_sha256": oda.EXPECTED_ARCHIVE_SHA256,
        "evidence_passed": evidence["passed"],
        "raw_record_count": evidence["validations"]["raw_record_count"],
        "source_death_receipt_count": evidence["validations"]["source_death_receipt_count"],
        "matched_cell_count": len(cells),
        "scalar_baseline": scalar,
        "candidate_reports": candidate_reports,
        "candidate_ranking": ranking,
        "factorial_c2d_hadamard_enrichment": factorial_enrichment,
        "algebraic_identity_checks": algebra,
        "schedule_pairing_and_temperature_report": pairing,
        "source_off_control_cube": source_off_cube,
        "hadamard_parity_report": hadamard,
        "residual_geometry": geometry,
        "decision": decision,
        "claim_boundary": {
            "scalar_q_readout_confirmed_prospectively": True,
            "clean_second_operator_dimension_found": decision["candidate_found"],
            "full_tomography_established": False,
            "relational_carrier_established": False,
            "physical_relational_memory_established": False,
            "small_wall_crossed": False,
        },
        "zero_live_activity_by_this_hunt": True,
    }
    report = dict(core)
    report["canonical_deep_hunt_digest"] = canonical_digest(core)
    json_dump(HERE / "OPERATOR_DIMENSION_DEEP_HUNT.json", report)
    json_dump(
        HERE / "SECOND_DIMENSION_CANDIDATE_RANKING.json",
        {
            "schema": "FAMILY10H_SECOND_DIMENSION_CANDIDATE_RANKING_V1",
            "postrun_seal_commit": oda.POSTRUN_SEAL_COMMIT,
            "ranking": ranking,
            "decision": decision,
        },
    )
    (HERE / "OPERATOR_DIMENSION_DEEP_HUNT.md").write_text(markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    run()
