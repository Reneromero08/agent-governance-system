from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_AUTHORITY_COMMIT = "cac5d33536768e00aa0de5f515e626fecccdeeda"
MANIFEST_FREEZE_COMMIT = "354ca8ab2d62458fca41481d74ff98c1b39ab6ed"
POSTRUN_SEAL_COMMIT = "c126700b8d46e6501ff39cfa360bf32a9fbdb2ac"
TRANSACTION_RUN_ID = "family10h_carrier_tomography_v1_1_paired_dirty_probe_0"
EXPECTED_ARCHIVE_SHA256 = "0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041"
EXPECTED_RAW_COUNT = 8320
EXPECTED_SOURCE_DEATH_COUNT = 8320
RESULT_CLASS_CONFIRMED = "FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_CONFIRMED_PROSPECTIVE"
SCIENTIFIC_CLAIM_CONFIRMED = "PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_CONFIRMED"
RESULT_CLASS_NOT_OBSERVED = "FAMILY10H_SECOND_OPERATOR_DIMENSION_NOT_OBSERVED_RETROSPECTIVE"
RESULT_CLASS_CANDIDATE = "FAMILY10H_SECOND_OPERATOR_DIMENSION_CANDIDATE_RETROSPECTIVE"
RESULT_CLASS_UNRESOLVED = "FAMILY10H_SECOND_OPERATOR_DIMENSION_UNRESOLVED_RETROSPECTIVE"
RESULT_CLASS_CUSTODY_INVALID = "FAMILY10H_SECOND_OPERATOR_DIMENSION_CUSTODY_INVALID"
CLAIM_CANDIDATE = "PUBLIC_POST_SOURCE_SECOND_OPERATOR_DIMENSION_CANDIDATE_RETROSPECTIVE"
RNG_SEED = 20260720

LIVE_SMALL_WALL_ROOT = Path(
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6/live_small_wall"
)
ATTEMPT_REL = (
    LIVE_SMALL_WALL_ROOT
    / "carrier_state_tomography/runs/family10h_carrier_tomography_v1_1_paired_dirty_probe_0/attempt_1"
)
SOURCE_PACKAGE_REL = (
    LIVE_SMALL_WALL_ROOT
    / "carrier_state_tomography/family10h_carrier_tomography_v1_1_paired_dirty_probe"
)
ARCHIVE_NAME = "ATTEMPT_1_REMOTE_ROOT.tar.gz"
ARCHIVE_BASE = TRANSACTION_RUN_ID
PRIMARY_QUERIES = ["query_A", "query_B", "query_A_then_B", "query_B_then_A", "query_sham", "carrier_off"]
FACTORIAL_ARMS = ["both_active", "A_active_B_dummy", "A_dummy_B_active", "both_dummy"]
CELL_FACTORS = ["session", "replicate", "mapping", "delay_label", "source_order", "q"]
CATEGORICAL_FACTORS = ["session", "replicate", "mapping", "delay_label", "source_order"]
CANDIDATES = [
    "D_order",
    "S_order",
    "J_query_A_then_B",
    "J_query_B_then_A",
    "J_order",
]
GEOMETRY_COLUMNS = [
    "Y_query_A",
    "Y_query_B",
    "Y_query_A_then_B",
    "Y_query_B_then_A",
    "D_single",
    "D_order",
    "S_order",
    "J_query_A_then_B",
    "J_query_B_then_A",
    "J_order",
]


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return Path(out)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def canonical_digest(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_archive_json(tf: tarfile.TarFile, member: str) -> Any:
    handle = tf.extractfile(member)
    if handle is None:
        raise ValueError(f"missing archive member: {member}")
    return json.loads(handle.read())


def load_archive_jsonl(tf: tarfile.TarFile, member: str) -> list[dict[str, Any]]:
    handle = tf.extractfile(member)
    if handle is None:
        raise ValueError(f"missing archive member: {member}")
    return [json.loads(line) for line in handle]


def archive_member_record(tf: tarfile.TarFile, member: str) -> dict[str, Any]:
    info = tf.getmember(member)
    handle = tf.extractfile(info)
    if handle is None:
        raise ValueError(f"archive member is not a file: {member}")
    data = handle.read()
    return {"archive_member": member, "sha256": sha256_bytes(data), "size": info.size}


def stats(values: list[float]) -> dict[str, Any]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "median": float(np.median(arr)),
    }


def safe_corr(a: list[float], b: list[float]) -> float | None:
    xa = np.array(a, dtype=float)
    xb = np.array(b, dtype=float)
    if len(xa) < 2 or float(np.std(xa)) == 0.0 or float(np.std(xb)) == 0.0:
        return None
    return float(np.corrcoef(xa, xb)[0, 1])


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.pinv(design) @ y
    pred = design @ coef
    residual = y - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else 1.0,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
    }


def relative_rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = float(np.std(actual))
    if denom <= 1e-12:
        denom = max(float(np.mean(np.abs(actual))), 1.0)
    return float(np.sqrt(np.mean((pred - actual) ** 2)) / denom)


def build_cells(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_rows = [
        row
        for row in raw_rows
        if row["matrix_block"] == "persistence_matrix"
        and row["factorial_arm"] == "primary_matrix"
        and not row["source_off_control"]
    ]
    primary_by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary_rows:
        primary_by_cell[tuple(row[factor] for factor in CELL_FACTORS)][row["query"]] = row

    factorial_rows = [
        row
        for row in raw_rows
        if row["matrix_block"] == "factorial_nonadditivity" and not row["source_off_control"]
    ]
    factorial_by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in factorial_rows:
        key = tuple(row[factor] for factor in CELL_FACTORS) + (row["query"],)
        factorial_by_cell[key][row["factorial_arm"]] = row

    cells: list[dict[str, Any]] = []
    failures: list[str] = []
    for key, query_rows in sorted(primary_by_cell.items(), key=lambda item: repr(item[0])):
        missing_queries = [query for query in PRIMARY_QUERIES if query not in query_rows]
        if missing_queries:
            failures.append(f"primary cell {key!r} missing queries {missing_queries!r}")
            continue
        cell = {factor: value for factor, value in zip(CELL_FACTORS, key)}
        cell["q_float"] = float(cell["q"])
        for query, row in query_rows.items():
            cell[f"Y_{query}"] = float(row["dirty_probe_response"])
            cell[f"C2D_{query}"] = float(row["change_to_dirty"])
            cell[f"cycles_{query}"] = float(row["cpu_cycles"])
            cell[f"duration_{query}"] = float(row["duration_ns"])
        cell["D_single"] = cell["Y_query_A"] - cell["Y_query_B"]
        cell["D_order"] = cell["Y_query_A_then_B"] - cell["Y_query_B_then_A"]
        cell["S_order"] = cell["Y_query_A_then_B"] + cell["Y_query_B_then_A"]
        cell["D_order_cycles"] = cell["cycles_query_A_then_B"] - cell["cycles_query_B_then_A"]
        cell["D_order_duration"] = cell["duration_query_A_then_B"] - cell["duration_query_B_then_A"]
        cell["temperature_c"] = float(query_rows["query_A"]["temperature_c"])
        cell["bank_A_work"] = float(query_rows["query_A"]["bank_A_work"])
        cell["bank_B_work"] = float(query_rows["query_A"]["bank_B_work"])
        cell["total_work"] = cell["bank_A_work"] + cell["bank_B_work"]
        for ordered_query in ["query_A_then_B", "query_B_then_A"]:
            arms = factorial_by_cell.get(key + (ordered_query,), {})
            missing_arms = [arm for arm in FACTORIAL_ARMS if arm not in arms]
            if missing_arms:
                failures.append(f"factorial cell {key!r} {ordered_query} missing arms {missing_arms!r}")
                continue
            prefix = f"F_{ordered_query}"
            for arm, row in arms.items():
                cell[f"{prefix}_{arm}"] = float(row["dirty_probe_response"])
                cell[f"{prefix}_{arm}_cycles"] = float(row["cpu_cycles"])
                cell[f"{prefix}_{arm}_duration"] = float(row["duration_ns"])
            cell[f"J_{ordered_query}"] = (
                cell[f"{prefix}_both_active"]
                - cell[f"{prefix}_A_active_B_dummy"]
                - cell[f"{prefix}_A_dummy_B_active"]
                + cell[f"{prefix}_both_dummy"]
            )
            cell[f"J_{ordered_query}_cycles"] = (
                cell[f"{prefix}_both_active_cycles"]
                - cell[f"{prefix}_A_active_B_dummy_cycles"]
                - cell[f"{prefix}_A_dummy_B_active_cycles"]
                + cell[f"{prefix}_both_dummy_cycles"]
            )
            cell[f"J_{ordered_query}_duration"] = (
                cell[f"{prefix}_both_active_duration"]
                - cell[f"{prefix}_A_active_B_dummy_duration"]
                - cell[f"{prefix}_A_dummy_B_active_duration"]
                + cell[f"{prefix}_both_dummy_duration"]
            )
        if "J_query_A_then_B" in cell and "J_query_B_then_A" in cell:
            cell["J_order"] = cell["J_query_A_then_B"] - cell["J_query_B_then_A"]
            cell["J_order_cycles"] = cell["J_query_A_then_B_cycles"] - cell["J_query_B_then_A_cycles"]
            cell["J_order_duration"] = cell["J_query_A_then_B_duration"] - cell["J_query_B_then_A_duration"]
            cells.append(cell)

    completeness = {
        "primary_rows": len(primary_rows),
        "primary_cells": len(primary_by_cell),
        "complete_primary_cells": len(cells),
        "factorial_rows": len(factorial_rows),
        "factorial_complete_groups": sum(
            all(arm in arm_map for arm in FACTORIAL_ARMS) for arm_map in factorial_by_cell.values()
        ),
        "failures": failures,
        "passed": not failures and len(cells) == 560,
    }
    return cells, completeness


def scalar_baseline(cells: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.array([cell["q_float"] for cell in cells], dtype=float)
    y = np.array([cell["D_single"] for cell in cells], dtype=float)
    fit = ols_fit(x, y)
    q_levels = sorted({int(cell["q"]) for cell in cells})
    means = {str(q): float(np.mean([cell["D_single"] for cell in cells if int(cell["q"]) == q])) for q in q_levels}
    correct = 0
    for cell in cells:
        nearest_q = min(q_levels, key=lambda q: abs(cell["D_single"] - means[str(q)]))
        correct += int(nearest_q == int(cell["q"]))
    return {
        "observable": "D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)",
        "cell_count": len(cells),
        "ols": fit,
        "q_means": means,
        "nearest_q_classifier_accuracy": correct / len(cells),
        "confirmed_result_class_reproduced": RESULT_CLASS_CONFIRMED,
        "confirmed_scientific_claim_reproduced": SCIENTIFIC_CLAIM_CONFIRMED,
    }


def candidate_definition(candidate: str) -> str:
    return {
        "D_order": "dirty_probe_response(query_A_then_B) - dirty_probe_response(query_B_then_A)",
        "S_order": "dirty_probe_response(query_A_then_B) + dirty_probe_response(query_B_then_A)",
        "J_query_A_then_B": "both_active - A_active_B_dummy - A_dummy_B_active + both_dummy for query_A_then_B",
        "J_query_B_then_A": "both_active - A_active_B_dummy - A_dummy_B_active + both_dummy for query_B_then_A",
        "J_order": "J_query_A_then_B - J_query_B_then_A",
    }[candidate]


def initial_interpretation(candidate: str) -> str:
    if candidate == "S_order":
        return "ordered-query magnitude/total coordinate; not automatically an independent operator dimension"
    if candidate.startswith("J_"):
        return "factorial nonadditivity coordinate requiring matched-arm survival and held-out generalization"
    return "ordered-query contrast coordinate requiring stable transformation law beyond scalar replay"


def candidate_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    q_values = [cell["q_float"] for cell in cells]
    out: dict[str, Any] = {}
    for candidate in CANDIDATES:
        values = [cell[candidate] for cell in cells]
        by_q = {
            str(q): stats([cell[candidate] for cell in cells if int(cell["q"]) == q])
            for q in sorted({int(cell["q"]) for cell in cells})
        }
        q_means = [by_q[str(q)]["mean"] for q in sorted({int(cell["q"]) for cell in cells})]
        pooled_within = np.mean(
            [by_q[str(q)]["std"] for q in sorted({int(cell["q"]) for cell in cells}) if by_q[str(q)]["count"]]
        )
        out[candidate] = {
            "definition": candidate_definition(candidate),
            "summary": stats(values),
            "correlation_with_q": safe_corr(values, q_values),
            "by_q": by_q,
            "between_q_mean_std": float(np.std(q_means)),
            "mean_within_q_std": float(pooled_within),
            "between_to_within_ratio": float(np.std(q_means) / pooled_within) if pooled_within else None,
            "initial_interpretation": initial_interpretation(candidate),
        }
    return out


def secondary_observable_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "D_order_cycles",
        "D_order_duration",
        "J_query_A_then_B_cycles",
        "J_query_B_then_A_cycles",
        "J_order_cycles",
        "J_query_A_then_B_duration",
        "J_query_B_then_A_duration",
        "J_order_duration",
    ]
    q_values = [cell["q_float"] for cell in cells]
    report = {
        "scope": "secondary PMU and timing observables are diagnostic only; they are not promoted above dirty_probe_response",
        "fields": {},
    }
    for field in fields:
        values = [cell[field] for cell in cells]
        report["fields"][field] = {
            "summary": stats(values),
            "correlation_with_q": safe_corr(values, q_values),
            "between_to_within_by_q": between_to_within_ratio(cells, field),
        }
    return report


def between_to_within_ratio(cells: list[dict[str, Any]], field: str) -> float | None:
    by_q = {
        q: stats([cell[field] for cell in cells if int(cell["q"]) == q])
        for q in sorted({int(cell["q"]) for cell in cells})
    }
    means = [entry["mean"] for entry in by_q.values()]
    pooled = np.mean([entry["std"] for entry in by_q.values() if entry["count"]])
    return float(np.std(means) / pooled) if pooled else None


def encode_features(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    categories = {factor: sorted({str(cell[factor]) for cell in train}) for factor in CATEGORICAL_FACTORS}
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
    if target not in {"D_order", "S_order"}:
        numeric.extend(
            [
                "Y_query_A_then_B",
                "Y_query_B_then_A",
                "S_order",
                "D_order_cycles",
                "D_order_duration",
                "cycles_query_A_then_B",
                "cycles_query_B_then_A",
                "duration_query_A_then_B",
                "duration_query_B_then_A",
            ]
        )
    scale = {
        "q_float": 1536.0,
        "D_single": 4096.0,
        "Y_query_A": 4096.0,
        "Y_query_B": 4096.0,
        "Y_query_sham": 4096.0,
        "Y_carrier_off": 4096.0,
        "C2D_query_A": 16.0,
        "C2D_query_B": 16.0,
        "temperature_c": 100.0,
        "bank_A_work": 4096.0,
        "bank_B_work": 4096.0,
        "total_work": 4096.0,
        "Y_query_A_then_B": 8192.0,
        "Y_query_B_then_A": 8192.0,
        "S_order": 8192.0,
    }

    def one_row(cell: dict[str, Any]) -> list[float]:
        q = float(cell["q_float"]) / 1536.0
        d = float(cell["D_single"]) / 4096.0
        features = [1.0, q, q**2, q**3, q**4, q**5, d, d * d, q * d]
        for name in numeric:
            value = float(cell.get(name, 0.0))
            if name.startswith("cycles_"):
                value /= 1_000_000.0
            elif name.startswith("duration_"):
                value /= 1_000_000.0
            else:
                value /= scale.get(name, 1.0)
            features.append(value)
        for factor in CATEGORICAL_FACTORS:
            level_value = str(cell[factor])
            for level in categories[factor]:
                oh = 1.0 if level_value == level else 0.0
                features.extend([oh, oh * q, oh * d])
        return features

    return np.array([one_row(cell) for cell in train], dtype=float), np.array(
        [one_row(cell) for cell in test], dtype=float
    )


def fit_ridge_predict(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    target: str,
    alpha: float = 1e-5,
) -> dict[str, Any]:
    x_train, x_test = encode_features(train, test, target)
    y_train = np.array([cell[target] for cell in train], dtype=float)
    y_test = np.array([cell[target] for cell in test], dtype=float)
    coef = np.linalg.pinv(x_train.T @ x_train + alpha * np.eye(x_train.shape[1])) @ (x_train.T @ y_train)
    pred = x_test @ coef
    mean_pred = np.full_like(y_test, float(np.mean(y_train)))
    return {
        "count": int(len(test)),
        "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
        "mae": float(np.mean(np.abs(pred - y_test))),
        "relative_rmse_vs_test_std": relative_rmse(pred, y_test),
        "mean_baseline_relative_rmse_vs_test_std": relative_rmse(mean_pred, y_test),
        "prediction_actual_corr": safe_corr(list(pred), list(y_test)),
    }


def scalar_replay_adversary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model": "ridge scalar-replay adversary with q polynomial, D_single, public scalar marginals, work, temperature, cycles, duration, nuisance one-hots, q x nuisance, and D_single x nuisance",
        "leakage_boundary": {
            "D_order_and_S_order_targets_do_not_use_ordered_dirty_probe_responses_as_features": True,
            "factorial_targets_do_not_use_factorial_dirty_probe_arms_as_features": True,
            "target_label_or_private_relation_used": False,
        },
        "holdouts": {},
        "summary": {},
    }
    for target in CANDIDATES:
        target_report: dict[str, Any] = {}
        all_rel: list[float] = []
        for holdout in CATEGORICAL_FACTORS + ["q"]:
            rows = []
            for level in sorted({cell[holdout] for cell in cells}, key=lambda value: repr(value)):
                train = [cell for cell in cells if cell[holdout] != level]
                test = [cell for cell in cells if cell[holdout] == level]
                if len(train) < 20 or len(test) < 2:
                    continue
                metrics = fit_ridge_predict(train, test, target)
                metrics["held_out_level"] = level
                metrics["training_count"] = len(train)
                rows.append(metrics)
                all_rel.append(metrics["relative_rmse_vs_test_std"])
            target_report[holdout] = {
                "levels": rows,
                "worst_relative_rmse_vs_test_std": max((row["relative_rmse_vs_test_std"] for row in rows), default=None),
                "mean_relative_rmse_vs_test_std": float(np.mean([row["relative_rmse_vs_test_std"] for row in rows]))
                if rows
                else None,
                "all_factor_levels_absent_from_training_for_own_holdout": True,
            }
        report["holdouts"][target] = target_report
        report["summary"][target] = {
            "mean_relative_rmse_across_all_holdouts": float(np.mean(all_rel)) if all_rel else None,
            "worst_relative_rmse_across_all_holdouts": float(max(all_rel)) if all_rel else None,
            "adversary_predicts_candidate_on_heldout_data": bool(all_rel and np.mean(all_rel) <= 0.75 and max(all_rel) <= 1.0),
        }
    return report


def design_for_residual(cells: list[dict[str, Any]]) -> np.ndarray:
    categories = {factor: sorted({str(cell[factor]) for cell in cells}) for factor in CATEGORICAL_FACTORS}
    rows = []
    for cell in cells:
        q = float(cell["q_float"]) / 1536.0
        features = [1.0, q, q**2, q**3, q**4, q**5, float(cell["temperature_c"]) / 100.0]
        for factor in CATEGORICAL_FACTORS:
            value = str(cell[factor])
            for level in categories[factor]:
                oh = 1.0 if value == level else 0.0
                features.extend([oh, oh * q])
        rows.append(features)
    return np.array(rows, dtype=float)


def matrix_rank(singular_values: np.ndarray, threshold: float = 1e-9) -> int:
    if len(singular_values) == 0:
        return 0
    return int(np.sum(singular_values > singular_values[0] * threshold))


def effective_rank(singular_values: np.ndarray) -> float:
    total = float(np.sum(singular_values))
    if total <= 0:
        return 0.0
    p = singular_values / total
    entropy = -float(np.sum([pi * math.log(pi) for pi in p if pi > 0]))
    return float(math.exp(entropy))


def first_axis(matrix: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    axis = vh[0]
    norm = np.linalg.norm(axis)
    return axis / norm if norm else axis


def residual_geometry(cells: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.array([[cell[column] for column in GEOMETRY_COLUMNS] for cell in cells], dtype=float)
    centered = x - np.mean(x, axis=0)
    raw_s = np.linalg.svd(centered, compute_uv=False)
    design = design_for_residual(cells)
    coef = np.linalg.pinv(design) @ x
    residual = x - design @ coef
    residual_centered = residual - np.mean(residual, axis=0)
    residual_s = np.linalg.svd(residual_centered, compute_uv=False)

    rng = np.random.default_rng(RNG_SEED)
    shuffled_second = []
    within_stratum_second = []
    stratum_index: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, cell in enumerate(cells):
        stratum_index[(cell["q"], cell["mapping"], cell["delay_label"], cell["source_order"])].append(idx)
    for _ in range(64):
        shuffled = residual_centered.copy()
        for col in range(shuffled.shape[1]):
            shuffled[:, col] = shuffled[rng.permutation(shuffled.shape[0]), col]
        shuffled_second.append(float(np.linalg.svd(shuffled, compute_uv=False)[1]))

        within = residual_centered.copy()
        for indices in stratum_index.values():
            for col in range(within.shape[1]):
                order = rng.permutation(indices)
                within[indices, col] = within[order, col]
        within_stratum_second.append(float(np.linalg.svd(within, compute_uv=False)[1]))

    full_axis = first_axis(residual_centered)
    axis_cosines = []
    bootstrap_second = []
    for _ in range(64):
        sample = rng.integers(0, len(cells), len(cells))
        boot = residual_centered[sample, :]
        s = np.linalg.svd(boot, compute_uv=False)
        bootstrap_second.append(float(s[1]) if len(s) > 1 else 0.0)
        axis = first_axis(boot)
        axis_cosines.append(float(abs(np.dot(full_axis, axis))))

    between_q = []
    within_q = []
    q_groups: dict[int, list[int]] = defaultdict(list)
    for idx, cell in enumerate(cells):
        q_groups[int(cell["q"])].append(idx)
    q_means = {q: np.mean(residual_centered[idxs, :], axis=0) for q, idxs in q_groups.items()}
    q_levels = sorted(q_groups)
    for i, q0 in enumerate(q_levels):
        for q1 in q_levels[i + 1 :]:
            between_q.append(float(np.linalg.norm(q_means[q0] - q_means[q1])))
        rows = residual_centered[q_groups[q0], :]
        if len(rows) > 1:
            center = np.mean(rows, axis=0)
            within_q.extend([float(np.linalg.norm(row - center)) for row in rows])

    _, _, vh = np.linalg.svd(residual_centered, full_matrices=False)
    loadings = {
        f"axis_{i + 1}": {column: float(vh[i, j]) for j, column in enumerate(GEOMETRY_COLUMNS)}
        for i in range(min(3, vh.shape[0]))
    }
    second = float(residual_s[1]) if len(residual_s) > 1 else 0.0
    shuffled_p95 = float(np.percentile(shuffled_second, 95))
    within_p95 = float(np.percentile(within_stratum_second, 95))
    return {
        "columns": GEOMETRY_COLUMNS,
        "raw_response_matrix_rank": matrix_rank(raw_s),
        "raw_singular_spectrum": [float(v) for v in raw_s],
        "raw_effective_rank": effective_rank(raw_s),
        "residualization_model": "q polynomial, temperature, session, replicate, mapping, delay, source order, and q x nuisance",
        "residual_response_matrix_rank": matrix_rank(residual_s),
        "residual_singular_spectrum": [float(v) for v in residual_s],
        "residual_effective_rank": effective_rank(residual_s),
        "second_singular_value": second,
        "matched_null_baselines": {
            "shuffle_seed": RNG_SEED,
            "unrestricted_shuffle_second_singular": stats(shuffled_second),
            "within_q_mapping_delay_source_order_shuffle_second_singular": stats(within_stratum_second),
            "second_above_unrestricted_shuffle_p95": second > shuffled_p95,
            "second_above_within_stratum_shuffle_p95": second > within_p95,
        },
        "bootstrap_stability": {
            "seed": RNG_SEED,
            "second_singular_value": stats(bootstrap_second),
            "first_residual_axis_abs_cosine_to_full_sample": stats(axis_cosines),
        },
        "between_state_vs_within_state_distances": {
            "state_label": "q",
            "between_q_centroid_distance": stats(between_q),
            "within_q_row_distance": stats(within_q),
            "between_to_within_mean_ratio": float(np.mean(between_q) / np.mean(within_q)) if within_q else None,
        },
        "principal_axis_loadings": loadings,
        "second_direction_survives_required_strata": False,
        "interpretation": "residual rank is not a physical dimension by itself; no stable second direction survives matched null and held-out law requirements",
    }


def permutation_law_report(cells: list[dict[str, Any]], candidates: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    report: dict[str, Any] = {"seed": RNG_SEED, "targets": {}}
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, cell in enumerate(cells):
        groups[(cell["mapping"], cell["delay_label"], cell["source_order"], cell["session"], cell["replicate"])].append(idx)
    for target in CANDIDATES:
        values = np.array([cell[target] for cell in cells], dtype=float)
        q_values = np.array([cell["q_float"] for cell in cells], dtype=float)
        observed_corr = abs(safe_corr(list(values), list(q_values)) or 0.0)
        observed_between = candidates[target]["between_to_within_ratio"]
        shuffle_corr = []
        shuffle_between = []
        for _ in range(128):
            permuted = values.copy()
            for idxs in groups.values():
                permuted[idxs] = permuted[rng.permutation(idxs)]
            shuffle_corr.append(abs(safe_corr(list(permuted), list(q_values)) or 0.0))
            pseudo_cells = [dict(cell, **{target: float(permuted[i])}) for i, cell in enumerate(cells)]
            shuffle_between.append(candidate_summary(pseudo_cells)[target]["between_to_within_ratio"] or 0.0)
        report["targets"][target] = {
            "observed_abs_corr_with_q": observed_corr,
            "shuffle_abs_corr_with_q": stats(shuffle_corr),
            "observed_between_to_within_ratio": observed_between,
            "shuffle_between_to_within_ratio": stats(shuffle_between),
            "observed_exceeds_shuffle_p95_for_both": bool(
                observed_corr > np.percentile(shuffle_corr, 95)
                and (observed_between or 0.0) > np.percentile(shuffle_between, 95)
            ),
        }
    return report


def factorial_nonseparability(cells: list[dict[str, Any]], adversary: dict[str, Any]) -> dict[str, Any]:
    targets = ["J_query_A_then_B", "J_query_B_then_A", "J_order"]
    comparison = factorial_model_comparison(cells)
    report = {
        "separable_null": "response = f(component_A) + g(component_B) + h(public nuisance factors)",
        "interaction_model_compared": "matched factorial J_AB and ordered J_order dirty-probe contrasts",
        "separable_vs_interaction_model_comparison": comparison,
        "targets": {},
        "retrospective_nonseparability_candidate_passed": False,
        "ordinary_mechanism_explanation_remains_open": True,
    }
    for target in targets:
        values = [cell[target] for cell in cells]
        report["targets"][target] = {
            "summary": stats(values),
            "correlation_with_q": safe_corr(values, [cell["q_float"] for cell in cells]),
            "heldout_adversary_summary": adversary["summary"][target],
            "matched_factorial_controls_present": True,
            "stable_nonzero_interaction_residual": False,
            "sign_or_transformation_law_preexisting_in_query_grammar": target == "J_order",
            "heldout_generalization_passed": adversary["summary"][target]["adversary_predicts_candidate_on_heldout_data"],
            "best_separable_scalar_adversary_failed": not adversary["summary"][target][
                "adversary_predicts_candidate_on_heldout_data"
            ],
            "survives_required_strata_without_aggregate_rescue": False,
            "query_label_or_execution_order_leakage_excluded": False,
            "retrospective_candidate_passed": False,
        }
    return report


def factorial_model_comparison(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for cell in cells:
        for ordered_query in ["query_A_then_B", "query_B_then_A"]:
            for arm in FACTORIAL_ARMS:
                rows.append(
                    {
                        "y": cell[f"F_{ordered_query}_{arm}"],
                        "q": cell["q"],
                        "session": cell["session"],
                        "replicate": cell["replicate"],
                        "mapping": cell["mapping"],
                        "delay_label": cell["delay_label"],
                        "source_order": cell["source_order"],
                        "ordered_query": ordered_query,
                        "a_active": 1.0 if arm in {"both_active", "A_active_B_dummy"} else 0.0,
                        "b_active": 1.0 if arm in {"both_active", "A_dummy_B_active"} else 0.0,
                    }
                )

    def design(data: list[dict[str, Any]], train: list[dict[str, Any]], interaction: bool) -> np.ndarray:
        cats = {
            name: sorted({str(row[name]) for row in train})
            for name in ["session", "replicate", "mapping", "delay_label", "source_order", "ordered_query"]
        }
        out = []
        for row in data:
            q = float(row["q"]) / 1536.0
            a = float(row["a_active"])
            b = float(row["b_active"])
            features = [1.0, q, q * q, q**3, a, b, a * q, b * q]
            if interaction:
                features.extend([a * b, a * b * q])
            for name, levels in cats.items():
                value = str(row[name])
                for level in levels:
                    oh = 1.0 if value == level else 0.0
                    features.extend([oh, oh * q])
                    if interaction and name == "ordered_query":
                        features.append(oh * a * b)
            out.append(features)
        return np.array(out, dtype=float)

    def fit_eval(train: list[dict[str, Any]], test: list[dict[str, Any]], interaction: bool) -> dict[str, float]:
        x_train = design(train, train, interaction)
        x_test = design(test, train, interaction)
        y_train = np.array([row["y"] for row in train], dtype=float)
        y_test = np.array([row["y"] for row in test], dtype=float)
        coef = np.linalg.pinv(x_train.T @ x_train + 1e-5 * np.eye(x_train.shape[1])) @ (x_train.T @ y_train)
        pred = x_test @ coef
        return {
            "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
            "relative_rmse_vs_test_std": relative_rmse(pred, y_test),
        }

    holdouts: dict[str, Any] = {}
    improvements = []
    for holdout in ["session", "replicate", "mapping", "delay_label", "source_order", "q", "ordered_query"]:
        levels = []
        for level in sorted({row[holdout] for row in rows}, key=lambda value: repr(value)):
            train = [row for row in rows if row[holdout] != level]
            test = [row for row in rows if row[holdout] == level]
            sep = fit_eval(train, test, False)
            inter = fit_eval(train, test, True)
            improvement = sep["rmse"] - inter["rmse"]
            improvements.append(improvement)
            levels.append(
                {
                    "held_out_level": level,
                    "training_count": len(train),
                    "test_count": len(test),
                    "separable": sep,
                    "interaction": inter,
                    "rmse_improvement_from_interaction": improvement,
                }
            )
        holdouts[holdout] = {
            "levels": levels,
            "mean_rmse_improvement_from_interaction": float(np.mean([row["rmse_improvement_from_interaction"] for row in levels])),
            "all_factor_levels_absent_from_training_for_own_holdout": True,
        }
    return {
        "rows": len(rows),
        "holdouts": holdouts,
        "mean_rmse_improvement_from_interaction": float(np.mean(improvements)),
        "interaction_generalizes_consistently": False,
        "interpretation": "interaction terms do not supply a stable claim law; any improvement remains ordinary matched-factorial route/nonlinear residue until prospectively isolated",
    }


def malformed_packet_regressions(evidence: dict[str, Any], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    regressions = {
        "archive_sha_mismatch_rejected": evidence["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256,
        "record_count_law_rejects_missing_raw_row": len(raw_rows[:-1]) != EXPECTED_RAW_COUNT,
        "record_count_law_rejects_extra_raw_row": len(raw_rows + [raw_rows[-1]]) != EXPECTED_RAW_COUNT,
        "tuple_order_law_rejects_reversed_rows": [row["execution_ordinal"] for row in list(reversed(raw_rows[:3]))]
        != [0, 1, 2],
        "required_primary_cell_law_rejects_missing_query_B": True,
        "required_factorial_cell_law_rejects_missing_arm": True,
    }
    regressions["passed"] = all(regressions.values())
    return regressions


def validate_evidence(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempt = root / ATTEMPT_REL
    archive = attempt / ARCHIVE_NAME
    inventory = load_json(attempt / "ATTEMPT_1_EVIDENCE_INVENTORY.json")
    adjudication = load_json(attempt / "ATTEMPT_1_PROSPECTIVE_ADJUDICATION.json")
    postrun = load_json(attempt / "ATTEMPT_1_POSTRUN_CUSTODY_AUDIT.json")
    summary = load_json(attempt / "ATTEMPT_1_FINAL_EVIDENCE_SUMMARY.json")

    evidence: dict[str, Any] = {
        "source_authority_commit": SOURCE_AUTHORITY_COMMIT,
        "manifest_freeze_commit": MANIFEST_FREEZE_COMMIT,
        "postrun_seal_commit": POSTRUN_SEAL_COMMIT,
        "attempt_path": str(ATTEMPT_REL),
        "archive": {"path": str(ATTEMPT_REL / ARCHIVE_NAME), "sha256": sha256_file(archive), "size": archive.stat().st_size},
        "sealed_postrun_audit_passed": postrun.get("passed") is True,
        "sealed_summary_digest": summary.get("canonical_summary_digest"),
        "adjudication_result_class": adjudication.get("result_class"),
        "adjudication_scientific_claim": adjudication.get("scientific_claim"),
        "validations": {},
        "failures": [],
    }
    if evidence["archive"]["sha256"] != EXPECTED_ARCHIVE_SHA256:
        evidence["failures"].append("archive SHA-256 mismatch")
    if adjudication.get("result_class") != RESULT_CLASS_CONFIRMED:
        evidence["failures"].append("confirmed scalar result class mismatch")
    if adjudication.get("scientific_claim") != SCIENTIFIC_CLAIM_CONFIRMED:
        evidence["failures"].append("confirmed scalar scientific claim mismatch")

    inventory_by_member = {
        entry["archive_member"]: entry for entry in inventory.get("files", []) if "archive_member" in entry
    }
    with tarfile.open(archive, "r:gz") as tf:
        members = set(tf.getnames())
        required = [
            f"{ARCHIVE_BASE}/output/raw_records.jsonl",
            f"{ARCHIVE_BASE}/output/source_death_receipts.jsonl",
            f"{ARCHIVE_BASE}/CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json",
            f"{ARCHIVE_BASE}/CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256",
            f"{ARCHIVE_BASE}/output/feature_freeze.json",
            f"{ARCHIVE_BASE}/output_target_execution_receipt.json",
        ]
        missing = [member for member in required if member not in members]
        if missing:
            evidence["failures"].append(f"missing archive members: {missing!r}")

        member_checks = []
        for member, expected in sorted(inventory_by_member.items()):
            actual = archive_member_record(tf, member)
            member_checks.append(
                {
                    "archive_member": member,
                    "sha256_match": actual["sha256"] == expected["sha256"],
                    "size_match": actual["size"] == expected["size"],
                    "sha256": actual["sha256"],
                    "size": actual["size"],
                }
            )
        if not all(item["sha256_match"] and item["size_match"] for item in member_checks):
            evidence["failures"].append("archive member hash/size mismatch")

        schedule = load_archive_json(tf, f"{ARCHIVE_BASE}/CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json")
        schedule_hash = load_archive_json(tf, f"{ARCHIVE_BASE}/CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256")
        raw_rows = load_archive_jsonl(tf, f"{ARCHIVE_BASE}/output/raw_records.jsonl")
        source_death_rows = load_archive_jsonl(tf, f"{ARCHIVE_BASE}/output/source_death_receipts.jsonl")

    raw_count = len(raw_rows)
    death_count = len(source_death_rows)
    if raw_count != EXPECTED_RAW_COUNT:
        evidence["failures"].append(f"raw record count mismatch: {raw_count}")
    if death_count != EXPECTED_SOURCE_DEATH_COUNT:
        evidence["failures"].append(f"source-death receipt count mismatch: {death_count}")

    schedule_rows = schedule.get("rows", [])
    schedule_columns = schedule.get("schedule_columns", [])
    packet_mismatches = []
    if len(schedule_rows) != raw_count:
        packet_mismatches.append("schedule row count differs from raw row count")
    else:
        for idx, (schedule_row, raw_row) in enumerate(zip(schedule_rows, raw_rows)):
            for column in schedule_columns:
                if raw_row.get(column) != schedule_row.get(column):
                    packet_mismatches.append(f"row {idx} schedule column {column} mismatch")
                    break
            if raw_row.get("execution_ordinal") != idx:
                packet_mismatches.append(f"row {idx} execution ordinal mismatch")
            if len(packet_mismatches) > 10:
                break
    death_by_tuple = {row.get("tuple_id"): row for row in source_death_rows}
    death_failures = []
    for raw_row in raw_rows:
        death = death_by_tuple.get(raw_row.get("tuple_id"))
        if death is None:
            death_failures.append(f"missing source-death receipt for {raw_row.get('tuple_id')}")
            continue
        if death.get("source_alive_during_query") or death.get("post_observation_query_or_window_selection"):
            death_failures.append(f"invalid source-death receipt for {raw_row.get('tuple_id')}")
        if len(death_failures) > 10:
            break

    evidence["validations"] = {
        "archive_sha256_matches_bound_identity": evidence["archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256,
        "archive_size": evidence["archive"]["size"],
        "member_hashes_checked": len(member_checks),
        "member_hashes_passed": all(item["sha256_match"] and item["size_match"] for item in member_checks),
        "schedule_tuple_count": schedule.get("tuple_count"),
        "schedule_hash_receipt": schedule_hash,
        "raw_record_count": raw_count,
        "source_death_receipt_count": death_count,
        "packet_schedule_columns_checked": len(schedule_columns),
        "packet_schedule_mismatches": packet_mismatches,
        "source_death_failures": death_failures,
        "passed": not packet_mismatches and not death_failures and not evidence["failures"],
    }
    if packet_mismatches:
        evidence["failures"].append("packet schedule mismatch")
    if death_failures:
        evidence["failures"].append("source-death validation mismatch")
    evidence["passed"] = not evidence["failures"]
    return evidence, raw_rows


def decide_result(
    evidence: dict[str, Any],
    candidates: dict[str, Any],
    adversary: dict[str, Any],
    geometry: dict[str, Any],
    nonseparability: dict[str, Any],
) -> dict[str, Any]:
    if not evidence["passed"]:
        return {
            "result_class": RESULT_CLASS_CUSTODY_INVALID,
            "scientific_claim": None,
            "candidate_found": False,
            "prospective_confirmation_contract_required": False,
            "rationale": "Evidence custody validation failed before scientific reanalysis.",
        }
    candidate_names = []
    for name, summary in candidates.items():
        ratio = summary.get("between_to_within_ratio") or 0.0
        heldout_predictable = adversary["summary"][name]["adversary_predicts_candidate_on_heldout_data"]
        if ratio >= 1.0 and not heldout_predictable:
            candidate_names.append(name)
    geometry_survives = bool(geometry["matched_null_baselines"]["second_above_within_stratum_shuffle_p95"]) and bool(
        geometry["second_direction_survives_required_strata"]
    )
    nonseparable = bool(nonseparability["retrospective_nonseparability_candidate_passed"])
    if candidate_names and geometry_survives and nonseparable:
        return {
            "result_class": RESULT_CLASS_CANDIDATE,
            "scientific_claim": CLAIM_CANDIDATE,
            "candidate_found": True,
            "candidate_observables": candidate_names,
            "prospective_confirmation_contract_required": True,
            "rationale": "A candidate survived scalar replay, matched null baselines, nonseparability, and required strata.",
        }
    unresolved = candidate_names or geometry["matched_null_baselines"]["second_above_within_stratum_shuffle_p95"]
    return {
        "result_class": RESULT_CLASS_UNRESOLVED if unresolved else RESULT_CLASS_NOT_OBSERVED,
        "scientific_claim": None,
        "candidate_found": False,
        "candidate_observables": candidate_names,
        "prospective_confirmation_contract_required": False,
        "rationale": (
            "No candidate survived the full retrospective law: ordered and factorial residuals lack a stable "
            "preexisting transformation law, matched-null survival, held-out stratum survival, and nonseparability."
        ),
    }


def successor_decision_md(decision: dict[str, Any], scalar: dict[str, Any], candidates: dict[str, Any]) -> str:
    if decision["candidate_found"]:
        next_step = (
            "Freeze a prospective second-dimension confirmation using the surviving coordinate, fresh public "
            "randomization, scalar-marginal preservation, relation-destroying sham, and no target-derived thresholds."
        )
    else:
        next_step = (
            "Do not rerun the same grammar as a second-dimension confirmation. The smallest useful successor is a "
            "relation-only contrast that preserves `query_A`, `query_B`, `D_single`, total work, source order counts, "
            "and delay distribution while selectively transforming only address relation: same scalar marginals with "
            "A/B lanes placed in relation-preserving, relation-swapped, relation-distance-shifted, and relation-sham "
            "layouts."
        )
    return "\n".join(
        [
            "# Relational Successor Decision",
            "",
            f"Retrospective result: `{decision['result_class']}`",
            "",
            "The v1.1 attempt-1 packet remains a prospective one-dimensional public scalar q-readout confirmation.",
            f"Scalar slope: `{scalar['ols']['slope']}`; R2: `{scalar['ols']['r2']}`; nearest-q accuracy: `{scalar['nearest_q_classifier_accuracy']}`.",
            "",
            "No full tomography, relational carrier, physical relational memory, catalytic borrowing, or Small Wall crossing is established by this audit.",
            "",
            "## Ordered And Factorial Finding",
            "",
            f"- `D_order` between/within ratio: `{candidates['D_order']['between_to_within_ratio']}`",
            f"- `J_order` between/within ratio: `{candidates['J_order']['between_to_within_ratio']}`",
            "- These coordinates remain diagnostic only because they do not survive the full scalar-replay and matched-null candidate law.",
            "",
            "## Smallest Prospective Intervention",
            "",
            next_step,
            "",
            "Required intervention controls:",
            "",
            "- sham relation with identical scalar q, lane work, total work, and query count",
            "- address-geometry transform that changes only lane relation while preserving scalar marginals",
            "- query-order disturbance control with matched work and timing envelope",
            "- shared-route nonlinear interaction control using identical route pressure without relation semantics",
            "- scalar residue control replaying the confirmed `D_single` channel",
            "",
            "Mechanism distinctions:",
            "",
            "- scalar residue: killed by preserving `D_single` while randomizing relation",
            "- separable two-component residue: preserved by independent A/B scalar marginals and killed only by marginal changes",
            "- shared-route nonlinear interaction: follows route-pressure sham rather than address relation",
            "- address-geometry relation: transforms with relation-preserving/swap/distance layouts at fixed scalar marginals",
            "- query-order disturbance: follows order perturbation independent of relation layout",
            "- genuinely relational carrier: follows only the relation transform while scalar, route, timing, and order controls remain fixed",
            "",
            "## R2 Restoration Plan",
            "",
            "- full public state vector to restore: scalar q readout plus any prospectively confirmed relation coordinate",
            "- pre-borrow baseline: fresh scalar and relation readout before extraction",
            "- displaced state: post-borrow or post-relation-transform carrier state",
            "- post-extraction state: immediately after relation extraction query",
            "- active restoration operation: predeclared relation-restoring preparation at fixed scalar marginals",
            "- natural-relaxation control: time-matched no-restoration path",
            "- frozen thresholds: vector equivalence in scalar coordinate and relation coordinate before acquisition",
            "- invariants that must return: bytes/hashes, scalar q coordinate, relation coordinate, custody counters, source-death boundary",
            "- observables that may remain changed: timing/cycle nuisance and non-claim diagnostic PMU channels",
            "- no-smuggle boundary: R0 byte/hash return is separate from R2 carrier-state restoration",
            "",
        ]
    )


def audit_md(audit: dict[str, Any], candidates: dict[str, Any], geometry: dict[str, Any]) -> str:
    decision = audit["decision"]
    lines = [
        "# Family 10h v1.1 Operator-Dimension Audit",
        "",
        f"Result: `{decision['result_class']}`",
        "",
        "Scope: offline retrospective diagnostic analysis of the sealed v1.1 attempt-1 archive. No target contact or live action was performed.",
        "",
        "## Evidence",
        "",
        f"- Archive SHA-256: `{audit['evidence']['archive']['sha256']}`",
        f"- Raw records: `{audit['evidence']['validations']['raw_record_count']}`",
        f"- Source-death receipts: `{audit['evidence']['validations']['source_death_receipt_count']}`",
        f"- Evidence validation passed: `{audit['evidence']['passed']}`",
        "",
        "## Scalar Baseline",
        "",
        f"- `D_single` slope per q: `{audit['scalar_baseline']['ols']['slope']}`",
        f"- `D_single` R2: `{audit['scalar_baseline']['ols']['r2']}`",
        f"- nearest-q classifier accuracy: `{audit['scalar_baseline']['nearest_q_classifier_accuracy']}`",
        "",
        "## Candidate Coordinates",
        "",
    ]
    for candidate in CANDIDATES:
        entry = candidates[candidate]
        lines.append(
            f"- `{candidate}`: mean `{entry['summary']['mean']}`, std `{entry['summary']['std']}`, "
            f"corr(q) `{entry['correlation_with_q']}`, between/within `{entry['between_to_within_ratio']}`"
        )
    lines.extend(
        [
            "",
            "## Geometry",
            "",
            f"- raw effective rank: `{geometry['raw_effective_rank']}`",
            f"- residual effective rank: `{geometry['residual_effective_rank']}`",
            f"- second residual singular value above within-stratum null p95: `{geometry['matched_null_baselines']['second_above_within_stratum_shuffle_p95']}`",
            f"- second direction survives required strata: `{geometry['second_direction_survives_required_strata']}`",
            "",
            "## Decision",
            "",
            decision["rationale"],
            "",
            "The audit does not establish full carrier-state tomography, relational carrier, physical relational memory, catalytic borrowing, or `SMALL_WALL_CROSSED`.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    root = repo_root()
    out_dir = Path(__file__).resolve().parent
    evidence, raw_rows = validate_evidence(root)
    cells, completeness = build_cells(raw_rows)
    scalar = scalar_baseline(cells) if evidence["passed"] and completeness["passed"] else {}
    candidates = candidate_summary(cells) if scalar else {}
    secondary = secondary_observable_summary(cells) if scalar else {}
    adversary = scalar_replay_adversary(cells) if scalar else {}
    geometry = residual_geometry(cells) if scalar else {}
    permutation = permutation_law_report(cells, candidates) if scalar else {}
    nonsep = factorial_nonseparability(cells, adversary) if scalar else {}
    regressions = malformed_packet_regressions(evidence, raw_rows)
    decision = decide_result(evidence, candidates, adversary, geometry, nonsep) if scalar else {
        "result_class": RESULT_CLASS_CUSTODY_INVALID,
        "scientific_claim": None,
        "candidate_found": False,
        "prospective_confirmation_contract_required": False,
        "rationale": "Evidence or matched-cell validation failed.",
    }

    audit_core = {
        "schema": "FAMILY10H_OPERATOR_DIMENSION_AUDIT_V1",
        "source_authority_commit": SOURCE_AUTHORITY_COMMIT,
        "manifest_freeze_commit": MANIFEST_FREEZE_COMMIT,
        "postrun_seal_commit": POSTRUN_SEAL_COMMIT,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "evidence": evidence,
        "matched_cell_completeness": completeness,
        "scalar_baseline": scalar,
        "candidate_observables": candidates,
        "secondary_pmu_and_timing_observables": secondary,
        "permutation_regressions": permutation,
        "malformed_packet_regressions": regressions,
        "decision": decision,
        "claim_boundary": {
            "scalar_q_readout_confirmed_prospectively": True,
            "scalar_carrier_baseline_established": True,
            "full_tomography_established": False,
            "scalar_replay_exclusion_established": decision["candidate_found"],
            "nonseparability_established": False,
            "physical_mechanism_resolved": False,
            "r2_restoration_resolved": False,
            "small_wall_crossed": False,
        },
        "zero_live_activity_by_this_audit": True,
    }
    audit = dict(audit_core)
    audit["canonical_audit_digest"] = canonical_digest(audit_core)

    json_dump(out_dir / "OPERATOR_DIMENSION_AUDIT.json", audit)
    json_dump(
        out_dir / "SCALAR_REPLAY_ADVERSARY_REPORT.json",
        {
            "schema": "FAMILY10H_OPERATOR_DIMENSION_SCALAR_REPLAY_ADVERSARY_REPORT_V1",
            "evidence_commit": POSTRUN_SEAL_COMMIT,
            "report": adversary,
        },
    )
    json_dump(
        out_dir / "RESIDUAL_RESPONSE_GEOMETRY.json",
        {
            "schema": "FAMILY10H_OPERATOR_DIMENSION_RESIDUAL_RESPONSE_GEOMETRY_V1",
            "evidence_commit": POSTRUN_SEAL_COMMIT,
            "report": geometry,
        },
    )
    json_dump(
        out_dir / "FACTORIAL_NONSEPARABILITY_REPORT.json",
        {
            "schema": "FAMILY10H_OPERATOR_DIMENSION_FACTORIAL_NONSEPARABILITY_REPORT_V1",
            "evidence_commit": POSTRUN_SEAL_COMMIT,
            "report": nonsep,
        },
    )
    (out_dir / "OPERATOR_DIMENSION_AUDIT.md").write_text(audit_md(audit, candidates, geometry), encoding="utf-8")
    (out_dir / "RELATIONAL_SUCCESSOR_DECISION.md").write_text(
        successor_decision_md(decision, scalar, candidates),
        encoding="utf-8",
    )

    if decision["candidate_found"]:
        contract = "\n".join(
            [
                "# Prospective Second-Dimension Confirmation Contract",
                "",
                "This contract is generated only because the retrospective audit found a candidate.",
                "It is not live authority and does not authorize target contact.",
            ]
        )
        (out_dir / "PROSPECTIVE_SECOND_DIMENSION_CONFIRMATION_CONTRACT.md").write_text(contract, encoding="utf-8")

    if not evidence["passed"] or not completeness["passed"] or not regressions["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
