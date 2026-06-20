#!/usr/bin/env python3
"""Complex linear geometry primitives for Phase 6B.5C.

The module preserves complex vectors until explicitly reported metrics.  It does
not implement a winner-first classifier, alter the frozen carrier verdict, or
claim identification of the complete physical operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

EPS = 1e-12


def wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def complex_vector(codeword: np.ndarray, theta_idx: int, phase_levels: int) -> np.ndarray:
    theta = 2.0 * np.pi * float(theta_idx) / float(phase_levels)
    return np.asarray(codeword, dtype=np.complex128) * np.exp(1j * theta)


def normalized_residual(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.complex128)
    predicted = np.asarray(predicted, dtype=np.complex128)
    return float(np.linalg.norm(observed - predicted) / (np.linalg.norm(observed) + EPS))


def phase_aligned_residual(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.complex128)
    predicted = np.asarray(predicted, dtype=np.complex128)
    cross = np.vdot(predicted, observed)
    phase = np.angle(cross) if abs(cross) > EPS else 0.0
    aligned = predicted * np.exp(1j * phase)
    return normalized_residual(observed, aligned)


def cosine_similarity(observed: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.linalg.norm(observed) * np.linalg.norm(predicted) + EPS
    return float(abs(np.vdot(predicted, observed)) / denom)


def phase_estimate(observed: np.ndarray, zero_phase_prediction: np.ndarray) -> float:
    return float(np.angle(np.vdot(zero_phase_prediction, observed)))


def circular_mean_abs(errors: Iterable[float]) -> float:
    values = np.asarray(list(errors), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(wrap_phase(values))))


def circular_resultant(errors: Iterable[float]) -> float:
    values = np.asarray(list(errors), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(abs(np.mean(np.exp(1j * values))))


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "q05": None, "median": None, "q95": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


@dataclass(frozen=True)
class ChartSpec:
    chart_id: str
    family: str
    rank: int | None = None
    ridge: float = 0.0


@dataclass
class ComplexChart:
    spec: ChartSpec
    matrix: np.ndarray
    fit_condition: float
    calibration_rows: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.complex128) @ self.matrix

    def to_json(self) -> dict[str, Any]:
        return {
            "chart_id": self.spec.chart_id,
            "family": self.spec.family,
            "rank": self.spec.rank,
            "ridge": self.spec.ridge,
            "fit_condition": self.fit_condition,
            "calibration_rows": self.calibration_rows,
            "matrix_real": self.matrix.real.tolist(),
            "matrix_imag": self.matrix.imag.tolist(),
        }


CHART_LADDER = (
    ChartSpec("C0_scalar", "scalar"),
    ChartSpec("C1_diagonal", "diagonal"),
    ChartSpec("C2_rank4", "low_rank", rank=4, ridge=1e-4),
    ChartSpec("C3_full_ridge", "full", ridge=1e-3),
)


def _condition_number(matrix: np.ndarray) -> float:
    try:
        value = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def fit_chart(spec: ChartSpec, x_rows: np.ndarray, z_rows: np.ndarray) -> ComplexChart:
    x_rows = np.asarray(x_rows, dtype=np.complex128)
    z_rows = np.asarray(z_rows, dtype=np.complex128)
    if x_rows.ndim != 2 or z_rows.ndim != 2 or x_rows.shape != z_rows.shape:
        raise ValueError("x_rows and z_rows must be equal two-dimensional arrays")
    nrows, nbin = x_rows.shape
    if nrows == 0:
        raise ValueError("at least one calibration row is required")

    if spec.family == "scalar":
        denom = np.vdot(x_rows, x_rows).real + EPS
        gain = np.vdot(x_rows, z_rows) / denom
        matrix = np.eye(nbin, dtype=np.complex128) * gain
        condition = 1.0
    elif spec.family == "diagonal":
        gain = np.zeros(nbin, dtype=np.complex128)
        for b in range(nbin):
            denom = np.vdot(x_rows[:, b], x_rows[:, b]).real + EPS
            gain[b] = np.vdot(x_rows[:, b], z_rows[:, b]) / denom
        matrix = np.diag(gain)
        nonzero = np.abs(gain) > EPS
        condition = float(np.max(np.abs(gain[nonzero])) / (np.min(np.abs(gain[nonzero])) + EPS)) if np.any(nonzero) else float("inf")
    elif spec.family in {"low_rank", "full"}:
        gram = x_rows.conj().T @ x_rows
        regularized = gram + float(spec.ridge) * np.eye(nbin)
        rhs = x_rows.conj().T @ z_rows
        try:
            full_matrix = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            full_matrix = np.linalg.pinv(regularized) @ rhs
        condition = _condition_number(regularized)
        if spec.family == "low_rank":
            u, singular, vh = np.linalg.svd(full_matrix, full_matrices=False)
            rank = min(int(spec.rank or 1), len(singular))
            matrix = (u[:, :rank] * singular[:rank]) @ vh[:rank, :]
        else:
            matrix = full_matrix
    else:
        raise ValueError(f"unknown chart family: {spec.family}")

    return ComplexChart(spec=spec, matrix=matrix, fit_condition=condition, calibration_rows=nrows)


def chart_validation(chart: ComplexChart, rows: list[dict[str, Any]], codebook: dict[str, np.ndarray], phase_levels: int) -> dict[str, Any]:
    residuals: list[float] = []
    aligned: list[float] = []
    margins: list[float] = []
    positive = 0
    mode_names = list(codebook)
    for row in rows:
        actual = str(row["actual_mode"])
        theta_idx = int(row["theta_idx"])
        predicted_actual = chart.predict(complex_vector(codebook[actual], theta_idx, phase_levels))
        actual_residual = normalized_residual(row["z"], predicted_actual)
        alternatives = [
            normalized_residual(row["z"], chart.predict(complex_vector(codebook[name], theta_idx, phase_levels)))
            for name in mode_names if name != actual
        ]
        margin = min(alternatives, default=actual_residual) - actual_residual
        residuals.append(actual_residual)
        aligned.append(phase_aligned_residual(row["z"], predicted_actual))
        margins.append(margin)
        positive += margin > 0.0
    return {
        "rows": len(rows),
        "normalized_residual": quantiles(residuals),
        "phase_aligned_residual": quantiles(aligned),
        "actual_mode_margin": quantiles(margins),
        "positive_margin_fraction": float(positive / len(rows)) if rows else None,
    }


def select_chart(validations: list[tuple[ComplexChart, dict[str, Any]]]) -> tuple[ComplexChart, dict[str, Any]]:
    """Select the smallest predeclared chart satisfying calibration-only criteria.

    Criteria are intentionally broad and diagnostic: median normalized residual <= 0.55,
    positive actual-mode margin on >= 75% of calibration-validation rows, and finite
    condition <= 1e10.  If no chart qualifies, choose the lowest median residual and
    mark the result as unstable.  Final odd/wrong/pseudo outcomes never influence this
    selection.
    """
    for chart, validation in validations:
        median = validation["normalized_residual"]["median"]
        positive = validation["positive_margin_fraction"]
        if median is not None and median <= 0.55 and positive is not None and positive >= 0.75 and chart.fit_condition <= 1e10:
            return chart, {
                "status": "MINIMAL_CALIBRATION_VALID_CHART",
                "criterion": "median_residual<=0.55 && positive_margin_fraction>=0.75 && condition<=1e10",
            }
    chart, validation = min(
        validations,
        key=lambda item: float("inf") if item[1]["normalized_residual"]["median"] is None else item[1]["normalized_residual"]["median"],
    )
    return chart, {
        "status": "CALIBRATION_CHART_UNSTABLE",
        "criterion": "no chart satisfied frozen calibration-only acceptance; selected lowest median residual for diagnostics",
        "selected_validation": validation,
    }


def evaluate_rows(chart: ComplexChart, rows: list[dict[str, Any]], codebook: dict[str, np.ndarray], phase_levels: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    mode_names = list(codebook)
    for row in rows:
        theta_idx = int(row["theta_idx"])
        residual_by_mode: dict[str, float] = {}
        aligned_by_mode: dict[str, float] = {}
        similarity_by_mode: dict[str, float] = {}
        for mode in mode_names:
            prediction = chart.predict(complex_vector(codebook[mode], theta_idx, phase_levels))
            residual_by_mode[mode] = normalized_residual(row["z"], prediction)
            aligned_by_mode[mode] = phase_aligned_residual(row["z"], prediction)
            similarity_by_mode[mode] = cosine_similarity(row["z"], prediction)
        actual = str(row["actual_mode"])
        declared = str(row["declared_mode"])
        other = [value for mode, value in residual_by_mode.items() if mode != actual]
        records.append({
            "symbol_index": row.get("symbol_index"),
            "family": row["family"],
            "trial": row["trial"],
            "actual_mode": actual,
            "declared_mode": declared,
            "theta_idx": theta_idx,
            "residual_by_mode": residual_by_mode,
            "phase_aligned_residual_by_mode": aligned_by_mode,
            "similarity_by_mode": similarity_by_mode,
            "actual_residual": residual_by_mode[actual],
            "declared_residual": residual_by_mode[declared],
            "actual_mode_margin": min(other, default=residual_by_mode[actual]) - residual_by_mode[actual],
            "actual_minus_declared_fit_margin": residual_by_mode[declared] - residual_by_mode[actual],
        })
    return {
        "rows": len(records),
        "positive_actual_mode_margin_fraction": float(np.mean([record["actual_mode_margin"] > 0.0 for record in records])) if records else None,
        "observed_norm": quantiles(np.linalg.norm(row["z"]) for row in rows),
        "actual_mode_margin": quantiles(record["actual_mode_margin"] for record in records),
        "actual_residual": quantiles(record["actual_residual"] for record in records),
        "phase_aligned_actual_residual": quantiles(record["phase_aligned_residual_by_mode"][record["actual_mode"]] for record in records),
        "records": records,
    }


def phase_equivariance(chart: ComplexChart, rows: list[dict[str, Any]], codebook: dict[str, np.ndarray], phase_levels: int, null_seed: int) -> dict[str, Any]:
    estimates: list[float] = []
    declared: list[float] = []
    per_row: list[dict[str, Any]] = []
    for row in rows:
        mode = str(row["actual_mode"])
        zero_prediction = chart.predict(np.asarray(codebook[mode], dtype=np.complex128))
        estimate = phase_estimate(row["z"], zero_prediction)
        target = 2.0 * np.pi * int(row["theta_idx"]) / phase_levels
        error = float(wrap_phase(estimate - target))
        estimates.append(estimate)
        declared.append(target)
        per_row.append({
            "symbol_index": row.get("symbol_index"),
            "trial": row["trial"],
            "family": row["family"],
            "actual_mode": mode,
            "theta_idx": int(row["theta_idx"]),
            "phase_estimate": estimate,
            "declared_phase": target,
            "wrapped_error": error,
        })
    errors = np.asarray(estimates) - np.asarray(declared)
    rng = np.random.default_rng(null_seed)
    shuffled = np.asarray(declared)[rng.permutation(len(declared))] if declared else np.asarray([])
    null_errors = np.asarray(estimates) - shuffled

    pair_errors: list[float] = []
    by_mode: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_mode.setdefault(str(row["actual_mode"]), []).append(index)
    for indices in by_mode.values():
        ordered = sorted(indices, key=lambda idx: (int(rows[idx]["trial"]), int(rows[idx].get("symbol_index", idx))))
        for left, right in zip(ordered, ordered[1:]):
            observed_delta = float(wrap_phase(estimates[right] - estimates[left]))
            declared_delta = float(wrap_phase(declared[right] - declared[left]))
            pair_errors.append(float(wrap_phase(observed_delta - declared_delta)))

    return {
        "rows": len(rows),
        "mean_absolute_circular_error": circular_mean_abs(errors),
        "phase_error_resultant": circular_resultant(errors),
        "pairwise_mean_absolute_error": circular_mean_abs(pair_errors),
        "shuffled_null_mean_absolute_error": circular_mean_abs(null_errors),
        "null_seed": null_seed,
        "records": per_row,
    }


def pseudo_covariance(chart: ComplexChart, rows: list[dict[str, Any]], codebook: dict[str, np.ndarray], phase_levels: int, null_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(null_seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        actual = str(row["actual_mode"])
        theta_idx = int(row["theta_idx"])
        permutation = np.asarray(row["bin_permutation"], dtype=int)
        exact_code = np.asarray(codebook[actual])[permutation]
        canonical_code = np.asarray(codebook[actual])
        alternative = np.arange(len(permutation))
        rng.shuffle(alternative)
        if np.array_equal(alternative, permutation):
            alternative = np.roll(alternative, 1)
        exact_residual = normalized_residual(row["z"], chart.predict(complex_vector(exact_code, theta_idx, phase_levels)))
        canonical_residual = normalized_residual(row["z"], chart.predict(complex_vector(canonical_code, theta_idx, phase_levels)))
        alternative_residual = normalized_residual(row["z"], chart.predict(complex_vector(np.asarray(codebook[actual])[alternative], theta_idx, phase_levels)))
        records.append({
            "symbol_index": row.get("symbol_index"),
            "trial": row["trial"],
            "actual_mode": actual,
            "declared_mode": row["declared_mode"],
            "theta_idx": theta_idx,
            "exact_permutation_residual": exact_residual,
            "canonical_residual": canonical_residual,
            "alternative_permutation_residual": alternative_residual,
            "exact_over_canonical_margin": canonical_residual - exact_residual,
            "exact_over_alternative_margin": alternative_residual - exact_residual,
        })
    return {
        "rows": len(records),
        "exact_better_than_canonical_fraction": float(np.mean([r["exact_over_canonical_margin"] > 0.0 for r in records])) if records else None,
        "exact_better_than_alternative_fraction": float(np.mean([r["exact_over_alternative_margin"] > 0.0 for r in records])) if records else None,
        "exact_over_canonical_margin": quantiles(r["exact_over_canonical_margin"] for r in records),
        "exact_over_alternative_margin": quantiles(r["exact_over_alternative_margin"] for r in records),
        "null_seed": null_seed,
        "records": records,
    }


def gram_geometry(chart: ComplexChart, codebook: dict[str, np.ndarray]) -> dict[str, Any]:
    names = list(codebook)
    vectors = np.stack([chart.predict(np.asarray(codebook[name], dtype=np.complex128)) for name in names])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + EPS
    normalized = vectors / norms
    gram = normalized @ normalized.conj().T
    singular = np.linalg.svd(vectors, compute_uv=False)
    return {
        "mode_names": names,
        "gram_real": gram.real.tolist(),
        "gram_imag": gram.imag.tolist(),
        "singular_values": singular.tolist(),
        "effective_rank_1e_6": int(np.sum(singular > max(singular[0] * 1e-6, EPS))) if singular.size else 0,
    }


def compare_gram(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_gram = np.asarray(left["gram_real"]) + 1j * np.asarray(left["gram_imag"])
    right_gram = np.asarray(right["gram_real"]) + 1j * np.asarray(right["gram_imag"])
    return {
        "frobenius_difference": float(np.linalg.norm(left_gram - right_gram)),
        "normalized_frobenius_difference": float(np.linalg.norm(left_gram - right_gram) / (np.linalg.norm(left_gram) + EPS)),
    }
