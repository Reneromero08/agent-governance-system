#!/usr/bin/env python3
"""Execute Phase 6B.5C transfer-aware carrier geometry analysis.

This tool consumes an immutable carrier campaign directory. It verifies input
bindings, estimates route/session complex charts from preamble and even-real
calibration rows only, evaluates held-out relations, computes deterministic
nulls, and writes compact JSON outputs. It never modifies the campaign.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from complex_geometry import (
    CHART_LADDER,
    ComplexChart,
    chart_validation,
    compare_gram,
    complex_vector,
    evaluate_rows,
    fit_chart,
    gram_geometry,
    normalized_residual,
    phase_equivariance,
    pseudo_covariance,
    quantiles,
    select_chart,
    wrap_phase,
)

SCHEMA_VERSION = "1.0.0"
ANALYSIS_ID = "phase6b5c_transfer_aware_geometry_v1"
MODE_NAMES = ("basis", "rotation", "residual", "mini")
REQUIRED_RUN_FILES = (
    "run.json",
    "schedule.json",
    "windows.csv",
    "raw_samples.bin",
    "summary.csv",
    "analysis.json",
    "run_manifest.json",
)
OUTPUT_NAMES = (
    "gate_layer_reconciliation.json",
    "chart_calibration.json",
    "heldout_equivariance.json",
    "execution_relation.json",
    "phase_equivariance.json",
    "pseudo_permutation_covariance.json",
    "route_conjugacy.json",
    "ordered_path_analysis.json",
    "seed4_transfer_report.json",
)


@dataclass
class RunData:
    path: Path
    run: dict[str, Any]
    schedule: dict[str, Any]
    rows: list[dict[str, Any]]
    windows: list[dict[str, Any]]
    codebook: dict[str, np.ndarray]
    selected_chart: ComplexChart | None = None
    chart_selection: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return str(self.run["run_id"])

    @property
    def route(self) -> str:
        route = self.run.get("route", {})
        return str(route.get("label", "unknown"))

    @property
    def seed(self) -> int:
        return int(self.run.get("seed", -1))

    @property
    def condition(self) -> str:
        return str(self.run.get("condition", "unknown"))

    @property
    def phase_levels(self) -> int:
        return int(self.schedule.get("phase_levels", 8))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_summary(path: Path, schedule: dict[str, Any]) -> list[dict[str, Any]]:
    symbols = {int(item["symbol_index"]): item for item in schedule.get("symbols", [])}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        column: dict[str, int] = {}
        data_index = 0
        for raw in reader:
            if not raw:
                continue
            if raw[0].startswith("#"):
                continue
            if header is None:
                header = raw
                column = {name: index for index, name in enumerate(header)}
                continue
            nbin = sum(name.endswith("_I") and name.startswith("b") for name in header)
            z = np.asarray(
                [
                    float(raw[column[f"b{b:02d}_I"]])
                    + 1j * float(raw[column[f"b{b:02d}_Q"]])
                    for b in range(nbin)
                ],
                dtype=np.complex128,
            )
            symbol = symbols.get(data_index)
            if symbol is None:
                raise ValueError(f"schedule missing symbol_index {data_index} for {path}")
            row = {
                "symbol_index": data_index,
                "family": raw[column["family"]],
                "declared_mode": raw[column["declared_mode"]],
                "actual_mode": raw[column["actual_mode"]],
                "trial": int(raw[column["trial"]]),
                "hash_restored": int(raw[column["hash_restored"]]),
                "theta_idx": int(raw[column["theta_idx"]]),
                "z": z,
                "bin_permutation": list(symbol.get("bin_permutation", range(nbin))),
                "drive_signs": list(symbol.get("drive_signs", [])),
                "phase_fractions": list(symbol.get("phase_fractions", [])),
            }
            for key in ("family", "declared_mode", "actual_mode", "trial", "theta_idx"):
                schedule_value = symbol.get(key)
                summary_value = row[key]
                if (
                    key == "family"
                    and schedule_value == "preamble"
                    and summary_value == "real"
                    and row["trial"] < 0
                ):
                    continue
                if schedule_value != summary_value:
                    raise ValueError(
                        f"summary/schedule mismatch {path.name} symbol {data_index} field {key}"
                    )
            rows.append(row)
            data_index += 1
    if len(rows) != len(symbols):
        raise ValueError(
            f"summary/schedule row count mismatch for {path}: {len(rows)} != {len(symbols)}"
        )
    return rows


def _parse_optional_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def read_windows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"windows CSV missing header: {path}")
        for raw in reader:
            rows.append(
                {
                    "window_index": int(raw["window_index"]),
                    "symbol_index": int(raw["symbol_index"]),
                    "bin_index": int(raw["bin_index"]),
                    "family": raw["family"],
                    "trial": int(raw["trial"]),
                    "theta_idx": int(raw["theta_idx"]),
                    "sample_count": int(raw["sample_count"]),
                    "tone_hz": float(raw["tone_hz"]),
                    "drive_sign": int(raw["drive_sign"]),
                    "phase_fraction": float(raw["phase_fraction"]),
                    "slot_start_tsc": int(raw["slot_start_tsc"]),
                    "capture_deadline_tsc": int(raw["capture_deadline_tsc"]),
                    "first_sample_tsc": int(raw["first_sample_tsc"]),
                    "last_sample_tsc": int(raw["last_sample_tsc"]),
                    "temp_before_c": _parse_optional_float(raw.get("temp_before_c")),
                    "temp_after_c": _parse_optional_float(raw.get("temp_after_c")),
                    "cur_khz_before": _parse_optional_float(raw.get("cur_khz_before")),
                    "cur_khz_after": _parse_optional_float(raw.get("cur_khz_after")),
                    "computed_I": float(raw["computed_I"]),
                    "computed_Q": float(raw["computed_Q"]),
                    "computed_magnitude": float(raw["computed_magnitude"]),
                    "computed_floor": float(raw["computed_floor"]),
                }
            )
    return rows


def verify_run_manifest(run_dir: Path, verify_raw_hash: bool) -> dict[str, Any]:
    manifest = json_load(run_dir / "run_manifest.json")
    errors: list[str] = []
    verified: dict[str, Any] = {}
    for name in REQUIRED_RUN_FILES:
        path = run_dir / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        if name == "run_manifest.json":
            continue
        entry = manifest.get("files", {}).get(name)
        if not isinstance(entry, dict):
            errors.append(f"manifest missing {name}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(entry.get("size", -1)):
            errors.append(f"size mismatch {name}: {actual_size} != {entry.get('size')}")
        should_hash = verify_raw_hash or name != "raw_samples.bin"
        actual_sha = sha256_file(path) if should_hash else str(entry.get("sha256"))
        if should_hash and actual_sha != entry.get("sha256"):
            errors.append(f"sha256 mismatch {name}")
        verified[name] = {
            "size": actual_size,
            "sha256": actual_sha,
            "hash_recomputed": should_hash,
        }
    if errors:
        raise ValueError(f"invalid run manifest {run_dir}: {'; '.join(errors)}")
    return {"run_id": manifest.get("run_id"), "files": verified}


def load_run(run_dir: Path, verify_raw_hash: bool) -> tuple[RunData, dict[str, Any]]:
    binding = verify_run_manifest(run_dir, verify_raw_hash)
    run = json_load(run_dir / "run.json")
    schedule = json_load(run_dir / "schedule.json")
    if (
        run.get("run_id") != schedule.get("run_id")
        or run.get("run_id") != binding.get("run_id")
    ):
        raise ValueError(f"run ID binding mismatch in {run_dir}")
    codebook = {
        name: np.asarray(schedule["codebook"][name], dtype=np.complex128)
        for name in MODE_NAMES
    }
    rows = read_summary(run_dir / "summary.csv", schedule)
    windows = read_windows(run_dir / "windows.csv")
    expected_windows = len(rows) * len(next(iter(codebook.values())))
    if len(windows) != expected_windows:
        raise ValueError(
            f"window count mismatch in {run_dir}: {len(windows)} != {expected_windows}"
        )
    return RunData(run_dir, run, schedule, rows, windows, codebook), binding


def discover_runs(
    campaign_root: Path, verify_raw_hash: bool
) -> tuple[list[RunData], list[dict[str, Any]]]:
    campaign = json_load(campaign_root / "campaign.json")
    expected_ids: list[str] = []
    for item in campaign.get("runs", []):
        if isinstance(item, str):
            expected_ids.append(item)
        elif isinstance(item, dict):
            expected_ids.append(str(item.get("run_id")))
    runs_root = campaign_root / "runs"
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    discovered_ids = {path.name for path in run_dirs}
    if expected_ids:
        missing = sorted(set(expected_ids) - discovered_ids)
        if missing:
            raise ValueError(f"campaign missing expected run directories: {missing}")
        extra = sorted(discovered_ids - set(expected_ids))
        if extra:
            raise ValueError(f"unbound run directories not in campaign.json: {extra}")
    runs: list[RunData] = []
    bindings: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run, binding = load_run(run_dir, verify_raw_hash)
        runs.append(run)
        bindings.append(binding)
    return runs, bindings


def calibration_partitions(
    run: RunData,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    calibration = [
        row
        for row in run.rows
        if row["family"] in {"preamble", "real"}
        and (row["trial"] < 0 or row["trial"] % 2 == 0)
    ]
    fit_rows = [
        row
        for row in calibration
        if row["trial"] < 0 or row["trial"] % 4 == 0
    ]
    validation_rows = [
        row
        for row in calibration
        if row["trial"] >= 0 and row["trial"] % 4 == 2
    ]
    if not validation_rows:
        validation_rows = [row for row in calibration if row["trial"] >= 0]
    evaluation = [
        row
        for row in run.rows
        if row["trial"] >= 0 and row["trial"] % 2 == 1
    ]
    return fit_rows, validation_rows, evaluation


def x_matrix(
    rows: list[dict[str, Any]],
    codebook: dict[str, np.ndarray],
    phase_levels: int,
) -> np.ndarray:
    return np.stack(
        [
            complex_vector(
                codebook[str(row["actual_mode"])],
                int(row["theta_idx"]),
                phase_levels,
            )
            for row in rows
        ]
    )


def z_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([row["z"] for row in rows])


def calibrate_run(run: RunData) -> dict[str, Any]:
    if run.condition != "matrix":
        return {
            "run_id": run.run_id,
            "condition": run.condition,
            "status": "CONTROL_NO_CHART_FIT",
        }
    fit_rows, validation_rows, evaluation = calibration_partitions(run)
    x_fit = x_matrix(fit_rows, run.codebook, run.phase_levels)
    z_fit = z_matrix(fit_rows)
    validations: list[tuple[ComplexChart, dict[str, Any]]] = []
    chart_records: list[dict[str, Any]] = []
    for spec in CHART_LADDER:
        chart = fit_chart(spec, x_fit, z_fit)
        validation = chart_validation(
            chart, validation_rows, run.codebook, run.phase_levels
        )
        validations.append((chart, validation))
        chart_records.append(
            {"chart": chart.to_json(), "calibration_validation": validation}
        )
    selected, selection = select_chart(validations)
    run.selected_chart = selected
    run.chart_selection = selection
    return {
        "run_id": run.run_id,
        "route": run.route,
        "seed": run.seed,
        "condition": run.condition,
        "partition": {
            "fit_rows": len(fit_rows),
            "validation_rows": len(validation_rows),
            "final_evaluation_rows": len(evaluation),
            "fit_rule": "preamble plus even-real trial%4==0",
            "validation_rule": "even-real trial%4==2",
            "evaluation_rule": "odd trials only",
        },
        "chart_ladder": chart_records,
        "selected_chart": selected.to_json(),
        "selection": selection,
    }


def deterministic_mode_null(
    rows: list[dict[str, Any]],
    chart: ComplexChart,
    codebook: dict[str, np.ndarray],
    phase_levels: int,
) -> dict[str, Any]:
    margins: list[float] = []
    names = list(codebook)
    for row in rows:
        actual = str(row["actual_mode"])
        null_mode = names[(names.index(actual) + 1) % len(names)]
        theta = int(row["theta_idx"])
        actual_residual = normalized_residual(
            row["z"],
            chart.predict(complex_vector(codebook[actual], theta, phase_levels)),
        )
        null_residual = normalized_residual(
            row["z"],
            chart.predict(complex_vector(codebook[null_mode], theta, phase_levels)),
        )
        margins.append(null_residual - actual_residual)
    return {
        "rows": len(rows),
        "actual_better_fraction": float(np.mean(np.asarray(margins) > 0.0))
        if margins
        else None,
        "actual_over_rotated_mode_margin": quantiles(margins),
    }


def matched_weight_codebook(
    codebook: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for name, vector in codebook.items():
        shuffled = np.asarray(vector).copy()
        rng.shuffle(shuffled)
        if np.array_equal(shuffled, vector):
            shuffled = np.roll(shuffled, 1)
        result[name] = shuffled
    return result


def evaluate_run(run: RunData, null_seed: int) -> dict[str, Any]:
    if run.condition != "matrix" or run.selected_chart is None:
        return {
            "run_id": run.run_id,
            "condition": run.condition,
            "status": "CONTROL_EVALUATED_SEPARATELY",
        }
    _, _, evaluation = calibration_partitions(run)
    real_rows = [row for row in evaluation if row["family"] == "real"]
    wrong_rows = [row for row in evaluation if row["family"] == "wrong"]
    pseudo_rows = [row for row in evaluation if row["family"] == "pseudo"]
    real_eval = evaluate_rows(
        run.selected_chart, real_rows, run.codebook, run.phase_levels
    )
    wrong_eval = evaluate_rows(
        run.selected_chart, wrong_rows, run.codebook, run.phase_levels
    )
    phase_eval = phase_equivariance(
        run.selected_chart,
        real_rows + wrong_rows,
        run.codebook,
        run.phase_levels,
        null_seed,
    )
    pseudo_eval = pseudo_covariance(
        run.selected_chart,
        pseudo_rows,
        run.codebook,
        run.phase_levels,
        null_seed + 1,
    )
    mode_null = deterministic_mode_null(
        real_rows + wrong_rows,
        run.selected_chart,
        run.codebook,
        run.phase_levels,
    )
    matched_null_book = matched_weight_codebook(run.codebook, null_seed + 2)
    matched_null = evaluate_rows(
        run.selected_chart,
        real_rows + wrong_rows,
        matched_null_book,
        run.phase_levels,
    )
    wrong_margins = [
        record["actual_minus_declared_fit_margin"]
        for record in wrong_eval.get("records", [])
    ]
    return {
        "run_id": run.run_id,
        "route": run.route,
        "seed": run.seed,
        "condition": run.condition,
        "selected_chart_id": run.selected_chart.spec.chart_id,
        "selection": run.chart_selection,
        "real": real_eval,
        "wrong": {
            **wrong_eval,
            "actual_better_than_declared_fraction": float(
                np.mean(np.asarray(wrong_margins) > 0.0)
            )
            if wrong_margins
            else None,
            "actual_over_declared_margin": quantiles(wrong_margins),
        },
        "phase": phase_eval,
        "pseudo": pseudo_eval,
        "nulls": {
            "rotated_mode_assignment": mode_null,
            "matched_weight_sign_codebook": {
                "null_seed": null_seed + 2,
                "positive_actual_mode_margin_fraction": matched_null.get(
                    "positive_actual_mode_margin_fraction"
                ),
                "actual_mode_margin": matched_null.get("actual_mode_margin"),
            },
        },
    }


def control_metrics(
    run: RunData, reference_charts: list[ComplexChart]
) -> dict[str, Any]:
    rows = [
        row
        for row in run.rows
        if row["trial"] >= 0 and row["trial"] % 2 == 1
    ]
    norms = [float(np.linalg.norm(row["z"])) for row in rows]
    floor_ratio: list[float] = []
    for window in run.windows:
        magnitude = abs(complex(window["computed_I"], window["computed_Q"]))
        floor_ratio.append(float(magnitude / (window["computed_floor"] + 1e-12)))
    chart_results: list[dict[str, Any]] = []
    for chart in reference_charts:
        result = evaluate_rows(chart, rows, run.codebook, run.phase_levels)
        chart_results.append(
            {
                "chart_id": chart.spec.chart_id,
                "positive_actual_mode_margin_fraction": result.get(
                    "positive_actual_mode_margin_fraction"
                ),
                "actual_residual": result.get("actual_residual"),
            }
        )
    return {
        "run_id": run.run_id,
        "condition": run.condition,
        "rows": len(rows),
        "observed_vector_norm": quantiles(norms),
        "window_magnitude_to_floor": quantiles(floor_ratio),
        "canonical_relation_under_reference_charts": chart_results,
        "chart_fit_on_control": False,
    }


def spearman_like(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3 or np.std(xa) < 1e-15 or np.std(ya) < 1e-15:
        return None
    xr = np.argsort(np.argsort(xa)).astype(float)
    yr = np.argsort(np.argsort(ya)).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def ordered_path(run: RunData) -> dict[str, Any]:
    tsc_hz = float(run.run.get("timing", {}).get("tsc_hz", 1.0))
    t0 = int(run.run.get("timing", {}).get("t0_tsc", 0))
    sample_counts = [window["sample_count"] for window in run.windows]
    lateness = [
        (window["first_sample_tsc"] - window["slot_start_tsc"]) / tsc_hz
        for window in run.windows
    ]
    window_duration = [
        (window["last_sample_tsc"] - window["first_sample_tsc"]) / tsc_hz
        for window in run.windows
    ]
    elapsed = [
        (window["slot_start_tsc"] - t0) / tsc_hz for window in run.windows
    ]
    magnitudes = [window["computed_magnitude"] for window in run.windows]
    floors = [window["computed_floor"] for window in run.windows]
    bins = [window["bin_index"] for window in run.windows]
    temperatures = [
        value
        for window in run.windows
        for value in (window["temp_before_c"], window["temp_after_c"])
        if value is not None
    ]
    frequencies = [
        value
        for window in run.windows
        for value in (window["cur_khz_before"], window["cur_khz_after"])
        if value is not None and value >= 0
    ]
    symbol_records: list[dict[str, Any]] = []
    by_symbol: dict[int, list[dict[str, Any]]] = {}
    for window in run.windows:
        by_symbol.setdefault(window["symbol_index"], []).append(window)
    for symbol_index, windows in sorted(by_symbol.items()):
        ordered = sorted(windows, key=lambda window: window["bin_index"])
        complex_values = np.asarray(
            [complex(window["computed_I"], window["computed_Q"]) for window in ordered]
        )
        prefix = np.cumsum(complex_values)
        suffix = np.cumsum(complex_values[::-1])[::-1]
        total = np.linalg.norm(complex_values) + 1e-12
        symbol_records.append(
            {
                "symbol_index": symbol_index,
                "family": ordered[0]["family"],
                "trial": ordered[0]["trial"],
                "prefix_half_fraction": float(
                    np.linalg.norm(prefix[len(prefix) // 2 - 1]) / total
                ),
                "suffix_half_fraction": float(
                    np.linalg.norm(suffix[len(suffix) // 2]) / total
                ),
                "first_last_phase_delta": float(
                    wrap_phase(
                        np.angle(complex_values[-1]) - np.angle(complex_values[0])
                    )
                ),
                "magnitude_slope_rank_correlation": spearman_like(
                    range(len(ordered)), [abs(value) for value in complex_values]
                ),
            }
        )
    return {
        "run_id": run.run_id,
        "route": run.route,
        "seed": run.seed,
        "condition": run.condition,
        "window_count": len(run.windows),
        "sample_count": quantiles(sample_counts),
        "capture_lateness_s": quantiles(lateness),
        "window_duration_s": quantiles(window_duration),
        "elapsed_s": quantiles(elapsed),
        "magnitude": quantiles(magnitudes),
        "floor": quantiles(floors),
        "temperature_c": quantiles(temperatures),
        "frequency_khz": quantiles(frequencies),
        "associations": {
            "elapsed_vs_magnitude_rank": spearman_like(elapsed, magnitudes),
            "elapsed_vs_floor_rank": spearman_like(elapsed, floors),
            "bin_index_vs_magnitude_rank": spearman_like(bins, magnitudes),
            "lateness_vs_magnitude_rank": spearman_like(lateness, magnitudes),
            "sample_count_vs_magnitude_rank": spearman_like(
                sample_counts, magnitudes
            ),
        },
        "fixed_order_confounding": "tone bin and within-symbol elapsed position are not independently varied in this campaign",
        "symbol_path_records": symbol_records,
    }


def cross_chart_transfer(matrix_runs: list[RunData]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for target in matrix_runs:
        _, _, target_eval = calibration_partitions(target)
        target_real_wrong = [
            row for row in target_eval if row["family"] in {"real", "wrong"}
        ]
        for source in matrix_runs:
            if source.run_id == target.run_id or source.selected_chart is None:
                continue
            result = evaluate_rows(
                source.selected_chart,
                target_real_wrong,
                target.codebook,
                target.phase_levels,
            )
            records.append(
                {
                    "source_run": source.run_id,
                    "source_route": source.route,
                    "source_seed": source.seed,
                    "target_run": target.run_id,
                    "target_route": target.route,
                    "target_seed": target.seed,
                    "same_route": source.route == target.route,
                    "same_seed": source.seed == target.seed,
                    "positive_margin_fraction": result.get(
                        "positive_actual_mode_margin_fraction"
                    ),
                    "actual_residual": result.get("actual_residual"),
                    "actual_mode_margin": result.get("actual_mode_margin"),
                }
            )
    return {"records": records}


def route_conjugacy(matrix_runs: list[RunData]) -> dict[str, Any]:
    per_run = {
        run.run_id: {
            "route": run.route,
            "seed": run.seed,
            "chart_id": run.selected_chart.spec.chart_id
            if run.selected_chart
            else None,
            "geometry": gram_geometry(run.selected_chart, run.codebook)
            if run.selected_chart
            else None,
        }
        for run in matrix_runs
    }
    pairs: list[dict[str, Any]] = []
    by_seed_route = {(run.seed, run.route): run for run in matrix_runs}
    routes = sorted({run.route for run in matrix_runs})
    if len(routes) >= 2:
        for seed in sorted({run.seed for run in matrix_runs}):
            left = by_seed_route.get((seed, routes[0]))
            right = by_seed_route.get((seed, routes[1]))
            if not left or not right or not left.selected_chart or not right.selected_chart:
                continue
            left_geometry = per_run[left.run_id]["geometry"]
            right_geometry = per_run[right.run_id]["geometry"]
            comparison = compare_gram(left_geometry, right_geometry)
            route_map = (
                np.linalg.pinv(left.selected_chart.matrix)
                @ right.selected_chart.matrix
            )
            orbit_residuals: list[float] = []
            for mode in MODE_NAMES:
                left_vector = left.selected_chart.predict(left.codebook[mode])
                right_vector = right.selected_chart.predict(right.codebook[mode])
                orbit_residuals.append(
                    normalized_residual(right_vector, left_vector @ route_map)
                )
            pairs.append(
                {
                    "seed": seed,
                    "left_run": left.run_id,
                    "right_run": right.run_id,
                    **comparison,
                    "diagnostic_route_map_orbit_residual": quantiles(
                        orbit_residuals
                    ),
                }
            )
    return {
        "per_run_geometry": per_run,
        "matched_seed_cross_route": pairs,
        "cross_chart_transfer": cross_chart_transfer(matrix_runs),
        "interpretive_limit": "diagnostic chart correspondence is not complete physical operator identification",
    }


def seed4_report(
    run_results: dict[str, Any],
    calibrations: dict[str, Any],
    path_results: dict[str, Any],
) -> dict[str, Any]:
    run_id = "v4s5_matrix_seed4"
    result = run_results.get(run_id)
    calibration = calibrations.get(run_id)
    path = path_results.get(run_id)
    if result is None:
        return {
            "run_id": run_id,
            "classification": "UNRESOLVED",
            "reason": "run not present",
        }
    phase_error = result.get("phase", {}).get("mean_absolute_circular_error")
    real_positive = result.get("real", {}).get(
        "positive_actual_mode_margin_fraction"
    )
    wrong_actual = result.get("wrong", {}).get(
        "actual_better_than_declared_fraction"
    )
    selection = (
        calibration.get("selection", {}).get("status") if calibration else None
    )
    residual = result.get("real", {}).get("actual_residual", {}).get("median")
    if (
        phase_error is not None
        and phase_error < 0.45
        and real_positive is not None
        and real_positive >= 0.75
        and selection == "CALIBRATION_CHART_UNSTABLE"
    ):
        classification = "CHART_FAILURE"
    elif (
        phase_error is not None
        and phase_error < 0.45
        and real_positive is not None
        and real_positive < 0.75
    ):
        classification = "TRANSFER_REGIME_SHIFT"
    elif (
        phase_error is not None
        and phase_error >= 0.9
        and real_positive is not None
        and real_positive < 0.6
    ):
        classification = "CARRIER_FAILURE"
    elif (
        wrong_actual is not None
        and wrong_actual < 0.6
        and residual is not None
        and residual > 0.8
    ):
        classification = "MIXED_FAILURE"
    else:
        classification = "UNRESOLVED"
    return {
        "run_id": run_id,
        "classification": classification,
        "phase_mean_absolute_error": phase_error,
        "real_positive_margin_fraction": real_positive,
        "wrong_actual_better_fraction": wrong_actual,
        "real_median_residual": residual,
        "chart_selection_status": selection,
        "ordered_path": path,
        "claim_limit": "classification is a derived diagnostic over the retained campaign",
    }


def aggregate_decision(
    run_results: dict[str, Any],
    controls: dict[str, Any],
    route_result: dict[str, Any],
) -> dict[str, Any]:
    matrix = [
        value for value in run_results.values() if value.get("condition") == "matrix"
    ]
    phase_good = [
        result.get("phase", {}).get(
            "mean_absolute_circular_error", float("inf")
        )
        < 0.5
        and result.get("phase", {}).get(
            "shuffled_null_mean_absolute_error", 0.0
        )
        > result.get("phase", {}).get(
            "mean_absolute_circular_error", float("inf")
        )
        for result in matrix
    ]
    transfer_good = [
        (
            result.get("real", {}).get(
                "positive_actual_mode_margin_fraction"
            )
            or 0.0
        )
        >= 0.75
        and (
            result.get("wrong", {}).get(
                "actual_better_than_declared_fraction"
            )
            or 0.0
        )
        >= 0.75
        for result in matrix
    ]
    pseudo_good = [
        (
            result.get("pseudo", {}).get(
                "exact_better_than_alternative_fraction"
            )
            or 0.0
        )
        >= 0.65
        for result in matrix
    ]
    chart_stable = [
        result.get("selection", {}).get("status")
        == "MINIMAL_CALIBRATION_VALID_CHART"
        for result in matrix
    ]
    matrix_norms = [
        result.get("real", {}).get("observed_norm", {}).get("median")
        for result in matrix
    ]
    matrix_norms = [float(value) for value in matrix_norms if value is not None]
    matrix_norm_reference = float(np.median(matrix_norms)) if matrix_norms else None
    silent_energy_null = True
    scramble_relation_null = True
    for control in controls.values():
        condition = control.get("condition")
        if condition == "silent":
            median_norm = control.get("observed_vector_norm", {}).get("median")
            if matrix_norm_reference is not None and median_norm is not None:
                silent_energy_null = (
                    float(median_norm) <= 0.5 * matrix_norm_reference
                )
        if condition == "scramble":
            fractions = [
                item.get("positive_actual_mode_margin_fraction")
                for item in control.get(
                    "canonical_relation_under_reference_charts", []
                )
                if item.get("positive_actual_mode_margin_fraction") is not None
            ]
            if fractions:
                scramble_relation_null = float(np.median(fractions)) <= 0.65
    null_low = silent_energy_null and scramble_relation_null
    n = max(len(matrix), 1)
    phase_fraction = sum(phase_good) / n
    transfer_fraction = sum(transfer_good) / n
    pseudo_fraction = sum(pseudo_good) / n
    chart_fraction = sum(chart_stable) / n
    if (
        transfer_fraction >= 0.75
        and phase_fraction >= 0.75
        and pseudo_fraction >= 0.65
        and null_low
    ):
        outcome = "TRANSFER_EQUIVARIANCE_SUPPORTED"
    elif phase_fraction >= 0.75 and transfer_fraction < 0.5:
        outcome = "PHASE_ONLY_TRANSPORT_SUPPORTED"
    elif transfer_fraction >= 0.75 and chart_fraction < 0.5:
        outcome = "ROUTE_SESSION_CHART_UNSTABLE"
    elif phase_fraction < 0.5 and transfer_fraction < 0.5:
        outcome = "NO_RELATIONAL_TRANSPORT_BEYOND_CONTROLS"
    else:
        outcome = "INCONCLUSIVE"
    return {
        "primary_outcome": outcome,
        "matrix_runs": len(matrix),
        "phase_supported_fraction": phase_fraction,
        "transfer_supported_fraction": transfer_fraction,
        "pseudo_covariance_supported_fraction": pseudo_fraction,
        "calibration_stable_fraction": chart_fraction,
        "controls_remain_null": null_low,
        "silent_energy_null": silent_energy_null,
        "scramble_canonical_relation_null": scramble_relation_null,
        "matrix_observed_norm_reference": matrix_norm_reference,
        "official_strict_closure_unchanged": "PARTIAL",
        "claim_ceiling": "DERIVED_TRANSFER_AWARE_ANALYSIS_ONLY",
        "route_geometry_available": bool(
            route_result.get("matched_seed_cross_route")
        ),
    }


CAMPAIGN_MANIFEST_SCHEMA_ID = "CAT_CAS_PDN_CARRIER_CAMPAIGN_MANIFEST_V1"


def _validate_relative_path(relative: str) -> None:
    if not relative or "\\" in relative or ":" in relative.split("/")[0]:
        raise ValueError(f"unsafe or non-portable manifest path: {relative!r}")
    normalized = Path(relative)
    if normalized.is_absolute():
        raise ValueError(f"absolute path in manifest: {relative!r}")
    parts = normalized.parts
    if ".." in parts:
        raise ValueError(f"path traversal in manifest: {relative!r}")
    if any(not part or part == "." for part in parts):
        raise ValueError(f"invalid path component in manifest: {relative!r}")


def verify_campaign_manifest(
    campaign_root: Path,
) -> dict[str, Any]:
    manifest_path = campaign_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"campaign manifest missing: {manifest_path}")
    manifest = json_load(manifest_path)
    if manifest.get("schema_id") != CAMPAIGN_MANIFEST_SCHEMA_ID:
        raise ValueError(
            f"unexpected campaign manifest schema: {manifest.get('schema_id')!r}"
        )
    campaign = json_load(campaign_root / "campaign.json")
    manifest_campaign_id = str(manifest.get("campaign_id", ""))
    campaign_campaign_id = str(campaign.get("campaign_id", ""))
    if manifest_campaign_id != campaign_campaign_id:
        raise ValueError(
            "campaign ID mismatch: manifest "
            f"{manifest_campaign_id!r} != campaign.json {campaign_campaign_id!r}"
        )
    manifest_commit = str(manifest.get("source_commit", ""))
    campaign_commit = str(campaign.get("source_commit", ""))
    if manifest_commit and campaign_commit and manifest_commit != campaign_commit:
        raise ValueError(
            "source commit mismatch: manifest "
            f"{manifest_commit!r} != campaign.json {campaign_commit!r}"
        )
    errors: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        _validate_relative_path(str(relative))
        candidate = campaign_root / relative
        if not candidate.is_file():
            errors.append(f"missing manifest file: {relative}")
            continue
        actual_size = candidate.stat().st_size
        expected_size = int(expected.get("size", -1))
        if actual_size != expected_size:
            errors.append(
                f"size mismatch {relative}: {actual_size} != {expected_size}"
            )
        actual_sha = sha256_file(candidate)
        expected_sha = str(expected.get("sha256", ""))
        if actual_sha != expected_sha:
            errors.append(f"sha256 mismatch {relative}")
    verified_run_ids: set[str] = set()
    for run_id, entry in manifest.get("run_manifests", {}).items():
        run_id_str = str(run_id)
        if run_id_str in verified_run_ids:
            errors.append(f"duplicate run_id in manifest: {run_id_str}")
            continue
        verified_run_ids.add(run_id_str)
        relative = str(entry.get("path", ""))
        _validate_relative_path(relative)
        candidate = campaign_root / relative
        if not candidate.is_file():
            errors.append(f"missing run manifest: {relative}")
            continue
        actual_size = candidate.stat().st_size
        expected_size = int(entry.get("size", -1))
        if actual_size != expected_size:
            errors.append(
                f"run manifest size mismatch {run_id_str}: {actual_size} != {expected_size}"
            )
        actual_sha = sha256_file(candidate)
        expected_sha = str(entry.get("sha256", ""))
        if actual_sha != expected_sha:
            errors.append(f"run manifest sha256 mismatch {run_id_str}")
        run_dir = candidate.parent
        if not run_dir.is_dir():
            errors.append(f"run manifest parent not a directory: {run_id_str}")
    expected_run_ids: set[str] = set()
    for item in campaign.get("runs", []):
        if isinstance(item, str):
            expected_run_ids.add(item)
        elif isinstance(item, dict):
            expected_run_ids.add(str(item.get("run_id", "")))
    missing_from_manifest = expected_run_ids - verified_run_ids
    if missing_from_manifest:
        errors.append(
            f"campaign runs missing from manifest: {sorted(missing_from_manifest)}"
        )
    extra_in_manifest = verified_run_ids - expected_run_ids
    if extra_in_manifest:
        errors.append(
            f"manifest runs not in campaign.json: {sorted(extra_in_manifest)}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return manifest


def gate_layer_reconciliation(campaign_root: Path) -> dict[str, Any]:
    del campaign_root
    source = Path("replication_discrepancy/results/official_gate_decomposition.json")
    path = Path(__file__).resolve().parent.parent / source
    available = path.is_file()
    return {
        "schema_id": "CAT_CAS_PHASE6B5C_GATE_LAYER_BINDING_V1",
        "source": source.as_posix(),
        "available_in_campaign_bundle": False,
        "available_in_analysis_source": available,
        "source_size": path.stat().st_size if available else None,
        "source_sha256": sha256_file(path) if available else None,
        "official_strict_closure_unchanged": "PARTIAL",
        "role_separation": [
            "protocol_integrity",
            "mode_transport",
            "phase_transport",
            "shared_schedule_specificity",
            "canonical_basis_fidelity",
            "metadata_leakage_sanity",
        ],
    }


def build_outputs(
    campaign_root: Path,
    output_dir: Path,
    verify_raw_hash: bool,
    null_seed: int,
) -> dict[str, Any]:
    verify_campaign_manifest(campaign_root)
    campaign = json_load(campaign_root / "campaign.json")
    runs, input_bindings = discover_runs(campaign_root, verify_raw_hash)
    matrix_runs = [run for run in runs if run.condition == "matrix"]
    controls_runs = [
        run for run in runs if run.condition in {"silent", "scramble"}
    ]

    calibrations: dict[str, Any] = {}
    for run in runs:
        calibrations[run.run_id] = calibrate_run(run)

    run_results: dict[str, Any] = {}
    for run in matrix_runs:
        run_results[run.run_id] = evaluate_run(
            run, null_seed + max(run.seed, 0) * 101
        )

    reference_charts = [
        run.selected_chart
        for run in matrix_runs
        if run.selected_chart and run.route == campaign.get("primary_route")
    ]
    controls = {
        run.run_id: control_metrics(run, reference_charts)
        for run in controls_runs
    }
    path_results = {run.run_id: ordered_path(run) for run in runs}
    route_result = route_conjugacy(matrix_runs)
    seed4 = seed4_report(run_results, calibrations, path_results)
    decision = aggregate_decision(run_results, controls, route_result)

    common = {
        "analysis_id": ANALYSIS_ID,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign.get("campaign_id"),
        "campaign_source_commit": campaign.get("source_commit"),
        "campaign_manifest_sha256": sha256_file(
            campaign_root / "campaign_manifest.json"
        ),
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "complex_geometry_source_sha256": sha256_file(
            Path(__file__).with_name("complex_geometry.py")
        ),
        "generated_utc": utc_now(),
        "null_seed": null_seed,
        "calibration_partition": "preamble + even-real; internal fit trial%4==0, validation trial%4==2",
        "final_evaluation_partition": "odd trials only",
        "official_closure_status_unchanged": "PARTIAL",
        "claim_ceiling": "DERIVED_TRANSFER_AWARE_ANALYSIS_ONLY",
    }

    outputs: dict[str, Any] = {
        "gate_layer_reconciliation.json": {
            **common,
            **gate_layer_reconciliation(campaign_root),
        },
        "chart_calibration.json": {**common, "runs": calibrations},
        "heldout_equivariance.json": {
            **common,
            "runs": {
                run_id: {
                    "real": result["real"],
                    "nulls": result["nulls"],
                    "selected_chart_id": result["selected_chart_id"],
                }
                for run_id, result in run_results.items()
            },
            "controls": controls,
        },
        "execution_relation.json": {
            **common,
            "runs": {
                run_id: {
                    "wrong": result["wrong"],
                    "selected_chart_id": result["selected_chart_id"],
                }
                for run_id, result in run_results.items()
            },
        },
        "phase_equivariance.json": {
            **common,
            "runs": {
                run_id: result["phase"] for run_id, result in run_results.items()
            },
            "controls": controls,
        },
        "pseudo_permutation_covariance.json": {
            **common,
            "runs": {
                run_id: result["pseudo"] for run_id, result in run_results.items()
            },
        },
        "route_conjugacy.json": {**common, **route_result},
        "ordered_path_analysis.json": {**common, "runs": path_results},
        "seed4_transfer_report.json": {**common, **seed4},
    }

    for name, value in outputs.items():
        json_write(output_dir / name, value)

    manifest_files: dict[str, Any] = {}
    for name in OUTPUT_NAMES:
        path = output_dir / name
        manifest_files[name] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    input_files = {
        "campaign.json": {
            "size": (campaign_root / "campaign.json").stat().st_size,
            "sha256": sha256_file(campaign_root / "campaign.json"),
        },
        "campaign_manifest.json": {
            "size": (campaign_root / "campaign_manifest.json").stat().st_size,
            "sha256": sha256_file(campaign_root / "campaign_manifest.json"),
        },
        "run_bindings": input_bindings,
    }
    manifest = {
        **common,
        "schema_id": "CAT_CAS_PHASE6B5C_ANALYSIS_MANIFEST_V1",
        "inputs": input_files,
        "outputs": manifest_files,
        "decision": decision,
        "manifest_excludes_itself": True,
    }
    json_write(output_dir / "analysis_manifest.json", manifest)
    return manifest


def verify_outputs(output_dir: Path) -> dict[str, Any]:
    manifest = json_load(output_dir / "analysis_manifest.json")
    errors: list[str] = []
    for name, expected in manifest.get("outputs", {}).items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing output: {name}")
            continue
        if path.stat().st_size != int(expected.get("size", -1)):
            errors.append(f"size mismatch: {name}")
        if sha256_file(path) != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": sha256_file(output_dir / "analysis_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("campaign_root", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--null-seed", type=int, default=65005)
    run.add_argument("--skip-raw-hash", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        manifest = build_outputs(
            args.campaign_root.resolve(),
            args.output.resolve(),
            not args.skip_raw_hash,
            args.null_seed,
        )
        print(json.dumps(manifest["decision"], indent=2, sort_keys=True))
        return 0
    result = verify_outputs(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
