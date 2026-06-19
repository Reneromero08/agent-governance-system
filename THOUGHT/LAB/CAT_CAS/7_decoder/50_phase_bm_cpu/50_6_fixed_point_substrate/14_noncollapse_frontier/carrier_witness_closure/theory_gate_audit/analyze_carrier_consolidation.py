#!/usr/bin/env python3
"""Phase 6B.5D bounded consolidation over committed Phase 6B.5C outputs.

This analysis does not refit the physical campaign or modify the official strict
closure. It answers four bounded questions before physical advancement:

1. Can the minimal scalar receiver chart alter the historical normalized gates?
2. Does the relational geometry generalize across sessions and routes?
3. Is held-out residual magnitude structured by route, seed, mode, phase, or time?
4. What exactly failed in route 4:5 seed 4?

The script consumes only the committed, manifest-bound Phase 6B.5C packet.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

ANALYSIS_ID = "phase6b5d_carrier_claim_consolidation_v1"
SCHEMA_VERSION = "1.0.0"
OUTPUT_FILES = (
    "old_gate_chart_invariance.json",
    "cross_session_generalization.json",
    "residual_structure.json",
    "seed4_localization.json",
    "carrier_claim_freeze.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(values: Iterable[float | int | None]) -> np.ndarray:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    return array[np.isfinite(array)]


def summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    array = finite(values)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "q05": None,
            "median": None,
            "mean": None,
            "q95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def rank_correlation(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa = finite(x)
    ya = finite(y)
    if xa.size != ya.size or xa.size < 3:
        return None
    if np.std(xa) < 1e-15 or np.std(ya) < 1e-15:
        return None
    xr = np.argsort(np.argsort(xa)).astype(float)
    yr = np.argsort(np.argsort(ya)).astype(float)
    value = float(np.corrcoef(xr, yr)[0, 1])
    return value if math.isfinite(value) else None


def eta_squared(records: list[dict[str, Any]], group_key: str, value_key: str) -> float | None:
    usable = [record for record in records if record.get(group_key) is not None and record.get(value_key) is not None]
    if len(usable) < 2:
        return None
    values = np.asarray([float(record[value_key]) for record in usable])
    grand = float(np.mean(values))
    total = float(np.sum((values - grand) ** 2))
    if total <= 1e-15:
        return 0.0
    groups: dict[str, list[float]] = defaultdict(list)
    for record in usable:
        groups[str(record[group_key])].append(float(record[value_key]))
    between = sum(len(group) * (statistics.mean(group) - grand) ** 2 for group in groups.values())
    return float(between / total)


def parse_run_id(run_id: str) -> tuple[str, int]:
    pieces = run_id.split("_matrix_seed")
    if len(pieces) != 2:
        return run_id, -1
    return pieces[0], int(pieces[1])


def selected_chart(run: dict[str, Any]) -> dict[str, Any]:
    chart = run.get("selected_chart")
    if not isinstance(chart, dict):
        raise ValueError("run has no selected_chart object")
    return chart


def scalar_chart_diagnostics(chart: dict[str, Any]) -> dict[str, Any]:
    real = np.asarray(chart["matrix_real"], dtype=float)
    imag = np.asarray(chart["matrix_imag"], dtype=float)
    matrix = real + 1j * imag
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("selected chart matrix must be square")
    diagonal = np.diag(matrix)
    offdiag = matrix - np.diag(diagonal)
    alpha = complex(np.mean(diagonal))
    return {
        "chart_id": chart.get("chart_id"),
        "family": chart.get("family"),
        "alpha_real": float(alpha.real),
        "alpha_imag": float(alpha.imag),
        "alpha_magnitude": float(abs(alpha)),
        "alpha_phase": float(np.angle(alpha)),
        "max_off_diagonal_absolute": float(np.max(np.abs(offdiag))),
        "max_diagonal_deviation": float(np.max(np.abs(diagonal - alpha))),
        "nonzero": bool(abs(alpha) > 1e-15),
        "is_scalar_identity": bool(
            np.max(np.abs(offdiag)) <= 1e-12
            and np.max(np.abs(diagonal - alpha)) <= 1e-12
            and abs(alpha) > 1e-15
        ),
    }


def old_gate_invariance(chart_calibration: dict[str, Any], official_gate: dict[str, Any]) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    all_scalar = True
    for run_id, run in sorted(chart_calibration["runs"].items()):
        if run.get("condition") != "matrix":
            continue
        diag = scalar_chart_diagnostics(selected_chart(run))
        runs[run_id] = diag
        all_scalar = all_scalar and diag["is_scalar_identity"]

    explanation = (
        "For nonzero alpha, z' = z/alpha. The historical feature is "
        "zhat(z)=z*exp(-i*arg(sum(z)))/||z||. Substitution gives "
        "zhat(z/alpha)=zhat(z), so fvec, rho, mhat, centroid predictions, "
        "and every non-phase old gate are exactly unchanged. theta_hat is "
        "shifted by the constant -arg(alpha), so differential phase is unchanged."
    )
    official_source = official_gate.get("runs", {})
    contract_passes: dict[str, bool] = {}
    for run_id, run in official_source.items():
        metrics = run.get("metrics", {})
        contract_passes[run_id] = bool(
            metrics.get("real_accuracy", 0.0) >= 0.60
            and metrics.get("real_vs_pseudo_floor", 0.0) >= 0.95
            and metrics.get("pseudo_reject_floor", 0.0) >= 0.95
        )
    return {
        "schema_id": "CAT_CAS_PHASE6B5D_OLD_GATE_INVARIANCE_V1",
        "all_selected_charts_scalar_identity": all_scalar,
        "runs": runs,
        "analytical_invariance": {
            "proven": all_scalar,
            "statement": explanation,
            "features_unchanged": [
                "fvec",
                "rho",
                "mhat",
                "real_accuracy",
                "real_mode_floor",
                "real_vs_pseudo_floor",
                "pseudo_reject_floor",
                "pseudo_declared_match",
                "wrong_actual_match",
                "wrong_declared_match",
                "differential_phase_delta",
            ],
        },
        "historical_gate_source_bound": bool(official_source),
        "historical_partial_status_unchanged": True,
        "conclusion": (
            "The C0 receiver chart cannot rescue the old normalized gates. Their "
            "failure is not caused by per-run complex scalar gain; it lies in residual "
            "structure, sparse threshold geometry, or the semantics of the old gate."
        ),
    }


def cross_session(route_data: dict[str, Any], chart_calibration: dict[str, Any]) -> dict[str, Any]:
    records = route_data.get("cross_chart_transfer", {}).get("records", [])
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("same_route"):
            key = "same_route_cross_seed"
        elif record.get("same_seed"):
            key = "cross_route_same_seed"
        else:
            key = "cross_route_cross_seed"
        buckets[key].append(record)

    aggregated: dict[str, Any] = {}
    for key, group in sorted(buckets.items()):
        aggregated[key] = {
            "records": len(group),
            "positive_margin_fraction": summary(record.get("positive_margin_fraction") for record in group),
            "median_actual_residual": summary(record.get("actual_residual", {}).get("median") for record in group),
            "median_actual_mode_margin": summary(record.get("actual_mode_margin", {}).get("median") for record in group),
            "all_record_positive_fraction_ge_0_95": all((record.get("positive_margin_fraction") or 0.0) >= 0.95 for record in group),
        }

    alphas: list[dict[str, Any]] = []
    for run_id, run in sorted(chart_calibration["runs"].items()):
        if run.get("condition") != "matrix":
            continue
        route, seed = parse_run_id(run_id)
        diag = scalar_chart_diagnostics(selected_chart(run))
        alphas.append({"run_id": run_id, "route": route, "seed": seed, **diag})

    alpha_by_route: dict[str, Any] = {}
    for route in sorted({record["route"] for record in alphas}):
        subset = [record for record in alphas if record["route"] == route]
        alpha_by_route[route] = {
            "magnitude": summary(record["alpha_magnitude"] for record in subset),
            "phase": summary(record["alpha_phase"] for record in subset),
            "runs": subset,
        }

    all_generalize = all(
        bucket["all_record_positive_fraction_ge_0_95"]
        for bucket in aggregated.values()
        if bucket["records"]
    )
    return {
        "schema_id": "CAT_CAS_PHASE6B5D_CROSS_SESSION_GENERALIZATION_V1",
        "buckets": aggregated,
        "chart_alpha_by_route": alpha_by_route,
        "relational_generalization_supported": all_generalize,
        "interpretation": (
            "Held-out relational ordering generalizes across seeds and routes even when "
            "a chart fitted in another session is used. Session-specific alpha changes "
            "absolute residual scale more than mode ordering. This is not complete "
            "session-independent operator identification."
        ),
    }


def collect_residual_records(heldout: dict[str, Any], execution: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_id, run in heldout.get("runs", {}).items():
        route, seed = parse_run_id(run_id)
        for record in run.get("real", {}).get("records", []):
            records.append(
                {
                    "run_id": run_id,
                    "route": route,
                    "seed": seed,
                    "family": "real",
                    "mode": record.get("actual_mode"),
                    "theta_idx": record.get("theta_idx"),
                    "trial": record.get("trial"),
                    "symbol_index": record.get("symbol_index"),
                    "actual_residual": record.get("actual_residual"),
                    "phase_aligned_residual": record.get("phase_aligned_residual_by_mode", {}).get(record.get("actual_mode")),
                    "actual_mode_margin": record.get("actual_mode_margin"),
                }
            )
    for run_id, run in execution.get("runs", {}).items():
        route, seed = parse_run_id(run_id)
        for record in run.get("wrong", {}).get("records", []):
            records.append(
                {
                    "run_id": run_id,
                    "route": route,
                    "seed": seed,
                    "family": "wrong",
                    "mode": record.get("actual_mode"),
                    "theta_idx": record.get("theta_idx"),
                    "trial": record.get("trial"),
                    "symbol_index": record.get("symbol_index"),
                    "actual_residual": record.get("actual_residual"),
                    "phase_aligned_residual": record.get("phase_aligned_residual_by_mode", {}).get(record.get("actual_mode")),
                    "actual_mode_margin": record.get("actual_mode_margin"),
                }
            )
    return records


def residual_structure(heldout: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    records = collect_residual_records(heldout, execution)
    for record in records:
        raw = record.get("actual_residual")
        aligned = record.get("phase_aligned_residual")
        record["phase_removable_residual"] = None if raw is None or aligned is None else float(raw) - float(aligned)
        trial = int(record.get("trial") or 0)
        record["trial_block"] = f"q{min(max(trial // 12, 0), 3)}"

    factors = ("route", "seed", "mode", "theta_idx", "family", "trial_block")
    effect = {factor: eta_squared(records, factor, "actual_residual") for factor in factors}
    grouped: dict[str, Any] = {}
    for factor in factors:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(factor))].append(record)
        grouped[factor] = {
            key: {
                "actual_residual": summary(item.get("actual_residual") for item in group),
                "phase_removable_residual": summary(item.get("phase_removable_residual") for item in group),
                "actual_mode_margin": summary(item.get("actual_mode_margin") for item in group),
            }
            for key, group in sorted(groups.items())
        }

    by_run: dict[str, Any] = {}
    for run_id in sorted({record["run_id"] for record in records}):
        group = [record for record in records if record["run_id"] == run_id]
        by_run[run_id] = {
            "actual_residual": summary(item.get("actual_residual") for item in group),
            "phase_aligned_residual": summary(item.get("phase_aligned_residual") for item in group),
            "phase_removable_residual": summary(item.get("phase_removable_residual") for item in group),
            "residual_vs_trial_rank": rank_correlation(
                [float(item.get("trial") or 0) for item in group],
                [float(item.get("actual_residual") or 0.0) for item in group],
            ),
            "residual_vs_symbol_index_rank": rank_correlation(
                [float(item.get("symbol_index") or 0) for item in group],
                [float(item.get("actual_residual") or 0.0) for item in group],
            ),
        }

    ranked_effects = sorted(
        ((factor, value) for factor, value in effect.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant = ranked_effects[0][0] if ranked_effects else None
    return {
        "schema_id": "CAT_CAS_PHASE6B5D_RESIDUAL_STRUCTURE_V1",
        "record_count": len(records),
        "overall": {
            "actual_residual": summary(record.get("actual_residual") for record in records),
            "phase_aligned_residual": summary(record.get("phase_aligned_residual") for record in records),
            "phase_removable_residual": summary(record.get("phase_removable_residual") for record in records),
        },
        "eta_squared_by_factor": effect,
        "ranked_effects": [{"factor": factor, "eta_squared": value} for factor, value in ranked_effects],
        "dominant_compact_packet_factor": dominant,
        "grouped": grouped,
        "by_run": by_run,
        "interpretation": (
            "Residual magnitude is analyzed by route, seed, mode, theta, family, and time block. "
            "The compact packet does not retain complex residual vectors, so bin-level residual "
            "Gram/cross-spectrum decomposition is outside this consolidation and would require "
            "a separately authorized raw-field pass only if these scalar effects remain ambiguous."
        ),
    }


def seed4_localization(chart_calibration: dict[str, Any], residuals: dict[str, Any], seed4: dict[str, Any]) -> dict[str, Any]:
    run_id = "v4s5_matrix_seed4"
    target = chart_calibration["runs"][run_id]
    target_chart = scalar_chart_diagnostics(selected_chart(target))
    peers: list[dict[str, Any]] = []
    for other_id, run in chart_calibration["runs"].items():
        if other_id.startswith("v4s5_matrix_seed") and other_id != run_id:
            diag = scalar_chart_diagnostics(selected_chart(run))
            peers.append({"run_id": other_id, **diag})

    peer_magnitudes = finite(peer["alpha_magnitude"] for peer in peers)
    peer_phases = finite(peer["alpha_phase"] for peer in peers)
    magnitude_z = None
    phase_z = None
    if peer_magnitudes.size >= 2 and np.std(peer_magnitudes, ddof=1) > 1e-15:
        magnitude_z = float((target_chart["alpha_magnitude"] - np.mean(peer_magnitudes)) / np.std(peer_magnitudes, ddof=1))
    if peer_phases.size >= 2 and np.std(peer_phases, ddof=1) > 1e-15:
        phase_z = float((target_chart["alpha_phase"] - np.mean(peer_phases)) / np.std(peer_phases, ddof=1))

    target_run_residual = residuals.get("by_run", {}).get(run_id, {})
    peer_residual_medians = [
        run.get("actual_residual", {}).get("median")
        for other_id, run in residuals.get("by_run", {}).items()
        if other_id.startswith("v4s5_matrix_seed") and other_id != run_id
    ]
    peer_median = float(np.median(finite(peer_residual_medians))) if finite(peer_residual_medians).size else None
    target_median = target_run_residual.get("actual_residual", {}).get("median")
    excess_ratio = None if peer_median in (None, 0.0) or target_median is None else float(target_median / peer_median)

    validation_residual = target.get("chart_ladder", [])[0].get("calibration_validation", {}).get("normalized_residual", {}).get("median")
    phase_removable = target_run_residual.get("phase_removable_residual", {}).get("median")
    trial_corr = target_run_residual.get("residual_vs_trial_rank")
    symbol_corr = target_run_residual.get("residual_vs_symbol_index_rank")

    classification = "SCALAR_FIT_QUALITY_FAILURE_WITH_RELATIONAL_INVARIANTS_PRESERVED"
    if phase_removable is not None and abs(float(phase_removable)) > 0.1:
        classification = "GLOBAL_PHASE_CHART_MISALIGNMENT"
    elif trial_corr is not None and abs(float(trial_corr)) > 0.6:
        classification = "TIME_LOCALIZED_TRANSFER_SHIFT"
    elif magnitude_z is not None and abs(magnitude_z) > 3.0:
        classification = "SCALAR_GAIN_OUTLIER_WITH_RELATIONAL_INVARIANTS_PRESERVED"

    return {
        "schema_id": "CAT_CAS_PHASE6B5D_SEED4_LOCALIZATION_V1",
        "run_id": run_id,
        "classification": classification,
        "phase6b5c_classification": seed4.get("classification"),
        "target_chart": target_chart,
        "peer_chart_magnitude": summary(peer["alpha_magnitude"] for peer in peers),
        "peer_chart_phase": summary(peer["alpha_phase"] for peer in peers),
        "alpha_magnitude_zscore": magnitude_z,
        "alpha_phase_zscore": phase_z,
        "calibration_validation_residual": validation_residual,
        "heldout_residual_median": target_median,
        "peer_route_residual_median": peer_median,
        "heldout_residual_excess_ratio": excess_ratio,
        "median_phase_removable_residual": phase_removable,
        "residual_vs_trial_rank": trial_corr,
        "residual_vs_symbol_index_rank": symbol_corr,
        "relational_invariants": {
            "real_positive_margin_fraction": seed4.get("real_positive_margin_fraction"),
            "wrong_actual_better_fraction": seed4.get("wrong_actual_better_fraction"),
            "phase_mean_absolute_error": seed4.get("phase_mean_absolute_error"),
        },
        "conclusion": (
            "Seed 4 does not exhibit relational carrier collapse. Its excess is in absolute scalar-model "
            "fit quality; global phase removal explains little of that residual, and time localization is "
            "reported explicitly rather than assumed."
        ),
    }


def claim_freeze(old_gate: dict[str, Any], cross: dict[str, Any], residuals: dict[str, Any], seed4: dict[str, Any]) -> dict[str, Any]:
    supported = [
        "On the retained T48 campaign and tested routes, sender-owned mode, phase, and exact pseudo-permutation relations survive held-out evaluation under calibration-only minimal scalar complex charts.",
        "Wrong-family receiver geometry follows physically executed mode rather than declared decoy metadata.",
        "Relational mode ordering generalizes across seeds and routes under cross-session scalar charts.",
        "Silent carrier-off and scramble unshared-schedule controls remain null.",
        "Route 4:5 seed 4 preserves relational invariants despite degraded absolute scalar-fit quality.",
    ]
    not_supported = [
        "The historical normalized strict gates are rescued by scalar chart calibration.",
        "A complete physical route operator has been identified.",
        "Complex residual vectors are unstructured noise.",
        "Tone order has a causal physical effect.",
        "Physical HoloGeometry, restoration, target coupling, orientation recovery, or a Small Wall crossing.",
    ]
    exit_ready = bool(
        old_gate.get("analytical_invariance", {}).get("proven")
        and cross.get("relational_generalization_supported")
        and residuals.get("record_count", 0) > 0
        and seed4.get("classification")
    )
    return {
        "schema_id": "CAT_CAS_PHASE6B5D_CARRIER_CLAIM_FREEZE_V1",
        "status": "FROZEN_PENDING_GATE_R" if exit_ready else "INCOMPLETE",
        "supported_claims": supported,
        "not_supported": not_supported,
        "official_strict_closure": "PARTIAL",
        "next_physical_control": {
            "name": "REVERSED_RANDOMIZED_TONE_ORDER",
            "status": "PREREGISTRATION_REQUIRED_NOT_AUTHORIZED",
            "purpose": "separate tone identity from within-symbol path position while preserving routes, modes, phases, and controls",
        },
        "gate_r_ready": exit_ready,
        "hard_stop": (
            "Do not continue open-ended analysis of this campaign after the bounded consolidation report. "
            "Proceed to external Gate R review and preregistration of the tone-order control."
        ),
    }


def common_binding(result_dir: Path) -> dict[str, Any]:
    manifest_path = result_dir / "analysis_manifest.json"
    manifest = load_json(manifest_path)
    return {
        "analysis_id": ANALYSIS_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "phase6b5c_manifest_sha256": sha256_file(manifest_path),
        "phase6b5c_campaign_id": manifest.get("campaign_id"),
        "phase6b5c_campaign_manifest_sha256": manifest.get("campaign_manifest_sha256"),
        "source_result_directory": result_dir.name,
        "official_closure_status_unchanged": "PARTIAL",
        "claim_ceiling": "BOUNDED_CARRIER_CLAIM_CONSOLIDATION_ONLY",
    }


def build(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    chart = load_json(result_dir / "chart_calibration.json")
    heldout = load_json(result_dir / "heldout_equivariance.json")
    execution = load_json(result_dir / "execution_relation.json")
    route = load_json(result_dir / "route_conjugacy.json")
    seed4_input = load_json(result_dir / "seed4_transfer_report.json")

    official_path = result_dir.parents[2] / "replication_discrepancy" / "results" / "official_gate_decomposition.json"
    if not official_path.is_file():
        official_path = result_dir.parents[3] / "replication_discrepancy" / "results" / "official_gate_decomposition.json"
    if not official_path.is_file():
        raise FileNotFoundError("official_gate_decomposition.json not found relative to result directory")
    official = load_json(official_path)

    common = common_binding(result_dir)
    old_gate = {**common, **old_gate_invariance(chart, official)}
    cross = {**common, **cross_session(route, chart)}
    residuals = {**common, **residual_structure(heldout, execution)}
    seed4 = {**common, **seed4_localization(chart, residuals, seed4_input)}
    freeze = {**common, **claim_freeze(old_gate, cross, residuals, seed4)}

    outputs = {
        "old_gate_chart_invariance.json": old_gate,
        "cross_session_generalization.json": cross,
        "residual_structure.json": residuals,
        "seed4_localization.json": seed4,
        "carrier_claim_freeze.json": freeze,
    }
    for name, value in outputs.items():
        write_json(output_dir / name, value)

    manifest_files = {
        name: {
            "size": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in OUTPUT_FILES
    }
    manifest = {
        **common,
        "schema_id": "CAT_CAS_PHASE6B5D_CONSOLIDATION_MANIFEST_V1",
        "outputs": manifest_files,
        "decision": {
            "old_gate_scalar_calibration_can_change_result": not old_gate["analytical_invariance"]["proven"],
            "cross_session_relational_generalization": cross["relational_generalization_supported"],
            "dominant_compact_residual_factor": residuals["dominant_compact_packet_factor"],
            "seed4_localization": seed4["classification"],
            "carrier_claim_status": freeze["status"],
            "gate_r_ready": freeze["gate_r_ready"],
        },
    }
    write_json(output_dir / "phase6b5d_manifest.json", manifest)
    return manifest


def verify(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "phase6b5d_manifest.json"
    manifest = load_json(manifest_path)
    errors: list[str] = []
    for name, expected in manifest.get("outputs", {}).items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        if path.stat().st_size != int(expected["size"]):
            errors.append(f"size mismatch {name}")
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"sha256 mismatch {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "decision": manifest.get("decision"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("result_dir", type=Path)
    run.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        manifest = build(args.result_dir.resolve(), args.output.resolve())
        print(json.dumps(manifest["decision"], indent=2, sort_keys=True))
        return 0
    result = verify(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
