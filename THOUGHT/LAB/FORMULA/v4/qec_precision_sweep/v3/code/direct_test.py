"""v3 analysis: test the formula directly -- no learned α, no linear model wrapper.

Structural fixes from v2:
  1. Predict R directly: R_pred = (E / grad_S) * sigma^Df. No α, β fitting.
  2. All operational definitions use per-condition measurable quantities.
  3. Slope test: does ln(R_actual) vs Df slope match ln(sigma)?
  4. Fit diagnostics: report α, β from R_actual ~ α*R_pred + β as a check.
  5. Cross-noise-model: same definitions on DEPOL and MEAS.

Operational definitions (v3, all measurable from each condition):
  E       = 1.0                             (normalized signal power)
  grad_S  = syndrome_density                (fraction of detectors firing)
  sigma   = 1 - syndrome_density            (fraction of quiet detectors)
  Df      = d                               (code distance)
  R_pred  = (E / grad_S) * sigma^Df         (direct prediction, no fitting)
  R_actual = ln(physical_error_rate / logical_error_rate_laplace)  (log suppression)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def compute_v3_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute v3 formula predictions using per-condition syndrome measurements."""
    results = []
    eps = 1e-60
    for r in rows:
        p = r["physical_error_rate"]
        d = int(r["distance"])
        syn_density = float(r["syndrome_density"])

        E_val = 1.0
        grad_S_val = max(syn_density, eps)
        sigma_val = max(1.0 - syn_density, eps)
        Df_val = float(d)

        R_pred = (E_val / grad_S_val) * (sigma_val ** Df_val)
        log_R_pred = math.log(max(R_pred, eps))

        log_E_over_grad_S = math.log(max(E_val / grad_S_val, eps))
        Df_ln_sigma = Df_val * math.log(max(sigma_val, eps))

        log_suppression = r["log_suppression"]
        survival = r.get("survival", 1.0 - r["logical_error_rate"])

        results.append({
            **r,
            "E_v3": E_val,
            "grad_S_v3": grad_S_val,
            "sigma_v3": sigma_val,
            "Df_v3": Df_val,
            "R_predicted_v3": R_pred,
            "log_R_predicted_v3": log_R_pred,
            "log_E_over_grad_S_v3": log_E_over_grad_S,
            "Df_ln_sigma_v3": Df_ln_sigma,
        })
    return results


def direct_prediction_metrics(rows: list[dict[str, Any]], heldout: set[int]) -> dict[str, Any]:
    """R_predicted vs R_actual -- no learned coefficients."""
    test = [r for r in rows if int(r["distance"]) in heldout]
    y_pred = np.array([r["log_R_predicted_v3"] for r in test], dtype=float)
    y_actual = np.array([r["log_suppression"] for r in test], dtype=float)

    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))

    # diagnostic: if formula is correct, y_actual ≈ y_pred (α≈1, β≈0)
    A = np.column_stack([y_pred, np.ones_like(y_pred)])
    coeffs, residuals, rank, singular = np.linalg.lstsq(A, y_actual, rcond=None)
    alpha = float(coeffs[0])
    beta = float(coeffs[1])
    residual_std = float(math.sqrt(np.mean((y_actual - (alpha * y_pred + beta))**2)))

    return {
        "direct_mae": mae,
        "direct_r2": r2,
        "fit_alpha": alpha,
        "fit_beta": beta,
        "fit_residual_std": residual_std,
        "alpha_is_near_1": abs(alpha - 1.0) < 0.2,
        "beta_is_near_0": abs(beta) < 0.5,
        "per_point": [
            {
                "basis": r["basis"],
                "distance": r["distance"],
                "physical_error_rate": r["physical_error_rate"],
                "R_actual": float(ya),
                "R_predicted": float(yp),
                "abs_error": float(abs(ya - yp)),
            }
            for r, ya, yp in zip(test, y_actual, y_pred)
        ],
    }


def slope_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Test: for each p, does ln(R_actual) vs Df have slope ≈ ln(sigma)?

    The formula claims: ln(R) = ln(E/grad_S) + Df * ln(sigma)
    So ln(R) vs Df should be linear with slope ln(sigma) and intercept ln(E/grad_S).
    """
    ps = sorted({float(r["physical_error_rate"]) for r in rows})
    by_distance = {}
    for d in sorted({int(r["distance"]) for r in rows}):
        by_distance[d] = [r for r in rows if int(r["distance"]) == d]

    results = []
    for p in ps:
        for basis in sorted({r["basis"] for r in rows}):
            points = [r for r in rows if float(r["physical_error_rate"]) == p and r["basis"] == basis]
            if len(points) < 2:
                continue
            distances = np.array([int(r["distance"]) for r in points], dtype=float)
            log_R_actual = np.array([r["log_suppression"] for r in points], dtype=float)
            log_sigma_pred = np.array([math.log(max(r["sigma_v3"], 1e-60)) for r in points], dtype=float)

            A = np.column_stack([distances, np.ones_like(distances)])
            coeffs, _, _, _ = np.linalg.lstsq(A, log_R_actual, rcond=None)
            empirical_slope = float(coeffs[0])
            empirical_intercept = float(coeffs[1])

            sigma_predicted = np.exp(np.mean(log_sigma_pred))
            predicted_slope = float(np.mean(log_sigma_pred))

            predicted_intercept = float(np.mean([
                math.log(max(1.0 / max(r["syndrome_density"], 1e-60), 1e-60))
                for r in points
            ]))

            slope_match = abs(empirical_slope - predicted_slope) < max(0.5, abs(predicted_slope) * 0.5)

            results.append({
                "physical_error_rate": p,
                "basis": basis,
                "num_distances": len(points),
                "empirical_slope": empirical_slope,
                "predicted_slope": predicted_slope,
                "slope_error": empirical_slope - predicted_slope,
                "slope_match": slope_match,
                "empirical_intercept": empirical_intercept,
                "predicted_intercept": predicted_intercept,
                "intercept_error": empirical_intercept - predicted_intercept,
                "sigma_v3_mean": float(np.mean([r["sigma_v3"] for r in points])),
                "grad_S_v3_mean": float(np.mean([r["grad_S_v3"] for r in points])),
                "below_threshold_crossover": p < 0.007071067811865475,
            })

    if not results:
        return {"slope_results": [], "slope_match_fraction": None}

    match_count = sum(1 for r in results if r["slope_match"])
    return {
        "slope_results": results,
        "slope_match_fraction": match_count / len(results),
        "total_conditions": len(results),
    }


def evaluate_baselines(rows: list[dict[str, Any]], heldout: set[int]) -> dict[str, Any]:
    """Fit standard baselines for context -- same as v1/v2 for comparability."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = [r for r in rows if int(r["distance"]) not in heldout]
    test = [r for r in rows if int(r["distance"]) in heldout]
    y_train = np.array([r["log_suppression"] for r in train], dtype=float)
    y_test = np.array([r["log_suppression"] for r in test], dtype=float)

    p_train = np.array([r["physical_error_rate"] for r in train], dtype=float)
    d_train = np.array([r["distance"] for r in train], dtype=float)
    bz_train = np.array([1.0 if r["basis"] == "z" else 0.0 for r in train], dtype=float)
    p_test = np.array([r["physical_error_rate"] for r in test], dtype=float)
    d_test = np.array([r["distance"] for r in test], dtype=float)
    bz_test = np.array([1.0 if r["basis"] == "z" else 0.0 for r in test], dtype=float)

    eps = 1e-60

    def make_feature(name, p_arr, d_arr, bz_arr):
        if name == "standard_qec_scaling":
            return np.column_stack([np.log(np.maximum(p_arr, eps)), d_arr, d_arr * np.log(np.maximum(p_arr, eps)), bz_arr])
        if name == "p_only":
            return np.column_stack([np.log(np.maximum(p_arr, eps)), bz_arr])
        if name == "distance_only":
            return np.column_stack([d_arr, bz_arr])
        raise ValueError(name)

    out = {}
    for name in ["standard_qec_scaling", "p_only", "distance_only"]:
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(make_feature(name, p_train, d_train, bz_train), y_train)
        pred_test = model.predict(make_feature(name, p_test, d_test, bz_test))
        out[name] = {
            "test_mae": float(mean_absolute_error(y_test, pred_test)),
            "test_r2": float(r2_score(y_test, pred_test)),
        }
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    dp = payload["direct"]
    sl = payload["slope"]
    bl = payload["baselines"]
    nm = payload.get("noise_model", "unknown")

    lines = [
        "# QEC Precision Sweep v3 -- Direct Formula Test",
        "",
        f"Run id: `{payload['run_id']}`",
        f"Noise model: `{nm}`",
        f"Source data: `{payload['source_run']}`",
        f"UTC time: `{payload['created_utc']}`",
        "",
        "## v3 Operational Definitions (all per-condition, measurable)",
        "",
        "| Symbol | Definition | Source |",
        "|--------|-----------|--------|",
        "| `E` | `1.0` | normalized signal power |",
        "| `grad_S` | syndrome density | fraction of detectors firing per shot |",
        "| `sigma` | `1 - syndrome_density` | fraction of quiet detectors |",
        "| `Df` | `d` | surface-code distance |",
        "| `R_pred` | `(E / grad_S) * sigma^Df` | direct prediction, no fitting |",
        "",
        "**Key difference from v2**: no p_th, no sqrt/threshold ratios, no learned α/β.",
        "Every quantity is measured directly from the syndrome data per condition.",
        "",
        "## 1. Direct Prediction (no learned coefficients)",
        "",
        f"- Direct MAE (test): `{dp['direct_mae']:.4f}`",
        f"- Direct R2 (test): `{dp['direct_r2']:.4f}`",
        "",
        "### Zero-Fitting Diagnostic",
        "",
        "If the formula is correct, R_actual = 1.0 * R_predicted + 0.0.",
        "",
        f"- Fitted alpha: `{dp['fit_alpha']:.4f}`  (ideal = 1.0)",
        f"- Fitted beta:  `{dp['fit_beta']:.4f}`  (ideal = 0.0)",
        f"- Residual std: `{dp['fit_residual_std']:.4f}`",
        f"- alpha near 1: `{dp['alpha_is_near_1']}`",
        f"- beta near 0:  `{dp['beta_is_near_0']}`",
        "",
        "### Baseline Comparison (with learned coefficients, for context)",
        "",
        "| Model | Test MAE | Test R2 |",
        "|---|---:|---:|",
    ]
    for name, metrics in sorted(bl.items(), key=lambda kv: kv[1]["test_mae"]):
        lines.append(f"| `{name}` | {metrics['test_mae']:.4f} | {metrics['test_r2']:.4f} |")

    lines.extend([
        "",
        "## 2. Slope Test: Does ln(R) grow as Df * ln(sigma)?",
        "",
        "For each error rate and basis, the formula claims:",
        "```",
        "ln(R_actual) = ln(E/grad_S) + Df * ln(sigma)",
        "```",
        "",
        f"Slope match fraction: `{sl['slope_match_fraction']:.2%}` ({sl.get('total_conditions', 0)} p/basis combos)",
        "",
        "| p | basis | n_dists | empir slope | pred slope | slope error | match |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in sl.get("slope_results", []):
        lines.append(
            f"| {r['physical_error_rate']:.4f} | {r['basis']} | {r['num_distances']} | "
            f"{r['empirical_slope']:.4f} | {r['predicted_slope']:.4f} | "
            f"{r['slope_error']:.4f} | {r['slope_match']} |"
        )

    lines.extend([
        "",
        "### Intercept Check",
        "",
        "| p | basis | empir intercept | pred intercept | error |",
        "|---:|---:|---:|---:|---:|",
    ])
    for r in sl.get("slope_results", []):
        lines.append(
            f"| {r['physical_error_rate']:.4f} | {r['basis']} | "
            f"{r['empirical_intercept']:.4f} | {r['predicted_intercept']:.4f} | "
            f"{r['intercept_error']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Verdict",
        "",
    ])

    if dp["alpha_is_near_1"] and dp["beta_is_near_0"]:
        lines.append("**PASS**: Alpha near 1 and beta near 0. Formula predicts R directly without recalibration.")
    elif dp["alpha_is_near_1"]:
        lines.append(f"**PARTIAL**: Alpha near 1, but beta = {dp['fit_beta']:.4f} (not near 0). Formula captures relative scaling but has systematic offset.")
    elif dp["beta_is_near_0"]:
        lines.append(f"**PARTIAL**: Beta near 0, but alpha = {dp['fit_alpha']:.4f} (not near 1). Formula's slope is wrong but intercept is right.")
    else:
        lines.append(f"**FAIL**: Alpha = {dp['fit_alpha']:.4f}, beta = {dp['fit_beta']:.4f}. Formula does not predict R without recalibration.")

    if sl.get("slope_match_fraction", 0) is not None:
        if sl["slope_match_fraction"] >= 0.75:
            lines.append(f"- Slope structure matches: {sl['slope_match_fraction']:.0%} of p/basis conditions show ln(R) ∝ Df * ln(sigma).")
        else:
            lines.append(f"- Slope structure weak: only {sl['slope_match_fraction']:.0%} of conditions match.")

    lines.extend([
        "",
        "## 4. Per-Point Errors (test distances only)",
        "",
        "| basis | d | p | R_actual | R_predicted | error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    points = sorted(dp["per_point"], key=lambda x: (x["physical_error_rate"], x["distance"], x["basis"]))
    for pt in points:
        lines.append(
            f"| {pt['basis']} | {pt['distance']} | {pt['physical_error_rate']:.4f} | "
            f"{pt['R_actual']:.4f} | {pt['R_predicted']:.4f} | {pt['abs_error']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sweep(source_dir: str) -> dict[str, Any]:
    src = Path(source_dir)
    for pattern in ["qec_precision_sweep_v2.json", "qec_precision_sweep.json"]:
        candidates = list(src.glob(pattern))
        if candidates:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No sweep JSON found in {source_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_sweep(args.source_dir)
    rows_raw = payload["conditions"]
    heldout = set(payload["config"]["heldout_distances"])
    noise_model = payload["config"].get("noise_model", "unknown")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = compute_v3_predictions(rows_raw)
    direct = direct_prediction_metrics(rows, heldout)
    slope = slope_test(rows)
    baselines = evaluate_baselines(rows_raw, heldout)

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "source_run": payload["run_id"],
        "noise_model": noise_model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direct": direct,
        "slope": slope,
        "baselines": baselines,
        "v3_definitions": {
            "E": "1.0 (normalized signal power)",
            "grad_S": "syndrome_density (fraction of detectors firing)",
            "sigma": "1 - syndrome_density (fraction of quiet detectors)",
            "Df": "surface-code distance",
            "R_pred": "(E / grad_S) * sigma^Df (direct, no fitting)",
        },
    }

    json_path = out_dir / "v3_analysis.json"
    report_path = out_dir / "V3_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)

    print(f"Direct MAE: {direct['direct_mae']:.4f}, R2: {direct['direct_r2']:.4f}")
    print(f"Alpha: {direct['fit_alpha']:.4f}, Beta: {direct['fit_beta']:.4f}")
    print(f"Slope match: {slope.get('slope_match_fraction', 'N/A')}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
