"""v4: empirically measured sigma from training-distance slopes.

Design:
  - sigma is MEASURED per physical error rate from training distances {3,5}:
      sigma_p = exp((ln(R_d5) - ln(R_d3)) / (5-3))
  - E is calibrated globally once from all training data:
      E = median( R_actual * grad_S / sigma_p^Df ) across training conditions
  - grad_S = syndrome_density (per condition, measured)
  - Df = d (code distance)
  - Prediction on held-out {7,9}: R_pred = (E / grad_S) * sigma_p^Df
  - Zero free parameters at test time.

Key difference from v3: sigma can naturally exceed 1 below threshold because it's
measured from the actual distance benefit, not computed as 1 - syndrome_density.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_sweep(source_dir: str) -> dict[str, Any]:
    src = Path(source_dir)
    for pattern in ["qec_precision_sweep_v2.json", "qec_precision_sweep.json"]:
        candidates = list(src.glob(pattern))
        if candidates:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No sweep JSON found in {source_dir}")


def measure_sigma_per_p(rows: list[dict[str, Any]], train_distances: set[int]) -> dict[tuple, float]:
    """Measure empirical sigma per (p, basis) from training-distance slope.

    sigma = exp(slope of ln(R) vs Df) across training distances.
    """
    groups = defaultdict(list)
    for r in rows:
        d = int(r["distance"])
        if d not in train_distances:
            continue
        key = (float(r["physical_error_rate"]), r["basis"])
        groups[key].append((d, r["log_suppression"]))

    sigma_map = {}
    for key, points in groups.items():
        if len(points) < 2:
            continue
        points.sort()
        ds = np.array([p[0] for p in points], dtype=float)
        log_Rs = np.array([p[1] for p in points], dtype=float)
        slope = (log_Rs[-1] - log_Rs[0]) / (ds[-1] - ds[0])
        sigma_map[key] = math.exp(slope)
    return sigma_map


def calibrate_E(rows: list[dict[str, Any]], train_distances: set[int],
                sigma_map: dict[tuple, float]) -> float:
    """Calibrate global E from all training conditions.

    Formula in log domain: ln(R) = ln(E) - ln(grad_S) + Df * ln(sigma)

    So: ln(E) = ln(R_actual) + ln(grad_S) - Df * ln(sigma)
      => E = exp(median of these estimates)
    """
    eps = 1e-60
    log_E_estimates = []
    for r in rows:
        d = int(r["distance"])
        if d not in train_distances:
            continue
        p = float(r["physical_error_rate"])
        key = (p, r["basis"])
        sigma_p = sigma_map.get(key)
        if sigma_p is None:
            continue

        log_R_actual = r["log_suppression"]
        log_grad_S = math.log(max(float(r["syndrome_density"]), eps))
        log_sigma = math.log(max(sigma_p, eps))

        log_E_est = log_R_actual + log_grad_S - d * log_sigma
        log_E_estimates.append(log_E_est)

    if not log_E_estimates:
        return 1.0
    return float(math.exp(np.median(log_E_estimates)))


def predict_v4(rows: list[dict[str, Any]], sigma_map: dict[tuple, float],
               E_val: float) -> list[dict[str, Any]]:
    """Compute v4 predictions for all conditions."""
    eps = 1e-60
    results = []
    for r in rows:
        d = int(r["distance"])
        p = float(r["physical_error_rate"])
        key = (p, r["basis"])
        sigma_p = sigma_map.get(key, 1.0)
        grad_S_val = max(float(r["syndrome_density"]), eps)

        R_pred = (E_val / grad_S_val) * (sigma_p ** d)
        log_R_pred = math.log(max(R_pred, eps))

        results.append({
            **r,
            "v4_sigma": sigma_p,
            "v4_E": E_val,
            "v4_grad_S": grad_S_val,
            "v4_Df": d,
            "v4_R_predicted": R_pred,
            "v4_log_R_predicted": log_R_pred,
        })
    return results


def evaluate_v4(rows: list[dict[str, Any]], heldout: set[int]) -> dict[str, Any]:
    """Direct prediction metrics on held-out distances."""
    test = [r for r in rows if int(r["distance"]) in heldout]
    y_pred = np.array([r["v4_log_R_predicted"] for r in test], dtype=float)
    y_actual = np.array([r["log_suppression"] for r in test], dtype=float)

    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))

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
                "sigma_used": r["v4_sigma"],
                "grad_S_used": r["v4_grad_S"],
            }
            for r, ya, yp in zip(test, y_actual, y_pred)
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    ev = payload["evaluation"]
    sigmas = payload["sigma_map_raw"]
    nm = payload.get("noise_model", "unknown")

    lines = [
        "# QEC Precision Sweep v4 -- Empirically Measured Parameters",
        "",
        f"Run id: `{payload['run_id']}`",
        f"Noise model: `{nm}`",
        f"Source data: `{payload['source_run']}`",
        f"UTC time: `{payload['created_utc']}`",
        "",
        "## Design",
        "",
        "- **sigma**: measured per (p, basis) from training-distance {3,5} slope",
        "  `sigma_p = exp((ln(R_d5) - ln(R_d3)) / (5-3))`",
        f"- **E**: calibrated globally as median across all training conditions: `E = {payload['E_calibrated']:.4f}`",
        "- **grad_S**: syndrome density (fraction of detectors firing, per condition)",
        "- **Df**: surface-code distance d",
        "- **No fitting at test time**: sigma and E are fixed before held-out prediction",
        "",
        "## Empirical Sigma Values (per p, basis -- from training distances only)",
        "",
        "| p | basis | sigma | ln(sigma) |",
        "|---:|---:|---:|---:|",
    ]
    for p, basis, sigma_val in sorted(sigmas):
        lines.append(f"| {p:.4f} | {basis} | {sigma_val:.4f} | {math.log(sigma_val):.4f} |")

    lines.extend([
        "",
        "## Prediction on Held-Out Distances (7, 9)",
        "",
        f"- Direct MAE: `{ev['direct_mae']:.4f}`",
        f"- Direct R2: `{ev['direct_r2']:.4f}`",
        "",
        "### Zero-Fitting Diagnostic",
        "",
        f"- Alpha: `{ev['fit_alpha']:.4f}` (ideal = 1.0)",
        f"- Beta:  `{ev['fit_beta']:.4f}` (ideal = 0.0)",
        f"- Residual std: `{ev['fit_residual_std']:.4f}`",
        f"- alpha near 1: `{ev['alpha_is_near_1']}`",
        f"- beta near 0:  `{ev['beta_is_near_0']}`",
        "",
        "### Verdict",
        "",
    ])

    if ev["alpha_is_near_1"] and ev["beta_is_near_0"]:
        lines.append("**PASS**: Alpha near 1 AND beta near 0. Formula predicts R directly with empirically measured parameters.")
    elif ev["alpha_is_near_1"]:
        lines.append(f"**PARTIAL**: Alpha near 1, beta = {ev['fit_beta']:.4f}. Slope structure correct, systematic offset remains.")
    elif ev["beta_is_near_0"]:
        lines.append(f"**PARTIAL**: Beta near 0, alpha = {ev['fit_alpha']:.4f}. Intercept correct, slope miscalibration.")
    else:
        lines.append(f"**FAIL**: Alpha = {ev['fit_alpha']:.4f}, beta = {ev['fit_beta']:.4f}. Structure does not transfer to held-out distances.")

    lines.extend([
        "",
        "### Per-Point Errors",
        "",
        "| basis | d | p | R_actual | R_pred | error |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pt in sorted(ev["per_point"], key=lambda x: (x["physical_error_rate"], x["distance"], x["basis"])):
        lines.append(
            f"| {pt['basis']} | {pt['distance']} | {pt['physical_error_rate']:.4f} | "
            f"{pt['R_actual']:.4f} | {pt['R_predicted']:.4f} | {pt['abs_error']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_sweep(args.source_dir)
    rows = payload["conditions"]
    train_distances = {3, 5}
    heldout = set(payload["config"]["heldout_distances"])
    noise_model = payload["config"].get("noise_model", "unknown")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    sigma_map = measure_sigma_per_p(rows, train_distances)
    E_val = calibrate_E(rows, train_distances, sigma_map)
    v4_rows = predict_v4(rows, sigma_map, E_val)
    evaluation = evaluate_v4(v4_rows, heldout)

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "source_run": payload["run_id"],
        "noise_model": noise_model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "E_calibrated": E_val,
        "sigma_map": {f"{p:.8f}|{b}": s for (p, b), s in sigma_map.items()},
        "sigma_map_raw": [(p, b, s) for (p, b), s in sigma_map.items()],
        "evaluation": evaluation,
        "design": {
            "sigma": "measured per (p,basis) from slope of ln(R) vs Df on training distances {3,5}",
            "E": f"calibrated globally as median across training conditions = {E_val:.4f}",
            "grad_S": "syndrome density per condition",
            "Df": "code distance",
            "prediction": "(E / grad_S) * sigma_p ^ Df on held-out distances {7,9}",
        },
    }

    json_path = out_dir / "v4_analysis.json"
    report_path = out_dir / "V4_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)

    print(f"E calibrated: {E_val:.4f}")
    print(f"Sigma values: {len(sigma_map)} p/basis combos")
    for (p, b), s in sorted(sigma_map.items()):
        print(f"  p={p:.4f} {b}: sigma={s:.4f}  ln(sigma)={math.log(s):.4f}")
    print(f"Direct MAE: {evaluation['direct_mae']:.4f}, R2: {evaluation['direct_r2']:.4f}")
    print(f"Alpha: {evaluation['fit_alpha']:.4f}, Beta: {evaluation['fit_beta']:.4f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
