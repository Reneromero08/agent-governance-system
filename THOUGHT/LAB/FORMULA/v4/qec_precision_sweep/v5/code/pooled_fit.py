"""v5: pooled bases, 3 training distances, robust sigma estimation.

Fixes from v4:
  - Pool X and Z bases (rotated surface code symmetry)
  - 3 training distances {3,5,7} for 3-point linear sigma fit
  - Test on {9}
  - Try grad_S = p (Light Cone canonical QEC mapping) alongside syndrome_density

Design:
  sigma_p = exp(slope of ln(R) vs Df across distances {3,5,7})
  E calibrated from training data (pooled bases)
  Predict on d=9 with zero free parameters.
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
    for pattern in ["combined_sweep.json", "qec_precision_sweep_v2.json", "qec_precision_sweep.json"]:
        candidates = list(src.glob(pattern))
        if candidates:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No sweep JSON found in {source_dir}")


def pool_bases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average log_suppression and syndrome_density across X/Z bases per (p,d)."""
    groups = defaultdict(list)
    for r in rows:
        key = (float(r["physical_error_rate"]), int(r["distance"]))
        groups[key].append(r)

    pooled = []
    for (p, d), group in groups.items():
        avg_log_supp = float(np.mean([r["log_suppression"] for r in group]))
        avg_syn_density = float(np.mean([r["syndrome_density"] for r in group]))
        avg_logical_rate = float(np.mean([r["logical_error_rate"] for r in group]))
        avg_logical_rate_lap = float(np.mean([r["logical_error_rate_laplace"] for r in group]))
        pooled.append({
            "physical_error_rate": p,
            "distance": d,
            "log_suppression": avg_log_supp,
            "syndrome_density": avg_syn_density,
            "logical_error_rate": avg_logical_rate,
            "logical_error_rate_laplace": avg_logical_rate_lap,
            "num_bases": len(group),
        })
    return pooled


def measure_sigma_per_p(rows: list[dict[str, Any]], train_distances: set[int]) -> dict[float, float]:
    """3-point linear fit of ln(R) vs Df across training distances, per p."""
    groups = defaultdict(list)
    for r in rows:
        d = int(r["distance"])
        if d not in train_distances:
            continue
        groups[float(r["physical_error_rate"])].append((d, r["log_suppression"]))

    sigma_map = {}
    for p, points in groups.items():
        if len(points) < 2:
            continue
        points.sort()
        ds = np.array([pt[0] for pt in points], dtype=float)
        log_Rs = np.array([pt[1] for pt in points], dtype=float)
        A = np.column_stack([ds, np.ones_like(ds)])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_Rs, rcond=None)
        slope = float(coeffs[0])
        sigma_map[p] = math.exp(slope)
    return sigma_map


def calibrate_E(rows: list[dict[str, Any]], train_distances: set[int],
                sigma_map: dict[float, float], grad_S_key: str = "syndrome_density") -> float:
    """Calibrate global E from training conditions in log domain."""
    eps = 1e-60
    log_E_estimates = []
    for r in rows:
        d = int(r["distance"])
        if d not in train_distances:
            continue
        p = float(r["physical_error_rate"])
        sigma_p = sigma_map.get(p)
        if sigma_p is None:
            continue

        if grad_S_key == "syndrome_density":
            grad_S_val = max(float(r["syndrome_density"]), eps)
        else:
            grad_S_val = max(p, eps)

        log_R = r["log_suppression"]
        log_grad_S = math.log(grad_S_val)
        log_sigma = math.log(max(sigma_p, eps))
        log_E_est = log_R + log_grad_S - d * log_sigma
        log_E_estimates.append(log_E_est)

    if not log_E_estimates:
        return 1.0
    return float(math.exp(np.median(log_E_estimates)))


def evaluate_v5(rows: list[dict[str, Any]], sigma_map: dict[float, float],
                E_val: float, heldout: set[int], grad_S_key: str = "syndrome_density") -> dict[str, Any]:
    """Predict held-out distances and compare to actual."""
    eps = 1e-60
    test_rows = [r for r in rows if int(r["distance"]) in heldout]
    y_pred = []
    y_actual = []
    per_point = []

    for r in test_rows:
        d = int(r["distance"])
        p = float(r["physical_error_rate"])
        sigma_p = sigma_map.get(p, 1.0)

        if grad_S_key == "syndrome_density":
            grad_S_val = max(float(r["syndrome_density"]), eps)
        else:
            grad_S_val = max(p, eps)

        R_pred = (E_val / grad_S_val) * (sigma_p ** d)
        log_R_pred = math.log(max(R_pred, eps))
        log_R_actual = r["log_suppression"]

        y_pred.append(log_R_pred)
        y_actual.append(log_R_actual)
        per_point.append({
            "physical_error_rate": p,
            "distance": d,
            "R_actual": log_R_actual,
            "R_predicted": log_R_pred,
            "abs_error": abs(log_R_actual - log_R_pred),
            "sigma_used": sigma_p,
            "grad_S_used": grad_S_val,
        })

    y_pred_arr = np.array(y_pred, dtype=float)
    y_actual_arr = np.array(y_actual, dtype=float)

    mae = float(mean_absolute_error(y_actual_arr, y_pred_arr))
    r2 = float(r2_score(y_actual_arr, y_pred_arr))

    A = np.column_stack([y_pred_arr, np.ones_like(y_pred_arr)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y_actual_arr, rcond=None)
    alpha = float(coeffs[0])
    beta = float(coeffs[1])

    return {
        "direct_mae": mae,
        "direct_r2": r2,
        "fit_alpha": alpha,
        "fit_beta": beta,
        "alpha_is_near_1": abs(alpha - 1.0) < 0.2,
        "beta_is_near_0": abs(beta) < 0.5,
        "num_test_points": len(test_rows),
        "per_point": per_point,
    }


def write_report(path: Path, results: dict[str, Any]) -> None:
    nm = results.get("noise_model", "unknown")
    lines = [
        "# QEC Precision Sweep v5 -- Pooled Bases, 3-Point Sigma",
        "",
        f"Run id: `{results['run_id']}`",
        f"Noise model: `{nm}`",
        f"Source data: `{results['source_run']}`",
        f"UTC time: `{results['created_utc']}`",
        "",
        "## Design",
        "",
        "- Bases: **X and Z pooled** (averaged log_suppression per condition)",
        "- Training distances: **{3, 5, 7}** (3-point linear fit for sigma)",
        "- Test distance: **{9}**",
        "- sigma: exp(slope of ln(R) vs Df across {3,5,7})",
        "- E: calibrated globally from training (log-domain median)",
        "",
    ]
    for key_label, res in results["grad_S_variants"].items():
        sigma_map = res["sigma_map"]
        ev = res["evaluation"]
        E_val = res["E"]
        lines.extend([
            f"## grad_S = {key_label}",
            "",
            f"E calibrated: `{E_val:.4f}`",
            "",
            "### Sigma per p (from 3-point training fit)",
            "",
            "| p | sigma | ln(sigma) |",
            "|---:|---:|---:|",
        ])
        for p in sorted(sigma_map.keys()):
            s = sigma_map[p]
            lines.append(f"| {p:.4f} | {s:.4f} | {math.log(s):.4f} |")

        lines.extend([
            "",
            "### Test on d=9",
            "",
            f"Direct MAE: `{ev['direct_mae']:.4f}`",
            f"Direct R2: `{ev['direct_r2']:.4f}`",
            f"Alpha: `{ev['fit_alpha']:.4f}` (ideal=1.0)",
            f"Beta:  `{ev['fit_beta']:.4f}` (ideal=0.0)",
            "",
        ])

        if ev["alpha_is_near_1"] and ev["beta_is_near_0"]:
            lines.append("**PASS**: Alpha near 1 AND beta near 0.")
        elif ev["alpha_is_near_1"]:
            lines.append(f"**PARTIAL**: Alpha near 1, beta={ev['fit_beta']:.4f}.")
        else:
            lines.append(f"**ISSUE**: Alpha={ev['fit_alpha']:.4f}, beta={ev['fit_beta']:.4f}.")

        lines.extend([
            "",
            "| p | R_actual | R_pred | error |",
            "|---:|---:|---:|---:|",
        ])
        for pt in sorted(ev["per_point"], key=lambda x: x["physical_error_rate"]):
            lines.append(f"| {pt['physical_error_rate']:.4f} | {pt['R_actual']:.4f} | {pt['R_predicted']:.4f} | {pt['abs_error']:.4f} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_sweep(args.source_dir)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    noise_model = payload["config"].get("noise_model", "unknown")

    rows = pool_bases(payload["conditions"])
    train_distances = {3, 5, 7}
    heldout = {9}

    grad_S_variants = {}
    for grad_S_key, grad_S_label in [("syndrome_density", "syndrome_density"), ("p", "p_raw")]:
        sigma_map = measure_sigma_per_p(rows, train_distances)
        E_val = calibrate_E(rows, train_distances, sigma_map, grad_S_key=grad_S_key)
        evaluation = evaluate_v5(rows, sigma_map, E_val, heldout, grad_S_key=grad_S_key)
        grad_S_variants[grad_S_label] = {
            "E": E_val,
            "sigma_map": sigma_map,
            "evaluation": evaluation,
        }

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "source_run": payload["run_id"],
        "noise_model": noise_model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_distances": sorted(train_distances),
        "heldout_distances": sorted(heldout),
        "pooled_bases": True,
        "grad_S_variants": grad_S_variants,
    }

    json_path = out_dir / "v5_analysis.json"
    report_path = out_dir / "V5_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)

    for key, res in grad_S_variants.items():
        ev = res["evaluation"]
        print(f"[{key}] E={res['E']:.4f}  MAE={ev['direct_mae']:.4f}  R2={ev['direct_r2']:.4f}  alpha={ev['fit_alpha']:.4f}  beta={ev['fit_beta']:.4f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
