"""v6: fractal scaling test — measure per-step sigma at each distance jump.

Hypothesis: if the formula describes fractal depth, then sigma should decay
systematically with distance — each additional unit of Df provides a different
per-step multiplier, following a predictable pattern.

For each (p, noise_model), pool X/Z, compute per-step sigma:
  sigma_{3->5} = exp((ln(R_d5) - ln(R_d3)) / 2)
  sigma_{5->7} = exp((ln(R_d7) - ln(R_d5)) / 2)
  sigma_{7->9} = exp((ln(R_d9) - ln(R_d7)) / 2)

Analysis:
  1. Does ln(sigma) decay with distance systematically?
  2. Is the decay pattern consistent across p?
  3. Cross-noise-model comparison
  4. Can sigma_{7->9} be predicted from sigma_{3->5}?
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


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_sweep(source_dir: str) -> dict[str, Any]:
    src = Path(source_dir)
    for pattern in ["qec_precision_sweep_v2.json", "qec_precision_sweep.json"]:
        candidates = list(src.glob(pattern))
        if candidates:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No sweep JSON found in {source_dir}")


def pool_bases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        key = (float(r["physical_error_rate"]), int(r["distance"]))
        groups[key].append(r)
    pooled = []
    for (p, d), group in groups.items():
        pooled.append({
            "physical_error_rate": p,
            "distance": d,
            "log_suppression": float(np.mean([r["log_suppression"] for r in group])),
            "syndrome_density": float(np.mean([r["syndrome_density"] for r in group])),
        })
    return pooled


def per_step_sigmas(pooled: list[dict[str, Any]]) -> dict[float, list[dict]]:
    """Compute sigma at each adjacent distance jump, per p."""
    by_p = defaultdict(list)
    for r in pooled:
        by_p[float(r["physical_error_rate"])].append((int(r["distance"]), r["log_suppression"]))

    results = {}
    for p, points in sorted(by_p.items()):
        points.sort()
        ds = [pt[0] for pt in points]
        log_Rs = [pt[1] for pt in points]

        steps = []
        for i in range(len(ds) - 1):
            d_lo = ds[i]
            d_hi = ds[i + 1]
            delta_d = d_hi - d_lo
            delta_logR = log_Rs[i + 1] - log_Rs[i]
            sigma_step = math.exp(delta_logR / delta_d)
            steps.append({
                "from_d": d_lo,
                "to_d": d_hi,
                "delta_d": delta_d,
                "delta_logR": round(delta_logR, 6),
                "sigma_step": round(sigma_step, 6),
                "ln_sigma_step": round(math.log(max(sigma_step, 1e-60)), 6),
            })
        results[p] = steps
    return results


def analyze_pattern(per_step: dict[float, list[dict]], noise_model: str) -> dict[str, Any]:
    """Analyze whether ln(sigma) decays systematically with distance."""
    midpoints = []
    ln_sigmas = []
    p_labels = []

    for p, steps in per_step.items():
        for step in steps[:3]:  # {3->5, 5->7, 7->9}
            mid_d = (step["from_d"] + step["to_d"]) / 2
            midpoints.append(mid_d)
            ln_sigmas.append(step["ln_sigma_step"])
            p_labels.append(p)

    if len(midpoints) < 3:
        return {"error": "Not enough data points"}

    midpoints_arr = np.array(midpoints)
    ln_sigmas_arr = np.array(ln_sigmas)

    # Fit: ln(sigma) = a + b * ln(d)  (power-law decay)
    log_mid = np.log(midpoints_arr)
    A = np.column_stack([log_mid, np.ones_like(log_mid)])
    coeffs_power, _, _, _ = np.linalg.lstsq(A, ln_sigmas_arr, rcond=None)
    power_slope = float(coeffs_power[0])
    power_intercept = float(coeffs_power[1])
    power_pred = power_slope * log_mid + power_intercept
    power_r2 = float(1 - np.sum((ln_sigmas_arr - power_pred)**2) / np.sum((ln_sigmas_arr - np.mean(ln_sigmas_arr))**2))

    # Fit: ln(sigma) = a + b * d  (linear decay)
    A_lin = np.column_stack([midpoints_arr, np.ones_like(midpoints_arr)])
    coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, ln_sigmas_arr, rcond=None)
    lin_slope = float(coeffs_lin[0])
    lin_intercept = float(coeffs_lin[1])
    lin_pred = lin_slope * midpoints_arr + lin_intercept
    lin_r2 = float(1 - np.sum((ln_sigmas_arr - lin_pred)**2) / np.sum((ln_sigmas_arr - np.mean(ln_sigmas_arr))**2))

    # Predict sigma_{7->9} from sigma_{3->5} using power-law
    #   ln(sigma_{7->9}) = power_slope * ln(8) + power_intercept
    #   sigma_{7->9} = exp(that)
    pred_ln_79 = power_slope * math.log(8.0) + power_intercept
    pred_sigma_79 = math.exp(pred_ln_79)

    # Get actual sigma_{7->9} for each p
    actual_79 = {}
    for p, steps in per_step.items():
        for step in steps:
            if step["from_d"] == 7 and step["to_d"] == 9:
                actual_79[p] = step["sigma_step"]

    # Compare prediction to actuals
    pred_vs_actual = []
    for p, actual in sorted(actual_79.items()):
        pred_vs_actual.append({
            "p": p,
            "sigma_79_actual": actual,
            "sigma_79_predicted": pred_sigma_79,
            "error": abs(actual - pred_sigma_79),
        })

    # Check sigma sign-flip: does sigma cross 1.0, and where?
    sigma_at_mid = {}
    for p, steps in per_step.items():
        for step in steps:
            mid_d = (step["from_d"] + step["to_d"]) / 2
            key = (p, mid_d)
            sigma_at_mid[key] = step["sigma_step"]

    crossings = []
    for p in sorted(per_step.keys()):
        vals = [(step["from_d"] + step["to_d"]) / 2 for step in per_step[p]]
        sigmas = [step["sigma_step"] for step in per_step[p]]
        for i in range(len(sigmas) - 1):
            if (sigmas[i] - 1.0) * (sigmas[i + 1] - 1.0) < 0:
                d1, d2 = vals[i], vals[i + 1]
                s1, s2 = sigmas[i], sigmas[i + 1]
                cross_d = d1 + (1.0 - s1) * (d2 - d1) / (s2 - s1)
                crossings.append({"p": p, "cross_distance": round(cross_d, 2)})

    return {
        "noise_model": noise_model,
        "num_p_values": len(per_step),
        "power_law_fit": {
            "slope": round(power_slope, 6),
            "intercept": round(power_intercept, 6),
            "r2": round(power_r2, 6),
            "form": f"ln(sigma) = {power_slope:.4f} * ln(d) + {power_intercept:.4f}",
        },
        "linear_fit": {
            "slope": round(lin_slope, 6),
            "intercept": round(lin_intercept, 6),
            "r2": round(lin_r2, 6),
        },
        "predicted_sigma_79": round(pred_sigma_79, 6),
        "pred_vs_actual_79": pred_vs_actual,
        "threshold_crossings": crossings,
        "sigma_consistency": {
            "mean_ln_sigma": round(float(np.mean(ln_sigmas_arr)), 6),
            "std_ln_sigma": round(float(np.std(ln_sigmas_arr)), 6),
            "cv": round(float(np.std(ln_sigmas_arr) / max(abs(np.mean(ln_sigmas_arr)), 1e-60)), 4),
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    nm = payload.get("noise_model", "unknown")
    ps = payload["per_step"]
    an = payload["analysis"]

    lines = [
        "# QEC Precision Sweep v6 -- Fractal Scaling Test",
        "",
        f"Run id: `{payload['run_id']}`",
        f"Noise model: `{nm}`",
        f"Source data: `{payload['source_run']}`",
        f"UTC time: `{payload['created_utc']}`",
        "",
        "## Design",
        "",
        "For each p, measure per-step sigma at each adjacent distance pair:",
        "```",
        "sigma_{d->d+2} = exp((ln(R_{d+2}) - ln(R_d)) / 2)",
        "```",
        "If the formula describes fractal depth, ln(sigma) should decay",
        "systematically with distance (power law or linear).",
        "",
        "## Per-Step Sigma Values",
        "",
        "| p | d_jump | delta_logR | sigma_step | ln(sigma) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(ps.keys()):
        for step in ps[p]:
            lines.append(
                f"| {p:.4f} | {step['from_d']}->{step['to_d']} | "
                f"{step['delta_logR']:+.4f} | {step['sigma_step']:.4f} | "
                f"{step['ln_sigma_step']:+.4f} |"
            )

    lines.extend([
        "",
        "## Fractal Pattern Analysis",
        "",
        f"### Power-law decay: ln(sigma) ∝ d^k",
        f"- Form: `{an['power_law_fit']['form']}`",
        f"- R2: `{an['power_law_fit']['r2']:.4f}`",
        f"- Slope (exponent): `{an['power_law_fit']['slope']:.4f}`",
        "",
        f"### Linear decay: ln(sigma) ∝ d",
        f"- Slope: `{an['linear_fit']['slope']:.4f}`",
        f"- R2: `{an['linear_fit']['r2']:.4f}`",
        "",
        f"### Sigma consistency across p values",
        f"- Mean ln(sigma): `{an['sigma_consistency']['mean_ln_sigma']:.4f}`",
        f"- Std ln(sigma): `{an['sigma_consistency']['std_ln_sigma']:.4f}`",
        f"- Coefficient of variation: `{an['sigma_consistency']['cv']}`",
        "",
    ])

    if an["power_law_fit"]["r2"] > 0.7:
        lines.append("**Strong fractal pattern: ln(sigma) follows power-law decay with distance.**")
    elif an["power_law_fit"]["r2"] > 0.4:
        lines.append("**Moderate fractal pattern: some structure in ln(sigma) vs distance.**")
    else:
        lines.append("**Weak or no fractal pattern: ln(sigma) vs distance is not systematic.**")

    lines.extend([
        "",
        "## Sigma Sign-Flip (Threshold Crossings)",
        "",
        "Sigma > 1 below threshold, sigma < 1 above. Where does it cross?",
        "",
    ])
    if an["threshold_crossings"]:
        lines.append("| p | crossing distance |")
        lines.append("|---:|---:|")
        for c in an["threshold_crossings"]:
            lines.append(f"| {c['p']:.4f} | d={c['cross_distance']} |")
    else:
        lines.append("No threshold crossings detected.")

    lines.extend([
        "",
        "## Predictive Test: sigma_{7->9} from sigma_{3->5}",
        "",
        "Using power-law fit across all p values, predict sigma at d=7->9:",
        f"- Predicted sigma_79: `{an['predicted_sigma_79']:.4f}`",
        "",
        "| p | actual sigma_79 | predicted | error |",
        "|---:|---:|---:|---:|",
    ])
    for row in an["pred_vs_actual_79"]:
        lines.append(
            f"| {row['p']:.4f} | {row['sigma_79_actual']:.4f} | "
            f"{row['sigma_79_predicted']:.4f} | {row['error']:.4f} |"
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
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    noise_model = payload["config"].get("noise_model", "unknown")

    pooled = pool_bases(payload["conditions"])
    per_step = per_step_sigmas(pooled)
    analysis = analyze_pattern(per_step, noise_model)

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "source_run": payload["run_id"],
        "noise_model": noise_model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "per_step": per_step,
        "analysis": analysis,
    }

    json_path = out_dir / "v6_analysis.json"
    report_path = out_dir / "V6_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)

    an = analysis
    pl = an["power_law_fit"]
    print(f"Power-law R2: {pl['r2']:.4f}  slope: {pl['slope']:.4f}")
    print(f"Linear R2: {an['linear_fit']['r2']:.4f}")
    print(f"Predicted sigma_79: {an['predicted_sigma_79']:.4f}")
    if an["threshold_crossings"]:
        print(f"Threshold crossings: {len(an['threshold_crossings'])}")
    else:
        print("No threshold crossings")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
