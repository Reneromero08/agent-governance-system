"""v7: information-theoretic sigma -- I(S:F) from syndrome-logical mutual information.

Light Cone canonical QEC mapping:
  sigma = code compression ratio = logical information per resource
        = I(S:F) = mutual information between logical state and syndrome fragment

Operationalization:
  I(S:F) = H2(p) - H2(p_L)  where:
    H2(p)   = binary entropy of the physical error rate (prior uncertainty)
    H2(p_L) = binary entropy of the logical error rate after decoding (residual)
    F      = the full syndrome from d rounds

Per-round sigma: sigma = I(S:F) / d

Then: E=1.0, grad_S=p, Df=d, R_pred = (E/grad_S) * sigma^Df

Comparison: log(R_pred) vs log_suppression on held-out distances.
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


def H2(x: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def load_sweep(source_dir: str) -> dict[str, Any]:
    src = Path(source_dir)
    for pattern in ["qec_precision_sweep_v2.json", "qec_precision_sweep.json"]:
        candidates = list(src.glob(pattern))
        if candidates:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No sweep JSON found in {source_dir}")


def compute_v7_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    eps = 1e-60
    for r in rows:
        p = r["physical_error_rate"]
        d = int(r["distance"])
        p_L = r["logical_error_rate_laplace"]

        I_SF = max(0.0, H2(p) - H2(p_L))
        sigma_v7 = I_SF / d

        E_val = 1.0
        grad_S_val = max(p, eps)
        Df_val = float(d)

        R_pred = (E_val / grad_S_val) * (sigma_v7 ** Df_val)
        log_R_pred = math.log(max(R_pred, eps))

        results.append({
            **r,
            "v7_H2_p": H2(p),
            "v7_H2_pL": H2(p_L),
            "v7_I_SF": I_SF,
            "v7_sigma": sigma_v7,
            "v7_grad_S": grad_S_val,
            "v7_R_predicted": R_pred,
            "v7_log_R_predicted": log_R_pred,
        })
    return results


def evaluate(rows: list[dict[str, Any]], heldout: set[int]) -> dict[str, Any]:
    test = [r for r in rows if int(r["distance"]) in heldout]
    y_pred = np.array([r["v7_log_R_predicted"] for r in test], dtype=float)
    y_actual = np.array([r["log_suppression"] for r in test], dtype=float)

    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))

    A = np.column_stack([y_pred, np.ones_like(y_pred)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y_actual, rcond=None)
    alpha = float(coeffs[0])
    beta = float(coeffs[1])

    return {
        "direct_mae": mae,
        "direct_r2": r2,
        "fit_alpha": alpha,
        "fit_beta": beta,
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
                "sigma_v7": r["v7_sigma"],
                "I_SF": r["v7_I_SF"],
            }
            for r, ya, yp in zip(test, y_actual, y_pred)
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    ev = payload["evaluation"]
    nm = payload.get("noise_model", "unknown")
    sigmas = payload["sigma_summary"]

    lines = [
        "# QEC Precision Sweep v7 -- Information-Theoretic Sigma",
        "",
        f"Run id: `{payload['run_id']}`",
        f"Noise model: `{nm}`",
        f"Source data: `{payload['source_run']}`",
        f"UTC time: `{payload['created_utc']}`",
        "",
        "## Design",
        "",
        "Light Cone canonical QEC mapping: sigma = code compression/fidelity factor.",
        "",
        "Operationalized as: `sigma = I(S:F) / d` where:",
        "- `I(S:F) = H2(p) - H2(p_L)` — mutual information between logical state and syndrome",
        "- `H2(p)` = binary entropy of physical error rate (prior uncertainty)",
        "- `H2(p_L)` = binary entropy of logical error rate after decoding (residual uncertainty)",
        "- `d` = code distance / rounds (per-round normalization)",
        "",
        "Then: `E = 1.0`, `grad_S = p`, `Df = d`",
        "",
        "## Sigma per (p, distance) — I(S:F) / d",
        "",
        "| p | d | I(S:F) | sigma | sigma > 1? |",
        "|---:|---:|---:|---:|---|",
    ]
    for s in sigmas:
        lines.append(
            f"| {s['p']:.4f} | {s['d']} | {s['I_SF']:.4f} | "
            f"{s['sigma']:.4f} | {s['sigma'] > 1.0} |"
        )

    lines.extend([
        "",
        "## Evaluation on Held-Out Distances",
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
        lines.append(f"**PARTIAL**: Alpha near 1 (structure correct), beta={ev['fit_beta']:.4f} (offset).")
    else:
        lines.append(f"**ISSUE**: Alpha={ev['fit_alpha']:.4f}, beta={ev['fit_beta']:.4f}.")

    lines.extend([
        "",
        "| basis | d | p | R_actual | R_pred | error | sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pt in sorted(ev["per_point"], key=lambda x: (x["physical_error_rate"], x["distance"], x["basis"])):
        lines.append(
            f"| {pt['basis']} | {pt['distance']} | {pt['physical_error_rate']:.4f} | "
            f"{pt['R_actual']:.4f} | {pt['R_predicted']:.4f} | {pt['abs_error']:.4f} | "
            f"{pt['sigma_v7']:.4f} |"
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
    heldout = set(payload["config"]["heldout_distances"])

    rows = compute_v7_predictions(payload["conditions"])
    ev = evaluate(rows, heldout)

    # summary stats per (p,d) for report table
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(float(r["physical_error_rate"]), int(r["distance"]))].append(r["v7_sigma"])
    sigma_summary = []
    for (p, d), sigs in sorted(groups.items()):
        avg_I_SF = float(np.mean([r["v7_I_SF"] for r in rows if float(r["physical_error_rate"]) == p and int(r["distance"]) == d]))
        sigma_summary.append({
            "p": p, "d": d,
            "I_SF": round(avg_I_SF, 6),
            "sigma": round(float(np.mean(sigs)), 6),
        })

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "source_run": payload["run_id"],
        "noise_model": noise_model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "sigma": "I(S:F)/d = (H2(p) - H2(p_L))/d",
            "E": 1.0,
            "grad_S": "p (physical error rate)",
            "Df": "d (code distance)",
        },
        "sigma_summary": sigma_summary,
        "evaluation": ev,
    }

    json_path = out_dir / "v7_analysis.json"
    report_path = out_dir / "V7_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)

    print(f"MAE: {ev['direct_mae']:.4f}  R2: {ev['direct_r2']:.4f}")
    print(f"Alpha: {ev['fit_alpha']:.4f}  Beta: {ev['fit_beta']:.4f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
