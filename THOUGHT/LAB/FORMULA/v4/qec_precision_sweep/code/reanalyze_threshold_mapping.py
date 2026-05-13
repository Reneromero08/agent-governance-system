"""Reanalyze QEC sweep results with a threshold-relative Formula mapping.

The first QEC mapping used sigma = 1/sqrt(H2(p)), which cannot represent the
central QEC fact that increasing code distance helps below threshold and hurts
above threshold. This script estimates a threshold from training distances only
and retests the Formula on held-out distances without rerunning simulations.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_rows(rows: list[dict[str, Any]], heldout_distances: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if int(r["distance"]) not in heldout_distances]
    test = [r for r in rows if int(r["distance"]) in heldout_distances]
    return train, test


def estimate_threshold_from_training(train: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate p_th using only training distances.

    A physical error rate is "below threshold" if the largest training distance
    has better log_suppression than the smallest training distance. The threshold
    is the geometric midpoint between the largest below-threshold p and the
    smallest above-threshold p.
    """
    distances = sorted({int(r["distance"]) for r in train})
    low_d = distances[0]
    high_d = distances[-1]
    ps = sorted({float(r["physical_error_rate"]) for r in train})
    votes = []
    for p in ps:
        by_basis = []
        for basis in sorted({r["basis"] for r in train}):
            low = [r for r in train if r["basis"] == basis and int(r["distance"]) == low_d and float(r["physical_error_rate"]) == p][0]
            high = [r for r in train if r["basis"] == basis and int(r["distance"]) == high_d and float(r["physical_error_rate"]) == p][0]
            by_basis.append(high["log_suppression"] - low["log_suppression"])
        mean_delta = float(np.mean(by_basis))
        votes.append({"p": p, "mean_delta_high_minus_low": mean_delta, "distance_helped": mean_delta > 0})

    below = [v["p"] for v in votes if v["distance_helped"]]
    above = [v["p"] for v in votes if not v["distance_helped"]]
    if below and above:
        p_low = max(p for p in below if p < min(above))
        p_high = min(p for p in above if p > p_low)
        threshold = math.sqrt(p_low * p_high)
        method = "geometric midpoint between last improving p and first degrading p"
    else:
        threshold = float(np.median(ps))
        method = "fallback median physical error rate; training did not bracket threshold"
    return {"threshold": threshold, "method": method, "votes": votes, "low_distance": low_d, "high_distance": high_d}


def feature_matrix(rows: list[dict[str, Any]], feature_set: str, threshold: float) -> np.ndarray:
    p = np.array([r["physical_error_rate"] for r in rows], dtype=float)
    d = np.array([r["distance"] for r in rows], dtype=float)
    basis_z = np.array([1.0 if r["basis"] == "z" else 0.0 for r in rows], dtype=float)
    pressure = np.maximum(p / threshold, 1e-12)
    sigma = np.sqrt(threshold / p)
    log_formula = -np.log(pressure) + d * np.log(sigma)

    if feature_set == "threshold_formula_score":
        return np.column_stack([log_formula])
    if feature_set == "threshold_formula_components":
        return np.column_stack([-np.log(pressure), d * np.log(sigma), basis_z])
    if feature_set == "original_formula_score":
        return np.column_stack([np.array([r["formula_log_score"] for r in rows], dtype=float)])
    if feature_set == "original_formula_components":
        h = np.array([r["grad_S"] for r in rows], dtype=float)
        sigma_h = np.array([r["sigma"] for r in rows], dtype=float)
        return np.column_stack([-np.log(h), d * np.log(sigma_h), basis_z])
    if feature_set == "p_only":
        return np.column_stack([np.log(p), basis_z])
    if feature_set == "distance_only":
        return np.column_stack([d, basis_z])
    if feature_set == "standard_qec_scaling":
        return np.column_stack([np.log(p), d, d * np.log(p), basis_z])
    raise ValueError(feature_set)


def evaluate(rows: list[dict[str, Any]], heldout_distances: set[int], threshold: float) -> dict[str, Any]:
    train, test = split_rows(rows, heldout_distances)
    y_train = np.array([r["log_suppression"] for r in train], dtype=float)
    y_test = np.array([r["log_suppression"] for r in test], dtype=float)
    out = {
        "target": "log_suppression = ln(physical_error_rate / logical_error_rate_laplace)",
        "heldout_distances": sorted(heldout_distances),
        "train_conditions": len(train),
        "test_conditions": len(test),
        "models": {},
    }
    for name in [
        "threshold_formula_score",
        "threshold_formula_components",
        "original_formula_score",
        "original_formula_components",
        "standard_qec_scaling",
        "p_only",
        "distance_only",
    ]:
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(feature_matrix(train, name, threshold), y_train)
        pred_train = model.predict(feature_matrix(train, name, threshold))
        pred_test = model.predict(feature_matrix(test, name, threshold))
        out["models"][name] = {
            "train_mae": float(mean_absolute_error(y_train, pred_train)),
            "test_mae": float(mean_absolute_error(y_test, pred_test)),
            "train_r2": float(r2_score(y_train, pred_train)),
            "test_r2": float(r2_score(y_test, pred_test)),
        }
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    models = payload["evaluation"]["models"]
    best = sorted(models.items(), key=lambda kv: kv[1]["test_mae"])[0]
    lines = [
        "# QEC Threshold Mapping Reanalysis",
        "",
        f"Run id: `{payload['run_id']}`",
        f"Source run: `{payload['source_run']}`",
        f"UTC time: `{payload['created_utc']}`",
        "",
        "## Logic Fix",
        "",
        "The first mapping used `sigma = 1 / sqrt(H2(p))`, which always stays above 1 for p < 0.5.",
        "That means it always predicts larger code distance improves retention.",
        "QEC has a threshold: distance helps below threshold and hurts above threshold.",
        "",
    ]
    if payload["threshold"]["method"].startswith("fixed"):
        threshold_line = "This reanalysis uses a fixed preregistered threshold and maps:"
    else:
        threshold_line = "This reanalysis uses a train-only threshold estimate and maps:"
    lines.extend(
        [
            threshold_line,
            "",
            "```text",
            "grad_S = p / p_threshold",
            "sigma = sqrt(p_threshold / p)",
            "Df = surface-code distance",
            "R = physical_error_rate / logical_error_rate_laplace",
            "```",
            "",
            "## Threshold Estimate",
            "",
            f"- Estimated threshold: `{payload['threshold']['threshold']:.8f}`",
            f"- Method: {payload['threshold']['method']}",
            f"- Training distances used: `{payload['threshold']['low_distance']}` and `{payload['threshold']['high_distance']}`",
            "",
            "| p | mean delta high-low | distance helped |",
            "|---:|---:|---|",
        ]
    )
    for vote in payload["threshold"]["votes"]:
        lines.append(
            f"| {vote['p']:.4f} | {vote['mean_delta_high_minus_low']:.4f} | {vote['distance_helped']} |"
        )
    lines.extend(
        [
            "",
            "## Held-Out Model Comparison",
            "",
            "| Model | Train MAE | Test MAE | Train R2 | Test R2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in sorted(models.items(), key=lambda kv: kv[1]["test_mae"]):
        lines.append(
            f"| `{name}` | {metrics['train_mae']:.4f} | {metrics['test_mae']:.4f} | "
            f"{metrics['train_r2']:.4f} | {metrics['test_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"Best held-out model by MAE: `{best[0]}`.",
            "",
        ]
    )
    if best[0].startswith("threshold_formula"):
        lines.append("The corrected threshold-relative Formula mapping won this held-out comparison.")
    else:
        lines.append("The corrected threshold-relative Formula mapping did not win this held-out comparison.")
    lines.append("")
    lines.append("This reanalysis reuses the recorded simulation data; it does not change the raw QEC results.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--run-id", default="threshold_reanalysis_v1")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use a preregistered threshold instead of estimating one from this run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_json)
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    rows = source_payload["conditions"]
    heldout_distances = set(source_payload["config"]["heldout_distances"])
    train, _ = split_rows(rows, heldout_distances)
    if args.fixed_threshold is None:
        threshold = estimate_threshold_from_training(train)
    else:
        threshold = {
            "threshold": args.fixed_threshold,
            "method": "fixed preregistered threshold supplied by --fixed-threshold",
            "votes": estimate_threshold_from_training(train)["votes"],
            "low_distance": min({int(r["distance"]) for r in train}),
            "high_distance": max({int(r["distance"]) for r in train}),
        }
    evaluation = evaluate(rows, heldout_distances, threshold["threshold"])
    out_dir = source.parent.parent / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "source_run": source_payload["run_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(source),
        "threshold": threshold,
        "mapping": {
            "R": "physical_error_rate / logical_error_rate_laplace",
            "E": "1.0 normalized initial logical-state integrity",
            "grad_S": "physical_error_rate / p_threshold",
            "sigma": "sqrt(p_threshold / physical_error_rate)",
            "Df": "surface-code distance",
            "p_threshold": threshold["threshold"],
        },
        "evaluation": evaluation,
    }
    (out_dir / "threshold_reanalysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir / "REPORT.md", payload)
    print(json.dumps(payload["threshold"], indent=2))
    print(json.dumps(payload["evaluation"]["models"], indent=2))
    print(f"Wrote {out_dir / 'threshold_reanalysis.json'}")
    print(f"Wrote {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
