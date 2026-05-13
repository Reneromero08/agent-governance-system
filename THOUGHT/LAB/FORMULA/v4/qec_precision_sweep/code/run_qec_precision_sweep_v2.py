"""QEC precision sweep v2 runner -- multiple noise models, frozen p_th mapping.

Preregistration: PREREGISTRATION_v2.md
Differences from v1:
  - E = 1 - p (non-trivial, instead of 1.0)
  - grad_S = p / p_th (threshold-relative, instead of H2(p))
  - sigma = sqrt(p_th / p) (threshold-aware, instead of 1/sqrt(H2(p)))
  - Supports DEPOL, MEAS, and PHENOM noise models
  - Records v2 formula scores for post-hoc evaluation with frozen p_th
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pymatching
import scipy
import sinter
import sklearn
import stim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FROZEN_P_THRESHOLD = 0.007071067811865475

NoiseModel = Literal["depol", "meas", "phenom"]


@dataclass(frozen=True)
class Condition:
    basis: str
    distance: int
    rounds: int
    physical_error_rate: float
    shots: int
    seed: int
    noise_model: NoiseModel


def condition_seed(base_seed: int, basis: str, distance: int, p: float, noise_model: str) -> int:
    key = f"{base_seed}|{noise_model}|{basis}|{distance}|{p:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def make_circuit(condition: Condition) -> stim.Circuit:
    task = f"surface_code:rotated_memory_{condition.basis}"
    p = condition.physical_error_rate
    nm = condition.noise_model

    if nm == "depol":
        return stim.Circuit.generated(
            task,
            distance=condition.distance,
            rounds=condition.rounds,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
    elif nm == "meas":
        return stim.Circuit.generated(
            task,
            distance=condition.distance,
            rounds=condition.rounds,
            after_clifford_depolarization=p * 0.2,
            after_reset_flip_probability=p * 2.0,
            before_measure_flip_probability=p * 3.0,
            before_round_data_depolarization=p * 0.5,
        )
    elif nm == "phenom":
        return stim.Circuit.generated(
            task,
            distance=condition.distance,
            rounds=condition.rounds,
            after_clifford_depolarization=0.0,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
    else:
        raise ValueError(f"Unknown noise model: {nm}")


def simulate_condition(condition: Condition) -> dict[str, Any]:
    started = time.perf_counter()
    circuit = make_circuit(condition)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=condition.seed)
    detectors, observables = sampler.sample(
        condition.shots,
        separate_observables=True,
    )
    predictions = matching.decode_batch(detectors)
    logical_errors = np.any(predictions != observables, axis=1)
    logical_error_count = int(np.count_nonzero(logical_errors))
    logical_error_rate = logical_error_count / condition.shots

    p = condition.physical_error_rate
    p_th = FROZEN_P_THRESHOLD

    # v2 mapping
    E_val = 1.0 - p
    grad_S_val = p / p_th
    sigma_val = math.sqrt(p_th / p)
    Df_val = float(condition.distance)
    formula_score_v2 = (E_val / grad_S_val) * (sigma_val**Df_val)
    formula_log_score_v2 = math.log(max(formula_score_v2, 1e-60))

    log_E_over_grad_S = math.log(max(E_val / grad_S_val, 1e-60))
    Df_ln_sigma = Df_val * math.log(max(sigma_val, 1e-60))

    # also compute v1 formula score for comparability
    h = -(p * math.log2(max(p, 1e-12)) + (1 - p) * math.log2(max(1 - p, 1e-12)))
    sigma_v1 = 1.0 / math.sqrt(max(h, 1e-12))
    formula_score_v1 = (1.0 / h) * (sigma_v1**Df_val)
    formula_log_score_v1 = math.log(max(formula_score_v1, 1e-60))

    # Laplace smoothing
    p_logical_smooth = (logical_error_count + 0.5) / (condition.shots + 1.0)
    suppression = p / p_logical_smooth
    log_suppression = math.log(max(suppression, 1e-60))
    survival = 1.0 - logical_error_rate

    syndrome_density = float(np.mean(detectors))

    p_th_is_below = p < p_th

    return {
        **asdict(condition),
        "logical_error_count": logical_error_count,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_laplace": p_logical_smooth,
        "survival": survival,
        "suppression_R": suppression,
        "log_suppression": log_suppression,
        "E_v2": E_val,
        "grad_S_v2": grad_S_val,
        "sigma_v2": sigma_val,
        "Df": Df_val,
        "formula_score_v2": formula_score_v2,
        "formula_log_score_v2": formula_log_score_v2,
        "log_E_over_grad_S": log_E_over_grad_S,
        "Df_ln_sigma": Df_ln_sigma,
        "p_threshold": p_th,
        "below_threshold": p_th_is_below,
        "E_v1": 1.0,
        "grad_S_v1": h,
        "sigma_v1": sigma_v1,
        "formula_score_v1": formula_score_v1,
        "formula_log_score_v1": formula_log_score_v1,
        "syndrome_density": syndrome_density,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "duration_seconds": time.perf_counter() - started,
    }


def feature_matrix(rows: list[dict[str, Any]], feature_set: str) -> np.ndarray:
    p = np.array([r["physical_error_rate"] for r in rows], dtype=float)
    d = np.array([r["distance"] for r in rows], dtype=float)
    basis_z = np.array([1.0 if r["basis"] == "z" else 0.0 for r in rows], dtype=float)

    if feature_set == "formula_score_v2":
        return np.column_stack([np.array([r["formula_log_score_v2"] for r in rows], dtype=float)])
    if feature_set == "formula_components_v2":
        x1 = np.array([r["log_E_over_grad_S"] for r in rows], dtype=float)
        x2 = np.array([r["Df_ln_sigma"] for r in rows], dtype=float)
        return np.column_stack([x1, x2, basis_z])
    if feature_set == "formula_score_v1":
        return np.column_stack([np.array([r["formula_log_score_v1"] for r in rows], dtype=float)])
    if feature_set == "formula_components_v1":
        h = np.array([r["grad_S_v1"] for r in rows], dtype=float)
        s = np.array([r["sigma_v1"] for r in rows], dtype=float)
        return np.column_stack([-np.log(np.maximum(h, 1e-60)), d * np.log(np.maximum(s, 1e-60)), basis_z])
    if feature_set == "p_only":
        return np.column_stack([np.log(np.maximum(p, 1e-60)), basis_z])
    if feature_set == "distance_only":
        return np.column_stack([d, basis_z])
    if feature_set == "standard_qec_scaling":
        return np.column_stack([np.log(np.maximum(p, 1e-60)), d, d * np.log(np.maximum(p, 1e-60)), basis_z])
    raise ValueError(feature_set)


def evaluate_models(rows: list[dict[str, Any]], heldout_distances: set[int]) -> dict[str, Any]:
    train = [r for r in rows if int(r["distance"]) not in heldout_distances]
    test = [r for r in rows if int(r["distance"]) in heldout_distances]
    y_train = np.array([r["log_suppression"] for r in train], dtype=float)
    y_test = np.array([r["log_suppression"] for r in test], dtype=float)

    out: dict[str, Any] = {
        "target": "log_suppression = ln(physical_error_rate / logical_error_rate_laplace)",
        "heldout_distances": sorted(heldout_distances),
        "train_conditions": len(train),
        "test_conditions": len(test),
        "models": {},
    }
    model_names = [
        "formula_score_v2",
        "formula_components_v2",
        "formula_score_v1",
        "formula_components_v1",
        "standard_qec_scaling",
        "p_only",
        "distance_only",
    ]
    for name in model_names:
        try:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(feature_matrix(train, name), y_train)
            pred_train = model.predict(feature_matrix(train, name))
            pred_test = model.predict(feature_matrix(test, name))
            out["models"][name] = {
                "train_mae": float(mean_absolute_error(y_train, pred_train)),
                "test_mae": float(mean_absolute_error(y_test, pred_test)),
                "train_r2": float(r2_score(y_train, pred_train)),
                "test_r2": float(r2_score(y_test, pred_test)),
                "test_predictions": [
                    {
                        "basis": r["basis"],
                        "distance": r["distance"],
                        "physical_error_rate": r["physical_error_rate"],
                        "actual_log_suppression": float(y),
                        "predicted_log_suppression": float(yhat),
                        "absolute_error": float(abs(y - yhat)),
                    }
                    for r, y, yhat in zip(test, y_test, pred_test)
                ],
            }
        except Exception as exc:
            out["models"][name] = {
                "train_mae": None,
                "test_mae": None,
                "train_r2": None,
                "test_r2": None,
                "error": str(exc),
            }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "stim": stim.__version__,
        "pymatching": pymatching.__version__,
        "sinter": sinter.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[5],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_report(path: Path, payload: dict[str, Any]) -> None:
    evals = payload["evaluation"]["models"]
    valid = {k: v for k, v in evals.items() if v.get("test_mae") is not None}
    if not valid:
        best_name = "NONE"
    else:
        best_name = sorted(valid.items(), key=lambda kv: kv[1]["test_mae"])[0][0]
    nm = payload["config"]["noise_model"]
    lines = [
        "# QEC Precision Sweep v2 -- Raw Sweep Report",
        "",
        f"Run id: `{payload['run_id']}`",
        f"Noise model: `{nm}`",
        f"UTC time: `{payload['created_utc']}`",
        f"Git revision: `{payload['git_revision']}`",
        "",
        "## Preregistered Mapping (v2, frozen p_th)",
        "",
        f"- `E`: `1 - p` (initial single-qubit survival probability)",
        f"- `grad_S`: `p / p_th` with `p_th = {FROZEN_P_THRESHOLD}`",
        f"- `sigma`: `sqrt(p_th / p)`",
        f"- `Df`: surface-code distance `d`",
        f"- `R`: `ln(physical_error_rate / logical_error_rate_laplace)`",
        "",
        "Formula score:",
        "",
        "```text",
        "R_hat = (E / grad_S) * sigma ** Df",
        "```",
        "",
        "## Sweep",
        "",
        f"- Conditions: `{len(payload['conditions'])}`",
        f"- Total shots: `{sum(r['shots'] for r in payload['conditions'])}`",
        f"- Distances: `{payload['config']['distances']}`",
        f"- Held-out distances: `{payload['config']['heldout_distances']}`",
        f"- Physical error rates: `{payload['config']['physical_error_rates']}`",
        f"- Bases: `{payload['config']['bases']}`",
        "",
        "## Preliminary Model Comparison (built-in evaluation)",
        "",
        "| Model | Train MAE | Test MAE | Train R2 | Test R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(valid.items(), key=lambda kv: kv[1]["test_mae"]):
        lines.append(
            f"| `{name}` | {metrics['train_mae']:.4f} | {metrics['test_mae']:.4f} | "
            f"{metrics['train_r2']:.4f} | {metrics['test_r2']:.4f} |"
        )
    for name, metrics in evals.items():
        if metrics.get("test_mae") is None:
            lines.append(f"| `{name}` | ERROR | ERROR | ERROR | ERROR | ({metrics.get('error', 'unknown')})")
    lines.extend(
        [
            "",
            "## Preliminary Verdict (built-in evaluation only)",
            "",
            f"Best held-out model by MAE: `{best_name}`.",
            "",
            "Note: The authoritative evaluation uses the separate evaluator script",
            "(`evaluate_qec_v2.py`) which applies the full pass/fail criteria from",
            "PREREGISTRATION_v2.md, including bootstrap confidence intervals.",
            "",
            "## Files",
            "",
            f"- Raw condition CSV: `{payload['files']['csv']}`",
            f"- Full JSON payload: `{payload['files']['json']}`",
            f"- This report: `{path.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--distances", default="3,5,7,9")
    parser.add_argument("--heldout-distances", default="7,9")
    parser.add_argument("--physical-error-rates", default="0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04")
    parser.add_argument("--bases", default="x,z")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--noise-model", default="depol", choices=["depol", "meas", "phenom"])
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    distances = [int(x) for x in args.distances.split(",")]
    heldout_distances = {int(x) for x in args.heldout_distances.split(",")}
    ps = [float(x) for x in args.physical_error_rates.split(",")]
    bases = [x.strip().lower() for x in args.bases.split(",")]
    noise_model: NoiseModel = args.noise_model
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        Condition(
            basis=basis,
            distance=d,
            rounds=d,
            physical_error_rate=p,
            shots=args.shots,
            seed=condition_seed(args.seed, basis, d, p, noise_model),
            noise_model=noise_model,
        )
        for basis in bases
        for d in distances
        for p in ps
    ]

    rows = []
    for index, condition in enumerate(conditions, start=1):
        print(
            f"[{index}/{len(conditions)}] noise={condition.noise_model} "
            f"basis={condition.basis} d={condition.distance} "
            f"p={condition.physical_error_rate} shots={condition.shots}",
            flush=True,
        )
        rows.append(simulate_condition(condition))

    evaluation = evaluate_models(rows, heldout_distances)
    csv_path = out_dir / "conditions.csv"
    json_path = out_dir / "qec_precision_sweep_v2.json"
    report_path = out_dir / "REPORT.md"
    write_csv(csv_path, rows)

    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "command": " ".join(sys.argv),
        "config": {
            "shots": args.shots,
            "distances": distances,
            "heldout_distances": sorted(heldout_distances),
            "physical_error_rates": ps,
            "bases": bases,
            "seed": args.seed,
            "noise_model": noise_model,
        },
        "versions": package_versions(),
        "preregistration": "PREREGISTRATION_v2.md",
        "mapping": {
            "R": "ln(physical_error_rate / logical_error_rate_laplace)",
            "E": "1.0 - p",
            "grad_S": "p / p_threshold",
            "sigma": "sqrt(p_threshold / p)",
            "Df": "surface-code distance",
            "p_threshold": FROZEN_P_THRESHOLD,
        },
        "conditions": rows,
        "evaluation": evaluation,
        "files": {
            "csv": str(csv_path.relative_to(ROOT)),
            "json": str(json_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(report_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
