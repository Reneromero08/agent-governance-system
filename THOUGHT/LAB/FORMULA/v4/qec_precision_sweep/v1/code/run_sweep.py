"""Run a recorded QEC precision sweep for Formula lab v4.

This experiment tests whether a preregistered domain mapping of

    R = (E / grad_S) * sigma ** Df

predicts logical-error suppression in surface-code simulations.
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
from typing import Any

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


@dataclass(frozen=True)
class Condition:
    basis: str
    distance: int
    rounds: int
    physical_error_rate: float
    shots: int
    seed: int


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def condition_seed(base_seed: int, basis: str, distance: int, p: float) -> int:
    key = f"{base_seed}|{basis}|{distance}|{p:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def make_circuit(condition: Condition) -> stim.Circuit:
    task = f"surface_code:rotated_memory_{condition.basis}"
    p = condition.physical_error_rate
    return stim.Circuit.generated(
        task,
        distance=condition.distance,
        rounds=condition.rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


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
    h = binary_entropy(p)
    sigma = 1.0 / math.sqrt(h)
    df = float(condition.distance)
    e = 1.0
    formula_score = (e / h) * (sigma**df)
    formula_log_score = math.log(formula_score)

    # Laplace smoothing prevents infinite suppression at low p with finite shots.
    p_logical_smooth = (logical_error_count + 0.5) / (condition.shots + 1.0)
    suppression = p / p_logical_smooth
    log_suppression = math.log(suppression)
    survival = 1.0 - logical_error_rate

    syndrome_density = float(np.mean(detectors))

    return {
        **asdict(condition),
        "logical_error_count": logical_error_count,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_laplace": p_logical_smooth,
        "survival": survival,
        "suppression_R": suppression,
        "log_suppression": log_suppression,
        "E": e,
        "grad_S": h,
        "sigma": sigma,
        "Df": df,
        "formula_score": formula_score,
        "formula_log_score": formula_log_score,
        "syndrome_density": syndrome_density,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "duration_seconds": time.perf_counter() - started,
    }


def feature_matrix(rows: list[dict[str, Any]], feature_set: str) -> np.ndarray:
    p = np.array([r["physical_error_rate"] for r in rows], dtype=float)
    d = np.array([r["distance"] for r in rows], dtype=float)
    h = np.array([r["grad_S"] for r in rows], dtype=float)
    sigma = np.array([r["sigma"] for r in rows], dtype=float)
    basis_z = np.array([1.0 if r["basis"] == "z" else 0.0 for r in rows], dtype=float)

    if feature_set == "formula_score":
        return np.column_stack([np.log(np.array([r["formula_score"] for r in rows]))])
    if feature_set == "formula_components":
        return np.column_stack([-np.log(h), d * np.log(sigma), basis_z])
    if feature_set == "p_only":
        return np.column_stack([np.log(p), basis_z])
    if feature_set == "distance_only":
        return np.column_stack([d, basis_z])
    if feature_set == "standard_qec_scaling":
        return np.column_stack([np.log(p), d, d * np.log(p), basis_z])
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
    for name in [
        "formula_score",
        "formula_components",
        "p_only",
        "distance_only",
        "standard_qec_scaling",
    ]:
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
    best = sorted(evals.items(), key=lambda kv: kv[1]["test_mae"])[0]
    lines = [
        "# QEC Precision Sweep Report",
        "",
        f"Run id: `{payload['run_id']}`",
        f"UTC time: `{payload['created_utc']}`",
        f"Git revision: `{payload['git_revision']}`",
        "",
        "## Preregistered Mapping",
        "",
        "- `R`: logical-error suppression, `physical_error_rate / logical_error_rate_laplace`.",
        "- `E`: normalized initial logical-state integrity, fixed at `1.0` for this first sweep.",
        "- `grad_S`: binary entropy of the physical error rate, `H2(p)`.",
        "- `sigma`: entropy-to-correction efficiency proxy, `1 / sqrt(H2(p))`.",
        "- `Df`: surface-code distance `d`.",
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
        "## Model Comparison",
        "",
        "| Model | Train MAE | Test MAE | Train R2 | Test R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(evals.items(), key=lambda kv: kv[1]["test_mae"]):
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
    if best[0] in {"formula_score", "formula_components"}:
        lines.append(
            "The preregistered formula mapping was competitive on held-out distances in this run."
        )
    else:
        lines.append(
            "The preregistered formula mapping did not beat the strongest held-out baseline in this run."
        )
    lines.extend(
        [
            "",
            "This is evidence about the current QEC domain mapping, not proof of the whole formula.",
            "A stronger result would require larger sweeps, more noise models, and a frozen mapping carried across them.",
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
    parser.add_argument("--physical-error-rates", default="0.001,0.002,0.005,0.01,0.02,0.04")
    parser.add_argument("--bases", default="x,z")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    distances = [int(x) for x in args.distances.split(",")]
    heldout_distances = {int(x) for x in args.heldout_distances.split(",")}
    ps = [float(x) for x in args.physical_error_rates.split(",")]
    bases = [x.strip().lower() for x in args.bases.split(",")]
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
            seed=condition_seed(args.seed, basis, d, p),
        )
        for basis in bases
        for d in distances
        for p in ps
    ]

    rows = []
    for index, condition in enumerate(conditions, start=1):
        print(
            f"[{index}/{len(conditions)}] basis={condition.basis} "
            f"d={condition.distance} p={condition.physical_error_rate} shots={condition.shots}",
            flush=True,
        )
        rows.append(simulate_condition(condition))

    evaluation = evaluate_models(rows, heldout_distances)
    csv_path = out_dir / "conditions.csv"
    json_path = out_dir / "qec_precision_sweep.json"
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
        },
        "versions": package_versions(),
        "mapping": {
            "R": "physical_error_rate / logical_error_rate_laplace",
            "E": "1.0 normalized initial logical-state integrity",
            "grad_S": "H2(physical_error_rate)",
            "sigma": "1 / sqrt(H2(physical_error_rate))",
            "Df": "surface-code distance",
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
