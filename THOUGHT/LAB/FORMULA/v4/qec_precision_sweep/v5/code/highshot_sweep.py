"""High-shot target sweep for v5 evaluation with clean sigma estimates.

Targets only the 4 lowest physical error rates (0.0005-0.004) where 20k-shot
sigma estimates were noisiest. runs at 100k shots per condition.

Distances: {3,5,7,9}. Bases: x,z poolable. Noise models: depol, meas.
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

NoiseModel = Literal["depol", "meas"]


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
    key = f"highshot|{base_seed}|{noise_model}|{basis}|{distance}|{p:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def make_circuit(condition: Condition) -> stim.Circuit:
    task = f"surface_code:rotated_memory_{condition.basis}"
    p = condition.physical_error_rate
    nm = condition.noise_model
    if nm == "depol":
        return stim.Circuit.generated(
            task, distance=condition.distance, rounds=condition.rounds,
            after_clifford_depolarization=p, after_reset_flip_probability=p,
            before_measure_flip_probability=p, before_round_data_depolarization=p,
        )
    elif nm == "meas":
        return stim.Circuit.generated(
            task, distance=condition.distance, rounds=condition.rounds,
            after_clifford_depolarization=p * 0.2, after_reset_flip_probability=p * 2.0,
            before_measure_flip_probability=p * 3.0, before_round_data_depolarization=p * 0.5,
        )
    else:
        raise ValueError(f"Unknown: {nm}")


def simulate_condition(condition: Condition) -> dict[str, Any]:
    started = time.perf_counter()
    circuit = make_circuit(condition)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=condition.seed)
    detectors, observables = sampler.sample(condition.shots, separate_observables=True)
    predictions = matching.decode_batch(detectors)
    logical_errors = np.any(predictions != observables, axis=1)
    logical_error_count = int(np.count_nonzero(logical_errors))
    logical_error_rate = logical_error_count / condition.shots
    p_L_laplace = (logical_error_count + 0.5) / (condition.shots + 1.0)

    p = condition.physical_error_rate
    suppression = p / p_L_laplace
    log_suppression = math.log(max(suppression, 1e-60))

    return {
        **asdict(condition),
        "logical_error_count": logical_error_count,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_laplace": p_L_laplace,
        "log_suppression": log_suppression,
        "syndrome_density": float(np.mean(detectors)),
        "duration_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--noise-model", default="depol", choices=["depol", "meas"])
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ps = [0.0005, 0.001, 0.002, 0.004]
    distances = [3, 5, 7, 9]
    bases = ["x", "z"]
    nm: NoiseModel = args.noise_model
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        Condition(basis=b, distance=d, rounds=d, physical_error_rate=p,
                  shots=args.shots, seed=condition_seed(args.seed, b, d, p, nm),
                  noise_model=nm)
        for b in bases for d in distances for p in ps
    ]

    rows = []
    for i, c in enumerate(conditions, 1):
        print(f"[{i}/{len(conditions)}] {nm} {c.basis} d={c.distance} p={c.physical_error_rate}")
        rows.append(simulate_condition(c))

    csv_path = out_dir / "highshot_conditions.csv"
    json_path = out_dir / "highshot_sweep.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    payload = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"shots": args.shots, "distances": distances, "bases": bases,
                   "physical_error_rates": ps, "seed": args.seed, "noise_model": nm},
        "conditions": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
