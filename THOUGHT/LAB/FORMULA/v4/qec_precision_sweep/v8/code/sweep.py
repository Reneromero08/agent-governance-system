"""v8: clean sweep with high shots, full p-grid, and distance {3,5,7,9,11}.

100k shots across all conditions. Both DEPOL and MEAS noise models.
X and Z bases (pooled in analysis). Designed around all findings from v1-v7.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np, pymatching, scipy, sinter, sklearn, stim

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

NoiseModel = Literal["depol", "meas"]

@dataclass(frozen=True)
class Condition:
    basis: str; distance: int; rounds: int
    physical_error_rate: float; shots: int; seed: int
    noise_model: NoiseModel

def condition_seed(base: int, basis: str, d: int, p: float, nm: str) -> int:
    key = f"v8|{base}|{nm}|{basis}|{d}|{p:.8f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

def make_circuit(c: Condition) -> stim.Circuit:
    task = f"surface_code:rotated_memory_{c.basis}"
    p = c.physical_error_rate
    if c.noise_model == "depol":
        return stim.Circuit.generated(task, distance=c.distance, rounds=c.rounds,
            after_clifford_depolarization=p, after_reset_flip_probability=p,
            before_measure_flip_probability=p, before_round_data_depolarization=p)
    return stim.Circuit.generated(task, distance=c.distance, rounds=c.rounds,
        after_clifford_depolarization=p*0.2, after_reset_flip_probability=p*2.0,
        before_measure_flip_probability=p*3.0, before_round_data_depolarization=p*0.5)

def simulate(c: Condition) -> dict:
    t0 = time.perf_counter()
    circ = make_circuit(c)
    dem = circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler(seed=c.seed)
    dets, obs = sampler.sample(c.shots, separate_observables=True)
    preds = matcher.decode_batch(dets)
    errs = np.any(preds != obs, axis=1)
    n_err = int(np.count_nonzero(errs))
    pL = (n_err + 0.5) / (c.shots + 1.0)
    suppression = c.physical_error_rate / pL
    return {**asdict(c),
        "logical_error_count": n_err,
        "logical_error_rate": n_err / c.shots,
        "logical_error_rate_laplace": pL,
        "log_suppression": math.log(max(suppression, 1e-60)),
        "syndrome_density": float(np.mean(dets)),
        "duration_seconds": time.perf_counter() - t0}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--noise-model", default="depol", choices=["depol","meas"])
    p.add_argument("--run-id", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    ps = [0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04]
    dists = [3,5,7,9,11]
    bases = ["x","z"]
    nm: NoiseModel = args.noise_model
    rid = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / rid; out.mkdir(parents=True, exist_ok=True)

    conds = [Condition(basis=b, distance=d, rounds=d, physical_error_rate=p,
               shots=args.shots, seed=condition_seed(args.seed,b,d,p,nm), noise_model=nm)
             for b in bases for d in dists for p in ps]

    rows = []
    for i, c in enumerate(conds, 1):
        print(f"[{i}/{len(conds)}] {nm} {c.basis} d={c.distance} p={c.physical_error_rate}", flush=True)
        rows.append(simulate(c))

    cp = out / "conditions.csv"
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    payload = {"run_id": rid, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"shots": args.shots, "distances": dists, "bases": bases,
                   "physical_error_rates": ps, "seed": args.seed, "noise_model": nm,
                   "heldout_distances": [9,11]},
        "conditions": rows}
    jp = out / "sweep.json"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {jp}")

if __name__ == "__main__":
    raise SystemExit(main())
