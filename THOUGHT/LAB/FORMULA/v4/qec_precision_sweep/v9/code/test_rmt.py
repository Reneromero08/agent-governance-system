"""RMT eigenvalue spacing test: Wigner-Dyson vs Poisson transition at QEC threshold.

Wigner-Dyson (GOE): eigenvalue spacings follow P(s) = (pi/2)s exp(-pi*s^2/4)
  -- characteristic of strongly correlated, structured systems
Poisson: P(s) = exp(-s)
  -- characteristic of uncorrelated, noisy systems

Framework predicts: below threshold (sigma>1), spacings follow Wigner-Dyson.
Above threshold (sigma<1), spacings follow Poisson.
The transition point should match sigma=1 (p ~ 0.007).
"""
import hashlib, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pymatching, stim

OUT = Path(__file__).parent / "rmt_results"
OUT.mkdir(exist_ok=True)

def t(d): return (d - 1) // 2
def seed(base, d, p, run):
    key = f"rmt|{base}|{d}|{p:.8f}|{run}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

# Parameters
TRAIN_DS = [3, 5, 7]
TEST_DS = [9, 11]
PS = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04]
SHOTS = 5000  # moderate shots for covariance estimation
RUNS = 3      # multiple runs for error bars
GEOM = "rotated"

print("=" * 65)
print("RMT EIGENVALUE SPACING TEST")
print("=" * 65)
print(f"Distances: train={TRAIN_DS}, test={TEST_DS}")
print(f"Error rates: {PS}")
print(f"Shots: {SHOTS} x {RUNS} runs")
print()

all_results = []

for d in TRAIN_DS + TEST_DS:
    for p_val in PS:
        for run_idx in range(RUNS):
            s = seed(20260518, d, p_val, run_idx)
            t0 = time.perf_counter()
            
            # Generate data
            task = f"surface_code:{GEOM}_memory_x"
            circ = stim.Circuit.generated(
                task, distance=d, rounds=d,
                after_clifford_depolarization=p_val,
                after_reset_flip_probability=p_val,
                before_measure_flip_probability=p_val,
                before_round_data_depolarization=p_val,
            )
            dem = circ.detector_error_model(decompose_errors=True)
            m = pymatching.Matching.from_detector_error_model(dem)
            det, obs = circ.compile_detector_sampler(seed=s).sample(SHOTS, separate_observables=True)
            pred = m.decode_batch(det)
            errs = np.any(pred != obs, axis=1)
            
            # Compute detection correlation matrix: (n_detectors, n_detectors)
            # Center the detection events
            det_centered = det.astype(np.float64) - np.mean(det, axis=0)
            # Correlation: C = (1/(n-1)) * D^T * D
            C = (det_centered.T @ det_centered) / (SHOTS - 1)
            
            # Eigenvalue spectrum
            ev = np.linalg.eigvalsh(C)
            ev = ev[ev > 1e-12]
            if len(ev) < 10:
                print(f"  d={d} p={p_val:.4f} run={run_idx}: too few eigenvalues ({len(ev)}), skipping")
                continue
            
            # Unfolding: normalize so mean spacing = 1
            ev_sort = np.sort(ev)
            # Estimate cumulative density and unfold
            n_ev = len(ev_sort)
            # Simple unfolding: rank/(n+1) gives uniform spacings on [0,1]
            # Then transform to exponential: -ln(1-rank/(n+1))
            unfolded = -np.log(1.0 - (np.arange(1, n_ev+1) / (n_ev + 1)))
            # But better: use local mean spacing
            spacings = np.diff(ev_sort)
            # Smooth: local mean over window
            window = max(5, n_ev // 10)
            local_mean = np.convolve(spacings, np.ones(window)/window, mode='same')
            local_mean = np.maximum(local_mean, 1e-15)
            unfolded_spacings = spacings / local_mean
            
            # Fit to Wigner-Dyson and Poisson
            s = unfolded_spacings
            s = s[(s > 0.01) & (s < 3.0)]  # remove outliers
            
            # Wigner-Dyson (GOE) PDF: (pi/2)*s*exp(-pi*s^2/4)
            wigner_pdf = (np.pi/2) * s * np.exp(-np.pi * s**2 / 4)
            # Poisson PDF: exp(-s)
            poisson_pdf = np.exp(-s)
            
            # Log-likelihood ratio: positive = Wigner favored, negative = Poisson favored
            llr = np.mean(np.log(wigner_pdf + 1e-15) - np.log(poisson_pdf + 1e-15))
            
            # Kolmogorov-Smirnov distances
            # Sort spacings for CDF comparison
            s_sort = np.sort(s)
            n_s = len(s_sort)
            
            # Empirical CDF
            emp_cdf = np.arange(1, n_s+1) / n_s
            
            # Wigner-Dyson CDF: 1 - exp(-pi*s^2/4)
            wigner_cdf = 1 - np.exp(-np.pi * s_sort**2 / 4)
            # Poisson CDF: 1 - exp(-s)
            poisson_cdf = 1 - np.exp(-s_sort)
            
            ks_wigner = np.max(np.abs(emp_cdf - wigner_cdf))
            ks_poisson = np.max(np.abs(emp_cdf - poisson_cdf))
            
            # Mean spacing ratio: <min(s_i, s_{i+1}) / max(s_i, s_{i+1})>
            # Wigner-Dyson: ~0.536, Poisson: ~0.386
            s_pairs = np.column_stack([spacings[:-1], spacings[1:]])
            min_s = np.min(s_pairs, axis=1)
            max_s = np.max(s_pairs, axis=1)
            max_s = np.maximum(max_s, 1e-15)
            ratios = min_s / max_s
            mean_ratio = float(np.mean(ratios[(ratios > 0) & (ratios < 1)]))
            
            result = {
                "d": d, "p": p_val, "run": run_idx,
                "log_likelihood_ratio": float(llr),
                "ks_wigner": float(ks_wigner),
                "ks_poisson": float(ks_poisson),
                "mean_spacing_ratio": mean_ratio,
                "wigner_wins": float(llr) > 0,
            }
            all_results.append(result)
            
            elapsed = time.perf_counter() - t0
            regime = "WIGNER" if llr > 0 else "POISSON"
            print(f"  d={d} p={p_val:.4f} r={run_idx}: llr={llr:+.3f} ks_w={ks_wigner:.3f} ks_p={ks_poisson:.3f} ratio={mean_ratio:.3f} [{regime}] {elapsed:.1f}s")

# Aggregate results
print(f"\n{'='*65}")
print("RESULTS")
print(f"{'='*65}")

# Group by (p, d)
from collections import defaultdict
groups = defaultdict(list)
for r in all_results:
    groups[(r["p"], r["d"])].append(r)

print(f"\n{'p':>8s} {'d':>4s} {'llr':>8s} {'ks_w':>8s} {'ks_p':>8s} {'ratio':>8s} {'regime':>10s}")
print("-" * 60)
for (p_val, d) in sorted(groups.keys()):
    grp = groups[(p_val, d)]
    llr = float(np.mean([r["log_likelihood_ratio"] for r in grp]))
    kw = float(np.mean([r["ks_wigner"] for r in grp]))
    kp = float(np.mean([r["ks_poisson"] for r in grp]))
    mr = float(np.mean([r["mean_spacing_ratio"] for r in grp]))
    regime = "WIGNER" if llr > 0 else "POISSON"
    print(f"{p_val:8.4f} {d:4d} {llr:+8.3f} {kw:8.3f} {kp:8.3f} {mr:8.3f} {regime:>10s}")

# Test: does the Wigner->Poisson transition match sigma=1?
print(f"\nWigner-Dyson ratio: 0.536")
print(f"Poisson ratio:      0.386")
print(f"\nTransition analysis:")
for p_val in PS:
    train_llrs = [r["log_likelihood_ratio"] for r in all_results if abs(r["p"]-p_val)<1e-10 and r["d"] in TRAIN_DS]
    test_llrs = [r["log_likelihood_ratio"] for r in all_results if abs(r["p"]-p_val)<1e-10 and r["d"] in TEST_DS]
    if train_llrs:
        print(f"  p={p_val:.4f}: train_llr={np.mean(train_llrs):+.3f} test_llr={np.mean(test_llrs):+.3f}")

# Save
with open(OUT / "rmt_results.json", "w") as f:
    json.dump({"results": all_results, "config": {"distances": TRAIN_DS + TEST_DS, "error_rates": PS, "shots": SHOTS, "runs": RUNS}}, f, indent=2)
print(f"\nSaved to {OUT / 'rmt_results.json'}")
