"""Lissajous Phase 3: DEM-level frequency rationality analysis.

Extracts detector error models from existing sweep circuits, builds correlation
matrices, finds dominant frequency modes via eigendecomposition, measures
frequency ratio rationality between distance levels, compares to sigma_empirical.
"""
import json, math, numpy as np, stim
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "qec_precision_sweep" / "v8" / "results" / "v8_depol" / "sweep.json"
OUT = ROOT

SIMPLE_RATIONALS = [(1,1),(2,1),(3,1),(3,2),(4,1),(4,3),(5,2),(5,3),(5,4),(5,1)]

def make_circuit(d, p, basis="x", noise="depol", rounds=None):
    task = f"surface_code:rotated_memory_{basis}"
    r = rounds or d
    if noise == "depol":
        return stim.Circuit.generated(task, distance=d, rounds=r,
            after_clifford_depolarization=p, after_reset_flip_probability=p,
            before_measure_flip_probability=p, before_round_data_depolarization=p)
    else:
        return stim.Circuit.generated(task, distance=d, rounds=r,
            after_clifford_depolarization=p*0.2, after_reset_flip_probability=p*2.0,
            before_measure_flip_probability=p*3.0, before_round_data_depolarization=p*0.5)

def build_correlation_matrix(d, p, noise="depol"):
    """Build detector correlation matrix from DEM."""
    c = make_circuit(d, p, "x", noise)
    dem = c.detector_error_model(decompose_errors=True)
    n = dem.num_detectors
    if n == 0: return np.eye(1)
    
    M = np.zeros((n, n))
    # Iterate over DEM instructions
    for inst in dem.flattened():
        if inst.type == "error":
            dets = []
            for t in inst.targets_copy():
                if t.is_separator: continue
                dets.append(t.val)
            if not dets: continue
            prob = inst.args_copy()[0]
            for i in dets:
                M[i, i] += prob  # diagonal: independent contribution
                for j in dets:
                    if i != j:
                        M[i, j] += prob  # correlated contribution
    
    return M

def rationality(r, alpha=10.0):
    best = float('inf')
    for p, q in SIMPLE_RATIONALS:
        best = min(best, abs(r - p/q))
    return math.exp(-alpha * best)

# Load existing sweep to get sigma_empirical
print("Loading existing sweep data...")
data = json.loads(SWEEP.read_text())
g = defaultdict(list)
for r in data["conditions"]:
    g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)

def t(d): return (d-1)//2
tg = defaultdict(list)
pooled = {}
for (p, d), vs in g.items():
    if p == 0.0005: continue
    pooled[(p, d)] = float(np.mean([x["log_suppression"] for x in vs]))
    if d in {3,5,7}: tg[p].append((t(d), pooled[(p,d)]))

sigma_emp = {}
for p, pts in tg.items():
    pts.sort()
    ts_arr = np.array([x[0] for x in pts])
    ls_arr = np.array([x[1] for x in pts])
    A = np.column_stack([ts_arr, np.ones_like(ts_arr)])
    sigma_emp[p] = math.exp(float(np.linalg.lstsq(A, ls_arr, rcond=None)[0][0]))

print(f"Sigma empirical: {len(sigma_emp)} p values")

# Phase 3: DEM-level analysis
print("\nExtracting DEMs and building correlation matrices...")
distances = [3,5,7,9,11]
ps = sorted(sigma_emp.keys())
spectra = {}  # (p,d) -> top eigenvalues

for p in ps:
    for d in distances:
        if (p,d) not in pooled: continue
        print(f"  p={p:.4f} d={d}", end=" ", flush=True)
        M = build_correlation_matrix(d, p)
        # Eigendecomposition
        eigenvals = np.linalg.eigvalsh(M)
        # Sort descending by absolute value
        eigenvals = np.sort(np.abs(eigenvals))[::-1]
        spectra[(p,d)] = eigenvals[:min(10, len(eigenvals))]
        print(f" top5={[round(v,6) for v in spectra[(p,d)][:5]]}", flush=True)

# Phase 4: Frequency ratios between distance levels
print("\nPhase 4: Frequency ratios between distance levels...")
freq_ratios = {}  # (p) -> list of rationality scores

for p in ps:
    ratios = []
    for d1, d2 in [(3,5), (5,7), (7,9)]:
        if (p,d1) not in spectra or (p,d2) not in spectra:
            continue
        e1 = spectra[(p,d1)]
        e2 = spectra[(p,d2)]
        
        # Match dominant modes between distance levels
        # Approach 1: ratio of largest eigenvalues
        r = e2[0] / max(e1[0], 1e-12)
        ratios.append(rationality(r))
        
        # Approach 2: ratio of eigenvalue means
        r_mean = np.mean(e2[:3]) / max(np.mean(e1[:3]), 1e-12)
        ratios.append(rationality(r_mean))
        
        # Approach 3: dominant mode frequency shift
        if len(e1) >= 2 and len(e2) >= 2:
            r_mode = e2[1] / max(e1[1], 1e-12)  # second mode ratio
            ratios.append(rationality(r_mode))
    
    if ratios:
        freq_ratios[p] = float(np.median(ratios))

# Phase 5: Compare to sigma_empirical
print("\nPhase 5: Rationality vs sigma_empirical...")
print(f"{'p':>8s}  {'sigma_emp':>10s}  {'rationality':>12s}  {'ratio_pred':>10s}")
rat_vals = []
sig_vals = []
for p in sorted(freq_ratios):
    rat = freq_ratios[p]
    sig = sigma_emp[p]
    rat_vals.append(rat)
    sig_vals.append(sig)
    print(f"{p:8.4f}  {sig:10.4f}  {rat:12.6f}  {rat/0.04:10.4f}")

rat_arr = np.array(rat_vals)
sig_arr = np.array(sig_vals)

# Pearson correlation
r_p = np.corrcoef(rat_arr, sig_arr)[0,1]

# Linear fit
A = np.column_stack([rat_arr, np.ones_like(rat_arr)])
coeffs = np.linalg.lstsq(A, sig_arr, rcond=None)[0]
pred = coeffs[0] * rat_arr + coeffs[1]
r2 = 1 - np.sum((sig_arr - pred)**2) / np.sum((sig_arr - np.mean(sig_arr))**2)
mae = float(np.mean(np.abs(sig_arr - pred)))

# Force intercept=0 fit
coeff0 = np.linalg.lstsq(rat_arr.reshape(-1,1), sig_arr, rcond=None)[0][0]
r2_0 = 1 - np.sum((sig_arr - coeff0 * rat_arr)**2) / np.sum((sig_arr - np.mean(sig_arr))**2)

print(f"\nPhase 6: Results")
print(f"  Pearson r: {r_p:+.4f}")
print(f"  R2 (with intercept): {r2:.4f}")
print(f"  R2 (forced intercept=0): {r2_0:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  sigma_emp = {coeffs[0]:.4f} * rationality + {coeffs[1]:.4f}")

# Compare to closed-form baseline
p_th = 0.00707  # from QEC paper
cf_sigma = {p: math.sqrt(p_th/p) for p in sig_vals if p > 0}
cf_arr = np.array([cf_sigma.get(p, 1) for p in sig_vals])
A_cf = np.column_stack([cf_arr, np.ones_like(cf_arr)])
coeffs_cf = np.linalg.lstsq(A_cf, sig_arr, rcond=None)[0]
pred_cf = coeffs_cf[0] * cf_arr + coeffs_cf[1]
r2_cf = 1 - np.sum((sig_arr - pred_cf)**2) / np.sum((sig_arr - np.mean(sig_arr))**2)
print(f"\n  Baseline sqrt(p_th/p): R2={r2_cf:.4f}")

# Verdict
print(f"\n{'='*60}")
if r_p > 0.7 and r2 > 0.70:
    print("CONFIRMED: Lissajous rationality predicts sigma.")
elif r_p > 0.4:
    print(f"PARTIAL: Moderate correlation (r={r_p:.2f}) but below threshold.")
else:
    print(f"NULL: r={r_p:.2f}, R2={r2:.2f}. DEM eigenvalue ratios do not predict sigma.")
    print("The Lissajous hypothesis, as operationalized through DEM eigendecomposition,")
    print("does not explain the fidelity factor for rotated surface codes.")
