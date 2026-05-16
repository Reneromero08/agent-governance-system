"""Lissajous Hypothesis Phase 1+2: Frequency extraction and rationality measure.

From existing QEC data, compute frequency ratios between distance levels
from syndrome density, measure rationality, compare to empirical sigma.
"""
import json, math, numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "qec_precision_sweep" / "v8" / "results" / "v8_depol" / "sweep.json"
RESULTS = ROOT

SIMPLE_RATIONALS = [
    (1,1), (2,1), (3,1), (4,1),
    (3,2), (5,2),
    (4,3), (5,3),
    (5,4),
]

def rationality(r, alpha=10.0):
    """Measure how close a ratio is to any simple rational."""
    best_dist = float('inf')
    for p, q in SIMPLE_RATIONALS:
        target = p / q
        dist = abs(r - target)
        if dist < best_dist:
            best_dist = dist
    return math.exp(-alpha * best_dist)

# Load and pool data
data = json.loads(SWEEP.read_text())
g = defaultdict(list)
for r in data["conditions"]:
    g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)

# Per (p,d): syndrome density and logR
pooled = {}
for (p, d), vs in g.items():
    if p == 0.0005: continue  # floor effect
    pooled[(p, d)] = {
        "syn": float(np.mean([x["syndrome_density"] for x in vs])),
        "logR": float(np.mean([x["log_suppression"] for x in vs]))
    }

# Compute empirical sigma per p from training {3,5,7}
def t(d): return (d-1)//2
tg = defaultdict(list)
for (p, d), r in pooled.items():
    if d in {3,5,7}: tg[p].append((t(d), r["logR"]))
sigma_emp = {}
for p, pts in tg.items():
    if len(pts) < 2: continue
    pts.sort()
    ts_arr = np.array([x[0] for x in pts])
    ls_arr = np.array([x[1] for x in pts])
    A = np.column_stack([ts_arr, np.ones_like(ts_arr)])
    coeff = float(np.linalg.lstsq(A, ls_arr, rcond=None)[0][0])
    if not math.isfinite(coeff) or abs(coeff) > 50:
        print(f"WARNING: Unstable regression coefficient {coeff:.2f} for p={p}")
        sigma_emp[p] = None
    else:
        sigma_emp[p] = math.exp(coeff)

print("=" * 70)
print("LISSAJOUS HYPOTHESIS: Phase 1 — Frequency Extraction")
print("=" * 70)

# Frequency ratios between adjacent distance levels
print(f"\n{'p':>8s}  {'sigma_emp':>10s}  ", end="")
ratios_labels = []
for d1, d2 in [(3,5), (5,7), (7,9), (9,11)]:
    rat = f"syn_{d2}/{d1}"
    ratios_labels.append(rat)
    print(f"{rat:>10s}  ", end="")
print(f"{'rat_mean':>10s}  {'rationality':>12s}")

freq_data = []
for p in sorted(sigma_emp):
    syns = {}
    for (pp, d), r in pooled.items():
        if pp == p and d in {3,5,7,9,11}:
            syns[d] = r["syn"]
    
    ratios = []
    rat_values = []
    for (d1, d2) in [(3,5), (5,7), (7,9)]:
        if d1 in syns and d2 in syns and syns[d1] > 0:
            ratio = syns[d2] / syns[d1]
            ratios.append(ratio)
            rat_values.append(ratio)
    
    if not ratios:
        continue
    
    mean_rat = float(np.mean(ratios))
    mean_rationality = float(np.mean([rationality(r) for r in ratios]))
    
    print(f"{p:8.4f}  {sigma_emp[p]:10.4f}  ", end="")
    for rv in rat_values:
        print(f"{rv:10.4f}  ", end="")
    print(f"{mean_rat:10.4f}  {mean_rationality:12.6f}")
    
    freq_data.append({"p": p, "sigma_emp": sigma_emp[p], "ratios": ratios,
                       "mean_ratio": mean_rat, "rationality": mean_rationality})

# Phase 2: Does rationality predict sigma?
print(f"\n{'=' * 70}")
print("PHASE 2: Rationality Measure")
print(f"{'=' * 70}")

rat_vals = np.array([d["rationality"] for d in freq_data])
sigma_vals = np.array([d["sigma_emp"] for d in freq_data])

# Fit: sigma_emp = a * rationality + b
A = np.column_stack([rat_vals, np.ones_like(rat_vals)])
coeffs = np.linalg.lstsq(A, sigma_vals, rcond=None)[0]
pred_sigma = coeffs[0] * rat_vals + coeffs[1]
residuals = sigma_vals - pred_sigma
r2 = 1 - np.sum(residuals**2) / np.sum((sigma_vals - np.mean(sigma_vals))**2)
r_p = np.corrcoef(rat_vals, sigma_vals)[0, 1]

# Also test: sigma ~ c * rationality (force intercept=0)
single_coeff = np.linalg.lstsq(rat_vals.reshape(-1,1), sigma_vals, rcond=None)[0][0]
pred2 = single_coeff * rat_vals
r2_force0 = 1 - np.sum((sigma_vals - pred2)**2) / np.sum((sigma_vals - np.mean(sigma_vals))**2)

print(f"\nSimple syndromensity frequency ratios:")
print(f"  Pearson r(rationality, sigma_emp): {r_p:+.4f}")
print(f"  sigma_emp = {coeffs[0]:.4f} * rationality + {coeffs[1]:.4f}")
print(f"  R2: {r2:.4f}")
print(f"  R2 (forced intercept=0): {r2_force0:.4f}")

logR_freq_data = []
# Phase 2b: Try different frequency definitions
# What if "frequency" = -log(syndrome_density)? (information-theoretic)
# What if "frequency" = sqrt(syn)? (our grad_S)
# What if ratio = sigma at d2 / sigma at d1? No — that's circular

# Try: ratio of logR at adjacent distances
print(f"\nUsing logR ratios as frequency proxy:")
rat_logR = []
for p in sorted(sigma_emp):
    logRs = {}
    for (pp, d), r in pooled.items():
        if pp == p and d in {3,5,7,9,11}:
            logRs[d] = r["logR"]
    
    ratios_lr = []
    for d1, d2 in [(3,5), (5,7), (7,9)]:
        if d1 in logRs and d2 in logRs:
            ratios_lr.append(logRs[d2] - logRs[d1])  # delta, not ratio
    
    if ratios_lr:
        # Rationality of the delta logR (closer to zero = more rational = sigma near 1)
        rat_score = float(np.mean([rationality(abs(d) + 1e-6, alpha=5.0) for d in ratios_lr]))
        logR_freq_data.append({"p": p, "sigma_emp": sigma_emp.get(p, 0), "rationality": rat_score, "ratios": [], "mean_ratio": 0})

lr_data = logR_freq_data
if lr_data:
    lr_rat = np.array([d["rationality"] for d in lr_data])
    lr_sig = np.array([d["sigma_emp"] for d in lr_data])
    r_p2 = np.corrcoef(lr_rat, lr_sig)[0, 1]
    print(f"  Pearson r(rationality(delta_logR), sigma_emp): {r_p2:+.4f}")

# Phase 2c: Direct frequency from eigenvalue spectrum analysis
# Use the actual stim circuit to get stabilizer coupling
# For now: use ratio of (1 - syn) as "coherence frequency"
print(f"\nUsing coherence (1-syn) ratios as frequency proxy:")
for p in sorted(sigma_emp):
    coh = {}
    for (pp, d), r in pooled.items():
        if pp == p and d in {3,5,7,9,11}:
            coh[d] = 1.0 - r["syn"]  # fraction of quiet detectors
    
    ratios_coh = []
    for d1, d2 in [(3,5), (5,7), (7,9)]:
        if d1 in coh and d2 in coh and coh[d1] > 0:
            ratios_coh.append(coh[d2] / coh[d1])
    
    if ratios_coh:
        mean_rat_coh = float(np.mean([rationality(r, alpha=20.0) for r in ratios_coh]))
        print(f"  p={p:.4f} sigma={sigma_emp[p]:.4f} coh_ratios={[round(r,4) for r in ratios_coh]} rationality={mean_rat_coh:.4f}")

# Summary
print(f"\n{'=' * 70}")
print("FINDINGS")
print(f"{'=' * 70}")

if abs(r_p) > 0.6:
    print("STRONG: Frequency rationality from syndrome density predicts sigma.")
elif abs(r_p) > 0.3:
    print("MODERATE: Some correlation, but not definitive.")
else:
    print(f"WEAK/NULL: r={r_p:.2f}. Syndrome density ratios do not capture frequency rationality.")
    print("Possible reasons:")
    print("  1. 'Frequency' is not syndrome_density but something else (eigendecomposition of DEM)")
    print("  2. Frequency ratios should be computed from stabilizer correlations, not raw rates")
    print("  3. The Lissajous mechanism requires per-stabilizer analysis, not aggregate statistics")
    print("  4. The hypothesis is wrong for this operationalization")

# Save results
out = ROOT / "phase1_results.json"
try:
    out.write_text(json.dumps({"freq_data": [{"p": d["p"], "sigma_emp": d["sigma_emp"],
        "mean_ratio_syn": d.get("mean_ratio", 0), "rationality": d["rationality"]} for d in freq_data if "mean_ratio" in d],
        "correlation": float(r_p), "r2": float(r2)}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
except (OSError, IOError) as e:
    print(f"ERROR: Failed to save results to {out}: {e}")
