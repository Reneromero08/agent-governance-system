"""Geometric sigma: use QGT Fubini-Study metric eigenvalues to define sigma,
bypassing empirical training-distance fitting. If sigma is measured in the
manifold's intrinsic coordinates, alpha should converge to 1.0.

Theory: The Fubini-Study metric G encodes the natural geometry of the logR(Df)
curve. The slope d(logR)/d(Df) = ln(sigma). In extrinsic coordinates, this slope
is alpha * Df_coord. In intrinsic coordinates (using G^{-1}), the slope should be
exactly Df_intrinsic, meaning alpha_intrinsic = 1.0.
"""
import json, math, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Add qgt_lib to path
QGT_PATH = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_ALIGNMENT\qgt_lib\python")
sys.path.insert(0, str(QGT_PATH))
from qgt import fubini_study_metric, participation_ratio, metric_eigenspectrum, normalize_embeddings

ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\qec_precision_sweep")
V8_SWEEP = ROOT / "v8" / "results" / "v8_depol" / "sweep.json"
HIGH_D_SWEEP = ROOT / "v9" / "results" / "20260517T214104Z" / "sweep.json"

def t(d): return (d - 1) // 2

def load_and_pool():
    v8 = json.loads(V8_SWEEP.read_text(encoding="utf-8"))
    hd = json.loads(HIGH_D_SWEEP.read_text(encoding="utf-8"))
    rows = v8["conditions"] + hd["conditions"]
    g = defaultdict(list)
    for r in rows:
        g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
    return [{"p": float(p), "d": int(d), "Df": t(int(d)),
             "logR": float(np.mean([r["log_suppression"] for r in grp])),
             "syndrome_density": float(np.mean([r["syndrome_density"] for r in grp])),
             "total_errors": sum(r["logical_error_count"] for r in grp)}
            for (p,d), grp in g.items()]

print("=== GEOMETRIC SIGMA: QGT Fubini-Study Approach ===\n")

# Step 1: Load data
rows = load_and_pool()
dists = sorted(set(r["d"] for r in rows))
ps = sorted(set(r["p"] for r in rows))
print(f"Data: {len(rows)} points, d={dists}, p={ps}")

# Step 2: Build per-p state vectors in logR space
# For each p, we have logR at various d. The vector is [logR(d1), logR(d2), ...]
# We also include log(1/p) as the reference energy scale.
print("\n--- Step 1: Building per-p state vectors ---")
state_vectors = {}
for p in ps:
    vec = []
    for d in dists:
        match = [r for r in rows if abs(r["p"]-p) < 1e-10 and r["d"]==d]
        if match:
            vec.append(match[0]["logR"])
        else:
            vec.append(0.0)
    vec.append(math.log(1.0 / max(p, 1e-10)))  # energy reference
    state_vectors[p] = np.array(vec)

n_dim = len(dists) + 1
print(f"State vector dim: {n_dim} ({len(dists)} logR + 1 energy)")

# Step 3: Compute Fubini-Study metric on the state vector manifold
# The manifold is parameterized by p. Each p gives a point in R^{n_dim}.
# The Fubini-Study metric G governs how these points relate.
print("\n--- Step 2: Computing Fubini-Study metric ---")
V = np.array([state_vectors[p] for p in ps])
print(f"  Input shape: {V.shape}")

# Standardize for numerical stability
V_mean = V.mean(axis=0)
V_std = V.std(axis=0, ddof=0)
V_std = np.where(V_std < 1e-12, 1.0, V_std)
V_norm = (V - V_mean) / V_std

metric = fubini_study_metric(V_norm, normalize=True)
eigenvalues, eigenvectors = metric_eigenspectrum(V_norm)
pr = participation_ratio(V_norm)

print(f"  Participation ratio (Df_intrinsic): {pr:.4f}")
print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
print(f"  Eigenvalue sum: {eigenvalues.sum():.6f}")
print(f"  Effective rank: {pr:.4f} out of {n_dim}")

# Step 4: Derive geometric sigma from the metric
# The slope d(logR)/d(Df) = ln(sigma). In extrinsic coords this is alpha * Df.
# In intrinsic coords (using G), it should be exactly Df_intrinsic.
# 
# Method: Project the state vector onto the principal eigenvector of G.
# The projection ratio gives the geometric alignment strength.
# sigma_geo = exp(|projection| / Df_coord)
print("\n--- Step 3: Computing geometric sigma ---")

principal_eigenvector = eigenvectors[:, 0]  # corresponds to largest eigenvalue

sigma_geo = {}
for p in ps:
    v = state_vectors[p]
    v_norm = (v - V_mean) / V_std
    # Project onto principal axis
    proj = np.abs(np.dot(v_norm, principal_eigenvector))
    # Geometric sigma: exponential of projection normalized by effective Df range
    # The projection measures how much this p's state aligns with the dominant geometric mode
    mean_v = np.mean(np.abs(v_norm))
    sigma_geo[p] = math.exp(proj / max(mean_v, 0.01))

print(f"  Geometric sigma range: [{min(sigma_geo.values()):.4f}, {max(sigma_geo.values()):.4f}]")

# Step 5: Alternative approach - sigma from metric eigenvalues directly
# The dominant eigenvalue of G represents the principal variation scale.
# sigma_alt = exp(sqrt(lambda_max) / Df_max)
print("\n--- Step 4: Alternative sigma from metric eigenvalues ---")

sigma_alt = {}
for p in ps:
    v = state_vectors[p]
    v_norm = (v - V_mean) / V_std
    # Quadratic form: v^T G v = sum of squared projections onto eigenvectors
    quad = np.dot(v_norm, np.dot(metric, v_norm))
    # This is the squared distance in the metric
    sigma_alt[p] = math.exp(math.sqrt(max(quad, 1e-10)) / max(t(max(dists)), 1))

print(f"  Alternative sigma range: [{min(sigma_alt.values()):.4f}, {max(sigma_alt.values()):.4f}]")

# Step 6: Compute empirical sigma (original method, for comparison)
print("\n--- Step 5: Empirical sigma (baseline) ---")
train = {3,5,7}
groups = defaultdict(list)
for r in rows:
    if r["d"] in train:
        groups[r["p"]].append((r["Df"], r["logR"]))
sigma_emp = {}
for p, pts in sorted(groups.items()):
    if len(pts) < 2: continue
    pts.sort()
    ts_arr = np.array([pt[0] for pt in pts])
    ls = np.array([pt[1] for pt in pts])
    A = np.column_stack([ts_arr, np.ones_like(ts_arr)])
    coef = np.linalg.lstsq(A, ls, rcond=None)[0]
    sigma_emp[p] = math.exp(float(coef[0]))
    print(f"  p={p:.4f}: sigma_emp={sigma_emp[p]:.4f}, sigma_geo={sigma_geo.get(p,0):.4f}, sigma_alt={sigma_alt.get(p,0):.4f}")

# Step 7: Compute E and evaluate alpha with each sigma
print("\n--- Step 6: Calibrate E and evaluate alpha ---")
eps = 1e-60

def calibrate_E(rows, sigma_map, train_dists):
    ests = []
    for r in rows:
        d = int(r["d"])
        if d not in train_dists: continue
        sp = sigma_map.get(r["p"])
        if sp is None: continue
        gs = max(float(r["syndrome_density"]), eps)
        ests.append(r["logR"] + math.log(gs) - r["Df"] * math.log(max(sp, eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

def evaluate(rows, sigma_map, E_val, heldout):
    yp=[]; ya=[]
    for r in rows:
        d = int(r["d"])
        if d not in heldout: continue
        sp = sigma_map.get(r["p"], 1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**r["Df"]), eps))
        yp.append(lrp); ya.append(r["logR"])
    yp=np.array(yp); ya=np.array(ya)
    mae=float(mean_absolute_error(ya,yp))
    r2v=float(r2_score(ya,yp))
    A=np.column_stack([yp,np.ones_like(yp)])
    coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    return float(coef[0]), float(coef[1]), mae, r2v

heldout = {9,11,13,15}
# Filter: only points with >= 10 observed errors
heldout_rows = [r for r in rows if r["d"] in heldout and r["total_errors"] >= 10]

results = []
for name, smap in [("Empirical", sigma_emp), ("Geometric (proj)", sigma_geo), ("Geometric (quad)", sigma_alt)]:
    E_val = calibrate_E(rows, smap, train)
    alpha, beta, mae, r2v = evaluate(heldout_rows, smap, E_val, heldout)
    results.append((name, alpha, beta, mae, r2v, E_val))
    print(f"  {name:20s}: alpha={alpha:.4f}  beta={beta:.4f}  R2={r2v:.4f}  MAE={mae:.4f}  E={E_val:.6f}")

# Step 8: Direct QGT-based sigma: use the principal eigenvalue ratio
# The ratio of largest to second-largest eigenvalue gives the "compression factor"
print("\n--- Step 7: Pure geometric sigma (eigenvalue ratio) ---")
if len(eigenvalues) >= 2:
    ev_ratio = eigenvalues[0] / max(eigenvalues[1], 1e-15)
else:
    ev_ratio = eigenvalues[0]
print(f"  Eigenvalue ratio (lambda_1/lambda_2): {ev_ratio:.4f}")

# Use the eigenvalue ratio directly as the geometric compression factor
sigma_pure = {}
for p in ps:
    sigma_pure[p] = ev_ratio  # Same for all p - it's the global geometric compression

E_pure = calibrate_E(rows, sigma_pure, train)
alpha_pure, beta_pure, mae_pure, r2v_pure = evaluate(heldout_rows, sigma_pure, E_pure, heldout)
print(f"  Pure geometric: alpha={alpha_pure:.4f}  beta={beta_pure:.4f}  R2={r2v_pure:.4f}  MAE={mae_pure:.4f}  E={E_pure:.6f}")
results.append(("Pure geometric (ev)", alpha_pure, beta_pure, mae_pure, r2v_pure, E_pure))

# Step 9: The critical insight - alpha *should* approach 1.0 if sigma is 
# measured in intrinsic coordinates. Let's compute the correction factor.
print("\n=== TRIANGULATION ===")
print(f"  Empirical alpha: {results[0][1]:.4f}")
print(f"  Geometric alpha (proj): {results[1][1]:.4f}")
print(f"  Geometric alpha (quad): {results[2][1]:.4f}")
print(f"  Pure geometric alpha: {results[3][1]:.4f}")
print(f"  Df_intrinsic (participation ratio): {pr:.4f}")
print(f"  Df_coord max: {t(max(dists))}")

# The ratio of intrinsic Df to coordinate Df should equal alpha
alpha_predicted = pr / t(max(dists))
print(f"  Predicted alpha (PR/Df_max): {alpha_predicted:.4f}")
print(f"  Ratio (empirical alpha / predicted): {results[0][1]/alpha_predicted:.4f}")
