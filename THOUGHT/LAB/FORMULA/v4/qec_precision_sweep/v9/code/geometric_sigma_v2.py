"""Geometric sigma v2: Natural gradient of logR w.r.t Df using local QGT.

The formula: ln(R) = ln(E/grad_S) + Df * ln(sigma)
Extrinsic slope: d(lnR)/d(Df) = ln(sigma)_extrinsic = alpha * Df
Natural gradient: G^{-1} * d(lnR)/d(Df) = ln(sigma)_intrinsic = Df (should be)

If alpha=1 in intrinsic coords, sigma_intrinsic = sigma_extrinsic^(1/alpha).
The goal: compute G^{-1} locally at each (p,Df) point to see if alpha_intrinsic = 1.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

QGT_PATH = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_ALIGNMENT\qgt_lib\python")
sys.path.insert(0, str(QGT_PATH))
from qgt import fubini_study_metric, normalize_embeddings

ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\qec_precision_sweep")
V8_SWEEP = ROOT / "v8" / "results" / "v8_depol" / "sweep.json"
HIGH_D_SWEEP = ROOT / "v9" / "results" / "20260517T214104Z" / "sweep.json"

def t(d): return (d - 1) // 2

v8 = json.loads(V8_SWEEP.read_text(encoding="utf-8"))
hd = json.loads(HIGH_D_SWEEP.read_text(encoding="utf-8"))
rows_raw = v8["conditions"] + hd["conditions"]
g = defaultdict(list)
for r in rows_raw:
    g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
rows = [{"p": float(p), "d": int(d), "Df": t(int(d)),
         "logR": float(np.mean([r["log_suppression"] for r in grp]))}
        for (p,d), grp in g.items()]

dists = sorted(set(r["d"] for r in rows))

print("=== GEOMETRIC SIGMA v2: Local Natural Gradient ===\n")

# For each error rate p, compute the local QGT in (Df, logR) space
# and extract the natural gradient. The natural gradient should give
# ln(sigma) exactly equal to Df * ln(sigma_emp) / alpha.
print("Per-p natural gradient analysis:")
print(f"{'p':>8s} {'sigma_emp':>10s} {'slope_ext':>10s} {'metric':>10s} {'slope_nat':>10s} {'alpha_nat':>10s}")
print("-" * 70)

alpha_nat_values = []
for p in sorted(set(r["p"] for r in rows)):
    # Get points for this p across distances
    pts = sorted([r for r in rows if abs(r["p"]-p) < 1e-10], key=lambda x: x["d"])
    if len(pts) < 3: continue
    
    df_vals = np.array([pt["Df"] for pt in pts])
    lr_vals = np.array([pt["logR"] for pt in pts])
    
    # Extrinsic slope via linear regression
    A = np.column_stack([df_vals, np.ones_like(df_vals)])
    slope_ext = float(np.linalg.lstsq(A, lr_vals, rcond=None)[0][0])
    sigma_emp = math.exp(slope_ext)
    
    # Build local QGT: 2D manifold parameterized by (Df, logR)
    # Normalize the 2D points
    pts_2d = np.column_stack([df_vals, lr_vals])
    pts_norm = normalize_embeddings(pts_2d)
    
    # Fubini-Study metric (2x2 covariance)
    metric_local = fubini_study_metric(pts_norm, normalize=False)
    metric_scalar = metric_local[0, 0]  # G_Df,Df component
    
    # Natural gradient: G^{-1} * extrinsic_gradient
    # For 1D parameter (Df), G is scalar, so natural gradient = slope_ext / G_DfDf
    if metric_scalar > 1e-15:
        slope_nat = slope_ext / metric_scalar
        alpha_nat = slope_nat / slope_ext if abs(slope_ext) > 1e-15 else 0
        alpha_nat_values.append(alpha_nat)
    else:
        slope_nat = slope_ext
        alpha_nat = 1.0
    
    print(f"{p:8.4f} {sigma_emp:10.4f} {slope_ext:10.4f} {metric_scalar:10.6f} {slope_nat:10.4f} {alpha_nat:10.4f}")

# Now compute the natural gradient sigma and test alpha
print(f"\n=== Natural gradient sigma vs empirical ===")
print(f"Mean alpha_nat = {np.mean(alpha_nat_values):.4f}")

# Use the natural gradient to define sigma_nat
sigma_nat = {}
for p in sorted(set(r["p"] for r in rows)):
    pts = sorted([r for r in rows if abs(r["p"]-p) < 1e-10], key=lambda x: x["d"])
    if len(pts) < 3: continue
    df_vals = np.array([pt["Df"] for pt in pts])
    lr_vals = np.array([pt["logR"] for pt in pts])
    pts_2d = np.column_stack([df_vals, lr_vals])
    pts_norm = normalize_embeddings(pts_2d)
    metric_local = fubini_study_metric(pts_norm, normalize=False)
    metric_scalar = metric_local[0, 0]
    if metric_scalar > 1e-15:
        A = np.column_stack([df_vals, np.ones_like(df_vals)])
        slope_ext = float(np.linalg.lstsq(A, lr_vals, rcond=None)[0][0])
        slope_nat = slope_ext / metric_scalar
        sigma_nat[p] = math.exp(slope_nat)
    else:
        sigma_nat[p] = math.exp(slope_ext)

# Evaluate alpha with natural gradient sigma
eps = 1e-60
def calibrate_E(rows, smap, train_dists):
    ests = []
    for r in rows:
        d = int(r["d"])
        if d not in train_dists: continue
        sp = smap.get(r["p"])
        if sp is None: continue
        gs = max(float(r.get("syndrome_density", 1e-10)), eps)
        ests.append(r["logR"] + math.log(gs) - r["Df"] * math.log(max(sp, eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

# Add syndrome_density back to rows
g2 = defaultdict(list)
for r in rows_raw:
    g2[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
rows_full = [{"p": float(p), "d": int(d), "Df": t(int(d)),
              "logR": float(np.mean([r["log_suppression"] for r in grp])),
              "syndrome_density": float(np.mean([r["syndrome_density"] for r in grp])),
              "total_errors": sum(r["logical_error_count"] for r in grp)}
             for (p,d), grp in g2.items()]

train = {3,5,7}
heldout = {9,11,13,15}
heldout_rows_f = [r for r in rows_full if r["d"] in heldout and r["total_errors"] >= 10]

E_nat = calibrate_E(rows_full, sigma_nat, train)

yp=[]; ya=[]
for r in heldout_rows_f:
    sp = sigma_nat.get(r["p"], 1.0)
    gs = max(float(r["syndrome_density"]), eps)
    lrp = math.log(max((E_nat/gs)*(sp**r["Df"]), eps))
    yp.append(lrp); ya.append(r["logR"])
yp=np.array(yp); ya=np.array(ya)
A=np.column_stack([yp,np.ones_like(yp)])
coef=np.linalg.lstsq(A,ya,rcond=None)[0]
alpha_nat_final = float(coef[0])
r2v = 1 - np.sum((ya-yp)**2)/np.sum((ya-ya.mean())**2)
mae = float(np.mean(np.abs(ya-yp)))

print(f"\nNatural gradient sigma: alpha={alpha_nat_final:.4f}  beta={float(coef[1]):.4f}  R2={r2v:.4f}  MAE={mae:.4f}")
print(f"Empirical sigma:      alpha=0.7482")

# The key test: does natural gradient alpha approach 1.0?
# If extrinsic alpha = 0.75, and G_DfDf ≈ 0.75, then natural alpha = 1.0
print(f"\n=== GEOMETRIC INTERPRETATION ===")
print(f"The Fubini-Study metric component G_DfDf measures how 'stretched'")
print(f"the Df coordinate is relative to the intrinsic geometry.")
print(f"If G_DfDf ≈ 0.75 (the empirical alpha), then the intrinsic slope is")
print(f"slope_nat = slope_ext / 0.75 = slope_ext * 1.33")
print(f"This would mean alpha_intrinsic = 1.0.")
print(f"")
print(f"Empirical alpha = {0.7482:.4f}")
print(f"Natural gradient alpha = {alpha_nat_final:.4f}")
print(f"Ratio = {alpha_nat_final/0.7482:.4f}")
