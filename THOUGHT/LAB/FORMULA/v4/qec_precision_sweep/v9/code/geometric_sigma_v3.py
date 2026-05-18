"""Geometric sigma v3: Use syndrome density participation ratio.

Theory: sigma measures how well the code compresses the error space.
At each (p,d), the syndrome density matrix has a participation ratio PR.
If PR << d (low effective rank), sigma is large (code is compressing well).
If PR ≈ d (full rank), sigma → 1.0 (no compression beyond physical limits).

Formula: sigma_geo = exp(Df_coord / max(PR, 1))
This captures the geometric compression ratio of the error space.
"""
import hashlib, json, math, sys, time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pymatching, stim

QGT_PATH = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_ALIGNMENT\qgt_lib\python")
sys.path.insert(0, str(QGT_PATH))
from qgt import participation_ratio

ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\qec_precision_sweep")
RESULTS = ROOT / "v9" / "results"

def t(d): return (d - 1) // 2

@dataclass(frozen=True)
class Cond:
    basis: str; distance: int; rounds: int; physical_error_rate: float
    shots: int; seed: int; geom: str

def seed(base, basis, d, p, geom):
    key = f"geo|{base}|{geom}|{basis}|{d}|{p:.8f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

def run_small_sample(c: Cond):
    """Run small sample to get syndrome density matrix."""
    task = f"surface_code:{c.geom}_memory_{c.basis}"
    circ = stim.Circuit.generated(
        task, distance=c.distance, rounds=c.rounds,
        after_clifford_depolarization=c.physical_error_rate,
        after_reset_flip_probability=c.physical_error_rate,
        before_measure_flip_probability=c.physical_error_rate,
        before_round_data_depolarization=c.physical_error_rate,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circ.compile_detector_sampler(seed=c.seed).sample(c.shots, separate_observables=True)
    pred = m.decode_batch(det)
    errs = np.any(pred != obs, axis=1)
    nc = int(np.count_nonzero(errs))
    pL = (nc + 0.5) / (c.shots + 1.0)
    return {
        "logical_error_count": nc,
        "logical_error_rate": nc / c.shots,
        "log_suppression": math.log(max(c.physical_error_rate / pL, 1e-60)),
        "syndrome_density": float(np.mean(det)),
        "detection_matrix": det.astype(np.float32),  # (shots, n_detectors)
        "num_detectors": circ.num_detectors,
    }

print("=== GEOMETRIC SIGMA v3: Syndrome PR ===\n")

# Small sample for syndrome PR (5000 shots is enough for covariance)
ps = [0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04]
ds = [3, 5, 7, 9, 11, 13, 15]
shots_pr = 5000
geom = "rotated"

print(f"Running syndrome PR samples: {len(ds)} distances x {len(ps)} rates x 2 bases x {shots_pr} shots")
print(f"Total: {len(ds)*len(ps)*2} conditions (~{len(ds)*len(ps)*2*5} seconds)\n")

pr_data = {}
for d in ds:
    for p_val in ps:
        cond_x = Cond(basis="x", distance=d, rounds=d, physical_error_rate=p_val,
                       shots=shots_pr, seed=seed(20260517, "x", d, p_val, geom), geom=geom)
        cond_z = Cond(basis="z", distance=d, rounds=d, physical_error_rate=p_val,
                       shots=shots_pr, seed=seed(20260517, "z", d, p_val, geom), geom=geom)
        
        r_x = run_small_sample(cond_x)
        r_z = run_small_sample(cond_z)
        
        # Compute PR on combined detection matrices
        det_combined = np.vstack([r_x["detection_matrix"], r_z["detection_matrix"]])
        # Center and compute participation ratio
        det_centered = det_combined - det_combined.mean(axis=0)
        pr_val = participation_ratio(det_centered)
        
        # Also compute PR on the covariance of the syndrome
        cov = np.cov(det_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        sum_lambda = np.sum(eigenvalues)
        sum_lambda_sq = np.sum(eigenvalues**2)
        pr_cov = (sum_lambda**2)/sum_lambda_sq if sum_lambda_sq > 1e-20 else 1.0
        
        # Average logical error rate from both bases
        avg_lr = (r_x["logical_error_rate"] + r_z["logical_error_rate"]) / 2.0
        avg_ls = (r_x["log_suppression"] + r_z["log_suppression"]) / 2.0
        avg_sd = (r_x["syndrome_density"] + r_z["syndrome_density"]) / 2.0
        total_errs = r_x["logical_error_count"] + r_z["logical_error_count"]
        
        pr_data[(p_val, d)] = {
            "p": p_val, "d": d, "Df": t(d),
            "PR_det": pr_val, "PR_cov": pr_cov,
            "logR": avg_ls, "syndrome_density": avg_sd,
            "total_errors": total_errs, "num_detectors": r_x["num_detectors"],
        }
        print(f"  d={d:2d} p={p_val:.4f}: PR={pr_val:.2f} PR_cov={pr_cov:.2f} n_det={r_x['num_detectors']} errs={total_errs}")

# Compute geometric sigma: sigma_geo = exp(Df_coord / max(PR, 1))
# If PR is small (code compresses well), sigma is large.
# If PR is large (near full rank), sigma approaches 1.
print(f"\n--- Sigma from syndrome PR ---")
sigma_geo_v3 = {}
for p_val in ps:
    pr_vals = [pr_data[(p_val, d)]["PR_cov"] for d in ds if (p_val, d) in pr_data]
    if not pr_vals: continue
    avg_pr = np.mean(pr_vals)
    sigma_geo_v3[p_val] = math.exp(t(max(ds)) / max(avg_pr, 1.0))
    print(f"  p={p_val:.4f}: avg_PR={avg_pr:.2f} sigma_geo={sigma_geo_v3[p_val]:.4f}")

# Also try: sigma_geo = exp(1/PR) per distance level
print(f"\n--- Per-distance sigma from syndrome PR ---")
sigma_geo_v3b = {}
for d in ds:
    for p_val in ps:
        if (p_val, d) not in pr_data: continue
        pr_val = pr_data[(p_val, d)]["PR_cov"]
        key = f"{p_val:.4f}_{d}"
        sigma_geo_v3b[key] = math.exp(1.0 / max(pr_val, 0.5))

# Now evaluate alpha with geometric sigma
print(f"\n--- Alpha evaluation ---")
eps = 1e-60
from sklearn.metrics import mean_absolute_error, r2_score

def calibrate_E(rows, smap, train_dists):
    ests = []
    for r in rows:
        d = int(r["d"])
        if d not in train_dists: continue
        sp = smap.get(r["p"])
        if sp is None: continue
        gs = max(float(r["syndrome_density"]), eps)
        ests.append(r["logR"] + math.log(gs) - r["Df"] * math.log(max(sp, eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

def evaluate(rows, smap, E_val, heldout):
    yp=[]; ya=[]
    for r in rows:
        d = int(r["d"])
        if d not in heldout: continue
        if r["total_errors"] < 10: continue
        sp = smap.get(r["p"], 1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**r["Df"]), eps))
        yp.append(lrp); ya.append(r["logR"])
    yp=np.array(yp); ya=np.array(ya)
    if len(yp) < 3: return float('nan'), float('nan'), float('nan'), float('nan')
    mae=float(mean_absolute_error(ya,yp))
    r2v=float(r2_score(ya,yp))
    A=np.column_stack([yp,np.ones_like(yp)])
    coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    return float(coef[0]), float(coef[1]), mae, r2v

train_set = {3,5,7}
heldout_set = {9,11,13,15}

# Convert pr_data to rows
geo_rows = list(pr_data.values())

E_geo = calibrate_E(geo_rows, sigma_geo_v3, train_set)
alpha_geo, beta_geo, mae_geo, r2_geo = evaluate(geo_rows, sigma_geo_v3, E_geo, heldout_set)
print(f"  Geometric (avg PR):  alpha={alpha_geo:.4f}  beta={beta_geo:.4f}  R2={r2_geo:.4f}  MAE={mae_geo:.4f}")

# Also try with the per-distance sigma
sigma_v3b_flat = {}
for key, val in sigma_geo_v3b.items():
    p_str, d_str = key.split("_")
    p_val = float(p_str)
    if p_val not in sigma_v3b_flat:
        sigma_v3b_flat[p_val] = val  # use first distance's value (d=3)

E_geo2 = calibrate_E(geo_rows, sigma_v3b_flat, train_set)

def evaluate_v3b(rows, smap, E_val, heldout):
    yp=[]; ya=[]
    for r in rows:
        d = int(r["d"])
        if d not in heldout: continue
        if r["total_errors"] < 10: continue
        key = f"{r['p']:.4f}_{d}"
        sp = smap.get(key, 1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**r["Df"]), eps))
        yp.append(lrp); ya.append(r["logR"])
    yp=np.array(yp); ya=np.array(ya)
    if len(yp) < 3: return float('nan'), float('nan'), float('nan'), float('nan')
    mae=float(mean_absolute_error(ya,yp))
    r2v=float(r2_score(ya,yp))
    A=np.column_stack([yp,np.ones_like(yp)])
    coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    return float(coef[0]), float(coef[1]), mae, r2v

alpha_geo2, beta_geo2, mae_geo2, r2_geo2 = evaluate_v3b(geo_rows, sigma_geo_v3b, E_geo2, heldout_set)
print(f"  Geometric (per-d):   alpha={alpha_geo2:.4f}  beta={beta_geo2:.4f}  R2={r2_geo2:.4f}  MAE={mae_geo2:.4f}")

# Also try: sigma_geo = num_detectors / PR (direct compression ratio)
print(f"\n--- Sigma from detector compression ratio ---")
sigma_comp = {}
for p_val in ps:
    comp_vals = []
    for d in ds:
        if (p_val, d) not in pr_data: continue
        det_count = pr_data[(p_val, d)]["num_detectors"]
        pr_val = pr_data[(p_val, d)]["PR_cov"]
        comp_ratio = det_count / max(pr_val, 1.0)
        comp_vals.append(comp_ratio)
    if comp_vals:
        sigma_comp[p_val] = math.exp(math.log(max(np.mean(comp_vals), 1.0)))

E_comp = calibrate_E(geo_rows, sigma_comp, train_set)
alpha_comp, beta_comp, mae_comp, r2_comp = evaluate(geo_rows, sigma_comp, E_comp, heldout_set)
print(f"  Detector compression: alpha={alpha_comp:.4f}  beta={beta_comp:.4f}  R2={r2_comp:.4f}  MAE={mae_comp:.4f}")

print(f"\n=== SUMMARY ===")
print(f"  Empirical sigma:  alpha = 0.7482 (baseline)")
print(f"  Geo PR (avg):     alpha = {alpha_geo:.4f}")
print(f"  Geo PR (per-d):   alpha = {alpha_geo2:.4f}")
print(f"  Detector comp:    alpha = {alpha_comp:.4f}")
