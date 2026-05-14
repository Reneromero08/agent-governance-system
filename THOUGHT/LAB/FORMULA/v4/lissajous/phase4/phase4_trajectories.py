"""Lissajous Phase 4: Detection-event trajectory analysis.

Samples surface code detection events, finds most correlated stabilizer pair,
analyzes joint detection distribution across p values. Tests whether the
Lissajous figure changes character crossing the threshold.
"""
import stim, numpy as np, json, math, hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase4"

def build_and_sample(d, p, shots=10000, seed=42):
    """Build circuit, sample detectors, return detection matrix."""
    base = stim.Circuit.generated(f"surface_code:rotated_memory_x", distance=d, rounds=d,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)
    sampler = base.compile_detector_sampler(seed=seed)
    detectors, _ = sampler.sample(shots, separate_observables=True)
    return detectors  # [shots, num_detectors]

def correlation_from_detections(detections):
    """Build empirical correlation matrix from detection samples."""
    n_shots, n_det = detections.shape
    # Correlation: P(i fires and j fires) - P(i fires)*P(j fires)
    p_i = detections.mean(axis=0)  # [n_det]
    p_ij = (detections.T @ detections) / n_shots  # [n_det, n_det]
    
    M = np.zeros((n_det, n_det))
    for i in range(n_det):
        for j in range(n_det):
            if i == j:
                M[i,j] = p_i[i] * (1 - p_i[i])  # variance
            else:
                corr = p_ij[i,j] - p_i[i] * p_i[j]
                M[i,j] = corr
    return M, p_i, p_ij

def find_best_pair(M, p_i):
    """Find detector pair with strongest normalized correlation."""
    n = M.shape[0]
    best_score = -1
    best_pair = (0,1)
    for i in range(n):
        for j in range(i+1, n):
            if p_i[i] * (1-p_i[i]) > 0 and p_i[j] * (1-p_i[j]) > 0:
                score = abs(M[i,j]) / math.sqrt(p_i[i]*(1-p_i[i])*p_i[j]*(1-p_i[j]))
                if score > best_score:
                    best_score = score
                    best_pair = (i,j)
    return best_pair, best_score

def lissajous_metrics(detections, i, j):
    """Characterize the joint distribution of two detectors."""
    # Joint distribution
    n_shots = detections.shape[0]
    joint = np.zeros((2,2))
    for s in range(n_shots):
        joint[int(detections[s,i]), int(detections[s,j])] += 1
    joint /= n_shots
    
    # Mutual information I(D_i; D_j)
    p_i = joint.sum(axis=1)
    p_j = joint.sum(axis=0)
    mi = 0.0
    for a in range(2):
        for b in range(2):
            if joint[a,b] > 0 and p_i[a] > 0 and p_j[b] > 0:
                mi += joint[a,b] * math.log(joint[a,b] / (p_i[a] * p_j[b]))
    
    # Determinant of 2x2 joint — a measure of "closedness" 
    # Det > 0 when P(0,0)*P(1,1) > P(0,1)*P(1,0) (positive correlation)
    det = joint[0,0]*joint[1,1] - joint[0,1]*joint[1,0]
    
    # Norm of off-diagonal vs diagonal in joint
    # "Closedness": how diagonal-dominated is the joint distribution?
    diag_sum = joint[0,0] + joint[1,1]
    off_sum = joint[0,1] + joint[1,0]
    
    return {
        "joint": joint.tolist(),
        "mutual_info": float(mi),
        "determinant": float(det),
        "diagonal_ratio": float(diag_sum / max(diag_sum + off_sum, 1e-12)),
        "off_diagonal": float(off_sum),
        "p_i": float(p_i[1]),  # probability detector i fires
        "p_j": float(p_j[1]),  # probability detector j fires
    }

print("LISSAJOUS PHASE 4: Detection-Event Trajectories")
print("=" * 60)

ps = [0.001, 0.004, 0.006, 0.008, 0.010, 0.020]
d = 3
shots = 10000
seed = 20260514

results = []
for p in ps:
    print(f"\np={p:.4f}: sampling {shots} shots...", end=" ", flush=True)
    key = f"lis_{seed}_{d}_{p:.8f}".encode()
    s = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    detections = build_and_sample(d, p, shots, s)
    n_det = detections.shape[1]
    print(f"{n_det} detectors", end=" ", flush=True)
    
    # Correlation matrix from samples
    M, p_i, p_ij = correlation_from_detections(detections)
    best_pair, score = find_best_pair(M, p_i)
    i, j = best_pair
    print(f"best_pair=({i},{j}) corr={score:.4f}", end=" ", flush=True)
    
    # Lissajous metrics
    metrics = lissajous_metrics(detections, i, j)
    metrics["p"] = p
    metrics["pair"] = (i,j)
    metrics["correlation"] = score
    metrics["n_shots"] = shots
    
    # Also check top-3 pairs
    print(f"MI={metrics['mutual_info']:.4f} det={metrics['determinant']:.6f} diag={metrics['diagonal_ratio']:.4f}", flush=True)
    results.append(metrics)
    
    # Report joint distribution
    jt = metrics["joint"]
    print(f"  Joint: P(0,0)={jt[0][0]:.4f} P(1,0)={jt[1][0]:.4f} P(0,1)={jt[0][1]:.4f} P(1,1)={jt[1][1]:.4f}")
    print(f"  P(i=1)={metrics['p_i']:.4f} P(j=1)={metrics['p_j']:.4f}")

# Phase 5: Does Lissajous closure predict sigma?
print(f"\n{'='*60}")
print("PHASE 5: Lissajous closure vs sigma")
print(f"{'='*60}")

# Load sigma_emp
def t(d): return (d-1)//2
sweep_data = json.load(open(ROOT.parent / "qec_precision_sweep" / "v8" / "results" / "v8_depol" / "sweep.json"))
g = defaultdict(list)
for r in sweep_data["conditions"]:
    g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r["log_suppression"])
tg = defaultdict(list)
for (p_val, d_val), vs in g.items():
    if p_val == 0.0005: continue
    if d_val in {3,5,7}: tg[p_val].append((t(d_val), float(np.mean(vs))))
sigma_emp = {}
for p_val, pts in tg.items():
    if len(pts) < 2: continue
    pts.sort(); ts=np.array([x[0] for x in pts]); ls=np.array([x[1] for x in pts])
    A=np.column_stack([ts,np.ones_like(ts)])
    sigma_emp[p_val] = math.exp(float(np.linalg.lstsq(A,ls,rcond=None)[0][0]))

print(f"\n{'p':>8s} {'sigma':>8s} {'MI':>8s} {'det':>10s} {'diag':>8s} {'corr':>8s}")
clos_vals = []
sig_vals = []
for r in results:
    p_val = r["p"]
    sig = sigma_emp.get(p_val, 1.0)
    # Closure metrics to test
    mi = r["mutual_info"]
    det = r["determinant"]
    diag = r["diagonal_ratio"]
    corr = r["correlation"]
    print(f"{p_val:8.4f} {sig:8.4f} {mi:8.4f} {det:10.6f} {diag:8.4f} {corr:8.4f}")

# Test each closure metric against sigma
for metric_name, metric_key in [("MI", "mutual_info"), ("det", "determinant"), 
                                  ("diag", "diagonal_ratio"), ("corr", "correlation")]:
    xs = np.array([r[metric_key] for r in results])
    ys = np.array([sigma_emp.get(r["p"], 1.0) for r in results])
    r_pearson = np.corrcoef(xs, ys)[0,1]
    print(f"\n{metric_name} vs sigma: r={r_pearson:+.4f}")
    if abs(r_pearson) > 0.5:
        print(f"  SIGNIFICANT: Lissajous {metric_name} predicts sigma!")
    else:
        print(f"  Weak or no correlation.")

# Save results
out_path = OUT / "phase4_results.json"
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved: {out_path}")
