"""Combine v8 data with new d=13,15 sweep and re-analyze."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\qec_precision_sweep")
V8_SWEEP = ROOT / "v8" / "results" / "v8_depol" / "sweep.json"
HIGH_D_SWEEP = ROOT / "v9" / "results" / "20260517T214104Z" / "sweep.json"
OUT_DIR = ROOT / "v9" / "results" / "v9_extended"

def t(d): return (d - 1) // 2

def pool(rows):
    g = defaultdict(list)
    for r in rows:
        g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
    return [{"physical_error_rate":p, "distance":d,
             "log_suppression":float(np.mean([r["log_suppression"] for r in grp])),
             "syndrome_density":float(np.mean([r["syndrome_density"] for r in grp]))}
            for (p,d), grp in g.items()]

def measure_sigma(rows, train_dists):
    groups = defaultdict(list)
    for r in rows:
        d = int(r["distance"])
        if d in train_dists:
            groups[float(r["physical_error_rate"])].append((t(d), r["log_suppression"]))
    smap = {}
    for p, pts in groups.items():
        if len(pts) < 2: continue
        pts.sort()
        ts_arr = np.array([pt[0] for pt in pts])
        ls = np.array([pt[1] for pt in pts])
        A = np.column_stack([ts_arr, np.ones_like(ts_arr)])
        smap[p] = math.exp(float(np.linalg.lstsq(A, ls, rcond=None)[0][0]))
    return smap

def calibrate_E(rows, train_dists, smap):
    eps = 1e-60; ests = []
    for r in rows:
        d = int(r["distance"])
        if d not in train_dists: continue
        p = float(r["physical_error_rate"]); sp = smap.get(p)
        if sp is None: continue
        gs = max(float(r["syndrome_density"]), eps)
        ests.append(r["log_suppression"] + math.log(gs) - t(d) * math.log(max(sp, eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

def evaluate(rows, smap, E_val, heldout):
    eps = 1e-60; yp=[]; ya=[]; pp=[]
    for r in rows:
        d = int(r["distance"])
        if d not in heldout: continue
        p = float(r["physical_error_rate"]); sp = smap.get(p, 1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**t(d)), eps))
        yp.append(lrp); ya.append(r["log_suppression"])
        pp.append({"p":p,"d":d,"Df":t(d),"sigma":sp,"logR_actual":r["log_suppression"],"logR_pred":lrp})
    yp=np.array(yp); ya=np.array(ya)
    mae=float(mean_absolute_error(ya,yp))
    r2v=float(r2_score(ya,yp))
    A=np.column_stack([yp,np.ones_like(yp)])
    coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    alpha=float(coef[0]); beta=float(coef[1])
    rng=np.random.RandomState(20260517)
    maes=[]; alphas=[]
    for _ in range(1000):
        idx=rng.choice(len(yp),size=len(yp),replace=True)
        maes.append(float(mean_absolute_error(ya[idx],yp[idx])))
        A2=np.column_stack([yp[idx],np.ones_like(yp[idx])])
        alphas.append(float(np.linalg.lstsq(A2,ya[idx],rcond=None)[0][0]))
    maes=np.array(maes); alphas=np.array(alphas)
    return {"mae":mae,"r2":r2v,"alpha":alpha,"beta":beta,
            "mae_ci95":[float(np.percentile(maes,2.5)),float(np.percentile(maes,97.5))],
            "alpha_ci95":[float(np.percentile(alphas,2.5)),float(np.percentile(alphas,97.5))],
            "points":pp}

# Load and merge
v8 = json.loads(V8_SWEEP.read_text(encoding="utf-8"))
hd = json.loads(HIGH_D_SWEEP.read_text(encoding="utf-8"))

all_conds = v8["conditions"] + hd["conditions"]
rows = pool(all_conds)

train = {3,5,7}
smap = measure_sigma(rows, train)
E_val = calibrate_E(rows, train, smap)

# Evaluate on all holdout combinations
evals = {}
for label, hset in [
    ("d9+d11+d13+d15", {9,11,13,15}),
    ("d9+d11", {9,11}),
    ("d13+d15", {13,15}),
    ("d9", {9}), ("d11", {11}), ("d13", {13}), ("d15", {15}),
]:
    evals[label] = evaluate(rows, smap, E_val, hset)

OUT_DIR.mkdir(parents=True, exist_ok=True)

result = {
    "run_id": "v9_extended",
    "source_runs": [v8["run_id"], hd["run_id"]],
    "noise_model": "depol",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "E": E_val,
    "train": sorted(train),
    "holdout_all": [9,11,13,15],
    "sigma_map": {str(k):v for k,v in sorted(smap.items())},
    "evaluations": evals,
}
(OUT_DIR / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

# Print results
print(f"\nE = {E_val:.6f}")
print(f"{'Holdout':<20s} {'Alpha':>8s} {'95% CI':>20s} {'R2':>8s} {'MAE':>8s}")
print("-" * 68)
for label, ev in evals.items():
    ci = f"[{ev['alpha_ci95'][0]:.4f}, {ev['alpha_ci95'][1]:.4f}]"
    print(f"{label:<20s} {ev['alpha']:8.4f} {ci:>20s} {ev['r2']:8.4f} {ev['mae']:8.4f}")

# Also print per-distance point details
print(f"\n--- Per-point predictions ---")
for label, ev in [("d9+d11+d13+d15", evals["d9+d11+d13+d15"])]:
    print(f"\n{label}:")
    print(f"{'p':>8s} {'d':>4s} {'Df':>4s} {'sigma':>8s} {'logR_act':>10s} {'logR_pred':>10s} {'error':>10s}")
    print("-" * 62)
    for pt in sorted(ev["points"], key=lambda x: (x["p"], x["d"])):
        err = abs(pt["logR_actual"] - pt["logR_pred"])
        print(f"{pt['p']:8.4f} {pt['d']:4d} {pt['Df']:4d} {pt['sigma']:8.4f} {pt['logR_actual']:10.4f} {pt['logR_pred']:10.4f} {err:10.4f}")
