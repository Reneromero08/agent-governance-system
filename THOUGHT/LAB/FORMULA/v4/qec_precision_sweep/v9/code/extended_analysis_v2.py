"""Re-analyze with statistical resolution filter. Exclude points with <10 observed errors."""
import json, math
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\qec_precision_sweep")
V8_SWEEP = ROOT / "v8" / "results" / "v8_depol" / "sweep.json"
HIGH_D_SWEEP = ROOT / "v9" / "results" / "20260517T214104Z" / "sweep.json"
OUT_DIR = ROOT / "v9" / "results" / "v9_extended"

def t(d): return (d - 1) // 2

# Pool with shot count tracking
def pool_with_stats(all_rows):
    g = defaultdict(list)
    for r in all_rows:
        g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
    result = []
    for (p,d), grp in g.items():
        total_errs = sum(r["logical_error_count"] for r in grp)
        total_shots = sum(r["shots"] for r in grp)
        result.append({
            "physical_error_rate": p,
            "distance": d,
            "log_suppression": float(np.mean([r["log_suppression"] for r in grp])),
            "syndrome_density": float(np.mean([r["syndrome_density"] for r in grp])),
            "total_errors": total_errs,
            "total_shots": total_shots,
        })
    return result

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

def evaluate_filtered(rows, smap, E_val, heldout, min_errors=10):
    eps = 1e-60; yp=[]; ya=[]; pp=[]
    skipped = 0
    for r in rows:
        d = int(r["distance"])
        if d not in heldout: continue
        if r.get("total_errors", 999) < min_errors:
            skipped += 1
            continue
        p = float(r["physical_error_rate"]); sp = smap.get(p, 1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**t(d)), eps))
        yp.append(lrp); ya.append(r["log_suppression"])
        pp.append({"p":p,"d":d,"Df":t(d),"sigma":sp,"logR_actual":r["log_suppression"],
                    "logR_pred":lrp,"errors":r.get("total_errors",0)})
    yp=np.array(yp); ya=np.array(ya)
    if len(yp) < 3:
        return {"mae":float('nan'),"r2":float('nan'),"alpha":float('nan'),"beta":float('nan'),
                "mae_ci95":[0,0],"alpha_ci95":[0,0],"points":pp,"skipped":skipped}
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
            "points":pp,"skipped":skipped,"n_points":len(yp)}

# Load
v8 = json.loads(V8_SWEEP.read_text(encoding="utf-8"))
hd = json.loads(HIGH_D_SWEEP.read_text(encoding="utf-8"))
all_conds = v8["conditions"] + hd["conditions"]
rows = pool_with_stats(all_conds)

# Show error statistics per condition
print("Error statistics (raw counts):")
print(f"{'p':>8s} {'d':>4s} {'total_errs':>12s} {'total_shots':>12s} {'pL_laplace':>14s}")
print("-" * 56)
for r in sorted(rows, key=lambda x: (x["physical_error_rate"], x["distance"])):
    pL = (r["total_errors"] + 0.5) / (r["total_shots"] + 1.0)
    flag = " *** UNDERSAMPLED" if r["total_errors"] < 10 else ""
    print(f"{r['physical_error_rate']:8.4f} {r['distance']:4d} {r['total_errors']:12d} {r['total_shots']:12d} {pL:14.6e}{flag}")

train = {3,5,7}
smap = measure_sigma(rows, train)
E_val = calibrate_E(rows, train, smap)

print(f"\nE calibrated: {E_val:.6f}")
print(f"\n--- ALPHA STABILITY (min 10 errors per condition) ---")
print(f"{'Holdout':<24s} {'Alpha':>8s} {'95% CI':>22s} {'R2':>8s} {'MAE':>8s} {'N':>4s} {'Skipped':>8s}")
print("-" * 90)

best_alpha = None
for min_err in [10]:
    for label, hset in [
        ("d9+d11+d13+d15 (>10 errs)", {9,11,13,15}),
        ("d9+d11 (>10 errs)", {9,11}),
        ("d13+d15 (>10 errs)", {13,15}),
        ("d9+d11+d13 (no d15)", {9,11,13}),
        ("d9+d11+d15 (no d13)", {9,11,15}),
    ]:
        ev = evaluate_filtered(rows, smap, E_val, hset, min_errors=min_err)
        ci = f"[{ev['alpha_ci95'][0]:.4f}, {ev['alpha_ci95'][1]:.4f}]" if not math.isnan(ev['alpha']) else "N/A"
        n = ev.get('n_points', '?')
        sk = ev.get('skipped', '?')
        print(f"{label:<24s} {ev['alpha']:8.4f} {ci:>22s} {ev['r2']:8.4f} {ev['mae']:8.4f} {str(n):>4s} {str(sk):>8s}")
        if "d9+d11+d13+d15" in label and not math.isnan(ev['alpha']):
            best_alpha = ev

# Also test: exclude the lowest 2 p values entirely for d=13,15
print(f"\n--- EXCLUDING p<0.002 for d>=13 ---")
rows_p002 = [r for r in rows if not (r["distance"] >= 13 and r["physical_error_rate"] < 0.002)]
smap2 = measure_sigma(rows_p002, train)
E2 = calibrate_E(rows_p002, train, smap2)
for label, hset in [
    ("d9+d11+d13+d15 (p>=0.002)", {9,11,13,15}),
    ("d13+d15 (p>=0.002)", {13,15}),
]:
    ev = evaluate_filtered(rows_p002, smap2, E2, hset, min_errors=0)
    ci = f"[{ev['alpha_ci95'][0]:.4f}, {ev['alpha_ci95'][1]:.4f}]" if not math.isnan(ev['alpha']) else "N/A"
    print(f"{label:<30s} {ev['alpha']:8.4f} {ci:>22s} {ev['r2']:8.4f} {ev['mae']:8.4f}")

# Also try: include d=13,15 only in training (recalibrate sigma)
print(f"\n--- WITH d=13 IN TRAINING (train=3,5,7,13, test=9,11,15) ---")
train_wide = {3,5,7,13}
smap3 = measure_sigma(rows, train_wide)
E3 = calibrate_E(rows, train_wide, smap3)
for label, hset in [
    ("d9+d11+d15 (train 3,5,7,13)", {9,11,15}),
    ("d9+d11 (train 3,5,7,13)", {9,11}),
]:
    ev = evaluate_filtered(rows, smap3, E3, hset, min_errors=0)
    ci = f"[{ev['alpha_ci95'][0]:.4f}, {ev['alpha_ci95'][1]:.4f}]" if not math.isnan(ev['alpha']) else "N/A"
    print(f"{label:<30s} {ev['alpha']:8.4f} {ci:>22s} {ev['r2']:8.4f} {ev['mae']:8.4f}")
