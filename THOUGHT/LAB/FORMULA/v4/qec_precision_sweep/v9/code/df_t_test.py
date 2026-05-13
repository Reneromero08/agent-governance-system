"""Quick test: fidelity-factor sigma with Df=t instead of Df=d."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def t(d): return (d - 1) // 2

def load(src):
    for pat in ["sweep.json"]:
        for f in Path(src).glob(pat):
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(src)

def pool(rows):
    g = defaultdict(list)
    for r in rows:
        g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
    return [{"physical_error_rate":p,"distance":d,
             "log_suppression":float(np.mean([r["log_suppression"] for r in grp])),
             "syndrome_density":float(np.mean([r["syndrome_density"] for r in grp]))}
            for (p,d),grp in g.items()]

def sigma_and_E(rows, train_dists):
    groups = defaultdict(list)
    for r in rows:
        d = int(r["distance"])
        if d in train_dists:
            groups[float(r["physical_error_rate"])].append((t(d), r["log_suppression"]))

    smap = {}; eps = 1e-60; ests = []
    for p, pts in groups.items():
        if len(pts) < 2: continue
        pts.sort(); ts_arr = np.array([pt[0] for pt in pts]); ls = np.array([pt[1] for pt in pts])
        A = np.column_stack([ts_arr, np.ones_like(ts_arr)])
        coef = np.linalg.lstsq(A, ls, rcond=None)[0]
        smap[p] = math.exp(float(coef[0]))

    for r in rows:
        d = int(r["distance"])
        if d not in train_dists: continue
        p = float(r["physical_error_rate"]); sp = smap.get(p)
        if sp is None: continue
        gs = max(float(r["syndrome_density"]), eps)
        ests.append(r["log_suppression"] + math.log(gs) - t(d) * math.log(max(sp, eps)))
    E = math.exp(float(np.median(ests))) if ests else 1.0
    return smap, E

def evaluate(rows, smap, E_val, heldout):
    eps = 1e-60; yp=[]; ya=[]
    for r in rows:
        d = int(r["distance"])
        if d not in heldout: continue
        p = float(r["physical_error_rate"]); sp = smap.get(p,1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**t(d)), eps))
        yp.append(lrp); ya.append(r["log_suppression"])
    yp=np.array(yp); ya=np.array(ya)
    A=np.column_stack([yp,np.ones_like(yp)])
    coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    return {"mae":float(mean_absolute_error(ya,yp)),"r2":float(r2_score(ya,yp)),
            "alpha":float(coef[0]),"beta":float(coef[1])}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source-dir",required=True)
    args=a.parse_args()
    payload=load(args.source_dir)
    nm=payload["config"]["noise_model"]
    rows=[r for r in pool(payload["conditions"]) if float(r["physical_error_rate"])!=0.0005]
    train={3,5,7}

    smap,E=sigma_and_E(rows,train)
    print(f"E={E:.4f}")
    for p in sorted(smap):
        print(f"  p={p:.4f} sigma={smap[p]:.4f}")

    for hname,hset in [("d9+d11",{9,11}),("d9",{9}),("d11",{11})]:
        ev=evaluate(rows,smap,E,hset)
        print(f"[{hname}] MAE={ev['mae']:.4f} R2={ev['r2']:.4f} alpha={ev['alpha']:.4f} beta={ev['beta']:.4f}")

if __name__=="__main__":
    raise SystemExit(main())
