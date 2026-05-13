"""v9 MEAS grad_S sweep: test alternative grad_S definitions for measurement-heavy noise."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
def t(d): return (d-1)//2
def load(src):
    for f in Path(src).glob("sweep.json"): return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(src)

def pool(rows):
    g = defaultdict(list)
    for r in rows:
        g[(float(r["physical_error_rate"]),int(r["distance"]))].append(r)
    return [{"p":p,"d":d,"logR":float(np.mean([r["log_suppression"] for r in grp])),
             "syn":float(np.mean([r["syndrome_density"] for r in grp]))} for (p,d),grp in g.items()]

def sigma_map(rows, train):
    g=defaultdict(list)
    for r in rows:
        if r["d"] in train: g[r["p"]].append((t(r["d"]),r["logR"]))
    sm={}
    for p,pts in g.items():
        if len(pts)<2: continue
        pts.sort(); ts=np.array([x[0] for x in pts]); ls=np.array([x[1] for x in pts])
        A=np.column_stack([ts,np.ones_like(ts)])
        sm[p]=math.exp(float(np.linalg.lstsq(A,ls,rcond=None)[0][0]))
    return sm

def calibrate_E(rows, train, sm, gs_fn):
    ests=[]; eps=1e-60
    for r in rows:
        if r["d"] not in train: continue
        sp=sm.get(r["p"]); gs=max(gs_fn(r),eps)
        if sp is None: continue
        ests.append(r["logR"]+math.log(gs)-t(r["d"])*math.log(max(sp,eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

def evaluate(rows, sm, E, holdout, gs_fn):
    eps=1e-60; yp=[]; ya=[]
    for r in rows:
        if r["d"] not in holdout: continue
        sp=sm.get(r["p"],1.0); gs=max(gs_fn(r),eps)
        lrp=math.log(max((E/gs)*(sp**t(r["d"])),eps))
        yp.append(lrp); ya.append(r["logR"])
    yp=np.array(yp); ya=np.array(ya)
    A=np.column_stack([yp,np.ones_like(yp)])
    coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    return {"mae":float(mean_absolute_error(ya,yp)),"r2":float(r2_score(ya,yp)),
            "alpha":float(coef[0]),"beta":float(coef[1])}

def main():
    a=argparse.ArgumentParser(); a.add_argument("--source-dir",required=True); args=a.parse_args()
    payload=load(args.source_dir)
    rows=[r for r in pool(payload["conditions"]) if r["p"]!=0.0005]
    train={3,5,7}; sm=sigma_map(rows,train)

    gs_defs={
        "syn": lambda r: r["syn"],
        "p": lambda r: r["p"],
        "sqrt_syn": lambda r: max(r["syn"],1e-60)**0.5,
        "p_eff": lambda r: max(r["p"]*1.5, 1e-60),  # rescale p upward for meas-heavy
        "syn_scaled": lambda r: max(r["syn"]*1.5, 1e-60),
        "p_mid": lambda r: 0.5*(r["p"]+r["syn"]),
    }

    for name,fn in gs_defs.items():
        E=calibrate_E(rows,train,sm,fn)
        ev=evaluate(rows,sm,E,{9,11},fn)
        print(f"[grad_S={name:12s}] E={E:.4f}  MAE={ev['mae']:.4f}  R2={ev['r2']:.4f}  alpha={ev['alpha']:.4f}  beta={ev['beta']:.4f}")

if __name__=="__main__":
    raise SystemExit(main())
