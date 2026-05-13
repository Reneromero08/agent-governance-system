"""Curvature residual analysis."""
import json, math, numpy as np
from collections import defaultdict

def t(d): return (d-1)//2
def pool(rows):
    g=defaultdict(list)
    for r in rows: g[(float(r["physical_error_rate"]),int(r["distance"]))].append(r)
    return [{"p":p,"d":d,"logR":float(np.mean([r["log_suppression"] for r in grp])),
             "syn":float(np.mean([r["syndrome_density"] for r in grp]))} for (p,d),grp in g.items()]

for nm, src in [("DEPOL","THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v8/results/v8_depol/sweep.json"),
                 ("MEAS","THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v8/results/v8_meas/sweep.json")]:
    data=json.load(open(src)); rows=[r for r in pool(data["conditions"]) if r["p"]!=0.0005]
    train={3,5,7}; eps=1e-60

    g=defaultdict(list)
    for r in rows:
        if r["d"] in train: g[r["p"]].append((t(r["d"]),r["logR"]))
    sm={}
    for p,pts in g.items():
        if len(pts)<2: continue
        pts.sort(); ts=np.array([x[0] for x in pts]); ls=np.array([x[1] for x in pts])
        A=np.column_stack([ts,np.ones_like(ts)]); sm[p]=math.exp(float(np.linalg.lstsq(A,ls,rcond=None)[0][0]))

    ests=[]
    for r in rows:
        if r["d"] not in train: continue
        sp=sm.get(r["p"]);
        if sp is None: continue
        gs=float(r["syn"])**0.5
        ests.append(r["logR"]+math.log(gs)-t(r["d"])*math.log(max(sp,eps)))
    E=math.exp(float(np.median(ests)))

    print(f"\n[{nm}] Residuals (actual - pred):")
    residuals=[]
    for r in rows:
        if r["d"] not in {9,11}: continue
        sp=sm.get(r["p"],1.0); gs=float(r["syn"])**0.5
        pred=math.log(E)-math.log(gs)+t(r["d"])*math.log(max(sp,eps))
        resid=r["logR"]-pred
        residuals.append((t(r["d"]),r["p"],resid))
        print(f"  p={r['p']:.4f} d={r['d']} t={t(r['d'])} actual={r['logR']:.4f} pred={pred:.4f} resid={resid:+.4f}")

    ts=np.array([x[0] for x in residuals])
    rs=np.array([x[2] for x in residuals])
    for ti in sorted(set(ts)):
        mask=ts==ti
        avg=np.mean(rs[mask]); std=np.std(rs[mask])
        print(f"  t={int(ti)}: mean_resid={avg:+.4f} std={std:.4f}")
