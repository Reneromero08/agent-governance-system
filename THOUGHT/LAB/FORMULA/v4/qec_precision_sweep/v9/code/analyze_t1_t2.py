"""Task 1+2 analysis: geometry test + threshold flattening."""
import json, math, numpy as np
from collections import defaultdict
from pathlib import Path

def t(d): return (d-1)//2
ROOT = Path(__file__).resolve().parent.parent.parent
V8_DEPOL = ROOT / "v8" / "results" / "v8_depol"
T1_UNROT = ROOT / "v9" / "results" / "t1_unrotated"
T2_FINE = ROOT / "v9" / "results" / "t2_thresh_fine"

def load(d):
    for f in d.glob("*.json"): return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(str(d))
def pool(rows):
    g=defaultdict(list)
    for r in rows: g[(float(r["physical_error_rate"]),int(r["distance"]))].append(r)
    return [{"p":p,"d":d,"logR":float(np.mean([r["log_suppression"] for r in grp])),
             "syn":float(np.mean([r["syndrome_density"] for r in grp])),
             "nq":grp[0].get("num_qubits",0),"nd":grp[0].get("num_detectors",0)}
            for (p,d),grp in g.items()]

def sigma_map(rows,train):
    g=defaultdict(list)
    for r in rows:
        if r["d"] in train: g[r["p"]].append((t(r["d"]),r["logR"]))
    sm={}
    for p,pts in g.items():
        if len(pts)<2: continue
        pts.sort(); ts=np.array([x[0] for x in pts]); ls=np.array([x[1] for x in pts])
        A=np.column_stack([ts,np.ones_like(ts)]); sm[p]=math.exp(float(np.linalg.lstsq(A,ls,rcond=None)[0][0]))
    return sm

def calibrate(rows,train,sm):
    eps=1e-60; ests=[]
    for r in rows:
        if r["d"] not in train: continue
        sp=sm.get(r["p"]);
        if sp is None: continue
        gs=float(r["syn"])**0.5
        ests.append(r["logR"]+math.log(gs)-t(r["d"])*math.log(max(sp,eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

# === TASK 1: Rotated vs Unrotated at t=2 ===
print("=== TASK 1: Geometry Test, t=2 ===")
for geom, src_dir in [("rotated",V8_DEPOL),("unrotated",T1_UNROT)]:
    data=load(src_dir)
    rows=pool(data["conditions"])
    rows=[r for r in rows if r["p"]!=0.0005]
    train={3,5}
    sm=sigma_map(rows,train)
    E=calibrate(rows,train,sm)

    print(f"\n{geom} — E={E:.4f}")
    print(f"  nQ={rows[0]['nq']} nD={rows[0]['nd']} (d=3)")
    for p in sorted({r["p"] for r in rows}):
        if p>0.01: break
        pts=[r for r in rows if r["p"]==p and r["d"] in {7,9}]
        for r in pts:
            sp=sm.get(p,1.0); gs=r["syn"]**0.5
            pred=math.log(max((E/gs)*(sp**t(r["d"])),1e-60))
            print(f"  p={p:.4f} d={r['d']} t={t(r['d'])}  logR={r['logR']:.4f}  pred={pred:.4f}  err={abs(r['logR']-pred):.4f}  sigma={sp:.4f}  syn={r['syn']:.4f}")

# === TASK 2: Threshold Flattening ===
print("\n=== TASK 2: Threshold Flattening ===")
data=load(T2_FINE)
rows=pool(data["conditions"])
rows=[r for r in rows if r["p"]>=0.004]
# Use formula to compute sigma and grad_S at each p, then predict flattening
for d in [3,5,7]:
    pts=[r for r in rows if r["d"]==d]
    print(f"\nd={d} t={t(d)}:")
    print(f"  {'p':>8s}  {'logR':>8s}  {'syn':>8s}  {'sqrt_syn':>8s}  {'stdQEC_slope':>12s}")
    for r in sorted(pts,key=lambda x:x["p"]):
        syn=r["syn"]; sqrt_syn=syn**0.5
        # standard QEC predicts: ln(R) = -t*ln(p) + const
        # d(ln(R))/dp = -t/p  (the slope with respect to p)
        stdqec_slope = -t(d)/r["p"]
        print(f"  {r['p']:.4f}  {r['logR']:8.4f}  {syn:8.4f}  {sqrt_syn:8.4f}  {stdqec_slope:12.4f}")
