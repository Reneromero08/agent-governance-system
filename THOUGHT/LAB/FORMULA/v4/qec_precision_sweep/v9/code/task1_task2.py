"""Task 1+2 combined: close MEAS gap and bootstrap closed-form sigma."""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]; RESULTS = ROOT / "results"
def t(d): return (d-1)//2
def load(src):
    for f in Path(src).glob("sweep.json"): return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(src)
def pool(rows):
    g=defaultdict(list)
    for r in rows: g[(float(r["physical_error_rate"]),int(r["distance"]))].append(r)
    return [{"p":p,"d":d,"logR":float(np.mean([r["log_suppression"] for r in grp])),
             "syn":float(np.mean([r["syndrome_density"] for r in grp]))} for (p,d),grp in g.items()]

def fidelity_sigma(rows,train):
    g=defaultdict(list)
    for r in rows:
        if r["d"] in train: g[r["p"]].append((t(r["d"]),r["logR"]))
    sm={}
    for p,pts in g.items():
        if len(pts)<2: continue
        pts.sort(); ts=np.array([x[0] for x in pts]); ls=np.array([x[1] for x in pts])
        A=np.column_stack([ts,np.ones_like(ts)]); sm[p]=math.exp(float(np.linalg.lstsq(A,ls,rcond=None)[0][0]))
    return sm

def calibrate(rows,train,sm,gs_fn):
    eps=1e-60; ests=[]
    for r in rows:
        if r["d"] not in train: continue
        sp=sm.get(r["p"]); gs=max(gs_fn(r),eps)
        if sp is None: continue
        ests.append(r["logR"]+math.log(gs)-t(r["d"])*math.log(max(sp,eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

def evaluate(rows,sm,E,holdout,gs_fn):
    eps=1e-60; yp=[]; ya=[]
    for r in rows:
        if r["d"] not in holdout: continue
        sp=sm.get(r["p"],1.0); gs=max(gs_fn(r),eps)
        lrp=math.log(max((E/gs)*(sp**t(r["d"])),eps))
        yp.append(lrp); ya.append(r["logR"])
    yp=np.array(yp); ya=np.array(ya)
    A=np.column_stack([yp,np.ones_like(yp)]); coef=np.linalg.lstsq(A,ya,rcond=None)[0]
    return {"mae":float(mean_absolute_error(ya,yp)),"r2":float(r2_score(ya,yp)),
            "alpha":float(coef[0]),"beta":float(coef[1])}

def main():
    a=argparse.ArgumentParser(); a.add_argument("--source-dir",required=True); args=a.parse_args()
    payload=load(args.source_dir); nm=payload["config"]["noise_model"]
    all_rows=pool(payload["conditions"])
    rows=[r for r in all_rows if r["p"]!=0.0005]
    train={3,5,7}

    # --- Fidelity sigma baseline ---
    fid_sm=fidelity_sigma(rows,train)
    sqrt_syn_grad = lambda r: max(r["syn"],1e-60)**0.5
    syn_grad = lambda r: r["syn"]

    # --- TASK 2: Closed-form sigma candidates ---
    # estimate p_th from fidelity sigma crossing point
    fid_sorted=sorted(fid_sm.items())
    p_th_est=None
    for i in range(len(fid_sorted)-1):
        p1,s1=fid_sorted[i]; p2,s2=fid_sorted[i+1]
        if (s1-1.0)*(s2-1.0)<0:
            p_th_est=p1+(1.0-s1)*(p2-p1)/(s2-s1); break
    if p_th_est is None: p_th_est=float(np.median([p for p,_ in fid_sorted]))
    print(f"[{nm}] p_th estimated from fidelity sigma crossing: {p_th_est:.6f}")

    # Closed-form candidates per p (from training only)
    train_ps=sorted({r["p"] for r in rows if r["d"] in train})
    cf_sigmas={}  # candidate name -> {p: sigma}
    for cf_name, cf_fn in [
        ("p_th/p", lambda p: p_th_est/p),
        ("sqrt(p_th/p)", lambda p: (p_th_est/p)**0.5),
        ("(p_th/p)^0.25", lambda p: (p_th_est/p)**0.25),
        ("(p_th/p)^1.5", lambda p: (p_th_est/p)**1.5),
        ("1+(p_th-p)/p_th", lambda p: max(1+(p_th_est-p)/p_th_est,1e-60)),
        ("linear_near_thresh", lambda p: max(1+2*(p_th_est-p)/p_th_est,1e-60)),
    ]:
        cf_sigmas[cf_name]={p:cf_fn(p) for p in train_ps}

    # Fit exponent k from training: sigma = (p_th/p)^k, find k that matches fidelity sigma
    log_ratios=[math.log(fid_sm.get(p,1.0))/math.log(max(p_th_est/p,1e-60)) for p in train_ps if fid_sm.get(p) and p_th_est/p>1e-60]
    if log_ratios:
        fitted_k=float(np.median(log_ratios))
        cf_sigmas[f"(p_th/p)^{fitted_k:.3f}"]={p:(p_th_est/p)**fitted_k for p in train_ps}
        print(f"  Fitted exponent k={fitted_k:.4f} from fidelity sigma")

    # Evaluate closed-form sigmas
    print("\n=== TASK 2: Closed-Form Sigma ===")
    for cf_name, sm_cf in cf_sigmas.items():
        E=calibrate(rows,train,sm_cf,sqrt_syn_grad)
        ev=evaluate(rows,sm_cf,E,{9},sqrt_syn_grad)
        print(f"[sigma={cf_name:20s}] E={E:.4f}  d=9 MAE={ev['mae']:.4f}  R2={ev['r2']:.4f}  alpha={ev['alpha']:.4f}  beta={ev['beta']:.4f}")

    # Compare to fidelity sigma baseline
    Ef=calibrate(rows,train,fid_sm,sqrt_syn_grad)
    evf=evaluate(rows,fid_sm,Ef,{9},sqrt_syn_grad)
    print(f"[sigma=fidelity_factor    ] E={Ef:.4f}  d=9 MAE={evf['mae']:.4f}  R2={evf['r2']:.4f}  alpha={evf['alpha']:.4f}  beta={evf['beta']:.4f}")

    # --- TASK 1: MEAS grad_S sweep ---
    if nm=="meas":
        print("\n=== TASK 1: MEAS grad_S sweep ===")
        # noise model params: gate=0.2p, reset=2p, measure=3p, data=0.5p
        gs_candidates={
            "sqrt_syn": lambda r: max(r["syn"],1e-60)**0.5,
            "syn": lambda r: r["syn"],
            "p": lambda r: r["p"],
            "sqrt_p*3": lambda r: max(r["p"]*3.0,1e-60)**0.5,  # measure error dominates
            "p_meas_only": lambda r: r["p"]*3.0,  # measurement error rate
            "p_eff": lambda r: 1-(1-r["p"]*0.2)*(1-r["p"]*2.0)*(1-r["p"]*3.0)*(1-r["p"]*0.5),  # effective error prob
            "sqrt_p_eff": lambda r: max(1-(1-r["p"]*0.2)*(1-r["p"]*2.0)*(1-r["p"]*3.0)*(1-r["p"]*0.5),1e-60)**0.5,
            "linear_comb": lambda r: max(0.2*r["syn"]+0.8*r["p"],1e-60),  # mix of syn and p
            "sqrt_linear_comb": lambda r: max(0.2*r["syn"]+0.8*r["p"],1e-60)**0.5,
        }
        for gs_name,gs_fn in gs_candidates.items():
            E=calibrate(rows,train,fid_sm,gs_fn)
            ev9=evaluate(rows,fid_sm,E,{9},gs_fn)
            ev11=evaluate(rows,fid_sm,E,{11},gs_fn)
            evall=evaluate(rows,fid_sm,E,{9,11},gs_fn)
            print(f"[{gs_name:20s}] E={E:.4f}  d9 alpha={ev9['alpha']:.4f} R2={ev9['r2']:.4f}  d11 alpha={ev11['alpha']:.4f} R2={ev11['r2']:.4f}  all alpha={evall['alpha']:.4f} R2={evall['r2']:.4f}")

if __name__=="__main__":
    raise SystemExit(main())
