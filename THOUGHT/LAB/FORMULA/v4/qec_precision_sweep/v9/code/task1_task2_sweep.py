"""Task 1+2: unrotated surface code sweep (t=1,2,3,4) + fine threshold grid on rotated."""
import argparse, csv, hashlib, json, math, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np, pymatching, stim

ROOT = Path(__file__).resolve().parents[1]; RESULTS = ROOT / "results"

@dataclass(frozen=True)
class Cond: basis:str; distance:int; rounds:int; physical_error_rate:float; shots:int; seed:int; geom:str

def seed(base,basis,d,p,geom):
    key=f"t1t2|{base}|{geom}|{basis}|{d}|{p:.8f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4],"little")

def run(c:Cond):
    t0=time.perf_counter()
    task=f"surface_code:{c.geom}_memory_{c.basis}"
    circ=stim.Circuit.generated(task,distance=c.distance,rounds=c.rounds,
        after_clifford_depolarization=c.physical_error_rate,after_reset_flip_probability=c.physical_error_rate,
        before_measure_flip_probability=c.physical_error_rate,before_round_data_depolarization=c.physical_error_rate)
    dem=circ.detector_error_model(decompose_errors=True)
    m=pymatching.Matching.from_detector_error_model(dem)
    det,obs=circ.compile_detector_sampler(seed=c.seed).sample(c.shots,separate_observables=True)
    pred=m.decode_batch(det); errs=np.any(pred!=obs,axis=1)
    nc=int(np.count_nonzero(errs)); pL=(nc+0.5)/(c.shots+1.0)
    return {**asdict(c),"logical_error_count":nc,"logical_error_rate":nc/c.shots,
        "logical_error_rate_laplace":pL,"log_suppression":math.log(max(c.physical_error_rate/pL,1e-60)),
        "syndrome_density":float(np.mean(det)),"num_detectors":circ.num_detectors,
        "num_qubits":circ.num_qubits,"duration_seconds":time.perf_counter()-t0}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--geom",default="unrotated",choices=["rotated","unrotated"])
    a.add_argument("--mode",default="task1",choices=["task1","task2"])
    a.add_argument("--run-id",default=None)
    args=a.parse_args()
    rid=args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.mode=="task1":
        ps=[0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04]
        dists=[3,5,7,9]; bases=["x","z"]; shots=100000
    else:
        ps=[0.004,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.009,0.01]
        dists=[3,5,7]; bases=["x","z"]; shots=100000

    conds=[Cond(basis=b,distance=d,rounds=d,physical_error_rate=p,shots=shots,
            seed=seed(20260513,b,d,p,args.geom),geom=args.geom)
           for b in bases for d in dists for p in ps]

    out=RESULTS/rid; out.mkdir(parents=True,exist_ok=True); rows=[]
    for i,c in enumerate(conds,1):
        print(f"[{i}/{len(conds)}] {args.geom} {args.mode} {c.basis} d={c.distance} p={c.physical_error_rate}",flush=True)
        rows.append(run(c))

    cp=out/"conditions.csv"
    with cp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    payload={"run_id":rid,"created_utc":datetime.now(timezone.utc).isoformat(),
        "config":{"geom":args.geom,"mode":args.mode,"shots":shots,"distances":dists,
                  "bases":bases,"physical_error_rates":ps,"seed":20260513},"conditions":rows}
    jp=out/"sweep.json"; jp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Wrote {jp}")

if __name__=="__main__":
    raise SystemExit(main())
