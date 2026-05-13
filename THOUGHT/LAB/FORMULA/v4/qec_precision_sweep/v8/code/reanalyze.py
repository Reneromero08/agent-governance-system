"""Re-run v8 analysis excluding p=0.0005 (floor effect at 100k shots)."""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def load(source):
    for pat in ["sweep.json","combined_sweep.json"]:
        for f in Path(source).glob(pat):
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(source)

def pool_and_filter(rows, exclude_ps):
    g = defaultdict(list)
    for r in rows:
        p = float(r["physical_error_rate"])
        if p in exclude_ps: continue
        g[(p, int(r["distance"]))].append(r)
    return [{"physical_error_rate":p, "distance":d,
             "log_suppression": float(np.mean([r["log_suppression"] for r in grp])),
             "syndrome_density": float(np.mean([r["syndrome_density"] for r in grp]))}
            for (p,d), grp in g.items()]

def sigma_per_p(rows, train_dists):
    g = defaultdict(list)
    for r in rows:
        if int(r["distance"]) in train_dists:
            g[float(r["physical_error_rate"])].append((int(r["distance"]), r["log_suppression"]))
    sm = {}
    for p, pts in g.items():
        if len(pts) < 2: continue
        pts.sort(); ds = np.array([t[0] for t in pts]); ls = np.array([t[1] for t in pts])
        A = np.column_stack([ds, np.ones_like(ds)])
        slope = float(np.linalg.lstsq(A, ls, rcond=None)[0][0])
        sm[p] = math.exp(slope)
    return sm

def calibrate_E(rows, train_dists, smap):
    eps = 1e-60; ests = []
    for r in rows:
        d = int(r["distance"])
        if d not in train_dists: continue
        p = float(r["physical_error_rate"]); sp = smap.get(p)
        if sp is None: continue
        gs = max(float(r["syndrome_density"]), eps)
        ests.append(r["log_suppression"] + math.log(gs) - d * math.log(max(sp, eps)))
    return math.exp(float(np.median(ests))) if ests else 1.0

def evaluate(rows, smap, E_val, heldout):
    eps = 1e-60; yp = []; ya = []; pp = []
    for r in rows:
        d = int(r["distance"])
        if d not in heldout: continue
        p = float(r["physical_error_rate"]); sp = smap.get(p, 1.0)
        gs = max(float(r["syndrome_density"]), eps)
        lrp = math.log(max((E_val/gs)*(sp**d), eps))
        yp.append(lrp); ya.append(r["log_suppression"])
        pp.append({"p":p,"d":d,"actual":r["log_suppression"],"pred":lrp,"sigma":sp})
    yp = np.array(yp); ya = np.array(ya)
    mae = float(mean_absolute_error(ya,yp))
    r2v = float(r2_score(ya,yp))
    A = np.column_stack([yp, np.ones_like(yp)])
    coeffs = np.linalg.lstsq(A, ya, rcond=None)[0]
    return {"mae":mae,"r2":r2v,"alpha":float(coeffs[0]),"beta":float(coeffs[1]),"points":pp}

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--source-dir", required=True)
    a.add_argument("--exclude-p", type=float, nargs="*", default=[0.0005])
    a.add_argument("--run-id", default=None)
    args = a.parse_args()
    payload = load(args.source_dir)
    nm = payload["config"]["noise_model"]
    rid = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = pool_and_filter(payload["conditions"], set(args.exclude_p))
    train = {3,5,7}
    smap = sigma_per_p(rows, train)
    E = calibrate_E(rows, train, smap)

    for holdout_label, holdout_set in [("d9+d11",{9,11}),("d9",{9}),("d11",{11})]:
        ev = evaluate(rows, smap, E, holdout_set)
        print(f"[{holdout_label}] MAE={ev['mae']:.4f} R2={ev['r2']:.4f} alpha={ev['alpha']:.4f} beta={ev['beta']:.4f}")

if __name__ == "__main__":
    raise SystemExit(main())
