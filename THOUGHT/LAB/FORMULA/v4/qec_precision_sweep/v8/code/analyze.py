"""v8 analysis: pooled bases, 3-point sigma fit on {3,5,7}, test on {9,11}.

Measures: alpha, beta, R2 per test distance. Curvature check across {9,11}.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def load(source: str):
    for pat in ["sweep.json","combined_sweep.json","qec_precision_sweep_v2.json"]:
        for f in Path(source).glob(pat):
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(source)

def pool(rows):
    g = defaultdict(list)
    for r in rows:
        g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
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

def report(path, data):
    nm = data["noise_model"]; smap = data["sigma_map"]
    lines = [f"# v8 -- {nm}", "",
        f"E={data['E']:.4f} | Train {sorted(data['train']) } | Holdout {sorted(data['holdout'])}",
        "", "## Sigma per p","| p | sigma | ln(sigma) |","|---:|---:|---:|"]
    for p in sorted(smap):
        lines.append(f"| {p:.4f} | {smap[p]:.4f} | {math.log(smap[p]):.4f} |")
    for label, ev in data["evaluations"].items():
        lines.extend(["", f"## {label}", f"MAE={ev['mae']:.4f} R2={ev['r2']:.4f} alpha={ev['alpha']:.4f} beta={ev['beta']:.4f}",
            "| p | d | actual | pred | error |","|---:|---:|---:|---:|---:|"])
        for pt in sorted(ev["points"], key=lambda x: (x["p"],x["d"])):
            lines.append(f"| {pt['p']:.4f} | {pt['d']} | {pt['actual']:.4f} | {pt['pred']:.4f} | {abs(pt['actual']-pt['pred']):.4f} |")
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--source-dir", required=True)
    args.add_argument("--run-id", default=None)
    a = args.parse_args()
    payload = load(a.source_dir)
    nm = payload["config"]["noise_model"]
    rid = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = pool(payload["conditions"])
    train = {3,5,7}
    holdout = {9,11}
    smap = sigma_per_p(rows, train)
    E = calibrate_E(rows, train, smap)

    ev_all = evaluate(rows, smap, E, {9,11})
    ev_9 = evaluate(rows, smap, E, {9})
    ev_11 = evaluate(rows, smap, E, {11})

    out = RESULTS / rid; out.mkdir(parents=True, exist_ok=True)
    result = {"run_id":rid, "source_run": payload["run_id"], "noise_model": nm,
        "train":sorted(train), "holdout":sorted(holdout), "E":E, "sigma_map":smap,
        "evaluations": {"d9+d11": ev_all, "d9": ev_9, "d11": ev_11}}
    jp = out / "analysis.json"; jp.write_text(json.dumps(result,indent=2),encoding="utf-8")
    rp = out / "REPORT.md"; report(rp, result)

    for label, ev in [("d9+d11",ev_all),("d9",ev_9),("d11",ev_11)]:
        print(f"[{label}] MAE={ev['mae']:.4f} R2={ev['r2']:.4f} alpha={ev['alpha']:.4f} beta={ev['beta']:.4f}")
    print(f"Wrote {jp}")

if __name__ == "__main__":
    raise SystemExit(main())
