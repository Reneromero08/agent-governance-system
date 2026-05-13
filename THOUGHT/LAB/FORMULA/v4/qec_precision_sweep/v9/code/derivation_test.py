"""v9: derivation-based test. Df = t = floor((d-1)/2), sigma = p_th/p, grad_S = p.

From QEC_DERIVATION.md: the formula R = (E/grad_S) * sigma^Df with these
definitions algebraically reduces to the standard QEC suppression law
P_L ∝ p^(t+1). Tests whether the derivation reproduces the v8 data.
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

def t(d): return (d - 1) // 2

def load(source):
    for pat in ["sweep.json"]:
        for f in Path(source).glob(pat):
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(source)

def pool(rows):
    g = defaultdict(list)
    for r in rows:
        g[(float(r["physical_error_rate"]), int(r["distance"]))].append(r)
    return [{"physical_error_rate": p, "distance": d,
             "log_suppression": float(np.mean([r["log_suppression"] for r in grp])),
             "syndrome_density": float(np.mean([r["syndrome_density"] for r in grp]))}
            for (p,d), grp in g.items()]

def estimate_p_th(rows, train_dists):
    """Estimate p_th as the geometric midpoint of sigma crossing 1.0, from training data only."""
    groups = defaultdict(list)
    for r in rows:
        d = int(r["distance"])
        if d in train_dists:
            groups[float(r["physical_error_rate"])].append((d, r["log_suppression"]))

    sigmas = {}
    for p, pts in groups.items():
        if len(pts) < 2: continue
        pts.sort()
        ds = np.array([t(pt[0]) for pt in pts])
        ls = np.array([pt[1] for pt in pts])
        A = np.column_stack([ds, np.ones_like(ds)])
        slope = float(np.linalg.lstsq(A, ls, rcond=None)[0][0])
        sigmas[p] = math.exp(slope)

    sigmas_sorted = sorted(sigmas.items())
    for i in range(len(sigmas_sorted)-1):
        p1,s1 = sigmas_sorted[i]; p2,s2 = sigmas_sorted[i+1]
        if (s1 - 1.0) * (s2 - 1.0) < 0:
            return float(p1 + (1.0 - s1) * (p2 - p1) / (s2 - s1))
    # fallback: use median of sigmas near 1
    return float(np.median([p for p,s in sigmas_sorted]))

def evaluate_derivation(rows, p_th, train_dists, heldout, grad_S_key="p"):
    eps = 1e-60
    yp, ya, pp = [], [], []
    for r in rows:
        d = int(r["distance"])
        if d not in heldout: continue
        p = float(r["physical_error_rate"])
        Df = float(t(d))
        sigma = max(p_th / max(p, eps), eps)
        gs = max(float(r["syndrome_density"]), eps) if grad_S_key == "syndrome" else max(p, eps)
        lrp = math.log(max((1.0/gs)*(sigma**Df), eps))
        yp.append(lrp); ya.append(r["log_suppression"])
        pp.append({"p":p,"d":d,"Df":Df,"sigma":sigma,"actual":r["log_suppression"],"pred":lrp})

    yp = np.array(yp); ya = np.array(ya)
    mae = float(mean_absolute_error(ya,yp))
    r2v = float(r2_score(ya,yp))
    A = np.column_stack([yp, np.ones_like(yp)])
    coeffs = np.linalg.lstsq(A, ya, rcond=None)[0]
    return {"mae":mae,"r2":r2v,"alpha":float(coeffs[0]),"beta":float(coeffs[1]),"points":pp}

def exponent_test(rows):
    """Fit log(P_L) vs log(p) for each distance, check exponent ≈ t+1."""
    results = []
    for d in sorted({int(r["distance"]) for r in rows}):
        pts = [r for r in rows if int(r["distance"])==d]
        if len(pts) < 3: continue
        log_ps = np.array([math.log(max(float(r["physical_error_rate"]),1e-60)) for r in pts])
        # compute P_L from log_suppression: P_L = p / exp(log_suppression)
        log_PLs = np.array([math.log(float(r["physical_error_rate"])) - r["log_suppression"] for r in pts])
        A = np.column_stack([log_ps, np.ones_like(log_ps)])
        coeffs,_,_,_ = np.linalg.lstsq(A, log_PLs, rcond=None)
        exp_fit = float(coeffs[0])
        exp_expected = t(d) + 1
        results.append({"d":d,"t":t(d),"expected_exp":exp_expected,"fitted_exp":round(exp_fit,4),
                        "error":round(exp_fit-exp_expected,4)})
    return results

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--source-dir", required=True)
    a.add_argument("--run-id", default=None)
    args = a.parse_args()
    payload = load(args.source_dir)
    nm = payload["config"]["noise_model"]
    rid = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = pool(payload["conditions"])
    rows = [r for r in rows if float(r["physical_error_rate"]) != 0.0005]  # floor
    train = {3,5,7}; holdout = {9,11}

    p_th = estimate_p_th(rows, train)
    print(f"p_th (from training): {p_th:.6f}")

    # Test 1: sigma = p_th/p, grad_S = p
    ev_depol = evaluate_derivation(rows, p_th, train, holdout, "p")
    print(f"[p_th/p, grad_S=p]  MAE={ev_depol['mae']:.4f} R2={ev_depol['r2']:.4f} alpha={ev_depol['alpha']:.4f} beta={ev_depol['beta']:.4f}")

    # Test 2: sigma = p_th/p, grad_S = syndrome_density
    ev_syn = evaluate_derivation(rows, p_th, train, holdout, "syndrome")
    print(f"[p_th/p, grad_S=syn] MAE={ev_syn['mae']:.4f} R2={ev_syn['r2']:.4f} alpha={ev_syn['alpha']:.4f} beta={ev_syn['beta']:.4f}")

    # Test 3: exponent check
    exp = exponent_test(rows)
    print(f"Exponent test: {len(exp)} distances")
    for e in exp:
        match = abs(e["error"]) < 0.5
        print(f"  d={e['d']} t+1={e['expected_exp']} fitted={e['fitted_exp']} error={e['error']:.4f} {'MATCH' if match else 'MISMATCH'}")

    out = RESULTS / rid; out.mkdir(parents=True, exist_ok=True)
    result = {"run_id":rid,"noise_model":nm,"p_th":p_th,
        "ev_gradS_p":ev_depol,"ev_gradS_syn":ev_syn,"exponents":exp}
    (out/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Wrote {out/'analysis.json'}")

if __name__ == "__main__":
    raise SystemExit(main())
