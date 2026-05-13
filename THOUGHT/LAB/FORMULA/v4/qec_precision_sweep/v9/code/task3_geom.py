"""Task 3: same t=1, rotated vs unrotated surface code."""
import stim, pymatching, numpy as np, math, json, hashlib
from pathlib import Path

def run(geom, b, d, p, shots, seed):
    task = f"surface_code:{geom}_memory_{b}"
    key = f"t3|{seed}|{geom}|{b}|{d}|{p:.8f}".encode()
    s = int.from_bytes(hashlib.sha256(key).digest()[:4],"little")
    c = stim.Circuit.generated(task, distance=d, rounds=d,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)
    dem = c.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    det, obs = c.compile_detector_sampler(seed=s).sample(shots, separate_observables=True)
    pred = m.decode_batch(det)
    errs = np.any(pred!=obs, axis=1)
    nc = int(np.count_nonzero(errs)); pL = (nc+0.5)/(shots+1.0)
    return {"geom":geom,"basis":b,"d":d,"p":p,"n_err":nc,"pL":pL,
            "log_suppression":math.log(max(p/pL,1e-60)),
            "syndrome_density":float(np.mean(det)),
            "num_detectors":c.num_detectors,"num_qubits":c.num_qubits}

results = []
for geom in ["rotated","unrotated"]:
    for p in [0.004, 0.006, 0.008, 0.01]:
        for b in ["x","z"]:
            r = run(geom, b, 3, p, 50000, 20260513)
            print(f"{geom:10s} {b} d={r['d']} t=1 p={r['p']:.4f} logR={r['log_suppression']:.4f} syn={r['syndrome_density']:.4f} n={r['n_err']} nQ={r['num_qubits']} nD={r['num_detectors']}")
            results.append(r)

out = Path("THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v9/results/task3_geom/sweep.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"conditions":results},indent=2))
print(f"\nWrote {out}")
