"""Color code sweep with ignore_decomposition_failures."""
import stim, pymatching, numpy as np, math, json, hashlib, time
from pathlib import Path

def t(d): return (d-1)//2
def run(b,d,p,shots,seed):
    key=f'cc|{seed}|{b}|{d}|{p:.8f}'.encode()
    s=int.from_bytes(hashlib.sha256(key).digest()[:4],'little')
    t0=time.perf_counter()
    c=stim.Circuit.generated('color_code:memory_xyz',distance=d,rounds=d,
        after_clifford_depolarization=p,after_reset_flip_probability=p,
        before_measure_flip_probability=p,before_round_data_depolarization=p)
    dem=c.detector_error_model(decompose_errors=True,ignore_decomposition_failures=True)
    m=pymatching.Matching.from_detector_error_model(dem)
    det,obs=c.compile_detector_sampler(seed=s).sample(shots,separate_observables=True)
    pred=m.decode_batch(det); errs=np.any(pred!=obs,axis=1)
    nc=int(np.count_nonzero(errs)); pL=(nc+0.5)/(shots+1.0)
    return {'basis':b,'d':d,'t':t(d),'p':p,'n_err':nc,'pL':pL,
            'logR':math.log(max(p/pL,1e-60)),'syn':float(np.mean(det)),
            'nQ':c.num_qubits,'nD':c.num_detectors,'sec':time.perf_counter()-t0}

results=[]
for d in [3,5,7]:
    shots=30000 if d==3 else 50000
    for p in [0.001,0.002,0.004,0.006,0.008,0.01]:
        for b in ['x','y','z']:
            r=run(b,d,p,shots,20260513); results.append(r)
            print(f'cc d={r["d"]} t={r["t"]} {r["basis"]} p={r["p"]:.4f} logR={r["logR"]:+.4f} n_err={r["n_err"]} nQ={r["nQ"]} nD={r["nD"]}')

out=Path('THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v9/results/color_code/sweep.json')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'config':{'code':'color_code','distances':[3,5,7],'bases':['x','y','z']},'conditions':results},indent=2))
print(f'Wrote {out} ({len(results)} conditions)')
