"""Minimal Phase 2 test: catalytic engine with hardcoded gold tokens."""
import sys, os, time, numpy as np, torch
from pathlib import Path
CAT_CAS = Path(r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference')
EIGEN = CAT_CAS.parent.parent / 'EIGEN_BUDDY'
sys.path.insert(0, str(CAT_CAS))
sys.path.insert(0, str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
os.chdir(str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
import catalytic_ffi
from experiment import CatalyticInferenceRuntime, COMPLEX_DIM, F32_DIM, NUM_LAYERS, HIDDEN_DIM

print('Init runtime...', flush=True)
t0 = time.perf_counter()
rt = CatalyticInferenceRuntime()
print(f'Done: {time.perf_counter()-t0:.1f}s', flush=True)

gold_tokens = [9707, 1879, 0, 358, 2776, 264, 48948, 304, 13027, 323]
hidden_real = []
hidden_imag = []
targets = []

for i, tid in enumerate(gold_tokens):
    emb = rt.tokenizer.embed(tid)
    tape = bytes(rt.tape)
    t0 = time.perf_counter()
    result = catalytic_ffi.catalytic_inference_step(tape, emb, NUM_LAYERS, rt.streamer.scrambled_weights, i)
    elapsed = time.perf_counter() - t0
    if 'working_region' in result:
        rt.tape[:len(result['working_region'])] = bytearray(result['working_region'])
    warm = result.get('warm_hit', False)
    label = 'WARM' if warm else 'COLD'
    hs = bytes(result.get('hidden_state', bytes(rt.tape[:COMPLEX_DIM])))
    hr = np.frombuffer(hs[:F32_DIM], dtype=np.float32).copy()
    hi = np.frombuffer(hs[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
    hr = np.nan_to_num(hr, nan=0, posinf=0, neginf=0)
    hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)
    hidden_real.append(hr)
    hidden_imag.append(hi)
    gold_next = gold_tokens[i+1] if i+1 < len(gold_tokens) else tid
    targets.append(gold_next)
    print(f'  [{i}] tid={tid} gold_next={gold_next} {label} {elapsed*1000:.0f}ms', flush=True)

print(f'\nCollected {len(targets)} pairs')
print(f'Real range: [{np.min(hidden_real):.2e}, {np.max(hidden_real):.2e}]')
print(f'Imag range: [{np.min(hidden_imag):.2e}, {np.max(hidden_imag):.2e}]')
print(f'NaN in real: {np.any(np.isnan(hidden_real))}')
print(f'NaN in imag: {np.any(np.isnan(hidden_imag))}')
rt.cleanup()
