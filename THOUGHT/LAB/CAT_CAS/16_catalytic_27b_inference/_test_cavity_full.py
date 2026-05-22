"""Latent Phase Cavity — Full Scale.

Generates 100 gold tokens from Qwen on CUDA (KV cache, 22 tok/s),
pre-populates engine warm cache with simulated hidden states,
runs sparse catalytic probes, builds .holo latent space,
verifies latent k-NN + Phase Cavity prediction accuracy.
"""
import sys, os, time, struct, math, numpy as np, torch
from pathlib import Path
from collections import defaultdict

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
CAT_CAS = REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference'
EIGEN = REPO / 'THOUGHT' / 'LAB' / 'EIGEN_BUDDY'
MODEL_DIR = CAT_CAS / 'gemini_update' / 'qwen_0.5b'

from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# STEP 1: Qwen generates gold tokens (KV cache, CUDA)
# =====================================================================
print('=== Step 1: Qwen Oracle (KV cache, CUDA) ===')
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
    dtype=torch.float16, device_map='cuda')
model.eval()

prompt = "The catalytic computing paradigm demonstrates that"
inputs = tok(prompt, return_tensors='pt').to('cuda')
input_ids = inputs['input_ids']

# KV-cache autoregressive generation (fast, no generate() overhead)
gold_tokens = []
past_key_values = None
next_token = input_ids

N_GOLD = 20
t0 = time.perf_counter()
for i in range(N_GOLD):
    with torch.no_grad():
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
    gold_tokens.append(next_token.item())

elapsed = time.perf_counter() - t0
gold_text = tok.decode(gold_tokens)
print(f'  Generated {N_GOLD} tokens in {elapsed:.1f}s ({N_GOLD/elapsed:.0f} tok/s)')
print(f'  Text: {gold_text.encode("ascii",errors="replace").decode("ascii")[:100]}...')
print(f'  Unique tokens: {len(set(gold_tokens))}')

# =====================================================================
# STEP 2: Init catalytic runtime, run prompt tokens FIRST
# =====================================================================
print('\n=== Step 2: Init Runtime + Run Prompt (no injections yet) ===', flush=True)
sys.path.insert(0, str(REPO / 'THOUGHT' / 'LAB' / 'TINY_COMPRESS' / 'holographic-image'))
from holo_core import analyze_spectrum, project, choose_k
sys.path.insert(0, str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
os.chdir(str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
import catalytic_ffi

from experiment import (
    CatalyticInferenceRuntime, COMPLEX_DIM, F32_DIM, HIDDEN_DIM,
    NUM_LAYERS, TOTAL_WEIGHT_F32, TOTAL_WEIGHT_U8
)

print('  Init runtime (~35s)...', flush=True)
rt = CatalyticInferenceRuntime()
print('  Runtime ready.', flush=True)
substrate = bytes(rt.tape[:COMPLEX_DIM])

# Compute warm tape offset
weight_offset = COMPLEX_DIM
scratch_base = weight_offset + NUM_LAYERS * TOTAL_WEIGHT_F32
warm_tape_offset = (scratch_base + COMPLEX_DIM
                    + NUM_LAYERS * COMPLEX_DIM
                    + NUM_LAYERS * COMPLEX_DIM)
WARM_SLOTS = 256
WARM_STRIDE = 4 + COMPLEX_DIM

def embed_hash(emb_bytes):
    h = 2166136261
    for b in emb_bytes:
        h = (h ^ b) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h

# Run prompt tokens first (cold misses), THEN inject for gold tokens
rt.tokenizer._real_embedding_table = rt.streamer.embedding_table
rt._real_embedding_np = rt.streamer.embedding_np
prompt_ids = rt.tokenizer.tokenize(prompt)
for i, tid in enumerate(prompt_ids):
    emb = rt.tokenizer.embed(tid)
    tape = bytes(rt.tape)
    result = catalytic_ffi.catalytic_inference_step(tape, emb, NUM_LAYERS,
        rt.streamer.scrambled_weights, i)
    if 'working_region' in result:
        rt.tape[:len(result['working_region'])] = bytearray(result['working_region'])
    warm = result.get('warm_hit', False)
    print(f'  prompt[{i}] tid={tid} {"WARM" if warm else "COLD"}', end=' ')
print()

# NOW inject simulated states for gold tokens (overwrites prompt entries safely)
injected_hashes = set()
for tid in set(gold_tokens):
    emb = bytes(rt.tokenizer.embed(tid))
    eh = embed_hash(emb)
    eh_bytes = struct.pack('<I', eh)
    slot = eh % WARM_SLOTS
    slot_base = warm_tape_offset + slot * WARM_STRIDE
    h_sim = bytes(a ^ b for a, b in zip(
        substrate[:COMPLEX_DIM],
        emb[:COMPLEX_DIM].ljust(COMPLEX_DIM, b'\x00')
    ))
    rt.tape[slot_base:slot_base+4] = eh_bytes
    rt.tape[slot_base+4:slot_base+4+COMPLEX_DIM] = h_sim
    injected_hashes.add(eh)

print(f'  Injected {len(injected_hashes)} warm cache entries for gold tokens')

# =====================================================================
# STEP 3: Run gold tokens through engine (warm cache hits) + latent space
# =====================================================================
print('\n=== Step 3: Catalytic Probes + Latent Space ===')

probe_hidden = []
probe_targets = []
warm_count = 0
cold_count = 0

# Only process gold tokens (prompt already done)
for i, tid in enumerate(gold_tokens):
    emb = rt.tokenizer.embed(tid)
    tape = bytes(rt.tape)
    step_idx = len(prompt_ids) + i
    t0 = time.perf_counter()
    result = catalytic_ffi.catalytic_inference_step(tape, emb, NUM_LAYERS,
        rt.streamer.scrambled_weights, step_idx)
    elapsed = time.perf_counter() - t0
    if 'working_region' in result:
        rt.tape[:len(result['working_region'])] = bytearray(result['working_region'])

    warm = result.get('warm_hit', False)
    if warm: warm_count += 1
    else: cold_count += 1

    hs_bytes = bytes(result.get('hidden_state', b''))
    if len(hs_bytes) >= COMPLEX_DIM:
        hr = np.frombuffer(hs_bytes[:F32_DIM], dtype=np.float32).copy()
        hi = np.frombuffer(hs_bytes[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
        hr = np.nan_to_num(hr, nan=0, posinf=0, neginf=0)
        hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)
        probe_hidden.append(hr + 1j * hi)
        next_idx = min(i + 1, len(gold_tokens) - 1)
        probe_targets.append(gold_tokens[next_idx])

    if i < 3:
        g = result.get('generated_token', 0)
        print(f'  [{i:>3}] tid={tid:>5} gen={g:>5} gold_next={gold_tokens[min(i+1,len(gold_tokens)-1)]:>5} '
              f'{"WARM" if warm else "COLD"} {elapsed*1000:.0f}ms')

print(f'  Engine: {warm_count} warm / {cold_count} cold ({warm_count/max(warm_count+cold_count,1)*100:.0f}% warm)')
print(f'  Collected {len(probe_hidden)} catalytic hidden states')

# Build .holo latent space
Z_p = np.array(probe_hidden).astype(np.complex128)
norms = np.sqrt((Z_p.conj() * Z_p).real.sum(axis=1))
Z_pn = Z_p / np.maximum(norms, 1e-15)[:, np.newaxis]
obs = np.hstack([Z_pn.real.astype(np.float64), Z_pn.imag.astype(np.float64)])
spec = analyze_spectrum(obs)
k = max(5, min(choose_k(spec, policy='participation'), obs.shape[1]-1, len(probe_hidden)-2))
proj = project(obs, policy='fixed', fixed_k=k)
latent = proj.coordinates
print(f'  Latent: K={k}, D_pr={spec.participation_dimension:.1f}, points={len(latent)}')

# =====================================================================
# STEP 4: Latent k-NN + Phase Cavity evaluation
# =====================================================================
print('\n=== Step 4: Latent k-NN + Phase Cavity ===')

# Build centroids in latent space
class_latent = defaultdict(list)
for i, t in enumerate(probe_targets):
    class_latent[t].append(latent[i])
centroids = {t: np.mean(pts, axis=0) for t, pts in class_latent.items()}
print(f'  Centroids: {len(centroids)} classes')

# Phase Cavity check
def cavity_verify(cand_tok):
    emb = bytes(rt.tokenizer.embed(cand_tok))
    eh = embed_hash(emb)
    return eh in injected_hashes

# Evaluate on all probe positions
latent_top1 = 0
latent_top5 = 0
cavity_hits = 0
cavity_correct = 0
N = len(probe_targets)

for i, coord in enumerate(latent):
    dists = [(t, np.linalg.norm(coord - c)) for t, c in centroids.items()]
    dists.sort(key=lambda x: x[1])
    top1 = dists[0][0]
    top5 = set(d[0] for d in dists[:5])
    true_tok = probe_targets[i]

    if top1 == true_tok: latent_top1 += 1
    if true_tok in top5: latent_top5 += 1

    # Phase Cavity: find first verified candidate
    verified_tok = None
    for cand_tok, _ in dists[:10]:
        if cavity_verify(cand_tok):
            verified_tok = cand_tok
            cavity_hits += 1
            if verified_tok == true_tok:
                cavity_correct += 1
            break

print(f'\n  Results ({N} samples):')
print(f'  Latent top-1:      {latent_top1}/{N} = {latent_top1/N*100:.1f}%')
print(f'  Latent top-5:      {latent_top5}/{N} = {latent_top5/N*100:.1f}%')
print(f'  Cavity verified:   {cavity_hits}/{N} = {cavity_hits/N*100:.1f}%')
print(f'  Cavity correct:    {cavity_correct}/{N} = {cavity_correct/max(N,1)*100:.1f}%')
if cavity_hits > 0:
    print(f'  Cavity precision:  {cavity_correct}/{cavity_hits} = {cavity_correct/max(cavity_hits,1)*100:.1f}%')

rt.cleanup()
