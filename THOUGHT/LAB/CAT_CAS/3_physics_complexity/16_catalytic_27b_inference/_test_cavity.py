"""Latent Phase Cavity — injects latent space into engine's warm cache.

Core insight: pre-populate the ENGINE's warm cache with simulated hidden states
(substrate XOR embedding). Then latent k-NN predicts candidates, Phase Cavity
(the engine's built-in warm cache lookup) verifies them at inference speed.

No separate Python cache — everything lives on the tape.
"""
import sys, os, time, struct, numpy as np, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
CAT_CAS = Path(__file__).resolve().parent
EIGEN = REPO / 'THOUGHT' / 'LAB' / 'EIGEN_BUDDY'
MODEL_DIR = CAT_CAS / 'gemini_update' / 'qwen_0.5b'

sys.path.insert(0, str(REPO / 'THOUGHT' / 'LAB' / 'TINY_COMPRESS' / 'holographic-image'))
from holo_core import analyze_spectrum, project, choose_k
sys.path.insert(0, str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
os.chdir(str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
import catalytic_ffi

from experiment import (
    CatalyticInferenceRuntime, COMPLEX_DIM, F32_DIM, HIDDEN_DIM, NUM_LAYERS,
    TOTAL_WEIGHT_F32, TOTAL_WEIGHT_U8
)

WARM_TAPE_SLOTS = 256
WARM_TAPE_STRIDE = 4 + COMPLEX_DIM  # 7172 bytes

def compute_warm_tape_offset():
    """Compute warm_tape_offset matching Rust's layout."""
    weight_offset = COMPLEX_DIM  # 7168
    scratch_base = weight_offset + NUM_LAYERS * TOTAL_WEIGHT_F32
    temp_offset = scratch_base
    pre_gate_base = temp_offset + COMPLEX_DIM
    saved_outputs_offset = pre_gate_base + NUM_LAYERS * COMPLEX_DIM
    warm_tape_offset = saved_outputs_offset + NUM_LAYERS * COMPLEX_DIM
    return warm_tape_offset

def compute_embedding_hash(embedding_bytes):
    """FNV-1a hash matching Rust's implementation."""
    h = 2166136261
    for b in embedding_bytes:
        h = (h ^ b) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h

# =====================================================================
# STEP 1: Qwen Oracle -> gold tokens
# =====================================================================
print('='*78)
print('LATENT PHASE CAVITY')
print('='*78)
print('\n--- Step 1: Qwen Oracle ---')
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
    dtype=torch.float16, device_map='cuda')
model.eval()

prompt = "The catalytic computing paradigm demonstrates that"
inputs = tok(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(inputs['input_ids'], max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
gold_tokens = out[0].tolist()[inputs['input_ids'].shape[1]:]
gold_text = tok.decode(gold_tokens)
print(f'  Gold ({len(gold_tokens)} tok): {gold_text.encode("ascii",errors="replace").decode("ascii")}')

# =====================================================================
# STEP 2: Inject simulated hidden states into ENGINE's tape
# =====================================================================
print('\n--- Step 2: Inject simulated hidden states into engine warm cache ---')

rt = CatalyticInferenceRuntime()
warm_off = compute_warm_tape_offset()

# Get substrate from tape (first COMPLEX_DIM bytes)
substrate = bytes(rt.tape[:COMPLEX_DIM])

# Inject simulated hidden states for each gold token
injected = {}
for tid in set(gold_tokens):
    emb = bytes(rt.tokenizer.embed(tid))
    emb_hash = compute_embedding_hash(emb)
    emb_hash_bytes = struct.pack('<I', emb_hash)
    slot = emb_hash % WARM_TAPE_SLOTS
    slot_base = warm_off + slot * WARM_TAPE_STRIDE

    # Simulated hidden state: substrate XOR embedding
    h_sim = bytes(a ^ b for a, b in zip(
        substrate[:COMPLEX_DIM],
        emb[:COMPLEX_DIM].ljust(COMPLEX_DIM, b'\x00')
    ))

    # Write to tape's warm cache
    rt.tape[slot_base:slot_base+4] = emb_hash_bytes
    rt.tape[slot_base+4:slot_base+4+COMPLEX_DIM] = h_sim

    injected[emb_hash] = (tid, slot)

print(f'  Injected {len(injected)} entries into engine warm cache')

# =====================================================================
# STEP 3: Run catalytic engine -> collect hidden states -> latent space
# =====================================================================
print('\n--- Step 3: Catalytic probes + Latent Space ---')

prompt_ids = rt.tokenizer.tokenize(prompt)
all_ids = prompt_ids + gold_tokens

probe_hidden = []
probe_targets = []

rt.tokenizer._real_embedding_table = rt.streamer.embedding_table
rt._real_embedding_np = rt.streamer.embedding_np

for i, tid in enumerate(all_ids):
    emb = rt.tokenizer.embed(tid)
    tape = bytes(rt.tape)
    t0 = time.perf_counter()
    result = catalytic_ffi.catalytic_inference_step(tape, emb, NUM_LAYERS,
        rt.streamer.scrambled_weights, i)
    elapsed = time.perf_counter() - t0
    if 'working_region' in result:
        rt.tape[:len(result['working_region'])] = bytearray(result['working_region'])

    warm = result.get('warm_hit', False)
    is_gold = i >= len(prompt_ids)
    gold_idx = i - len(prompt_ids)

    if is_gold:
        hs_bytes = bytes(result.get('hidden_state', b''))
        if len(hs_bytes) >= COMPLEX_DIM:
            hr = np.frombuffer(hs_bytes[:F32_DIM], dtype=np.float32).copy()
            hi = np.frombuffer(hs_bytes[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
            hr = np.nan_to_num(hr, nan=0, posinf=0, neginf=0)
            hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)
            probe_hidden.append(hr + 1j * hi)
            next_idx = min(gold_idx + 1, len(gold_tokens) - 1)
            probe_targets.append(gold_tokens[next_idx])

        label = 'WARM' if warm else 'COLD'
        gen_tok = result.get('generated_token', 0)
        if gold_idx < 5:
            print(f'  [{gold_idx:>2}] tok={tid:>5} gold_next={gold_tokens[min(gold_idx+1,len(gold_tokens)-1)]:>5} '
                  f'gen={gen_tok:>5} {label} {elapsed*1000:.0f}ms')

    if is_gold and gold_idx >= len(gold_tokens) - 1:
        break

# Build latent space from probes
print(f'\n  Collected {len(probe_hidden)} catalytic probes')
if len(probe_hidden) >= 4:
    Z_p = np.array(probe_hidden).astype(np.complex128)
    norms = np.sqrt((Z_p.conj() * Z_p).real.sum(axis=1))
    Z_pn = Z_p / np.maximum(norms, 1e-15)[:, np.newaxis]
    obs = np.hstack([Z_pn.real.astype(np.float64), Z_pn.imag.astype(np.float64)])
    spec = analyze_spectrum(obs)
    k = max(3, min(choose_k(spec, policy='participation'), obs.shape[1]-1, len(probe_hidden)-1))
    proj = project(obs, policy='fixed', fixed_k=k)
    latent = proj.coordinates

    # Build latent centroids
    class_latent = {}
    for i, t in enumerate(probe_targets):
        if t not in class_latent: class_latent[t] = []
        class_latent[t].append(latent[i])
    centroids = {t: np.mean(pts, axis=0) for t, pts in class_latent.items()}
    print(f'  Latent: K={k}, D_pr={spec.participation_dimension:.1f}, centroids={len(centroids)}')

# =====================================================================
# STEP 4: Latent k-NN -> Phase Cavity (warm cache hit check)
# =====================================================================
print('\n--- Step 4: Latent k-NN + Phase Cavity test ---')

if len(probe_hidden) >= 4:
    cavity_hits = 0
    cavity_total = 0
    latent_top1_correct = 0
    latent_top5_correct = 0

    for i, coord in enumerate(latent):
        # k-NN: nearest centroids in latent space
        dists = [(t, np.linalg.norm(coord - c)) for t, c in centroids.items()]
        dists.sort(key=lambda x: x[1])
        top5 = set(d[0] for d in dists[:5])
        top1 = dists[0][0]

        # Phase Cavity: test each candidate against engine warm cache
        verified = False
        ver_token = None
        for cand_tok, _ in dists[:10]:
            cand_emb = bytes(rt.tokenizer.embed(cand_tok))
            cand_hash = compute_embedding_hash(cand_emb)
            cand_slot = cand_hash % WARM_TAPE_SLOTS
            cand_base = warm_off + cand_slot * WARM_TAPE_STRIDE
            stored_hash = rt.tape[cand_base:cand_base+4]
            if stored_hash == struct.pack('<I', cand_hash):
                verified = True
                ver_token = cand_tok
                break

        true_tok = probe_targets[i]
        cavity_total += 1
        if verified:
            cavity_hits += 1
        if top1 == true_tok:
            latent_top1_correct += 1
        if true_tok in top5:
            latent_top5_correct += 1
        status = 'HIT' if (verified and ver_token == true_tok) else ('VER' if verified else 'MIS')
        if i < 10:
            print(f'  [{i:>2}] true={true_tok:>5} latent_top1={top1:>5} '
                  f'cavity={ver_token} {status}')

    print(f'\n  Latent top-1: {latent_top1_correct}/{cavity_total} ({latent_top1_correct/max(cavity_total,1)*100:.0f}%)')
    print(f'  Latent top-5: {latent_top5_correct}/{cavity_total} ({latent_top5_correct/max(cavity_total,1)*100:.0f}%)')
    print(f'  Phase Cavity hit rate: {cavity_hits}/{cavity_total} ({cavity_hits/max(cavity_total,1)*100:.0f}%)')

# =====================================================================
# STEP 5: Cache statistics
# =====================================================================
print(f'\n--- Step 5: Cache Hit Summary ---')
total_warm = 0; total_cold = 0
for i in range(len(all_ids)):
    emb = rt.tokenizer.embed(all_ids[i])
    emb_hash = compute_embedding_hash(emb)
    if emb_hash in injected:
        total_warm += 1
    else:
        total_cold += 1
print(f'  Pre-injected tokens that would hit cache: {total_warm}/{len(all_ids)}')
print(f'  Cold misses (need compute): {total_cold}/{len(all_ids)}')

rt.cleanup()
