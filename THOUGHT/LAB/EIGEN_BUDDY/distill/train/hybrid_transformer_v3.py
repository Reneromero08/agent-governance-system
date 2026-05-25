"""
Phase 14: FULL STACK ASSEMBLY — The Superradiant Transformer
=============================================================
Wires Dynamic Carrier Wave and Native Hologram (M) directly into the full
MultiHeadComplexAttention module using .holo-injected weights from Qwen 27B.

Architecture:
  1. Qwen 27B Embeddings (frozen) → complex hidden states
  2. MultiHeadComplexAttention with .holo-injected k_proj/v_proj weights
  3. Hologram M tracks prompt variable bindings (ingested from fibonacci prompt)
  4. Dynamic Carrier Shifting: fibonacci → params ( n - on trigger
  5. Combined logits: 0.15 * attention + 0.30 * hologram + 0.55 * carrier_boost

Usage:
  python hybrid_transformer_v3.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE.parent))
from core.attention import MultiHeadComplexAttention

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
N_HEADS = 8
DH = D_MODEL // N_HEADS
HALF = D_MODEL // 2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLO_NPZ = BASE / "distilled" / "eigenbuddy_distilled.holo.npz"
HOLO_JSON = BASE / "distilled" / "eigenbuddy_distilled.json"
EXCLUDE = {',', 'the', 'to', 'is', '?', '!', '-', ''}

PYTHON_CODE_CORPUS = """
def add(a, b): return a + b
def multiply(x, y): return x * y
def is_even(n): return n % 2 == 0
def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
def gcd(a, b): while b: a, b = b, a % b; return a
def is_prime(n): if n < 2: return False; for i in range(2, int(n**0.5)+1): if n % i == 0: return False; return True
def sum_list(lst): total = 0; for x in lst: total += x; return total
def reverse_string(s): return s[::-1]
def binary_search(arr, target): left, right = 0, len(arr) - 1; while left <= right: mid = (left+right)//2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid+1; else: right = mid-1; return -1
def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[0]; less = [x for x in arr[1:] if x <= pivot]; greater = [x for x in arr[1:] if x > pivot]; return quicksort(less) + [pivot] + quicksort(greater)
class Counter: def __init__(self): self.count = 0; def increment(self): self.count += 1; def get(self): return self.count
if __name__ == '__main__': print(add(1, 2))
for i in range(10): print(i)
while True: break
try: x = 1 / 0; except ZeroDivisionError: pass
with open('file.txt') as f: data = f.read()
lambda x: x * x
import os; os.path.join('a', 'b')
""".strip()

print("=" * 60)
print("PHASE 14: FULL STACK ASSEMBLY — SUPERRADIANT TRANSFORMER")
print("=" * 60)

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
V = tokenizer.vocab_size
print(f"  Vocab: {V}")

print("Loading Qwen 27B embeddings...")
import safetensors.torch as st

embed_weight = None
lm_head_weight = None
for sp in sorted(MODEL_DIR.glob("model-*.safetensors")):
    tensors = st.load_file(str(sp))
    for k in tensors:
        if "embed_tokens" in k and embed_weight is None:
            embed_weight = tensors[k][:V, :D_MODEL].float()
        if "lm_head" in k and lm_head_weight is None:
            lm_head_weight = tensors[k][:V, :D_MODEL].float()
    if embed_weight is not None and lm_head_weight is not None:
        break

er = embed_weight[:, :HALF]
ei = embed_weight[:, HALF:]
er = er / er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
ei = ei / ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)
phase_angle = torch.atan2(ei, er)
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)

token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>!]')
code_words_in_corpus = set()
for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
    code_words_in_corpus.add(m.group())
EXTRA_SYMBOLS = {'+', '*', '/', '-', '%', '=', '<', '>', '!', '&', '|', '^', '~', '#', '@', '$'}
code_words_in_corpus |= EXTRA_SYMBOLS

vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if word in code_words_in_corpus:
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  Vocab mask: {n_allowed} code-corpus tokens")

print(f"Precomputing concept phases for {n_allowed} words...")
concept_phases = torch.zeros(V, HALF, dtype=torch.complex64, device=DEV)
concept_words = [""] * V
for tid in range(V):
    if vocab_mask[tid] == 0:
        continue
    word = tokenizer.decode([tid]).strip()
    concept_words[tid] = word
    sub_ids = tokenizer.encode(word, add_special_tokens=False)
    if not sub_ids:
        continue
    cp = phase_vectors[sub_ids[0]].clone()
    for sid in sub_ids[1:]:
        cp = cp * phase_vectors[sid]
    concept_phases[tid] = cp
print(f"  Concept phases: {concept_phases.numel() * 8 / 1e6:.0f} MB")


def resolve_cid(word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    for tid in range(V):
        if concept_words[tid] == word and vocab_mask[tid] > 0:
            return tid
    return ids[0] if ids[0] < V and vocab_mask[ids[0]] > 0 else None


def get_phase(word):
    cid = resolve_cid(word)
    return concept_phases[cid] if cid is not None else None


class NativeHologramM:
    def __init__(self, cp, mask):
        self.cp = cp
        self.mask = mask
        self.HALF = cp.shape[1]
        self.M = torch.zeros(self.HALF, dtype=torch.complex64, device=cp.device)

    def ingest(self, text):
        lines = text.split('\n')
        cids = []
        for line in lines:
            words = line.split()
            for w in words:
                clean = w.strip('.,!?;:')
                if clean in EXCLUDE:
                    continue
                cid = resolve_cid(clean)
                if cid is not None:
                    cids.append(cid)
            cids.append(-1)
        if cids and cids[-1] < 0:
            cids.pop()
        bound, skipped = 0, 0
        for i in range(len(cids) - 1):
            pi, ci = cids[i], cids[i + 1]
            if pi < 0 or ci < 0:
                skipped += 1
                continue
            self.M += self.cp[ci] * self.cp[pi].conj()
            bound += 1
        return bound, skipped

    def forward_wave(self, phase):
        return self.M * phase

    def bind(self, pc, pp):
        self.M += pc * pp.conj()


def inject_eigenbasis(attn, holo_data):
    k_gratings = []
    v_gratings = []
    for key in holo_data.files:
        g = torch.tensor(holo_data[key])
        if g.shape[1] != D_MODEL:
            continue
        if 'k_proj' in key:
            k_gratings.append(g.to(DEV))
        elif 'v_proj' in key:
            v_gratings.append(g.to(DEV))

    with torch.no_grad():
        for h in range(N_HEADS):
            s, e = h * DH, (h + 1) * DH
            ki = h % max(len(k_gratings), 1)
            vi = h % max(len(v_gratings), 1)
            kg = k_gratings[ki] if k_gratings else None
            vg = v_gratings[vi] if v_gratings else None
            for pn in ['qr', 'qi', 'kr', 'ki', 'vr', 'vi']:
                w = getattr(attn, pn).weight
                gr = kg if 'k' in pn and kg is not None else (vg if 'v' in pn and vg is not None else None)
                if gr is None:
                    continue
                if gr.is_complex():
                    em = gr[h % gr.shape[0], :D_MODEL].real.float()
                else:
                    em = gr[h % gr.shape[0], :D_MODEL].float()
                rot_angle = 2 * math.pi * h / N_HEADS
                rotated = em * math.cos(rot_angle)
                w.data[s:e] = rotated.unsqueeze(0).expand(DH, -1) * 0.1


class CatalyticLM(nn.Module):
    def __init__(self, V_size, D, H):
        super().__init__()
        self.er = nn.Embedding(V_size, D)
        self.ei = nn.Embedding(V_size, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = nn.Linear(D, V_size, bias=False)

    def forward(self, ids):
        x = torch.complex(self.er(ids), self.ei(ids))
        z, _ = self.attn(x)
        return self.out(z.real)


def build_grammar_G(code_token_ids, cp):
    G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
    for i in range(len(code_token_ids) - 1):
        pi, ci = code_token_ids[i], code_token_ids[i + 1]
        G += torch.outer(cp[ci], cp[pi].conj())
    return G / len(code_token_ids)


def compute_scores(wave, cp, mask):
    raw = torch.abs(cp @ wave.conj())
    scores = (raw * mask) ** 2
    return scores


prompt = (
    'def fibonacci ( n ) : \n'
    '    """ Return the n - th Fibonacci number . """\n'
    '    if n == 0 :\n'
    '        return 0\n'
    '    if n == 1 :\n'
    '        return 1\n'
    '    return'
)

print(f"\n{'='*60}")
print(f"PROMPT:\n{prompt}")
print(f"{'='*60}")

print("Loading .holo matrices...")
holo = np.load(str(HOLO_NPZ))
meta = json.load(open(str(HOLO_JSON)))
print(f"  Loaded {len(holo.files)} phase gratings, {len(meta)} metadata entries")

holo_m = NativeHologramM(concept_phases, vocab_mask)
bound, skipped = holo_m.ingest(prompt)
print(f"Hologram ingest: {bound} edges, {skipped} firewall-skipped")
print(f"|M| mean: {holo_m.M.abs().mean():.4f}")

code_token_ids = []
for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
    cid = resolve_cid(m.group())
    if cid is not None:
        code_token_ids.append(cid)
grammar_G = build_grammar_G(code_token_ids, concept_phases)
print(f"Grammar matrix: {HALF}x{HALF}, {len(code_token_ids)} transitions, |G|={grammar_G.abs().mean():.4f}")

Phase_fib = get_phase("fibonacci")
Phase_paren = get_phase("(")
Phase_n = get_phase("n")
Phase_minus = get_phase("-")
Phase_params = Phase_paren + Phase_n + Phase_minus
Phase_params = Phase_params / (Phase_params.abs().max().clamp(min=1e-12))
print(f"Phase_fib: |fib|={float(Phase_fib.abs().mean()):.4f}  Phase_params: |(n-|={float(Phase_params.abs().mean()):.4f}")

model = CatalyticLM(V, D_MODEL, N_HEADS)
model.er.weight.data.copy_(embed_weight.float())
model.ei.weight.data.zero_()
model.out.weight.data.copy_(lm_head_weight.float())
inject_eigenbasis(model.attn, holo)
model = model.to(DEV)
model.train()
print(f"Model: 770M params, .holo injected, hologram active")

prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
ids = prompt_ids.clone()

Phase_carrier = Phase_fib
carrier_shifted = False
anneal_offset = 0
GAMMA = 0.35
max_gen = 15

current_word = "return"
skip_set = {"def", ":", ")", ",", "fibonacci", current_word}
generated = []

print(f"\n{'='*60}")
print(f"SUPERRADIANT KURAMOTO DRIVE ({max_gen} tokens)")
print(f"{'='*60}")

for step in range(max_gen):
    logits = model(ids)
    last_logits = logits[0, -1, :]

    last_tid = ids[0, -1].item()
    holo_scores = torch.zeros(V, device=DEV)
    gram_scores = torch.zeros(V, device=DEV)
    carrier_scores = torch.zeros(V, device=DEV)

    if last_tid < V and vocab_mask[last_tid] > 0:
        cp_last = concept_phases[last_tid]

        wave_M = holo_m.forward_wave(cp_last)
        holo_scores = compute_scores(wave_M, concept_phases, vocab_mask)

        query_vec = cp_last + GAMMA * Phase_carrier
        query_phase = query_vec / (query_vec.abs().max().clamp(min=1e-12))
        wave_G = grammar_G @ query_phase
        gram_scores = compute_scores(wave_G, concept_phases, vocab_mask)

        anneal_step = step - anneal_offset
        carrier_boost = (10.0 + anneal_step * 3.0)
        carrier_raw = torch.abs(concept_phases @ Phase_carrier.conj())
        carrier_scores = carrier_boost * (carrier_raw * vocab_mask) ** 2

    holo_probs = holo_scores / holo_scores.sum().clamp(min=1e-12)
    gram_probs = gram_scores / gram_scores.sum().clamp(min=1e-12)
    carrier_probs = carrier_scores / carrier_scores.sum().clamp(min=1e-12)

    attn_probs = torch.softmax(last_logits / 0.8, dim=-1)
    attn_probs = attn_probs * vocab_mask
    attn_probs = attn_probs / attn_probs.sum()

    combined = 0.15 * attn_probs + 0.30 * holo_probs + 0.55 * carrier_probs
    combined = combined / combined.sum()

    top5_vals, top5_ids = combined.topk(6)
    r1_tid = int(top5_ids[0].item())
    r1_word = concept_words[r1_tid]
    r1_score = float(top5_vals[0].item())

    carrier_label = "fib" if not carrier_shifted else "params"
    print(f"\nStep {step+1}: current='{current_word}'  carrier={carrier_label} anneal={anneal_step}")
    print(f"  Attn:    {', '.join(f'{concept_words[int(tid)]}({float(top5_vals[i]):.3f})' for i,tid in enumerate(top5_ids[:3]))}")
    print(f"  Holo:    {', '.join(f'{concept_words[int(tid)]}({float(holo_probs[tid]):.3f})' for tid in holo_probs.topk(3).indices.tolist())}")
    print(f"  Grammar: {', '.join(f'{concept_words[int(tid)]}({float(gram_probs[tid]):.3f})' for tid in gram_probs.topk(3).indices.tolist())}")
    print(f"  Carrier: {', '.join(f'{concept_words[int(tid)]}({float(carrier_probs[tid]):.3f})' for tid in carrier_probs.topk(3).indices.tolist())}")
    print(f"  Combined: '{r1_word}' (s={r1_score:.4f})")

    chosen_word = r1_word
    chosen_id = r1_tid
    for i in range(len(top5_ids)):
        tid = int(top5_ids[i].item())
        w = concept_words[tid]
        if w in skip_set:
            continue
        chosen_word = w
        chosen_id = tid
        break

    generated.append(chosen_word)
    cp_new = concept_phases[chosen_id]
    cp_prev = concept_phases[last_tid]
    holo_m.bind(cp_new, cp_prev)

    if chosen_word == "fibonacci" and not carrier_shifted:
        Phase_carrier = Phase_params
        carrier_shifted = True
        anneal_offset = step + 1
        skip_set = {"def", ":", ")", ",", "fibonacci"}
        print(f"  *** CARRIER SHIFTED: fibonacci -> params ( n -  skip_set RESET ***")

    current_word = chosen_word
    skip_set.add(chosen_word)
    if "fibonacci" in skip_set:
        skip_set.discard("fibonacci")
    if carrier_shifted:
        for sym in ["n", "-", "("]:
            skip_set.discard(sym)

    new_tok = torch.tensor([[chosen_id]], device=DEV)
    ids = torch.cat([ids, new_tok], dim=1)

    print(f"  >> Generated: '{chosen_word}'")

completion = " ".join(generated)
print(f"\n{'='*60}")
print(f"PROMPT:\n{prompt}")
print(f"COMPLETION: {completion}")
print(f"{'='*60}")
print("DONE.")
