"""
Phase 12: SUPERRADIANT ASSEMBLY — Core Splice
================================================
Splices Native Hologram (M) into MultiHeadComplexAttention forward pass.
.holo matrices route syntax; Hologram M tracks variables in scope.

Architecture:
  HybridAttention extends MultiHeadComplexAttention.
  In forward(), Q consults M to identify active variable bindings.
  Attention scores are modulated by hologram resonance before softmax.
  The modulation boosts attention toward tokens that M predicts follow the query.

Usage:
  python hybrid_transformer.py
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

print("=" * 60)
print("PHASE 12: SUPERRADIANT ASSEMBLY — CORE SPLICE")
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

token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>!]')
code_words_in_corpus = set()
for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
    code_words_in_corpus.add(m.group())
EXTRA_SYMBOLS = {'+', '*', '/', '-', '%', '=', '<', '>', '!', '&', '|', '^', '~', '#', '@', '$'}
code_words_in_corpus |= EXTRA_SYMBOLS

ascii_code = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if word in code_words_in_corpus:
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  Vocab mask: {n_allowed} code-corpus tokens, {V - n_allowed} blocked")

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


def inject_eigenbasis(attn, holo, meta):
    k_gratings = []
    v_gratings = []
    for key in holo.files:
        g = torch.tensor(holo[key])
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
                    em = gr[h % gr.shape[0], :].real.float()
                else:
                    em = gr[h % gr.shape[0], :D_MODEL].float()
                rot_angle = 2 * math.pi * h / N_HEADS
                rotated = em * math.cos(rot_angle)
                w.data[s:e] = rotated.unsqueeze(0).expand(DH, -1) * 0.1


class NativeHologramM:
    def __init__(self, cp, mask, cw):
        self.cp = cp
        self.mask = mask
        self.cw = cw
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
        bound = 0
        for i in range(len(cids) - 1):
            pi, ci = cids[i], cids[i + 1]
            if pi < 0 or ci < 0:
                continue
            self.M += self.cp[ci] * self.cp[pi].conj()
            bound += 1
        return bound

    def ingest_ids(self, token_ids):
        bound = 0
        for i in range(len(token_ids) - 1):
            pi, ci = token_ids[i], token_ids[i + 1]
            if pi < 0 or ci < 0:
                continue
            self.M += self.cp[ci] * self.cp[pi].conj()
            bound += 1
        return bound

    def forward_wave(self, phase):
        return self.M * phase

    def bind(self, pc, pp):
        self.M += pc * pp.conj()


class HolographicModulatedAttention(nn.Module):
    def __init__(self, d_model, n_heads, holo_m, concept_phases):
        super().__init__()
        self.attn = MultiHeadComplexAttention(d_model, n_heads, geo_init=False)
        self.holo = holo_m
        self.cp = concept_phases
        self.gamma = nn.Parameter(torch.tensor(0.15))

    def forward(self, x, token_ids):
        B, S, D = x.shape
        H = self.attn.H
        dh = self.attn.dh

        qr = self.attn.qr(x.real) - self.attn.qi(x.imag)
        qi = self.attn.qr(x.imag) + self.attn.qi(x.real)
        kr = self.attn.kr(x.real) - self.attn.ki(x.imag)
        ki = self.attn.kr(x.imag) + self.attn.ki(x.real)
        vr = self.attn.vr(x.real) - self.attn.vi(x.imag)
        vi = self.attn.vr(x.imag) + self.attn.vi(x.real)

        qr = qr.view(B, S, H, dh).transpose(1, 2)
        qi = qi.view(B, S, H, dh).transpose(1, 2)
        kr = kr.view(B, S, H, dh).transpose(1, 2)
        ki = ki.view(B, S, H, dh).transpose(1, 2)
        vr = vr.view(B, S, H, dh).transpose(1, 2)
        vi = vi.view(B, S, H, dh).transpose(1, 2)

        sr = (qr @ kr.transpose(-2, -1) + qi @ ki.transpose(-2, -1)) * self.attn.scale
        si = (qi @ kr.transpose(-2, -1) - qr @ ki.transpose(-2, -1)) * self.attn.scale

        holo_bias = torch.zeros(S, S, device=x.device)
        tids = token_ids[0].tolist()
        for i in range(S):
            ti = tids[i]
            if ti >= len(self.cp) or self.holo.mask[ti] == 0:
                continue
            wave_i = self.holo.forward_wave(self.cp[ti])
            for j in range(i + 1):
                tj = tids[j]
                if tj >= len(self.cp) or self.holo.mask[tj] == 0:
                    continue
                resonance = float(torch.abs(torch.dot(self.cp[tj].conj(), wave_i)))
                holo_bias[i, j] = resonance / HALF

        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf'))
        si = si.masked_fill(mask, 0.0)

        sr = sr + self.gamma.abs() * holo_bias.unsqueeze(0).unsqueeze(0)

        attn_mag = F.softmax(sr, dim=-1)
        cos_p = torch.cos(si)
        sin_p = torch.sin(si)
        out_r = (attn_mag * cos_p) @ vr - (attn_mag * sin_p) @ vi
        out_i = (attn_mag * sin_p) @ vr + (attn_mag * cos_p) @ vi

        out_r = out_r.transpose(1, 2).contiguous().view(B, S, -1)
        out_i = out_i.transpose(1, 2).contiguous().view(B, S, -1)
        or_ = self.attn.or_(out_r) - self.attn.oi(out_i)
        oi_ = self.attn.or_(out_i) + self.attn.oi(out_r)

        return torch.complex(or_, oi_), si


class SuperradiantHybridLM(nn.Module):
    def __init__(self, V, D, H, holo_m, concept_phases):
        super().__init__()
        self.er = nn.Embedding(V, D)
        self.ei = nn.Embedding(V, D)
        self.attn = HolographicModulatedAttention(D, H, holo_m, concept_phases)
        self.out = nn.Linear(D, V, bias=False)

    def forward(self, ids):
        x = torch.complex(self.er(ids), self.ei(ids))
        z, si = self.attn(x, ids)
        return self.out(z.real), si


print("\nLoading .holo matrices...")
holo = np.load(str(HOLO_NPZ))
meta = json.load(open(str(HOLO_JSON)))
print(f"  Loaded {len(holo.files)} phase gratings, {len(meta)} metadata entries")

holo_m = NativeHologramM(concept_phases, vocab_mask, concept_words)

grammar_G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
code_token_ids = []
for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
    word = m.group()
    cid = resolve_cid(word)
    if cid is not None:
        code_token_ids.append(cid)
for i in range(len(code_token_ids) - 1):
    pi, ci = code_token_ids[i], code_token_ids[i + 1]
    grammar_G += torch.outer(concept_phases[ci], concept_phases[pi].conj())
grammar_G = grammar_G / len(code_token_ids)
print(f"  Grammar matrix: {HALF}x{HALF}, {len(code_token_ids)} code tokens, |G|={grammar_G.abs().mean():.4f}")

model = SuperradiantHybridLM(V, D_MODEL, N_HEADS, holo_m, concept_phases)
model.er.weight.data.copy_(embed_weight.float())
model.ei.weight.data.zero_()
model.out.weight.data.copy_(lm_head_weight.float())
inject_eigenbasis(model.attn.attn, holo, meta)
model = model.to(DEV)
model.train()

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Model: {n_params:.0f}M params, .holo injected, hologram active")

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

prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
bound = holo_m.ingest(prompt)
print(f"Hologram ingest: {bound} directed edges burned into M")
print(f"|M| mean: {holo_m.M.abs().mean().item():.4f}")

ids = prompt_ids.clone()
generated_tokens = []
max_gen = 12

print(f"\n{'='*60}")
print(f"AUTOREGRESSIVE GENERATION ({max_gen} tokens)")
print(f"{'='*60}")

for step in range(max_gen):
    logits, si = model(ids)

    holo_scores = torch.zeros(V, device=DEV)
    grammar_scores = torch.zeros(V, device=DEV)
    last_tid = ids[0, -1].item()
    if last_tid < V and vocab_mask[last_tid] > 0:
        wave_h = holo_m.forward_wave(concept_phases[last_tid])
        holo_scores = (torch.abs(concept_phases @ wave_h.conj()) * vocab_mask) ** 2
        wave_g = grammar_G @ concept_phases[last_tid]
        grammar_scores = (torch.abs(concept_phases @ wave_g.conj()) * vocab_mask) ** 2

    holo_probs = holo_scores / holo_scores.sum().clamp(min=1e-12)
    grammar_probs = grammar_scores / grammar_scores.sum().clamp(min=1e-12)

    combined = 0.3 * holo_probs + 0.7 * grammar_probs
    combined = combined / combined.sum()

    top5_vals, top5_ids = combined.topk(5)
    chosen_id = top5_ids[0].item()
    chosen_word = tokenizer.decode([chosen_id]).strip()

    print(f"\nStep {step+1}:")
    for rank in range(min(5, len(top5_ids))):
        tid = top5_ids[rank].item()
        word = tokenizer.decode([tid]).strip()
        prob = top5_vals[rank].item()
        holo_p = holo_probs[tid].item() if tid < V else 0
        gram_p = grammar_probs[tid].item() if tid < V else 0
        marker = " ***" if rank == 0 else ""
        print(f"  Rank {rank+1}: '{word}' (holo={holo_p:.4f} gram={gram_p:.4f} comb={prob:.4f}){marker}")

    generated_tokens.append(chosen_id)
    ids = torch.cat([ids, torch.tensor([[chosen_id]], device=DEV)], dim=1)

    cid = resolve_cid(chosen_word)
    if cid is not None:
        prev_id = ids[0, -2].item()
        prev_word = tokenizer.decode([prev_id]).strip()
        prev_cid = resolve_cid(prev_word)
        if prev_cid is not None:
            holo_m.bind(concept_phases[cid], concept_phases[prev_cid])

    if chosen_word in {'', '<|endoftext|>'} or '\n' in chosen_word:
        if step > 2:
            break

completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
completion_clean = ''.join(c for c in completion if ord(c) < 128)

print(f"\n{'='*60}")
print(f"PROMPT:\n{prompt}")
print(f"COMPLETION: {completion_clean}")
print(f"{'='*60}")
print("DONE.")
