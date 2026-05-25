"""
Phase 11: HYBRID ENGINE — Dual-Resonance Kuramoto Drive
=========================================================
Fuses Native Hologram (M vector, HRR variable tracking) with
a matrix grammar projector from code corpus transitions.

Memory:  Wave_M = M * Phase_curr        (Hadamard, forward unbind)
Grammar: Wave_H = G_mat @ Phase_curr    (Matrix-vector, outer-product syntax)
Fused:   Wave_F = 0.4 * Wave_M + 0.6 * Wave_H
Collapse: Token = argmax(|Vocab @ Wave_F*|)

M is a Hadamard state (HALF-dim vector) using directional binding.
G_mat is a (HALF x HALF) matrix built from outer-product code transitions.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = D_MODEL // 2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {',', 'the', 'to', 'is', '?', '!', '-', ''}
HOLO_NPZ = Path(__file__).parent.parent / "distilled" / "eigenbuddy_distilled.holo.npz"
HOLO_JSON = Path(__file__).parent.parent / "distilled" / "eigenbuddy_distilled.json"

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
print("PHASE 11: HYBRID ENGINE — DUAL-RESONANCE KURAMOTO DRIVE")
print("=" * 60)

print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
V = tokenizer.vocab_size
print(f"  Vocab: {V}")

print("Loading Qwen 27B embed_tokens...")
import safetensors.torch as st

embed = None
for sp in sorted(MODEL_DIR.glob("model-*.safetensors")):
    tensors = st.load_file(str(sp))
    for k in tensors:
        if "embed_tokens" in k:
            embed = tensors[k][:V, :D_MODEL].float()
            break
    if embed is not None:
        break

if embed is None:
    raise FileNotFoundError(f"embed_tokens not found in {MODEL_DIR}")

er = embed[:, :HALF]
ei = embed[:, HALF:]
er = er / er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
ei = ei / ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)
phase_angle = torch.atan2(ei, er)
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
del embed

token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>!]')
code_words_in_corpus = set()
for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
    code_words_in_corpus.add(m.group())
EXTRA_SYMBOLS = {'+', '*', '/', '-', '%', '=', '<', '>', '!', '&', '|', '^', '~', '#', '@', '$'}
code_words_in_corpus |= EXTRA_SYMBOLS
code_words_in_corpus.add(' ')  # space is important for token boundaries

vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if word in code_words_in_corpus:
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  Vocab mask: {n_allowed} code-corpus tokens, {V - n_allowed} blocked")
symbol_hits = {s: sum(1 for tid in range(V) if tokenizer.decode([tid]).strip() == s) for s in EXTRA_SYMBOLS}
print(f"  Symbol availability: {', '.join(f'{k}={v}' for k,v in symbol_hits.items())}")

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


def resolve_word_to_cid(word, cw, mask):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    for tid in range(len(cw)):
        if cw[tid] == word and mask[tid] > 0:
            return tid
    return ids[0] if ids[0] < len(cw) and mask[ids[0]] > 0 else None


grammar_G = None
holo_available = HOLO_NPZ.exists() and HOLO_JSON.exists()

if holo_available:
    print("Loading distilled .holo matrices...")
    holo = np.load(str(HOLO_NPZ))
    meta = json.load(open(str(HOLO_JSON)))
    print(f"  Loaded {len(holo.files)} phase gratings, {len(meta)} metadata entries")

    Q_stack = []
    K_stack = []
    for key in holo.files:
        g = torch.tensor(holo[key])
        if g.shape[1] != D_MODEL:
            continue
        if g.is_complex():
            g_comp = g.to(DEV)
        else:
            g_comp = torch.complex(g[:, :HALF], g[:, HALF:]).to(DEV)
        if 'k_proj' in key:
            K_stack.append(g_comp[:, :HALF])
        elif 'v_proj' in key:
            Q_stack.append(g_comp[:, :HALF])

    if Q_stack and K_stack:
        Q_all = torch.cat([q for q in Q_stack], dim=0)
        K_all = torch.cat([k for k in K_stack], dim=0)
        Q_all = Q_all / (Q_all.norm(dim=-1, keepdim=True) + 1e-12)
        K_all = K_all / (K_all.norm(dim=-1, keepdim=True) + 1e-12)
        G_holo = (K_all.T.conj() @ Q_all) / K_all.shape[0]
        G_holo = G_holo.to(DEV)
        g_strength = G_holo.abs().mean().item()
        print(f"  .holo grammar: {HALF}x{HALF}, K={K_all.shape[0]}+V={Q_all.shape[0]} modes, |G|={g_strength:.4f}")
        if g_strength > 0.01:
            grammar_G = G_holo
            print(f"  Using .holo grammar projector (signal sufficient).")
        else:
            print(f"  .holo signal too weak (|G|={g_strength:.4f} < 0.01). Falling back to n-gram.")
    else:
        print(f"  No matching k_proj/v_proj found (K={len(K_stack)}, V={len(Q_stack)}).")

if grammar_G is None:
    print("Building outer-product grammar matrix from code corpus...")
    code_token_ids = []
    for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
        word = m.group()
        cid = resolve_word_to_cid(word, concept_words, vocab_mask)
        if cid is not None:
            code_token_ids.append(cid)
    print(f"  Resolved {len(code_token_ids)} code tokens from corpus")

    grammar_G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
    ngram_count = 0
    for i in range(len(code_token_ids) - 1):
        pi, ci = code_token_ids[i], code_token_ids[i + 1]
        grammar_G += torch.outer(concept_phases[ci], concept_phases[pi].conj())
        ngram_count += 1
    grammar_G = grammar_G / len(code_token_ids)
    print(f"  Grammar matrix: {HALF}x{HALF} complex64, {ngram_count} outer-product transitions")


class HybridEngine:
    def __init__(self, cp, pv, mask, cw, grammar_G):
        self.cp = cp
        self.pv = pv
        self.mask = mask
        self.cw = cw
        self.G = grammar_G
        self.HALF = cp.shape[1]
        self.M = torch.zeros(self.HALF, dtype=torch.complex64, device=cp.device)

    def _get_cid(self, word):
        return resolve_word_to_cid(word, self.cw, self.mask)

    def _get_phase(self, word):
        cid = self._get_cid(word)
        return self.cp[cid] if cid is not None else None

    def ingest_prompt(self, text):
        lines = text.split('\n')
        cids = []
        for line in lines:
            words = line.split()
            for w in words:
                clean = w.strip('.,!?;:')
                if clean in EXCLUDE:
                    continue
                cid = self._get_cid(clean)
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

    def memory_wave(self, phase):
        return self.M * phase

    def grammar_wave(self, phase):
        return self.G @ phase

    def measure(self, wave, topk=5):
        raw = torch.abs(self.cp @ wave.conj())
        scores = raw ** 2
        masked = scores * self.mask
        top = masked.topk(topk)
        return [(self.cw[int(tid)], int(tid), float(masked[int(tid)]), float(raw[int(tid)]))
                for tid in top.indices.tolist()]

    def bind(self, phase_curr, phase_prev):
        self.M += phase_curr * phase_prev.conj()


prompt = "def add ( a , b ) : \n return a"

print(f"\n{'='*60}")
print(f"PROMPT: {repr(prompt)}")
print(f"{'='*60}")

engine = HybridEngine(concept_phases, phase_vectors, vocab_mask, concept_words, grammar_G)
bound, skipped = engine.ingest_prompt(prompt)
print(f"Ingest: {bound} directed edges, {skipped} firewall-skipped")
print(f"|M| mean: {engine.M.abs().mean().item():.4f}  |G| mean: {engine.G.abs().mean().item():.4f}")

p_curr = engine._get_phase("a")
current_word = "a"
generated = []
skip_set = {current_word, "def", "return", ":", "(", ")", ","}

print(f"\n{'='*60}")
print("DUAL-RESONANCE KURAMOTO DRIVE (3 tokens)")
print(f"{'='*60}")

for step in range(3):
    wave_M = engine.memory_wave(p_curr)
    wave_H = engine.grammar_wave(p_curr)
    wave_F = 0.4 * wave_M + 0.6 * wave_H

    rM = engine.measure(wave_M, topk=3)
    rH = engine.measure(wave_H, topk=3)
    rF = engine.measure(wave_F, topk=6)

    r1_word, r1_id, r1_score, r1_raw = rF[0]
    r2_word = rF[1][0] if len(rF) > 1 else ""
    r3_word = rF[2][0] if len(rF) > 2 else ""

    print(f"\nStep {step+1}: current='{current_word}'")
    print(f"  Memory:   {', '.join(f'{w}({s:.1e})' for w,_,s,_ in rM[:3])}")
    print(f"  Grammar:  {', '.join(f'{w}({s:.1e})' for w,_,s,_ in rH[:3])}")
    print(f"  Combined: '{r1_word}' (s={r1_score:.1e})  #2 '{r2_word}'  #3 '{r3_word}'")

    chosen_word = r1_word
    chosen_id = r1_id
    for w, tid, s, raw in rF:
        if w in skip_set:
            continue
        chosen_word = w
        chosen_id = tid
        break

    generated.append(chosen_word)
    p_new = engine.cp[chosen_id]
    engine.bind(p_new, p_curr)

    print(f"  >> Generated: '{chosen_word}'")
    p_curr = p_new
    current_word = chosen_word
    skip_set.add(chosen_word)

print(f"\n{'='*60}")
completion = " ".join(generated)
print(f"PROMPT:   {prompt}")
print(f"COMPLETE: {prompt} {completion}")
print(f"{'='*60}")
print("DONE.")
