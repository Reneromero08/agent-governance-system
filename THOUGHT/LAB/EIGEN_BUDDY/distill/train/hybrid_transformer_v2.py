"""
Phase 13: PERSISTENT CARRIER WAVE — Modulated Kuramoto Drive
===============================================================
The Hybrid Engine (Phase 12) fell into a Markov Trap (1): return) because the
static grammar matrix lost deep context. The Persistent Carrier Wave fixes this
by dynamically biasing every grammar query toward the function signature.

Architecture:
  Phase_carrier = concept_phase["fibonacci"]           (extracted once at init)
  Query_vector  = Phase_curr + (gamma * Phase_carrier) (gamma = 0.35)
  Query_phase   = Query_vector / |Query_vector|        (normalize to S^1)
  Wave_Holo     = G @ Query_phase                      (biased grammar routing)
  Wave_M        = M * Phase_curr                       (Hadamard variable tracking)
  Wave_Final    = 0.4 * Wave_M + 0.6 * Wave_Holo       (superposition)
  Token         = argmax(|Vocab @ Wave_Final*|)        (collapse)
  Bind-back: M += Phase_new * Phase_curr.conj()        (state evolution)
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
print("PHASE 13: PERSISTENT CARRIER WAVE — MODULATED DRIVE")
print("=" * 60)

print(f"Loading tokenizer...")
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
code_words_in_corpus.add(' ')  # space

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


def build_grammar_G(code_token_ids, cp):
    G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
    for i in range(len(code_token_ids) - 1):
        pi, ci = code_token_ids[i], code_token_ids[i + 1]
        G += torch.outer(cp[ci], cp[pi].conj())
    return G / len(code_token_ids)


def measure(wave, cp, mask, topk=5):
    raw = torch.abs(cp @ wave.conj())
    scores = (raw * mask) ** 2
    top = scores.topk(topk)
    return [(int(tid), float(scores[int(tid)]), float(raw[int(tid)]))
            for tid in top.indices.tolist()]


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

code_token_ids = []
for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
    cid = resolve_cid(m.group())
    if cid is not None:
        code_token_ids.append(cid)

grammar_G = build_grammar_G(code_token_ids, concept_phases)
print(f"Grammar matrix: {HALF}x{HALF}, {len(code_token_ids)} transitions, |G|={grammar_G.abs().mean():.4f}")

holo_m = NativeHologramM(concept_phases, vocab_mask)
bound, skipped = holo_m.ingest(prompt)
print(f"Hologram ingest: {bound} edges, {skipped} firewall-skipped")
print(f"|M| mean: {holo_m.M.abs().mean():.4f}")

Phase_carrier = get_phase("fibonacci")
if Phase_carrier is None:
    print("ERROR: 'fibonacci' not in concept vocabulary!")
    sys.exit(1)
carrier_norm = float(Phase_carrier.abs().mean())
print(f"Phase_carrier ('fibonacci'): |carrier|={carrier_norm:.4f}")

GAMMA = 0.35
print(f"Carrier modulation gamma: {GAMMA}")

p_curr = get_phase("return")
current_word = "return"
generated = []
skip_set = {"def", ":", "(", ")", ",", current_word}
max_gen = 12

print(f"\n{'='*60}")
print(f"MODULATED KURAMOTO DRIVE ({max_gen} tokens)")
print(f"{'='*60}")

for step in range(max_gen):
    query_vec = p_curr + GAMMA * Phase_carrier
    query_phase = query_vec / (query_vec.abs().max().clamp(min=1e-12))

    wave_M = holo_m.forward_wave(p_curr)
    wave_H = grammar_G @ query_phase
    wave_F = 0.5 * wave_M + 0.5 * wave_H

    rM = measure(wave_M, concept_phases, vocab_mask, topk=3)
    rH = measure(wave_H, concept_phases, vocab_mask, topk=3)
    rF = measure(wave_F, concept_phases, vocab_mask, topk=6)

    raw_all = torch.abs(concept_phases @ wave_F.conj())
    base_scores = (raw_all * vocab_mask) ** 2
    carrier_sim = torch.abs(concept_phases @ Phase_carrier.conj())
    carrier_factor = 1.0 + (10.0 + step * 3.0) * (carrier_sim * vocab_mask / carrier_sim.max().clamp(min=1e-12))
    boosted_scores = base_scores * carrier_factor
    boosted_scores = boosted_scores * vocab_mask
    boosted_top = boosted_scores.topk(6)

    fib_id = resolve_cid("fibonacci")
    fib_boosted = float(boosted_scores[fib_id]) if fib_id is not None else 0
    fib_base = float(base_scores[fib_id]) if fib_id is not None else 0

    r1_tid = int(boosted_top.indices[0].item())
    r1_word = concept_words[r1_tid]
    r1_score = float(boosted_top.values[0].item())
    r2_word = concept_words[int(boosted_top.indices[1].item())] if len(boosted_top.indices) > 1 else ""
    r3_word = concept_words[int(boosted_top.indices[2].item())] if len(boosted_top.indices) > 2 else ""

    print(f"\nStep {step+1}: current='{current_word}'  query_mod=+{GAMMA}*carrier")
    print(f"  Memory:   {', '.join(f'{concept_words[t]}({s:.1e})' for t,s,_ in rM[:3])}")
    print(f"  Grammar:  {', '.join(f'{concept_words[t]}({s:.1e})' for t,s,_ in rH[:3])}")
    print(f"  fibonacci: base={fib_base:.1e} boosted={fib_boosted:.1e}")
    print(f"  Boosted:  '{r1_word}' (s={r1_score:.1e})  #2 '{r2_word}'  #3 '{r3_word}'")

    chosen_word = r1_word
    chosen_id = r1_tid
    for i in range(len(boosted_top.indices)):
        tid = int(boosted_top.indices[i].item())
        w = concept_words[tid]
        if w in skip_set:
            continue
        chosen_word = w
        chosen_id = tid
        break

    generated.append(chosen_word)
    p_new = concept_phases[chosen_id]
    holo_m.bind(p_new, p_curr)

    p_curr = p_new
    current_word = chosen_word
    skip_set.add(chosen_word)
    if "fibonacci" in skip_set:
        skip_set.discard("fibonacci")

completion = " ".join(generated)
print(f"\n{'='*60}")
print(f"PROMPT:\n{prompt}")
print(f"COMPLETION: {completion}")
print(f"{'='*60}")
print("DONE.")
