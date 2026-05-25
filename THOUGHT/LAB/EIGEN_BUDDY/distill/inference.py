"""
Phase 17: INFERENCE ENGINE — Production Lock (Coefficient Tuning)
===================================================================
Wraps the Superradiant Transformer architecture with tuned coefficients.
During decoherence delay (gamma=0 vacuum), applies 3.0x grammar boost
to surface linking syntax (1, )) that the hologram would otherwise suppress.

Usage:
  python inference.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

BASE = Path(__file__).parent
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

GRAMMAR_BOOST_VACUUM = 5.0
HOLO_WEIGHT = 0.40
GRAM_WEIGHT = 0.60
VACUUM_HOLO = 0.15
VACUUM_GRAM = 0.85


class InferenceEngine:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        V = self.tokenizer.vocab_size
        self._load_embeddings(V)
        self._build_vocabulary(V)
        self._load_holo_matrices()
        self._build_model(V)

    def _load_embeddings(self, V):
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
        self.phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
        self.embed_weight = embed_weight
        self.lm_head_weight = lm_head_weight

    def _build_vocabulary(self, V):
        token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>!]')
        code_words = set()
        for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
            code_words.add(m.group())
        for sym in {'+', '*', '/', '-', '%', '=', '<', '>', '!', '&', '|', '^', '~', '#', '@', '$'}:
            code_words.add(sym)

        self.vocab_mask = torch.zeros(V, device=DEV)
        for tid in range(V):
            if self.tokenizer.decode([tid]).strip() in code_words:
                self.vocab_mask[tid] = 1.0

        n_allowed = int(self.vocab_mask.sum().item())
        self.concept_phases = torch.zeros(V, HALF, dtype=torch.complex64, device=DEV)
        self.concept_words = [""] * V
        for tid in range(V):
            if self.vocab_mask[tid] == 0:
                continue
            word = self.tokenizer.decode([tid]).strip()
            self.concept_words[tid] = word
            sub_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not sub_ids:
                continue
            cp = self.phase_vectors[sub_ids[0]].clone()
            for sid in sub_ids[1:]:
                cp = cp * self.phase_vectors[sid]
            self.concept_phases[tid] = cp

        self._resolve_cid_cache = {}
        code_token_ids = []
        for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
            cid = self._resolve_cid(m.group())
            if cid is not None:
                code_token_ids.append(cid)
        self.grammar_G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
        for i in range(len(code_token_ids) - 1):
            pi, ci = code_token_ids[i], code_token_ids[i + 1]
            self.grammar_G += torch.outer(self.concept_phases[ci], self.concept_phases[pi].conj())
        self.grammar_G = self.grammar_G / len(code_token_ids)

    def _resolve_cid(self, word):
        if word in self._resolve_cid_cache:
            return self._resolve_cid_cache[word]
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        for tid in range(len(self.concept_words)):
            if self.concept_words[tid] == word and self.vocab_mask[tid] > 0:
                self._resolve_cid_cache[word] = tid
                return tid
        tid = ids[0] if ids[0] < len(self.concept_words) and self.vocab_mask[ids[0]] > 0 else None
        self._resolve_cid_cache[word] = tid
        return tid

    def _get_phase(self, word):
        cid = self._resolve_cid(word)
        return self.concept_phases[cid] if cid is not None else None

    def _load_holo_matrices(self):
        self.holo = np.load(str(HOLO_NPZ))
        self.meta = json.load(open(str(HOLO_JSON)))

    def _build_model(self, V):
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

        self.model = CatalyticLM(V, D_MODEL, N_HEADS)
        self.model.er.weight.data.copy_(self.embed_weight.float())
        self.model.ei.weight.data.zero_()
        self.model.out.weight.data.copy_(self.lm_head_weight.float())

        k_gratings, v_gratings = [], []
        for key in self.holo.files:
            g = torch.tensor(self.holo[key])
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
                    w = getattr(self.model.attn, pn).weight
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

        self.model = self.model.to(DEV)
        self.model.train()

    def _sum_phases(self, tokens):
        result = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
        for t in tokens:
            p = self._get_phase(t)
            if p is not None:
                result = result + p
        return result / (result.abs().max().clamp(min=1e-12))

    def _compute_scores(self, wave):
        raw = torch.abs(self.concept_phases @ wave.conj())
        return (raw * self.vocab_mask) ** 2

    def generate(self, prompt, max_tokens=25, intent_phase=None, params_list=None, cassette=None):
        if intent_phase is None:
            intent_phase = self._get_phase("fibonacci")
        if params_list is None:
            params_list = ["n"]

        holo_m = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
        lines = prompt.split('\n')
        cids = []
        for line in lines:
            words = line.split()
            for w in words:
                clean = w.strip('.,!?;:')
                if clean in EXCLUDE:
                    continue
                cid = self._resolve_cid(clean)
                if cid is not None:
                    cids.append(cid)
            cids.append(-1)
        if cids and cids[-1] < 0:
            cids.pop()
        for i in range(len(cids) - 1):
            pi, ci = cids[i], cids[i + 1]
            if pi < 0 or ci < 0:
                continue
            holo_m += self.concept_phases[ci] * self.concept_phases[pi].conj()

        prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
        ids = prompt_ids.clone()

        Phase_carrier = intent_phase
        carrier_shifted = False
        intent_consumed = False
        params_consumed = False
        carrier_active = {"intent"}
        anneal_offset = 0
        GAMMA = 0.35
        delay_steps = 0
        next_carrier = None
        skip_set = {"def", ":", ",", "return"}
        generated = []

        for step in range(max_tokens):
            logits = self.model(ids)
            last_logits = logits[0, -1, :]
            last_tid = ids[0, -1].item()

            holo_scores = torch.zeros(len(self.concept_words), device=DEV)
            gram_scores = torch.zeros(len(self.concept_words), device=DEV)
            carrier_scores = torch.zeros(len(self.concept_words), device=DEV)

            if last_tid < len(self.concept_words) and self.vocab_mask[last_tid] > 0:
                cp_last = self.concept_phases[last_tid]
                wave_M = holo_m * cp_last
                holo_scores = self._compute_scores(wave_M)

                if cassette is not None:
                    wave_G = cassette * cp_last
                else:
                    query_vec = cp_last + GAMMA * Phase_carrier
                    query_phase = query_vec / (query_vec.abs().max().clamp(min=1e-12))
                    wave_G = self.grammar_G @ query_phase
                gram_scores = self._compute_scores(wave_G)

                anneal_step = step - anneal_offset
                carrier_boost = (10.0 + anneal_step * 3.0)
                carrier_raw = torch.abs(self.concept_phases @ Phase_carrier.conj())
                carrier_scores = carrier_boost * (carrier_raw * self.vocab_mask) ** 2

            holo_probs = holo_scores / holo_scores.sum().clamp(min=1e-12)
            gram_probs = gram_scores / gram_scores.sum().clamp(min=1e-12)
            carrier_probs = carrier_scores / carrier_scores.sum().clamp(min=1e-12)

            attn_probs = torch.softmax(last_logits / 0.8, dim=-1)
            attn_probs = attn_probs * self.vocab_mask
            attn_probs = attn_probs / attn_probs.sum()

            in_vacuum = (delay_steps > 0 or GAMMA == 0.0)

            if in_vacuum:
                gram_probs = gram_probs * GRAMMAR_BOOST_VACUUM
                gram_probs = gram_probs / gram_probs.sum().clamp(min=1e-12)
                combined = VACUUM_HOLO * holo_probs + VACUUM_GRAM * gram_probs
            else:
                combined = attn_probs * 0.05 + holo_probs * HOLO_WEIGHT + gram_probs * GRAM_WEIGHT + carrier_probs * 0.55

            combined = combined / combined.sum()
            top5_vals, top5_ids = combined.topk(6)
            r1_tid = int(top5_ids[0].item())
            r1_word = self.concept_words[r1_tid]
            r1_score = float(top5_vals[0].item())

            chosen_word = r1_word
            chosen_id = r1_tid
            for i in range(len(top5_ids)):
                tid = int(top5_ids[i].item())
                w = self.concept_words[tid]
                if w in skip_set:
                    continue
                chosen_word = w
                chosen_id = tid
                break

            generated.append(chosen_word)
            cp_new = self.concept_phases[chosen_id]
            cp_prev = self.concept_phases[last_tid]
            holo_m += cp_new * cp_prev.conj()

            if cassette is not None:
                theta = math.pi * 0.6180339887498949
                U = complex(math.cos(theta), math.sin(theta))
                cassette = cassette * U
                if last_tid < len(self.concept_phases) and self.vocab_mask[last_tid] > 0:
                    cassette = cassette - 0.3 * self.concept_phases[last_tid]
                    cassette = cassette / (cassette.abs().max().clamp(min=1e-12))

            if not intent_consumed:
                intent_consumed = True
                carrier_active = set(params_list[:3])
                carrier_active.discard("")
                if carrier_active:
                    Phase_carrier = self._sum_phases(carrier_active)
                    carrier_shifted = True
                    anneal_offset = step + 1

            if carrier_shifted and chosen_word in carrier_active:
                carrier_active.discard(chosen_word)
                if carrier_active:
                    Phase_carrier = self._sum_phases(carrier_active)
                    anneal_offset = step + 1
                else:
                    params_consumed = True
                    delay_steps = 2
                    GAMMA = 0.0
                    carrier_shifted = False

            if delay_steps > 0:
                delay_steps -= 1
                if delay_steps == 0:
                    GAMMA = 0.0

            skip_set.add(chosen_word)
            if carrier_shifted:
                skip_set.discard(":")

            new_tok = torch.tensor([[chosen_id]], device=DEV)
            ids = torch.cat([ids, new_tok], dim=1)

        return generated


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 17: INFERENCE ENGINE — PRODUCTION LOCK")
    print("=" * 60)

    engine = InferenceEngine()
    print(f"Grammar boost (vacuum): {GRAMMAR_BOOST_VACUUM}x  depth_map: [1->'1', 2->'2']")
    print(f"Weights: holo={HOLO_WEIGHT} gram={GRAM_WEIGHT}")
    print(f"Vacuum: holo={VACUUM_HOLO} gram={VACUUM_GRAM}")

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

    tokens = engine.generate(prompt, max_tokens=25)
    completion = " ".join(tokens)
    print(f"\nCOMPLETION: {completion}")
    print(f"{'='*60}")
    print("DONE.")
