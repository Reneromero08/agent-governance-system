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
import torch.nn.functional as F
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
HOLO_WEIGHT = 0.45
GRAM_WEIGHT = 0.25
ATTN_WEIGHT = 0.15
CARR_WEIGHT = 0.15
VACUUM_HOLO = 0.55
VACUUM_GRAM = 0.30
VACUUM_ATTN = 0.15
M_DEPLETE = 0.6
VSA_HOLO = 0.30
VSA_GRAM = 0.20
VSA_CARR = 0.50
VSA_RESONANCE = 0.85
VSA_TIMEOUT = 3

PUSH_TOKENS = {"(", "[", "{", "if", "for", "def", "while", "try", "class", "with", "lambda"}
POP_TOKENS = {")", "]", "}", "else", "return", "break", "continue", "pass", "except", "finally"}
VACUUM_HOLO = 0.55
VACUUM_GRAM = 0.30
VACUUM_ATTN = 0.15
M_DEPLETE = 0.6


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
        class CatalyticTensorLM:
            def __init__(self, embed_weight, lm_head_weight, attn):
                self.er_w = embed_weight.float()
                self.ei_w = torch.zeros_like(embed_weight)
                self.attn = attn
                self.lm_head = lm_head_weight.float()

            def forward(self, ids):
                x_r = self.er_w[ids]
                x_i = self.ei_w[ids]
                x = torch.complex(x_r, x_i)
                z, _ = self.attn(x)
                return z.real @ self.lm_head.T

            def to(self, dev):
                self.er_w = self.er_w.to(dev)
                self.ei_w = self.ei_w.to(dev)
                self.lm_head = self.lm_head.to(dev)
                self.attn = self.attn.to(dev)
                return self

        from core.attention import MultiHeadComplexAttention
        attn = MultiHeadComplexAttention(D_MODEL, N_HEADS, geo_init=False)

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

        attn = attn.to(DEV)
        self.model = CatalyticTensorLM(self.embed_weight, self.lm_head_weight, attn)
        self.model.to(DEV)

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

    def generate(self, prompt, max_tokens=25, intent_phase=None, params_list=None,
                 cassette=None, ref_phase=None, local_var_phases=None, local_var_names=None,
                 vsa_fsm=None):
        if intent_phase is None:
            intent_phase = self._get_phase("fibonacci")
        if params_list is None:
            params_list = ["n"]
        if local_var_phases is None:
            local_var_phases = []
        if local_var_names is None:
            local_var_names = []

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

        for var_phase in local_var_phases:
            var_phase_dev = var_phase.to(DEV) if var_phase.device != holo_m.device else var_phase
            holo_m += var_phase_dev * var_phase_dev.conj()
            for other_phase in local_var_phases:
                if other_phase is var_phase:
                    continue
                other_dev = other_phase.to(DEV) if other_phase.device != holo_m.device else other_phase
                holo_m += 0.5 * var_phase_dev * other_dev.conj()

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
        vsa_trigger = "start"
        vsa_state = "init"
        vsa_timeout = 0
        stack_S = torch.zeros(HALF, dtype=torch.complex64, device=DEV)

        for step in range(max_tokens):
            logits = self.model.forward(ids)
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
                elif vsa_fsm is not None:
                    wave_G = self.grammar_G @ cp_last
                else:
                    query_vec = cp_last + GAMMA * Phase_carrier
                    query_phase = query_vec / (query_vec.abs().max().clamp(min=1e-12))
                    wave_G = self.grammar_G @ query_phase

                if vsa_fsm is not None:
                    try:
                        vsa_wave = vsa_fsm.query(vsa_trigger, vsa_state)
                        vsa_wave = vsa_wave / (vsa_wave.abs().max().clamp(min=1e-12))
                        vsa_scores = self._compute_scores(vsa_wave)
                        carrier_scores = vsa_scores
                    except Exception:
                        pass

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
                combined = VACUUM_HOLO * holo_probs + VACUUM_GRAM * gram_probs + VACUUM_ATTN * attn_probs
            else:
                combined = ATTN_WEIGHT * attn_probs + HOLO_WEIGHT * holo_probs + GRAM_WEIGHT * gram_probs + CARR_WEIGHT * carrier_probs

            combined = combined / combined.sum()
            if vsa_fsm is not None:
                holo_probs = holo_probs / holo_probs.sum().clamp(min=1e-12)
                carrier_probs = carrier_probs / carrier_probs.sum().clamp(min=1e-12)
                combined = 0.65 * carrier_probs + 0.35 * holo_probs
                if stack_S.abs().max() > 1e-12:
                    stack_wave = self.grammar_G @ stack_S
                    stack_scores = self._compute_scores(stack_wave)
                    stack_mask = (stack_scores > stack_scores.max() * 0.03).float()
                    gram_mask = (gram_scores > gram_scores.max() * 0.05).float()
                    combined_mask = stack_mask * gram_mask
                    if combined_mask.sum() > 0:
                        combined = combined * combined_mask
                else:
                    gram_mask = (gram_scores > gram_scores.max() * 0.05).float()
                    if gram_mask.sum() > 0:
                        combined = combined * gram_mask
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
            if chosen_id < len(self.concept_phases) and self.vocab_mask[chosen_id] > 0:
                depletion = M_DEPLETE + step * 0.02
                holo_m = holo_m - depletion * self.concept_phases[chosen_id]
                holo_m = holo_m / (holo_m.abs().max().clamp(min=1e-12))

            if cassette is not None:
                theta = math.pi * 0.6180339887498949
                U = complex(math.cos(theta), math.sin(theta))
                cassette = cassette * U
                if last_tid < len(self.concept_phases) and self.vocab_mask[last_tid] > 0:
                    cassette = cassette - 0.3 * self.concept_phases[last_tid]
                    cassette = cassette / (cassette.abs().max().clamp(min=1e-12))

                if ref_phase is not None:
                    c_norm = cassette / (cassette.abs().max().clamp(min=1e-12))
                    r_coh = float(torch.abs(torch.dot(c_norm.conj(), ref_phase))) / HALF
                    if r_coh < 0.7:
                        corr = (1.0 - r_coh) * (ref_phase - c_norm)
                        cassette = cassette + corr.to(cassette.device)
                        cassette = cassette / (cassette.abs().max().clamp(min=1e-12))

            if not intent_consumed:
                intent_consumed = True
                carrier_active = set(params_list[:3])
                carrier_active.discard("")
                if carrier_active:
                    Phase_carrier = self._sum_phases(carrier_active)
                elif local_var_phases:
                    Phase_carrier = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
                    for vp in local_var_phases:
                        Phase_carrier = Phase_carrier + vp.to(DEV)
                    Phase_carrier = Phase_carrier / (Phase_carrier.abs().max().clamp(min=1e-12))
                else:
                    Phase_carrier = None
                if Phase_carrier is not None:
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

            if params_consumed and not carrier_shifted and not delay_steps and local_var_phases:
                Phase_carrier = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
                for vp in local_var_phases:
                    Phase_carrier = Phase_carrier + vp.to(DEV)
                Phase_carrier = Phase_carrier / (Phase_carrier.abs().max().clamp(min=1e-12))
                carrier_shifted = True
                params_consumed = False
                carrier_active = set(local_var_names[:3])
                anneal_offset = step + 1

            skip_set.add(chosen_word)
            if carrier_shifted:
                skip_set.discard(":")

            if chosen_word in PUSH_TOKENS and chosen_id < len(self.concept_phases):
                cp_tok = self.concept_phases[chosen_id]
                stack_S = cp_tok + torch.roll(stack_S, shifts=1)
                stack_S = stack_S / (stack_S.abs().max().clamp(min=1e-12))
            elif chosen_word in POP_TOKENS and chosen_id < len(self.concept_phases):
                cp_tok = self.concept_phases[chosen_id]
                stack_S = torch.roll(stack_S, shifts=-1) - cp_tok
                stack_S = stack_S / (stack_S.abs().max().clamp(min=1e-12))

            if vsa_fsm is not None:
                if chosen_id < len(self.concept_phases) and self.vocab_mask[chosen_id] > 0:
                    gen_phase = self.concept_phases[chosen_id]
                    state_seed = vsa_fsm.states.get(vsa_state)
                    if state_seed is not None:
                        sim = float(torch.abs(torch.dot(state_seed.conj(), gen_phase))) / HALF
                        vsa_timeout += 1
                        if sim > VSA_RESONANCE or vsa_timeout >= VSA_TIMEOUT:
                            vsa_timeout = 0
                            trigger_vec = vsa_fsm.triggers.get(vsa_trigger)
                            if trigger_vec is not None:
                                next_noisy = vsa_fsm.query(vsa_trigger, vsa_state)
                                results = vsa_fsm.measure(next_noisy)
                                if results:
                                    vsa_state = results[0][0]
                                    if vsa_trigger == "start":
                                        vsa_trigger = "true"
                                    elif vsa_state in ("body", "true_body", "false_body"):
                                        vsa_trigger = "step"
                                    elif vsa_state in ("inc",):
                                        vsa_trigger = "step"
                                    elif vsa_state in ("cond",):
                                        vsa_trigger = "true"
                                    elif vsa_state in ("done", "end"):
                                        vsa_trigger = "done"

            new_tok = torch.tensor([[chosen_id]], device=DEV)
            ids = torch.cat([ids, new_tok], dim=1)

        return generated


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 17: INFERENCE ENGINE — PRODUCTION LOCK")
    print("=" * 60)

    engine = InferenceEngine()
    print(f"Grammar boost (vacuum): {GRAMMAR_BOOST_VACUUM}x  depth_map: [1->'1', 2->'2']")
    print(f"Weights: holo={HOLO_WEIGHT} gram={GRAM_WEIGHT} attn={ATTN_WEIGHT} carr={CARR_WEIGHT}  M_deplete={M_DEPLETE}")
    print(f"VSA mode: carrier=0.65 holo=0.35  grammar=multiplicative_mask  resonance>{VSA_RESONANCE}  timeout={VSA_TIMEOUT}")
    print(f"Vacuum: holo={VACUUM_HOLO} gram={VACUUM_GRAM} attn={VACUUM_ATTN}")

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
