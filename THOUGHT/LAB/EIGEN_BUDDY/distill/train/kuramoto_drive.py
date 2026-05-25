"""
Phase 10: Kuramoto Autoregressive Drive — Autonomous Code Generation
======================================================================
Transitions from manual V-trace to autonomous state generation.
Kuramoto-style coherence threshold governs token selection.
Each generated token is bound back into M, evolving the hologram.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import torch
from pathlib import Path
from transformers import AutoTokenizer

print("=" * 60)
print("PHASE 10: KURAMOTO AUTOREGRESSIVE DRIVE")
print("=" * 60)

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = D_MODEL // 2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {',', 'the', 'to', 'is', '?', '!', '-', ''}

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

ascii_letter = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_letter.match(word) and word != '':
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  Vocab mask: {n_allowed} allowed, {V - n_allowed} blocked")

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
print(f"  M matrix: 512x512 complex64 = 2.1 MB")


class KuramotoDrive:
    def __init__(self, cp, pv, mask, cw):
        self.cp = cp
        self.pv = pv
        self.mask = mask
        self.cw = cw
        self.HALF = cp.shape[1]
        self.M = torch.zeros(self.HALF, dtype=torch.complex64, device=cp.device)

    def _get_cid(self, word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        for tid in range(len(self.cw)):
            if self.cw[tid] == word and self.mask[tid] > 0:
                return tid
        return ids[0] if ids[0] < len(self.cw) and self.mask[ids[0]] > 0 else None

    def _get_phase(self, word):
        cid = self._get_cid(word)
        return self.cp[cid] if cid is not None else None

    def _measure(self, wave, power=3, topk=5):
        raw = torch.abs(self.cp @ wave.conj())
        scores = (raw * self.mask) ** power
        top = scores.topk(topk)
        return [(self.cw[tid], tid, float(score), float(raw[tid]))
                for tid, score in zip(top.indices.tolist(), top.values.tolist())]

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

        bound = 0
        for i in range(len(cids) - 1):
            pi, ci = cids[i], cids[i + 1]
            if pi < 0 or ci < 0:
                continue
            self.M += self.cp[ci] * self.cp[pi].conj()
            bound += 1
        return bound

    def forward(self, phase):
        return self.M * phase

    def bind(self, phase_curr, phase_prev):
        self.M += phase_curr * phase_prev.conj()


# 2. THE PROMPT
prompt = "def add ( a , b ) : \n return a"

print(f"\n{'='*60}")
print(f"PROMPT: {prompt}")
print(f"{'='*60}")

drive = KuramotoDrive(concept_phases, phase_vectors, vocab_mask, concept_words)
bound = drive.ingest_prompt(prompt)
print(f"Ingest: {bound} edges bound into M")

# 3. KURAMOTO AUTOREGRESSIVE DRIVE
print(f"\n{'='*60}")
print("AUTOREGRESSIVE GENERATION (3 tokens)")
print(f"{'='*60}")

# Start from last token of prompt: 'a'
p_prev = drive._get_phase("a")
current_word = "a"
generated = []

for step in range(3):
    wave = drive.forward(p_prev)
    results = drive._measure(wave, power=3, topk=5)

    r1_word, r1_id, r1_score, r1_raw = results[0]
    r2_word = results[1][0] if len(results) > 1 else ""

    print(f"\nStep {step+1}: current='{current_word}'")
    print(f"  Rank 1: '{r1_word}'  (r={r1_raw:.0f}, r^3={r1_score:.1e})")
    print(f"  Rank 2: '{r2_word}'")

    # Exclude self-loop and common structural tokens from generated output
    skip_set = {current_word, "def", "return", ":", "(", ")", ",", "."}
    chosen_word = r1_word
    chosen_id = r1_id
    for w, tid, s, raw in results:
        if w in skip_set:
            continue
        chosen_word = w
        chosen_id = tid
        break

    generated.append(chosen_word)
    p_new = drive.cp[chosen_id]

    # Bind new token back into M (autoregressive memory update)
    drive.bind(p_new, p_prev)

    print(f"  >> Generated: '{chosen_word}'")
    p_prev = p_new
    current_word = chosen_word

print(f"\n{'='*60}")
print(f"PROMPT: {prompt}")
print(f"GENERATED: {' '.join(generated)}")
print(f"{'='*60}")
print("DONE.")
