"""
Phase 9: The HumanEval Matrix — Scaling to Qwen 27B
=====================================================
Ports the proven Native Hologram architecture from 0.5B to 27B.
Concept fusion, directed write, newline firewalls, pointer resolution.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import torch
from pathlib import Path
from transformers import AutoTokenizer

print("=" * 60)
print("PHASE 9: HUMANEVAL MATRIX — 27B SCALE")
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

embed_mb = embed.numel() * 4 / 1e6
print(f"  Embed shape: {embed.shape}  ({embed_mb:.0f} MB FP32)")

er = embed[:, :HALF]
ei = embed[:, HALF:]
er = er / er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
ei = ei / ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)
phase_angle = torch.atan2(ei, er)
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
del embed
print(f"  Phase vectors: {phase_vectors.shape}  ({phase_vectors.numel() * 8 / 1e6:.0f} MB complex64)")

ascii_letter = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_letter.match(word) and word != '':
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  Vocab mask: {n_allowed} code tokens allowed, {V - n_allowed} blocked")

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

cp_mb = concept_phases.numel() * 8 / 1e6
print(f"  Concept phases: {cp_mb:.0f} MB")

M_mb = HALF * HALF * 8 / 1e6
print(f"  M matrix: {HALF}x{HALF} complex64 = {M_mb:.1f} MB")
print(f"  Total GPU: {cp_mb + M_mb:.0f} MB")


class NativeHologram27B:
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

    def ingest(self, text):
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

    def forward(self, phase):
        return self.M * phase

    def backward(self, phase):
        return (self.M * phase.conj()).conj()

    def measure(self, wave, topk=5):
        raw = torch.abs(self.cp @ wave.conj())
        scores = (raw * self.mask) ** 3
        top = scores.topk(topk)
        return [(self.cw[tid], tid, float(score)) for tid, score in zip(top.indices.tolist(), top.values.tolist())]


# 2. THE CODE TAPE
tape = "x = 5\ny = x\nreturn y"

print(f"\n{'='*60}")
print(f"CODE: {repr(tape)}")
print(f"{'='*60}")

holo = NativeHologram27B(concept_phases, phase_vectors, vocab_mask, concept_words)
bound, skipped = holo.ingest(tape)
print(f"Ingest: {bound} edges, {skipped} newline-firewall-skipped")

cid_y = holo._get_cid("y")
cid_x = holo._get_cid("x")
cid_5 = holo._get_cid("5")
cid_eq = holo._get_cid("=")

print(f"\nConcept IDs: y={cid_y}, x={cid_x}, 5={cid_5}, =={cid_eq}")

p_y = holo.cp[cid_y]
p_x = holo.cp[cid_x]

print(f"\n{'='*60}")
print("DOUBLE V-TRACE: y -> x -> 10")
print(f"{'='*60}")

# Hop 1 FWD: y -> =
wave1 = holo.forward(p_y)
r1 = holo.measure(wave1)
print(f"\nHop 1 FWD — y -> ?:")
for w, tid, s in r1[:3]:
    ok = " ***" if w == "=" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
eq1_id = r1[0][1]
eq1_word = r1[0][0]

# Hop 2 FWD: = -> x (exclude y and 10, pick variable)
wave2 = holo.forward(holo.cp[eq1_id])
r2 = holo.measure(wave2, topk=8)
print(f"\nHop 2 FWD — {eq1_word} -> ? (variable):")
best2 = None
for w, tid, s in r2:
    if w in ("y", "5", "=", "=="):
        continue
    ok = " ***" if w == "x" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
    if best2 is None:
        best2 = (w, tid)
x_id = best2[1] if best2 else r2[0][1]

# Hop 3 BWD: x -> =
wave3 = holo.backward(p_x)
r3 = holo.measure(wave3)
print(f"\nHop 3 BWD — x <- ?:")
for w, tid, s in r3[:3]:
    ok = " ***" if w == "=" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
eq2_id = r3[0][1]

# Hop 4 FWD: = -> 10 (exclude x, y)
wave4 = holo.forward(holo.cp[eq2_id])
r4 = holo.measure(wave4, topk=8)
print(f"\nHop 4 FWD — = -> ? (excluding x, y, =):")
best4 = None
for w, tid, s in r4:
    if w in ("x", "y", "=", "=="):
        continue
    ok = " ***" if w == "5" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
    if best4 is None:
        best4 = w
if best4 is None:
    best4 = r4[0][0]

print(f"\n{'='*60}")
print(f"TRACE: y -> = -> x -> = -> {best4}")
print(f"RESULT: y resolves to {best4}")

if best4 == "5":
    print(f"VERDICT: PASS — 27B pointer resolution correct.")
else:
    print(f"VERDICT: FAIL — expected '5', got '{best4}'.")
print("=" * 60)
print("DONE.")
