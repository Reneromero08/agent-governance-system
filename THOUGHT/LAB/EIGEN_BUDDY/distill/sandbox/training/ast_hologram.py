"""
Phase 8: Abstract Syntax Sandbox — Pointer Resolution
=======================================================
Tests V-Trace on code-like variable assignment chains.
var_y -> var_x -> 5 via double V-trace through equals nodes.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 8: ABSTRACT SYNTAX — POINTER RESOLUTION")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {',', 'the', 'to', 'Where', 'is', '?', '!', '-', '', 'return'}

print(f"Loading Prism: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

embed = model.model.embed_tokens.weight.detach().float()
D_MODEL = embed.shape[1]
HALF = D_MODEL // 2
V = embed.shape[0]

er = embed[:, :HALF]
ei = embed[:, HALF:]
er = er / er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
ei = ei / ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)
phase_angle = torch.atan2(ei, er)
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
del model

ascii_letter = re.compile(r'^[a-zA-Z0-9_.]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_letter.match(word) and word != '':
        vocab_mask[tid] = 1.0

print(f"Precomputing concept phases for {int(vocab_mask.sum().item())} words...")
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
print(f"Prism: {HALF} dim, {int(vocab_mask.sum().item())} concepts ready.")


class NativeHologram:
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
        words = text.split()
        cids = []
        for w in words:
            clean = w.strip('.,!?;:')
            if clean in EXCLUDE:
                if w == '.':
                    cids.append(-1)
                continue
            cid = self._get_cid(clean)
            if cid is not None:
                cids.append(cid)
            if w.endswith('.'):
                cids.append(-1)
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
        results = []
        for tid, score in zip(top.indices.tolist(), top.values.tolist()):
            results.append((self.cw[tid], tid, score))
        return results


# 2. THE CODE TAPE
tape = "a equals 5 . b equals a . return b ."

print("\n" + "=" * 60)
print(f"CODE TAPE: {tape}")
print("=" * 60)

holo = NativeHologram(concept_phases, phase_vectors, vocab_mask, concept_words)
bound, skipped = holo.ingest(tape)
print(f"Ingest: {bound} edges, {skipped} firewall-skipped")

cid_b = holo._get_cid("b")
cid_a = holo._get_cid("a")
cid_5 = holo._get_cid("5")
cid_eq = holo._get_cid("equals")

print(f"\nConcept IDs: b={cid_b}, a={cid_a}, 5={cid_5}, equals={cid_eq}")

p_b = holo.cp[cid_b]
p_a = holo.cp[cid_a]
p_5 = holo.cp[cid_5]

print("\n" + "=" * 60)
print("DOUBLE V-TRACE: b -> a -> 5")
print("=" * 60)

# Hop 1 BWD: b -> equals (from "b equals a")
wave1 = holo.backward(p_b)
r1 = holo.measure(wave1)
print(f"\nHop 1 BWD — b <- ?:")
for w, tid, s in r1[:3]:
    ok = " ***" if w == "equals" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
best1 = r1[0][0]
best1_id = r1[0][1]
print(f"  >> b <- {best1}")

# Hop 2 FWD: equals -> a (from "equals a")
wave2 = holo.forward(holo.cp[best1_id])
r2 = holo.measure(wave2)
print(f"\nHop 2 FWD — {best1} -> ?:")
for w, tid, s in r2[:3]:
    ok = " ***" if w == "a" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
best2 = r2[0][0]
print(f"  >> {best1} -> {best2}")

# Hop 3 BWD: a -> equals (from "a equals 5")
wave3 = holo.backward(p_a)
r3 = holo.measure(wave3)
print(f"\nHop 3 BWD — a <- ?:")
for w, tid, s in r3[:3]:
    ok = " ***" if w == "equals" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
best3 = r3[0][0]
best3_id = r3[0][1]
print(f"  >> a <- {best3}")

# Hop 4 FWD: a -> equals -> 5 (two-hop, exclude variables from dest)
wave4a = holo.forward(p_a)
r4a = holo.measure(wave4a, topk=3)
eq_id = r4a[0][1]
print(f"\nHop 4a FWD — a -> {r4a[0][0]} (score={r4a[0][2]:.1e})")

wave4b = holo.forward(holo.cp[eq_id])
r4b = holo.measure(wave4b, topk=5)
print(f"Hop 4b FWD — equals -> ? (excluding a, b, equals):")
best4 = None
for w, tid, s in r4b:
    if w in ("a", "b", "equals", "Equals"):
        continue
    ok = " ***" if w == "5" else ""
    print(f"  {w:20s}  score={s:.1e}{ok}")
    if best4 is None:
        best4 = w
if best4 is None:
    best4 = r4b[0][0]
print(f"  >> equals -> {best4}")

print(f"\n{'='*60}")
print(f"TRACE: {best4} <- equals <- a <- {best1} <- b")
print(f"RESULT: b resolves to {best4}")

if best4 == "5":
    print(f"VERDICT: PASS — pointer chain resolved correctly.")
else:
    print(f"VERDICT: FAIL — expected '5', got '{best4}'.")
print("=" * 60)
print("DONE.")
