"""
Phase 3: Directional Time Binding — Conjugate-Enforced Arrow of Time
======================================================================
Fixes the bidirectional oscillation from Phase 2.5 by using complex
conjugation to create directed edges.

Write:  M += Phase_curr * Phase_prev.conj()    (prev -> curr)
Backward: Wave = (M.conj() * Phase_curr).conj()  (retrieve prev from curr)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 3: DIRECTIONAL TIME BINDING — 3-HOP BACKWARD TRACE")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {'.', ',', 'the', 'to', 'Where', 'is', '?'}

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

print(f"Prism Ready: {HALF} dim, {V} tokens on {DEV}.")

def measure(wave, label, power=3, topk=3):
    raw = torch.abs(phase_vectors @ wave.conj())
    scores = raw ** power
    top = scores.topk(topk)
    print(f"\n  {label}:")
    for rank, (tid, score) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
        word = tokenizer.decode([tid]).strip()
        raw_score = raw[tid].item()
        print(f"    Rank {rank+1}: '{word}'  (raw: {raw_score:.1f}, sharp: {score:.1e})")
    return top.indices[0].item()

# 2. THE DATA
story = "Mary went to the bathroom. John moved to the hallway. Mary dropped the football."

print("\n" + "=" * 60)
print("STORY:", story)
print("=" * 60)

story_ids = tokenizer.encode(story)

# 3. SEMANTIC FILTER
filtered_words = []
filtered_ids = []
for tid in story_ids:
    word = tokenizer.decode([tid]).strip()
    if word not in EXCLUDE and word != '':
        filtered_words.append(word)
        filtered_ids.append(tid)

print(f"\nFiltered chain ({len(filtered_ids)} tokens):")
print(f"  {' -> '.join(filtered_words)}")

# 4. DIRECTIONAL TIME BINDING
M = torch.zeros(HALF, dtype=torch.complex64, device=DEV)

print(f"\nDirected write: M += Phase_curr * Phase_prev.conj()")
for i in range(len(filtered_ids) - 1):
    p_prev = phase_vectors[filtered_ids[i]]
    p_curr = phase_vectors[filtered_ids[i + 1]]
    M += p_curr * p_prev.conj()

print(f"  {len(filtered_ids) - 1} directed edges. |M| mean: {M.abs().mean().item():.4f}")

# 5. 3-HOP BACKWARD TRACE
# Derivation: M = C * P*, so P = (M * C*)*
def backward_unbind(M, cur_phase):
    return (M * cur_phase.conj()).conj()

print("\n" + "=" * 60)
print("3-HOP BACKWARD TRACE: football")
print("=" * 60)

p_football = phase_vectors[tokenizer.encode("football", add_special_tokens=False)[0]]

# Hop 1: football -> dropped
wave1 = backward_unbind(M, p_football)
hop1_id = measure(wave1, "Hop 1 — football <- ? (expect dropped)")
hop1_word = tokenizer.decode([hop1_id]).strip()
print(f"  >> Step 1: {hop1_word} <- football")

# Hop 2: dropped -> Mary
p_hop1 = phase_vectors[hop1_id]
wave2 = backward_unbind(M, p_hop1)
hop2_id = measure(wave2, f"Hop 2 — {hop1_word} <- ? (expect Mary)")
hop2_word = tokenizer.decode([hop2_id]).strip()
print(f"  >> Step 2: {hop2_word} <- {hop1_word}")

# Hop 3: Mary -> [went, hallway] (superposition from both sentences)
p_hop2 = phase_vectors[hop2_id]
wave3 = backward_unbind(M, p_hop2)
_ = measure(wave3, f"Hop 3 — {hop2_word} <- ? (superposition: went, hallway)")

print("\n" + "=" * 60)
print(f"TRACE: {hop2_word} <- {hop1_word} <- football")
print("=" * 60)
print("DONE.")
print("=" * 60)
