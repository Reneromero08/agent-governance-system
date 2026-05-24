"""
Phase 2.5: Semantic Filter & 3-Hop Backward Trace
===================================================
Filters punctuation/stopwords from the ingestion stream, binds only
content tokens via rolling knot, then traces the causal chain backwards.

Trace: football -> dropped -> Mary -> [went, dropped, hallway]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 2.5: SEMANTIC FILTER — 3-HOP BACKWARD TRACE")
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
query = "football"

print("\n" + "=" * 60)
print("STORY:", story)
print("=" * 60)

story_ids = tokenizer.encode(story)
print(f"\nRaw tokens: {len(story_ids)}")

# 3. SEMANTIC FILTER
filtered_words = []
filtered_ids = []
for tid in story_ids:
    word = tokenizer.decode([tid]).strip()
    if word not in EXCLUDE and word != '':
        filtered_words.append(word)
        filtered_ids.append(tid)

print(f"Filtered ({len(EXCLUDE)} excluded): {len(filtered_ids)} content tokens")
print(f"  Chain: {' -> '.join(filtered_words)}")

# 4. ROLLING KNOT on filtered chain
M = torch.zeros(HALF, dtype=torch.complex64, device=DEV)

print(f"\nBinding {len(filtered_ids) - 1} bigram pairs into M...")
for i in range(len(filtered_ids) - 1):
    p_prev = phase_vectors[filtered_ids[i]]
    p_curr = phase_vectors[filtered_ids[i + 1]]
    M += p_curr * p_prev

print(f"  |M| mean: {M.abs().mean().item():.4f}")

# 5. 3-HOP BACKWARD TRACE
print("\n" + "=" * 60)
print(f"3-HOP BACKWARD TRACE: {query}")
print("=" * 60)

p_football = phase_vectors[tokenizer.encode("football", add_special_tokens=False)[0]]

# Hop 1: football -> dropped
wave1 = M * p_football.conj()
hop1_id = measure(wave1, "Hop 1 — Unbind 'football' (expect dropped)", power=3)
hop1_word = tokenizer.decode([hop1_id]).strip()
print(f"  >> Step 1: football -> {hop1_word}")

# Hop 2: dropped -> Mary
p_hop1 = phase_vectors[hop1_id]
wave2 = M * p_hop1.conj()
hop2_id = measure(wave2, f"Hop 2 — Unbind '{hop1_word}' (expect Mary)", power=3)
hop2_word = tokenizer.decode([hop2_id]).strip()
print(f"  >> Step 2: {hop1_word} -> {hop2_word}")

# Hop 3: Mary -> superposition (went, dropped, hallway)
p_hop2 = phase_vectors[hop2_id]
wave3 = M * p_hop2.conj()
_ = measure(wave3, f"Hop 3 — Unbind '{hop2_word}' (superposition: went, dropped, hallway)", power=3)

print("\n" + "=" * 60)
print(f"TRACE: football -> {hop1_word} -> {hop2_word}")
print("=" * 60)
print("DONE.")
print("=" * 60)
