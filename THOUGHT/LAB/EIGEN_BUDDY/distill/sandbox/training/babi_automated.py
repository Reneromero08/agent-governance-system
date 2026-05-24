"""
Phase 2: Automated Ingestion & Temporal Binding
=================================================
Moves from hardcoded binding to automated n-gram ingestion.
Rolling knot: for every adjacent token pair, M += Phase_t ⊙ Phase_{t-1}.
Superposition filter: score sharpening via exponentiation (thermodynamic lens).
Multi-hop query: football -> [hop1] -> [hop2].
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 2: AUTOMATED INGESTION — ROLLING KNOT")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_phase(word):
    tid = tokenizer.encode(word, add_special_tokens=False)
    return phase_vectors[tid[0]] if tid else None

def measure(wave, label, power=3):
    raw = torch.abs(phase_vectors @ wave.conj())
    scores = raw ** power
    top3 = scores.topk(3)
    print(f"\n  {label}:")
    for rank, (tid, score) in enumerate(zip(top3.indices.tolist(), top3.values.tolist())):
        word = tokenizer.decode([tid]).strip()
        raw_score = raw[tid].item()
        print(f"    Rank {rank+1}: '{word}'  (raw: {raw_score:.1f}, sharp: {score:.1e})")
    return top3.indices[0].item()

# 2. THE DATA
story = "Mary went to the bathroom. John moved to the hallway. Mary dropped the football."
query = "Where is the football?"
target = "bathroom"

print("\n" + "=" * 60)
print("STORY:", story)
print("QUERY:", query)
print("=" * 60)

story_ids = tokenizer.encode(story)
query_ids = tokenizer.encode(query)

print(f"\nStory tokens: {len(story_ids)} | Query tokens: {len(query_ids)}")

# 3. AUTOMATED ROLLING KNOT (n-gram binding)
M = torch.zeros(HALF, dtype=torch.complex64, device=DEV)

print("\nBurning rolling knot into M...")
pairs = 0
for i in range(len(story_ids) - 1):
    t_prev = story_ids[i]
    t_curr = story_ids[i + 1]
    p_prev = phase_vectors[t_prev]
    p_curr = phase_vectors[t_curr]
    M += p_curr * p_prev
    pairs += 1

print(f"  {pairs} bigram pairs bound into M.  |M| mean: {M.abs().mean().item():.4f}")

# 4. MULTI-HOP QUERY with Superposition Filter
print("\n" + "=" * 60)
print("MULTI-HOP QUERY: Where is the football?")
print("=" * 60)

p_football = get_phase("football")

print("\n--- HOP 1: Unbind 'football' ---")
wave1 = M * p_football.conj()
hop1_token = measure(wave1, "football → ?")
hop1_word = tokenizer.decode([hop1_token]).strip()
print(f"  >> Selected: '{hop1_word}'")

print("\n--- HOP 2: Unbind '{hop1_word}' ---")
p_hop1 = phase_vectors[hop1_token]
wave2 = M * p_hop1.conj()
hop2_token = measure(wave2, f"{hop1_word} → ?")
hop2_word = tokenizer.decode([hop2_token]).strip()
print(f"  >> Selected: '{hop2_word}'")

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"  Trace: football -> {hop1_word} -> {hop2_word}")
if hop2_word == target:
    print(f"  VERDICT: PASS. Traced state correctly to '{target}'.")
else:
    print(f"  VERDICT: MISS. Expected '{target}', got '{hop2_word}'.")

print("=" * 60)
print("DONE.")
print("=" * 60)
