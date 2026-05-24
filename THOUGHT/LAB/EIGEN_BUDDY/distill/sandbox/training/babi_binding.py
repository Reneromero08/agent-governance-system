"""
Phase 1.5: HRR Complex Binding — Two-Hop State Tracking
=========================================================
Upgrades the hologram from Markov chain A->B to true Vector Symbolic
Architecture (VSA) using element-wise Hadamard product binding.

Binding:    M += Phase_A ⊙ Phase_B        (complex Hadamard, phase addition)
Unbinding:  Wave = M ⊙ Phase_A_conj        (phase subtraction)
Retrieval:  scores = |Phase_vocab · Wave|   (complex dot product magnitude)
"""
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 1.5: HRR COMPLEX BINDING — STATE TRACKING")
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
    tid = tokenizer.encode(word, add_special_tokens=False)[0]
    return phase_vectors[tid]

def measure(wave, label):
    scores = torch.abs(phase_vectors @ wave.conj())
    top3 = scores.topk(3)
    print(f"\n  {label}:")
    for rank, (tid, score) in enumerate(zip(top3.indices.tolist(), top3.values.tolist())):
        word = tokenizer.decode([tid]).strip()
        print(f"    Rank {rank+1}: '{word}' (score: {score:.2f})")

# 2. THE DATA
print("\n--- The Tape ---")
print("Bind 1: Mary + bathroom")
print("Bind 2: John + hallway")
print("Bind 3: Mary + football")

p_mary     = get_phase("Mary")
p_john     = get_phase("John")
p_bathroom = get_phase("bathroom")
p_hallway  = get_phase("hallway")
p_football = get_phase("football")

# 3. THE HOLOGRAPHIC WRITE — Hadamard product binding
M = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
M += p_mary * p_bathroom
M += p_john * p_hallway
M += p_mary * p_football
print(f"\nM built. |M| mean: {M.abs().mean().item():.4f}")

# 4. TWO-HOP QUERY
print("\n========================================")
print("QUERY: Where is the football?")
print("========================================")

# Hop 1: unbind football -> should get Mary
wave1 = M * p_football.conj()
measure(wave1, "Hop 1 — Unbind 'football' (expect Mary)")

# Hop 2: unbind Mary -> should get bathroom AND football
wave2 = M * p_mary.conj()
measure(wave2, "Hop 2 — Unbind 'Mary' (expect bathroom, football)")

# Verify: unbind John -> should get hallway
wave3 = M * p_john.conj()
measure(wave3, "Verify — Unbind 'John' (expect hallway)")

print("\n" + "=" * 60)
print("DONE.")
print("=" * 60)
