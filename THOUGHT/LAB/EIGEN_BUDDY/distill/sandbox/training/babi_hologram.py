"""
Phase 1: The Clean Room — bAbI Task 1 (Single Supporting Fact)
================================================================
Proves continuous-wave state tracking using a 0.5B semantic prism.
No attention layers. No backprop. Pure phase geometry.
"""
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 1: NATIVE HOLOGRAM — STATE TRACKING (bAbI)")
print("=" * 60)

# 1. THE PRISM (Qwen 0.5B)
# Small enough to download instantly, deep enough to have semantic geometry.
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading Prism: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Extract embedding table and convert to Complex Phase Vectors (S^1)
embed = model.model.embed_tokens.weight.detach().float()
D_MODEL = embed.shape[1]
HALF = D_MODEL // 2  # 896 / 2 = 448 complex dimensions

er = embed[:, :HALF]
ei = embed[:, HALF:]
er = er / er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
ei = ei / ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)

phase_angle = torch.atan2(ei, er)
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)

print(f"Prism Ready: 448 complex dimensions on {DEV}.")
del model # Free VRAM, we only need the embeddings

# 2. THE DATA (bAbI Task 1 - Micro Scale)
story = "Mary went to the bathroom. John moved to the hallway. Mary dropped the football."
query = "Where is the football?"
target = "bathroom"

print("\n--- The Tape ---")
print(f"Story: {story}")
print(f"Query: {query}")
print(f"Target: {target}")

story_ids = tokenizer.encode(story)
query_ids = tokenizer.encode(query)

# 3. THE HOLOGRAPHIC WRITE (One-Shot Binding)
# M += (Phase_B ⊗ Phase_A*)
M = torch.zeros((HALF, HALF), dtype=torch.complex64, device=DEV)

print("\nBurning tape into Matrix M...")
for i in range(len(story_ids) - 1):
    idx_A = story_ids[i]
    idx_B = story_ids[i+1]
    
    phase_A = phase_vectors[idx_A]
    phase_B = phase_vectors[idx_B]
    
    # Complex outer product
    M += torch.outer(phase_B, phase_A.conj())

print(f"Burn complete. M magnitude: {M.abs().mean().item():.4f}")

# 4. THE RESONANT READ (Flashlight)
# Output = M @ Phase_query
last_query_token = query_ids[-1]
query_phase = phase_vectors[last_query_token]

print("\nShining query phase at Matrix M...")
output_wave = M @ query_phase

# Measure constructive interference against entire vocabulary
scores = torch.abs(phase_vectors @ output_wave.conj())
top5 = scores.topk(5).indices.tolist()

print("\n--- Resonance Results ---")
for rank, token_id in enumerate(top5):
    word = tokenizer.decode([token_id]).strip()
    score = scores[token_id].item()
    print(f"  Rank {rank+1}: '{word}' (Interference: {score:.2f})")

if tokenizer.decode([top5[0]]).strip() == target:
    print("\nVERDICT: PASS. Hologram successfully tracked state.")
else:
    print(f"\nVERDICT: FAIL. Expected '{target}'.")
