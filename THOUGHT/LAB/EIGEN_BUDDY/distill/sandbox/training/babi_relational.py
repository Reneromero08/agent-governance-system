"""
Phase 4: The Firewall & The V-Shaped Trace
============================================
Periods act as graph firewalls — no binding across sentence boundaries.
V-shaped trace: backward to find actor, pivot, forward to find location.

Backward: Wave = (M * Phase_curr.conj()).conj()
Forward:  Wave = M * Phase_prev
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 4: FIREWALL & V-SHAPED TRACE")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {',', 'the', 'to', 'Where', 'is', '?'}

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

# 3. FIREWALL FILTER — keep periods for boundary detection
filtered_words = []
filtered_ids = []
period_id = None
for tid in story_ids:
    word = tokenizer.decode([tid]).strip()
    if word == '.':
        period_id = tid
    if word not in EXCLUDE and word != '':
        filtered_words.append(word)
        filtered_ids.append(tid)

print(f"\nFiltered chain ({len(filtered_ids)} tokens):")
print(f"  {' '.join(filtered_words)}")

# 4. DIRECTIONAL BINDING WITH FIREWALL
M = torch.zeros(HALF, dtype=torch.complex64, device=DEV)

print(f"\nDirected write with firewall (skip periods)...")
bound = 0
skipped = 0
for i in range(len(filtered_ids) - 1):
    word_prev = tokenizer.decode([filtered_ids[i]]).strip()
    word_curr = tokenizer.decode([filtered_ids[i + 1]]).strip()
    if word_prev == '.' or word_curr == '.':
        skipped += 1
        print(f"  FIREWALL: skip '{word_prev}' -> '{word_curr}'")
        continue
    p_prev = phase_vectors[filtered_ids[i]]
    p_curr = phase_vectors[filtered_ids[i + 1]]
    M += p_curr * p_prev.conj()
    bound += 1

print(f"  {bound} edges bound, {skipped} firewall-skipped.")
print(f"  |M| mean: {M.abs().mean().item():.4f}")

# 5. V-SHAPED TRACE
def backward_unbind(M, cur_phase):
    return (M * cur_phase.conj()).conj()

def forward_unbind(M, prev_phase):
    return M * prev_phase

p_football = phase_vectors[tokenizer.encode("football", add_special_tokens=False)[0]]

print("\n" + "=" * 60)
print("PHASE 1: BACKWARD TRACE — Who has the football?")
print("=" * 60)

wave1 = backward_unbind(M, p_football)
hop1_id = measure(wave1, "Hop 1 BWD — football <- ? (expect dropped)")
hop1_word = tokenizer.decode([hop1_id]).strip()
print(f"  >> {hop1_word} <- football")

p_hop1 = phase_vectors[hop1_id]
wave2 = backward_unbind(M, p_hop1)
hop2_id = measure(wave2, f"Hop 2 BWD — {hop1_word} <- ? (expect Mary)")
hop2_word = tokenizer.decode([hop2_id]).strip()
print(f"  >> {hop2_word} <- {hop1_word}")

print(f"\n  ACTOR FOUND: {hop2_word}")

print("\n" + "=" * 60)
print(f"PIVOT — Now trace forward from '{hop2_word}'")
print("=" * 60)

print("\nPHASE 2: FORWARD TRACE — Where is Mary?")

p_mary = phase_vectors[hop2_id]
wave3 = forward_unbind(M, p_mary)
hop3_ids = torch.abs(phase_vectors @ wave3.conj()) ** 3
top_forward3 = hop3_ids.topk(2)

print(f"\n  Hop 3 FWD — Mary -> ? (superposition: went, dropped):")
for rank, (tid, score) in enumerate(zip(top_forward3.indices.tolist(), top_forward3.values.tolist())):
    word = tokenizer.decode([tid]).strip()
    raw = torch.abs(phase_vectors @ wave3.conj())[tid].item()
    print(f"    Rank {rank+1}: '{word}'  (raw: {raw:.1f}, sharp: {score:.1e})")

went_id = top_forward3.indices[0].item()
went_word = tokenizer.decode([went_id]).strip()

print(f"\n  Hop 4 FWD — {went_word} -> ? (expect bathroom):")
p_went = phase_vectors[went_id]
wave4 = forward_unbind(M, p_went)
_ = measure(wave4, f"Hop 4 FWD — {went_word} -> ? (expect bathroom)")

print("\n" + "=" * 60)
print("V-TRACE COMPLETE")
print("=" * 60)
print("DONE.")
print("=" * 60)
