"""
Phase 5: Semantic Routing — The Query Beam
============================================
When Mary has multiple outgoing edges, the query word "Where" acts
as a semantic beam, modulating the forward superposition toward
locomotion verbs via multiplicative AND-gating.

modulated_scores = base_scores * query_scores
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 5: SEMANTIC ROUTING — THE QUERY BEAM")
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

def measure(wave, label, power=3, topk=3, extra_scores=None):
    raw = torch.abs(phase_vectors @ wave.conj())
    if extra_scores is not None:
        scores = (raw * extra_scores) ** power
    else:
        scores = raw ** power
    top = scores.topk(topk)
    print(f"\n  {label}:")
    for rank, (tid, score) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
        word = tokenizer.decode([tid]).strip()
        raw_score = raw[tid].item()
        qs = extra_scores[tid].item() if extra_scores is not None else None
        if qs is not None:
            print(f"    Rank {rank+1}: '{word}'  (base: {raw_score:.0f}, query: {qs:.3f}, mod: {score:.1e})")
        else:
            print(f"    Rank {rank+1}: '{word}'  (raw: {raw_score:.1f}, sharp: {score:.1e})")
    return top.indices[0].item()

# 2. THE DATA
story = "Mary went to the bathroom. John moved to the hallway. Mary dropped the football."

print("\n" + "=" * 60)
print("STORY:", story)
print("=" * 60)

story_ids = tokenizer.encode(story)

# 3. FIREWALL FILTER
filtered_words = []
filtered_ids = []
for tid in story_ids:
    word = tokenizer.decode([tid]).strip()
    if word not in EXCLUDE and word != '':
        filtered_words.append(word)
        filtered_ids.append(tid)

print(f"\nFiltered chain ({len(filtered_ids)} tokens):")
print(f"  {' '.join(filtered_words)}")

# 4. DIRECTIONAL BINDING WITH FIREWALL
M = torch.zeros(HALF, dtype=torch.complex64, device=DEV)

print(f"\nDirected write with firewall...")
bound = 0
skipped = 0
for i in range(len(filtered_ids) - 1):
    word_prev = tokenizer.decode([filtered_ids[i]]).strip()
    word_curr = tokenizer.decode([filtered_ids[i + 1]]).strip()
    if word_prev == '.' or word_curr == '.':
        skipped += 1
        continue
    M += phase_vectors[filtered_ids[i + 1]] * phase_vectors[filtered_ids[i]].conj()
    bound += 1

print(f"  {bound} edges, {skipped} firewall-skipped. |M| mean: {M.abs().mean().item():.4f}")

# 5. THE QUERY BEAM — additive modulation avoids multiplicative self-resonance
beam_word = "located"
beam_ids = tokenizer.encode(beam_word, add_special_tokens=False)
p_beam = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
for bid in beam_ids:
    p_beam += phase_vectors[bid]
p_beam = p_beam / p_beam.abs().max()

query_scores_all = torch.abs(phase_vectors @ p_beam.conj())
mu = query_scores_all.mean()
# Clip self-resonance spike: any token with query > 3*mean gets clamped
# This removes "located" and its subword variants without killing location signal
spike_mask = query_scores_all > 3.0 * mu
n_spikes = spike_mask.sum().item()
query_scores_all[spike_mask] = mu
print(f"  Query beam: '{beam_word}' — clipped {n_spikes} self-resonant tokens to mean ({mu:.1f})")

def backward_unbind(M, cur_phase):
    return (M * cur_phase.conj()).conj()

def forward_wave(M, prev_phase):
    return M * prev_phase

p_football = phase_vectors[tokenizer.encode("football", add_special_tokens=False)[0]]

# ========== BACKWARD TRACE ==========
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

# ========== FORWARD TRACE WITH QUERY BEAM ==========
print("\n" + "=" * 60)
print(f"PIVOT — Forward from '{hop2_word}' with query beam '{beam_word}'")
print("=" * 60)

p_mary = phase_vectors[hop2_id]
wave3 = forward_wave(M, p_mary)
base_raw = torch.abs(phase_vectors @ wave3.conj())

# Show raw ranking first (proves 'dropped' was winning)
print("\n  Hop 3 FWD RAW (no query beam) — Mary -> ?:")
raw_top = (base_raw ** 3).topk(2)
for rank, (tid, score) in enumerate(zip(raw_top.indices.tolist(), raw_top.values.tolist())):
    word = tokenizer.decode([tid]).strip()
    print(f"    Rank {rank+1}: '{word}'  (raw: {base_raw[tid].item():.0f}, sharp: {score:.1e})")

# Now apply query beam to DESTINATIONS, not intermediate verbs.
# Strategy: forward-expand from Mary, score the TWO-STEP destinations.
# Mary -> went -> bathroom  vs  Mary -> dropped -> football
print(f"\n  Hop 3 FWD MODULATED — beam-search from Mary using '{beam_word}':")
# Get raw forward candidates from Mary
raw_from_mary = torch.abs(phase_vectors @ wave3.conj())
mary_top = (raw_from_mary ** 3).topk(5)

best_dest_score = -1
best_verb = None
best_dest = None
best_verb_id = None

for rank, (verb_id, _) in enumerate(zip(mary_top.indices.tolist(), mary_top.values.tolist())):
    verb_word = tokenizer.decode([verb_id]).strip()
    if verb_word == hop2_word:
        continue
    # Look ahead one step from this verb
    wave_dest = forward_wave(M, phase_vectors[verb_id])
    dest_raw = torch.abs(phase_vectors @ wave_dest.conj())
    # Score destinations by query resonance
    dest_scores = (dest_raw * query_scores_all) ** 3
    top_dest = dest_scores.topk(2)
    dest1_word = tokenizer.decode([top_dest.indices[0].item()]).strip()
    dest1_score = top_dest.values[0].item()
    dest2_word = tokenizer.decode([top_dest.indices[1].item()]).strip()
    dest2_score = top_dest.values[1].item()
    print(f"    {verb_word}: -> '{dest1_word}' ({dest1_score:.1e})  |  '{dest2_word}' ({dest2_score:.1e})")
    if dest1_score > best_dest_score:
        best_dest_score = dest1_score
        best_verb = verb_word
        best_dest = dest1_word
        best_verb_id = verb_id

hop3_word = best_verb
hop3_id = best_verb_id
print(f"  >> Selected: Mary -> {hop3_word} (best destination: {best_dest})")

# ========== HOP 4: FINAL DESTINATION ==========
print("\n" + "=" * 60)
print(f"PHASE 2: Final Hop — Where did Mary go?")
print("=" * 60)

p_hop3 = phase_vectors[hop3_id]
wave4 = forward_wave(M, p_hop3)
hop4_id = measure(wave4, f"Hop 4 FWD — {hop3_word} -> ? (expect bathroom)")
hop4_word = tokenizer.decode([hop4_id]).strip()

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"  V-TRACE: bathroom <- {hop3_word} <- {hop2_word} <- {hop1_word} <- football")
if hop4_word == "bathroom":
    print(f"  VERDICT: PASS. The football is in the {hop4_word}.")
else:
    print(f"  VERDICT: MISS. Expected 'bathroom', got '{hop4_word}'.")
print("=" * 60)
print("DONE.")
print("=" * 60)
