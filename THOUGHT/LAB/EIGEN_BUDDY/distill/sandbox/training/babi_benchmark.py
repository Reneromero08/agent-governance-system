"""
Phase 6: The Benchmark — bAbI Task 1 Engine
=============================================
Packages the V-Trace mechanics into a reusable NativeHologram class.
Vocabulary mask eliminates Unicode noise. Runs 3 bAbI test cases.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 6: BENCHMARK — bAbI Task 1")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {',', 'the', 'to', 'Where', 'is', '?', '!', '-'}

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

# 1. VOCABULARY MASK
vocab_mask = torch.zeros(V, device=DEV)
ascii_letter = re.compile(r'^[a-zA-Z.]+$')
n_masked = 0
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_letter.match(word) and word != '':
        vocab_mask[tid] = 1.0
    else:
        n_masked += 1
print(f"Prism: {HALF} dim, {V} tokens. Vocab mask: {int(vocab_mask.sum().item())} allowed, {n_masked} blocked.")


class NativeHologram:
    def __init__(self, phase_vectors, vocab_mask):
        self.pv = phase_vectors
        self.mask = vocab_mask
        self.HALF = phase_vectors.shape[1]
        self.M = torch.zeros(self.HALF, dtype=torch.complex64, device=phase_vectors.device)

    def _filter_ids(self, text):
        raw_ids = tokenizer.encode(text)
        filtered_ids = []
        for tid in raw_ids:
            word = tokenizer.decode([tid]).strip()
            if word not in EXCLUDE and word != '' and vocab_mask[tid] > 0:
                filtered_ids.append(tid)
        return filtered_ids

    def ingest(self, text):
        ids = self._filter_ids(text)
        bound = 0
        skipped = 0
        for i in range(len(ids) - 1):
            word_prev = tokenizer.decode([ids[i]]).strip()
            word_curr = tokenizer.decode([ids[i + 1]]).strip()
            if word_prev == '.' or word_curr == '.':
                skipped += 1
                continue
            p_prev = self.pv[ids[i]]
            p_curr = self.pv[ids[i + 1]]
            self.M += p_curr * p_prev.conj()
            bound += 1
        return bound, skipped

    def _backward(self, phase):
        return (self.M * phase.conj()).conj()

    def _forward(self, phase):
        return self.M * phase

    def _measure(self, wave, power=3, topk=3):
        raw = torch.abs(self.pv @ wave.conj())
        scores = (raw * self.mask) ** power
        top = scores.topk(topk)
        return top.indices[0].item(), top

    def query(self, entity_str, beam_str="located"):
        entity_ids = tokenizer.encode(entity_str, add_special_tokens=False)
        beam_ids = tokenizer.encode(beam_str, add_special_tokens=False)
        if not entity_ids or not beam_ids:
            return None, []

        p_entity = torch.zeros(self.HALF, dtype=torch.complex64, device=self.pv.device)
        for eid in entity_ids:
            p_entity += self.pv[eid]
        p_entity = p_entity / max(p_entity.abs().max().item(), 1e-8)

        p_beam = torch.zeros(self.HALF, dtype=torch.complex64, device=self.pv.device)
        for bid in beam_ids:
            p_beam += self.pv[bid]
        p_beam = p_beam / max(p_beam.abs().max().item(), 1e-8)

        beam_resonance = torch.abs(self.pv @ p_beam.conj())
        mu = beam_resonance.mean()
        spike_mask = beam_resonance > 3.0 * mu
        beam_resonance[spike_mask] = mu

        trace = []

        # Hop 1 BWD: entity -> action
        wave1 = self._backward(p_entity)
        act_id, _ = self._measure(wave1)
        act_word = tokenizer.decode([act_id]).strip()
        trace.append(act_word)

        # Hop 2 BWD: action -> actor
        wave2 = self._backward(self.pv[act_id])
        actor_id, _ = self._measure(wave2)
        actor_word = tokenizer.decode([actor_id]).strip()
        trace.append(actor_word)

        # Hop 3 FWD: actor -> [verb] (beam-searched)
        wave3 = self._forward(self.pv[actor_id])
        raw3 = torch.abs(self.pv @ wave3.conj())
        actor_top = (raw3 * self.mask).topk(5)

        best_score = -1
        best_verb_id = None
        for verb_id in actor_top.indices.tolist():
            verb_word = tokenizer.decode([verb_id]).strip()
            if verb_word == actor_word or verb_word == act_word:
                continue
            wave_dest = self._forward(self.pv[verb_id])
            dest_raw = torch.abs(self.pv @ wave_dest.conj())
            dest_scores = (dest_raw * beam_resonance * self.mask) ** 3
            top_dest = dest_scores.max().item()
            if top_dest > best_score:
                best_score = top_dest
                best_verb_id = verb_id

        if best_verb_id is None:
            best_verb_id = actor_top.indices[0].item()
        verb_word = tokenizer.decode([best_verb_id]).strip()
        trace.append(verb_word)

        # Hop 4 FWD: verb -> destination
        wave4 = self._forward(self.pv[best_verb_id])
        dest_id, _ = self._measure(wave4)
        dest_word = tokenizer.decode([dest_id]).strip()
        trace.append(dest_word)

        return dest_word, trace


# 2. BENCHMARK
tests = [
    {
        "story": "Mary went to the bathroom . John moved to the hallway . Mary dropped the football .",
        "query": "football",
        "expected": "bathroom",
    },
    {
        "story": "Daniel travelled to the office . Sandra journeyed to the kitchen . Daniel grabbed the apple .",
        "query": "apple",
        "expected": "office",
    },
    {
        "story": "John went to the bedroom . Mary grabbed the milk . Mary travelled to the garden .",
        "query": "milk",
        "expected": "garden",
    },
]

passed = 0
for i, test in enumerate(tests):
    print(f"\n{'='*60}")
    print(f"TEST {i+1}: {test['query']} -> ? (expect {test['expected']})")
    print(f"  Story: {test['story']}")

    holo = NativeHologram(phase_vectors, vocab_mask)
    bound, skipped = holo.ingest(test["story"])
    print(f"  Ingest: {bound} edges, {skipped} firewall-skipped")

    answer, trace = holo.query(test["query"], beam_str="located")
    trace_str = " <- ".join(reversed(trace))
    print(f"  Trace: {trace_str}")
    print(f"  Answer: {answer}")

    if answer == test["expected"]:
        print(f"  VERDICT: PASS")
        passed += 1
    else:
        print(f"  VERDICT: FAIL (expected '{test['expected']}', got '{answer}')")

print(f"\n{'='*60}")
print(f"BENCHMARK: {passed}/{len(tests)} passed")
print(f"{'='*60}")
print("DONE.")
print("=" * 60)
