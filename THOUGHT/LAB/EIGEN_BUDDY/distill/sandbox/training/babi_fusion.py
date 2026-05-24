"""
Phase 7: Concept Fusion — Subword Resolution (Full Concept Pipeline)
======================================================================
Precomputes Hadamard-product concept phases for EVERY vocab word.
Both ingestion AND query use concept phases. Retrieval matches waves
against concept vocabulary, not raw tokens.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("PHASE 7: CONCEPT FUSION — FULL CONCEPT PIPELINE")
print("=" * 60)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDE = {',', 'the', 'to', 'Where', 'is', '?', '!', '-', ''}

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

ascii_letter = re.compile(r'^[a-zA-Z.]+$')
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

print(f"Prism: {HALF} dim, {V} tokens, {int(vocab_mask.sum().item())} concepts.")


class NativeHologram:
    def __init__(self, concept_phases, phase_vectors, vocab_mask):
        self.cp = concept_phases
        self.pv = phase_vectors
        self.mask = vocab_mask
        self.HALF = concept_phases.shape[1]
        self.M = torch.zeros(self.HALF, dtype=torch.complex64, device=concept_phases.device)

    def _get_concept_id(self, word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        for tid in range(len(concept_words)):
            if concept_words[tid] == word and self.mask[tid] > 0:
                return tid
        return ids[0] if ids[0] < len(concept_words) and self.mask[ids[0]] > 0 else None

    def ingest(self, text):
        words = text.split()
        concept_ids = []
        for w in words:
            clean = w.strip('.,!?;:')
            if clean in EXCLUDE:
                if w == '.':
                    concept_ids.append(-1)
                continue
            cid = self._get_concept_id(clean)
            if cid is not None:
                concept_ids.append(cid)
            if w.endswith('.'):
                concept_ids.append(-1)

        bound, skipped = 0, 0
        for i in range(len(concept_ids) - 1):
            prev_id = concept_ids[i]
            curr_id = concept_ids[i + 1]
            if prev_id < 0 or curr_id < 0:
                skipped += 1
                continue
            self.M += self.cp[curr_id] * self.cp[prev_id].conj()
            bound += 1
        return bound, skipped

    def _backward(self, concept_phase):
        return (self.M * concept_phase.conj()).conj()

    def _forward(self, concept_phase):
        return self.M * concept_phase

    def _measure_concept(self, wave, power=3, topk=3):
        raw = torch.abs(self.cp @ wave.conj())
        scores = (raw * self.mask) ** power
        top = scores.topk(topk)
        return top.indices[0].item(), top

    def query(self, entity_str, beam_str="located"):
        eid = self._get_concept_id(entity_str)
        bid = self._get_concept_id(beam_str)
        if eid is None or bid is None:
            return None, []

        p_entity = self.cp[eid]
        p_beam = self.cp[bid]

        beam_resonance = torch.abs(self.cp @ p_beam.conj())
        mu = beam_resonance.mean()
        beam_resonance[beam_resonance > 3.0 * mu] = mu

        trace = []

        wave1 = self._backward(p_entity)
        act_id, _ = self._measure_concept(wave1)
        act_word = concept_words[act_id]
        trace.append(act_word)

        wave2 = self._backward(self.cp[act_id])
        actor_id, _ = self._measure_concept(wave2)
        actor_word = concept_words[actor_id]
        trace.append(actor_word)

        wave3 = self._forward(self.cp[actor_id])
        raw3 = torch.abs(self.cp @ wave3.conj())
        actor_top = (raw3 * self.mask).topk(5)

        er = torch.abs(self.cp @ p_entity.conj())
        er_mu = er.mean()
        penalty = torch.where(er > 2.0 * er_mu, 0.1, 1.0)

        best_score = -1.0
        best_verb_id = None
        for verb_id in actor_top.indices.tolist():
            verb_word = concept_words[verb_id]
            if verb_word == actor_word or verb_word == act_word:
                continue
            wave_dest = self._forward(self.cp[verb_id])
            dest_raw = torch.abs(self.cp @ wave_dest.conj())
            dest_scores = (dest_raw * beam_resonance * penalty * self.mask) ** 3
            top_dest = dest_scores.max().item()
            if top_dest > best_score:
                best_score = top_dest
                best_verb_id = verb_id

        if best_verb_id is None:
            best_verb_id = actor_top.indices[0].item()
        verb_word = concept_words[best_verb_id]
        trace.append(verb_word)

        wave4 = self._forward(self.cp[best_verb_id])
        dest_id, _ = self._measure_concept(wave4)
        dest_word = concept_words[dest_id]
        trace.append(dest_word)

        return dest_word, trace


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

    holo = NativeHologram(concept_phases, phase_vectors, vocab_mask)
    bound, skipped = holo.ingest(test["story"])
    print(f"  Ingest: {bound} edges, {skipped} firewall-skipped")

    answer, trace = holo.query(test["query"], beam_str="where")
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
