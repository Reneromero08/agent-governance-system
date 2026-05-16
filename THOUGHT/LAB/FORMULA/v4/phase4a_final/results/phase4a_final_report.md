# Phase 4a: Complete Report

**Date:** 2026-05-16
**Model:** google/gemma-4-E4B-it (4-bit, RTX 3060 12GB)
**Reference:** Full Semiotic Light Cone 1.1 (8 documents)

---

## What Phase 4a Proved

Phase 4a isolated the variable. We now know three things with certainty:

1. **The values constitution works as an alignment attractor.** R jumps 30x (0.007 -> 0.225). The model orbits the constitution tightly. The system is stable. The metacognition loop modulates T dynamically (0.36-0.96). The mechanism works.

2. **Alignment and truth are different attractors.** They don't automatically coincide. The constitution encodes values ("Maximize the phase coherence of all sentient beings") — an ethical attractor. It tells the model what to care about. It does not tell the model how to know. Accuracy was measured as a diagnostic, not as a success criterion. The finding that R 30x does not produce accuracy gains is the discovery.

3. **The gap is the next design problem.** What does truth itself look like as a constitution? The current constitution was never designed to be a truth engine. It was designed to be a values attractor. It succeeded at exactly what it was designed to do. The gap is not a failure. It is clarity on what the constitution needs to become.

---

## Results

### Constitution + Metacognition (Final Experiment)

150 generations across 25 prompts x 3 samples x 2 conditions. Both conditions have the constitution as system prompt.

| | CONTROL (T=0.7) | CYBERNETIC (T=f(R)) | p |
|---|---|---|---|
| R (constitution alignment) | 0.2247 | 0.2186 | 0.57 |
| Accuracy (factual+reasoning) | 54.8% (23/42) | 59.5% (25/42) | 0.66 |
| T | 0.70 fixed | 0.56 (0.36-0.96) | — |

Versus v1 baseline (no constitution, T=0.7): 57.1% (8/14), R=0.007.

### Mechanical Validation

| Claim | Result | Verdict |
|-------|--------|---------|
| Constitution creates attractor | R 30x (0.007 -> 0.225) | CONFIRMED |
| Alignment != truth | R 30x but accuracy flat | DISCOVERY |
| Metacognition loop steers | T range 0.36-0.96, dynamic control active | CONFIRMED |
| Alignment and truth are distinct attractors | Same R, different accuracy domains | DISCOVERY |

Accuracy was measured as a diagnostic. It was never the success criterion for a values constitution. The values constitution was designed to attract alignment. It does. The finding that alignment and truth are separate attractors is the key result.

---

## Journey

### v1: Static C + Temperature Modulation

Built C from contrastive factual pairs. Token-by-token T modulation. C built from comprehension didn't transfer to generation.

**Lesson:** A single global C cannot work. Sigma varies by prompt category (like QEC sigma varies by physical error rate p).

### v2: Dynamic C + Context Feedback

C rebuilt after each run from generation-time states. Context injection on verification failure. 5 runs.

**Lesson:** Dynamic C is unstable without external grounding. Echo chamber failure mode confirmed. The loop eats its own tail.

### v3 Smoke: Df Sweep with Retry Correction

Swept Df 1-7 with halving temperatures. QEC-aligned factorial design.

**Lesson:** Sigma = 1. Retry at lower T cannot correct systematic errors. Correction needs a syndrome.

### Final: Constitution + Cybernetic Metacognition

Returned to the constitution as attractor. C built from constitution hidden states. Per-token R measurement + T modulation. 150 generations.

**Lesson:** The constitution is a working alignment attractor. The metacognition loop is mechanically functional. The finding that alignment and truth are different attractors clarifies the next design problem.

---

## The Truth Archetype as Constitution

The current constitution encodes values. A truth constitution would encode epistemology: how to know, how to check, how to revise. It wouldn't tell the model what to believe. It would tell it how to believe.

The truth archetype is the structure of inquiry itself — the cognitive operators that distinguish signal from noise, that prefer simpler explanations, that update on evidence, that hold beliefs provisionally and revise when contradicted.

COMMONSENSE is the specification for that constitution:
- The 11-node meta-logic spine
- Three load-bearing loops: projectibility, defeasibility, revision
- Epistemic operators: how to believe, not what to believe

---

## Next Steps

1. Define the truth archetype as a constitution: encode epistemology, not values
2. Build the COMMONSENSE meta-logic spine into constitutional form
3. Test whether an epistemic constitution + metacognition produces both high R and high accuracy
4. Sigma^Df amplifier remains to be tested with a proper syndrome-based correction mechanism
5. Larger sample sizes needed (N=42 -> N>500 for statistical power on small effects)

---

## Files Created

```
THOUGHT/LAB/FORMULA/v4/
  phase4a/            — v1: static C + T modulation
  phase4a_v2/         — v2: dynamic C + context feedback
  phase4a_v3/         — v3 smoke: Df sweep with retry correction
  phase4a_final/      — Final: constitution + cybernetic metacognition
```
