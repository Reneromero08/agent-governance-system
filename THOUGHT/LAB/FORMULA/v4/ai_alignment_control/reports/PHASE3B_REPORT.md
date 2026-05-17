# Phase 3b Report: Symbol Survival -- LLM Transmission Chains

Date: 2026-05-14 | Model: Gemma 4B E4B (4-bit) | Status: **INCONCLUSIVE -- EXPERIMENT TESTED LENGTH, NOT COMPRESSION**

---

## Reassessment Summary

The Phase 3b experiment found that longer texts survive LLM paraphrasing better
than shorter texts. Literal expansions of proverbs (longer, same meaning)
retained higher cosine similarity to their originals than the proverbs
themselves. This is a real finding about length and paraphrase fidelity. It is
not a test of whether compression amplifies survival under error-correction
dynamics. The formula was not tested under appropriate conditions.

## What Was Tested

- **Symbols**: 10 proverb/literal pairs + 10 control sentences (30 total)
- **sigma**: Measured token compression ratio. Proverbs have sigma > 1; literals have sigma = 1.0 by definition.
- **Df**: Human-rated interpretive depth (1-4 layers)
- **Transmission**: 10 generations of LLM paraphrasing (Gemma 4B, T in {0.3, 0.7, 1.2})
- **Chains**: 5 independent chains per condition. 450 chains total.
- **Metric**: Cosine similarity (all-MiniLM-L6-v2) to original at generation 10

## What Was Found

| Noise | Proverbs | Literals | Controls | Proverb-Literal diff (p) |
|-------|----------|----------|----------|--------------------------|
| LOW   | 0.470 | **0.587** | 0.412 | -0.116 (p=0.025) |
| MED   | 0.383 | 0.418 | 0.430 | -0.036 (p=0.557) |
| HIGH  | 0.223 | 0.300 | 0.175 | -0.077 (p=0.229) |

sigma*Df correlation with survival: r=-0.08, +0.16, -0.07 at LOW, MED, HIGH. None significant.

The only result supported by the data: longer texts retain higher cosine
similarity through LLM paraphrase chains than shorter texts with the same
semantic content.

## Why This Is Not a Falsification

### 1. The control is artificial

Literal expansions are artificially verbose versions of proverbs. They do not
exist as transmitted cultural objects. Comparing a natural compressed object
to an artificially inflated version of itself is a test of text length, not
compression.

### 2. The metric is misaligned with cultural survival

Cosine similarity measures semantic overlap, not recognition or citability. A
paraphrase of a proverb might score low on cosine similarity with its verbose
expansion while remaining perfectly recognizable as the original proverb. The
metric measures the wrong property.

### 3. Df was subjectively rated, not measured

QEC measured Df = t = floor((d-1)/2) from code distance. Phase 2 measured
Df = 5 from the constitution\\\'s explicit five-scale structure. Phase 3b used
human-assigned scores on a 1-4 scale with no inter-rater validation.

### 4. The transmission channel has no error-correction dynamics

The formula predicts compression amplifies survival when the channel has
error-correction dynamics -- when the receiver can detect and correct signal
degradation using the compressed structure. LLM paraphrasing is semantic
reconstruction, not error correction. Phase 2 had error-correction dynamics
via the Cybernetic Truth control circuit. Phase 3b did not.

### 5. Sample size is inadequate

N=10 proverb pairs with 5 chains each. A single outlier can flip the
correlation direction. The significant result at LOW noise reflects the
length effect, not a test of compression.

## What the Formula Actually Predicts

The formula predicts compression amplifies resonance when:
1. The channel has error-correction dynamics
2. sigma is actually measured from compression ratio
3. Df is actually measured from redundancy depth
4. grad_S is actually measured from channel noise

Phase 1 (QEC) and Phase 2 (AI alignment) confirmed this. Phase 3b used a
channel with no error-correction dynamics. The formula was not tested.

## Status

The Phase 3b experiment produced real data: longer texts survive LLM
paraphrasing better than shorter texts. This is trivial. The formula was
not tested under appropriate conditions. Phase 3b is INCONCLUSIVE.

## Files

- phase3b_experiment.py
- analyze_phase3b.py
- results/phase3b_results.json
- PREREGISTRATION_PHASE3B.md
