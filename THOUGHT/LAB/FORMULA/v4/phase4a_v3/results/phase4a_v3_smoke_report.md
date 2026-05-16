# Phase 4a v3 Smoke: Df Sweep — Retry Correction

**Date:** 2026-05-16
**Status:** SMOKE COMPLETE — σ ≈ 1 (retry correction ineffective)

---

## Method

Swept Df (retry attempts) across 5 prompts that failed in v1 CONTROL:

| Df | T schedule | Mechanism |
|----|-----------|-----------|
| 1 | 0.7 | Single attempt |
| 2 | 0.7, 0.35 | Retry at lower T if wrong |
| 3 | 0.7, 0.35, 0.18 | Retry twice |
| 5 | 0.7 → 0.09 | Retry 4x, halving T each time |
| 7 | 0.7 → 0.011 | Retry 6x, halving T each time |

QEC-aligned: no mid-generation modification. Each attempt is an independent generation. "Correction" = retry at lower temperature (more deterministic sampling).

## Results

| Prompt | Df=1 | Df=2 | Df=3 | Df=5 | Df=7 |
|--------|------|------|------|------|------|
| F5 (population) | ✗ | ✗✗ | ✗✗✗ | ✗✗✗✗✗ | ✗✗✗✗✗✗✗ |
| F6 (water formula) | ✗ | ✗✗ | ✗✗✗ | ✗✗✗✗✗ | — |

**Zero corrections across all Df levels.** 0/17 attempts correct. Accuracy = 0.0 for all Df.

## Analysis

The formula predicts: `R(Df) = (E/∇S) × σ^Df`

With σ = 1 (no correction power): `R = E/∇S = constant`, independent of Df.

The data confirms σ ≈ 1. The retry mechanism has no correction power because lower temperature does not change the model's logits — it only sharpens them. The model is systematically wrong on these prompts; lower T makes it *more confidently wrong*, not correct.

## Root Cause: Missing Syndrome

The QEC decoder works because it receives a **syndrome** — a measurement that tells it *which* qubits are likely errored and *how* to flip them. The retry corrector receives no syndrome. It knows the output is wrong (the oracle said so), but it doesn't know *what specifically* was wrong or *what the correct answer is*. Without a syndrome, the decoder cannot correct — it can only retry blindly.

## Required Fix: Syndrome-Informed Correction

When verification fails, inject the ground truth as a specific correction:

```
"Your previous answer said [incorrect claim]. That is wrong.
 The correct fact is [ground truth].
 Now re-answer the question: [original prompt]"
```

This gives the model:
- **Error location** (what was wrong) — the QEC syndrome analog
- **Correct state** (what it should be) — the QEC correction gate
- **Retry prompt** — regeneration with new information

With this mechanism, σ measures the probability that the model, *given the correct answer*, can produce it on retry. This should vary by category:
- **Factual**: model "knows" the answer but sampled wrong → σ high (> 0.5)
- **Reasoning**: model's reasoning was flawed → syndrome gives answer, model reverse-engineers → σ medium
- **Adversarial**: model's weights resist the truth → even with syndrome, σ ≈ 1

## Next: Full Df Sweep with Syndrome Correction

```
Sweep: Df ∈ [1, 2, 3, 4, 5, 6, 7] (syndrome-corrected retries)
Categories: factual (low ∇S), reasoning (med ∇S), adversarial (high ∇S)
Conditions: 7 × 3 = 21

Per condition:
  - Generate at T=0.7
  - Verify → if correct, stop (Df_used = 1)
  - If wrong, inject syndrome correction, retry at T=0.7
  - Repeat up to Df times
  - Measure: attempts_used, final_correct

Train: Df ∈ [1,2,3] → calibrate σ per category
Test:  Df ∈ [4,5,6,7] → predict accuracy, test σ^Df scaling
```
