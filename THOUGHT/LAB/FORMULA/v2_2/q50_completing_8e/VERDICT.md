# Q50 Verification Report: Df * alpha Conserved Across Architectures

**Date:** 2026-05-17
**Status:** PARTIALLY VERIFIED (8e value FALSIFIED, architecture invariance CONFIRMED)

## Updated Verdict (2026-05-18)

The architecture invariance at fixed N is genuine — confirmed across MiniLM, MPNet, MultiQA, Paraphrase (CV 0.5-5.4%), and independently reproduced in the Q49 probe analysis. The shared f(N) function is the real invariant, not the value.

However, the v1 claim was specifically "Df × alpha = 8e is a universal constant." This is false — the product varies with N. The value 8e = 21.746 is merely f(75), an artifact of measuring at the single vocabulary size N=75.

**What survived:** The function f(N) is shared across architectures. Df × alpha at fixed N is invariant. α → 0.5 is the underlying universal exponent.

**What didn't:** 8e = 21.746 is not a universal constant. It's N-dependent. The v1 naming conflated the structural constant (α ≈ 0.5 from Chern class topology) with a scale-dependent measurement at N=75.
**Reviewer:** Fresh verification

---

## Claim

D_f × alpha is a universal conservation law holding across all trained embedding architectures, with the specific value 8e = 21.746.

---

## Test

Four sentence-transformer models (MiniLM-L6, MPNet-base, MultiQA-MiniLM, Paraphrase-MiniLM) plus BERT token embeddings were tested at vocabulary sizes N = 30, 50, 75, 100, 130.

| N | Mean product | CV across models | Span |
|---|-------------|------------------|------|
| 30 | 8.57 | 5.4% | 1.25 |
| 50 | 14.89 | 2.5% | 0.93 |
| 75 | 21.40 | 2.0% | 1.10 |
| 100 | 28.01 | 1.2% | 0.76 |
| 130 | 34.70 | 0.5% | 0.47 |

At N=75: mean = 21.55, CV = 2.2% across 5 models including BERT token embeddings. The v1 claim of cross-architecture agreement at N≈75 is **reproduced**.

---

## Findings

1. **Architecture invariance is real.** Different models produce nearly identical D_f × alpha values at the same N. The CV drops as N increases (5.4% → 0.5%), suggesting convergence.

2. **The product is N-dependent.** As established in Q49, D_f × alpha = f(N), approximately 0.27 × N in the under-sampled regime. It is not a constant.

3. **8e is not the value.** The product equals 8e = 21.746 only at N ≈ 75. At N = 130 it's 34.7. The v1 "constant" was measuring f(75), not a universal value.

4. **The function f(N) appears universal.** All tested models follow the same curve. This suggests a shared geometric property of the embedding covariance spectrum that's independent of architecture, training data, and dimensionality.

---

## Verdict

**PARTIALLY VERIFIED.** Df x alpha is architecture-invariant at fixed N (the shared f(N) function is the real finding, confirmed across 5 models including BERT). The v1 claim that Df x alpha = 8e specifically is false — 8e = 21.746 is f(75), an artifact of measuring at N=75. At N=130 the product is 34.7. The architecture invariance is real and confirmed (CV 0.5-5.4%). The specific value 8e is not. The real invariant is the shared f(N) function and the underlying exponent alpha ~ 0.5.
