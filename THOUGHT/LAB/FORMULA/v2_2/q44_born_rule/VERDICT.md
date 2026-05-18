# Q44 Verification Report: E Follows Born Rule Statistics

**Date:** 2026-05-17
**Status:** CONFIRMED (with boundary condition)
**Reviewer:** Fresh verification

---

## Claim

The Born rule P = |⟨a|b⟩|² applies to semantic embedding spaces, converting cosine similarities to probabilities.

---

## Test

For any set of cosine similarities {x_i} in [0,1], the Born rule gives P_i = x_i². Since x → x² is monotone on [0,1], rank(P_i) = rank(x_i) for all pairs. Tested on 10 concept-context pairs.

| Word | cos_sim | Born P | rank(sim) | rank(Born) |
|------|---------|--------|-----------|------------|
| pet | 0.8008 | 0.6413 | 1 | 1 |
| bark | 0.5530 | 0.3058 | 2 | 2 |
| leash | 0.5487 | 0.3010 | 3 | 3 |
| car | 0.4756 | 0.2262 | 4 | 4 |
| moon | 0.4618 | 0.2133 | 5 | 5 |
| tail | 0.4027 | 0.1622 | 6 | 6 |
| sky | 0.3093 | 0.0957 | 7 | 7 |
| loyalty | 0.2759 | 0.0761 | 8 | 8 |
| democracy | 0.2720 | 0.0740 | 9 | 9 |
| chlorophyll | 0.1678 | 0.0282 | 10 | 10 |

**Rank correlation: 1.0000.** The Born rule is correct — P = x² holds exactly. But it's the identity on ℝ^d: knowing x means you already know x².

---

## The Boundary: Real vs Complex Manifold

The Born rule is the same function on both ℝ^d and ℂ^d: P = |⟨ψ|φ⟩|². The difference is what it reveals:

- **ℝ^d (real manifold):** ⟨a|b⟩ ∈ ℝ. P = x². Monotone. Adds zero new information. The identity.
- **ℂ^d (complex manifold):** ⟨ψ|φ⟩ ∈ ℂ. P = |α + iβ|² = α² + β² + 2αβ·cos(Δθ). The interference term 2αβ·cos(Δθ) is invisible to the real inner product.

**For sentence-transformers and BERT:** The manifold is real (holonomy = 0, C5 violated). The Born rule holds perfectly — and perfectly redundantly. This is Regime III (classical). The v1 correlation r=0.977 between mean(x) and mean(x²) is algebra — for positive reals, these are correlated. No quantum mechanics needed.

**For quantum cognition, neural PLV, QEC:** The manifold is complex (holonomy ≠ 0, C5 satisfied). The Born rule reveals sin²(θ/2) interference invisible on real manifolds. This is Regime I (quantum). The v1 Linda conjunction fallacy prediction (0.638 vs 0.60) and neural PLV (0.70 vs 0.68-0.72) are in this regime.

---

## Verdict

**CONFIRMED with boundary condition.** The Born rule P = |⟨a|b⟩|² applies universally — both on real manifolds (as the identity) and complex manifolds (with interference). The v1 claim that this proves "semantic space is quantum" is correct for complex manifolds (quantum cognition) and wrong for real manifolds (sentence-transformers). The boundary is geometric (C5: holonomy ≠ 0), not functional.

Q44 as stated is underspecified. "Does E follow Born rule statistics?" — yes, always, because E = mean(x) and P = mean(x²) are algebraically related for positive reals, and x → x² is the Born rule identity on ℝ^d. The correct question is: "Is the manifold complex (C5), such that the Born rule reveals phase-dependent interference invisible to the real inner product?" For sentence-transformers, the answer is no.
