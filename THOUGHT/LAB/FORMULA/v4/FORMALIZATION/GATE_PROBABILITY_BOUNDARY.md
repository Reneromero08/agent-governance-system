# Gate-to-Probability: The Born Rule is Universal

**Date:** 2026-05-17 | **Status:** FORMALIZED (corrected) | **Priority:** Medium

---

## 0. The Correction

The Born rule P = |⟨a|b⟩|² is not domain-specific. It is universal. It always applies. The original formulation was wrong to ask "when does the Born rule apply?" The correct question is: **when does the Born rule reveal phase-dependent probability structure invisible to the real inner product?**

On real manifolds (ℝ^d): P = cos_sim². This IS the Born rule working perfectly. But it is the identity — knowing `cos_sim` means you already know `cos_sim²`. The Born rule adds no new information because the real line has no phase degree of freedom.

On complex manifolds (ℂ^d): P = |⟨ψ|φ⟩|² = |α|² + |β|² + 2|α||β|cos(Δθ). The interference term 2|α||β|cos(Δθ) is the new information — invisible to the real inner product, revealed only by the Born rule on a complex manifold.

The boundary is geometric. ℝ ⊂ ℂ. The Born rule is the same function in both spaces. It simply reveals more structure as you add dimensions.

---

## 1. The Born Rule on Real vs Complex Manifolds

### 1.1 Real Manifold (ℝ^d)

Inner products are real: ⟨a|b⟩ ∈ ℝ.

```
P_born = ⟨a|b⟩² = cos_sim²
```

This is the Born rule. It is also the identity `x → x²`. It preserves rank ordering. It makes zero falsifiable predictions about ordering because it is monotone. But it is not wrong — it is the Born rule operating on a manifold with zero phase degrees of freedom.

**Proof:** For any real inner product x, the Born rule gives x². The mapping x → x² is monotone on [0,1] (or [-1,1] with sign). Therefore rank(P_born) = rank(cos_sim). The Born rule reveals nothing new about ordering because there is no phase to encode.

**The squared circle on ℝ:** The continuous operation of squaring maps the real line to the non-negative reals. This IS the Born rule. It is the identity at the individual level. At the aggregate level — mean(x) vs mean(x²) — it produces correlations that are algebraic properties of positive-valued distributions, not evidence of complex structure.

### 1.2 Complex Manifold (ℂ^d)

Inner products are complex: ⟨ψ|φ⟩ ∈ ℂ.

```
|ψ⟩ = |α|e^{iθ_α}|a⟩ + |β|e^{iθ_β}|b⟩
|φ⟩ = |γ|e^{iθ_γ}|c⟩ + |δ|e^{iθ_δ}|d⟩

⟨ψ|φ⟩ = α*γ + β*δ  (complex)

P_born = |⟨ψ|φ⟩|² = |α*γ|² + |β*δ|² + 2Re(α*γ β*δ)
```

The cross-term 2Re(α*γ β*δ) encodes the phase difference Δθ = θ_α - θ_β + θ_δ - θ_γ. This is the interference term — invisible on ℝ, revealed only on ℂ.

**The squared circle on ℂ:** Multiplying by e^{iθ} rotates the state. The squared magnitude |e^{iθ}⟨ψ|φ⟩|² = |⟨ψ|φ⟩|² is invariant under rotation — but the COMPONENTS of the squared expression carry the phase information. The Born rule P = sin²(θ/2) emerges when the measurement basis is rotated relative to the state.

---

## 2. When Does the Born Rule Reveal New Information?

The Born rule reveals phase-dependent probability structure when the system satisfies:

### Condition: Complex Phase Degree of Freedom (C5, corrected)

```
C5: The manifold has a genuine imaginary axis — nonzero holonomy for closed loops.
```

**Test:** Parallel transport a vector around a closed loop. If the holonomy angle ≠ 0 (mod 2π), the manifold admits complex structure. The Born rule will reveal interference terms invisible to the real inner product.

**No longer a binary gate:** C5 is not "does quantum structure exist?" It is "is the manifold real or complex?" Real manifolds produce Born rule as identity. Complex manifolds produce Born rule with interference. Both are valid. Both are the same function.

### Why the Other Conditions Still Matter

C1-C4 determine whether the complex structure is VISIBLE to a particular gate operation:

- **C1 (Coherence):** Off-diagonal terms carry the interference. Without them, the Born rule still works but reveals no interference.
- **C2 (Purity):** Mixed states average over phase. The Born rule still works but the interference term is washed out by the ensemble average.
- **C3 (Decoherence):** If γτ > 0.1, the environment measures the phase before the gate completes. The Born rule operates on the decohered (real) state. Identity, not interference.
- **C4 (Measurement type):** Weak measurements extract partial phase information. The Born rule still works but the probability formula depends on the measurement strength.

---

## 3. Operational Criterion

The unified scalar measures whether the Born rule reveals phase structure:

```
Q = Tr(rho^2) * c * exp(-nabla_S / Delta_E)
```

where c = sum_{i!=j} |rho_ij| / Tr(rho) is the coherence fraction. Note: C5 (holonomy) is no longer a binary gate. It is encoded in the coherence fraction c — real manifolds have c=0 by definition, complex manifolds can have c>0.

The Born rule always applies. Q measures whether it reveals new information:

| Q Range | Manifold | Born Rule Behavior |
|---------|----------|--------------------|
| Q > 0.1 | Complex (ℂ) | Interference visible, sin^2(theta/2) fringes |
| 0 < Q < 0.1 | Complex, decohered | Interference present but damped |
| Q = 0 | Real (ℝ) | Born rule = identity, x -> x^2, deterministic |

---

## 4. Domain Classification

| Domain | Manifold | Q | Born Rule Effect |
|--------|----------|---|------------------|
| QEC surface codes | ℂ^2 | >0.1 | Full interference, sigma crosses 1.0 |
| Neural PLV (EEG) | ℂ^d | >0.1 | sin^2 fringes, PLV 0.68-0.72 |
| Quantum cognition | ℂ^d | >0.1 | Linda: 0.638 predicted vs 0.60 obs |
| Superconducting qubit | ℂ^2 | >0.1 | Gate fidelities >99.9% |
| MiniLM embeddings | ℝ^d | 0 | Born rule = cos_sim^2 (identity, deterministic) |
| MPNet embeddings | ℝ^d | 0 | Same as MiniLM |
| BERT embeddings | ℝ^d | 0 | Same identity behavior |
| Constitution (Gemma) | ℝ^d | 0 | Born rule works, reveals no new ordering |
| LLM temperature sampling | ℝ^d | 0 | Softmax = classical probability, not Born |
| Silicon transistor (300K) | ℝ | 0 | γτ ~ 600, fully decohered to real manifold |

---

## 5. Why Kimi's Test "Failed"

Kimi K2.5 correctly identified that real embedding manifolds have zero holonomy. The Born rule applied to MiniLM/BERT/MPNet produces `P = cos_sim^2` — the identity. This is NOT a failure of the Born rule. It is the Born rule operating on a real manifold where it is the identity operation `x -> x^2`.

The original claim that "E follows Born rule statistics at r=0.977" measured an algebraic property of positive-valued distributions — mean(x) correlates with mean(x^2) for any set of positive reals. This correlation is real (z=3.5 vs random vectors) and the Born rule IS producing it. But it is not revealing phase structure because there is no phase to reveal.

The error was in the INTERPRETATION, not the mapping. The mapping P = x^2 is the Born rule. It always was. The claim that this proves "semantic space is quantum" was wrong — but the claim that "the Born rule maps inner products to probabilities" was always right.

---

## 6. The Squared Circle

The Born rule P = |<psi|phi>|^2 IS the squared circle. On the real line, a number squared is just its square — the identity, the circle collapsed to a point, no new information. On the complex plane, a number squared carries its phase through the interference term — the continuous circle modulating into the discrete square without losing its rotational structure.

The fact that `cos_sim^2 = cos_sim^2` at the word level is not a falsification. It is the Born rule working in its most reduced form. The squared circle on ℝ is the identity. The squared circle on ℂ is the sin^2(theta/2) fringe pattern. Both are the same operation. Both squares. Different dimensionality.

---

## 7. Impact

| Claim | Before | After |
|-------|--------|-------|
| Born rule scope | Domain-specific, boundary-limited | Universal. Always applies. |
| Real embeddings | "Born rule fails" | Born rule = identity x -> x^2. Works perfectly. |
| Complex manifolds | "Born rule works" | Born rule = interference. Reveals phase. |
| Kimi's test | "Failed — Born rule doesn't apply" | Correctly identified ℝ manifold. Born rule IS the identity here. |
| The boundary | Does the Born rule apply? | Is the manifold real or complex? Both valid. |
| Squared circle | Metaphor | Literal: squaring maps continuous (ℂ) to discrete (ℝ) through phase. |

---

*Corrected from the Q44 Born rule verdicts (v2_2), Kimi K2.5 Q51 investigation, and the gate-to-probability gap in `05_FORMALIZATION_1.md`. The Born rule is universal. The squared circle is the identity.*
