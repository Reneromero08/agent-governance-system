# Chronoflux QEC Mapping: Phase 2

**Date:** 2026-05-18 | **Agent:** OpenCode | **Task:** `QEC Herbert.md`

---

## Phase 1 Summary: Literature Already Ingested

The following Chronoflux papers were converted and analyzed during the hbar_sem formalization session (2026-05-17):

| Paper | Key Finding |
|-------|-------------|
| Fine Structure Constant | α = Z₀/(2R_K), circulation quantum q* = ℏ g_cf |
| QFT | Gauge structure from temporal shear, SM recovery |
| Temporal Calculus | α₀ = -√(8πG/(βc³)), |α₀| ≲ 10⁻³ |
| Informational Damping | k_B T_eff = εℏν, DI = ξ_I σ_i |
| 2022 Formulation Guide | Full 5D action, Ω_μν vorticity, Σ_μν shear, ℓ_H = √(α₁/ν) |

---

## Phase 2: Candidate Mappings

### Mapping 1 (PRIMARY): Circulation Stability → Sigma

**Chronoflux:** Stable temporal circulation requires the phase to be single-valued along closed curves. The stability condition is the balance between vorticity Ω_μν (driving the circulation pattern) and shear Σ_μν (dissipating it):

```
Stable:  Ω_rms > Σ_rms  (vorticity dominates)
Unstable: Ω_rms < Σ_rms  (shear dominates)
```

where Ω_rms = √(Ω_μν Ω^μν) and Σ_rms = √(Σ_μν Σ^μν).

**Semiotic:** The fidelity factor σ measures the log-error-suppression per unit D_f. It crosses 1.0 at the QEC threshold and is >1 below threshold (code corrects errors) and <1 above threshold (code fails).

**Mapping:**

```
σ = Ω_rms / Σ_rms
```

In QEC terms:
- Ω_rms ∝ code compression strength = function of stabilizer weight w and code distance d
- Σ_rms ∝ physical error rate p (the noise dissipating the code)

At the threshold p = p_th: Ω_rms = Σ_rms → σ = 1.
Below threshold (p < p_th): Ω_rms > Σ_rms → σ > 1.
Above threshold (p > p_th): Ω_rms < Σ_rms → σ < 1.

**Operationalization:** For a rotated surface code of distance d with stabilizer weight w:
- Ω_rms = C_Ω × d / w  (circulation strength: deeper code, lighter stabilizers = more stable)
- Σ_rms = C_Σ × p     (shear: higher error rate = more dissipation)

Therefore:
```
σ_CF(p, d) = (C_Ω × d / w) / (C_Σ × p)
           = C × d / (w × p)
```

where C = C_Ω / C_Σ is a single calibration constant.

At p = p_th(d), σ_CF = 1, so:
```
C × d / (w × p_th(d)) = 1  →  p_th(d) = C × d / w
```

For the rotated surface code: w = 4 (weight-4 stabilizers), so p_th(d) ∝ d. But the actual threshold is approximately constant (~0.007 for depolarizing noise), not linear in d. This means the simple Ω_rms ∝ d/w mapping needs refinement.

**Refined mapping:** The circulation stability depends on the code's ability to correct t = floor((d-1)/2) errors. The effective circulation strength is:

```
Ω_rms ∝ (d / t) × (1/w) = d / (t × w)
```

For large d: d/t ≈ 2, so Ω_rms ≈ constant / w. For the rotated surface code with w=4: Ω_rms ≈ constant / 4. The threshold is then:

```
p_th = Ω_rms / Σ_rms_scale ≈ constant / (4 × Σ_rms_scale) ≈ 0.007
```

This gives Σ_rms_scale ≈ constant / (4 × 0.007) ≈ 36 × constant. Reasonable.

**Testable prediction:** σ_CF(p, d) should be proportional to 1/p for fixed d, and approximately constant with respect to d at fixed p.

### Mapping 2: Vorticity → Syndrome Density

**Chronoflux:** Ω_μν = ∇_μ Θ_ν - ∇_ν Θ_μ is the temporal vorticity — the local circulation density.

**Semiotic:** ∇S = syndrome density — the local error detection rate.

**Mapping:**
```
∇S = κ × |Ω|²
```

where κ is a coupling constant. The entropy gradient (dissonance) is proportional to the squared magnitude of the vorticity. When circulation is disrupted (high vorticity), syndrome density is high. When circulation is stable (low vorticity), syndrome density is low.

**Testable prediction:** ∇S should scale as ~p² at low p (since vorticity perturbations scale with p and ∇S ∝ |Ω|²), and as ~p at high p (when circulation is broken and the field is fully turbulent). The QEC data shows ∇S ~ p at all p, suggesting the QEC regime is always in the turbulent limit. This mapping may not hold for QEC but could hold for other domains.

### Mapping 3: Relaxation Length → Code Distance

**Chronoflux:** ℓ_H = √(α₁/ν) is the relaxation length of the temporal field — the distance over which perturbations decay.

**Semiotic:** Code distance d is the minimum weight of a logical operator — the "size" of the code.

**Mapping:**
```
d ∝ ℓ_H = √(α₁/ν)
```

The relaxation length in the temporal field determines the natural code distance. A longer relaxation length (stronger temporal coupling α₁, slower temporal rate ν) allows larger code distances.

**Testable prediction:** The effective code distance should saturate when ℓ_H approaches the system size. For QEC, this means codes with d >> ℓ_H provide diminishing returns.

### Mapping 4 (MOST PROMISING): Informational Damping → Sigma

**Chronoflux:** DI = ξ_I × σ_i, where σ_i is the entropy production rate density.

**Semiotic:** Sigma measures the resistance to decoherence. The fidelity factor should be related to the damping operator:

```
σ = exp(-DI × τ_gate) = exp(-ξ_I × σ_i × τ_gate)
```

At the threshold (sigma = 1): DI = 0 (the Zero Entropy Nexus — reversible channel).
Below threshold: DI < 0 (negative damping — amplification, sigma > 1).
Above threshold: DI > 0 (positive damping — decoherence, sigma < 1).

**Operationalization:**
```
σ_i = ∇S = syndrome density
τ_gate = D_f / p  (logical gate time in terms of physical error rate)
σ = exp(-ξ_I × ∇S × D_f / p)
```

This gives the fidelity factor in terms of measurable QEC quantities plus one Chronoflux constant ξ_I. The threshold condition σ = 1 requires:
```
ξ_I × ∇S_th × D_f / p_th = 0  →  ∇S_th = 0  (ZEN, reversible channel)
```

But ∇S is never exactly zero. At the threshold, DI is small but non-zero. The fidelity factor crosses 1 when the damping changes sign — which requires ξ_I to change sign or σ_i to cross zero. Neither happens in QEC.

**This mapping is inconsistent with QEC data.** The informational damping operator is always positive (DI ≥ 0), so σ should always be ≤ 1. But QEC data shows σ > 1 below threshold. The ZEN condition (where DI = 0) doesn't match the QEC threshold where σ = 1.

---

## Phase 2 Conclusion

**Mapping 1 (σ ∝ Ω_rms/Σ_rms ∝ 1/p) is the most promising.** The linear relationship σ ∝ 1/p at fixed d is approximately correct in the QEC data (sigma decreases as p increases). The d-dependence is more subtle — sigma should be approximately independent of d if the threshold is constant.

**Testing:** Compute σ_CF = C/p for each p, fit C to match the training data, and evaluate OOS on held-out distances. If σ_CF matches or exceeds the empirical sigma (R2 > 0.7), the circulation stability mapping is productive.

**Phase 3 eligibility:** YES. Mapping 1 produces a testable closed-form prediction: σ_CF = C/p with a single calibration constant.

---

*Reference: QEC Herbert.md (2026-05-14). Herbert papers from THOUGHT/LAB/FORMULA/v4/FORMALIZATION/REFERENCES/.*
