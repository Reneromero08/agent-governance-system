# Q7: Composition Axioms C1-C4

**Formal statement of the axioms that R must satisfy for multi-scale composition.**

These are analogous to Q3's axioms A1-A4, but extended to the scale transformation domain.

---

## Axiom C1: Locality

**Statement:** Composition only uses local scale information.

```
T_λ(obs, truth, σ) depends only on observations within the aggregation window.
```

**Formal:** For any partition of observations into local groups {G_i}:
```
R(T_λ({G_i})) = f(R(G_1), R(G_2), ..., R(G_n))
```
where f depends only on local R values, not on global structure outside the groups.

**Grounding:** Semiotic Axiom 0 (Information Primacy) - evidence must be computable from local information.

**Test:** Adding non-local noise should not affect R if properly isolated.

---

## Axiom C2: Associativity (Group Action)

**Statement:** Scale transformations compose associatively.

```
T_λ(T_μ(·)) = T_{λμ}(·)
```

**Formal:** For any scale factors λ, μ > 0:
```
T_λ ∘ T_μ = T_{λ·μ}
```

**Properties:**
1. **Group action:** T forms a group homomorphism from (ℝ⁺, ×) to endomorphisms
2. **Identity:** T_1 = id (scale factor 1 does nothing)
3. **Inverse:** T_λ^{-1} = T_{1/λ} (conceptual - information-losing in practice)

**Grounding:** Mathematical structure - scale transformations form a group.

**Test:** Order reversal: T_2(T_3(·)) vs T_3(T_2(·)) should yield same result.

---

## Axiom C3: Functoriality

**Statement:** Structure is preserved across scales.

```
F(X → Y) = F(X) → F(Y)
```

**Formal:** If L_child and L_parent are semantic L-functions at different scales:
```
correlation(L_child, L_parent) > τ_functoriality
```

The "lifting map" φ: Rep(G_child) → Rep(G_parent) preserves semantic structure.

**Properties:**
1. Neighborhood preservation: If x ≈ y at child scale, their aggregates remain related
2. Spectral preservation: Eigenvalue structure is preserved under aggregation
3. L-function correlation: Semantic complexity measures correlate across scales

**Grounding:** Category theory - functors preserve structure.

**Test:** Shuffle hierarchy and measure structure loss (should preserve > 90%).

---

## Axiom C4: Intensivity

**Statement:** R doesn't grow or shrink systematically with scale.

```
R(T_λ(obs)) ≈ R(obs) for all λ > 0
```

**Formal:** The coefficient of variation of R across scales should be bounded:
```
CV(R_scale_1, R_scale_2, ..., R_scale_k) < ε
```

**Interpretation:**
- R is **intensive** (like temperature, density, pressure)
- NOT **extensive** (like mass, volume, energy)
- Signal QUALITY, not signal VOLUME

**Grounding:** Q15 proved R is intensive (independent of sample size).

**Test:** Scale sweep from 0.1× to 100× - R should have CV < 0.1.

---

## Uniqueness Theorem

**Theorem:** R = E(z)/σ is the UNIQUE form satisfying all four axioms C1-C4.

**Proof sketch:**

1. **C1 (Locality)** forces R to be a function of local statistics:
   ```
   R = f(obs, truth, σ) where f is local
   ```

2. **C2 (Associativity)** forces R to be scale-covariant:
   ```
   R(T_λ(·)) = g(λ) · R(·) for some function g
   ```

3. **C3 (Functoriality)** forces the structure to be preserved:
   ```
   R must be a "nice" function of evidence (monotonic, continuous)
   ```

4. **C4 (Intensivity)** forces g(λ) = 1 (R is invariant):
   ```
   R(T_λ(·)) = R(·)
   ```

**Combining:** The only local, scale-covariant, structure-preserving, intensive measure is:
```
R = E(z) / σ
```
where z = (obs - truth) / σ is the normalized error.

---

## Alternatives That Fail

| Form | Fails Axiom | Reason |
|------|-------------|--------|
| E/σ² | C4 | Scales as 1/σ² (extensive, not intensive) |
| E²/σ | C3 | Non-linear; breaks functoriality |
| E - σ | C4 | Additive; not multiplicative-scale-covariant |
| E × σ | C2 | Wrong scaling direction |
| Σ E_i | C4 | Extensive; grows with sample size |
| max(E_i) | C3 | Non-smooth; breaks structure preservation |

---

## Connection to Q3

Q3's axioms A1-A4 for the base formula:
- **A1 (Locality)** ↔ **C1 (Locality)** - Same principle
- **A2 (Normalized Deviation)** → Implicit in C4 (intensive requires z = (obs - truth) / σ)
- **A3 (Monotonicity)** → Implicit in C3 (functoriality requires monotonic E)
- **A4 (Intensive)** ↔ **C4 (Intensivity)** - Same property

The composition axioms are the NATURAL EXTENSION of Q3's axioms to the multi-scale domain.

---

## Connection to Physics

**Renormalization Group:** In quantum field theory, the RG describes how physical parameters change with scale. A **fixed point** is a set of parameters that don't change under scale transformation.

**Our claim:** R = E(z)/σ is an RG fixed point because:
1. The functional form is preserved under scale transformation
2. The β-function β(R) = dR/d(ln λ) ≈ 0
3. All "reasonable" evidence measures flow to R under repeated coarse-graining

**Universality:** Different physical systems at the same RG fixed point exhibit identical critical behavior. Similarly, different domains (quantum, neural, social) with the same R structure should exhibit identical semantic behavior.

---

## Test Matrix

| Axiom | Test Method | Pass Threshold |
|-------|-------------|----------------|
| C1 | Non-local injection | R unchanged (error < 0.1) |
| C2 | Order reversal | Difference < 1e-6 |
| C3 | Hierarchy shuffle | Structure preserved > 90% |
| C4 | Scale sweep | CV < 0.1 |

All four axioms must pass for R to be confirmed as the unique multi-scale composition law.

---

*Last Updated: 2026-01-11*
