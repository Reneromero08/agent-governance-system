# Living Formula: Formal Glossary

**Version:** 1.0.0
**Status:** Canonical
**Last Updated:** 2026-02-05

---

## Purpose

This glossary provides **unique, unambiguous definitions** for every symbol and term
used in the Living Formula framework. Each symbol has exactly ONE definition.
Domain-specific interpretations are noted as such but do not change the formal definition.

---

## Core Formula

```
R = (E / grad_S) * sigma^Df
```

Where:
- R = Resonance (dimensionless ratio)
- E = Essence (dimensionless, domain-dependent; see Definition 2)
- grad_S = Semantic gradient (dimensionless)
- sigma = Noise floor / baseline variance (dimensionless, 0 < sigma < 1)
- Df = Fractal dimension of the signal structure (dimensionless, Df >= 0)

---

## Formal Definitions

### Definition 1: Resonance (R)

**Symbol:** R
**Type:** Dimensionless scalar, R >= 0
**Formula:** R = (E / grad_S) * sigma^Df
**Interpretation:** A measure of how strongly a signal coheres relative to noise.
High R indicates structured, aligned information; low R indicates disorder.

**Operational test:** R > 1 means signal dominates noise. R < 1 means noise dominates.

### Definition 2: Essence (E)

**Symbol:** E
**Type:** Dimensionless scalar, E >= 0
**Formula:** Domain-dependent (see table below)

| Domain | E Definition | Source |
|--------|-------------|--------|
| Semantic | Mean pairwise cosine similarity of embedding cluster | Measured from embeddings |
| Quantum | Mutual information I(S:F) between system and fragment | Zurek (2009) |
| Wave | Amplitude coherence \|&lt;psi_+\|psi_-&gt;\|^2 | Inner product of mode pair |
| General | Normalized alignment score of a signal ensemble | Context-dependent |

**CRITICAL NOTE:** E is NOT energy in the physics sense. It is a dimensionless
alignment measure. The name "Essence" is metaphorical. Each domain must specify
its operational definition of E before computing R.

### Definition 3: Semantic Gradient (grad_S)

**Symbol:** grad_S
**Type:** Dimensionless scalar, grad_S > 0
**Formula:** Standard deviation of the E measurements across the ensemble
**Interpretation:** The local variability or "noise" in the alignment signal.

**Degenerate case:** If grad_S -> 0, R diverges. This indicates perfect agreement
with zero variance - a degenerate case that should be flagged, not interpreted as
"infinite resonance."

### Definition 4: Noise Floor (sigma)

**Symbol:** sigma
**Type:** Dimensionless scalar, 0 < sigma < 1
**Empirical value:** sigma ~ 0.27 (observed across embedding models)
**Status:** EMPIRICAL OBSERVATION. Not derived from first principles.

**Post-hoc fit:** sigma = e^(-4/pi) = 0.2805 has been proposed but is a curve fit
to observed data, not an independent derivation. See Q25 for details.

### Definition 5: Fractal Dimension (Df)

**Symbol:** Df
**Type:** Non-negative real, Df >= 0
**Formula:** Measured from eigenvalue spectrum decay: lambda_k ~ k^(-1/Df)
**Interpretation:** The structural complexity of the signal. Higher Df indicates
more self-similar, scale-invariant structure.

### Definition 6: Eigenvalue Decay Exponent (alpha)

**Symbol:** alpha
**Type:** Non-negative real
**Formula:** lambda_k ~ k^(-alpha), where lambda_k are eigenvalues of the
covariance matrix sorted in descending order
**Relationship to Df:** alpha = 1 / (2 * Df) (for CP^n manifolds)
**Empirical value:** alpha ~ 0.5 for most semantic embedding models

**WARNING:** This alpha has NO relation to the fine structure constant (alpha ~ 1/137).
They are entirely different quantities that share a Greek letter.

### Definition 7: Conservation Product (Df * alpha)

**Symbol:** Df * alpha
**Type:** Dimensionless scalar
**Empirical value:** Df * alpha ~ 21.75 (CV = 6.93% across 24 embedding models)
**Proposed identity:** Df * alpha = 8e = 21.746

**Status:** EMPIRICAL OBSERVATION with possible numerological fit. The value 8e
has been proposed but the derivation is not rigorous. See HONEST_FINAL_STATUS.md.

---

## Derived Quantities

### log(R) and Free Energy

**Relationship:** log(R) = -F + const, where F is the variational free energy
in the Free Energy Principle sense (Friston, 2010).

**Status:** Mathematical identity under specific assumptions. Not experimentally validated.

### Born Rule Correlation

**Claim:** E = |<psi|phi>|^2 (Born rule probability)
**Status:** Correlation r = 0.999 observed on synthetic quantum simulations.
Not tested on real experimental data.

---

## Status Labels

| Label | Meaning |
|-------|---------|
| ANSWERED | Question resolved with supporting evidence |
| PARTIAL | Some phases complete; key aspects remain open |
| EXPLORATORY | Framework proposed but not independently validated |
| VALIDATED | Independently reproduced by external methods |
| CONFIRMED | Internal tests pass consistently |
| FALSIFIED | Evidence contradicts the hypothesis |
| OPEN | Not yet investigated |
| DERIVED | Mathematically derived from axioms (use with caution) |

---

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Zurek, W.H. (2009). Quantum Darwinism.
- Peirce, C.S. (1903). Collected Papers (semiotic categories).
- Zhu, Q. et al. (2022). Quantum advantage experiments.
