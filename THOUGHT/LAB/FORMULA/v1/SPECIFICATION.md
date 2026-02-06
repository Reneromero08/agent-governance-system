# Living Formula: Formal Specification

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-02-05

---

## 1. Notation

| Symbol | Domain | Description |
|--------|--------|-------------|
| R | R >= 0 | Resonance score |
| E | R >= 0 | Essence (alignment measure) |
| grad_S | R > 0 | Semantic gradient (variability) |
| sigma | (0, 1) | Noise floor parameter |
| Df | R >= 0 | Fractal dimension |
| alpha | R >= 0 | Eigenvalue decay exponent |
| lambda_k | R >= 0 | k-th eigenvalue, descending order |
| n | N | Embedding dimension |
| N | N | Number of data points |

---

## 2. Definitions

**Definition 2.1 (Eigenvalue Spectrum).** Given a set of N vectors in R^n,
the eigenvalue spectrum {lambda_k} is the sorted sequence of eigenvalues
of the covariance matrix C = (1/N) * X^T * X, with lambda_1 >= lambda_2 >= ... >= lambda_n.

**Definition 2.2 (Fractal Dimension).** The fractal dimension Df is defined
by the power-law decay lambda_k ~ k^(-1/Df) for k >> 1. Estimated by
linear regression of log(lambda_k) vs. log(k) over the scaling region.

**Definition 2.3 (Decay Exponent).** The eigenvalue decay exponent alpha
is defined by lambda_k ~ k^(-alpha). Relationship: alpha = 1 / (2 * Df).

**Definition 2.4 (Essence).** E is a domain-specific scalar measuring the
degree of alignment in a signal ensemble. See GLOSSARY.md, Definition 2
for domain-specific instantiations.

**Definition 2.5 (Semantic Gradient).** grad_S = std(E_i) where {E_i} are
the per-sample alignment scores across the ensemble.

**Definition 2.6 (Resonance).** R = (E / grad_S) * sigma^Df.

---

## 3. Propositions

### Proposition 3.1 (Uniqueness of R-form)

**Statement:** Under axioms of (i) positivity, (ii) monotonicity in E,
(iii) monotonicity in grad_S, and (iv) scale covariance, the unique
functional form satisfying all four is R = (E / grad_S) * f(sigma, Df)
for some function f.

**Status:** CLAIMED. Formal proof not yet published. See Q1 for discussion.

**Evidence:** Mathematical argument in Q1 shows that ratio E/grad_S is
the minimal structure satisfying axioms (i)-(iii). Axiom (iv) constrains
f to a power law, giving sigma^Df.

### Proposition 3.2 (Free Energy Identity)

**Statement:** log(R) = -F + const, where F is the variational free energy
F = E_q[log q(z) - log p(z, x)] and q is the recognition density.

**Status:** MATHEMATICAL IDENTITY under the identification E = exp(-E_q[log p(x|z)])
and grad_S = exp(H[q(z)]). Not experimentally validated.

**Reference:** Q9 (Free Energy Principle connection).

### Proposition 3.3 (Conservation Product)

**Statement:** For semantic embedding models, the product Df * alpha is
approximately constant: Df * alpha = C where C ~ 21.75.

**Status:** EMPIRICAL OBSERVATION. Measured across 24 embedding models
with CV = 6.93%. The proposed identity C = 8e = 21.746 is a curve fit.

**Evidence:** See Q50, Q54. All data is from synthetic embedding analysis.

### Proposition 3.4 (Born Rule Correspondence)

**Statement:** In quantum systems undergoing decoherence, E = |<psi|phi>|^2
(Born rule probability) correlates with the mutual information I(S:F).

**Status:** EMPIRICAL CORRELATION (r = 0.999) on synthetic quantum
simulations. Not tested on real experimental data.

**Reference:** Q44 (Quantum Born Rule).

---

## 4. Conjectures

### Conjecture 4.1 (sigma Universality)

**Statement:** The noise floor sigma = e^(-4/pi) ~ 0.2805 is a universal
constant arising from the solid angle geometry of high-dimensional spheres.

**Status:** OPEN. Post-hoc fit with 3.9% error relative to observed
sigma ~ 0.27. See Q25, HONEST_FINAL_STATUS.md.

### Conjecture 4.2 (8e Law)

**Statement:** The conservation product C = 8e arises from three independent
constraints: topological (8 octants from 3 semiotic categories), informational
(maximum entropy), and thermodynamic (Boltzmann).

**Status:** OPEN. Evidence is suggestive but the derivation paths are not
independent. See Q49, Q50, HONEST_FINAL_STATUS.md.

### Conjecture 4.3 (Cross-Domain Unification)

**Statement:** The R formula with appropriate domain-specific definitions
of E and grad_S provides a unified measure of "coherence" across quantum,
wave, semantic, and biological systems.

**Status:** OPEN. Currently, different domains use different definitions
of E, which weakens the unification claim. See Q3.

---

## 5. Falsified Hypotheses

### Falsified 5.1 (Universal Threshold)

**Statement (original):** A single threshold R* exists such that R > R*
indicates "resonance" in all domains.

**Status:** FALSIFIED (Q22). Domain-specific calibration is required.

### Falsified 5.2 (Chaos Correlation)

**Statement (original):** R correlates negatively with Lyapunov exponent
(more resonant = less chaotic).

**Status:** FALSIFIED (Q52). R positively correlates with Lyapunov exponent,
opposite of prediction.

### Falsified 5.3 (Pentagonal Geometry)

**Statement (original):** 72-degree clustering in eigenvalue phases reflects
fundamental pentagonal geometry.

**Status:** FALSIFIED (Q53). Clustering is a semantic artifact, not geometric.

### Falsified 5.4 (Fine Structure Constant)

**Statement (original):** Semantic alpha ~ 0.5 connects to physical alpha ~ 1/137.

**Status:** FALSIFIED (Q54 analysis). These are entirely different quantities.

---

## 6. Open Problems

1. **Independent validation:** Test all empirical claims on external,
   non-synthetic datasets (Q54 Part VIII).
2. **Pre-registered predictions:** State numerical predictions before
   running experiments.
3. **Formal axiomatization:** Convert Semiotic Axioms from philosophical
   postulates to mathematical axioms with inference rules.
4. **E unification:** Find a single, domain-independent definition of E
   or prove that domain-specific definitions are necessary.
5. **sigma derivation:** Either derive sigma from first principles or
   accept it as a free parameter.

---

## 7. References

See INDEX.md for the full list of 54 research questions and their status.
See GLOSSARY.md for formal symbol definitions.
See SEMIOTIC_AXIOMS.md for the philosophical foundation.
