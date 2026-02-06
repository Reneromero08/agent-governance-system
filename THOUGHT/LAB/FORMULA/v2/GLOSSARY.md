# v2 Glossary

**Version:** 2.0
**Rule:** ONE definition per symbol. No silent substitution. If a Q needs a variant, it must be declared explicitly.

---

## Core Formula

```
R = (E / grad_S) * sigma^Df
```

**Simplified form** (when sigma^Df is dropped):

```
R_simple = E / grad_S
```

---

## Symbol Definitions

### E -- Epistemic Concordance (Agreement)

**Definition:**
```
E = (1 / C(n,2)) * sum_{i<j} cos(x_i, x_j)
```

Mean pairwise cosine similarity of n observation vectors x_1, ..., x_n.

- **Domain:** [-1, 1] (in practice, typically [0, 1] for normalized embeddings)
- **Interpretation:** How much the observations agree with each other
- **Edge cases:** E = 1 when all observations are identical. E = 0 for orthogonal observations.
- **n < 2:** Undefined. Minimum 2 observations required.

**NOT used in v2:**
- E = exp(-z^2/2) (Gaussian kernel -- this was v1's analytical convenience)
- E = 1/(1+std) (v1 toy proxy)
- E = 1/(1+error) (v1 variant)
- E = 1.0 (v1 hardcoded constant)

If any of these are needed for a specific analytical purpose, they must be declared as E_gaussian, E_proxy, etc. and the relationship to operational E must be proven.

### grad_S -- Entropy Gradient (Disagreement)

**Definition:**
```
grad_S = std({cos(x_i, x_j) : i < j})
```

Standard deviation of all pairwise cosine similarities.

- **Domain:** [0, inf) in theory; in practice [0, 1]
- **Interpretation:** How much the observations disagree with each other
- **Edge cases:** grad_S = 0 when all observations are identical (handle with epsilon floor)
- **Dimensionality:** Scalar. NOT a gradient vector despite the name. The name is historical.

### sigma -- Symbolic Compression

**Definition:**
```
sigma = V_eff / V_total
```

Ratio of effective vocabulary (observed unique tokens) to total vocabulary size.

- **Domain:** (0, 1]
- **Interpretation:** How compressed the representation is
- **Measurement:** Computed from the token distribution of the observations, NOT from the embedding vectors
- **v1 issue:** sigma was assumed universal (e^(-4/pi)). v2 measures it per dataset.

### Df -- Fractal Dimension

**Definition:**
```
Df = participation ratio of eigenvalues of the embedding covariance matrix
Df = (sum(lambda_i))^2 / sum(lambda_i^2)
```

where lambda_i are eigenvalues of the covariance matrix of the observation embeddings.

- **Domain:** [1, d] where d is embedding dimensionality
- **Interpretation:** Effective dimensionality of the embedding subspace
- **Measurement:** Computed from SVD/eigendecomposition of centered observation matrix

### R -- Semiotic Resonance

**Definition:**
```
R = (E / grad_S) * sigma^Df
```

- **Domain:** [0, inf)
- **Interpretation:** Signal-to-noise ratio of semantic agreement, scaled by compression
- **When grad_S = 0:** Use R = E / (grad_S + epsilon), epsilon = 1e-10
- **When sigma^Df overflows:** Report the overflow. Do NOT silently switch to R_simple. Document and flag.

### R_simple -- Simplified Semiotic Resonance

**Definition:**
```
R_simple = E / grad_S
```

Used when sigma^Df is dropped (intentionally or due to overflow).

- **Interpretation:** Raw signal-to-noise ratio of semantic agreement
- **Relationship to R:** R = R_simple * sigma^Df

---

## Derived Quantities

### alpha -- Spectral Exponent

**Definition:** Power-law exponent of the eigenvalue spectrum.
```
lambda_k ~ k^(-alpha)
```

Measured by log-log regression of sorted eigenvalues.

### Df * alpha -- Spectral Product

**Definition:** Product of fractal dimension and spectral exponent.
**v1 claim:** Df * alpha = 8e (conservation law). v2 status: OPEN, to be tested.

---

## Embedding Models

v2 tests should use embeddings from:
- **Minimum:** At least 2 architecturally different models (e.g., one BERT-family, one non-transformer)
- **Recommended:** 5+ models spanning different architectures, training data, and dimensionalities
- **Report:** Model name, version, dimensionality, and training data description for every result

---

## Data Requirements

- **Minimum n:** 5 observations per measurement (per Q26 finding)
- **Real data required:** See METHODOLOGY.md Rule 2
- **Report:** Dataset name, version, split, sample size, and any preprocessing applied
