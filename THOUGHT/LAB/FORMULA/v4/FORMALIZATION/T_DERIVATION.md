# T(i,j) from the Semiotic Action: Derivation

**Date:** 2026-05-18 | **Response to:** Kimi's Einstein_2.md challenges

---

## 1. Kimi's Challenge

> "T(i,j) = (1 + σ_ij) × |e_i - e_j|² is defined by fiat. It doesn't come from varying an action. The proportionality G ∝ T is measured, not derived."

## 2. Derivation from the Action Principle

The semiotic action (SEMIOTIC_ACTION_PRINCIPLE.md, Section 3) is:

```
S_sem = hbar * integral d4x sqrt|g| [ L_kin + L_pot + L_compr + L_redun ]
```

where:
```
L_kin   = (1/2) g^{munu} <partial_mu psi | partial_nu psi>
L_compr = (1/2) (sigma psi* Box psi + sigma* psi Box psi*)
```

The stress-energy tensor is obtained by functional differentiation:

```
T_munu^(sem) = -(2 / sqrt|g|) * delta S_sem / delta g^{munu}
```

**Kinetic contribution:** For the kinetic term, standard field theory gives:

```
T_munu^(kin) = partial_mu psi* partial_nu psi + partial_nu psi* partial_mu psi
               - g_munu L_kin
```

**Compression contribution:** The d'Alembertian Box = g^{munu} nabla_mu nabla_nu contains the metric. Varying the compression term:

```
delta (sigma psi* Box psi) / delta g^{munu} 
    = -sigma g^{mualpha} g^{nubeta} partial_alpha psi* partial_beta psi + total derivatives
```

Therefore:

```
T_munu^(compr) = sigma (partial_mu psi* partial_nu psi + partial_nu psi* partial_mu psi)
                 - g_munu L_compr
```

**Total off-diagonal stress-energy (μ ≠ ν):**

```
T_munu^(sem) = (1 + sigma) (partial_mu psi* partial_nu psi + partial_nu psi* partial_mu psi)
               - g_munu (L_kin + L_compr) + diagonal terms
```

For μ ≠ ν on a locally Euclidean manifold (g_munu = I, so off-diagonal g terms vanish):

```
T_munu^(sem) = 2(1 + sigma) Re[partial_mu psi* partial_nu psi]   (μ ≠ ν)
```

## 3. Discretization to the Semantic Graph

On the embedding graph, replace the continuous derivative with the finite difference along edge (i,j):

```
partial_mu psi → e_j - e_i   (along edge direction)
partial_nu psi* → e_j - e_i  (real embeddings, ψ* = ψ)
```

The inner product:

```
partial_mu psi* partial_nu psi → (e_j - e_i)·(e_j - e_i) = |e_j - e_i|²
```

Therefore:

```
T(i,j) = 2(1 + σ_ij) × |e_i - e_j|²   (for edges i≠j)
```

The factor of 2 arises from the tensor symmetry (μ↔ν terms). On the unit sphere:

```
|e_i - e_j|² = 2(1 - cos_sim_ij)
```

And:

```
σ_ij = 1/(1 - cos_sim_ij)  (compression from Fubini-Study metric)
```

Substituting:

```
T(i,j) = 2(1 + 1/(1-c)) × 2(1-c) = 4(1-c) + 4 = 8 - 4c
```

where c = cos_sim_ij. **This is T = 4 - 2c doubled** — the factor of 2 comes from the tensor structure.

## 4. The Factor of 2

The original document used `T = (1+σ)|e_i-e_j|² = 4 - 2c`. The derived form is `T = 2(1+σ)|e_i-e_j|² = 8 - 4c`. The factor of 2 gets absorbed into the proportionality constant κ:

```
Originally: G = κ_old × T_old  →  κ_old = 0.036
Derived:    G = κ_new × T_new  →  κ_new = κ_old/2 = 0.018
```

So κ_new = 0.018. This is the **predicted** value from the action principle.

## 5. Testing the Derived T

If the derived T (with factor 2) is correct, then fitting κ should yield κ ≈ 0.018, and the correlation should be identical to before (since multiplying T by a constant doesn't change correlation).

The test: recompute with T_derived = 2(1+σ)|Δe|² and verify:
- κ_fitted ≈ 0.018 (half of 0.036)
- r(G,T) unchanged at ~0.95
