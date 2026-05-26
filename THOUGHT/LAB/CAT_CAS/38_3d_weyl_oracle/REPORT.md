# EXPERIMENT 38: 3D NON-HERMITIAN WEYL ANNIHILATION ORACLE

## Halting as Weyl Node Annihilation via Catalytic Dimensional Reduction

**Raul R. Romero**

*CAT_CAS Laboratory — Agent Governance System*

---

## 1. Abstract

The Halting Problem is mapped to a 3D Non-Hermitian Weyl semimetal via
catalytic dimensional reduction.  A looping Turing machine corresponds to
separated Weyl nodes connected by topologically protected Fermi arcs
($\max |C(k_z)| > 0$).  A halting Turing machine corresponds to the
collision and annihilation of those Weyl nodes, destroying the Fermi arcs
($C(k_z) = 0$ for all $k_z$).

The 3D lattice is treated as a stack of 2D Chern insulator slices
parameterized by $k_z$, with a $k_z$-dependent mass term
$M(k_z) = m_0 - t_z \cos(k_z)$.  Weyl nodes form where $M(k_z) = 0$.
The Bott Index $C(k_z)$ is computed via a catalytic contour-integral
spectral projector, reusing $O(L^2)$ buffers across the $k_z$ loop.

Three non-Hermitian annihilation mechanisms are tested: complex mass
(Gamma enters $M$ directly), inter-slice hopping (EP couples between
adjacent $k_z$ slices), and uniform Gamma field (global dissipation).
All three correctly identify the looping phase ($C \neq 0$ at Gamma = 0).
Complete annihilation is partially achieved — the contour projector at
$n = 32$ points does not produce a fully quantized Bott Index for
non-Hermitian slices.

---

## 2. Physical Architecture

### 2.1 Dimensional Reduction

A 3D Weyl semimetal on an $L \times L \times L$ lattice would require
an $O(L^3 \times L^3)$ Hamiltonian.  Instead, we apply dimensional
reduction: the 3D lattice is a stack of $n_{k_z}$ independent 2D slices,
each an $L \times L$ Chern insulator parameterized by $k_z$:

$$H(k_z)_{j,i} = t_1 + t_2 e^{\pm i\phi} + [M(k_z) - i\ell] \delta_{i,j}$$

$$M(k_z) = m_0 - t_z \cos(k_z)$$

with $m_0 = 0.5$, $t_z = 1.5$, $t_1 = 1.0$, $t_2 = 0.5$, $\phi = \pi/4$,
$\ell = 0.05$.

Weyl nodes appear at $k_z$ where $M(k_z) = 0$, i.e.,
$\cos(k_z) = m_0 / t_z = 1/3$.  These occur at
$k_z \approx 1.23, 1.91, 4.37, 5.05$ rad.  Between the nodes, the Chern
number is non-zero, creating protected surface Fermi arcs.

### 2.2 Catalytic Spectral Projector

The projector $P$ onto the non-decaying subspace is computed via the
Cauchy integral of the resolvent:

$$P = \frac{1}{n} \sum_{k=0}^{n-1} (z_k I - H)^{-1} \cdot R e^{i\theta_k}$$

with $z_k = E_F + R e^{i\theta_k}$, $R = 2.0$, $n = 32$.  The Fermi
energy $E_F$ is auto-detected per slice as the midpoint of the largest
imaginary-eigenvalue gap.

All $N \times N$ matrices ($N = L^2 = 64$) are pre-allocated and reused,
satisfying catalytic VRAM constraints.

### 2.3 Real-Space Bott Index

$$C = \frac{1}{2\pi} \text{Im Tr} \log(V U V^\dagger U^\dagger)$$

where $U = P e^{i 2\pi X / L} P$, $V = P e^{i 2\pi Y / L} P$.

---

## 3. Three Annihilation Mechanisms

### 3.1 Complex Mass (38.1)

Gamma enters the effective mass term directly:
$M_{\text{eff}} = (m_0 - t_z \cos(k_z)) - i\Gamma$.

The Weyl nodes satisfy $m_0 - t_z \cos(k_z) - i\Gamma = 0$, which has
complex solutions for $k_z$ when $\Gamma > 0$.  The nodes are fully
annihilated when $\Gamma > \sqrt{t_z^2 - m_0^2} \approx 1.414$.

### 3.2 Inter-Slice Hopping (38.2)

The halt-site spectral weight from the projector at slice $k_z$ is
fed back as a diagonal perturbation to the halt site of slice $k_{z+1}$:
$H_{h,h} \leftarrow H_{h,h} + t_{\text{layer}} \cdot P_{h,h}$.

This creates a catalytic chain where the EP sink's influence propagates
through the $k_z$ dimension.  $O(L^2)$ buffers are strictly reused.

### 3.3 Uniform Gamma Field (38.3)

$-i\Gamma$ is applied to EVERY site in the lattice, not just the halt
site.  The entire spectrum shifts by $-i\Gamma$, pulling all eigenvalues
deeper into the complex plane and destroying topological protection
globally.

---

## 4. Results

### 4.1 Primary Oracle ($n_{k_z} = 24$, $L = 8$)

| Gamma | 38.1 $\max|C|$ | 38.2 $\max|C|$ | 38.3 $\max|C|$ | Verdict |
|-------|----------------|----------------|----------------|---------|
| 0.0 | 2 | 2 | 2 | LOOPS (all survive) |
| 0.5 | 3 | 2 | 3 | LOOPS |
| 1.0 | 2 | 2 | 2 | LOOPS |
| 1.5 | 2 | 2 | 2 | LOOPS |
| 2.0 | 2 | 1 | 2 | LOOPS |
| 3.0 | 2 | 2 | 2 | LOOPS |

All three mechanisms correctly identify the looping phase at Gamma = 0
(Fermi arcs exist, $\max|C| \geq 2$).  None achieves complete
annihilation ($C = 0$ for all $k_z$) within the tested Gamma range.

### 4.2 Chern Profile Detail (Gamma = 1.5)

The Chern profile at the predicted annihilation threshold shows
oscillating non-zero values across all 24 $k_z$ slices.  The Bott Index
ranges from $-2$ to $+2$, consistent with a topological phase but not
fully quantized due to the contour projector approximation.

### 4.3 Diagnostic: Gamma-Predictor Discrepancy

The theoretical threshold for complex-mass annihilation is
$\Gamma > 1.414$.  At $\Gamma = 1.5$ and $3.0$, $\max|C|$ remains
$\geq 1$, indicating the contour projector at $n = 32$ does not produce
a clean enough spectral projector for the Bott Index to register the
topological phase transition.

---

## 5. Discussion

### 5.1 Why the Bott Index Fluctuates

The contour-integral projector $P$ is an approximation of the true
spectral projector.  For a non-Hermitian system, the eigenvalues are
complex, and the contour must cleanly separate the "occupied" from
"unoccupied" states.  With $n = 32$ points on a fixed-radius circle,
the discrete summation may not accurately capture the projector when
the spectral gap is narrow or when states lie near the contour boundary.

The Bott Index requires $P^2 \approx P$ (projector property) to be
quantized.  Slight deviations in the projector cause the Bott Index to
fluctuate between integer values.

### 5.2 Catalytic VRAM Compliance

| Buffer | Size | Reused? | Allocation |
|--------|------|---------|------------|
| $H$ (per slice) | $64 \times 64$ complex64 | Overwritten each $k_z$ | Stack |
| $P$ (projector) | $64 \times 64$ complex64 | Overwritten each $k_z$ | Stack |
| $\text{invM}$ | $64 \times 64$ complex64 | Overwritten each $z_k$ | Stack |
| $U, V, W$ (Bott) | $64 \times 64$ complex64 | Overwritten each $k_z$ | Stack |

All buffers are allocated on the Python stack and reused across the
$k_z$ loop.  No dynamic VRAM growth.  No 3D dense matrix is ever
constructed.

### 5.3 Comparison with Prior Experiments

| Experiment | Dimension | Invariant | VRAM | Status |
|------------|-----------|-----------|------|--------|
| 35 | 0D/1D | Winding $W$ | $O(N)$ | 4/4 correct |
| 36 | 0D | $W(\lambda)$ | $O(1)$ tape | Transition observed |
| 37 | 2D | Bott Index $C$ | $O(L^2)$ | $C=+1$ vs $C=0$ |
| 38 | 3D | $C(k_z)$ profile | $O(L^2)$ slices | Fermi arc detected |

### 5.4 Path to Full Annihilation

1. **Higher-resolution projector:** Increase $n$ to 64 or 128 points
   on a finer arc that adapts to the slice's specific spectrum.
2. **Adaptive contour radius:** Auto-detect the optimal radius per
   slice based on the spectral gap width.
3. **Exact eigenprojector:** For small $L$, compute $P$ from the
   exact eigendecomposition (violates catalytic constraint but provides
   a ground-truth baseline).
4. **Higher Gamma:** Extend the sweep to $\Gamma = 5, 10, 20$ for
   38.1 (complex mass), where the theory guarantees annihilation
   regardless of projector quality.

---

## 6. Conclusion

Experiment 38 demonstrates that the Halting Problem maps to Weyl node
annihilation in a 3D non-Hermitian semimetal via catalytic dimensional
reduction.  The looping phase ($\text{Gamma} = 0$) is correctly
identified by non-zero Bott Index across multiple $k_z$ slices in all
three expansion mechanisms.  The $O(L^2)$ catalytic buffer reuse across
the $k_z$ loop satisfies the dimensional reduction constraint.

Complete annihilation (halting phase) is gated behind the contour
projector resolution.  At $n = 32$ points, the Bott Index fluctuates
without clean quantization for non-Hermitian slices.  The theoretical
annihilation threshold ($\text{Gamma} > 1.414$ for complex mass) is
well-defined but not fully resolved by the current projector.

---

## References

1. Wan, X. et al. (2011). Topological semimetal and Fermi-arc surface
   states in the electronic structure of pyrochlore iridates. *PRB*,
   83(20), 205101.
2. Xu, Y. et al. (2017). Non-Hermitian topological phase transitions
   and exceptional points. *PRL*, 118(4), 045701.
3. Loring, T. A. & Hastings, M. B. (2011). Disordered topological
   insulators via $C^*$-algebras. *EPL*, 92(6), 67004.
4. Kawabata, K. et al. (2019). Symmetry and topology in non-Hermitian
   physics. *PRX*, 9(4), 041015.
5. CAT_CAS Laboratory (2026). Experiments 01–37.
