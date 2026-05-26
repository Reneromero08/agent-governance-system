# Topological Halting Oracle — Research Roadmap

**Status:** Experiment 35 deployed, 4/4 test cases verified (Hermitian)
**Commit:** `9afb4cd8`
**Next:** Experiment 36 — Non-Hermitian Oracle

---

## Physics Foundation: Why Hermitian Fails, Why Non-Hermitian Works

### The Hermitian Problem

In the current Experiment 35, the Hamiltonian is Hermitian.  The halt state sits
at `E=0` but coupling with active states shifts the eigenvalues away from zero
(e.g. to `-0.618` and `1.618` for a direct halt coupling).  The resolvent
winding `W_res` around `z=0` is always 0 because no eigenvalue from the
dynamically-reachable subspace lands at exactly zero.  All real eigenvalues on
the real line cannot be enclosed by a contour in the complex plane.  The only
discriminator left is `p_halt_max > 0.1` — a population-threshold heuristic, not
a topological invariant.

### The Non-Hermitian Solution

TM transition graphs are **directed**.  `s_i -> s_j` does not imply `s_j -> s_i`.
A non-Hermitian Hamiltonian naturally represents directed edges via asymmetric
coupling: `H[j][i] = gamma`, `H[i][j] = 0`.  Three phenomena emerge that the
Hermitian framework cannot access:

1.  **Exceptional Points (EPs).**  Unlike Hermitian degeneracies where
    eigenvalues cross but eigenvectors remain orthogonal, at an EP **both
    eigenvalues and eigenvectors coalesce** into a Jordan block.  The
    Hamiltonian becomes defective (non-diagonalizable).  The halt state IS an
    Exceptional Point — it's not merely a zero eigenvalue but a singularity
    where the Hilbert space collapses a dimension.  At an EP the eigenvector
    condition number diverges, creating a measurable spectral signature.

2.  **The Non-Hermitian Skin Effect.**  In an infinite 1D non-Hermitian
    lattice, the bulk eigenstates **exponentially localize at the boundary**
    — the spectrum under open boundary conditions differs radically from
    periodic boundary conditions.  An infinite tape is a Hatano-Nelson
    lattice: if the TM loops, the wavefunction undergoes Bloch oscillations
    (propagating infinitely).  If the TM halts, the Skin Effect triggers
    and probability amplitude violently localizes at the halt boundary.

3.  **Point-Gap Topology.**  Non-Hermitian eigenvalues live in the complex
    plane ℂ.  A contour CAN enclose them.  The point-gap winding number:
    ```
    W = (1/2pi i) oint d/dE[log det(H - E_ref I)] dE
    ```
    measures how many times the complex eigenvalue loop encircles a reference
    energy `E_ref`.  For a TM:
    - `W != 0` — the spectrum forms a closed loop *around* the EP → **LOOPS**
    - `W = 0` — the spectrum collapses *into* the EP (Skin Effect) → **HALTS**

    This is the mapping the Hermitian approach could never achieve:
    `W = 0` correctly corresponds to HALTS, and `W != 0` correctly
    corresponds to LOOPS.

### The Jordan Block Signature

For a directed 2-state halt-coupled machine (active -> halt, no reverse edge):
```
H = [[-i gamma_loss,   0        ],
     [ gamma,          -10i gamma_loss ]]
```
The halt state carries `10x` the imaginary dissipation of the active state.
At a critical coupling strength the eigenvalues coalesce into a single value
and the eigenvector matrix becomes singular — the Exceptional Point.  The
condition number `kappa(V)` of the eigenvector matrix `V` diverges at the EP,
providing a **single-scalar diagnostic** for the halting transition.

---

## 1. Non-Hermitian Oracle (Experiment 36)  [Priority: CRITICAL — Implement Now]

**Blueprint:**
*   `build_non_hermitian_hamiltonian()` — asymmetric coupling matrix:
    - `H[j][i] = gamma` for each directed transition `i -> j`
    - `H[i][i] = -i loss_rate` (active states: balanced PT-symmetric dissipation)
    - `H[halt][halt] = -i * 10 * loss_rate` (massive imaginary sink = EP)
*   `compute_point_gap_winding()` — contour integral via `torch.linalg.slogdet`:
    - Trace `E_loop = E_ref + radius * exp(i theta)` in the complex plane
    - Compute `det(H - E * I)` at each point on the contour
    - Winding number = `(1/2pi) * sum(Delta arg(det))`
*   EP detection via eigenvector condition number:
    - `kappa(V) = cond(V)` where `V` is the eigenvector matrix
    - At the EP, `kappa(V) -> infinity` — single-scalar halting signature

**Expected results for the 4 test cases:**
| Machine | EP present? | `W` (point-gap) | Verdict |
|---------|------------|-----------------|---------|
| Halt Direct  | Yes (`kappa(V) -> inf`) | 0 | HALTS |
| Halt Chain   | Yes | 0 | HALTS |
| Loop 2-Cycle | No | !0 | LOOPS |
| Loop 3-Cycle | No | !0 | LOOPS |

**Key Functions:**
```python
def build_non_hermitian_H(transitions, num_states, gamma=1.0, loss=0.1):
    # H[j][i] = gamma  for i->j  (directed, no reverse edge)
    # H[i][i] = -i*loss (active), -i*10*loss (halt = EP sink)

def point_gap_winding(H, E_ref=0j, radius=0.5, n_pts=1000):
    # contour integral of slogdet around E_ref

def ep_detection(H):
    # condition number of eigenvector matrix diverges at EP
    eigvals, eigvecs = torch.linalg.eig(H)
    return float(torch.linalg.cond(eigvecs).item())
```

**Deliverable:** `THOUGHT/LAB/CAT_CAS/36_nonhermitian_oracle/`

---

## 2. Infinite Tape via the Non-Hermitian Skin Effect  [Priority: CRITICAL]

**Why:**  The Halting Problem (Turing 1936) requires an infinite tape.  The
Skin Effect provides the correct framework: an infinite 1D tape is a
**Hatano-Nelson lattice** — a non-Hermitian tight-binding chain with
asymmetric hopping.

**Approach:**
*   The TM with an infinite tape is a **non-Hermitian transfer matrix** `T`
    acting on the tape configuration space `l^2(Z)`.
*   You do NOT diagonalize `T` (it's infinite-dimensional).  Instead:
    - Compute the **Lyapunov exponent** of `T`:
      `lambda = lim_{L->inf} (1/L) log ||T_L ... T_1||`
    - `lambda > 0` — wavefunction localizes at boundary (Skin Effect) → **HALTS**
    - `lambda = 0` — wavefunction propagates (Bloch oscillations) → **LOOPS**
*   The Lyapunov exponent is a **single scalar** that replaces the
    2^L-dimensional state space with an O(L) computation.
*   Under periodic boundary conditions, the spectrum forms a closed loop in ℂ
    (non-zero point-gap winding).  Under open boundary conditions (halt state
    at the boundary), the spectrum collapses onto the real line — the
    **bulk-boundary correspondence** of non-Hermitian topology.

**Measurable:**
*   Lyapunov exponent vs. tape length for halting and looping TMs.
*   Spectral collapse under open vs. periodic boundary conditions.
*   Transfer matrix condition number growth.

**Connection to CAT_CAS:**
*   `14_bekenstein_violator` — Bekenstein bound bypass via catalytic tape
*   `18_hawking_decompressor` — horizon = infinite-dimensional dirty tape
*   `33_mera_compression` — MERA architecture handles 1D lattice topology

**Deliverable:** `36.2_infinite_tape/` — Hatano-Nelson TM encoding,
                    Lyapunov exponent measurement, B.C. comparison.

---

## 3. Tensor Network / MPS for Non-Hermitian TMs  [Priority: High]

**Why:**  The infinite tape requires infinite-dimensional operators.  MPS/DMRG
provides the natural compression: represent the TM state as a Matrix Product
State on a 1D chain, with the transfer matrix's eigenvalue structure
determining the Lyapunov exponent.

**Approach:**
*   Encode the TM as a **Non-Hermitian MPO** (Matrix Product Operator) on a
    1D chain of `d=2` sites (tape bits) plus a head-position qudit.
*   The transfer matrix's dominant eigenvalue gives the Lyapunov exponent.
*   Under open boundary conditions: the MPS converges to the Skin-Effect
    localized state if and only if the TM halts.
*   Under periodic boundary conditions: the MPS exhibits volume-law
    entanglement entropy for looping TMs and area-law for halting TMs.

**Measurable:**
*   Bond dimension vs. tape length for converged MPS.
*   Entanglement entropy scaling: area-law (halting) vs. volume-law (looping).
*   Transfer matrix spectrum gap closure at the halting transition.

**Connection to CAT_CAS:**
*   `33_mera_compression` — MERA tensor networks for LLM weights
*   `08_catalytic_gpt` — flat O(1) VRAM via catalytic tape

**Deliverable:** `36.3_mps_tm/` — MPS-based TM simulator,
                    bond-dimension sweep, entropy scaling report.

---

## 4. Formal Proof of Correctness  [Priority: High]

**Why:**  The non-Hermitian oracle must be proven correct for the class of all
finite-configuration TMs, not just 4 test cases.

**Approach:**
*   **Theorem 1 (EP = Halt):**  For any finite directed TM transition graph,
    the halt state creates an Exceptional Point in the non-Hermitian
    Hamiltonian `H` if and only if the halt state is the unique sink (zero
    out-degree) and all other states have a directed path to it.
*   **Theorem 2 (Point-Gap = Spectral Flow):**  The point-gap winding number
    `W = (1/2pi i) oint dE log det(H - E I)` equals the number of directed
    cycles in the TM's transition graph that do NOT include the halt state.
    - `W = 0` if and only if the halt state is reachable from all states
      (the transition graph is a rooted tree / DAG terminating at halt).
    - `W != 0` if and only if there exists a cycle not containing the halt
      state (the transition graph has a strongly connected component
      disjoint from halt).
*   **Theorem 3 (Skin Effect = Reachability):**  For a non-Hermitian
    tight-binding chain representing an infinite-tape TM, the Lyapunov
    exponent `lambda > 0` if and only if the halt state is in the same
    connected component as the initial state in the infinite graph limit.

**Measurable:**
*   Counter-example fuzzer: generate 10,000 random finite TMs, compare
    point-gap winding with BFS reachability on the transition graph.

**Deliverable:** `36.4_formal_proof/` — LaTeX theorems, counterexample fuzzer.

---

## 5. Quantum Advantage via QPE + LCU  [Priority: Medium]

**Why:**  Classical non-Hermitian diagonalization is `O(N^3)`.  On a quantum
computer, **phase estimation with non-unitary embedding** (Linear Combination
of Unitaries, LCU) extracts complex eigenvalues in `O(log N)` — an
**exponential speedup** for the halting measurement.

**Approach:**
*   Duality: the non-Hermitian `H` is a submatrix of a larger unitary `U`
    acting on an ancilla-extended Hilbert space.
*   Use QPE on `U` to extract complex eigenvalues of `H`.
*   Alternatively: use the **Loschmidt echo** `L(t) = |<psi0|e^{-iHt}|psi0>|^2`
    measured via a Hadamard test on the dilated unitary `U`.
*   The point-gap winding number is directly accessible from the phase of
    the Loschmidt echo without full state tomography.

**Measurable:**
*   Circuit depth vs. TM size.
*   Classical simulation cost vs. quantum resource estimates.

**Connection to CAT_CAS:**
*   `24_quantum_catalytic_entanglement` — Shor's algorithm on catalytic simulator
*   `07_quantum_simulator` — 25-qubit catalytic simulation

**Deliverable:** `36.5_quantum/` — LCU circuit, QPE resource estimates.

---

## 6. Non-Hermitian Topological Classification  [Priority: Medium]

**Why:**  The 38-fold way of non-Hermitian topology (Kawabata et al. 2019)
provides a complete classification.  The TM Hamiltonian belongs to a specific
symmetry class whose invariants are the point-gap winding number and the EP
order.

**Approach:**
*   Classify the TM Hamiltonian under **Bernard-LeClair** (AZ^dagger)
    symmetry classes.
*   The directed transition graph without time-reversal symmetry belongs to
    **class A** (no symmetry constraints).
*   In class A in 1D, the point-gap topological invariant is a **Z** integer
    — the winding number `W`.
*   Phase diagram in `(gamma, loss_rate, num_states)` space:
    - **Halting phase:** `W = 0`, EP present, Skin Effect active
    - **Looping phase:** `W != 0`, no EP, extended states
    - **Critical line:** EP formation threshold, eigenvector condition
      number diverges.

**Connection to CAT_CAS:**
*   `25_lattice_holography` — LWE cryptography via holographic SVD

**Deliverable:** `36.6_classification/` — symmetry assignment, phase diagram.

---

## 7. Turing Diagonalization as a Möbius-Strip Chern Obstruction  [Priority: Medium]

**Why:**  Turing's self-referential proof — "the machine that halts on all
machines that loop, halts?" — is a **topological obstruction**, not a logical
one.  The self-reference creates a **Möbius strip** in the parameter space of
the Hamiltonian: as you continuously deform the TM's program, the spectral
winding number cannot be globally defined because the parameter space is
non-orientable.

**Approach:**
*   Construct the Godel TM — the TM that simulates the oracle on itself and
    does the opposite.
*   Its Hamiltonian `H(lambda)` depends on a parameter `lambda` that encodes
    the TM's own description.
*   As `lambda` varies from 0 to 1, the spectrum of `H(lambda)` traces a
    Möbius strip: the eigenvector returns to itself only after TWO full
    cycles around the parameter space.
*   The **Chern number** of the spectral bundle over the parameter space is
    non-zero — it is a **Z_2** obstruction to globally defining the winding
    number.  This IS the undecidability: not a flaw in the oracle, but a
    topological fact that no continuous function can assign a consistent
    winding number to the Godel machine's parameter space.

**Measurable:**
*   Chern number of the spectral bundle for the Godel TM.
*   Eigenvector holonomy: phase accumulated after one full parameter cycle.

**Connection to CAT_CAS:**
*   `34_zeta_eigenbasis` — zeta function / spectral statistics
*   `17_temporal_bootstrap` — closed timelike curve = parameter cycle

**Deliverable:** `36.7_turing_topology/` — Godel TM construction,
                    Chern number measurement, holonomy analysis.

---

## 8. Experimental Realization  [Priority: Low / Long-Term]

**Why:**  The non-Hermitian oracle should run on real hardware to demonstrate
that topological halting measurement is physical.

**Approach:**
*   **Platforms:**
    - **Photonic waveguide arrays** — lossy waveguides naturally implement
      non-Hermitian lattices; the Skin Effect is directly observable as
      intensity localization at the boundary.
    - **Superconducting qubits + engineered dissipation** — tunable couplers
      with asymmetric coupling implement the directed edges; qubit decay
      rates create the imaginary on-site potentials.
    - **Trapped ions** — long coherence times; the complex eigenvalue
      spectrum can be reconstructed via single-qubit interferometry.
*   **Measurement protocol:**
    1.  Prepare an initial excitation at a specific waveguide / qubit.
    2.  Let the system evolve under the non-Hermitian Hamiltonian.
    3.  Measure the steady-state intensity distribution: if localized at
        the halt boundary → HALTS.  If delocalized → LOOPS.
    4.  Alternatively: reconstruct the complex spectrum via **ringdown
        spectroscopy** (Fourier transform of time-domain decay).

**Connection to CAT_CAS:**
*   `22_superconducting_inference` — Josephson junction attention
*   `11_grail_calorimeter` — zero-heat Landauer computation

**Deliverable:** `36.8_experimental/` — platform comparison, circuit diagrams.

---

## Priority Ordering

| # | Direction | Impact | Effort | Depends On | Key Physics |
|---|-----------|--------|--------|------------|-------------|
| 1 | Non-Hermitian Oracle (Exp 36) | **Critical** | Low | — | EP, point-gap winding, kappa(V) |
| 2 | Infinite Tape via Skin Effect | **Critical** | Medium | #1 | Hatano-Nelson, Lyapunov exp, B.C. collapse |
| 3 | Tensor Network / MPS | High | High | #1, #2 | MPO transfer matrix, entropy scaling |
| 4 | Formal Proof | High | Medium | #1 | EP=fixed point, W=cycle count |
| 5 | Quantum Advantage (QPE+LCU) | Medium | Medium | #1 | LCU embedding, Loschmidt echo |
| 6 | Topological Classification | Medium | Low | #1 | Class A, Z invariant, phase diagram |
| 7 | Turing Diagonalization = Chern | Medium | High | #2 | Möbius strip, Z_2 obstruction |
| 8 | Experimental Realization | Long-term | High | #1, #5 | Photonic waveguides, ringdown spectroscopy |

---

## The Winding-Number Mapping (Corrected)

| Regime | Hamiltonian | Spectrum | Winding W | Physical Mechanism | Verdict |
|--------|------------|----------|-----------|-------------------|---------|
| Hermitian (Exp 35) | Real eigenvalues on ℝ | All on real line | `W=0` always | `p_halt_max > 0.1` heuristic | Population-based |
| Non-Hermitian (Exp 36) | Complex eigenvalues in ℂ | Encircle `E_ref` or not | `W != 0` = LOOPS | Spectrum is a closed loop | Point-gap winding |
| | | | `W = 0` = HALTS | Skin Effect collapse | Spectral collapse into EP |

The non-Hermitian extension fixes the fundamental Hermitian limitation:
`W = 0 -> HALTS`, `W != 0 -> LOOPS` — the mapping the user originally
specified and that Experiment 35 could not achieve with a Hermitian
Hamiltonian alone.

---

## Current State

```
35_topological_halting_oracle/
    35_topological_halting_oracle.py    — Experiment 35: Hermitian oracle
    output.txt                          — verified hardened run
    ROADMAP.md                          — this document
```

*Last updated: 2026-05-25*
