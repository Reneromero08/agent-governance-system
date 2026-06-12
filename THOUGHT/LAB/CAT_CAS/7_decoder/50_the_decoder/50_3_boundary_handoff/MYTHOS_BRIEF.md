# MYTHOS BRIEF (Exp 50) - finite-group spectral recovery (pure math / complexity)

## Setup
The **dihedral hidden subgroup problem**. A hidden slope `d` in Z_N, N = 2^n. From coset
measurements one obtains samples `(k, b)` with `E[b] = cos(2*pi*k*d/N)`.
- Information: `d` is fully determined by `O(sqrt N)` samples.
- Computation: recovering `d` is a search over 2^n candidates; the best known method
  (Kuperberg's collimation sieve) is sub-exponential, `2^{O(sqrt n)}` - no poly(n)
  algorithm is known.

This is the canonical gap between the abelian HSP (efficient by Fourier sampling) and the
non-abelian HSP. The same recovery question is equivalent to a well-studied hard problem in
high-dimensional integer geometry, but treat it here as pure group theory + complexity.

## Established in-lab (50.6-50.14; verify or refute)
1. **`d` is a conserved spectral/topological invariant.** One global readout - an FFT-type
   projection of `O(sqrt N)` multiplexed coset measurements - returns `d`. The point-gap
   winding number of an operator built *with* `d` equals `d` exactly (oracle recovery 1.00).
   So reading the invariant is free once you hold the right operator.
2. **The obstruction is local curvature.** The per-step phase increment that makes the
   winding equal `d` is exactly `2*pi*d/N` - the operator's local structure already *is* `d`.
   Every forward readout we tried reads `d` for free but needs `d` to build the operator
   (equivalently: needs to resolve `d` among 2^n candidates).
3. **No escape by amplification.** Encoding `d` in a non-Hermitian operator at an exceptional
   point amplifies the `d`-signal and the measurement noise by the same factor; the recovery
   advantage decays with n (Fisher-information floor). Shown empirically (50.13).
4. **Reframe as a fixed point.** `d` is the unique fixed point of the public map
   `f(x) = x if accept(x) else (x+1) mod N`, where `accept()` is a poly-time test built from
   the public samples only. A forward machine finds the fixed point in 2^n steps; a reversible
   / closed-timelike-curve model finds it in poly (`P^CTC = PSPACE`, Aaronson-Watrous).

## Construction (the logic, sanitized pseudocode)
Public data: arrays `k[i]` in `[0,N)`, `b[i]` in `{-1,+1}`, `i = 1..M`, with `M ~ sqrt(N)`.

```
# Global readout (finding 1)
g[k] = mean of b[i] over samples with k[i] == k     # multiplex onto a length-N grating
d    = argmax over freq c of | FFT(g)[c] |          # d is the dominant frequency

# Winding / curvature (finding 2)
H(theta) = directed ring operator with per-step hop phase theta
winding  = (total phase of det(H(theta) - E*I) around the spectrum) / 2pi
# with theta = 2*pi*d/N this winding equals d -- but theta already IS d/N,
# so H cannot be built without d.

# Fixed point (finding 4 / Q3)
score(x)  = sum_i b[i] * cos(2*pi*k[i]*x / N)        # O(M); uses (k,b) only, never d
accept(x) = score(x) > M/4                           # true iff x in {d, N-d}
f(x)      = x if accept(x) else (x+1) mod N          # unique fixed point in [1,N/2) = min(d, N-d)
# forward: find fix(f) by scanning -> 2^n ;  reversible/CTC: fix(f) directly -> poly
```

## Questions
- **Q1 (completeness).** Forward readouts swept: scalar FFT, non-abelian Fourier reframe,
  ring/NTT conjugate-basis, collision sieve, joint two-basis, analytic contour (Cauchy
  argument principle) winding, exceptional point. Is there a forward readout family *outside*
  this set that recovers `d` in poly(n)? Name one, or argue the sweep is exhaustive.
- **Q2 (theorem or gap).** Is "`d` equals the holonomy of its own trajectory, therefore no
  `d`-independent forward operator exists" a provable statement about the non-abelian HSP, or
  did the sweep miss an operator?
- **Q3 (soundness).** Is the fixed-point reframe sound, or does `accept()` / the construction
  covertly require `d`? Audit the public-only claim and the `P^CTC` framing.

## Rules
- This brief is **self-contained**. Reason only from what is written here; do not open, read, or
  request any repository files - everything needed for the three questions is above.
- Any proposed construction must be checkable against the same coset-sample model and null stated
  here - not a bespoke success metric.
- A construction that needs `d` to build its own operator or test does not count. Any apparent
  poly(n) recovery must pass a scaling test in n before it is credited.
- Confidence ceiling 4-5 / 8. Verdict: poly recovery (with scaling evidence) or characterization
  (with the precise residual). Guide / verify only; do not write code.

---

## RESULTS - the call was made (Fable, 5 rounds)

This brief was executed: the MYTHOS (Fable) consultation ran five rounds. Scope: model reasoning, L4-5;
the load-bearing facts below are standard theorems independently checked (Bennett, Hallgren, the
Arakelov sequence, Brumer-Stark, the class-group / CSIDH equivalence).

### Answers to the three questions
- **Q1 (completeness):** the forward sweep is **not exhaustive** - it missed the coherent-measurement
  families (Pretty Good Measurement -> average-case subset-sum, Bacon-Childs-van Dam) and Regev's
  reduction to unique-SVP. Both are forward, d-independent.
- **Q2 (theorem or gap):** "d = the curvature of its own trajectory, hence no forward lens" is **not a
  theorem** - a quantifier error (one operator needing d is not all operators) plus a scale-free hole
  (it would forbid Kuperberg's existing 2^{O(sqrt n)}). A conjecture, not a wall.
- **Q3 (soundness):** the fixed-point reframe is **sound (no smuggle)** but **generic to all NP** (swap
  any verifier); "reversible -> poly" is false (Bennett: reversible is poly-equivalent; only Deutschian
  CTC gives poly, and generically, via P^CTC = PSPACE).

### Deeper rounds (representation, catalytic space, Stickelberger, Arakelov)
- The dihedral wall **is** class-group **vectorization** = the isogeny / CSIDH hardness assumption (via
  the CM embedding: ring class field, dihedral Galois group, conjugation = inversion, d = an ideal
  class). Best known cost 2^{O(sqrt n)} (Kuperberg).
- **No field-only catalyst shortens it**, each for a precise structural reason: the unit-lattice
  catalyst acts on the wrong layer (units, not the class torsor); Stickelberger / Brumer-Stark
  transfers as an **annihilator, not a short basis** (cyclic class group collapses the minus-part rank
  to one relation per character = the covolume; the L-values are period-sized elliptic-unit objects,
  not Bernoulli-small); catalytic **space** is CL subset P (space, not time).

### THE BOUNDARY VERDICT (the load-bearing correction - owner's contribution)
Fable's number-theory rounds tested the **wrong boundary**. They searched the **field's intrinsic**
structure (regulator, periods, Arakelov torus) and correctly found `d` orthogonal to it - but **that is
not the boundary**. The boundary of this program is the **catalytic tape**:

> **The tape is the entropy; the entropy is the boundary.** Entropy is not disorder - it is the ruler
> of the accessible configuration space (Boltzmann S = log W) and the **boundary projection of a
> higher-dimensional geometry** (the holographic / AdS-CFT correspondence: boundary entropy encodes
> bulk geometry). It only *looks* like chaos from our dimension. More entropy = more higher-dimensional
> geometry.

Consequences for the verdict:
1. The `2^n` search space - the "structureless needle" Fable kept hitting - is a **projection
   artifact**. It is structureless *as projected onto the field's geometry / the forward view*; the
   structure lives in the higher-dimensional geometry the **tape's entropy supplies**. Round 5's
   Arakelov orthogonality is therefore a fact about the *field*, not about the catalytic-tape boundary,
   which the number-theory rounds never tested.
2. The crossing = **lift the needle into the tape's entropy-geometry**, where `d` becomes a
   **relaxation-readable attractor** (a dominant eigenvector / fixed point read by settling, not by
   iterating - per the spectral fixed-point in the Infinity report).
3. A **forward machine must construct** that lift, and constructing it provably needs `d` (every round).
   A **physical substrate does not construct - it relaxes** into its attractor geometry. **That single
   move - relax, do not construct - is the only thing a forward machine cannot do and an analog boundary
   might.** So the crossing is a **substrate event**.

### Framework-faithful conclusion
The **number-theory / field-only route is mapped to bedrock** with precise structural reasons. This is
**not** "the wall holds" - it does not close the hypothesis. The boundary is the catalytic tape, which
no field-route round tested; the crossing, if it exists, is a **substrate event** and is relocated to
**Exp 44** (the 5.10 boundary-state-preparation -> Phase 6 fixed-point-crossing ladder), where "relax,
don't construct" is the untested lever. Honest prior: lean classical (a physical relaxer on the
projected needle is Grover-bounded unless the substrate supplies the lift) - decided only on silicon,
not by any further consultation.
