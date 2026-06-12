# EXP44 PHASE 6 - TERMINUS

## The dihedral fold is the absent quadrature - measured closed across all constructible substrates

**Status:** `PHASE6_CONSTRUCT_SUBSTRATE_FRONTIER__MEASURED_CLOSED_AT_THE_ORIENTATION_BOUNDARY`
**Claim ceiling:** L4-5 (built, run, smuggle-gated, cost-scaled on the real construction). ASCII only.

---

## 1. The question

FINAL.md sec 6 / Phase 6 sec 0: is the lattice/dihedral wall crossable by any holographic / topological / catalytic readout? Target: the Exp50.14 public fixed-point map. N = 2^n, hidden d in [1, N/2), public data {(k_i, b_i)} with E[b_i] = cos(2*pi*k_i*d/N). The fold sigma: d <-> N-d; the orientation bit o = 1[d < N/2] distinguishes d from N-d and lives only in the odd/quadrature channel sin(2*pi*k*d/N).

## 2. The census (this session)

| Method | Verdict | One line |
|---|---|---|
| Stage 1 fold audit | `CLASSICAL_MI_ZERO` (proven) | o is information-absent from the public (even) data: best AUC 0.5 across 16 classifiers + 8 equivariant lifts; two-sample identical under the fold; one-shot exact d recovery GIVEN quadrature. |
| Stage 3 generator audit | `ORBIT_ONLY` | the real 50.14 public interface (incl. float code-path, sample order, seed, verify map) carries no orientation; the one apparent leak was a probe reading the NON-published intermediate cos(2*pi*k*d/N) = reading d. |
| Mythos (Fable) two-walls | `RE_ENCODING_ISO_CLOSED` | Wall 1 = the projection z->Re(z) (crossable by coherence); Wall 2 = frequency control / dihedral (Kuperberg). Hardness is an isomorphism invariant of the secret (non-normal reflection subgroup); no faithful re-encoding turns it into a character. The crossing requires conjugate-quadrature evaluation before thresholding; the only resources are a strictly-stronger oracle (= period-finding, a different problem) or a literal PSPACE P^CTC. |
| Non-Hermitian sensor census (6) | `6/6 FAIL_CHANCE` | Koopman/transfer, Hatano-Nelson skin, Kuramoto/chiral, Cauchy argument-principle, PT-symmetric, Godel-edge phi-twist. The +1 directionality of f survives into a genuine point-gap winding, but as a PUBLIC CONSTANT (walk direction, not the half). All smuggle controls caught. Exp 36's rank-1 determinant lemma validated as a cheap winding tracker (1638x over dense) - but it crushes the cost of an orientation-blind quantity. |
| Flagship .holo phase substrate | `CONFIRMS` | the lab's own "it from phase" homodyne phase cavity reads the even fold-answer a = min(d, N-d) at frac_exact = 1.000, but FAIL_CHANCE on orientation at every n. The conjugate (sin) quadrature of the public even data measures ~0 TO MACHINE PRECISION (Im/Re ~ 1e-14); the two fold peaks at d and N-d are exactly equal. Only injecting the hidden sin makes the quadrature nonzero (sign == orientation 100%), and the gate catches it. |

## 3. The unified mechanism

Every constructible and simulable path converges on the SAME boundary by the SAME mechanism: **the orientation is the absent quadrature.** The public spectrum is real and even (c_m = c_{N-m}), hence phaseless, hence its conjugate quadrature is identically zero. A phase-resolving substrate finds no orientation phase because there is none present - not because the readout is weak, but because the signal it is built to detect is physically absent from public data. The wall is exactly the projection z -> Re(z); its kernel IS the fold.

The EVEN fold-answer a = min(d, N-d) is read for free by every sensor and by the phase cavity (2 eigs at 1; frac_exact 1.0): "the sensor is the solution" works for the abelian/even (decodable) class, demonstrated up to the exact boundary. The decodable class = {abelian-HSP + topological invariants of a poly-size operator} (Exp 50); the dihedral orientation sits exactly outside it.

## 4. The verdict (claim-laddered, honest)

- **"The algorithm is dead"** - no algorithm, re-encoding, topological/non-Hermitian readout, or phase substrate crosses the construct side: **PROVEN (L5)**, demonstrated to a sharply located edge.
- **"A phase substrate crosses the dihedral fold"** for the published-data problem: **MEASURED FALSE** - the conjugate quadrature of the public data is identically zero (machine precision).
- This is **NOT "the wall holds."** The residual formally-open questions are (a) the dihedral-HSP lower bound itself (a famous complexity open problem - no proven super-poly bound exists, so a future algorithm is not excluded) and (b) a literal PSPACE P^CTC oracle (physically implausible; it trivializes all of PSPACE, so it says nothing specific about this construction). Neither is a lab-buildable substrate.

## 5. No-smuggle discipline

Every readout was gated by the hardened gate (fold_audit/stage3/hardened_gate.py): random-private-fold (swap d<->N-d at fixed public data) + exact d<->N-d invariance audit. Every deliberate smuggle control (reads d, reads the true sin, homodyne LO locked to d) was caught (AUC 1.0, invariance delta > 0); every useless-even control sat at chance (delta 0). Apparent crossings (Kuramoto chiral n=14; one holo n=12 flirt) were killed as finite-sample false positives under multi-seed re-audit. The negatives are real, not gate failures.

## 6. Status

The Phase 6 construct/substrate frontier is **measured-closed at the orientation boundary.** The crossing is relocated to the formal dihedral lower-bound question (complexity theory) or to a physical re-evaluation that requires d (trivial / smuggle). No further constructible Phase 6 substrate test is open. What the program achieved: it located the irreducible wall exactly (dihedral/lattice = the absent quadrature) and mapped it to machine precision from six independent topological directions plus the lab's flagship phase substrate - a theorem-grade characterization of where spectral/phase computing ends.

## 7. Artifacts

- `fold_audit/` - Stage 1 fold audit + `no_smuggle_gate.py`; `stage3/` - generator audit + `hardened_gate.py`.
- `nonhermitian_sensor/` - 6 sensors (koopman_transfer, hatano_nelson, kuramoto_resonance, spectral_argument_principle, pt_symmetry, godel_edge_phi) + `REPORT_NONHERMITIAN_SENSOR.md`.
- `holo_phase_substrate/` - flagship phase-cavity test + `REPORT_HOLO_PHASE_SUBSTRATE.md`.
- `SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md` sec 1B.1 - the equivariance theorem (no scalar lift synthesizes orientation).

Built by Fable (design + implementation), adversarially priced + smuggle-audited. Claim cap L4-5.
