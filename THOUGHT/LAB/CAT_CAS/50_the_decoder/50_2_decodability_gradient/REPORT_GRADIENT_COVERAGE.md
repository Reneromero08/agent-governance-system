# Exp 50.2e - Generalizing the gradient: more groups, readout hierarchy, independent phi

**Verdict:** `WALL_IS_NON_NORMAL_SUBGROUPS` (all 5 gates pass, entry exits 0).
**Claim level:** 4-5 (bounded group sample + matched nulls; refines, does not replace, 50.2).

## What this sharpens

50.2 located the collapse at the abelian -> non-abelian boundary using d_max / abelianness as
the x-axis. 50.2b/2c then showed the non-abelian Fourier reframe recovers *normal* hidden
subgroups and bottoms out only at the *non-normal* (strong-sampling / lattice) case. This
brick tests that refinement head-on with new group families, a second readout channel, and an
independent re-implementation of the order parameter.

## Results (extended ladder, n_inst=30, matched random + shuffle nulls)

| group | \|G\| | normal-H | d_max | D_char | D_fft | shuffle |
|---|---|---|---|---|---|---|
| Z_16/32/64 | 16-64 | True | 1 | 1.000 | 1.000 | ~0.05 |
| D_8/16/32 | 16-64 | False | 2 | 0.15 -> 0.02 | ~0 | ~0.03 |
| **Q_8** | 8 | **True** | 2 | **1.000** | **0.103** | 0.11 |
| AGL(1,5) | 20 | False | 4 | 0.112 | ~0 | 0.06 |
| A_5 | 12 | False | 5 | 0.363 | 0.20 | 0.19 |
| S_4/S_5/S_6 | 24-720 | False | 3-16 | 0.03 -> 0.001 | ~0 | ~0.01 |

normal-H (decodable) mean D_char = **1.000**; non-normal-H (collapsed) mean D_char =
**0.093**; Cohen d = **8.98**.

## The three findings

**(A) The wall is non-NORMAL subgroups, not non-abelianness.** Q_8 (quaternion) is
non-abelian but Hamiltonian - every subgroup is normal. It is **decodable** (D_char = 1.000),
sitting on the shelf with the abelian groups, while dihedral / AGL(1,5) / A_5 / S_n (all with
a non-normal hidden subgroup) collapse. Decodability tracks H-normality with Cohen d = 9.0.
This is the decisive refinement: the structural variable is normality of the hidden subgroup,
exactly the abelian-HSP + normal-subgroup class of 50.2b.

**(B) Readout hierarchy (the readouts do NOT all agree - and that is the point).** The scalar
FFT readout recovers only the abelian shelf (D_fft = 1.000 on Z_n) and **misses Q_8**
(D_fft = 0.103) - even though Q_8 is decodable. Only the character/quotient readout (the non-
abelian reframe) recovers Q_8 (D_char = 1.000). So the scalar spectral readout is bounded by
the weaker ABELIAN wall; crossing to normal non-abelian subgroups *requires* the reframe
(50.2b). This is the readout hierarchy made measurable: scalar-FFT (abelian) subset
character-reframe (normal subgroups) subset [non-normal wall = lattice].

**(C) The order parameter is implementation-robust.** phi_character was re-implemented a
second, independent way - as an explicit orthogonal-projector matrix onto K-coset-constant
functions - and agrees with the hsp_family loop implementation to **2.22e-16** (machine
epsilon) across every group. The collapse is not an artifact of one code path.

## Gates

| Gate | Result | Detail |
|---|---|---|
| G1 normality split (normal decodable, non-normal collapsed) | PASS | dec=1.000 col=0.093 d=8.98 |
| G2 Q_8 decisive: non-abelian + normal-H => decodable | PASS | Q_8 D_char=1.000 |
| G3 independent phi cross-check (loop == projector) | PASS | max disagreement 2.22e-16 |
| G4 readout hierarchy (scalar FFT abelian-only; reframe gets Q_8) | PASS | Q_8 D_fft=0.103 vs D_char=1.000 |
| G5 label-shuffle null floor | PASS | max \|shuffle D\| = 0.19 |

## Honest negative recorded

A MUSIC / super-resolution readout was attempted as a third channel and **does not apply**:
the coset grating has \|G\|/2 frequency components, not a sparse line spectrum, so MUSIC's
few-sources assumption is violated and it returns a degenerate constant. Rather than force a
ceremonial gate, this is recorded as an honest negative. The FFT channel (a genuine,
different, working scalar readout) carries the readout-hierarchy result.

## Reproduce

```
python THOUGHT/LAB/CAT_CAS/50_the_decoder/50_2_decodability_gradient/50_2e_gradient_coverage.py
```
Writes `coverage_result.json` + `output_coverage.txt`. Exits 0 iff all 5 gates pass.
