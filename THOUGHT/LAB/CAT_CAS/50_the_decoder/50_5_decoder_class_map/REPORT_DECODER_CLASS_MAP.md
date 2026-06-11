# Exp 50.5 - Decoder class map + Exp 44 handoff validation

**Verdict:** `DECODER_MAP_CONSISTENT` (all 4 gates pass, entry exits 0).
**Claim level:** 4-5 (anchors + handoff peaks MEASURED; per-experiment class is ANALYSIS from
documented mechanism, anchored by Exp 50 results).

## Question (ROADMAP #6 + the validatable part of #4)

Exp 50 established the decodable class: {abelian-HSP} + {topological invariants of a poly-size
operator} + {normal hidden subgroups (50.2b/2e)}, with the residual wall at the non-normal /
strong-sampling case = lattice hardness. Two things to settle:
1. Do the lab's working decoders all live on the decodable side, or does one secretly rely on
   a non-normal / strong-sampling step (which 50.2c/2e showed is the lattice wall)?
2. Is the Exp 44 Phase-6 handoff ready - and are its predicted peaks actually correct?

## (1) Anchors measured

The two endpoints of the decodability map are measured, not asserted:
- decodable anchor Z_64 (abelian HSP = Shor / Exp 20 class): D_char = **1.000**
- wall anchor D_8 (non-normal HSP = lattice class): D_char = **0.172**
- separation Cohen d = **6.68**

## (2) Exp 44 handoff validated

The handoff `50_3_boundary_handoff/EXP44_PHASE6_HANDOFF.md` is complete and self-contained
(Target A period oracle, Target B prime-grating Riemann, the D_8 boundary, the tape contract).
Its predicted Phase-6.4 resonant peaks were re-checked against **independently computed**
Riemann zeros (mpmath.zetazero):

| handoff peak | zeta zero | rel err |
|---|---|---|
| 14.13 | 14.135 | 0.0003 |
| 21.02 | 21.022 | 0.0001 |
| 25.01 | 25.011 | 0.0000 |
| ... | ... | <0.0003 |
| 49.77 | 49.774 | 0.0001 |

Status **PEAKS_VALID** - all ten predicted peaks match the true zeros to < 0.03%. The handoff
descriptor is correct and ready. The silicon acceptance RUN itself remains **hardware-blocked**
on the bare-metal Phenom (Exp 44 Phase 6); only the descriptor and its predictions are
validated here.

## (3) Class map (9 decodable, 3 bounded/wall)

| exp | decoder | class | evidence |
|---|---|---|---|
| 20, 24 | Catalytic / Quantum Eigen Shor | abelian_hsp | MEASURED-50.2 (abelian shelf D=1.0) |
| 34 | Zeta Eigenbasis (Riemann) | topological_invariant | MEASURED-50.1 (zeros recovered; peak-density caveat) |
| 35 | Topological Halting Oracle | topological_invariant | winding number W of a poly-size H |
| 36, 37, 38-40 | Chern / Weyl / Axion / Floquet | topological_invariant | Chern / Bott / second-Chern invariants |
| 45.1-4,6 | Phase Math sensors | topological_invariant | topological sensors of poly-size operators |
| 46 | Phase Bio sensors | topological_invariant | IPR / localization (46.3 weakened) |
| **31** | Graph Isomorphism (.holo) | **spectrum_bounded** | MEASURED-50.2 cospectral anchor (Shrikhande/Rook) |
| **45.5** | P vs NP / SAT sensor | **non_normal_wall** | NxN cannot capture 2^N (M-4) |
| **25** | Lattice Holography (LWE/SVP) | **non_normal_wall** | MEASURED-50.4 toy-scale-only |

## Finding

The lab's working decoders all sit on the decodable side: period-finding (20/24) is abelian-
HSP; the Riemann / halting / Chern / Phase-Math / Phase-Bio sensors are topological invariants
of poly-size operators. **The three decoders that touch the non-normal / strong-sampling /
lattice side are exactly the bounded or negative cases** - Exp 31 (cospectral-bounded, 50.2
anchor), Exp 45.5 (NxN cannot capture 2^N), and Exp 25 (toy-scale-only, 50.4). None of the
working decoders secretly relies on crossing the located wall. The decodable class and the
located barrier together give a consistent partition of the entire lab's decoder arsenal.

## Gates

| Gate | Result | Detail |
|---|---|---|
| G1 anchors measured (decodable abelian vs collapsed non-normal) | PASS | Z_64 D=1.000 vs D_8 D=0.172, d=6.68 |
| G2 Exp 44 handoff predicted peaks validated | PASS | PEAKS_VALID (<0.03% error vs mpmath zeros) |
| G3 class-map partition consistent; wall set = {25,31,45.5} | PASS | matches expected |
| G4 no working decoder relies on a non-normal/strong-sampling step | PASS | all 9 decodable are abelian-HSP / topological-invariant |

## Honest scope

The per-experiment classification (rows 34-46) is ANALYSIS from each experiment's documented
mechanism, not a re-run of every experiment - the No-Subagents / direct-verification rule
forbids treating those as independently re-measured here. What IS measured: the two anchors of
the map and the handoff peaks. The classification is anchored by the Exp 50 results that ARE
measured (50.1 extractive, 50.2 abelian/cospectral, 50.4 lattice).

## Reproduce

```
python THOUGHT/LAB/CAT_CAS/50_the_decoder/50_5_decoder_class_map/decoder_class_map.py
```
Writes `decoder_class_map_result.json` + `output_class_map.txt`. Exits 0 iff all gates pass.
