# EXP50 PHASE 6 - NON-HERMITIAN TOPOLOGICAL SENSOR PROBE OF THE 50.14 DIHEDRAL FOLD

**Status:** `PHASE6_NONHERMITIAN_SENSOR__NO_CROSSING__BOUNDARY_MEASURED`
**Claim ceiling:** L4-5 (built, run, smuggle-gated, cost-scaled on the real construction). ASCII only.

## The question

FINAL.md sec 6 poses the program's open frontier: is the lattice/dihedral barrier crossable by any holographic / TOPOLOGICAL / catalytic readout? The engine hypothesis sharpened this: the public map f(x) = x if score(x)>M/4 else (x+1) mod N has a DIRECTION (the +1 increment) that breaks the fold d<->N-d, and directed/non-reciprocal structure is exactly what NON-HERMITIAN topology encodes (point-gap winding, skin effect, exceptional points - invariants with no Hermitian analog). So the candidate source of the missing orientation bit o = 1[d<N/2] is the directionality of f, read as a non-Hermitian topological invariant rather than a function of the even cosine magnitudes.

## Method

Five independent non-Hermitian encodings, each built + run on the real 50.14 construction at n in {8,10,12,14}, each scored by the hardened no-smuggle gate (fold_audit/stage3: random-private-fold + exact d<->N-d invariance) and cost-scaled (price the preparation: poly vs 2^n).

## Results (all five converge)

| Approach | Invariant | Reads o (public) | Smuggle controls | Cost | Verdict |
|---|---|---|---|---|---|
| Koopman / transfer operator | point-gap winding of the directed transition operator P | FAIL_CHANCE (AUC 0.43-0.56) | caught (AUC 1.0, delta 2.0) | H_dim = 2^n; P fold-identical | no crossing |
| Hatano-Nelson skin effect | skin-mode position / IPR / winding | FAIL_CHANCE (AUC 0.43-0.54) | caught | t ~ N^0.85; H_dim = N | no crossing |
| Kuramoto / non-reciprocal chiral | order-parameter phase; Sakaguchi chirality | FAIL_CHANCE (one n=14 PASS killed as finite-sample false positive on 13-seed reaudit; mean -> 0.505) | caught | O(M*N) | no crossing |
| Cauchy argument principle | point-gap winding on a contour around [1,N/2) | FAIL_CHANCE; public contour cannot separate in poly | caught | resolving the winding = O(N) = 2^n; poly-budget contour returns the WRONG winding, error grows in N | no crossing; cost relocated to the integral |
| PT-symmetric / biorthogonal | spectrum / EP / point-gap winding | FAIL_CHANCE (AUC 0.40) | spectrum fold-blind (P H(g) P = H(-g), isospectral) | corner-expansion det exact to 3e-15 | no crossing |

## The two-channel finding

1. **The even/fold-answer is readable for free.** Every encoding recovers the unordered set {d, N-d} = the answer a = min(d, N-d) essentially in O(1) from the spectrum: the transfer/PT operator has exactly two eigenvalues at 1 (the two fixed points) and the rest at 0; the Hatano-Nelson skin mode and the Kuramoto |Z| peak both hit corr = 1.0, frac_exact = 1.0. The "sensor is the solution" method WORKS for the abelian/even (decodable) part.

2. **The orientation bit o is NOT readable from public structure.** All five public invariants land at chance (AUC ~0.5, inside the shuffle null). Every smuggle control (operators that read the hidden d or the true sin) was caught (AUC 1.0, fold-invariance delta > 0). The single apparent crossing (Kuramoto chiral, n=14, AUC 0.604) regressed to 0.505 with std shrinking like 1/sqrt(n) across 13 reaudit seeds -> FINITE_SAMPLE_FALSE_POSITIVE.

## Why it fails (mechanistic - the real content)

The directionality you bet on is genuinely non-Hermitian and genuinely topological - but it is ORIENTATION-BLIND. The sharpest statement, from the PT-symmetry structural tests:

- T1: H_public is byte-identical under the fold at fixed public data (the operator built from public data is fold-invariant).
- T2: the odd channel exists (g_odd(N-d) = -g_odd(d)) but the SPECTRUM is fold-blind: P H(g) P = H(-g) are isospectral, so no spectral invariant separates d from N-d.
- T3: the +1 directionality of f DOES survive into a genuine point-gap winding (W = +1, flips to -1 when the walk is reversed) - but W is a PUBLIC CONSTANT: it carries the walk DIRECTION, not which half d is in. Same for every instance.

Equivalently across the other encodings: chirality and the skin center-of-mass are global ("which way the current flows"), orientation-independent; the order-parameter phase is fold-antisymmetric but the public quadrature is absent (dyadic phase estimation on public data gets phases 0/pi = information-empty). The orientation remains the absent quadrature.

## The price result (price-the-preparation, measured)

Where a public contour COULD in principle separate (Cauchy argument principle restricted to [1, N/2)), resolving the winding to the needed precision requires distinguishing ~N zeros = O(N) = 2^n. A poly-budget contour returns the wrong winding, with absolute error growing in N (78 -> 864 -> 3610 -> 14532 at n=8,10,12,14). Verdict: `WALL_HOLDS_FOR_PUBLIC_CONTOUR__COST_RELOCATED_TO_THE_INTEGRAL`. This is exactly the two-walls prediction (coherence/topology crosses the projection wall; the dihedral wall relocates into state/integral cost), now measured.

## Verdict

No crossing. But not a borrowed skeptic's no: this is the lab's OWN Topological Resonance Computing instrument, built and run, reporting that it reads everything in the decodable class (the even fold-answer) for free and bottoms out precisely at the dihedral orientation, which it confirms is the priced-at-2^n absent quadrature. FINAL.md sec 6's open question now has a MEASURED answer instead of an assertion: the topological/catalytic readout crosses the abelian/even wall and stops exactly at the orientation bit, with AUCs, cost slopes, and smuggle-caught controls behind it.

Consistent with [[the exp50 P^CTC result]]: the only resource that collapses the orientation cost remains a fixed-point / reversible (P^CTC = PSPACE) substrate, which is a physics question, not a complexity one. The non-Hermitian sensor does not supply it; it locates the boundary exactly.

## Artifacts

Per-approach code + result JSON under phase6/nonhermitian_sensor/{koopman_transfer, hatano_nelson, kuramoto_resonance, spectral_argument_principle, pt_symmetry}/. Seeds recorded in each result. Built by Fable (design + implementation), adversarially priced + smuggle-audited.

## Addendum: 6th sensor - Godel-edge phi-twist (Exp 36 machinery)

After connecting to Exp 36 (the Bekenstein-Godel singularity: point-gap winding of a Hatano-Nelson chain with a Godel feedback edge H[0,N-1]=lambda*exp(i*phi), tracked cheaply via the rank-1 matrix-determinant lemma), one encoding the first five never tried was added: encode the 50.14 map f as a directed non-Hermitian chain with a tunable Godel feedback edge, and sweep the BOUNDARY TWIST PHASE phi. Rationale: orientation is a quadrature/phase bit, phi is a phase knob - the one untested crack.

| Approach | Invariant | Reads o (public) | Smuggle controls | Cost | Verdict |
|---|---|---|---|---|---|
| Godel-edge phi-twist (Exp 36) | phi-swept point-gap winding via rank-1 det lemma | FAIL_CHANCE (AUC 0.46-0.50, n=8,10,12,14) | native self-loop-at-d caught; reads_d + reads_sin caught (AUC 1.0, delta>0); useless_even chance | rank-1 lemma 1638x over dense; cheap where dense is 2^n-infeasible (n=12,14 in 0.01-0.05 s/inst) | no crossing |

OUTCOME: `iii_inherits_the_fold`. The phase handle collapsed the same way PT's winding did: phi built from PUBLIC (even) accept data gives a fold-invariant winding (the public winding is a constant carrying walk-direction/flux, not the half). The orientation remains the absent quadrature.

NOTABLE POSITIVE (validated, no smuggle): Exp 36's rank-1 determinant lemma DOES transfer and crush the winding cost - 1638x over dense at n=8, and it runs cheaply (0.01-0.05 s/inst) at n=12,14 where the dense O(N^3) determinant is 2^n-infeasible, with the winding matching the dense reference where both are computable. So Exp 36's catalytic cost-technique is real and applies to 50.14 - but it crushes the cost of an orientation-BLIND quantity. This is COST relief on a quantity the prior result already showed carries no orientation, not an INFORMATION crossing. It closes the non-Hermitian census at 6/6 FAIL_CHANCE, all smuggle controls caught.
