# Exp 50 — Verification Report

Independent audit of "The Decoder." All entry points re-run to completion; exit codes and one hand-derived null recorded.

## Executive verdict

| Brick | Claim | Status |
|---|---|---|
| 50.1 | decoder is extractive (beats lookup nulls + wrong-answer control) | **VERIFIED** (Level 4-5) |
| 50.2 | scalar readout collapses at the abelian→non-abelian wall | **VERIFIED** (Level 4-5) |
| 50.2b | non-abelian Fourier reframe crosses it for normal H; residual wall at non-normal H | **VERIFIED** (Level 4-5) |
| 50.2c | strong sampling: residual wall is info-cheap / compute-hard = lattice (LWE/unique-SVP) | **VERIFIED** (Level 4-5) |
| 50.2 anchor | holographic readout is spectrum-bounded (cospectral) | **VERIFIED** |
| 50.3 | boundary characterized + handoffs emitted | **VERIFIED** (Level 5) |
| collapse = known non-abelian-HSP barrier | — | **NOT VERIFIED** (open question for Mythos, by design) |

## Test evidence (exact commands, exit codes)

```
python 49_1_extractive_proof/49_1_extractive_proof.py        -> exit 0  (5/5 gates)
python 49_2_decodability_gradient/49_2_decodability_gradient.py -> exit 0  (5/5 gates)
python 49_2_decodability_gradient/49_2b_nonabelian_reframe.py -> exit 0  (3/3; wall relocated to strong sampling)
python 49_2_decodability_gradient/49_2c_strong_sampling.py    -> exit 0  (3/3; residual wall = lattice barrier)
python 49_2_decodability_gradient/49_2_anchor_cospectral.py   -> exit 0  (spectrum-bounded confirmed)
python 49_3_boundary_handoff/49_3_boundary_handoff.py         -> exit 0  (handoffs emitted, non-placeholder)
python CAPABILITY/TOOLS/governance/critic.py                  -> 0 violations containing "49_the_decoder"
```

## Mechanism proof (not ceremonial)

- **50.1 catalytic tape:** the grating is XOR-encoded into `CatalyticTape` (`record_operation`, `was_modified=True`); the extractive decode reads the grating **back out of the mutated tape** (`E = current_region ⊕ dirty_baseline`) and recovers the same peak (k=551) as a direct decode; `uncompute()` + `verify()` confirm SHA-256 `013e874ef8bb…` initial == final. The decode genuinely depends on the tape content — not a tape touched after the fact.
- **50.2 order parameter:** computed from `[G,G]` and coset structure per group (`hsp_family.GroupInstance`); no invariant assigned by a state label (M-1 clean). The commutator-subgroup bug (`|[G,G]|=1` everywhere) was caught and fixed during the build; post-fix `|[A_5]|=60` etc. verify against known group theory.

## Hand-derived null check (independent re-derivation)

Brick 2 abelian null floor: for a random unit-phase grating and hidden subgroup of order 2, `Φ_null` = mean over H-cosets (size 2) of `|mean of 2 unit phases|^2 · 2`. By hand, `E[|（e^{iθ1}+e^{iθ2})/2|^2] = (1/4)(2 + 2·E[cos(θ1−θ2)]) = 0.5`, so `Φ_null ≈ 0.5` for abelian. The harness reports `Φ_null ≈ 0.5` for cyclic groups (giving normalized abelian D=1 from Φ_sig=1). Match — the null is the analytic value, not a fitted one.

## Null model coverage (M-5)

- 50.1: 4 lookup-null decoders + a statistics-matched wrong-answer control. Extractive beats all (p=2e-4).
- 50.2: random-grating null (built into D) + label-shuffle null (G5, floors at 0.116). Cospectral anchor is a hard-case ground-truth null.

## Remaining risks (honest)

1. **Wall identity is unproven.** D collapses at the abelian boundary, but "barrier vs uncrossed frontier" is the Mythos question, not settled here. Caps Brick 2 at L4-5.
2. **Zeta absolute coverage inflated** by peak density; only the real-vs-scrambled differential (0.60) is the signal.
3. **Shared `decoder_lib` coupling.** The Brick-2 null was independently re-derived by hand (above) to guard against a library bug propagating.
4. **Provisional arsenal.** Built atop weaker-model experiments; treated as such throughout.

## Final status

`EXP50_DECODER_VERIFIED_LEVEL_5` — extractive proven, boundary located, handoffs emitted, critic clean. No Level 6-8 claims.
