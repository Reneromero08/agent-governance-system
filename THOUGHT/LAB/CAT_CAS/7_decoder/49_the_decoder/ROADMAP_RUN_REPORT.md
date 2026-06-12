# Exp 50 - Roadmap Run Report

A single-session execution of the Exp 50 ROADMAP, **excluding the Mythos call (#3) by
instruction**. All work stayed in `THOUGHT/LAB/CAT_CAS/`; built, run, and verified directly
(no delegated verification, per CAT_CAS_OS s.26). Claims capped Level 4-5.

## Status of every roadmap item

| Item | Status | Brick | Verdict |
|---|---|---|---|
| #1 Lattice audit (Exp 25) | DONE | `49_4_lattice_audit/` | `EXP25_TOY_SCALE_ONLY` |
| #2 Kuperberg subexponential rung | DONE | `49_2_decodability_gradient/49_2d_kuperberg_sieve.py` | `DIHEDRAL_BARRIER_SUBEXPONENTIAL_UPPER_BOUND` |
| #3 Call Mythos | SKIPPED (instruction) | - | sandbox ready, not invoked |
| #4 Exp 44 Phase 6 handoff | VALIDATED; silicon HARDWARE-BLOCKED | `49_5_decoder_class_map/` | handoff peaks correct (<0.03%) |
| #5 Generalize the gradient | DONE | `49_2_decodability_gradient/49_2e_gradient_coverage.py` | `WALL_IS_NON_NORMAL_SUBGROUPS` |
| #6 Connect to bigger claims | DONE | `49_5_decoder_class_map/` | `DECODER_MAP_CONSISTENT` |

## Verification

- **All 10 Exp 50 entry points exit 0** (6 original + 4 new).
- **Lab critic (M-1..M-8) clean** across all of CAT_CAS.
- Every new brick carries matched nulls, statistics (CI / Cohen d / chance baselines), and a
  written report. No catalytic tape was added where the mechanism does not borrow a substrate
  (anti-ceremonial).

## What the run established (and what it did not)

**1. Exp 25's LWE "break" does not cross the located barrier (50.4).** Running Exp 25's own
holographic phase-resonance attack under matched nulls: it recovers a secret only at a tiny
modulus (q=5, n=2; exact_rate 0.25), and recovers **nothing at its own shipped default**
(n=128, q=3329), nothing at realistic noise (sigma>=2), and its objective cannot distinguish a
planted instance from a no-secret null (resonance Cohen d=0.16). The `LATTICE BROKEN!` message
is a toy-scale / error-free artifact. This relocates the lab's boldest crypto claim onto the
bedrock located in 50.2c and shows it does not survive. *Not claimed:* a proof about all
lattice attacks - only this one, under this null discipline.

**2. The wall is NON-NORMAL hidden subgroups, not non-abelianness (50.2e).** Q_8 (quaternion,
non-abelian but Hamiltonian - every subgroup normal) is **decodable** (D_char=1.000), while
dihedral / AGL(1,5) / A_5 / S_n (non-normal H) collapse (Cohen d=8.98). Decodability tracks
H-normality. The scalar FFT readout is bounded by the weaker abelian wall (it misses Q_8); only
the non-abelian character/quotient reframe crosses to normal subgroups. The order parameter is
implementation-robust (two independent phi agree to 2.2e-16).

**3. The barrier is subexponential-but-superpolynomial (50.2d).** Kuperberg's collimation sieve
recovers the dihedral slope in subexponential queries (2^n/M_needed widens 4x -> 4.2e6x over
n=6..30; conditional readout correctness 1.000). Combined with 50.2c's poly(n)-budget failure
this sandwiches the barrier: super-polynomial (no poly readout) but subexponential (not full
2^n) - the exact shape of unique-SVP hardness. *Not claimed:* an empirical separation of
subexp from polynomial (the fits are indistinguishable over the reachable n-range); the
super-poly side is inherited from 50.2c + the standard Regev/Kuperberg result.

**4. The lab's decoder arsenal partitions consistently (50.5).** 9 working decoders are
decodable (Exp 20/24 abelian-HSP; Exp 34/35/36-40/45/46 topological invariants of poly-size
operators); the 3 that touch the non-normal / lattice side are exactly the bounded or negative
cases (Exp 31 cospectral-bounded, Exp 45.5 NxN-cannot-capture-2^N, Exp 25 toy-scale). None of
the working decoders secretly relies on crossing the located wall. The Exp 44 handoff's
predicted resonant peaks were validated against independently computed Riemann zeros (<0.03%).

## The single open thread deliberately left

**The Mythos question (#3):** is the unique-SVP barrier itself crossable by any
holographic/topological/catalytic readout? The sandbox is ready and now has a concrete result
to reason about (Exp 25's toy-only failure). It was not invoked this session by instruction.
The silicon acceptance run (#4) remains hardware-blocked on the live Phenom.

## Net effect on the Exp 50 picture

The decodable class is now `{abelian-HSP} + {normal hidden subgroups} + {topological invariants
of a poly-size operator}`; the residual wall is the **non-normal / strong-sampling case =
lattice (unique-SVP)**, characterized as subexponential-but-superpolynomial; and the lab's one
claim to cross that wall (Exp 25) has been audited and does not. Everything stayed at Level 4-5
with honest negatives recorded (the degenerate MUSIC readout, the un-separable subexp/poly fit,
the toy-scale recovery that keeps Exp 25 from being called simply "non-functional").
