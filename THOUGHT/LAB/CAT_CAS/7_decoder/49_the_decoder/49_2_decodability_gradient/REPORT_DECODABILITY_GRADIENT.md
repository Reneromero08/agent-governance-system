# Exp 50.2 — The Decodability Gradient

**Verdict:** `BOUNDED_AT_ABELIAN_HSP_WALL` (5/5 gates) + anchor `SPECTRUM_BOUNDED_CONFIRMED`. Claim level **4-5**, provisional.

## Overview

- **Mechanism:** slide a Hidden Subgroup Problem family from abelian (cyclic Z_n) through dihedral (D_m) to symmetric (S_n), and measure where the holographic/scalar readout's recovery of the hidden subgroup collapses.
- **Order parameter:** `D = (Φ_signal − Φ_null)/(1 − Φ_null)`, where Φ is the fraction of the coset-grating energy accessible to the group's 1-dimensional (abelian) characters — i.e. the energy surviving pushforward to `G/([G,G]·H)`. Rigorous and ordering-independent. Confirmed by the literal FFT-readout channel.
- **Claim (provisional):** the holographic readout decodes exactly the abelian-HSP class and collapses at the first non-abelian group. Whether the collapse is the *known* non-abelian-HSP barrier or a frontier the scalar readout simply hasn't crossed is the **question handed to Mythos** — not a verdict.

## Method

Permutation-group engine (`hsp_family.py`): closure, commutator subgroup `[G,G]`, cosets — all numeric, no per-group formulas. Uniform hard hidden subgroup `H` of order 2 (transverse to `[G,G]` for non-abelian groups — the hard HSP case). `dim_G`-dimensional regular-representation gratings; no SAT vocabulary, no `(N,N)` matrix (M-4 cannot fire). 40 random coset instances per group; null = random grating; label-shuffle null in addition.

## Results

| group | \|G\| | \|[G,G]\| | d_max | abelianness | D_char [95% CI] | D_fft |
|---|---|---|---|---|---|---|
| Z_16…Z_64 | 16–64 | 1 | 1 | 1.000 | **1.000** [1.000,1.000] | 1.000 |
| D_8 | 16 | 4 | 2 | 0.250 | 0.172 [0.122,0.227] | ~0 |
| D_16 | 32 | 8 | 2 | 0.125 | 0.056 [0.032,0.080] | ~0 |
| D_32 | 64 | 16 | 2 | 0.062 | 0.033 [0.020,0.048] | ~0 |
| A_4 | 12 | 4 | 3 | 0.250 | 0.399 [0.310,0.480] | ~0 |
| S_4 | 24 | 12 | 3 | 0.083 | 0.063 [0.036,0.095] | ~0 |
| S_5 | 120 | 60 | 4 | 0.017 | 0.009 [0.004,0.014] | ~0 |

**Shelf mean D=1.000, pole mean D=0.110.** Both the rigorous character channel and the literal FFT channel collapse from 1.0 (abelian) to ~0 (non-abelian).

## Located collapse

**First dihedral group D_8 (|G|=16, d_max 1→2, abelianness 0.250), D_char=0.172.** Cohen d (shelf vs pole) = **8.82**, permutation p = 1.3×10⁻³ (at the ~8×10⁻⁴ resolution floor for n=13 groups; effect size is the decisive statistic).

## Gates

| Gate | Result |
|---|---|
| G1 shelf vs pole discrimination | PASS (shelf 1.000, pole 0.110, d=8.82) |
| G2 knobs agree (abelianness & d_max) | PASS (collapse at d_max 1→2 under both orderings) |
| G3 scale-independence | PASS (at every shared \|G\|∈{16,24,32,48,64}: cyclic 1.00, dihedral collapsed — the wall is *not* a \|G\| artifact) |
| G4 robustness (seed) | PASS (seed-2 shelf 1.000, pole 0.090) |
| G5 label-shuffle null floor | PASS (max \|shuffle D\| = 0.116) |

## Cospectral anchor (the Exp 31 case, finally reported)

Shrikhande vs Rook(4×4), both SRG(16,6,2,2): **identical spectra** (signature distance 2.3×10⁻¹⁵, participation 13.5 both) yet **non-isomorphic** (4-cliques: Rook 8, Shrikhande 0). The holographic/spectral readout **cannot distinguish them** → `SPECTRUM_BOUNDED_CONFIRMED`. This pins the deep-non-abelian pole: the readout fails the cospectral graph-iso case exactly as it fails non-abelian HSP — it is spectrum-bounded. (Exp 31 had this code but never reported the result; the honest result is failure at the hard case.)

## Anti-toy defenses (all held)

Structural x-axis (`d_max`/abelianness, computed from the group, code-independent); two knobs agree (G2); scale-independence (G3 — the decisive test: an artifact scales with |G|, the wall doesn't); geometric concern handled by the character channel being ordering-independent; null-crossing required; cospectral ground-truth anchor. The collapse is a genuine measurement, not a size artifact.

## 50.2b — The non-abelian Fourier reframe (we crossed the wall ourselves)

Before handing the wall to a stronger model, we pushed the readout harder: the
**non-abelian Fourier transform** (weak sampling). Its distinguishing power depends
on the hidden subgroup only through its **conjugacy-class composition** (since the
weak-sampling distribution `P(ρ) ∝ d_ρ Σ_{h∈H} χ_ρ(h)` is a class function of H), so it
is computable from conjugacy classes alone — no explicit irreps. `49_2b_nonabelian_reframe.py`.

| readout | non-abelian groups, D |
|---|---|
| scalar / 1-D-character (Brick 2) | 0.123 (collapsed) |
| non-abelian Fourier, **normal** hidden H | **1.000 — CROSSED** |
| non-abelian Fourier, **non-normal** hidden H | 0.157 — residual wall |

(separation 0.843; gates R1/R2/R3 all PASS; verdict `WALL_RELOCATED_TO_STRONG_SAMPLING`.)

**The "abelian vs non-abelian" wall was not the real barrier — we passed it.** The
reframe recovers every *normal* hidden subgroup. The **residual wall** is at
*non-normal* subgroups (dihedral reflections, transpositions), where weak sampling
cannot separate conjugates (they share a class composition). That residual is exactly
**strong Fourier sampling**, whose dihedral case is reducible to/from unique-SVP
(Regev) — lattice-hard. This is textbook-correct HSP theory, reproduced from scratch.

## 50.2c — Strong Fourier sampling: we hit the lattice bedrock

We climbed the next rung too: **strong** (within-irrep) Fourier sampling on the dihedral
coset states (`49_2c_strong_sampling.py`). The coset state for hidden slope `d` is
`(|0> + e^{2πi k d/N}|1>)/√2` for a random measured `k`. Results (3/3 gates,
`STRONG_SAMPLING_CONFIRMS_LATTICE_BARRIER`):

- **Zero info per state:** averaged over random `k`, a single coset state is `I/2`,
  independent of `d` (measured coherence ≈ 0.004).
- **Info-cheap:** the slope is determined by O(√N) coset states (T_min/N: 0.5→0.098) —
  not an information-theoretic wall (Ettinger-Hoyer).
- **Compute-hard:** recovering `d` is a search over the 2ⁿ secret space. A poly(n)-budget
  search succeeds only with probability ~2B/N → 0 (measured 0.20→0.07→0.03→0.00 at
  N=256..2048), exactly tracking the budget fraction — no structural shortcut — while
  full search = 1.000.

The residual wall **is** the 1-bit-LWE / dihedral-HSP ↔ unique-SVP lattice problem
(Regev); best known (Kuperberg) is subexponential, still not poly(n). It is the same
lattice hardness Exp 25 (LWE/SVP) claims to break.

## Conclusion

The scalar readout collapses at the abelian→non-abelian boundary — but that boundary is **crossable** (50.2b non-abelian Fourier recovers all normal hidden subgroups). We then climbed to strong sampling (50.2c) and reached **bedrock**: the genuine residual wall is the dihedral slope of non-normal subgroups, which is *information-cheap but computationally a 2ⁿ secret-space search = the lattice (LWE/unique-SVP) barrier*. We crossed every rung we could ourselves; the remaining question — *is the lattice barrier itself crossable?* — is a cryptographic-hardness question, now precisely scoped for Mythos. Provisional, given the weak-model arsenal.
