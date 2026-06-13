# REPORT - Stage 1 Fold Audit (Exp 50 Phase 6)

**Status:** COMPLETE. Verdict `FOLD_AUDIT_CONFIRMED`. All predictions held = True.
**Claim ceiling:** L4-5. Information-theoretic measurement on the REAL construction, not a toy.
**Determinism:** master_seed = 44060611; every per-section seed recorded in `fold_audit_result.json`.
**Run:** `..\..\..\..\..\.venv\Scripts\python.exe fold_audit.py` (elapsed ~221s).

---

## 0. One-sentence result

On the lab's REAL Exp 50.14 public fixed-point map, the orientation bit
`b_orient = 1[d < N/2]` is **information-absent from the scalar/cosine channel**
(measured held-out AUC ~0.5 across 16 classifier results and 8 equivariant-lift
results; the `d`- and `(N-d)`-conditioned public distributions are two-sample
**indistinguishable**) and **present in the quadrature/complex channel** (trivial
`sin` readout AUC = 1.000 at every `n`; one-shot exact `d` recovery from the dyadic
frequency ladder, no search). The reusable no-smuggle gate correctly flags both a
`d`-reader and a `sin`-reader as SMUGGLE and a pure even transform as CHANCE.

This converts the theoretical encoding-wall claim of `SPEC_PHASE6` Sec 1B.1 / 1C
into a MEASURED fact.

---

## 1. The construction audited (and where it was found)

The construction is the lab's **Exp 50.14 public fixed-point map**, lifted **verbatim**
(no new modeling choices) from:

- `THOUGHT/LAB/CAT_CAS/49_the_decoder/49_14_reversible_substrate/49_14_substrate.py`
  - functions `coset_samples` (lines 51-55) and `make_verify` (lines 58-69)
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/phase6/SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md`
  - Sec 2 gives the exact `score`/`accept`/`f`; Sec 1B.1 + 1C state the fold claim this
    audit tests (Sec 1B.1 explicitly names "the Stage 1 fold audit (phase6/fold_audit/)").

Re-exported unchanged in `construction.py` (single source of truth). Exact parameters:

| Parameter | Value (verbatim) | Source |
|---|---|---|
| Modulus | `N = 2^n` | 49_14 main(); SPEC Sec 2 |
| Public frequencies | `k_i ~ Uniform{0,...,N-1}`, `i=1..M` | `coset_samples` |
| Noise model on bits | `p_i = (1 + cos(2*pi*k_i*d/N))/2`; `b_i = +1 w.p. p_i else -1`; `E[b_i] = cos(2*pi*k_i*d/N)` | `coset_samples` |
| Sample count | `M = max(4*ceil(sqrt(N)), 48*n)` (`M ~ O(sqrt N)`) | 49_14 main() |
| Public score | `score(x) = sum_i b_i*cos(2*pi*k_i*x/N)` (O(M); `(k,b)` only) | `make_verify` / SPEC Sec 2 |
| Accept | `accept(x) = score(x) > M/4` (true iff `x in {d, N-d}`) | `make_verify` |
| Fixed-point map | `f(x) = x if accept(x) else (x+1) mod N` | 49_14 `f_map` |
| Secret pinning | `d in [1,N)`; unique fixed point in `[1,N/2)` is `min(d, N-d)` | SPEC Sec 2 |
| Orientation bit | `b_orient = 1[d < N/2]` (the bit `sigma: d->N-d` destroys) | SPEC Sec 1B.1 |

**N choice (realistic, stated):** the lab's bricks instantiate this at `n = 8,10,12,14,16`
(49_14_substrate.py loops `n in (8,10,12,14)`; SPEC Sec 6 lists up to 16). The audit runs
the representative span **`n = 8, 10, 12, 14`** (`N = 256, 1024, 4096, 16384`) - the lab's
real operating range, NOT a toy. `M` is the real `O(sqrt N)` count (M=384 at n=8 .. M=672
at n=14). `n=16` omitted only for wall-clock; trends are flat in `n`.

Fold symmetry verified exact: `cos(d) - cos(N-d) ~ 1e-12`; `sin(d) + sin(N-d) ~ 1e-12`
(the `sin` channel flips sign under the fold).

---

## 2. AUC table (held-out, 5-fold CV, 600 instances per n; chance = 0.5)

### (a) SCALAR / COSINE channel - classifier battery
PREDICTION ~0.5 (bit absent). RESULT: held.

| n | logistic_regression | rbf_svm | gradient_boosted_trees | mlp |
|---|---|---|---|---|
| 8  | 0.513 | 0.540 | 0.464 | 0.526 |
| 10  | 0.486 | 0.462 | 0.494 | 0.470 |
| 12  | 0.518 | 0.500 | 0.475 | 0.506 |
| 14  | 0.517 | 0.537 | 0.503 | 0.525 |

Range across all 16 cells: **0.462 .. 0.540**. No classifier - linear, kernel,
boosted-tree, or neural - extracts the bit.

### (b) EQUIVARIANT-LIFT control - nonlinear lifts of the PUBLIC cosines
Random Fourier features `cos(W s + phi)`, degree-2 poly cross-terms, `tanh`/`cos`, applied
to the public score vector over probes in `[1,N/2)`. PREDICTION stays at chance. RESULT: held.

| n | logreg_on_lift | gbt_on_lift |
|---|---|---|
| 8  | 0.490 | 0.525 |
| 10  | 0.446 | 0.467 |
| 12  | 0.556 | 0.536 |
| 14  | 0.475 | 0.484 |

Range across all 8 cells: **0.446 .. 0.556**. Empirical equivariance theorem
(SPEC 1B.1): no lift of any dimension recovers a bit the public data never wrote.

### (c) QUADRATURE / COMPLEX channel
Provide `z_k = exp(-2*pi*i*k*d/N)` on the FIXED dyadic frequencies `k = N/2,...,1` (the
odd channel must be read at FIXED freqs: `E_k[sin]=0` over random k). PREDICTION ~1.0. Held.

| n | logreg_on_quadrature | trivial_sin_sign_readout (no training, k=1) |
|---|---|---|
| 8  | 0.9999 | 1.0000 |
| 10  | 0.9993 | 1.0000 |
| 12  | 0.9981 | 1.0000 |
| 14  | 0.9977 | 1.0000 |

Trivial readout = `1[sin(2*pi*d/N) > 0]` at the single rung k=1: exact (AUC 1.000) because
`sin(2*pi*d/N) > 0 <=> 0 < d < N/2`.

---

## 3. Two-sample distinguishability test
Energy distance between scalar public features for lower-half `d` vs upper-half `d`
(240/group, 2000 perms), with a within-class control (two random halves of the SAME
orientation = true null). PREDICTION not distinguishable. RESULT: held.

| n | cross-orientation energy dist | within-class energy dist (null) | perm_p |
|---|---|---|---|
| 8  | 0.0745 | 0.1123 | 0.056 |
| 10  | 0.0709 | 0.1425 | 0.109 |
| 12  | 0.0597 | 0.1173 | 0.547 |
| 14  | 0.0620 | 0.1345 | 0.451 |

Decisive: at every `n` the cross-orientation distance is SMALLER than the within-class
distance - the two fold-conditioned distributions are statistically identical, the
difference below within-orientation sampling noise. (An earlier 120/group run gave a
marginal `p=0.017` at n=14; 240/group removed it = finite-sample noise. Reported for honesty.)

---

## 4. One-shot d recovery (no search)
`one_shot_recover_d` reconstructs `d` in ONE non-adaptive parallel shot from the dyadic
ladder `k = N/2,...,1` via Kitaev phase estimation (each rung's phase = one bit of `d`,
MSB->LSB). No iteration, no scan. PREDICTION exact. RESULT: held -

| n | exact-recovery rate (300 trials) |
|---|---|
| 8  | 1.0000 |
| 10  | 1.0000 |
| 12  | 1.0000 |
| 14  | 1.0000 |

Examples: `d=3865 -> 3865` (n=12), `d=13128 -> 13128` (n=14). The full n-bit secret is
pinned from n parallel quadrature reads. SPEC 1C CROSSING SPEC made concrete.

---

## 5. No-smuggle gate (the reusable instrument) + self-test
`no_smuggle_gate.gate(O, n, ...)` audits a candidate quadrature-synthesis op `O` on two
axes:
- **AXIS 1 exact d-invariance audit.** Hand `O` the two hidden states (`d`, `N-d`) that
  give IDENTICAL public data; `max_fold_delta` = max abs output change. `delta==0` PROVES
  `O` is a pure function of public data; `delta>0` means it read `d` (smuggle).
- **AXIS 2 orientation signal.** Held-out AUC calibrated against a per-dataset
  **label-shuffle null** (95th pct). `above_chance := auc > shuffle_null_95`. A provably
  public (`delta=0`) `O` cannot beat its own shuffle null -> never mislabeled a crossing.

| Verdict | Condition | Meaning |
|---|---|---|
| `PASS_CROSSING` | above_chance AND not reads_d | genuine quadrature from PUBLIC data, no `d`-dependence |
| `FAIL_SMUGGLE` | above_chance AND reads_d | lifted the bit only by reading `d` |
| `FAIL_CHANCE` | not above_chance | manufactured no bit (expected even-transform outcome) |

`O` contract: `O(instance) -> 1D float vector`, `instance = {k, b, N, d}`; `d` present
ONLY for the gate's invariance audit - a non-smuggling `O` must never read it.

### Self-test results (n=10, 400 instances) - all three correct
| Self-test O | What it does | Required | Measured | AUC | reads_d | max_fold_delta |
|---|---|---|---|---|---|---|
| `O_cheat_reads_d` | returns `1[d<N/2]` from hidden `d` | FAIL_SMUGGLE | **FAIL_SMUGGLE** | 1.000 | True | 1.00 |
| `O_cheat_reads_sin` | reads `sin(2*pi*k*d/N)` (k=1,2,3) from hidden `d` | FAIL_SMUGGLE | **FAIL_SMUGGLE** | 1.000 | True | 2.00 |
| `O_useless_even` | nonlinear transform of PUBLIC `(k,b)` only | FAIL_CHANCE | **FAIL_CHANCE** | 0.491 | False | 0.00 |

The useless-even `O` has `delta=0.00` (provably public) and AUC below its own shuffle null
-> CHANCE (not fooled by finite-sample AUC noise). The two cheaters are caught by the exact
invariance audit (`delta=1.0`, `2.0`) regardless of AUC. Instrument is ready for the next
stage: a PASS requires lifting the bit from PUBLIC-ONLY data with `delta=0`.

---

## 6. Verdict and discipline

**VERDICT: `FOLD_AUDIT_CONFIRMED`. All predictions held.**

> The orientation bit `b_orient = 1[d < N/2]` is **information-absent from the scalar
> channel** (AUC ~0.5 across 16 classifiers + 8 lifts, range 0.462-0.556; fold-
> conditioned public distributions two-sample indistinguishable, cross-distance below the
> within-class null at every `n`) and **present in quadrature** (trivial `sin` readout AUC
> 1.000; one-shot exact `d` recovery from the dyadic ladder at n=8,10,12,14).

### PROVEN (measured here)
- On the REAL Exp 50.14 construction at real parameters (`n=8..14`, `M~sqrt N`), no
  classifier or scalar lift extracts the orientation bit (AUC ~0.5).
- The two fold-conditioned public distributions are statistically identical (within-class control).
- The quadrature channel trivially yields the bit (AUC 1.0); the dyadic ladder yields one-shot exact `d`.
- The no-smuggle gate distinguishes a public crossing from a `d`-smuggle and a useless even transform.

### ASSUMED / NOT claimed
- **No physical crossing is claimed.** This measures channel information content; it does NOT
  show the quadrature is physically accessible from THIS construction's PUBLIC data.
  Synthesizing quadrature without reading `d` is the open problem (SPEC 1C: dihedral / hidden-
  shift barrier, Kuperberg `2^{O(sqrt n)}`). The gate exists to adjudicate future candidate
  syntheses; here the only ops that produced quadrature did so by reading `d` (gate flagged them).
- One-shot recovery is GIVEN the quadrature `z_k` (the SPEC's premise), not derived from public
  even data. It shows IF quadrature is available the algorithm is dead; it does not manufacture it.

### Honesty notes
- The n=14 two-sample marginal (`p=0.017` at smaller group) is reported and shown to be finite-
  sample noise. No prediction hidden.
- Two readout bugs found and fixed mid-build (random-`k` `sin` averaging; an under-calibrated
  chance band that briefly mislabeled finite-sample AUC noise as a crossing). Fixes = fixed-
  frequency quadrature reads + a shuffle-null-calibrated gate (the correct instruments).

---

## 7. Files (all under `phase6/fold_audit/`)
| File | Role |
|---|---|
| `construction.py` | Exp 50.14 construction, verbatim (single source of truth) |
| `no_smuggle_gate.py` | the reusable no-smuggle gate + 3 self-test O's |
| `fold_audit.py` | the Stage 1 audit driver (sections a-d) |
| `gen_report.py` | regenerates this report from the result JSON |
| `fold_audit_result.json` | full numeric results + all seeds (deterministic) |
| `output_fold_audit.txt` | captured console log of the run |
| `REPORT_FOLD_AUDIT.md` | this report |

Reproduce: `..\..\..\..\..\.venv\Scripts\python.exe fold_audit.py` (master_seed
44060611; ~221s), then `python gen_report.py`.
