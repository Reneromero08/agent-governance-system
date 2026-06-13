# Stage 3 - Generator Audit + No-Smuggle Gate Hardening

Phase 6 quadrature-crossing campaign. Claim ceiling L4-5. ASCII only. All RNGs seeded
(master_seed = 44060611); per-cell seeds derived deterministically and recorded in the
JSON sidecars. Foreground only, n in {8, 10}, every computation < 60 s.

Real construction audited: THOUGHT/LAB/CAT_CAS/49_the_decoder/49_14_reversible_substrate/49_14_substrate.py
(functions coset_samples, make_verify, f_map, forward_find_fixedpoint), via the
verbatim reimplementation phase6/fold_audit/construction.py. Instruments reused
verbatim: phase6/fold_audit/no_smuggle_gate.py, stage3/candidates.py.

## Bedrock prior (given)

The published data is {(k_i, b_i)} with b_i a binary sign in {-1,+1} and
E[b_i] = cos(2*pi*k_i*d/N), N = 2^n, secret d in [1, N/2). Orientation
o = 1[d < N/2] distinguishes d from N-d. The consult proved: if b_i is a pure
binary sign with the cos-mean as its WHOLE conditional law, then P(D|d) = P(D|N-d)
pointwise on every realization, so I(o ; T(D)) = 0 for ANY transform T at any compute.
The only escape: if the ACTUAL construction exposes something richer/asymmetric than that
cos-mean. This audit hunts that, on the real code.

---

## (A) Per-quantity generator audit

Every quantity the real construction makes public, tested for ODD dependence on d
(anything not invariant under the fold sigma: d <-> N-d). Script:
generator_audit.py -> generator_audit_result.json, output_generator_audit.txt.

| # | Public quantity | Source | Odd dependence? | Leak magnitude | Classification |
|---|---|---|---|---|---|
| Q1 | published bit b_i in {-1,+1} | coset_samples | **NO** | pure binary; unique={-1,+1}; fold moment-gaps == same-d redraw noise (ratio ~0.8-1.1x) | airtight: whole law = cos-mean, even in d |
| Q2 | frequencies k_i | coset_samples | **NO** | k drawn before d is touched; identical for d and N-d at fixed RNG state | orbit-only (independent of d) |
| Q3 | sample count M | M_for(n) | **NO** | M = max(4*ceil(sqrt N), 48n); pure function of n | orbit-only (no d) |
| Q4 | score / threshold / accept | score, make_verify | **NO** | score(d)-score(N-d) on same (k,b) is sub-1e-6 float; accept-set {d,N-d} symmetric; thresh=M/4 (no d) | orbit-only |
| Q5 | verify map make_verify | make_verify | **NO** | verify(x) built from (k,b,N) only; truth invariant to relabel d<->N-d; the [1,N/2) pin is in the SEARCH, not the published data | orbit-only |
| Q6 | float code-path of cos(2*pi*k*d/N) | coset_samples arg | **NO** (published) | intermediate cos low bits differ ~97% (sub-ULP, ~1e-13); **published bit diffs = 0** | float artifact of a NON-published intermediate (see below) |
| Q7 | sample order / PRNG seed | coset_samples | **NO** | order preserved; no default_rng(d); published bits b bitwise-identical for d vs N-d at shared RNG | orbit-only |

**Verdict (Part A): ORBIT_ONLY_PUBLIC_INTERFACE_BEDROCK_APPLIES.** No odd/sin-dependence
in ANY published quantity. The real construction public interface {(k,b)} (and the
verify map, score, threshold, M, order, seed, and float code-path) is a pure function of
the orbit {d, N-d}. The bedrock impossibility P(D|d)=P(D|N-d) applies WITHOUT caveat.

### The three initial flags, diagnosed and closed (skeptic pass)

The first audit pass (naive nulls) flagged Q1, Q6, Q7. Treated with maximum skepticism,
all three are **test-harness artifacts, not leaks** (a found "leak" was, as predicted, a
bug in the probe -- here in MY probe, not the construction):

- **Q6 / "AUC 0.92 on orientation"** -- the dangerous-looking number. It came from reading
  the LOW FLOAT BITS of the INTERMEDIATE cos(2*pi*k*d/N), which is a direct function of
  the HIDDEN d and is **never published** (the published object is the binary bit b,
  whose float64 holds only the two patterns for +-1.0 -> zero information). Control proof
  (in the JSON): the same low-bit classifier scores **parity-of-d at AUC = 1.000** and a
  **random label at ~0.51**. It classifies any smooth function of d because the bits
  encode d itself. That is the textbook smuggle the gate exists to catch, not a property
  of the construction. Decisive closure: published_bit_b_differences_d_vs_Nd = 0 -- the
  sub-ULP rounding never crosses a uniform threshold, so the published data is bitwise
  identical for d and N-d.
- **Q1 / "moment asymmetry ~0.05"** -- the apparent fold-asymmetry in mean/var/m3/m4 is
  **equal in scale to the gap between two independent redraws at the SAME d** (the correct
  paired null): ratio fold-gap / same-d-null ~ 0.8-1.1x. Pure sampling noise. The bit is
  exactly {-1,+1} and cos(2*pi*k*d/N) = cos(2*pi*k*(N-d)/N) up to ~1e-14 that never
  reaches b.
- **Q7 / "accept-prob not bitwise identical"** -- the intermediate p = (1+cos)/2 differs
  by ~1e-14 (even cos, different rounded argument), but the PUBLISHED bits are
  bitwise-identical at shared RNG and the order is preserved. Verdict hinges on the
  published object only.

Classification of all three: **(ii) NOT a leak; an artifact of probing a non-published
intermediate / using the wrong null.** Audit thresholds were corrected to the paired-null
and published-object criteria; the corrected audit reports all seven quantities even.

---

## (B) Hardened gate: the random-private-fold test

New instrument: hardened_gate.py (wraps no_smuggle_gate.gate verbatim, adds the test).

### Spec

Per instance, draw the orbit representative d0 from the lower half [1, N/2), flip a
PRIVATE coin r in {0,1}, and publish data for the lower half (d_used = d0, r=0) or the
upper half (d_used = N-d0, r=1) -- BOTH from the SAME orbit {d0, N-d0}. The candidate
O must predict r from PUBLIC data only. Pinning d0 to the representative makes r
EXACTLY the orientation of the published secret. Because the two branches have provably
identical public laws (P(D|d0)=P(D|N-d0)), an honest public-only O scores AUC = 0.5 in
expectation; any lift above the calibrated shuffle null localizes a side-channel.

### Verdict logic (two axes + effect-size margin)

- AXIS 1 (exact, zero-false-positive): byte-equal d-invariance -- O(inst) vs
  O(folded_instance(inst)). delta > 0 => the candidate reads the private fold / secret.
- AXIS 2 (statistical): random-fold held-out AUC vs per-dataset shuffle-null 95th pct,
  **plus a 0.05 effect-size margin**. The margin is justified by a 12-seed sweep: honest
  public-only transforms have (rf_auc - null95) <= 0.016, while true orientation/sin
  smugglers sit at margin ~0.45 -- 0.05 separates them with large headroom and removes the
  ~5% borderline false-positive rate of a bare percentile threshold. The same margin is
  applied to the orientation axis (it was the source of a spurious crossing; see Part C).
- FAIL_SMUGGLE if delta > 0 OR random-fold AUC clears its null by the margin.
  PASS_CROSSING if orientation lifts by the margin with both other axes clean.
  FAIL_CHANCE otherwise.

Axis-power demonstration (random-fold AUC, full strength): a candidate that reads the
orientation bit directly, or the hidden sin channel, scores random-fold **AUC = 1.000**
at both n; an honest even transform stays at chance (margin <= 0.016). The test has teeth.

### Phase-retrieval smuggle demonstration (the mechanism)

O_gerchberg_saxton_SMUGGLE (candidates.py): alternating-projection phase retrieval WITH
the support constraint d in [1, N/2). To impose that support, the iteration must know
which half d lives on -- the ONLY source of that bit is the hidden secret in the instance:

    d = inst[d-key]                    # <-- reads the secret
    lower_half = int(d % N) < (N / 2)  # <-- this IS the orientation bit
    support[N//2:] = 0  (or [:N//2]=0) # <-- half-plane mask = range prior by fiat

The support constraint injects the range prior = orientation by fiat. Under the fold
d -> N-d the mask flips, so O output changes -> exact d-invariance audit yields
delta = 1.98 > 0. **Hardened gate flags it FAIL_SMUGGLE at n=8 and n=10**
(reason: exact_d_invariance_delta>0). The honest symmetric-support variant
(O_gerchberg_saxton_PUBLIC) keeps a real-even fixed point, delta = 0, random-fold at
chance -> FAIL_CHANCE. Self-test: all 10 hardened-gate cells match expectation
(hardened_gate_result.json).

---

## (C) Bounded battery confirmation (n in {8, 10})

Every candidates.py transform through the HARDENED gate. Script: battery_confirm.py
-> battery_confirm_result.json. n_instances = 300, n_shuffles = 25.

| Candidate | n=8 verdict | n=10 verdict | mechanism |
|---|---|---|---|
| 1 Hilbert / analytic-signal | FAIL_CHANCE | FAIL_CHANCE | Hilbert of the even binned cos-spectrum -> odd sequence with no absolute d-sign reference |
| 2 Double-angle ladder | FAIL_CHANCE | FAIL_CHANCE | double-angle gives only abs(sin) + even consistency residuals; sin(2t) needs an absent odd channel |
| 3 Bispectrum | FAIL_CHANCE | FAIL_CHANCE | bispectrum of a real-even spectrum is real (phase 0/pi); odd orientation sign not encoded |
| 4 Autocorrelation / x->N-x asymmetry | FAIL_CHANCE | FAIL_CHANCE | score(x) even in x (peaks at d and N-d); antisymmetric part ~0 by construction |
| 5a Gerchberg-Saxton PUBLIC (symmetric support) | FAIL_CHANCE | FAIL_CHANCE | symmetric support -> real-even fixed point; phase 0/pi, delta=0 |
| 5b Gerchberg-Saxton SMUGGLE (support = d half) | **FAIL_SMUGGLE** | **FAIL_SMUGGLE** | support mask reads the hidden secret half; flips under fold -> delta=1.98 |
| 6 Half-angle / Chebyshev sign-lift | FAIL_CHANCE | FAIL_CHANCE | Chebyshev/half-angle lift invariant under theta -> -theta (d -> N-d); branch sign unrecoverable |

**No PASS_CROSSING. All 14 cells match expectation.** This is the empirical echo of the
bedrock proof on the real construction: no transform of the public even data lifts
orientation; the designated smuggle is caught.

### One spurious crossing caught and closed (skeptic pass)

Before the orientation-axis margin was added, candidate 4 (autocorrelation) at n=10 tripped
PASS_CROSSING with orient_auc = 0.557 vs null95 = 0.555 -- a margin of **0.002**, with
random-fold AUC 0.469 (at chance) and delta = 0 (provably d-invariant). An 8-seed sweep
confirmed it oscillates around chance (margin range -0.136 .. +0.035, above_chance fires
3/8 by noise). This is exactly the spurious one-axis "crossing" the campaign must not claim;
the 0.05 effect-size margin on the orientation axis closes it. The gate is now robust to
finite-sample AUC variance on BOTH axes.

---

## Crisp verdict

**The REAL construction does NOT leak orientation anywhere.** Its entire public interface
{(k_i, b_i)} -- and the verify map, score, threshold, sample count M, sample order,
PRNG seeding, and the float code-path of the sampler -- is a provable PURE FUNCTION OF THE
ORBIT {d, N-d}. The published bit is a pure binary sign with E[b]=cos as its whole
conditional law, and cos(2*pi*k*d/N) is bitwise-identical to cos(2*pi*k*(N-d)/N) on the
actual sampler path up to a sub-ULP rounding that never reaches the published bit. The
bedrock impossibility I(o ; T(D)) = 0 therefore applies to this instance WITHOUT caveat.

Every apparent "leak" found in the audit was, on hard re-audit, **a bug in the probe**
(reading a non-published intermediate, or a bare-percentile threshold firing on
finite-sample noise) -- classification (ii), not a classical crossing. The no-smuggle gate
was hardened with the random-private-fold test (now demonstrated at AUC = 1.000 against true
smugglers and effect-size-margined against false positives on both axes) and correctly flags
the phase-retrieval support-constraint smuggle. The wall is not on the readout; it remains
on the substrate, as established by prior work.

## Files

Scripts (all under phase6/fold_audit/stage3/):
- generator_audit.py -- Part A, the per-quantity generator audit
- hardened_gate.py -- Part B, the random-private-fold test + hardened gate (importable)
- battery_confirm.py -- Part C, bounded battery through the hardened gate

Result sidecars (same dir): generator_audit_result.json, output_generator_audit.txt,
hardened_gate_result.json, battery_confirm_result.json.

Read-only inputs (untouched): 49_14_substrate.py, fold_audit/construction.py,
fold_audit/no_smuggle_gate.py, stage3/candidates.py.