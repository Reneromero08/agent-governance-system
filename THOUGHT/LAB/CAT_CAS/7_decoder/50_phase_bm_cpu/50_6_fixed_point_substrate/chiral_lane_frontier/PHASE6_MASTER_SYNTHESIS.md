# Phase 6 Master Synthesis Report

**Phase:** Phase 6 -- Chiral Lane Frontier (Exp 50, bare-metal CPU substrate push).
**Status:** BOUNDARY MAPPED. HANDOFF TO SUBSTRATE FRONTIER.
**Date:** 2026-06-14.
**Claim ceiling:** L4 (invariant survives calibrated nulls on Phenom II hardware).

---

## 0. One-Sentence Result

Phase 6 confirmed through five gates, two hardware-negative tracks, two mathematical-reference tracks, and three rejected label-smuggle designs that the Exp 50.14 public cosine-only oracle does not generate a fold-odd physical carrier: candidate-value structure may be visible in some transforms, but the orientation bit `1[d < N/2]` is information-theoretically absent from the public data and has not been recovered by any no-smuggle measurement on Phenom II hardware.

---

## 1. Complete Ledger

### Sprint 0: Detector Spine

| Gate | Status | Claim | Key Result | Commit |
|---|---|---|---|---|
| E5/E1 (Oracle Integrity) | PASS | L4 | Published bits bitwise-identical for d and N-d. No PRNG leakage. No orientation channel. | `33d2b776` (Exp 50) |
| Track Z (Orientation Conservation) | PASS | L4 | Public schedule SHA-256 invariant under fold. Candidate labels blinded. | `54bce282` |
| Track 0 (Transfer Function) | PASS | L4 | Detector threshold binary: eps=0→null, eps>0→trivial. MDE not SNR-limited, sampling-limited. | `efe6f340` |
| Track B (I/Q Receiver) | PASS | L4 | I/Q receiver live. Q channel separates candidates (candidate-value) but Q_diff always positive → orientation-blind. | `efe6f340` |
| Track I (Topology Map) | ROUTE_SELECTED | L4 | Route 4:5 confirmed from T300 measured data (6/6 seeds). Route 2:3 partial (2/6). 10/12 routes deferred. | `62d3428d` |

### Sprint 1: Primary Public Lane Attempt

| Track | Status | Claim | Key Result | Commit |
|---|---|---|---|---|
| Track A-Seq (Sequential PDN) | NEGATIVE | L4 | No measurable PDN differential. c0=180.041, c1=180.040, diff=0.0009. p=0.72. | `e58f2b77` |
| Track A-Full (Simultaneous Dual-Lane) | NEGATIVE | L3 | All 4 modes identical within 0.001. Hidden positive failed. Detector not live for integer multiply load. | `e58f2b77` |
| Track A-Lockin (Corrected Lock-In) | NEGATIVE | L4 | 12/12 controls. c0=0.2475, c1=0.2390, diff=3.4%, p=0.093. AUC 0.587 < null95 0.612. | `4b31715e` |
| **Track A Final Closure** | **CLOSED** | **L4** | Three architectures, all negative on Phenom II route 4:5. Integer multiply operand asymmetry below measurement resolution. | `4b31715e` |

### Sprint 2: Phase-Geometry Attacks

| Track | Status | Claim | Key Result | Commit |
|---|---|---|---|---|
| Track F (Candidate HW Accumulation) | PARTIAL | L3 | Weak seed-dependent candidate-value hint. Mean AUC 0.514 across 8 seeds. Orientation FAIL_CHANCE (AUC 0.486). delta=0. NOT a physical Loschmidt echo. | `6e284cfd` |
| Track D (Commutator Lane) | NEGATIVE | L3 | Multi-seed AUC 0.516 (n=8/10/12), below null95 ~0.540. Same-candidate and dummy exactly 0.500. | `45bf2f01` |
| Track C (Chiral QFT) | REJECTED | -- | Roadmap Sec 10 lines 783-788: manual ROL→c0, ROR→c1 label assignment. Needs redesign. | -- |
| Track E (Geometric Phase) | REJECTED | -- | Roadmap Sec 12 lines 949-955: manual cw→c0, ccw→c1 label assignment. Needs redesign. | -- |

---

## 2. The Wall

The public oracle data is `{(k_i, b_i)}` with `E[b_i] = cos(2*pi*k_i*d/N)`. Since `cos(theta) = cos(-theta)`, the public data is identical for hidden secret `d` and its fold `N-d`. This is information-theoretic: `P(D | d) = P(D | N-d)` pointwise on every realization. Therefore:

- **Candidate value `a = min(d, N-d)`** may be recoverable from some public transforms (the fold magnitude). This has been observed weakly in Tracks B, F, and A-Lockin at some seeds.
- **Orientation `1[d < N/2]`** is not present in the public data. No no-smuggle track has recovered it. Every apparent signal was either a statistical artifact (washout across seeds), a detector-insensitivity artifact (noise floor), or a manual label encoding (smuggle).

**Candidate-value coupling is not orientation coupling.** The boundary between them is the structural boundary of the cosine-only oracle.

---

## 3. Evidence Chain

```
E5/E1 → Oracle fold-symmetric at implementation level. No PRNG/float/metadata leaks.
   ↓
Track Z → Public schedule invariant under fold. Candidate blinding enforced.
   ↓
Track 0 → Detector transfer function binary. eps>0 trivial, eps=0 null.
   ↓
Track B → I/Q receiver live. Q channel reads candidate-value, never orientation.
   ↓
Track I → Phenom route 4:5 confirmed. Full topology sweep deferred but prior validated.
   ↓
Track A → Real Phenom II hardware. Three architectures (DC, simultaneous, lock-in).
          12/12 controls on final run. All three negative. No measurable PDN differential.
   ↓
Track D → Commutator mathematical reference. Multi-seed AUC 0.516, below null95.
          No candidate-value signal from noncommutative execution order.
   ↓
Track F → Hamming-weight accumulation reference. Weak seed-dependent hint.
          Orientation FAIL_CHANCE per no-smuggle gate.
   ↓
Tracks C/E → Rejected at staging. Manual label encoding (ROL/ROR, cw/ccw).
   ↓
Synthesis → All no-smuggle tracks converged on the same boundary shape.
            The remaining frontier is a substrate question, not another track.
```

---

## 4. Recurring Pattern

| Classification | Tracks |
|---|---|
| **Null (no measurable signal)** | A-Seq, A-Full, A-Lockin, D |
| **Weak candidate-value hint (seed-dependent, below threshold at multi-seed)** | F |
| **Candidate-value coupled, NOT orientation coupled** | B, all of Track A, F |
| **Label-smuggle rejected** | A-phase-encoding, C, E |
| **Detector/method limitation** | A-Full (popcount below PDN threshold), A-correction (operand HW below noise) |
| **Mathematical reference only** | D, F |
| **Hardware negative (Phenom II)** | A-Seq, A-Full, A-Lockin |

---

## 5. Claims That Are Supported

1. **The Exp 50.14 public cosine-only oracle is fold-symmetric at the implementation level** (E5/E1). Published bits are bitwise-identical for `d` and `N-d` at shared RNG state. No PRNG seed depends on `d`. No float rounding reaches the published bits. The boundary is information-theoretic, not a code bug.

2. **No no-smuggle measurement on Phenom II hardware has recovered the orientation bit** from public data. Three Track A architectures, simultaneous and sequential, two different drive mechanisms, all with calibrated nulls -- none produce a statistically significant orientation signal.

3. **Weak candidate-value separation is occasionally observable** (c0 vs c1 lock-in magnitude differences of ~3% in Track A-Lockin at some seeds; Hamming-weight accumulation echo differences in Track F at some seeds). These signals are seed-dependent, do not survive multi-seed robustness checks, and do not encode orientation.

4. **Manual label encoding is the most common design failure mode.** Three proposed tracks (A-phase-encoding, C, E) assigned operation/direction/phase to candidate labels manually rather than deriving them from public computation geometry. All were rejected at audit.

5. **The Phenom II cross-core PDN coupling is LIVE for alu_burst + lock-in measurement** (T300 preflight SNR 13-38, Track A-Lockin preflight ratio 25x, off-tone selectivity 9.3x). The detector works. The signal absence is not a detector failure -- it is a workload-class limitation.

---

## 6. Claims That Are NOT Supported

1. **"No hardware can ever carry a candidate-value or orientation signal."** All hardware measurements are on a single Phenom II X6 1090T. A different CPU, measurement frequency, or drive architecture might produce different results.

2. **"PDN coupling is absent."** The T300 slot2 proved PDN coupling works for alu_burst + lock-in at audio frequencies. The integer multiply operand asymmetry does not excite it measurably.

3. **"All possible no-smuggle designs have been tested."** Only the designs in the Phase 6 roadmap were tested or audited. Other public transforms may exist.

4. **"The boundary is proven mathematically."** The boundary is a measured property of the Exp 50.14 construction on Phenom II hardware. It is not a mathematical proof that no oracle or platform could produce a different result.

5. **"The substrate (the Exp 50 Phase 6 substrate frontier, catalytic/reversible/CTC) cannot cross the boundary."** Phase 6 measured the FORWARD boundary. Whether a reversible, catalytic, or CTC substrate can cross it is a hardware question for Exp 50 (this experiment), not a Phase 6 conclusion.

---

## 7. The Substrate Handoff

Phase 6 was named "the frontier" for a reason: it was designed to find the boundary between what forward public-data measurement can do and what requires a substrate event.

The boundary has been located. The remaining question is not "can we find another public transform that reads orientation?" -- every no-smuggle transform tested has failed that question. The remaining question is:

**Can a catalytic/reversible/CTC substrate preserve path information through architectural restoration in a way that forward transforms cannot?**

This is the handoff that Exp 50.14 made to the Exp 50 substrate frontier: `d` is the unique fixed point of a public map `f(x) = x if verify(x) else (x+1) mod N`. Forward, finding it is `2^n`. On a reversible/CTC fixed-point substrate, it is `poly`. Phase 6 confirmed that forward transforms on a classical (non-catalytic) Phenom II do not reach that fixed point. The test of whether a reversible substrate does is hardware work (the Phenom, running catalytic), not another Python track.

Phase 6's role in this handoff is now complete: it provided the detector calibration (Tracks 0, B, I), confirmed the route (Track I, route 4:5), built the no-smuggle gate (E5/E1, Track Z, reusable `no_smuggle_gate.py`), and measured the forward boundary across multiple architectures (Track A). The target is mapped. The fence is located. The crossing, if it exists, is a substrate event.

---

## 8. Track C/E Disposition

Both tracks are rejected in their current roadmap form (manual label encoding). Public-derived redesigns are possible:

- Track C: derive ROL/ROR from `candidate * k_j mod 2` (LSB parity of per-step product)
- Track E: derive loop direction from `parity(candidate * K_mid mod N)` (cumulative sum parity)

However, even if redesigned correctly, both will likely collapse to the same candidate-value-coupled-not-orientation-coupled pattern. The candidate-value asymmetry they would measure is structurally identical to what Tracks A, B, D, and F already tested: the difference between `a*k` and `(N-a)*k` intermediate values. The orientation bit would remain inaccessible through the cosine channel.

**Recommendation: DEFER.** Do not build Track C or E under the current Phase 6 charter. If the substrate demonstrates a crossing, these tracks may become relevant as target generators. Until then, they are redundant with the already-completed A/D/F measurements.

---

## 9. Next Roadmap State

```
## Phase 6 Status: BOUNDARY MAPPED. HANDOFF TO SUBSTRATE FRONTIER.

Sprint 0: COMPLETE (5/5).  Sprint 1: COMPLETE (1/1, negative).
Sprint 2: PARTIAL (D negative, F weak/L3, C/E rejected).

All no-smuggle tracks executed or file-audited under the Exp 50.14 public-data
oracle either produce null, weak candidate-value-only signals, or fail
orientation recovery. The boundary is measured and confirmed.

Remaining open: substrate-level crossing (Exp 50 Phase 6, catalytic/reversible/
CTC). Forward public-data measurement alone does not generate a fold-odd lane.
The crossing, if it exists, is a substrate event.

Tracks C/E: DEFERRED pending substrate charter.
```

---

## 10. Files

This report: `PHASE6_MASTER_SYNTHESIS.md`. All referenced track reports, results, and code live under `chiral_lane_frontier/` subdirectories. All commits referenced above.
