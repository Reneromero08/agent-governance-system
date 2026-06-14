# Exp 50 Substrate Mechanism Definition

**Date:** 2026-06-14.
**Status:** EXP50_SUBSTRATE_MECHANISM_DEFINITION: PUBLIC_VERIFY_PATH_BLOCKED__MECHANISM_DEFINITION_REQUIRED

---

## Executive Summary

Public `verify(x)` cannot select orientation because it is fold-even -- it accepts
both `d` and `N-d` identically. The candidate value `a = min(d, N-d)` is public
and recoverable at AUC 1.000. Full `d` recovery requires the orientation bit
`1[d < N/2]`, which is information-theoretically absent from the public cosine
channel (Phase 6, all tracks). Any valid substrate mechanism must access, preserve,
or amplify fold-odd information through substrate dynamics, not through verifier
enumeration. This document defines the criteria such a mechanism must satisfy,
the classes of mechanism that could theoretically qualify, and the minimum
experimental screen required before any L4 recovery attempt.

---

## Problem Restatement

| Element | Definition |
|---|---|
| Public oracle | `{(k_i, b_i)}` with `E[b_i] = cos(2*pi*k_i*d/N)`, `N = 2^n` |
| Hidden secret | `d` drawn from `[1, N)` excluding `N/2` (`construction.py` line 71-77) |
| Fold pair | `{d, N-d}` -- produce identical public data |
| Candidate value | `a = min(d, N-d)` -- public, recoverable at AUC 1.000 |
| Orientation bit | `1[d < N/2]` -- distinguishes `d` from `N-d` |
| verify(x) | Accepts both `d` and `N-d` (fold-even). Cannot discriminate. |
| Forward scan on verify | Finds `a`, not `d`. O(N) cost. Blocked (L4 audit). |

**Why public verifier iteration cannot recover full d:** `verify(x)` is a function
of `cos(2*pi*k*x/N)`, which is even in `x`. Therefore `verify(d) == verify(N-d)`
for every instance. A loop whose only branch condition is `verify(x)` can stop at
the first accepting value in its scan domain. If the domain is `[1, N/2)`, it
finds `a`, not `d`. If the domain is `[1, N)`, it finds either `d` or `N-d`
depending on scan order, neither of which is distinguishable. The verifier
provides zero bits of orientation information.

---

## Invalid Mechanisms

The following are rejected and must not be presented as substrate mechanisms:

1. **Sequential verify scan.** `for x in 0..N: if verify(x): return x`. O(N) search.
2. **SHA-wrapped forward scan.** Sequential verify inside tape lifecycle. Ceremonial.
3. **Restricted-domain recovery of a.** Returns `min(d, N-d)`. Candidate-value, not orientation.
4. **Preseeded hidden d.** Tape initialized with `d`. Hidden d in runtime. Smuggle.
5. **Candidate label phase/sign encoding.** Manual assignment of observable to `c0`/`c1`. Smuggle.
6. **Post-hoc seed selection.** Run many seeds, pick the one that "worked." HARKing.
7. **Candidate-value readout presented as orientation.** Claiming `a` recovery = `d` recovery.
8. **Any map whose only branch condition is fold-even verify(x).** The verifier itself carries no orientation.

---

## Valid Mechanism Criteria

A valid substrate mechanism for orientation recovery must satisfy ALL of:

| # | Criterion |
|---|---|
| 1 | No hidden `d` in the runtime path. |
| 2 | No true/false labels in the runtime path. |
| 3 | No manual assignment of phase/sign/orientation to candidate labels. |
| 4 | No sequential enumeration of candidates. |
| 5 | No post-hoc winner selection across seeds. |
| 6 | State is tape-resident or physically substrate-resident (not a C register). |
| 7 | Update is reversible or explicitly catalytic (tape restored). |
| 8 | Observable is predeclared before scoring. |
| 9 | A fold-odd observable exists that is NOT a function of fold-even verify(x). |
| 10 | Behavior beats forward scan baseline under scaling (n increasing). |
| 11 | Controls: same-candidate null, dummy null, label-swap null, shuffle null95, wrong-restore, replay. |

---

## Possible Mechanism Classes

These are conceptual categories. None are claimed to exist or to be implementable.

### A. Physical Relaxation Mechanism
A substrate state (thermal, cache, PDN, metastable) relaxes into one basin
over the fold pair due to a fold-odd physical asymmetry. The asymmetry is not
present in the mathematical verifier but emerges from the hardware's response
to different intermediate operand values `a*k_j` vs `(N-a)*k_j`.

**Requires:** A measurable physical degree of freedom that responds to the
sign of the intermediate computation, not just its cosine output. Track A
tested PDN/timing and found null. Other degrees of freedom (cache coherence,
DRAM row-buffer, thermal gradient) remain untested.

### B. Reversible Echo Mechanism
A reversible/catalytic loop `U_candidate → U_candidate_dagger` preserves
path-dependent information in a physical degree of freedom while restoring
the architectural tape state. The echo residue differs for `a` vs `N-a`
because the intermediate write patterns differ.

**Track F reference:** Weak, seed-dependent candidate-value hint found in
Hamming-weight accumulation. Not orientation. Not physical memory echo.
Requires genuine physical memory subsystem modeling.

### C. Analog Phase-Lane Mechanism
Fold-odd orientation appears as a phase, timing, or frequency-domain signal
under no-smuggle controls. The observable is not `verify(x)` but a physical
lock-in measurement at a known carrier frequency.

**Track B reference:** I/Q receiver separates candidates (Q channel detects
candidate-value) but Q_diff is always positive -- orientation-blind. Requires
a phase observable whose sign flips under the fold.

### D. CTC / Fixed-Point Oracle Mechanism
A theoretical fixed-point substrate selects a self-consistent solution without
forward enumeration. P^CTC = PSPACE (Aaronson-Watrous). Not implementable in
classical C on Phenom II.

**Exp 49.14 reference:** The handoff identified this as the untested lever.
Remains a theoretical complexity-class claim, not a hardware experiment.

### E. Null Mechanism
No fold-odd substrate observable exists within the current constraints.
The boundary holds. This is the consistent finding of Phase 6 and the L4 audit.

---

## Minimum Testable Mechanism Template

For any proposed mechanism, the following must be specified BEFORE
implementation:

| Field | Required Content |
|---|---|
| Mechanism name | Unique identifier |
| State representation | Where does the state live? (tape bytes, cache lines, thermal map, etc.) |
| Physical carrier | What physical degree of freedom carries the signal? |
| Fold-odd source hypothesis | Where does fold asymmetry enter? Must be public computation geometry. |
| Public inputs | k, b, N, candidate values only |
| Update rule | How does state evolve per cycle? Must be reversible or catalytic. |
| Observable | What is measured? Must be predeclared. |
| Stopping rule | When does the loop terminate? |
| Scaling prediction | How does cost/SNR scale with n? |
| Positive control | How is the detector proven live for this mechanism class? |
| Negative controls | Same-candidate, dummy, label-swap, shuffle, wrong-restore, replay |
| Null model | What outcome proves the mechanism does NOT work? |
| Failure modes | What artifacts could produce a false positive? |
| Claim ceiling | Maximum claim level if successful |

---

## Phenom II Feasibility Audit

**Can test:**
- Tape lifecycle (SHA-256, deterministic replay) -- L2 proven.
- Timing, PDN, cache, thermal observables -- T300 and Track A hardware paths exist.
- Deterministic replay under seed control.
- Hardware-dependent weak effects (if SNR permits).

**Cannot honestly claim from C alone:**
- CTC behavior or nonclassical fixed-point oracle.
- Orientation recovery from fold-even verify alone (L4 audit).
- Substrate crossing without a pre-existing fold-odd observable.
- That the Phenom II "proves" any substrate impossibility.

---

## Next Valid Experiment Shape

The next valid experiment is NOT L4 recovery. It is:

**L4A: Mechanism Candidate Screen**

Goal: Search for any predeclared fold-odd substrate observable under public
oracle workloads, without claiming orientation recovery.

Protocol:
1. Propose a mechanism from classes A, B, or C (class D is theoretical; class E is null).
2. Complete the mechanism template above.
3. Build a C probe that measures the proposed observable on Phenom II.
4. Run same-candidate, dummy, label-swap, and shuffle controls.
5. If the observable shows candidate-value separation (c0 vs c1) above null95
   with controls passing, the mechanism survives the screen.
6. If no observable survives, L4 remains blocked.

**Do NOT attempt orientation recovery until a mechanism survives this screen.**

---

## Claim Ledger

| Stage | Claim Level |
|---|---|
| Mechanism definition (this document) | L0 (conceptual) |
| Mechanism candidate screen (C probe, controls) | L2-L3 (hardware-measured) |
| Orientation recovery (no-smuggle, controls pass) | L4 |
| Multi-seed / multi-session | L5 |
| Independent reproduction | L6 |

---

## Roadmap Update

```
[>] Next frontier: define and screen substrate mechanisms for a predeclared
    fold-odd observable. Do not attempt L4 recovery until a mechanism
    survives controls.
```
