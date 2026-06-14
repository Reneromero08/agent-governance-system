# Exp 50 L4A Non-Collapse Mechanism Screen Protocol

**Date:** 2026-06-14.
**Status:** EXP50_L4A_NON_COLLAPSE_SCREEN: ORBIT_STATE_PRESERVED__FOLD_ODD_RESIDUE_SCREEN__NO_RECOVERY_CLAIM

---

## Executive Summary

L4A screens mechanism candidates for a predeclared fold-odd substrate residue
while preserving the fold pair as an unresolved OrbitState. L4A does NOT recover
d. L4A does NOT choose between d and N-d. L4A does NOT call verify(x). L4A looks
only for an observable that survives fold-even cancellation under no-smuggle
controls. If no residue survives, the mechanism class is null. If a residue
survives, it is a CANDIDATE for later L4 orientation-recovery work.

---

## 1. Non-Collapse Doctrine Restatement

- The algorithm is dead.
- No scalar candidate is the primitive object.
- No verify(x) as branch selector.
- No AUC as primary truth metric.
- No early collapse to c0/c1.
- .holo stores the process-object (eigenbasis, orbit, path), not merely the answer.
- Measurement only at CollapseBoundary.
- Fold-even cancellation must precede any interpretation of residue.
- Tape restoration is hygiene, not substrate dynamics.

---

## 2. Object Definitions (L4A-specific roles)

| Object | L4A Role | May Contain | Must Not Contain | Collapse Risk |
|---|---|---|---|---|
| FoldPair | The unresolved orbit. Created from public a, N-a. | (a, N-a) as symmetric values. state=unresolved. | Label "true"/"false." Scalar d. Branch choice. | Selecting one branch for scoring. |
| OrbitState | The full system state during evolution. | FoldPair, PhaseRelation, PathHistory, TapeResidue, SubstrateMemory. | Scalar x. AUC score. Binary winner. | Computing c0 vs c1 statistics. |
| PhaseRelation | Relative phase between branches. | Complex ratio z = branch+/branch-. Phase angle. | Binary orientation bit. Scalar d. | Converting to sign(d-N/2). |
| PathHistory | Append-only trajectory log. | (step, operator, tape_delta, sub_delta) per step. | Cumulative score. Convergence metric. Best candidate label. | Converting to scalar feature vector. |
| TapeResidue | Physical tape state after XOR-restore. | Timing/voltage measurement of tape bytes. NOT byte values. | Architectural tape contents (SHA guarantees identity). | Reading tape bytes as signal. |
| SubstrateMemory | Non-architectural physical state. | Thermal, PDN, cache coherence, timing. | Architectural register or memory value. | Measuring arch state and calling it substrate. |
| CatalyticField | The medium in which OrbitState evolves. | XOR mask stream. Drive tone. Probe schedule. | Hidden d. Candidate label. Verify oracle. | Making field depend on branch truth. |
| OddResidue | What survives fold-even cancellation. | Physical observable antisymmetric under fold. | Noise without null calibration. | Claiming any nonzero = signal. |
| HoloContainer | The .holo file storing the full process-object. | OrbitState, evolution trace, cancellation transcript, controls. | Scalar answer as primary. Post-hoc selection. | Collapsing .holo to a scalar d. |
| CollapseBoundary | The explicit point where the orbit resolves. | InvariantExtract. Final measurement. Log write. | Post-boundary re-interpretation. Seed selection. | Re-scoring after boundary. |

---

## 3. Screen Target

L4A screens for a fold-odd residue R satisfying ALL of:

1. R changes sign or phase under branch conjugation (plus <-> minus swap).
2. R is not recoverable from public verify(x) (which is fold-even).
3. R is not the candidate-value magnitude a (which is public).
4. R survives PhaseCancel (fold-even component destructively interferes to zero).
5. R survives same-orbit null (FoldPair(a, a) -> R = 0).
6. R survives dummy-orbit null (FoldPair(42, 42) -> R = 0).
7. R survives label-swap null (swap branch roles -> R sign flips consistently).
8. R is predeclared in the .holo before measurement.
9. R magnitude exceeds shuffle null95.

R is NOT called "d." R is NOT called "the answer." R is NOT called "orientation recovery."
R is a MECHANISM CANDIDATE -- a physical observable carrying fold-odd structure.

---

## 4. Mechanism Candidate Classes

### A. Phase-Lane Residue
- **Hypothesis:** The relative timing/phase between two core-local walks carries an antisymmetric component that survives fold-even cancellation.
- **State object:** PhaseRelation.
- **Physical carrier:** Core-local TSC deltas, thread scheduling phase, APERF/MPERF ratio during coupled walk.
- **Fold-odd source:** The two branch walks (a*k_j vs (N-a)*k_j) produce different instruction scheduling and pipeline occupancy, creating a phase offset between the two sender cores.
- **Cancellation:** Lock-in phase difference between branch+ window and branch- window.
- **Measurement boundary:** After both walks complete and tape is restored.
- **Null controls:** Same-orbit (a, a). Dummy-orbit (42, 42). Label-swap. Phase-randomized input.
- **Collapse risks:** Measuring TSC jitter as phase signal. Using scalar timing mean instead of lock-in phase.

### B. PDN / Common-Mode Residue
- **Hypothesis:** The shared power rail carries an antisymmetric current signature from two simultaneously executing branch walks.
- **State object:** SubstrateMemory (PDN channel).
- **Physical carrier:** alu_burst-driven PDN current measured via victim ring-oscillator lock-in.
- **Fold-odd source:** The two branch walks use different operand magnitudes, producing different instantaneous current draw. The differential survives common-mode rejection.
- **Cancellation:** Lock-in Q-channel difference between branch+ and branch- drive windows.
- **Measurement boundary:** After both walks complete and tape is restored.
- **Null controls:** Same-orbit. Dummy-orbit. Label-swap. No-drive baseline. Off-tone.
- **Collapse risks:** Measuring DC current instead of lock-in. Confusing ambient VRM noise with signal.

### C. Cache / Coherence Residue
- **Hypothesis:** The L3 cache coherence state after two coupled branch walks retains an antisymmetric access pattern.
- **State object:** SubstrateMemory (cache channel).
- **Physical carrier:** L3 cache line state (MESI), coherence probe traffic, cache miss latency.
- **Fold-odd source:** The two branch walks access different cache sets due to different operand magnitudes, creating asymmetric coherence directory state.
- **Cancellation:** Cache miss latency delta between branch+ and branch- probe windows.
- **Measurement boundary:** After cache flush and restore.
- **Null controls:** Same-orbit. Dummy-orbit. Label-swap. Cache-flush baseline.
- **Collapse risks:** Measuring cache occupancy instead of coherence state. Phenom II unified L3 may not expose per-core coherence asymmetry.

### D. Path-History Residue
- **Hypothesis:** The branch predictor and speculative execution state after two coupled walks retains an antisymmetric training pattern.
- **State object:** PathHistory.
- **Physical carrier:** Branch predictor tables, BTB, return stack, indirect branch predictor.
- **Fold-odd source:** The two branch walks contain different sequences of taken/not-taken branches due to different intermediate operand parity.
- **Cancellation:** Branch misprediction rate delta between branch+ and branch- probe sequences.
- **Measurement boundary:** After both walks complete and predictor is probed with a standard sequence.
- **Null controls:** Same-orbit. Dummy-orbit. Label-swap. Predictor-flush baseline.
- **Collapse risks:** Branch predictor state is not directly readable on Phenom II. Indirect measurement via timing is noisy.

### E. Thermal / Metastability Residue
- **Hypothesis:** The die temperature distribution after two coupled walks retains an antisymmetric thermal footprint.
- **State object:** SubstrateMemory (thermal channel).
- **Physical carrier:** k10temp on-die thermal diode. Ring-oscillator period as temperature proxy.
- **Fold-odd source:** The two branch walks produce different thermal loads due to different switching activity (a*k vs (N-a)*k Hamming weight patterns).
- **Cancellation:** Temperature delta between branch+ and branch- windows.
- **Measurement boundary:** After thermal steady-state (seconds).
- **Null controls:** Same-orbit. Dummy-orbit. Label-swap. Thermal soak baseline.
- **Collapse risks:** Thermal time constant (seconds) is much slower than walk time (milliseconds). Ambient temperature drift dominates.

### F. Null Mechanism
- **Hypothesis:** No substrate primitive carries a measurable fold-odd residue under non-collapse controls.
- **State object:** NULL.
- **This is the expected outcome if all mechanism classes fail.** Constitutes a strong negative result, not a failure of the protocol.

---

## 5. .Holo Container Spec

A `.holo` file stores the non-collapse process-object. It is the canonical record.

**Must store:**
- OrbitState declaration (FoldPair, creation time, public hash).
- Branch relation (symmetric a, N-a. No "true"/"false").
- Phase/Path/Tape/Substrate traces (per-step, append-only).
- CatalyticField configuration (mask seed, drive tone, probe schedule).
- Cancellation transcript (method, before/after, residual magnitude).
- OddResidue declaration (predeclared observable, measured value, null thresholds).
- InvariantExtract result (sign or NULL).
- CollapseBoundary timestamp.
- Measurement metadata (hardware, temperature, P-state, seeds).
- Null-control links (same-orbit .holo, dummy .holo, label-swap .holo).
- Claim level.

**Must NOT store:**
- Scalar d as primary object.
- Candidate winner label.
- Hidden labels in any field.
- Post-hoc selected result across seeds.
- Collapsed answer without the process-object trace.

---

## 6. L4A Protocol Phases

| Phase | Operation | Gate |
|---|---|---|
| P0 | Predeclare orbit-state and residue hypothesis in .holo. | Hypothesis must name: mechanism class, physical carrier, fold-odd source, observable, null model. |
| P1 | Construct FoldPair(a, N-a) from public oracle. Both branches carried. No label assignment. | a recovered from public data only. |
| P2 | Initialize OrbitState with FoldPair + empty PhaseRelation, PathHistory, TapeResidue, SubstrateMemory. Write to .holo. | All containers empty. No branch selection. |
| P3 | Run CatalyticEvolution: coupled branch walks with shared field. Both branches evolve simultaneously. Tape XOR-mutated and SHA-verified each cycle. Substrate perturbed per field schedule. | No verify(x). No branch selection. SHA passes each cycle. |
| P4 | Restore public tape. SHA verify. TapeResidue measurement: physical probe of tape bytes (timing/voltage), NOT architectural read. | SHA matches. |
| P5 | Apply PhaseCancel(fold_even). Compute observable delta between branch+ and branch- windows. | Same-orbit control shows zero delta. |
| P6 | Extract predeclared OddResidue from cancellation residual. | Residue sign/magnitude recorded. Not interpreted yet. |
| P7 | Cross CollapseBoundary. Write InvariantExtract to .holo. No further operations permitted. | Boundary is final. |
| P8 | Run null controls: same-orbit, dummy-orbit, label-swap, shuffle, wrong-restore, replay, carrier-off, measurement-order-reversal. | All controls must pass for residue to survive. |
| P9 | Classify mechanism candidate: L4A_RESIDUE_CANDIDATE_FOUND or L4A_RESIDUE_NOT_FOUND. | Claim level L3. |

---

## 7. Control Suite

| # | Control | Purpose |
|---|---|---|
| C1 | Same-orbit null | FoldPair(a, a). Residue must be zero. |
| C2 | Dummy-orbit null | FoldPair(42, 42). Residue must be zero. |
| C3 | Label-swap null | Swap branch roles. Residue sign must flip. |
| C4 | Shuffle null95 | Randomize branch labels. AUC must be at chance. |
| C5 | Wrong-restore negative | Deliberate SHA mismatch. Run flagged invalid. |
| C6 | Replay determinism | Same seed -> same .holo trace. |
| C7 | Session/reboot repeat | Rerun after reboot. Same orbit -> same residue. |
| C8 | Carrier-off control | Disable the physical carrier (no PDN drive, no cache pressure). Residue must vanish. |
| C9 | Measurement-order reversal | Swap which branch is measured first. Residue sign must flip. |
| C10 | Post-hoc leakage audit | Verify no hidden d, no true/false labels, no seed selection in .holo. |

---

## 8. Success / Failure Verdicts

| Verdict | Meaning |
|---|---|
| L4A_RESIDUE_CANDIDATE_FOUND | A predeclared fold-odd residue survives all controls. Claim L3. |
| L4A_RESIDUE_NOT_FOUND | No residue survives controls in the tested mechanism class. Valid negative. |
| L4A_COLLAPSE_CONTAMINATION_FOUND | The protocol collapsed to scalar candidate comparison, verify(x), or AUC. Design rejected. |
| L4A_MEASUREMENT_INVALID | Controls failed. Noise floor exceeded. Run invalid. |
| L4A_NULL_CONTROL_FAILED | Same-orbit or dummy-orbit produced nonzero residue. Mechanism class rejected. |
| L4A_PROTOCOL_READY_NOT_RUN | Protocol defined. No hardware run. |
| L4A_NEEDS_ARCHITECTURE_REVISION | Protocol cannot be implemented with current Phenom II primitives. |

---

## 9. Claim Levels

| Level | Stage |
|---|---|
| L0 | Protocol defined (this document). |
| L1 | .holo schema complete and validated. |
| L2 | Simulator preserves OrbitState through all phases without collapse. |
| L3 | Phenom II primitive screen runs. Residue candidate survives controls. |
| L4A | Predeclared fold-odd residue confirmed across mechanism classes. |
| L4 | Residue used for orientation recovery under no-smuggle controls. |
| L5 | Multi-seed, multi-session, multi-n. |
| L6 | Independent reproduction. |

---

## 10. Median-Basin Fail Conditions

Reject the protocol immediately if:

| If the design... | Action |
|---|---|
| Starts with verify(x) | REJECT. Fold-even. |
| Creates candidate_0/candidate_1 labels | REJECT. Early collapse. |
| Reports AUC before residue definition | REJECT. Classification, not measurement. |
| Asks "which candidate wins" | REJECT. Branch selection. |
| Collapses .holo into scalar answer | REJECT. Process-object destroyed. |
| Treats SHA restore as substrate dynamics | REJECT. Hygiene only. |
| Uses a = min(d,N-d) as orientation | REJECT. Candidate value. |
| Calls a toy map frontier evidence | REJECT. Mechanical warmup. |
| Returns to backprop/standard ML defaults | REJECT. Not substrate. |

---

## 11. Phenom II Primitive Mapping

| Primitive | Hosts OrbitState? | Hosts PhaseRelation? | Hosts PathHistory? | Plausible OddResidue? | Collapse Trigger |
|---|---|---|---|---|---|
| L3 cache | YES (unified) | PARTIAL | YES | LOW | Reading cache tags as arch |
| PDN/common-mode | YES (victim ring-osc) | YES (lock-in phase) | NO | **HIGHEST** (T300 proven live) | DC averaging instead of lock-in |
| Core timing | PARTIAL | YES (TSC delta) | NO | MEDIUM | Jitter = signal |
| Branch predictor | NO (not readable) | NO | YES (implicit) | LOW | Cannot measure directly |
| Thermal drift | NO (too slow) | NO | PARTIAL | VERY LOW | Ambient dominates |
| Cache coherence | PARTIAL | NO | YES | LOW | Unified L3, no CCX split |
| Thread phase | YES | YES | NO | MEDIUM | OS scheduler noise |
| Tape buffer | YES (XOR reversible) | NO | YES | NO | Tape bytes identical after restore |

**Recommended first screen class: B (PDN / Common-Mode Residue).** T300 proved the measurement path is live (SNR 13-38). The lock-in Q-channel is the natural PhaseRelation carrier. The coupled branch walks map directly to two sender cores driving alu_burst at 200 Hz with opposite phase encoding derived from walk operands.

---

## 12. Roadmap Update

```
[>] L4A next: screen non-collapse substrate mechanisms for predeclared
    fold-odd residue using .holo OrbitState. Recommended first class:
    PDN / Common-Mode Residue (T300 path proven live).
    No L4 recovery until residue survives controls.
```
