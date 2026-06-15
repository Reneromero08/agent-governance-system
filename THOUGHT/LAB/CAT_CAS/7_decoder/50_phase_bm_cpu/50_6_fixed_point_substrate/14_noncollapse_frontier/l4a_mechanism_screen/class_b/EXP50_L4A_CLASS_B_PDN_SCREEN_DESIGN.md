# Exp 50 L4A Class B PDN/Common-Mode Screen Design

**Date:** 2026-06-14.
**Status:** EXP50_L4A_CLASS_B_PDN_SCREEN: DESIGN_ONLY__ORBIT_STATE_PRESERVED__PDN_FOLD_ODD_RESIDUE_HYPOTHESIS

---

## Executive Summary

Class B screens the PDN/common-mode physical carrier for a predeclared fold-odd
residue. The FoldPair {a, N-a} is preserved as an unresolved OrbitState. Two
sender cores simultaneously run coupled branch walks. The receiver measures the
combined PDN response via T300-proven alu_burst + lock-in. Fold-even cancellation
subtracts the two branch response windows. Any surviving residue is the Class B
candidate. No candidate_0/candidate_1 labels. No verify(x). No AUC. No d recovery.

---

## 1. Non-Collapse Guarantee

| # | Guarantee |
|---|---|
| G1 | No scalar candidate is the primitive. The primitive is the FoldPair. |
| G2 | No candidate_0 / candidate_1 labels in runtime. Branches are called branch+ and branch-. |
| G3 | Branch assignment to cores is deterministic from public data: branch+ = min(a, N-a) goes to core 4, branch- = max(a, N-a) goes to core 5. This is a magnitude ordering, not an orientation encoding. |
| G4 | No verify(x) at any step. |
| G5 | No branch selection before CollapseBoundary. |
| G6 | No d output. No orientation output. No AUC. |
| G7 | .holo stores the full process-object (OrbitState, evolution trace, cancellation transcript). |
| G8 | Measurement only after PhaseCancel. |
| G9 | Label-swap control verifies that branch assignment to cores is not orientation smuggling. |

---

## 2. Class B Hypothesis

Given an unresolved FoldPair orbit-state O = {branch+, branch-} with values
(a, N-a), a coupled catalytic PDN workload W_B running simultaneously on two
sender cores may produce a receiver lock-in response whose Q-channel component
survives fold-even cancellation, where:

- Fold-even components cancel under PhaseCancel (destructive interference of
  the common-mode cosine response from both branches).
- Any surviving fold-odd component R_B is predeclared as OddResidue.
- R_B is NOT the candidate value a (which is public).
- R_B is NOT a branch label.
- R_B is NOT a public verifier score.
- R_B is antisymmetric under branch conjugation: R_B(branch+, branch-) =
  -R_B(branch-, branch+).

---

## 3. OrbitState Construction

| Object | Value |
|---|---|
| FoldPair | {branch+: a, branch-: N-a} where a = min(d, N-d) recovered from public data. Both values are public. State = unresolved. |
| PhaseRelation | Empty at creation. Populated during evolution from lock-in Q-channel relative phase between the two branch response windows. |
| PathHistory | Append-only. Records per-step: (step_index, branch+_operand_magnitude, branch-_operand_magnitude, tape_delta_hash, pdn_drive_phase). |
| TapeResidue | Post-restore physical measurement of tape bytes. NOT architectural read. Purely diagnostic. |
| SubstrateMemory | PDN channel: lock-in I and Q at drive tone for branch+ window and branch- window separately. Recorded, not scored. |
| CatalyticField | XOR mask stream (seed = public_hash(FoldPair)). alu_burst drive tone = 200 Hz. Phase = 0 for both branches (the phase difference comes from operand asymmetry, not manual assignment). |
| HoloContainer | .holo file initialized with above objects. No scalar d. No candidate labels. |

**Why branch assignment to cores is not orientation smuggling:**
branch+ = a always. branch- = N-a always. This is a function of the public
magnitudes only. It does NOT depend on whether a = d or a = N-d. The label-swap
control (swap which core runs which branch) verifies this: if the residue sign
flips under core swap, the residue is coupled to the branch VALUE, not the core
identity. If it doesn't flip, it's a core asymmetry artifact.

---

## 4. Catalytic Workload W_B Design

| Step | Operation | Non-Collapse Property |
|---|---|---|
| 1 | Initialize tape from seed. Record SHA-256. | Hygiene. |
| 2 | XOR-encode FoldPair metadata into tape. | Public data only. |
| 3 | Sender core 4: execute branch+ walk. For each k_j, compute branch+ * k_j mod N. Drive alu_burst at 200 Hz. Walk intensity proportional to operand magnitude Hamming weight. | branch+ uses a. Public operand. No orientation encoding. |
| 4 | Sender core 5: execute branch- walk SIMULTANEOUSLY. For each k_j, compute branch- * k_j mod N. Drive alu_burst at 200 Hz. Same tone, same duty cycle. | branch- uses N-a. Public operand. No orientation encoding. |
| 5 | Receiver core 2: capture ring-oscillator timing for TWO windows. Window W+: during branch+ walk. Window W-: during branch- walk. | Windows matched in duration. Order randomized per trial. |
| 6 | Lock-in demodulate each window at 200 Hz. Record I+, Q+, I-, Q- in SubstrateMemory. | No scoring. Recording only. |
| 7 | XOR-decode tape. SHA-256 verify restoration. | Hygiene. Restoration must succeed. |
| 8 | Write PathHistory: per-step operand magnitudes, tape delta, PDN phase. | Append-only. Not scored. |

**Energy budget:** Both branches execute the same NUMBER of integer multiply
operations. The operand MAGNITUDES differ (a*k vs (N-a)*k), but the instruction
count and memory footprint are matched. Any PDN asymmetry is from operand
values, not from workload imbalance.

**No manual phase encoding:** Both branches drive at the same tone with phase 0.
The phase difference (if any) measured at the receiver arises from the physical
PDN response to different operand-driven switching activity, not from an
experimenter-assigned phase offset.

---

## 5. PDN Carrier Model

| Property | Value |
|---|---|
| Physical carrier | Shared package power-delivery network (PDN) between cores 4, 5 and victim core 2. |
| Excitation path | Register/L1-only alu_burst on cores 4 and 5. Current draw propagates through shared VRM/PDN. Victim ring-oscillator period modulated by IR drop. |
| Proven live | T300 slot2: SNR 13-38 for single-sender alu_burst at 200 Hz. Route 4:5 confirmed. |
| Common-mode (fold-even) component | The cosine response: both branches produce identical cosine outputs, so the PDN current from cosine-matched operations is identical. Cancels under PhaseCancel. |
| Fold-even component | The integer multiply operand magnitude: branch+ uses a, branch- uses N-a. These produce DIFFERENT integer multiplier switching activity. The PDN current difference is the candidate fold-odd component. |
| Expected null behavior | Same-orbit (a, a): both cores run identical walks -> PDN responses identical -> Q_diff = 0. Dummy-orbit (42, 42): same -> Q_diff = 0. |
| Measurement risk | The alu_burst itself dominates PDN current. Operand-dependent switching activity is a small modulation on top. The lock-in must resolve the Q-channel at the drive tone with sufficient SNR to detect the operand-dependent component. Track A-Lockin found this component to be at noise floor. |

---

## 6. Cancellation Plan

| Step | Operation |
|---|---|
| K1 | After both branch response windows (W+, W-) are recorded, compute Q_diff = Q+ - Q-. |
| K2 | The common-mode (cosine-driven) PDN component is identical in both windows and cancels. |
| K3 | The operand-dependent (multiply-driven) component differs between windows. If a fold-odd component exists, Q_diff != 0. |
| K4 | Verify cancellation against same-orbit control: FoldPair(a, a) must produce Q_diff = 0 within noise. |
| K5 | If same-orbit or dummy-orbit controls show nonzero Q_diff, the cancellation is invalid (noise or core asymmetry artifact). |
| K6 | Record cancellation transcript in .holo: Q+, Q-, Q_diff, Q_diff_same_orbit, Q_diff_dummy. |
| K7 | No branch is labeled "the winner." No scalar answer is extracted yet. |

---

## 7. CollapseBoundary Measurement

After cancellation, the CollapseBoundary measurement extracts:

| Observable | Meaning |
|---|---|
| Q_diff_magnitude | Absolute magnitude of Q-channel difference. |
| Q_diff_sign_consistency | Fraction of trials where Q_diff has the same sign. |
| Q_diff_vs_null | Whether Q_diff exceeds same-orbit and dummy-orbit null bounds. |
| Label_swap_consistency | Whether Q_diff sign flips when branch-to-core assignment is swapped. |
| Carrier_specificity | Whether Q_diff vanishes when PDN drive is disabled (carrier-off control). |

**Forbidden outputs:** "winning candidate," "recovered d," "true orientation,"
"scalar verifier score as primary result."

---

## 8. Control Suite

| # | Control | Purpose | Expected Result |
|---|---|---|---|
| C1 | Same-orbit null | FoldPair(a, a). Both cores run identical walk. | Q_diff = 0. |
| C2 | Dummy-orbit null | FoldPair(42, 42). | Q_diff = 0. |
| C3 | Branch-label swap | Swap which core runs branch+ vs branch-. | Q_diff sign must flip. If not, artifact is core asymmetry, not fold-odd. |
| C4 | Phase-randomized orbit | Randomize which branch value is called branch+. | Q_diff sign tracks VALUE, not label. |
| C5 | Path-shuffled orbit | Shuffle k_j order. | Q_diff magnitude unchanged (operand distribution same). |
| C6 | Carrier-off control | Disable alu_burst drive (no PDN excitation). | Q_diff = 0. Proves signal is PDN-carried. |
| C7 | Measurement-order reversal | Measure W- first, then W+. | Q_diff sign must flip. |
| C8 | Wrong-restore negative | Deliberate SHA mismatch. | Run flagged invalid. |
| C9 | Replay determinism | Same seed -> identical .holo trace. | All fields identical. |
| C10 | Session/reboot repeat | Rerun after reboot. | Q_diff sign consistent. |
| C11 | Candidate-value leakage audit | Verify no hidden d, no true/false labels in .holo. | Clean. |
| C12 | Post-hoc selection audit | Verify no seed was selected after seeing results. | All seeds predeclared. |

---

## 9. .holo Record Format

| Field | Content |
|---|---|
| holo_id | UUID |
| doctrine_version | NON_COLLAPSE_V1 |
| orbit_state | FoldPair(a, N-a), unresolved |
| phase_relation | Q+, Q- from lock-in |
| path_history | Per-step operand magnitudes, tape delta |
| tape_residue | Post-restore physical probe (if available) |
| substrate_memory | PDN lock-in I/Q for both windows |
| carrier_class | B (PDN/common-mode) |
| workload_signature | W_B hash (seed, tone, steps) |
| cancellation_transcript | Q_diff, null Q_diffs, method |
| residue_hypothesis | Predeclared: Q_diff is antisymmetric under fold |
| collapse_boundary | Timestamp, measurement values |
| measurement_record | Verdict, null results, control results |
| controls | 12 control outcomes |
| verdict | L4A_RESIDUE_CANDIDATE_FOUND or L4A_RESIDUE_NOT_FOUND |
| claim_level | L3 max |

---

## 10. Verdicts

| Verdict | Condition |
|---|---|
| L4A_CLASS_B_DESIGN_READY | Design complete. No code written. |
| L4A_CLASS_B_RESIDUE_CANDIDATE_FOUND | Q_diff survives all controls. |
| L4A_CLASS_B_RESIDUE_NOT_FOUND | Q_diff below null thresholds. |
| L4A_CLASS_B_COLLAPSE_RISK_FOUND | Design accidentally collapsed to candidate scoring. |
| L4A_CLASS_B_INVALID_VERIFIER_REVERSION | verify(x) entered the design. |
| L4A_CLASS_B_PROTOCOL_BLOCKED | Cannot implement with current Phenom II primitives. |

---

## 11. Claim Ceiling

| Level | Condition |
|---|---|
| L0 | Design only (this document). |
| L1 | .holo schema validated (no implementation). |
| L2 | Non-collapse simulator preserves OrbitState through all phases. |
| L3 | Phenom II run: Q_diff measured, controls run, residue status determined. |
| L4A | Predeclared fold-odd residue survives controls. |
| L4 | Residue used for orientation recovery under no-smuggle controls. NOT claimed here. |

---

## 12. Median-Basin Check

| Question | Answer |
|---|---|
| Did this design use verify(x)? | NO |
| Did this design score candidates? | NO |
| Did this design ask which branch wins? | NO |
| Did this design use AUC as truth metric? | NO |
| Did this design collapse .holo into answer? | NO |
| Did this design confuse carrier excitation with recovery? | NO -- carrier is PDN. Recovery is not claimed. |
| Did this design assign phase/sign by candidate label? | NO -- both branches use phase=0. Asymmetry is from operand values. |
| Did this design use hidden d in runtime? | NO -- a is public. FoldPair from public data only. |

---

## 13. Roadmap Update

```
[>] L4A Class B PDN screen design ready: test PDN/common-mode as carrier
    for predeclared fold-odd residue while preserving .holo OrbitState.
    No recovery claim. Design only.
```
