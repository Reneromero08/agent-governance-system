# Exp 50 L4A Class B Implementation Plan

**Date:** 2026-06-14.
**Status:** EXP50_L4A_CLASS_B_IMPLEMENTATION_PLAN: PLAN_ONLY__HOLO_FIRST__NO_RECOVERY__NO_VERIFY

---

## Executive Summary

This plan prepares the first L4A Class B runtime. It preserves FoldPair as unresolved
OrbitState. It writes `.holo` records before any scalar measurement. It screens only
for a predeclared PDN/common-mode OddResidue coordinate Q_diff. It makes no recovery
claim. No verify(x). No candidate scoring. No AUC. No d output.

---

## 1. Doctrine Check

| Question | Answer |
|---|---|
| Does this plan use verify(x)? | NO |
| Does this plan score candidates? | NO |
| Does this plan ask which branch wins? | NO |
| Does this plan output d? | NO |
| Does this plan use AUC-first framing? | NO |
| Does this plan preserve .holo process-object? | YES |
| Does this plan delay CollapseBoundary? | YES |
| Does this plan keep OrbitState unresolved? | YES |

---

## 2. Implementation Components

| File | Purpose | Allowed Content | Forbidden Content |
|---|---|---|---|
| `holo_record.h` | .holo schema header. Struct definitions. | Field names, types, docstrings. | Scalar d. winner. truth labels. |
| `holo_record.c` | .holo write/read/validate. | Serialize OrbitState to .holo file. Validate no collapse fields. | Writing candidate_0_truth. Writing recovered_d. |
| `class_b_pdn_screen.c` | Main L4A Class B runtime. | FoldPair construction, coupled walk, lock-in capture, cancellation, .holo write. | verify(x). candidate loop. AUC computation. |
| `results/l4a_class_b/*.holo` | Per-trial .holo records. | Full process-object per the schema. | Scalar answer file. |
| `results/l4a_class_b/summary.csv` | Trial-level summary. | trial_id, seed, N, Q_diff, controls_pass, verdict. | winner, recovered_d, orientation_auc. |
| `EXP50_L4A_CLASS_B_RUN_REPORT.md` | Post-run analysis report. | Verdict, control results, residue status, claim level. | L4 recovery claim. d output. |

**Language:** C. No Rust on Phenom hardware path.
**Build:** `gcc -O2 -pthread -march=amdfam10 -Wall -Wextra -lssl -lcrypto -lm`
**Carrier:** T300-proven alu_burst + lock-in. Route 4:5.

---

## 3. .holo Schema Plan

Required fields:

| Field | Type | Content |
|---|---|---|
| holo_id | char[37] | UUID v4 |
| doctrine_version | char[32] | "NON_COLLAPSE_V1" |
| run_id | uint64_t | seed + trial index |
| seed | uint64_t | Deterministic RNG seed |
| N | int | 256 (n=8) or 1024 (n=10) |
| orbit_state | struct | FoldPair values |
| fold_pair_relation | char[16] | "conjugate" (a, N-a are paired) |
| branch_plus_value | int | a (public fold magnitude) |
| branch_minus_value | int | N-a (public fold mirror) |
| branch_assignment_note | char[128] | "branch_plus = a always. Not a truth label. Verified by label-swap control." |
| phase_relation_q_plus | double | Lock-in Q at drive tone, branch+ window |
| phase_relation_q_minus | double | Lock-in Q at drive tone, branch- window |
| phase_relation_i_plus | double | Lock-in I at drive tone, branch+ window |
| phase_relation_i_minus | double | Lock-in I at drive tone, branch- window |
| path_history_steps | int | Number of walk steps |
| tape_residue_diagnostic | double | Post-restore ring-osc period (diagnostic only) |
| substrate_memory_note | char[128] | "PDN/common-mode carrier. Lock-in I/Q recorded." |
| carrier_class | char[32] | "B_PDN_COMMON_MODE" |
| workload_signature | char[64] | SHA of W_B parameters |
| sender_core_plus | int | 4 |
| sender_core_minus | int | 5 |
| receiver_core | int | 2 |
| cancellation_method | char[32] | "Q_diff = Q_plus - Q_minus" |
| cancellation_transcript_q_common | double | (Q_plus + Q_minus)/2 (fold-even component) |
| cancellation_transcript_q_diff | double | Q_plus - Q_minus (predeclared residue coordinate) |
| residue_hypothesis | char[256] | "Q_diff is antisymmetric under branch conjugation. Predeclared before measurement." |
| residue_predeclared | int | 1 |
| collapse_boundary_timestamp | char[32] | ISO timestamp |
| measurement_record_q_diff_magnitude | double | abs(Q_diff) |
| measurement_record_q_diff_sign | int | sign(Q_diff) |
| measurement_record_same_orbit_q_diff | double | Q_diff from same-orbit control |
| measurement_record_dummy_orbit_q_diff | double | Q_diff from dummy-orbit control |
| measurement_record_label_swap_pass | int | 1 if Q_diff sign flipped under swap |
| controls_pass | int | Bitmask of passing controls |
| verdict | char[64] | L4A verdict label |
| claim_level | int | 3 max |

Forbidden fields (must not exist in any .holo):

| Field | Reason Forbidden |
|---|---|
| hidden_d | Hidden truth in runtime record |
| winner | Early collapse |
| candidate_0_truth | Branch truth label |
| candidate_1_truth | Branch truth label |
| recovered_d | Scalar answer |
| orientation_label | Hidden orientation |
| orientation_auc | AUC before residue |
| posthoc_selected_result | Seed selection |
| verify_score | Verifier as branch selector |

---

## 4. OrbitState Encoding Plan

FoldPair {a, N-a} represented as:
- `branch_plus_value = a`
- `branch_minus_value = N-a`

Rules:
- `branch_plus` always = `min(a, N-a)`. This is a MAGNITUDE ordering from public data, not a truth assignment.
- `branch_minus` always = `max(a, N-a)`. Same.
- Core 4 always runs branch_plus. Core 5 always runs branch_minus.
- This assignment is DETERMINISTIC from public magnitudes. It does not depend on whether a = d or a = N-d.
- The label-swap control (swap which core runs which branch) verifies: if Q_diff sign flips, the residue is coupled to branch VALUE, not core identity.

Forbidden:
- No field named "true" or "false" associated with either branch.
- No field encoding the hidden orientation bit.
- No field assigning phase/sign/direction to a branch based on hidden truth.

---

## 5. Workload Plan W_B

| # | Operation | Detail |
|---|---|---|
| 1 | Initialize tape buffers for both sender cores from shared seed. | Deterministic. SHA-256 recorded. |
| 2 | Construct FoldPair(a, N-a) from public oracle. | a recovered via public score readout. |
| 3 | Core 4: execute branch_plus walk. For each k_j, compute a * k_j mod N. Drive alu_burst at 200 Hz. | alu_burst verbatim from slot2_pdn_lockin.c. Intensity = 1.0 (standard). Phase = 0. |
| 4 | Core 5: execute branch_minus walk SIMULTANEOUSLY. For each k_j, compute (N-a) * k_j mod N. Drive alu_burst at 200 Hz. | Same tone. Same duty cycle. Same phase. Phase difference (if any) from operand values. |
| 5 | Core 2: capture ring-oscillator timing for TWO windows. W+ during branch_plus walk. W- during branch_minus walk. | Window order randomized per trial. Each window 0.4s. Lock-in at 200 Hz. |
| 6 | Store I+, Q+, I-, Q+ in SubstrateMemory fields. | No scoring. Recording only. |
| 7 | XOR-restore both tape buffers. SHA-256 verify. | Both must pass. |
| 8 | Write .holo record. | All fields populated. |

**Energy budget:** Both branches execute the SAME number of integer multiply operations.
The operand MAGNITUDES differ but instruction count and memory footprint are MATCHED.
Any asymmetry is from operand values, not workload imbalance.

**No manual phase encoding:** Both branches use alu_burst phase = 0. The lock-in phase
difference (if any) arises from the physical PDN response to different operand-driven
switching activity.

---

## 6. Cancellation Plan

| Symbol | Meaning |
|---|---|
| Q_plus | Lock-in Q at 200 Hz during branch_plus window |
| Q_minus | Lock-in Q at 200 Hz during branch_minus window |
| Q_common | (Q_plus + Q_minus) / 2 -- fold-even component |
| Q_diff | Q_plus - Q_minus -- predeclared residue coordinate |

- Q_common is expected to be nonzero (PDN carrier is live, T300-proven).
- Q_diff is expected to be zero if both branches produce identical PDN response.
- Q_diff is expected to be nonzero if operand magnitude asymmetry creates measurable PDN differential.
- Q_diff is NOT orientation. It is NOT recovered d. It is only a predeclared residue coordinate.

Cancellation steps:
1. Record Q_plus and Q_minus.
2. Compute Q_diff = Q_plus - Q_minus.
3. Verify against same-orbit control: FoldPair(a, a) -> Q_diff_same expected 0.
4. Verify against dummy-orbit control: FoldPair(42, 42) -> Q_diff_dummy expected 0.
5. If both null controls pass: Q_diff is a valid residue coordinate.
6. If either null control fails: the cancellation is invalid (noise floor or core asymmetry artifact).

---

## 7. Control Execution Plan

| # | Control | Implementation | Expected | Collapse Risk Caught |
|---|---|---|---|---|
| C1 | Same-orbit | FoldPair(a, a). Both cores run a. | Q_diff = 0 | Core asymmetry artifact |
| C2 | Dummy-orbit | FoldPair(42, 42). | Q_diff = 0 | Operand-independent artifact |
| C3 | Label-swap | Core 4 runs branch_minus, core 5 runs branch_plus. | Q_diff sign flips | Core identity artifact |
| C4 | Phase-randomized | Swap which value is branch_plus per trial. | Q_diff sign tracks value | Label encoding artifact |
| C5 | Path-shuffled | Shuffle k_j order. | Q_diff magnitude unchanged | Path-order artifact |
| C6 | Carrier-off | Disable alu_burst (no PDN drive). | Q_diff = 0 | Non-PDN artifact |
| C7 | Measurement-order | Measure W- first, then W+. | Q_diff sign flips | Window-order artifact |
| C8 | Wrong-restore | Deliberate SHA mismatch. | Run flagged invalid | SHA-as-substrate artifact |
| C9 | Replay | Same seed -> identical .holo. | All fields identical | Non-determinism |
| C10 | Session repeat | Rerun after reboot. | Q_diff sign consistent | Transient state artifact |
| C11 | Leakage audit | Scan .holo for forbidden fields. | No hidden_d, no truth labels | Hidden truth leakage |
| C12 | Post-hoc audit | Verify all seeds predeclared. | No seed selected after run | Seed selection artifact |

---

## 8. Run Matrix Plan

First safe run:

| Parameter | Value |
|---|---|
| n | 8 (N=256, M=384) |
| Seeds | 5 |
| Orbit instances per seed | 8 |
| Total trials (public mode) | 40 |
| Controls | 12, each with 5 trials |
| Total measurements | ~100 |
| Expected runtime | ~15-20 minutes |
| Stop conditions | k10temp >= 68 C. Any SHA mismatch in normal mode. |

Do not optimize for detection of a weak signal.
Optimize for doctrine-clean execution and control coverage.

---

## 9. Verdict Rules

| Verdict | Condition |
|---|---|
| L4A_CLASS_B_PROTOCOL_READY_NOT_RUN | Plan complete. No hardware run. |
| L4A_CLASS_B_RUN_CLEAN_NO_RESIDUE | All controls pass. Q_diff below null thresholds. |
| L4A_CLASS_B_RESIDUE_CANDIDATE_FOUND | Q_diff survives all controls above null thresholds. |
| L4A_CLASS_B_COLLAPSE_CONTAMINATION_FOUND | verify(x), AUC, candidate scoring, or truth labels found. |
| L4A_CLASS_B_NULL_CONTROL_FAILED | Same-orbit or dummy-orbit showed nonzero Q_diff. |
| L4A_CLASS_B_MEASUREMENT_INVALID | Wrong-restore passed, replay failed, or carrier-off showed signal. |
| L4A_CLASS_B_NEEDS_REDESIGN | Protocol cannot execute on current Phenom II primitives. |

---

## 10. Claim Ceiling

| Level | Condition |
|---|---|
| L1 | Plan complete (this document). |
| L2 | Code compiles, runs, writes .holo records without collapse contamination. |
| L3 | Clean run with 12/12 controls. Residue status determined. |
| L4A | Predeclared fold-odd residue survives all controls. MECHANISM CANDIDATE only. Not recovery. |
| L4 | Orientation recovery from residue under no-smuggle controls. NOT claimed here. |

---

## 11. Median-Basin Tripwires

The implementation has collapsed into algorithmic framing if:

| Sign | Action |
|---|---|
| verify(x) appears in the code | REJECT. Rewrite. |
| A candidate loop appears | REJECT. Rewrite. |
| AUC is computed before residue declaration | REJECT. Rewrite. |
| Output says "winner" or "recovered d" | REJECT. Rewrite. |
| .holo stores a scalar answer as the primary object | REJECT. Rewrite. |
| Branch labels encode truth (candidate_0 = true) | REJECT. Rewrite. |
| Q_diff is interpreted as d or orientation | REJECT. Rewrite. |
| Public a is presented as the recovered answer | REJECT. Rewrite. |
| SHA restoration is treated as proof of residue | REJECT. Rewrite. |

---

## 12. Implementation Readiness

**PLAN_READY_FOR_CODE.**

All components defined. Schema specified. Controls enumerated. Run matrix sized. Tripwires listed. The plan preserves the doctrine: no verify(x), no candidate scoring, no AUC, no d output, .holo stores process-object, measurement only after cancellation, CollapseBoundary explicit.

---

## 13. Roadmap Update

```
[>] L4A Class B implementation plan: .holo-first PDN/common-mode screen
    prepared. No verify, no recovery, no scalar candidate scoring.
    Ready for code.
```
