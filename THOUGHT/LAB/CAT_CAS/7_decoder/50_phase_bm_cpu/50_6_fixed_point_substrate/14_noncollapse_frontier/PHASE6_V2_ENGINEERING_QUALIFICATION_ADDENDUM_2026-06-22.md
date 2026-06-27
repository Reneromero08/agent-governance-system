# Phase 6 V2 Engineering Qualification Addendum

**Status:** `ACTIVE_BINDING_ENGINEERING_ADDENDUM`  
**Date:** 2026-06-22  
**Current stage:** `PHASE6B5D_V2_CALIBRATION_ARCHITECTURE_DRAFT__GATE_R_STILL_NEXT`  
**Base authority:** `COURSE_CORRECTION.md` and `COURSE_CORRECTION_ADDENDUM_2026-06-19.md`  
**Physical acquisition authorized:** false  
**Restoration authorized:** false  
**Target coupling authorized:** false  
**Small Wall execution authorized:** false

---

## 1. Purpose

This addendum gives the forward-only V2 executor and spectral-calibration work an explicit place in the Phase 6 authority stack.

It does not change the non-collapse doctrine, promote the V1 campaign, accept Gate R, enter Phase 6B.6, or authorize hardware execution.

The V2 branch is an engineering-qualification object. It exists to prove that a corrected waveform, executor, custody chain, calibration schedule, and analyzer can be made internally coherent before any new physical control is considered.

---

## 2. Preserved scientific state

The following state remains binding:

```text
V1 historical acquisition:
PERMANENT_RETROSPECTIVE_NEGATIVE_ADJUDICATION
NO_STABLE_PREDICTIVE_OPERATOR
PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN

T48 carrier result:
TRANSFER_EQUIVARIANCE_SUPPORTED under a minimal C0 receiver chart
STRICT_CARRIER_CLOSURE_PARTIAL

Gate R:
PENDING

Phase 6B.6:
NOT ENTERED

physical restoration:
NOT ESTABLISHED

target-to-carrier coupling:
NOT ESTABLISHED

fold-odd invariant:
NOT ESTABLISHED

Small Wall crossing:
NOT ESTABLISHED
```

No V2 software result may rewrite those scientific records.

---

## 3. V2 engineering object

The current V2 lane contains:

- corrected eight-state waveform semantics at the requested fundamental;
- 1/8, 2/8, and 3/8 amplitude duties;
- restored Slot2 drive-source identity protection;
- exact four-session calibration contracts;
- pre- and post-reboot repetitions on routes `v4s5` and `v2s3`;
- 672 windows per session and 2,688 windows per campaign;
- sender-on and sender-off raw captures;
- exact authorization, source-bundle, session, run, and telemetry bindings;
- C/Python waveform equivalence tests;
- spectral reconstruction and frozen calibration thresholds;
- fail-closed non-acquisition labels.

The object remains a draft until final-head qualification and evidence binding are complete.

---

## 4. Separation from the selected next physical control

Phase 6B.5D selected reversed or randomized tone order as the proposed next physical control.

The current V2 calibration plan uses ascending tone order. Therefore:

```text
V2 ascending-order spectral calibration
!=
reversed/randomized tone-order scientific control
```

The V2 plan qualifies the executor and measurement contract. It does not execute or replace the selected next-control experiment.

Any reversed or randomized tone-order campaign requires its own versioned plan, review decision, authorization artifact, source bundle, and evidence chain.

---

## 5. Required qualification gates

### Q1. Source and contract closure

The V2 source must:

- reject malformed command-line numerics completely rather than accepting valid prefixes;
- reject malformed, duplicate, unknown, or trailing authorization content;
- bind runtime capture-quality thresholds mechanically to the frozen plan or prove exact source identity between the two definitions;
- enforce capture coverage, empirical sample-rate range, Nyquist margin, and maximum timestamp gap in both runtime and analyzer adjudication;
- validate exact plan, authorization, source-bundle, session, evidence-map, run-manifest, CSV, and telemetry schemas;
- reject extra files, directories, and symlinks in immutable run directories;
- preserve the same input bytes used for parsing in every recorded digest.

### Q2. Repository CI closure

GitHub CI must compile and test `holo_runtime_v2`, not only the historical `holo_runtime`.

The required CI lane includes:

- strict C compile with `-Wall -Wextra -Werror`;
- V2 runner tests;
- C/Python waveform equivalence;
- Slot2 primitive identity;
- V2 calibration-contract tests;
- V2 analyzer tests;
- ASan and UBSan execution.

### Q3. Exact final-head Linux closure

The exact final source head must pass on the Linux target:

- strict release compilation;
- C/Python waveform equivalence;
- ASan/UBSan executor tests;
- target-toolchain source identity checks;
- the V2 focused Python suite;
- the no-write full repository gate.

No earlier commit's log may qualify a later source head.

### Q4. Evidence and authority synchronization

After Q1 through Q3 pass:

- regenerate calibration contracts from the final source commit;
- regenerate source-bundle and session-manifest digests;
- commit final-head Windows, GitHub, and Linux logs;
- regenerate the evidence inventory against those exact bytes;
- update the PR description, architecture review, navigation, and roadmaps;
- keep the PR draft until an independent review finds no remaining blocker.

---

## 6. Authorization boundary

Engineering qualification does not authorize execution.

A future V2 hardware calibration requires a new singleton authorization per session, bound to:

- exact executor commit and binary digest;
- exact campaign source commit;
- exact plan digest;
- exact singleton source bundle and session manifest;
- exact route/core pair;
- exact runtime parameters;
- exact output root;
- named project-owner authorization.

The authorization must keep these fields false:

```text
acquisition_authorized
restoration_authorized
target_coupling_authorized
small_wall_authorized
```

---

## 7. Transition rule

The allowed sequence is:

```text
V2 source and contract closure
-> GitHub V2 CI closure
-> exact final-head Linux qualification
-> evidence and roadmap synchronization
-> independent PR review
-> Gate R project-owner decision
```

Only after Gate R acceptance and a separate explicit authorization may the lab enter physical observability/operator acquisition.

The algorithm remains dead. The V2 lane qualifies the physical interface without collapsing the unresolved relation into a scalar verifier, candidate ranking, or benchmark-first substitute.