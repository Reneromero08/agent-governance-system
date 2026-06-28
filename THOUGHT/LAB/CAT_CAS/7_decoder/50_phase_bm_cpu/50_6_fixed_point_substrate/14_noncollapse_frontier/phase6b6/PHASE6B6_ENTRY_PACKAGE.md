# Phase 6B.6 Entry Package

**Status:** `SOFTWARE_ENTRY_APPROVED__IMPLEMENTATION_AUTHORIZED__HARDWARE_BLOCKED`
**Base main head:** `19afa4a4696c068c45e9bb6fcc03f65b466f0500`
**Gate R decision:** `APPROVED_FOR_INTEGRATION`
**Gate R review:** `4585403632`
**Phase 6B.6 entered:** true
**Implementation authorized:** true
**Hardware execution authorized:** false
**Scientific acquisition authorized:** false

This package freezes the scientific architecture that governs Phase 6B.6 software implementation. Project-owner approval `APPROVE_PHASE6B6_SOFTWARE_ENTRY_ONLY` was recorded on June 28, 2026 and bound by `PHASE6B6_SOFTWARE_ENTRY_APPROVAL.json`. It authorizes software implementation, software-only qualification, and sealed-snapshot non-hardware target qualification. It does not authorize physical execution on the Phenom target.

The algorithm is dead. Phase 6B.6 does not reduce the carrier to scalar candidate ranking. It preserves measured complex response, executed control, topology, phase, ordered path memory, and explicit projection boundaries. Any accepted operator must act on a declared measured equivalence class without smuggling route, session, target labels, or public candidate identity into state.

---

## 1. Bound predecessor chain

The entry object inherits, but does not mutate, the qualified V2 engineering lane.

```text
reviewed source = ba48125d15009a044bb869b5716c412b1a8baa1b
source review = 4584742973
generated contracts = 500f7dfcd198e6e70dc3f999248aa61224d530cd
generated review = 4584795315
corrective evidence = 9291d61ab3eb8d27e2bff347f1ec90a046726228
final evidence review = 4585386261
Gate R review = 4585403632
Gate R integration = APPROVED_FOR_INTEGRATION
plan SHA-256 = 3c1b8d3da4d24e97a4395747dc8f587f60d21ef6d789bd27da8cd95908b7ebb3
source-bundle SHA-256 = bec71b2369587e68a88e9e2b5cb47837a07d5cdef6f13990417e0c0928e85f2f
Phase 6B.6 architecture review = 4588082595
Phase 6B.6 software entry = APPROVED
software entry approval commit = 44cd771a06698436c49034cfd1b16bb76cdbf6ef
```

V2 remains a sealed engineering reference. Phase 6B.6 must not rewrite its source, contracts, evidence, qualification claims, or authority state.

---

## 2. Architectural decision

Phase 6B.6 is a new scientific campaign layer over the qualified V2 hardware and capture interface.

It is not:

- a new interpretation of the ascending-order V2 calibration plan;
- a mutation of historical V1 or V2 evidence;
- a restoration experiment;
- target coupling;
- a Small Wall attempt;
- authorization to execute hardware.

The implementation namespace should be new and forward-only, under `14_noncollapse_frontier/phase6b6/` or an equivalent repository-consistent location. It may reuse qualified V2 primitives by exact source binding, but it must generate its own scientific contracts, schedules, analyzers, evidence, and authority records.

The qualified V2 runner currently encodes engineering timing that includes implicit off behavior. Phase 6B.6 requires explicit contiguous drive and drive-off trajectories. Therefore the scientific scheduler must own every acquisition slot. No hidden inter-window sender action is allowed. Every slot must serialize the actual `drive_on` state and all executed controls.

---

## 3. Scientific question and claim ceiling

Primary question:

> Does a measured complex response equivalence class support a predictive state and operator across held-out sessions and routes, and does any distinguishable state persist after the sender is physically disabled?

Maximum positive claim:

```text
EMPIRICAL_PREDICTIVE_OBSERVABILITY_OF_TESTED_MEASURED_EQUIVALENCE_CLASS
```

Permitted subordinate classifications:

```text
SHARED_PREDICTIVE_OPERATOR_SUPPORTED
ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY
DRIVEN_RELATIONAL_TRANSPORT_ONLY
PERSISTENT_STATE_CANDIDATE
CONFOUNDED_NO_OPERATOR_CLAIM
INSTRUMENTATION_BOUNDARY_REJECTED
```

Forbidden claims remain:

```text
complete physical observability
physical HoloGeometry
inverse physical dynamics
physical restoration
target coupling
orientation recovery
fold-odd invariant recovery
Small Wall crossing
```

---

## 4. Object partition

The Gate R partition remains binding.

```text
r_t = directly measured response
u_t = physically executed control
c_t = nuisance and topology context
g_s = preamble-only session gauge
```

### 4.1 Measured response

```text
r_t = [lockin_I, lockin_Q, ring_osc_period]
```

`r_t` contains measured values only. It must not contain route, core identity, tone identity, sender mode, session identity, TSC origin, window index, declared labels, or target labels.

The semantic response is complex and phase-native. Numerical implementations may use an equivalent real block form for linear algebra, but the contract, transformations, metrics, and reports must preserve the complex interpretation and identify every projection.

### 4.2 Executed control

```text
u_t = [
  drive_on,
  executed_mode,
  amplitude_level,
  phase_action,
  physical_tone_index,
  executed_order_family,
  executed_order_position,
  codeword_bin_permutation
]
```

Declared values must be stored separately from executed values. A declaration is never evidence that a physical action occurred.

### 4.3 Context

```text
c_t = [
  route,
  sender_core,
  receiver_core,
  reboot_block,
  session_chronology,
  session_tsc_origin,
  actual_start_tsc,
  actual_end_tsc,
  measured_tsc_hz,
  empirical_sample_rate,
  temperature,
  P_state,
  capture_quality
]
```

Context may define blocks, stratification, random effects, or a route-conditioned operator. It may not be serialized into a claimed measured state. Session identity is prohibited as an accepted predictive feature.

### 4.4 Session gauge

Each session has a preamble-only gauge:

```text
g_s = [
  per-tone complex anchor alpha_s[k],
  per-tone amplitude floor,
  preamble drift estimate,
  local idle covariance for quality diagnostics
]
```

The session gauge is frozen before prepared-state, trajectory, validation, or test rows.

A global whitening covariance used by accepted models is estimated from training-session preambles only. Validation and test preambles may estimate their local complex anchor and quality diagnostics, but they may not update the global covariance or any learned parameter.

---

## 5. Acquisition geometry

The scientific layer inherits the qualified physical interface unless a later implementation review proves a required change.

```text
routes = [v4s5, v2s3]
v4s5 cores = [4, 5]
v2s3 cores = [2, 3]
read_hz = 8000
slot_s = 0.5
nominal_samples_per_slot = 4000
pin_khz = 1600000
temperature_veto_c = 68.0
automatic_retry = false
```

Every slot is explicit. `drive_on=false` means the sender workload is physically disabled for the entire slot. No periodic refresh, hidden sender thread, or implicit inter-slot drive is allowed.

### 5.1 TSC alignment

At session start, measure and record `measured_tsc_hz`. Define a frozen `session_tsc_origin`. Slot `k` begins at:

```text
session_tsc_origin + round(k * slot_s * measured_tsc_hz)
```

The allowed start error is at most one empirical sample period. Record requested and actual TSC boundaries. A slot that exceeds the frozen capture-gap or sample-rate tolerances fails quality review. The runtime must not silently catch up, repeat, or reorder a failed slot.

### 5.2 Capture custody

Every session must bind:

- exact scientific contract and schedule digests;
- exact runtime source bundle;
- route and core identities;
- reboot block and chronology;
- raw samples, derived CSV, telemetry, and complete stderr/stdout;
- actual executed controls and declaration fields;
- file sizes and SHA-256 digests;
- authority fields, all checked before physical fields are accepted.

---

## 6. Session topology and frozen data splits

Use six reboot blocks, numbered `b0` through `b5`. Each reboot block contains one session on each route, for twelve sessions total.

```text
session_count = 12
reboot_block_count = 6
sessions_per_reboot = 2
routes_per_reboot = [v4s5, v2s3]
```

Route execution order alternates by reboot block:

```text
even block: v4s5 then v2s3
odd block:  v2s3 then v4s5
```

Splits are frozen by complete reboot block so no reboot context crosses a split.

```text
training reboot blocks = [b0, b1, b2]
validation reboot blocks = [b3]
test reboot blocks = [b4, b5]
```

This yields:

```text
training sessions = 6
validation sessions = 2
test sessions = 4
```

Reboot block `b4` is the required stress block corresponding to retained seed 4 behavior. It must not be removed because it looks anomalous.

No row-level random split is permitted. No validation or test row may contribute to gauge-learning policy, covariance learning, feature selection, delay selection, regularization, model selection, threshold adjustment, or tone selection.

---

## 7. Tone-order controls

The physical tone set remains the frozen twelve-tone interface inherited from V2. Tone frequencies and codeword mappings must be imported by exact digest rather than copied by hand.

Frozen physical order arrays:

```text
FWD  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
REV  = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
RND1 = [3, 10, 1, 8, 5, 0, 11, 6, 2, 9, 4, 7]
RND2 = [8, 2, 11, 4, 0, 7, 3, 10, 5, 1, 9, 6]
```

The fifth family is `ORDER_LABEL_SHAM`. On even reboot blocks it physically executes `RND1` while declaring `RND2`. On odd reboot blocks it physically executes `RND2` while declaring `RND1`. Both declared and executed orders remain serialized.

Within-session order-family sequencing is defined by the binding entry review addendum and must be deterministic. Runtime randomization is prohibited.

Physical-tone-indexed and execution-order-indexed views must both be reported. Path-memory interpretation is forbidden until FWD, REV, RND1, RND2, and order-label sham are complete and accepted.

---

## 8. Frozen session schedule

Each session contains exactly 864 acquisition slots.

```text
preamble = 96 slots
prepared-state/order stage = 360 slots
trajectory stage = 384 slots
tail drift check = 24 slots
total = 864 slots
```

Across twelve sessions:

```text
total slots = 10368
active acquisition time = 86.4 minutes, excluding reboot and setup overhead
nominal_campaign_sample_count = 41472000
```

Actual sample counts are empirical and must pass the frozen capture-quality contract. No padding, interpolation, synthetic replacement, or silent repetition is permitted.

### 8.1 Preamble, 96 slots

```text
48 sender-off idle slots
12 carrier-off slots
12 time-matched declaration-sham slots
24 amplitude-level-2 anchors, one positive and one negative sign for each tone
```

Only the preamble may estimate `g_s`. Carrier-off and declaration-sham rows remain separate analysis strata. The 24 anchors are not part of model evaluation.

### 8.2 Prepared-state and order stage, 360 slots

For every order family:

```text
12 tones * 3 amplitude levels * 2 signs = 72 slots
5 order families * 72 = 360 slots
```

This stage tests prepared response geometry and tone-order disentanglement. It does not by itself establish path memory.

### 8.3 Trajectory stage, 384 slots

For every physical tone, execute four eight-slot packets:

```text
IMPULSE:
  1 driven slot at amplitude level 2
  7 physically sender-off slots

STEP:
  4 contiguous driven slots at amplitude level 2 on one absolute session timeline
  4 physically sender-off slots

PHASE_SHIFT:
  2 contiguous driven slots at phase 0
  2 contiguous driven slots at phase +pi/2 or -pi/2 on the same sender epoch
  4 physically sender-off slots

CARRIER_OFF_SHAM:
  8 physically sender-off slots with time-matched declarations
```

Packet sign and phase-shift direction alternate deterministically across tone, reboot block, and route. No trajectory packet may be split across training, validation, or test because the split unit is the complete session.

### 8.4 Tail drift check, 24 slots

```text
12 sender-off slots
12 amplitude-level-2 anchors, one per tone with counterbalanced sign
```

Tail rows measure drift and capture quality only. They do not update `g_s`, covariance, model parameters, or thresholds.

---

## 9. State ladder

Evaluate the simplest measured state first.

```text
S0 = r_t
S1 = gauge_normalize(r_t, g_s, Sigma_train)
S2(L) = [S1_t ... S1_(t-L+1); u_(t-1) ... u_(t-L+1)]
```

Frozen delay candidates:

```text
L in [2, 4, 8, 16]
```

Delay selection uses validation sessions only. Test sessions remain sealed until the complete state, operator, regularization, and threshold choice is frozen.

No state may contain session ID, target label, public candidate identity, declared order without executed order, route unless explicitly evaluating a route-conditioned model, or future information.

State escalation is sequential:

1. test S0;
2. test S1 only if S0 fails the frozen sufficiency gates;
3. test S2 only if S1 fails;
4. select the smallest `L` within 2 percent validation NRMSE of the best accepted delay candidate.

If S0 passes, delay embedding is not promoted. If S1 passes, S2 is diagnostic only.

---

## 10. Operator ladder

Models remain complex and phase-native. No backpropagation, deep network, scalar candidate verifier, or AUC-first model is permitted.

Evaluate in order:

```text
O0 nulls and baselines:
  training mean
  last value
  return to baseline
  input only
  time index only
  session lookup diagnostic null

O1 shared complex affine operator
O2 route-conditioned complex affine operator
O3 complex bilinear state-control operator
O4 fixed phase-native lift with regularized linear evolution
```

The O4 lift may contain only preregistered features such as:

```text
z
conjugate(z)
abs(z)^2
z tensor u
exp(i * executed_phase)
fixed quadratic cross terms
```

The lift is fixed before validation. Only the linear evolution coefficients and frozen regularization ladder are fit. A learned neural representation is outside this phase.

Operator escalation is allowed only when the previous operator class fails on validation. Choose the simplest operator within 2 percent validation multi-horizon NRMSE of the best accepted candidate.

---

## 11. Prediction and metrics

Primary object:

```text
s_(t+1) = F(s_t, u_t, c_t; parameters) + epsilon_t
```

Required rollout horizons:

```text
H = [1, 2, 4, 8] slots
```

Required metrics:

- complex NRMSE for measured response;
- complex correlation;
- phase error and amplitude error reported separately;
- multi-horizon rollout degradation;
- per-session, per-route, and pooled results;
- raw and gauge-normalized results;
- FWD, REV, RND1, RND2, and order-sham strata;
- sender-on and sender-off trajectory strata.

Diagnostic classification may be reported, but it is subordinate to held-out trajectory prediction and cannot define the accepted state.

---

## 12. Frozen model selection

Training sessions fit parameters. Validation sessions select:

- state level;
- delay length;
- operator class;
- regularization from a fixed ladder;
- phase-native lift choice from a fixed finite list.

The test set is opened once after the analysis manifest binds all choices.

Regularization and bootstrap seeds must be derived deterministically from the frozen analysis-contract digest. No seed may be changed after observing results.

The canonical accepted model is the simplest model that satisfies every applicable gate. Best average score alone is insufficient.

---

## 13. Acceptance gates

### 13.1 Partition and gauge integrity

```text
zero measured-state fields sourced from input or context
zero declared-label substitution for executed control
preamble-only gauge
training-only global covariance
raw and normalized results both reported
stress reboot block b4 retained
```

### 13.2 Predictive sufficiency

On final test sessions, the selected model must:

1. improve one-step NRMSE by at least 10 percent over the strongest non-session baseline;
2. improve eight-step rollout NRMSE by at least 5 percent over the strongest non-session baseline;
3. have a 95 percent session-block-bootstrap interval excluding zero gain for both improvements;
4. achieve pooled complex correlation of at least 0.80 on each route at one-step horizon;
5. have no test session worse than the strongest baseline by more than 5 percent;
6. beat the session-lookup diagnostic by more than 5 percent, otherwise a shared-operator claim is blocked.

Bootstrap resampling operates on whole trajectory packets nested within complete sessions. Test-session identities remain intact.

### 13.3 Route transfer

Evaluate both directions without target-route refitting:

```text
v4s5 training and validation to v2s3 test
v2s3 training and validation to v4s5 test
```

A shared operator requires a positive lower 95 percent bound on gain over the strongest target-route baseline in both directions. Failure of either direction blocks `SHARED_PREDICTIVE_OPERATOR_SUPPORTED`.

A route-conditioned model may support `ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY` when within-route gates pass but bidirectional transfer fails.

### 13.4 Drive-off classification

Classify `PERSISTENT_STATE_CANDIDATE` only when both Gate R conditions hold on held-out test sessions:

1. the lower 95 percent session-block-bootstrap bound of post-drive distance from sham exceeds the sham upper bound for at least three consecutive frozen sender-off slots;
2. a zero-input decay model improves held-out NRMSE by at least 10 percent over mean, return-to-baseline, and last-value baselines, with a 95 percent interval excluding zero gain.

Otherwise classify `DRIVEN_RELATIONAL_TRANSPORT_ONLY`.

### 13.5 Tone-order disentanglement

A delay-state or path-memory interpretation is blocked when:

- physical-tone-indexed and execution-order-indexed conclusions disagree materially;
- order-label sham predicts response comparably to executed order;
- a time-index baseline is within 5 percent of the dynamic model;
- performance depends on one order family or one chronology position.

### 13.6 Instrumentation-boundary rejection

If no S0, S1, or S2 state with O1 through O4 passes the required held-out gates, the final verdict is:

```text
INSTRUMENTATION_BOUNDARY_REJECTED
```

This is a valid scientific result. It must not trigger unplanned state expansion, threshold relaxation, higher-capacity model search, or acquisition repetition.

---

## 14. Adjudication matrix

```text
shared predictive gates pass + drive-off persistence passes:
  SHARED_PREDICTIVE_OPERATOR_SUPPORTED
  PERSISTENT_STATE_CANDIDATE

shared predictive gates pass + drive-off persistence fails:
  SHARED_PREDICTIVE_OPERATOR_SUPPORTED
  DRIVEN_RELATIONAL_TRANSPORT_ONLY

within-route gates pass + bidirectional route transfer fails:
  ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY

only order or session labels explain response:
  CONFOUNDED_NO_OPERATOR_CLAIM

no measured state and operator pass:
  INSTRUMENTATION_BOUNDARY_REJECTED
```

No outcome in this matrix authorizes restoration or target coupling.

---

## 15. Authorized software implementation package

The project owner has approved software entry. An implementation agent may now create:

```text
phase6b6 scientific contract and schema
frozen tone-order permutations and schedule generator
explicit-slot runtime wrapper over qualified V2 primitives
capture and custody validators
state-construction library
phase-native operator ladder
analysis preregistration manifest
synthetic fixtures and negative controls
strict software-only CI workflow
non-hardware target qualification package
```

The first implementation milestone is software-only. It must prove schedule determinism, partition integrity, no hidden sender action, state/input/context separation, analysis determinism, and authority fail-closed behavior.

Hardware acquisition remains separately blocked after implementation qualification.

---

## 16. Required evidence before acquisition authority

The software implementation must pass:

- deterministic double generation of every scientific contract and session schedule;
- exact source and generated-artifact binding;
- strict C compilation where applicable;
- functional tests;
- separate ASan and UBSan lanes;
- synthetic trajectory recovery tests;
- negative tests for session leakage, route leakage, declaration substitution, future leakage, and hidden drive;
- frozen-split verification;
- phase-native metric verification;
- sealed-snapshot Phenom software qualification without hardware execution;
- independent implementation and evidence review.

Only then may a separate authority record be proposed for physical acquisition.

---

## 17. Current authority envelope

```json
{
  "schema_id": "CAT_CAS_PHASE6B6_ENTRY_DECISION_V1",
  "design_status": "SOFTWARE_ENTRY_APPROVED__IMPLEMENTATION_AUTHORIZED__HARDWARE_BLOCKED",
  "base_main_head": "19afa4a4696c068c45e9bb6fcc03f65b466f0500",
  "gate_r_review": 4585403632,
  "architecture_review": 4588082595,
  "gate_r_integration_approved": true,
  "phase6b6_entry_approved": true,
  "phase6b6_entered": true,
  "implementation_authorized": true,
  "software_qualification_authorized": true,
  "non_hardware_target_qualification_authorized": true,
  "hardware_ran": false,
  "authorization_artifact_created": false,
  "calibration_authorized": false,
  "scientific_acquisition_authorized": false,
  "restoration_authorized": false,
  "target_coupling_authorized": false,
  "small_wall_authorized": false,
  "next_boundary": "INDEPENDENT_SOFTWARE_IMPLEMENTATION_AND_EVIDENCE_REVIEW_BEFORE_ACQUISITION_AUTHORITY"
}
```

The next work is software implementation and software-only qualification of the frozen effective design. No physical execution is authorized.
