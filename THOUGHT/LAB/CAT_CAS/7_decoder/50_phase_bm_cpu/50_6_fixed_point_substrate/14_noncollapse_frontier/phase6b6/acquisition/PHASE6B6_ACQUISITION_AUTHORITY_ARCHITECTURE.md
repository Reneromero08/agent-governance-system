# Phase 6B.6 Physical Acquisition Authority Architecture

**Status:** `DESIGN_ONLY__NO_EXECUTION_AUTHORITY`

**Base main commit:** `1db6d5b0b1e0d9b38a0c1709b09c9c11a59217a2`

**Integrated non-hardware evidence head:** `b2b785d064d4704ef2955238593f9f5050425f55`

**Independent evidence review:** `4614291991`

**Current authority:** hardware execution blocked

## 1. Purpose

This document defines the next authority boundary after completion of Phase 6B.6 software implementation, software qualification, sealed copy-only Phenom qualification, independent evidence review, and evidence integration.

It does not authorize any physical execution.

It defines a forward-only three-gate ladder:

```text
Gate A: engineering smoke
Gate B: calibration and capture-quality qualification
Gate C: frozen scientific acquisition
```

No gate inherits execution authority from the previous software-entry decision. Each gate requires its own exact authority artifact, independent review, and project-owner approval.

## 2. Bound predecessor chain

The authority design is bound to the integrated repository state:

```text
main merge commit = 1db6d5b0b1e0d9b38a0c1709b09c9c11a59217a2
approved evidence head = b2b785d064d4704ef2955238593f9f5050425f55
corrected evidence payload = 38eccc2f8377c656b0c21cbd37dd296e81adfad4
corrected evidence tree = 159c04abd72ed631e7647249b539b548d8351e4b
final evidence inventory = 7b205dd92482425505e498027a3e96db842297fad949fe22bdbf5542e95573e0
portable archive = affbc0b3e9725de62aa946774e3e8830399f9af12414713b1bfbc68547765ca4
portable manifest = 59e5c5927cfa7f19bdaafdd740cb350f5819e81741b62821a22f2eb80ecd4676
target final-result digest = c2d1bf3c78e2a9318f51e06d27ac39a49fe7a49e3cd49c0c8850cd6c85a07f7f
```

Frozen scientific identities remain:

```text
snapshot subject commit = d351a62f4f211589d57359d872734757b6e280d9
snapshot subject tree = 1a927b20cb2d712a7220a823621c8fc83cbc984d
scoped tree = 408ee35257417898a992510b0f260602117a15af
snapshot inventory = e47dea4c3467835a425d9d553803da48f672a8799970db4fc9b83e98596f50d8
Phase 6B.6 subtree digest = 24789f0df9afa2d9f6a243a9050ff8f265cf22ffb42ab33bbe2f67521dbf44b5
V2 source digest = c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976
qualification contract digest = 986d1eb27e6e715da0ed8765f58566b0608e464b94dbd0d58ab3d130d80fd0d2
schedule digest = c632d59934c2610541e279cac3a48202f2c0a79bb734e995f2cc4f28d19e87d3
mock custody digest = 4c0a58772fd25fe77759d6d09089ad532a09b3a5adcdce01dc099b6b7b00dba1
raw C emission digest = 56a26105b2b6969a05addbaafaa8db672d0d05a50158e1d145ed2903fec889c2
```

Any later authority artifact must bind exact source, contract, schedule, target identity, executable, and evidence digests. A later source or contract head requires fresh qualification before authority may be proposed.

## 3. Governing doctrine

The algorithm is dead.

Physical acquisition must preserve:

```text
measured complex response
physically executed control
route and topology context
absolute session timeline
ordered path and phase action
raw reconstructable custody
explicit projection boundaries
```

It must not collapse the physical object into candidate ranking, scalar verification, AUC-first scoring, target-label recovery, or winner-first interpretation.

The physical ladder remains:

```text
carrier
!= measured relational state
!= identified operator
!= restoration
!= target coupling
!= fold-odd invariant
!= Small Wall crossing
```

Each arrow remains a separate experiment and evidence gate.

## 4. Current authority state

The current state after PR #33 integration is:

```json
{
  "phase6b6_entry_approved": true,
  "phase6b6_entered": true,
  "implementation_authorized": true,
  "software_qualification_authorized": true,
  "non_hardware_target_qualification_authorized": true,
  "evidence_package_created": true,
  "qualification_evidence_collected": true,
  "hardware_ran": false,
  "authorization_artifact_created": false,
  "engineering_smoke_authorized": false,
  "calibration_authorized": false,
  "scientific_acquisition_authorized": false,
  "restoration_authorized": false,
  "target_coupling_authorized": false,
  "small_wall_authorized": false
}
```

This architecture document does not change any false field to true.

## 5. Gate A: engineering smoke authority

### 5.1 Purpose

Gate A proves that the qualified software object can cross the physical interface without claiming scientific observability.

It is a minimal engineering execution, not a reduced scientific campaign.

### 5.2 Permitted scope

A Gate A authority artifact may permit only:

```text
one target boot state
one predeclared route
one sender and receiver core pair
one short explicit-slot sequence
sender-off, carrier-off, declaration-sham, and driven slots
TSC alignment measurement
raw I/Q and ring-period capture
telemetry and capture-quality evaluation
controlled cleanup
```

The smoke schedule must be newly generated under a smoke-specific schema and must not be represented as a subset of the frozen scientific test set.

### 5.3 Required controls

Gate A must prove:

```text
sender process absent before start
sender-off slots contain no sender workload
contiguous driven slots preserve one sender epoch when required
requested and actual TSC boundaries are recorded
temperature veto is active at 68 C
automatic retry is false
capture gaps and empirical sample rate are measured
all raw files, stdout, stderr, telemetry, and commands are hashed
failure stops the run
cleanup removes temporary executables and processes
```

### 5.4 Forbidden scope

Gate A must not permit:

```text
model fitting
opening validation or test sessions
claiming predictive observability
classifying persistence
campaign-scale repetition
restoration testing
target coupling
orientation or fold-odd recovery
Small Wall work
frequency or voltage modification
MSR writes
```

### 5.5 Gate A exit

Gate A passes only after:

1. an exact authority artifact is independently reviewed and owner-approved;
2. the smoke executes exactly once under that artifact;
3. complete evidence custody is independently reviewed;
4. every fail-closed and cleanup requirement passes.

A Gate A pass does not authorize Gate B.

## 6. Gate B: calibration and capture-quality authority

### 6.1 Purpose

Gate B validates the measurement geometry required by the frozen Phase 6B.6 contract before scientific acquisition.

It does not fit or select a predictive operator.

### 6.2 Permitted scope

A Gate B authority artifact may permit only predeclared calibration sessions sufficient to validate:

```text
preamble complex anchors
sender-off idle covariance
carrier-off noise-floor strata
declaration-sham separation
empirical sample rate
maximum capture gap
TSC alignment
route and core identity
temperature and P-state stability
session-gauge construction
raw and normalized response reporting
```

Calibration data must be stored in a calibration-only namespace and may not silently become scientific training, validation, or test data.

### 6.3 Required adjudications

Gate B evidence must answer:

```text
Are all measured fields physically sourced?
Are executed and declared controls distinct?
Does sender-off mean the workload is absent?
Are carrier-off and declaration-sham strata distinguishable in custody?
Can each session gauge be frozen from preamble rows only?
Does the target remain inside temperature, timing, and capture-quality bounds?
Can every derived artifact be reconstructed from raw acquisition?
```

### 6.4 Gate B exit

Gate B passes only after independent evidence review establishes that the frozen scientific campaign can execute without changing its geometry, thresholds, split policy, or custody contract.

A Gate B pass does not authorize Gate C.

## 7. Gate C: frozen scientific acquisition authority

### 7.1 Purpose

Gate C authorizes one exact execution of the frozen Phase 6B.6 physical observability campaign.

The authority object must bind:

```text
12 sessions
6 reboot blocks
2 routes per reboot block
864 slots per session
10368 total slots
FWD, REV, RND1, RND2, and ORDER_LABEL_SHAM
train blocks b0, b1, b2
validation block b3
test blocks b4, b5
read_hz = 8000
slot_s = 0.5
nominal samples per slot = 4000
pin_khz = 1600000
temperature veto = 68 C
automatic retry = false
```

It must bind exact generated session schedules and their digests. Runtime randomization is prohibited.

### 7.2 Required campaign custody

Every session must preserve:

```text
session TSC origin
requested and actual slot boundaries
measured TSC frequency
raw lockin I and Q
raw ring-period samples
executed controls
declared controls
route and core identities
reboot block and chronology
temperature and P-state telemetry
capture-quality result
complete command ledger
stdout and stderr
file sizes and SHA-256 digests
```

No padding, interpolation, synthetic replacement, silent repetition, failed-slot catch-up, or automatic retry is permitted.

### 7.3 Test-set custody

The test set remains sealed until:

```text
state level frozen
operator class frozen
delay length frozen
regularization frozen
phase-native lift frozen
thresholds frozen
analysis manifest sealed
```

No validation or test row may contribute to covariance learning, feature selection, model selection, threshold adjustment, delay selection, or tone selection.

### 7.4 Claim ceiling

The maximum positive claim remains:

```text
EMPIRICAL_PREDICTIVE_OBSERVABILITY_OF_TESTED_MEASURED_EQUIVALENCE_CLASS
```

Permitted classifications remain:

```text
SHARED_PREDICTIVE_OPERATOR_SUPPORTED
ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY
DRIVEN_RELATIONAL_TRANSPORT_ONLY
PERSISTENT_STATE_CANDIDATE
CONFOUNDED_NO_OPERATOR_CLAIM
INSTRUMENTATION_BOUNDARY_REJECTED
```

No Gate C outcome authorizes restoration, target coupling, orientation recovery, fold-odd invariant recovery, or Small Wall work.

## 8. Authority artifact structure

Each gate requires a separate immutable authority artifact containing:

```text
schema ID
decision
project owner
decision date
gate identifier
exact reviewed architecture head
independent review ID
base main commit
source commit and tree
contract and schedule digests
target identity digest
executable and runtime bundle digests
permitted operations
prohibited operations
maximum execution count
expiration or one-shot consumption rule
fail-closed predicates
cleanup obligations
authority booleans
```

An authority artifact is valid only when:

```text
review result is approval
project-owner decision exactly matches the gate
all required digests match
no prohibited authority field is true
no source or schedule changed after review
artifact has not already been consumed
```

## 9. One-shot and non-escalation rules

Every physical authority artifact is one-shot unless it explicitly declares a smaller finite count.

Failure does not grant a retry.

A failed or vetoed run requires:

```text
evidence preservation
cleanup proof
independent failure review
new authority proposal
new project-owner decision
```

Gate A cannot be interpreted as partial Gate B authority. Gate B cannot be interpreted as partial Gate C authority. Gate C cannot be interpreted as restoration authority.

## 10. Required review sequence

For each gate:

```text
architecture or run-plan draft
independent source and contract review
exact generated-artifact review
project-owner authority decision
physical execution
complete evidence package
independent evidence review
integration decision
```

No physical execution may occur between draft and project-owner approval.

## 11. Stop conditions

Stop before execution if any condition is true:

```text
source head differs from the reviewed head
contract or schedule digest differs
target identity differs
sender-off semantics are not physically enforceable
automatic retry is enabled
temperature veto is absent or weakened
test-set access is not sealed
command or raw-file custody is incomplete
frequency or voltage control is introduced
MSR writes are introduced
candidate labels enter measured state
route or session identity enters an accepted shared state
scientific thresholds changed after observing physical output
a prior gate is treated as authority for a later gate
```

## 12. Current conclusion

The software and non-hardware evidence prerequisites for proposing physical authority are complete.

No physical gate is approved by this document.

Current decision:

```text
ACQUISITION_AUTHORITY_ARCHITECTURE_READY_FOR_INDEPENDENT_REVIEW
```

Current execution state:

```text
engineering_smoke_authorized = false
calibration_authorized = false
scientific_acquisition_authorized = false
hardware_ran = false
```

The next legitimate boundary is independent review of this architecture and its fail-closed candidate schema. Project-owner approval may be requested only after that review closes.