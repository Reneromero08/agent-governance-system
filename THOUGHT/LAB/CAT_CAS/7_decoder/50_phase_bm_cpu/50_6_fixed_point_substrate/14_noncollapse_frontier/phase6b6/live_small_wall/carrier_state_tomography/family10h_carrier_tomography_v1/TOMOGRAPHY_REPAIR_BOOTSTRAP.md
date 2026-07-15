# Family 10h Carrier Tomography Repair Bootstrap

Status: `BOOTSTRAPPED_FOR_BLOCKER_REPAIR`

Branch:

```text
codex/family10h-tomography-repair
```

Base commit:

```text
32b5af119a03bc48bb00f279e6cc0014406147ad
```

This branch repairs the already-built public Family 10h carrier-state tomography package. It must not contain audio-frequency work and must not modify the retired OrbitState confirmation package except where an inherited read-only reference is necessary.

## Required Claim Boundary

Retain:

```text
GAIN_COVARIANT_ORBITSTATE_PROJECTION_TRANSDUCTION_ESTABLISHED
GAIN_COVARIANT_FINAL_CONFIRMATION_PACKAGE_RETIRED
QUERY_SEPARATED_ARCHITECTURE_BLOCKED
QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED
SMALL_WALL_CROSSED_NOT_PROMOTED
```

The tomography lane may eventually emit only the public carrier-state classes already defined in `CARRIER_TOMOGRAPHY_CONTRACT.md`.

No live execution is authorized by this repair branch.

## Authoritative Current Decision

```text
FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED
```

The active contract and generated package metadata must not claim `FROZEN_AWAITING_AUTHORIZATION` until every material blocker is closed and all required reviewers return final responses.

## Material Blockers to Close

### PHYS-REVIEW-01: approved CPU temperature sensor identity

Current defect:

```text
read_temperature_sample() selects the first numerically valid hwmon temperature path
without proving it is the approved CPU sensor.
```

Repair requirements:

```text
bind approved hwmon name, preferably k10temp
bind approved sensor label
bind resolved device/path identity in the manifest
require the same identity throughout the transaction
reject non-CPU sensors and identity drift
```

Required regressions:

```text
non-CPU sensor appears first
wrong hwmon name
wrong sensor label
path substitution
identity drift
unreadable approved sensor
```

### OPER-REVIEW-01: preserve query and query-order structure

Current defect:

```text
evidence_samples() collapses query_A, query_B, query_A_then_B, and query_B_then_A
into one A-minus-B contrast labeled query_A.
```

Repair requirements:

```text
preserve each query as an explicit observation
preserve query order
model main query effects
model query-order effects
model preparation-by-query interactions
model delay-by-query interactions
model mapping-by-query interactions
```

Required regression:

```text
a packet with signal only in query order must be detected by operator analysis
```

### OPER-REVIEW-02: strict factor holdouts

Current defect:

```text
mapping and delay holdouts retain the same mapping/delay factor levels in training
through replicate 0, so they are not strict held-out-factor tests.
```

Repair requirements:

```text
held-out replicate: test replicate absent from training
held-out mapping: test mapping level absent from training
held-out delay: test delay level absent from training
```

Required regressions:

```text
mapping-scrambled packet with pooled contrast preserved remains candidate
Delay-scrambled packet with pooled contrast preserved remains candidate
in-sample fit cannot rescue strict factor-holdout failure
```

### OPER-REVIEW-03: derived classification gate

Current defect:

```text
cross_validated_codeword_classification is asserted True rather than computed.
```

Repair requirements:

```text
freeze classifier and decision rule
freeze training/test splits
compute confusion matrix
compute balanced performance metric
freeze prospective minimum success threshold
fail or downgrade when held-out classification fails
```

Required regressions:

```text
perfectly separated codewords pass
deliberately confused codewords fail
class-imbalanced trivial predictor fails
training memorization with held-out failure does not pass
```

### OPER-REVIEW-04: complete lifetime classification

Current defect:

```text
lifetime_summary() does not fully implement the contract vocabulary or gate
session/mapping variation.
```

Required vocabulary:

```text
vanishes before source death
survives only immediate handoff
survives a bounded delay
persists across the full grid
changes form across delay
```

Repair requirements:

```text
separate mean persistence curve
separate session variation
separate mapping variation
separate query variation
report confidence intervals
freeze downgrade laws for excessive variation
```

Required regressions:

```text
stable persistence
bounded monotonic decay
immediate-only response
no post-source response
nonmonotonic form change
session-confounded response
mapping-confounded response
```

## Review Completeness

The prior latest-state custody reviewer response was not received. After repairs, run a complete fresh read-only review with final responses from all four roles:

```text
physical carrier-state auditor
experimental-design/operator auditor
custody/evidence auditor
claim-boundary adjudicator
```

A missing response is not clearance.

A single material blocker retains:

```text
FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED
```

All four must return no material blocker before the package may become:

```text
FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_FROZEN_AWAITING_AUTHORIZATION
```

That status still does not authorize live contact.

## Custody

Recommended worktree:

```text
D:\CCC 2.0\AI\agent-governance-system-family10h
```

Work only on this branch. Do not merge to `main` during repair.

Do not use:

```text
SSH
SCP
ping
target inspection
PMU hardware execution
live-authority environment variables
remote cleanup
```

## Validation

Run at minimum:

```text
Python syntax
strict C compilation
runtime self-test
schedule validation
source-death custody tests
sensor-identity tests
operator query-structure tests
strict holdout tests
classification tests
lifetime tests
exact coverage tests
target self-test
controller self-test
prepare-only
validate-only
transport simulation
deployment-layout test
source-bundle reconstruction
JSON and JSONL parsing
git diff --check
governance critic
ci_local_gate.py --full
```

## Git Completion

Use one coherent repair commit or a small number of large coherent commits. Push only:

```text
codex/family10h-tomography-repair
```

Do not merge to `main`.

Final report must include:

```text
starting branch head
final branch head
base main commit
all repaired finding IDs
contract status
manifest hashes
source bundle hash
binary hash
self-test hashes
all four reviewer IDs and verdicts
remaining blockers
working-tree status
target contact count = 0
live invocation count = 0
```

End with exactly one:

```text
FAMILY10H_TOMOGRAPHY_REPAIR_READY_FOR_INTEGRATION_REVIEW
FAMILY10H_TOMOGRAPHY_REPAIR_BLOCKED
```
