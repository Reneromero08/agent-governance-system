# I2E Prospective Measurement Harness Skeleton

## Result

```text
I2E_PROSPECTIVE_MEASUREMENT_HARNESS_SKELETON_COMPLETE
FIXTURE_ONLY
TWELVE_NEGATIVE_FIXTURE_CLASSES_FAIL_CLOSED
PHYSICAL_BACKEND_NOT_IMPLEMENTED
NUMERICAL_THRESHOLDS_NOT_FROZEN
PHYSICAL_PACKAGE_NOT_FREEZE_READY
I2F_TOPOLOGY_BACKEND_AND_NDESTRUCTIVE_PROBE_SPEC_NEXT
NO_LIVE_EXECUTION_AUTHORIZED
```

I2E turns the I2D reversibility-gap contract into an executable transaction schema. It
validates records and custody only. It does not execute a Family 10h operation, open PMU
events, contact the target, or select numerical acceptance thresholds.

## Transaction surface

Every future record must contain:

```text
identity
carrier
source-death seal
post-death challenge
operator specifications
per-operation receipts
multivariate state observations
independent accumulator record
restoration record
environment accounting
threshold contract
claim boundary
```

The same carrier ID must appear at all state checkpoints:

```text
baseline_pre
after_A
after_B
after_forward_commutator
after_output_decoupled
after_inverse_commutator
```

Every state observation includes logical-byte custody, `D_single`, `D_local`, four
observer channels, four overlap strata, a timing/probe vector, and a measurement receipt.

## Valid fixture ceiling

The included valid record is deliberately nonphysical:

```text
mode = synthetic_fixture
physical_backend = false
live_authority = false
qualified inverses = false
physical state equivalence = false
R2 claimed = false
all numeric thresholds = null
thresholds frozen = false
```

Passing this fixture means only that the record is complete and fail-closed. It cannot
promote any physical gate.

## Negative fixtures

The harness mutates the valid record into twelve forbidden cases. Each must fail with its
own stable class:

```text
EARLY_CHALLENGE_SELECTION
SOURCE_STILL_ALIVE
OPEN_SOURCE_IPC
CARRIER_ID_SUBSTITUTION
MISSING_OBSERVER
MISSING_OVERLAP_STRATUM
READOUT_ONLY_STATE_VECTOR
DESTRUCTIVE_RESET_AS_INVERSE
WORD_LABEL_ACCUMULATOR
CARRIER_OFF_OUTPUT_NONZERO
POSTHOC_THRESHOLD
PHYSICAL_CLAIM_WITHOUT_BACKEND
```

A failure in one class is not accepted under a different label. This prevents later
implementations from satisfying the harness by weakening or reordering the custody law.

## Accumulator boundary

The fixture accumulator is labeled synthetic and carrier-coupled. Its carrier-off and
word-only replay outputs are both zero. It freezes one output before the inverse word and
retains that synthetic output afterward.

This validates the required record flow. It does not establish a physical accumulator or
causal hardware coupling.

## Threshold boundary

I2E forbids all numeric thresholds. No equivalence margin, use floor, inverse deadline,
null ceiling, or error rate may be inserted into the fixture contract. Those values
require a future public calibration package and prospective freeze.

## Next gate

```text
I2F_TOPOLOGY_BACKEND_AND_NDESTRUCTIVE_PROBE_SPEC
```

I2F must specify, without executing, the exact core topology, carrier allocation,
directional ownership backend, multi-observer probe interface, disturbance receipts,
and capability-isolated source process needed to populate the I2E schema. It must still
leave every numerical threshold and live authorization unset.
