# Combined Campaign Analysis Contract

**Status:** `FROZEN_BEFORE_ACQUISITION`  
**Selection unit:** whole session  
**Restoration analysis:** forbidden

## 1. Integrity before science

Verify campaign, session, and run manifests; exact window counts; contiguous window indices; executed/declared separation; sender-off drive state; telemetry; raw hashes; no overwritten output; and complete seed/route coverage. Any integrity failure blocks scientific adjudication for that session and is not silently imputed.

## 2. Session gauge

Estimate complex `g_s` only from `A_GAUGE` preamble rows. Freeze it before all B/C/D rows. Report raw and gauge-normalized coordinates. Seed 4 remains included.

## 3. Tone/order adjudication

Evaluate held-out sessions in two views:

```text
physical-tone indexed
execution-position indexed
```

Use real, wrong, pseudo, order-sham, silent, and scramble rows. Actual execution must beat declared sham metadata. Choose exactly one outcome:

```text
TONE_IDENTITY_EQUIVARIANCE_SUPPORTED
ORDER_PATH_COVARIANCE_SUPPORTED
MIXED_TONE_PATH_GEOMETRY_SUPPORTED
FIXED_ORDER_ARTIFACT_SUPPORTED
NO_ORDER_RESOLUTION
```

No path-memory interpretation is allowed when order shams, random orders, or nulls fail.

## 4. Persistence adjudication

During `C_PERSISTENCE_OFF`, sender drive must be physically absent. Compare with time-matched silent/sham windows.

Choose `PERSISTENT_STATE_CANDIDATE` only when both hold on held-out sessions:

1. lower 95% session-block-bootstrap distance from sham exceeds the sham upper bound for at least three consecutive frozen windows;
2. a zero-input decay/transition model improves held-out NRMSE by at least 10% over mean, return-to-baseline, and last-value baselines, with a 95% interval excluding zero gain.

Otherwise choose `DRIVEN_RELATIONAL_TRANSPORT_ONLY`.

## 5. Predictive operator ladder

State candidates:

```text
S0 = raw measured response
S1 = preamble-gauge-normalized response
S2(L) = response history plus prior executed-input history
```

Operator candidates:

```text
O0 affine
O1 route/context-conditioned affine
O2 bilinear response/control
O3 compact nonlinear
```

Training uses seeds 0–2, model/history selection uses seed 3, seed 4 is a mandatory stress report, and seed 5 is untouched final test. Report leave-one-session-out and cross-route transfer.

Nulls include mean, last value, return-to-baseline, input-only, route-only, time-index, session lookup, shuffled input, and random linear operators. Block a shared-operator claim when session lookup is within 5% of the selected dynamic model.

Choose exactly one:

```text
S0_SUFFICIENT
S1_SUFFICIENT
S2_HISTORY_REQUIRED
NO_STABLE_PREDICTIVE_OPERATOR
```

## 6. Claim ordering

Tone/order failure cannot be rescued by persistence or operator fit. Persistence failure cannot be rescued by driven prediction. Operator success does not establish hidden substrate state. No outcome authorizes restoration, target coupling, orientation recovery, or Small Wall claims.
