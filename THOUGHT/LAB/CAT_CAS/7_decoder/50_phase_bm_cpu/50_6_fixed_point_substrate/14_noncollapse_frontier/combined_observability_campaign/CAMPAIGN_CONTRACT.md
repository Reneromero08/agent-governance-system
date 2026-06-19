# Combined Tone-Order Observability Campaign

**Status:** `AUTHORIZED_FOR_IMPLEMENTATION_AND_POST_PREFLIGHT_ACQUISITION`  
**Owner decision:** `RATIFY_AND_AUTHORIZE_COMBINED_TONE_ORDER_OBSERVABILITY_CAMPAIGN`  
**Restoration:** blocked

## Question

Separate three claims without collapsing them:

```text
transport
observable measured-state dynamics
post-drive persistence
```

The campaign also separates physical tone identity from within-symbol path position.

## Sessions and partitions

```text
routes = v4s5, v2s3
seeds = 0,1,2,3,4,5 per route
train = seeds 0,1,2
validation = seed 3
stress = seed 4
final test = seed 5
```

Seed 4 is mandatory and may not be excluded.

## Tone-order conditions

```text
FWD  = ascending physical-tone order
REV  = reversed order
RND1 = frozen deterministic permutation
RND2 = independent frozen deterministic permutation
```

Order chronology is Latin-balanced across the 12 sessions. Codeword-bin permutation is generated independently from tone order.

## Stages

### A — preflight and session gauge

Machine/instrumentation checks precede samples. The scientific preamble contains only clean executed controls and is the sole source of session gauge `g_s`. The gauge is frozen before evaluated rows.

### B — tone/path disentanglement

For every order condition:

```text
real
wrong declaration
pseudo codeword permutation
order-label sham
silent
scramble
```

Executed and declared mode/order are stored separately.

### C — sender-off persistence

Counterbalanced structured and randomized orders receive impulse and step preparations. The sender is physically disabled during frozen post-drive windows. No refresh or hidden replay is allowed.

### D — predictive trajectories

Persistent-excitation trajectories include frozen zero-input segments. Acquisition is allowed regardless of earlier scientific outcomes; later interpretation cannot rescue failed Stage B or C gates.

## State partition

```text
r_t = measured I/Q/ring response
u_t = executed control
c_t = route/core/time/thermal/P-state context
g_s = preamble-only session gauge
```

S0 is raw response, S1 is gauge-normalized response, and S2 is response/input history. Context and declared labels are not state fields.

## Primary outcomes

Tone/order:

```text
TONE_IDENTITY_EQUIVARIANCE_SUPPORTED
ORDER_PATH_COVARIANCE_SUPPORTED
MIXED_TONE_PATH_GEOMETRY_SUPPORTED
FIXED_ORDER_ARTIFACT_SUPPORTED
NO_ORDER_RESOLUTION
```

Persistence:

```text
PERSISTENT_STATE_CANDIDATE
DRIVEN_RELATIONAL_TRANSPORT_ONLY
```

Operator:

```text
S0_SUFFICIENT
S1_SUFFICIENT
S2_HISTORY_REQUIRED
NO_STABLE_PREDICTIVE_OPERATOR
```

## Hard prohibitions

No result authorizes restoration, target coupling, orientation recovery, or Small Wall claims. Trial counts, thresholds, schedules, and permutations cannot change after outcome inspection. A negative result is final evidence for this campaign, not permission to rerun until positive.
