# Family 10h Paired Dirty-Probe Tomography Repair Audit

This sidecar preserves the official frozen package and the official attempt-3 result. It does not rewrite `family10h_carrier_tomography_v1`, does not authorize live contact, and does not promote `SMALL_WALL_CROSSED`.

## Frozen Boundary

```text
retained run = family10h_carrier_tomography_v1_0 attempt_3
official result = FAMILY10H_POST_SOURCE_STATE_NOT_OBSERVED
sidecar result vocabulary = retrospective diagnostic only
official_result_replaced = false
small_wall_promoted = false
```

## Defect Diagnosed

The frozen adjudicator reduced the measured PMU vector to `change_to_dirty` only. The contract, however, freezes a measured observable vector that includes Change-to-Dirty, dirty probe response, CPU cycles, and duration.

The live evidence shows `change_to_dirty` near zero for almost all rows, but `dirty_probe_response` carries the public q-codeword structure. The original adjudicator also multiplies by a map sign. The runtime maps logical queries through the same map variant used for preparation, so logical `query_A` and `query_B` should be compared consistently across map0/map1 rather than sign-inverted.

## Repaired Diagnostic Law

The repaired offline diagnostic law is:

```text
D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)
```

Group matching keys are:

```text
session, replicate, delay, mapping, source_order, q, source_off_control
```

Mapping is treated as a consistency factor for logical query names, not as a sign reversal. This follows the runtime's `mapped_query_lane` behavior.

## Attempt-3 Retrospective Result

The copied attempt-3 evidence passes custody under the original packet validator and passes the repaired paired dirty-probe diagnostic law.

Summary from `PAIRED_DIRTY_PROBE_ADJUDICATION.json`:

```text
result = FAMILY10H_PAIRED_DIRTY_PROBE_TOMOGRAPHY_SUPPORTED_RETROSPECTIVE
small_wall_promoted = false
```

Hardened margins:

```text
active paired samples = 560
source-off paired samples = 80
active full-factor-q cells = 560/560 unique, min count 1, max count 1
source-off full-factor cells = 80/80 unique, min count 1, max count 1
global slope = 1.8797825404575892 dirty-probe counts per q
global R2 = 0.9955421082831637
global RMSE = 128.80767152163438
ordinary least-squares intercept = -3.6464285714285714
source-off abs max = 45.0
source-off abs mean = 1.45
signal/null ratio = 64.00916666666666
nearest-q exact accuracy = 1.0
nearest-q nonzero sign accuracy = 1.0
```

Odd-symmetry residuals:

```text
q=512 relative odd residual = 0.001951422587065971
q=1024 relative odd residual = 0.0020569722925183782
q=1536 relative odd residual = 0.005605613380974637
```

Slope disagreement:

```text
mapping = 0.00468036624768273
delay = 0.009284744479029102
session = 0.0034496969360207717
source_order = 0.0004628118183032302
```

Query observations remain separated:

```text
persistence query_A = 640
persistence query_B = 640
persistence query_A_then_B = 640 retained, not fit
persistence query_B_then_A = 640 retained, not fit
factorial query_A_then_B = 2240 retained, not fit
factorial query_B_then_A = 2240 retained, not fit
```

Channel specificity:

```text
dirty_probe_response passes: R2 = 0.9955421082831637
change_to_dirty fails: R2 = 0.007805525669818403
cpu_cycles fails: R2 = 0.6863528000322364
duration_ns fails: R2 = 0.6635832051280859
```

The hardened self-test rejects:

```text
change_to_dirty-only replay
cpu_cycles-channel replay
duration-channel replay
legacy map-sign inversion replay
source-off paired smuggle
flat query_A/query_B signal
swapped query-pair values
negated q labels
```

This is retrospective analysis repair. It is evidence that the first-light data contains a strong public tomography signal, not by itself a final wall-crossing claim.
