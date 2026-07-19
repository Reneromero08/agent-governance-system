# Prospective Paired Dirty-Probe Tomography Confirmation Contract

This contract freezes the repaired analysis law for the next confirmation package. It does not authorize live execution.

## New Identity

```text
science package = family10h_carrier_tomography_v1_1_paired_dirty_probe
base evidence = family10h_carrier_tomography_v1_0 attempt_3 diagnostic only
claim ceiling = public Family 10h carrier-state tomography
small wall promotion = forbidden by this contract
```

## Observable

The primary observable is the matched paired dirty-probe contrast:

```text
D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)
```

The matched key is frozen as:

```text
session, replicate, delay_label, mapping, source_order, q, source_off_control
```

`query_A` and `query_B` are logical query names. Because the runtime maps query lanes through the same `map_variant` used during preparation, map0/map1 must be evaluated as consistency strata, not as a sign inversion.

## Fixed Thresholds

The next run must use these thresholds without post-run revision:

```text
source_off abs(D_single) max <= 128
source_off mean abs(D_single) <= 32
q=0 abs mean D_single per stratum <= 64
abs(mean D_single at q=1536) >= 2000 in every stratum
abs(mean D_single at q=-1536) >= 2000 in every stratum
linear q model R2 >= 0.98 in every stratum
ordinary least-squares intercept abs <= 64
odd-symmetry relative residual <= 0.10 for q pairs 512, 1024, and 1536
signal/null ratio >= 20
held-out replicate relative RMSE <= 0.10
held-out mapping relative RMSE <= 0.10
held-out delay relative RMSE <= 0.10
held-out session relative RMSE <= 0.10
held-out source-order relative RMSE <= 0.10
map0/map1 slope relative disagreement <= 0.10
delay slope relative disagreement <= 0.10
session slope relative disagreement <= 0.10
source-order slope relative disagreement <= 0.10
nearest-q exact classifier accuracy >= 0.95
nearest-q nonzero sign classifier accuracy = 1.0
```

## Gates

A passing prospective confirmation requires:

```text
original custody and exact evidence validation pass
all query_A/query_B pairs complete
query_A, query_B, query_A_then_B, and query_B_then_A observations remain separately recorded
ordered-query observations are reported separately and are not used to fit D_single
active cells balanced across session, replicate, mapping, delay_label, source_order, and q
source-off cells balanced across session, replicate, mapping, delay_label, and source_order
all source-off controls pass
all source-off controls pass separately by session, replicate, mapping, delay_label, and source_order
all q-sign and q-magnitude laws pass in every session and replicate
odd symmetry passes globally and in every session/replicate stratum
linear intercept bound passes globally and in every session/replicate stratum
all held-out session/replicate/mapping/delay/source-order tests pass
all held-out session/replicate/mapping/delay/source-order nearest-q classifiers pass
map, delay, session, and source-order slope consistency pass
dirty_probe_response passes while change_to_dirty, cpu_cycles, and duration_ns fail the same paired law
no target-derived feature selection
no post-run threshold revision
no SMALL_WALL_CROSSED promotion from this package alone
```

## Required Negative Regressions

The package must reject:

```text
change_to_dirty-only replay
cpu_cycles-channel replay
duration-channel replay
legacy map-sign inversion replay
source-off paired smuggle
flat query_A/query_B signal
swapped query-pair values
negated q labels
invalid/missing packet rows
```
