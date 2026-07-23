# Relation R-Spatial Shift Sweep Discovery

Run ID: `family10h_relation_rspatial_shift_sweep_discovery_v1_0`
Archive SHA-256: `f3500f3dc1907eae5a1e0ea2eadb722b4dec7a9ec5d2d54f1fbfa374c7b61d0f`
Analysis SHA-256: `931be822322a7664c78116d59b9605a338b8cd37642335b7c0db868bb6740f8d`

| Shift | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(shift 1 D) | max stratum ratio | all strata D+ | all strata D- |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.054762984 | -0.008839211 | 0.063602196 | 1.000 | 1.000 | True | False |
| 2 | 0.071007567 | 0.024985932 | 0.046021635 | 0.724 | 1.558 | True | False |
| 4 | -0.000831228 | -0.003516174 | 0.002684946 | 0.042 | 0.457 | False | False |
| 8 | 0.027033135 | -0.002968400 | 0.030001534 | 0.472 | 0.763 | False | False |
| 16 | 0.002983199 | 0.001853378 | 0.001129821 | 0.018 | 0.269 | False | False |
| 32 | 0.068123575 | 0.001164887 | 0.066958688 | 1.053 | 2.559 | True | False |
| 64 | -0.003401004 | -0.008944149 | 0.005543146 | 0.087 | 0.320 | False | False |
| 128 | 0.007098278 | -0.000707116 | 0.007805394 | 0.123 | 0.440 | False | False |

| Shift | prep/query phase rad | prep/query mag ratio | diag/skew phase rad | diag/skew mag ratio |
|---:|---:|---:|---:|---:|
| 1 | -0.846705 | 1.000 | 0.794139 | 1.000 |
| 2 | -2.748835 | 0.378 | -0.201290 | 0.518 |
| 4 | 0.002238 | 0.257 | 1.346493 | 0.133 |
| 8 | 0.180279 | 0.374 | 0.431279 | 0.364 |
| 16 | 1.667994 | 0.251 | -1.481256 | 0.139 |
| 32 | -0.407362 | 0.819 | 0.635073 | 0.917 |
| 64 | 2.400341 | 0.264 | -1.257730 | 0.198 |
| 128 | -1.786399 | 0.320 | 0.962255 | 0.150 |

Discovery interpretation:
- D sign changes across shifts: `False`
- all shift D positive: `True`
- best absolute-D shift: `32`
- log2(shift) Spearman vs D: `-0.261905`
- log2(shift) Spearman vs abs(D): `-0.261905`
- oriented shift-selectivity candidate: `False`

This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.
