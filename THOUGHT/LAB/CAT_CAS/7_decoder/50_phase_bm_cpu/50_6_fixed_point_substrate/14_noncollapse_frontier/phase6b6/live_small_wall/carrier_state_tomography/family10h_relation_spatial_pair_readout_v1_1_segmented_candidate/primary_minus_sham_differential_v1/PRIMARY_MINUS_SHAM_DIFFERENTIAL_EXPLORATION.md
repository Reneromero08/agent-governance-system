# Primary-Minus-Sham Differential Exploration

Archive SHA-256: `54ca4dcd53b300d4fe2857e4f9366a754b4316b1f951daa735a5de3bb4f16e89`
Archive size: `110187151` bytes
Copy-back verified: `True`
Analysis SHA-256: `5a613977264935d04fa5382ebb4856c1590320774e543c82e4fcffd5fcb4f63d`

Diagnostic disposition: `PRIMARY_MINUS_SHAM_DIFFERENTIAL_REPRODUCED_WITH_GENERIC_CONTROL_Q99_WARNINGS`
Coordinate freeze status: `FAMILY10H_PRIMARY_MINUS_SHAM_DIFFERENTIAL_COORDINATE_FROZEN_FOR_PROSPECTIVE_PACKAGE`

## Round Results

| Round | R_primary | R_sham | D = primary - sham | max abs generic | generic/D | generic <= 0.25 D | generic q99 all pass |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.058764853 | -0.020682295 | 0.079447149 | 0.008175433 | 0.103 | `True` | `False` |
| 1 | 0.068888641 | -0.009604907 | 0.078493548 | 0.008869480 | 0.113 | `True` | `False` |

## Query Means

| Round | Query | mean R_spatial | abs(mean) | q99 null | null exceeded |
|---:|---|---:|---:|---:|---|
| 0 | `query_relation_pair` | 0.058764853 | 0.058764853 | 0.008469489 | `True` |
| 0 | `relation_sham` | -0.020682295 | 0.020682295 | 0.006228310 | `True` |
| 0 | `scrambled_pair_control` | 0.005759887 | 0.005759887 | 0.006549076 | `False` |
| 0 | `distance_matched_control` | -0.005919383 | 0.005919383 | 0.006530858 | `False` |
| 0 | `route_pressure_control` | -0.008175433 | 0.008175433 | 0.006828173 | `True` |
| 1 | `query_relation_pair` | 0.068888641 | 0.068888641 | 0.008135467 | `True` |
| 1 | `relation_sham` | -0.009604907 | 0.009604907 | 0.006051137 | `True` |
| 1 | `scrambled_pair_control` | 0.008869480 | 0.008869480 | 0.007542406 | `True` |
| 1 | `distance_matched_control` | 0.001522399 | 0.001522399 | 0.006700747 | `False` |
| 1 | `route_pressure_control` | 0.001404704 | 0.001404704 | 0.007062131 | `False` |

## Factor-Stratum Gate

Primary positive, sham negative, and differential positive: `28` / `28` strata.

## Interpretation

The signed primary-minus-sham contrast reproduced in both fresh reset rounds and in every reported factor stratum. `relation_sham` behaves as an anti-relation comparator rather than a null control. The three generic controls remain small relative to the differential, but two isolated generic-control segments exceeded their own tiny matched-permutation q99 null. The next prospective package should therefore freeze the differential coordinate and use a predeclared relative generic-control envelope rather than treating relation_sham as a null.

This does not promote the claim boundary or Small Wall state.
