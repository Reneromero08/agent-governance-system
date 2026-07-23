# Local Paired Differential Adjudication

Result class: `FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CONFIRMED_PROSPECTIVE`
Scientific claim: `PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_CONFIRMED`
Archive SHA-256: `b2137e3107d15f42659e304213768763047eb42ffca6e838230a2862f9cc4017`
Adjudication SHA-256: `14866d3c25cde02f18c269c0498527380d437edf2db7a2d28fea4cd09330358c`

## Round Results

| Round | R_primary | R_sham | D | max abs generic | generic/D | envelope | sham<0 diagnostic |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.054808499 | -0.003600438 | 0.058408937 | 0.003549561 | 0.061 | `True` | `True` |
| 1 | 0.089138930 | 0.001846948 | 0.087291982 | 0.001861141 | 0.021 | `True` | `False` |

This replay applies the frozen prospective local paired differential law: `D = R_primary - R_sham`, `R_primary > 0`, one-factor matched stratum ordering, and the `0.25 * D` generic-control envelope.

The original matched-permutation q99 diagnostic path timed out locally after acquisition. That diagnostic is not a gate for this local paired law; thresholds and result gates were not changed.

The adjudicator aggregation bug was repaired by moving false claim-boundary invariants, including `small_wall_crossed = false`, out of the pass-gate set.

Claim boundary: no full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing is established by this package.
