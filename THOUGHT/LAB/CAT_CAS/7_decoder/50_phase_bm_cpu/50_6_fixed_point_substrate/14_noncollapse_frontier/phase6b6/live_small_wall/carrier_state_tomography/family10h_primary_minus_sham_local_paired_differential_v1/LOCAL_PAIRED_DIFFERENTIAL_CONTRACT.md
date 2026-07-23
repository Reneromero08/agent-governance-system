# Family 10h Local Paired Differential Package

This package implements the frozen law in:

```text
PROSPECTIVE_LOCAL_PAIRED_DIFFERENTIAL_LAW.md
```

Coordinate:

```text
R_primary = mean R_spatial(query_relation_pair)
R_sham = mean R_spatial(relation_sham)
D_local = R_primary - R_sham
```

Acquisition law:

- Use the existing Family 10h relation-spatial runtime, PMU helper, source/receiver boundary, target identity, sensor identity, and source schedule package.
- Use fresh carrier state and a fresh runtime process for each segment.
- Run two fresh independently reset rounds.
- Preserve factor strata for session, replicate, mapping, source order, query order, and cyclic origin.
- Do not add new factors, probes, controls, schedules, PMU events, or thresholds.

Prospective success law:

- Every round: `R_primary > 0`.
- Every round: `D_local > 0`.
- Every one-factor matched stratum for session, replicate, mapping, source order, query order, and cyclic origin: `R_primary > 0` and `D_local > 0`.
- Generic controls pass when `max(abs(R_scrambled), abs(R_distance), abs(R_route)) <= 0.25 * D_local` in every round.
- `relation_sham` is the local comparator and must not be included in the generic-null set.
- `R_sham < 0` is diagnostic only and must not be restored as a gate.
- Complete six-factor crossed cells are diagnostic only and must not be promoted to gates.
- Generic matched-permutation q99 results are reported diagnostics and are not the generic-control gate.

Claim boundary:

The maximum claim is:

```text
PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_CONFIRMED
```

This package does not establish full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing.

Live authority:

```text
live_attempt_authority = false
```
