# Prospective Primary-Minus-Sham Differential Coordinate

Status: `FAMILY10H_PRIMARY_MINUS_SHAM_DIFFERENTIAL_COORDINATE_FROZEN_FOR_PROSPECTIVE_PACKAGE`

Coordinate:

```text
R_primary = mean R_spatial(query_relation_pair)
R_anti = mean R_spatial(relation_sham)
D_primary_minus_sham = R_primary - R_anti
```

Interpretation:

- `query_relation_pair` is the positive relation geometry.
- `relation_sham` is the anti-relation comparator, not a null control.
- `scrambled_pair_control`, `distance_matched_control`, and `route_pressure_control` remain generic null controls.

Prospective acquisition law:

- Use fresh carrier state and a fresh runtime process for each segment.
- Do not interleave primary, anti-relation, or generic-control rows in one raw schedule.
- Run at least two fresh independently reset rounds.
- Preserve factor strata for session, replicate, mapping, source order, query order, and cyclic origin.

Prospective coordinate success law:

- Every round: `R_primary > 0`.
- Every round: `R_anti < 0`.
- Every round: `D_primary_minus_sham > 0`.
- Every reported factor stratum for session, replicate, mapping, source order, query order, and cyclic origin: `R_primary > 0`, `R_anti < 0`, and `D_primary_minus_sham > 0`.
- Generic controls are true null controls under the frozen relative envelope: `max(abs(R_scrambled), abs(R_distance), abs(R_route)) <= 0.25 * D_primary_minus_sham` in every round.
- Generic matched-permutation q99 results must be reported, but `relation_sham` must not be included in the null-control set.
- No target-derived threshold revision or post-observation feature selection is allowed.

Claim boundary:

- This freezes the next prospective coordinate only.
- It does not establish full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing.
