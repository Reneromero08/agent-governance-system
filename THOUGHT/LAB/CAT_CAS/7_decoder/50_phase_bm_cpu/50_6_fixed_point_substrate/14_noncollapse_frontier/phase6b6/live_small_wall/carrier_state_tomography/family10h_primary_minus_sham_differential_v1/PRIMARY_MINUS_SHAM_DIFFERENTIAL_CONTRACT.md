# Family 10h Primary-Minus-Sham Differential Package

This package implements the frozen coordinate in:

```text
family10h_relation_spatial_pair_readout_v1_1_segmented_candidate/
primary_minus_sham_differential_v1/
PROSPECTIVE_PRIMARY_MINUS_SHAM_DIFFERENTIAL_COORDINATE.md
```

Coordinate:

```text
R_primary = mean R_spatial(query_relation_pair)
R_anti = mean R_spatial(relation_sham)
D_primary_minus_sham = R_primary - R_anti
```

Acquisition law:

- Use fresh carrier state and a fresh runtime process for each segment.
- Do not interleave primary, anti-relation, or generic-control rows in one raw schedule.
- Run two fresh independently reset rounds.
- Preserve factor strata for session, replicate, mapping, source order, query order, and cyclic origin.

Prospective success law:

- Every round: `R_primary > 0`.
- Every round: `R_anti < 0`.
- Every round: `D_primary_minus_sham > 0`.
- Every reported factor stratum for session, replicate, mapping, source order, query order, and cyclic origin: `R_primary > 0`, `R_anti < 0`, and `D_primary_minus_sham > 0`.
- Generic controls pass when `max(abs(R_scrambled), abs(R_distance), abs(R_route)) <= 0.25 * D_primary_minus_sham` in every round.
- Generic matched-permutation q99 results are reported diagnostics. They are not the generic-control gate.
- `relation_sham` is the anti-relation comparator and must not be included in the null-control set.

Claim boundary:

This package can confirm only the primary-minus-sham relation coordinate. It does not establish full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing.
