# Family 10h Segmented Relation-Spatial Candidate Freeze

Status: `DISCOVERY_CANDIDATE_READY_FOR_PROSPECTIVE_PACKAGE_VALIDATION`

This is a discovery-only freeze record. It does not establish a prospective claim and does not cross the Small Wall.

## Found Candidate

The live discovery runs show that the original full schedule suppressed the relation coordinate by interleaving raw control rows inside the carrier sequence. The clean shape is segmented:

1. run the primary four-cell relation matrix first with `query_relation_pair`;
2. run full R-shaped control matrices after it for `relation_sham`, `scrambled_pair_control`, `distance_matched_control`, and `route_pressure_control`;
3. adjudicate primary and controls with the same R-spatial algebra, not raw one-row `C_pair` versus an R null.

## Key Live Numbers

- Initial all-B relation-only: ratio `12.697` over discovery q99.
- Repeat all-B relation-only: ratio `11.496`.
- All-A relation-only: ratio `11.580`.
- Alternating relation-only: ratio `11.677`.
- Combined transaction primary segment: abs mean `0.081525`, q99 `0.006812`, ratio `11.967`.

Combined controls all stayed below their own matched q99 and were not factor-stable. Largest combined control abs mean: `0.003412`.

## Next Freeze Rule

A prospective v1.1 package should freeze the segmented schedule and fail closed unless:

- primary `query_relation_pair` R-spatial exceeds matched-permutation q99;
- primary sign is stable across session, mapping, source order, query order, and cyclic origin;
- every R-shaped control is at or below its own matched-permutation q99;
- primary abs mean is at least 4x the largest control abs mean;
- custody, no-smuggle, and no-positive-claim leakage gates pass.

Maximum next claim remains a public spatial relation-coordinate readout claim only. No full tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing is established by this record.

Freeze SHA-256: `6366dbb67f7b1ff09252d0451a7dd91e8e776904c2884c643936ecbe80f01aa9`
