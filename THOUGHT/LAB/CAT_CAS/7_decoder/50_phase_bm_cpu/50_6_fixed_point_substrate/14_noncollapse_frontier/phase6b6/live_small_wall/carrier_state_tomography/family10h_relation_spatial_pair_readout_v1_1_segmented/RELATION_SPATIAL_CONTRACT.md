# Family 10h Spatial Pair Readout V1

This frozen package tests whether relation geometry is readable through spatially resolved first-touch latency pairs.

- q is fixed at 0.
- source_lifetime is fixed to alive_during_query.
- each row measures 256 deterministic A/B line pairs exactly once.
- primary coordinate: C_pair Spearman rank correlation.
- relation coordinate: R_spatial = 0.5 * (r0/r0 + r1/r1 - r0/r1 - r1/r0).
- calibrated result requires exceeding the frozen matched-permutation q99 null plus all stability and custody gates.
- no result in this package may claim full tomography, relation memory, R2 restoration, catalytic borrowing, or SMALL_WALL_CROSSED.
