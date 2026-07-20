# Family 10h Relation-Only Matched-Permutation Contract

Package decision: `FAMILY10H_RELATION_ONLY_PACKAGE_FROZEN_AWAITING_AUTHORIZATION`

Transaction identity: `family10h_relation_only_matched_permutation_v1_0`

This is an offline source and validation package. It does not authorize target contact, SSH, SCP, PMU acquisition, runtime execution, deployment, cleanup, borrowing, restoration, or any Small Wall promotion.

## Confirmed Baseline

The prior Family 10h v1.1 result confirms only the scalar primitive:

`D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)`

The relation-only package must preserve that scalar coordinate. A relation result is invalid if relation labels change q slope, q intercept, scalar classifier behavior, scalar source-off residual, scalar mapping consistency, or scalar delay consistency beyond the frozen equivalence law.

## Relation Grammar

Two public address relations are frozen:

- `relation_r0`: `B_index = (A_index + 1) mod 4096`
- `relation_r1`: `B_index = (A_index - 1) mod 4096`

They are distinct inverse one-step permutations over the same A and B logical line sets. The machine-readable proof in `RELATION_MARGINAL_EQUALITY_PROOF.json` binds equal A/B address sets, work counts, total work, page counts, line counts, cache-set histograms, pair-distance histograms, cycle-structure class, source/receiver loop lengths, branch structure, source/receiver CPU, PMU group, delay distribution, source-order counts, query-order counts, allocation behavior, and prefault behavior.

## Primary Matrix

The source prepares `r_prepare`. The receiver queries `r_query`.

Every matched condition acquires a complete fresh-preparation matrix:

- prepare `relation_r0`, query `relation_r0`
- prepare `relation_r0`, query `relation_r1`
- prepare `relation_r1`, query `relation_r0`
- prepare `relation_r1`, query `relation_r1`

The primary endpoint is `dirty_probe_response`.

The primary relation law is:

`R_match = 0.5 * ((Y_r0_r0 + Y_r1_r1) - (Y_r0_r1 + Y_r1_r0))`

Either sign may be meaningful. Sign or magnitude alone is insufficient: scalar, route, timing, distance, order, and label-leak controls must pass prospectively.

## Required Controls

The public schedule freezes:

- scalar controls: `query_A`, `query_B`
- relation sham
- label-scramble control
- query-relation mismatch
- source-relation mismatch
- route-pressure sham
- independent-marginal replay
- source-order counterbalance
- query-order counterbalance
- distance-preserved relation pair

The schedule uses public block randomization and Latin-square counterbalancing. Tuple IDs are opaque hashes and must not encode relation labels.

## Prospective Candidate Law

A future physical relation coordinate may be considered only if all of the following pass without aggregate rescue:

- exact evidence and custody validation
- complete relation matrix
- exact matched scalar marginals
- exact schedule balance
- stable nonzero `R_match`
- relation-sham null
- label-scramble collapse
- scalar q equivalence across relation variants
- held-out session, replicate, mapping, delay, source order, query order, and q where valid
- best scalar-replay adversary fails
- best separable A/B marginal model fails
- route-pressure model fails
- distance-only model fails or is separately bounded
- relation transform follows the preparation-query match law
- matched permutation nulls fail to reproduce the result
- bootstrap or resampling stability survives

## Result Vocabulary

Future physical result classes are predeclared:

- `FAMILY10H_RELATION_MATCH_COORDINATE_CONFIRMED_PROSPECTIVE`
- `FAMILY10H_RELATION_MATCH_COORDINATE_NOT_CONFIRMED_PROSPECTIVE`
- `FAMILY10H_RELATION_MATCH_COORDINATE_CANDIDATE_PROSPECTIVE`
- `FAMILY10H_RELATION_MATCH_COORDINATE_CUSTODY_INVALID`

Maximum future claim:

`PUBLIC_POST_SOURCE_RELATION_MATCH_COORDINATE_CONFIRMED`

Even a passing relation package does not establish full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or `SMALL_WALL_CROSSED`.

## Mechanism And Restoration Boundary

A future confirmed relation coordinate would feed later work in this order:

1. distinguish route interaction from address relation
2. identify physical carrier mechanism
3. design a controlled borrow or displacement
4. extract the relation coordinate
5. actively restore the relation coordinate
6. compare against a time-matched natural-relaxation control
7. test R2 vector equivalence across scalar and relation coordinates

Borrowing and restoration are not implemented here.
