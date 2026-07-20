# Family 10h Deep Operator-Dimension Hunt

Result: `FAMILY10H_SECOND_OPERATOR_DIMENSION_UNRESOLVED_RETROSPECTIVE`

Scope: offline-only exploratory follow-up over the sealed v1.1 attempt-1 archive. No target contact, PMU acquisition, runtime execution, SSH, SCP, deployment, or cleanup was performed.

## What Changed

This pass explicitly tests the easy-to-miss transforms: source-order aligned order contrasts, mapping-aligned order contrasts, log/ratio order contrasts, factorial symmetric and antisymmetric terms, factorial `change_to_dirty` interactions, Hadamard arm-balance coordinates, sham/carrier negative controls, timing-normalized terms, and the tempting cross-order residual.

## Main Findings

- The strongest new-looking coordinate, `CrossOrderResid`, is rejected because `(query_A_then_B - query_A) - (query_B_then_A - query_B) = D_order - D_single`; it is dominated by the already-confirmed scalar q readout.
- The sham-referenced difference also cancels to exactly `D_single`, so it is now a mechanical scalar-leakage rejection.
- Source-order and mapping-aligned order transforms reduce the scalar contamination but do not produce a stable held-out operator law.
- Sham/carrier terms expose a large ordinary carrier/query route offset, not a relation-specific coordinate.
- The q=0 source-off cube separates true active response from source-off null for `D_order`, but the sham/carrier route offset is nearly the same active and source-off, so it is not a carrier-state dimension.
- Factorial `change_to_dirty` and Hadamard arm coordinates are present but sparse/noisy; they do not supply a clean held-out operator law.
- Hadamard parity over mapping and source-order shows an invariant q-even curvature lead, but it is q=0 referenced, scalar-replayable, and fails session/replicate transport.
- Timing and cycle transforms are diagnostic only and do not survive as primary carrier coordinates.
- Algebra-preserving residual geometry now uses only independent measured base columns, not derived columns shuffled as if they were independent evidence.

## Top Candidate Ranking

- `OrderedVsShamExcess`: family `negative_control_referenced`, corr(q) `0.061847377318885365`, mean held-out rel RMSE `6.937139128603394`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata; diagnostic negative-control/timing channel, not a primary carrier operator coordinate.
- `OrderedVsSingleExcess`: family `ordered_total`, corr(q) `0.052881303394646466`, mean held-out rel RMSE `6.300145003358546`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.
- `K_L_C2D`: family `factorial_c2d`, corr(q) `-0.0625852220736158`, mean held-out rel RMSE `4.63608379810509`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.
- `K_L_C2D_per_cycle`: family `factorial_c2d_exposure`, corr(q) `-0.06725229611558771`, mean held-out rel RMSE `4.389297511550003`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.
- `K_C2D`: family `factorial_c2d`, corr(q) `0.03970296438186154`, mean held-out rel RMSE `2.8580433087610624`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.
- `K_T_C2D`: family `factorial_c2d`, corr(q) `0.00561016381916292`, mean held-out rel RMSE `2.755799450606883`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.
- `K_T_C2D_per_cycle`: family `factorial_c2d_exposure`, corr(q) `0.006112362286388364`, mean held-out rel RMSE `2.6748910505542765`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.
- `K_T_dirty`: family `factorial_hadamard`, corr(q) `0.005056072251842121`, mean held-out rel RMSE `2.3464503169308464`, clean candidate `False`
  Rejected: no strong between-state structure after matched within-state noise; q-profile or transformation profile is not stable across required strata.

## Residual Geometry

- residual effective rank: `12.86105650084094`
- second singular value above column-shuffle p95: `True`
- second singular value above column-shuffle p99: `True`
- minimum second-axis alignment across required strata: `0.011908297788603826`
- stable second axis across required strata: `False`

## Decision

No clean second operator dimension survived. The strongest apparent signals are either algebraic scalar-q leakage, negative-control route offsets, or residual axes that do not remain stable across the required session/replicate/mapping/delay/source-order strata.

This does not establish full carrier-state tomography, relational carrier, physical relational memory, catalytic borrowing, or `SMALL_WALL_CROSSED`.

## Smallest Next Grammar

The next useful experiment should create a relation-only contrast: hold `query_A`, `query_B`, `D_single`, total work, route pressure, query count, source order, and delay fixed while transforming only the A/B address relation. Include relation-preserving, relation-swapped, relation-distance-shifted, and relation-sham layouts plus scalar replay and route-pressure controls.
