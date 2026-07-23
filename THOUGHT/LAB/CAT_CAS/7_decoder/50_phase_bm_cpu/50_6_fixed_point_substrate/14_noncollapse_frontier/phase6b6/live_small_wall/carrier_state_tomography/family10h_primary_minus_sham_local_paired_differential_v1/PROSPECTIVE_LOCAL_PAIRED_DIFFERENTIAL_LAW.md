# Prospective Local Paired Differential Law

Package identity:

```text
family10h_primary_minus_sham_local_paired_differential_v1
```

Purpose:

Freeze the smallest prospective repair to the primary-minus-sham differential law. The repaired coordinate treats `relation_sham` as a local anti-relation comparator whose absolute sign may drift across fresh carrier resets. The prospective coordinate is the paired differential/order statistic:

```text
D_local = R_primary - R_sham
R_primary = mean R_spatial(query_relation_pair)
R_sham = mean R_spatial(relation_sham)
```

The law does not require `R_sham < 0`. The old absolute-sham-sign gate is retired for this coordinate because retained evidence shows that the sham baseline can cross zero while the local primary-minus-sham ordering persists.

## Frozen Acquisition Shape

- Reuse the existing Family 10h relation-spatial runtime, PMU helper, source/receiver boundary, target identity, sensor identity, factors, and five query variants.
- Use fresh carrier state and a fresh runtime process for each segment.
- Run the same two independently reset rounds.
- Preserve the existing factor strata: session, replicate, mapping, source order, query order, and cyclic origin.
- Do not add new queries, factors, thresholds, post-run feature selection, or schedule edits.
- Preserve generic matched-permutation q99 as a reported diagnostic only.

## Prospective Success Law

All of the following must pass in a fresh prospective physical attempt:

1. All custody, source lifecycle, target identity, sensor identity, runtime hash, PMU preflight, manifest, schedule, and no-fixture gates pass.
2. Every round has `D_local > 0`.
3. Every round has `R_primary > 0` as a source-orientation sanity gate.
4. Every one-factor matched stratum for session, replicate, mapping, source order, query order, and cyclic origin has `D_local > 0`.
5. Every one-factor matched stratum has `R_primary > 0`.
6. No stratum or round requires `R_sham < 0`.
7. Generic controls pass the existing envelope in every round:

```text
max(abs(R_scrambled), abs(R_distance), abs(R_route)) <= 0.25 * D_local
```

8. `relation_sham` remains excluded from the generic-null set.
9. No post-observation threshold revision or target-derived fitting is permitted.

The six-factor crossed-cell audit is diagnostic only for this law. Each complete crossed cell has four block samples in the retained schedule; freezing a full crossed-cell sign gate would overfit sparse cells and change the experiment more than this repair requires.

## Result Vocabulary

```text
FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CONFIRMED_PROSPECTIVE
FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_NOT_CONFIRMED_PROSPECTIVE
FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CUSTODY_INVALID
```

Maximum scientific claim:

```text
PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_CONFIRMED
```

This law cannot establish full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing by itself.

## Freeze Basis

This law is frozen from offline audit `2826e10317dd96d7f73c8471e0e6365b466a003fa59dde8c88ec200a0d1b3cf9` of the retained exploratory and official physical evidence. The audit supports a minimal prospective confirmation package, not retrospective promotion.
