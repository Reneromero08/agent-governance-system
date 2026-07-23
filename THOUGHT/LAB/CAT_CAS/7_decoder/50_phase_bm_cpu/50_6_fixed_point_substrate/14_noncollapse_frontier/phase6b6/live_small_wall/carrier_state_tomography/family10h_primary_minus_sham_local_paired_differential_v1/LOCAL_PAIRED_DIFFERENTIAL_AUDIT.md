# Local Paired Differential Audit

Audit SHA-256: `2826e10317dd96d7f73c8471e0e6365b466a003fa59dde8c88ec200a0d1b3cf9`
Decision: `FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_LAW_SUPPORTED_FOR_PROSPECTIVE_FREEZE`

## Diagnosis

The sham baseline crosses zero because its absolute mean is a weak near-zero baseline that is sensitive to fresh-process carrier offset. The position exploration shows the negative sham response reproduces when sham is alone, first, or after primary, so the official round-0 positive sham is not explained by primary-before-sham positioning alone.

The durable candidate is the local ordering: `query_relation_pair` stays above `relation_sham`, and the generic controls stay small relative to that separation. The repaired law therefore gates on `D = R_primary - R_sham`, not on the absolute sign of `R_sham`.

## Round Evidence

| Evidence | Round | R_primary | R_sham | D | sham<0 | D>0 | generic/D | envelope |
|---|---:|---:|---:|---:|---|---|---:|---|
| exploratory_differential | 0 | 0.058764853 | -0.020682295 | 0.079447149 | `True` | `True` | 0.103 | `True` |
| exploratory_differential | 1 | 0.068888641 | -0.009604907 | 0.078493548 | `True` | `True` | 0.113 | `True` |
| official_differential | 0 | 0.053253644 | 0.010112171 | 0.043141473 | `False` | `True` | 0.207 | `True` |
| official_differential | 1 | 0.075491216 | -0.004201392 | 0.079692608 | `True` | `True` | 0.100 | `True` |

## Local Strata

| Evidence | one-factor cells | D>0 | primary>0 | sham<0 | weakest D |
|---|---:|---:|---:|---:|---:|
| exploratory_differential | 28 | 28 | 28 | 28 | 0.052441903 |
| official_differential | 28 | 28 | 28 | 13 | 0.032292091 |

Complete six-factor crossed cells are retained as diagnostics, not gates. The retained schedule gives only four block samples per crossed cell, and those sparse cells invert in both exploratory and official evidence even while round and one-factor strata are stable.

## Generic Controls

All four differential rounds pass the retained `0.25 * D` generic-control envelope. q99 warnings remain diagnostics and are not removed.

## Frozen Next Step

The audit supports freezing the revised prospective local paired differential law. It does not retroactively confirm the official failed attempt, and it does not authorize a new live attempt. The smallest confirmation package is the existing segmented shape plus the repaired adjudication law that removes only `R_sham < 0`.

## Claim Boundary

Retrospective audit result: `FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_LAW_SUPPORTED_FOR_PROSPECTIVE_FREEZE`
Prospective law frozen: `True`
Small Wall crossed: `false`
