# Stationary Memory Strata

## Status

```text
STATIONARY_MEMORY_STRATA_CLASSIFIED
BOTH_MEMORY_COORDINATES_INTERIOR_EXCLUDED
EIGHT_PER_CLAUSE_STRATUM_TYPES_REMAIN
PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Stationary equations

For one clause with exact violation `C`, short memory `s`, long memory `l`, and long
memory cap `L`:

```text
s_dot = beta (C-gamma) s(1-s)
l_dot = alpha (C-delta) l(1-l/L).
```

At a stationary point:

```text
s in {0,1} or C = gamma
l in {0,L} or C = delta.
```

Because the frozen thresholds satisfy

```text
gamma = 1/4 != 1/20 = delta,
```

both memory coordinates cannot be interior simultaneously.

## Complete per-clause classification

Every stationary clause lies in one of eight algebraic strata:

```text
s in {0,1}, l in {0,L}, C arbitrary                 (4 types)
s interior with C = gamma, l in {0,L}               (2 types)
s in {0,1}, l interior with C = delta               (2 types)
```

No ninth both-interior type exists.

## Consequence

Combined with the exact memory-logit clock, this reduces the non-solution recurrence
problem to memory-boundary geometry. A proof no longer needs to classify arbitrary
interior cycles. It must show that the declared public seed cannot approach or remain on
any non-solution invariant set assembled from these eight per-clause strata before
reaching a satisfying terminal section.

The known satisfiable non-solution stationary example lies in the

```text
SHORT_ONE__LONG_CAP__C_ARBITRARY
```

stratum for each clause.

This classification does not prove that every other stratum is harmless, nor that the
public seed avoids their stable manifolds. It makes that remaining task finite at the
local memory level while leaving global phase coupling unresolved.
