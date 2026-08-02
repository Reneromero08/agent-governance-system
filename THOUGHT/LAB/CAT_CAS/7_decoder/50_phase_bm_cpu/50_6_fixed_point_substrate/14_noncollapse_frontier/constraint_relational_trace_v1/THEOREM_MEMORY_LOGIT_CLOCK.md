# Exact Memory-Logit Clock

## Status

```text
MEMORY_LOGIT_CLOCK_ESTABLISHED
INTERIOR_PERIODIC_ORBITS_EXCLUDED
BOUNDED_INTERIOR_RECURRENCE_EXCLUDED
OMEGA_LIMIT_SET_MUST_APPROACH_MEMORY_BOUNDARY_STRATA
PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Identity

For every public clause, let

```text
u = log(s/(1-s))
v = log((l/L)/(1-l/L)),
```

where `s` is short memory, `l` is long memory, and `L` is its public cap. The native
memory equations imply exactly

```text
u_dot = beta (C-gamma)
v_dot = alpha (C-delta).
```

Therefore

```text
d/dt (v/alpha - u/beta) = gamma-delta.
```

The right-hand side is independent of the clause violation `C`, phase state, formula,
selectors, and memory values. With the frozen public parameters

```text
gamma = 1/4
delta = 1/20,
```

the normalized difference drifts at the constant rate

```text
gamma-delta = 1/5.
```

## Consequences

No trajectory that remains in the interior memory chart can be periodic. More strongly,
no bounded recurrent orbit can remain entirely in that interior, because the exact clock
coordinate changes linearly forever.

Thus any non-solution omega-limit behavior must approach a memory boundary stratum:

```text
short memory -> 0 or 1
or
long memory -> 0 or its public cap.
```

This sharply reduces the recurrent-set analysis. The campaign no longer needs to rule
out arbitrary interior limit cycles. It must classify boundary-stratum stationary sets,
heteroclinic chains, and switching behavior reachable from the declared public seed.

## Relationship to the stationary-point wall

The known satisfiable non-solution stationary carrier lies on memory boundaries. It is
therefore consistent with this theorem. The public seed starts in the interior and is
not stationary, but the clock identity alone does not prove it cannot approach a
non-solution boundary stratum.

## Remaining theorem

The central statement is now:

```text
Every satisfiable public formula, from the declared public seed and under every declared
presentation gauge, reaches a robust satisfying phase section before approaching any
non-solution memory-boundary invariant set, within one formula-uniform polynomial
deadline.
```

This theorem is not yet established.
