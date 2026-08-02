# Public-Seed Memory Reduction

## Status

```text
PUBLIC_SEED_MEMORY_ODDS_RELATION_EXACT
FINITE_TIME_MEMORY_BOUNDARY_CONTACT_EXCLUDED
THREE_STATIONARY_STRATA_FORWARD_INCOMPATIBLE
FIVE_FORWARD_COMPATIBLE_STATIONARY_STRATA_REMAIN
PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Exact memory chart

For one clause, let

```text
s in (0,1)                 short memory
l in (0,L)                 long memory
r = l/L in (0,1)           normalized long memory
C                           exact clause violation
```

and define the logits

```text
u = log(s/(1-s))
v = log(r/(1-r)).
```

The native memory equations give

```text
u_dot = beta (C-gamma)
v_dot = alpha (C-delta).
```

Therefore

```text
D = v/alpha - u/beta
D_dot = gamma-delta,
```

and for every finite trajectory segment

```text
v(t) = (alpha/beta) u(t)
       + alpha D(0)
       + alpha (gamma-delta) t.
```

Equivalently,

```text
odds(r(t))
= exp(alpha D(0))
  exp(alpha (gamma-delta) t)
  odds(s(t))^(alpha/beta).
```

This identity is formula-independent because the clause-violation term cancels.

## Public-seed specialization

The declared public seed uses

```text
s(0) = 1/2
l(0) = 1
L = long_memory_cap_factor * max(1,m).
```

Hence

```text
odds(s(0)) = 1
odds(r(0)) = 1/(L-1).
```

For the frozen parameters

```text
alpha = 5
beta = 20
gamma = 1/4
delta = 1/20,
```

we have

```text
alpha/beta = 1/4
alpha(gamma-delta) = 1.
```

Thus every clause on every public-seed trajectory satisfies the exact relation

```text
odds(r(t)) = e^t/(L-1) * odds(s(t))^(1/4).
```

The long-memory coordinate is therefore not an independent public-seed degree of
freedom. It can be reconstructed from short memory and elapsed time.

## Finite-time boundary exclusion

The clause violation is bounded on the phase circles. Therefore the logit derivatives
are bounded on every finite interval. Starting from interior memory coordinates, both
`u(t)` and `v(t)` remain finite for finite `t`, so

```text
0 < s(t) < 1
0 < l(t) < L
```

for every finite deadline.

Memory-boundary strata can only occur as asymptotic limit sets. They cannot be reached in
finite native time from the declared public seed.

## Forward-compatible stationary strata

The exact odds relation excludes three of the eight algebraically stationary per-clause
strata from any public-seed omega limit.

### Excluded

```text
SHORT_ONE__LONG_ZERO__C_ARBITRARY
SHORT_INTERIOR_C_EQ_GAMMA__LONG_ZERO
SHORT_ONE__LONG_INTERIOR_C_EQ_DELTA
```

Reason:

- if short memory tends to one, `odds(s)` diverges, and the additional `e^t` factor
  forces normalized long memory to tend to one;
- if short memory remains interior, the same `e^t` factor forces normalized long
  memory to tend to one.

### Still compatible

```text
SHORT_ZERO__LONG_ZERO__C_ARBITRARY
SHORT_ZERO__LONG_CAP__C_ARBITRARY
SHORT_ONE__LONG_CAP__C_ARBITRARY
SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP
SHORT_ZERO__LONG_INTERIOR_C_EQ_DELTA
```

When a stationary limit has a limiting clause violation `C_*`, these five classes align
with the threshold ordering `delta < gamma`:

```text
C_* < delta          -> short zero, long zero
C_* = delta          -> short zero, long interior
 delta < C_* < gamma -> short zero, long cap
C_* = gamma          -> short interior, long cap
C_* > gamma          -> short one, long cap.
```

This is a classification of forward-compatible stationary memory behavior, not a proof
that each class contains a globally invariant phase set.

## Reduced proof target

The public-seed completeness proof no longer needs to attack all eight stationary memory
strata. It must exclude non-solution invariant sets assembled from the five
forward-compatible classes above.

The exact next questions are:

1. Which phase and selector configurations make each remaining stratum invariant?
2. What are the normal stability exponents of those invariant sets?
3. Can the declared public seed or any declared presentation gauge lie in their stable
   manifolds?
4. Can escape time from every non-solution neighborhood be bounded by one polynomial in
   the public formula size?

This reduction does not establish eventual convergence, a polynomial deadline, a robust
terminal margin, or `P = NP`.
