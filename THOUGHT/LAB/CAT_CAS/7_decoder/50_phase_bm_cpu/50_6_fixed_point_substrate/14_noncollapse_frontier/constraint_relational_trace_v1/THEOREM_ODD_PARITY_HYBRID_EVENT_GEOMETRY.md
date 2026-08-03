# Odd-Parity Public-Seed Hybrid Event Geometry

## Status

```text
ODD_PARITY_HYBRID_EVENT_GEOMETRY_CROSS_SOLVER_REFERENCE_PASS
FIRST_PUBLIC_SEED_WITNESS_ENTRY_TRANSVERSE
FIRST_WITNESS_DWELL_TIME_ORDER_ONE_ON_THIS_FORMULA
FIRST_ENTRY_ACTIVE_SET_GAP_SMALL_POSITIVE
FIRST_INTERVAL_GUARD_MARGIN_SMALL_POSITIVE
UNIFORM_PUBLIC_SEED_EVENT_LOWER_BOUND_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Formula and public trajectory

Use the exact four-clause CNF for odd parity on three variables:

```text
( x1 OR ~x2 OR ~x3)
(~x1 OR  x2 OR ~x3)
(~x1 OR ~x2 OR  x3)
( x1 OR  x2 OR  x3).
```

The declared public seed thresholds to `(True,False,True)`, which is not a witness. The
current exact-product flow later enters a witness orthant, exits it, and eventually
returns toward the non-solution midpoint manifold.

This audit resolves the first witness interval using the exact semialgebraic guard

```text
G_F(c) = min_j max_r q_jr c_i.
```

## Independent solver controls

The trajectory was integrated independently with:

```text
DOP853
Radau
```

using:

```text
relative tolerance = 1e-9
absolute tolerance = 1e-11
maximum step = 2e-2
deadline = 3.
```

Both solvers locate the first witness entry near

```text
t_entry approximately 0.208699.
```

They agree on entry time to better than `1e-6`, on entry transverse speed to better than
`1e-8`, on the entry active-set gap to better than `1e-10`, and on the maximum guard
margin to better than `1e-8`.

## First entry geometry

At the first entry, the crossing is a simple event after tolerance-based active-set
resolution. The measured reference values are approximately:

```text
guard directional derivative kappa approximately 2.905e-3
active-set gap Delta approximately 3.160e-7.
```

Thus this concrete public-seed event is transverse, but the separation between the
controlling min/max branch and its competitors is already small.

The event does not support a structural transversality theorem: an independent exact
native control already exhibits a smooth simple guard surface with zero normal velocity.
The positive derivative here is a trajectory-specific fact.

## First witness interval

The first witness sign interval persists until approximately:

```text
DOP853 exit time approximately 1.741
Radau exit time approximately 1.749.
```

Hence the first witness dwell time is approximately:

```text
1.53 to 1.54 native time units.
```

The maximum guard margin during this interval is only approximately:

```text
max G_F approximately 1.57e-4
```

and occurs near time one.

This combination is important:

```text
long dwell on this finite control
does not imply a large guard margin.
```

A trajectory may remain in a witness orthant for order-one time while staying close to
its threshold boundary.

## Relationship to the conditional transfer theorem

For a simple event, the established local transfer theorem requires lower bounds on:

```text
entry transverse speed kappa
active-set gap Delta
```

and uses the already established polynomial speed and acceleration upper bounds to
derive witness margin and dwell time.

This finite control supplies positive values for both quantities, but it does not prove
that either remains inverse-polynomial across formula size or presentation gauge.

The nonsmooth-cone alternative also remains relevant: positive directional entry can
occur with zero active-set gap. In that case the needed object is a uniform lower bound
on the lower directional guard derivative throughout a witness-entry cone.

## What this establishes

```text
one actual public-seed witness event is cross-solver reproducible
its entry is transverse rather than grazing
its first witness interval has order-one dwell time
its guard margin and active-set gap are small but positive
terminal verification at T=3 still fails after the transient interval.
```

## What remains open

This result does not establish:

```text
polynomial first-passage time for every satisfiable formula
inverse-polynomial crossing speed for every public seed
inverse-polynomial active-set gap or nonsmooth directional-cone margin
inverse-polynomial witness guard margin
polynomial event-location precision
presentation-gauge uniformity
recorder restoration
P = NP.
```

The exact next empirical/theorem target is to measure and bound the pair

```text
(kappa, Delta)
```

or its nonsmooth directional-cone replacement across scalable SAT families and declared
presentation gauges, while retaining the existing claim ceiling.
