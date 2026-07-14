# Carrier Capacity And Query Space

Status: `CAPACITY_SEPARATION_NOT_ESTABLISHED`

## Required Inequality

To separate a compact relation state from a precomputed answer table, a future
protocol must freeze:

```text
C_prep = carrier preparation capacity after source closure
B_answer_raw = |Q| * response_bits
B_answer_min = minimum admissible ordinary answer-equivalent code length, including
               compression, formulas, circuits, seeds, low-rank bases, and public
               side information
B_relation = relation_representation_bits
H_query = log2(|Q|)
```

Capacity separation requires:

```text
B_answer_min > C_prep
B_relation <= C_prep
H_query high enough that source-visible guessing is negligible
```

## Current State

The proposed shared-pair Family 10h carrier has not measured:

```text
number of independent persistent carrier coordinates
bits per coordinate
state lifetime after source death
measurement disturbance
route/bank/address/order capacity
whether PMU/timing observables are state variables or readout artifacts
```

Therefore this package does not assign a numeric `C_prep`. Inventing one would hide
the blocker instead of resolving it.

## Answer-Table Requirement

For `m = |Q|` and `r` response bits:

```text
B_answer_raw = m * r
```

This is not a lower bound. If `Q` is small, public, and enumerable before source
closure, the raw table may fit ordinary metadata capacity. Even when the raw table is
large, a compressed answer generator, response formula, low-rank basis, coefficient
set, seed, circuit, or compact relation encoding may be far smaller. In either case
query separation is not enough.

## Relation-Representation Requirement

A compact state is allowed only when it is specified as a physical representation:

```text
P(R) = carrier state encoding the unordered relation, not the response vector
```

The representation must be invariant under blinded branch-to-lane permutation and
must support held-out query evaluation without storing every answer.

## Held-Out Strategy

If capacity cannot be independently measured, the next lawful strategy is held-out
generalization:

```text
training relations
held-out relations
training query bases
held-out query bases
training query parameters
held-out query parameters
training seeds
held-out seeds
training sessions
held-out sessions
```

A lookup table that memorizes the training cross-product must fail on held-out cells.
This still rejects only bounded old-boundary predictors, not an arbitrary unbounded
lookup table.

## Result

Capacity separation is not established for the proposed Family 10h mechanism.
Held-out generalization remains a required future strategy, not a passed gate.
