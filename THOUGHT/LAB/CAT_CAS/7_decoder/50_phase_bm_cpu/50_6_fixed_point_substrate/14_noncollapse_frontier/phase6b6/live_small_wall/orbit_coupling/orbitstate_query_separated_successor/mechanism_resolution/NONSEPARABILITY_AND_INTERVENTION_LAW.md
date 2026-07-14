# Nonseparability And Intervention Law

Status: `SYNTHETIC_JOINT_GATE_DEFINED__PHYSICAL_BINDING_MISSING`

## Joint Observable

For a query `q`, define:

```text
Y_q(a,b)
Y_q(a,empty)
Y_q(empty,b)
Y_q(empty,empty)
```

and:

```text
J_q =
  Y_q(a,b)
  - Y_q(a,empty)
  - Y_q(empty,b)
  + Y_q(empty,empty)
```

## Required Interpretation

`J_q != 0` demonstrates observed nonadditivity under the measured observable. It does
not by itself demonstrate unresolved OrbitState, catalytic borrowing, physical
relational memory, or `SMALL_WALL_CROSSED`.

Claim ceiling for a future passing physical gate:

```text
QUERY_SEPARATED_RELATIONAL_CARRIER_CANDIDATE
```

## Required Nulls

The gate must produce:

```text
shared topology: predicted nonzero J_q
separated topology: null J_q
additive branch-local synthetic data: null J_q
label swap: null change in preparation and J_q
relation mutation: predicted change in J_q
carrier-off: null J_q
geometry-null: null J_q
```

## Relation Mutations

Every relation mutation must bind:

```text
same pages or address population
same bank population
same operation count
same source timing envelope
same route/core allocation
same order distribution
same branch-local occupancy
changed joint incidence or topology only
```

## Mechanism Blocker

The current Family 10h proposal does not define a physically realizable mutation that
changes only joint incidence while preserving all ordinary carrier variables. That is
the remaining `PHYSICAL-MECHANISM-01` and `NONSEPARABILITY-01` overlap.
