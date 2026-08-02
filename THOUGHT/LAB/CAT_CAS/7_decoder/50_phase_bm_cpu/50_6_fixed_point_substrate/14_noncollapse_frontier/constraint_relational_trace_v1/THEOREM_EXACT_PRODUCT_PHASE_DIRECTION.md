# Exact Product Phase Direction Checkpoint

## Status

```text
EXACT_PRODUCT_PHASE_DIRECTION_ESTABLISHED_REFERENCE_CANDIDATE
SELECTOR_MIN_DIRECTION_FALSIFIED_ON_SEED_38
UNIFORM_POLYNOMIAL_TRAJECTORY_BOUND_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Exact direction law

For one public clause with literal defects

```text
d_i = 1 - q_i c_i,
```

and exact polynomial truth channel

```text
C = 4 d_1 d_2 d_3 / 8 = d_1 d_2 d_3 / 2,
```

the exact negative coordinate gradient is

```text
- partial C / partial c_i = q_i d_j d_k / 2.
```

The phase carrier now uses this exact public direction by default:

```text
G_i = q_i d_j d_k / 2.
```

The selector-min direction remains available only as a calibration. It approximates a
local minimum through replicator weights and is not the default mechanism.

## Seed-38 separating control

The deterministic 12-variable, 51-clause seed-38 formula is SAT with one reference
witness.

At the same public seed and deadline `T = 3`:

```text
selector-min direction:
  no terminal witness
  negative clause margin
  long memory above 500
  sustained high-gain switching

exact-product direction:
  first passage approximately 1.365
  terminal witness verified
  terminal clause margin 1
  maximum long memory approximately 43.2
  native trajectory length approximately 213
```

This is a mechanism-separating control, not a parameter-only speedup. The exact gradient
removes the one-clause switching regime produced by selector-min direction.

## Reference evidence

The exact-product direction passed:

```text
complete 256-formula three-variable census
255 SAT witnesses and one UNSAT no-false-witness control
parity SAT and UNSAT
pigeonhole SAT and UNSAT
graph-coloring SAT and UNSAT
128 deterministic near-threshold 12-variable formulae
101 SAT -> 101 terminal witnesses
27 UNSAT -> zero false positives
zero invalid carriers
```

Across the 128-case near-threshold campaign at deadline three:

```text
latest observed SAT first passage approximately 2.997
maximum long memory approximately 539
maximum native trajectory length approximately 3409
```

The near-deadline unique-witness seed 86 remained stable under:

```text
reverse variable renaming
cyclic variable renaming
reverse clause order
reverse literal order.
```

Additional capped-reference controls passed:

```text
14 variables, 60 clauses: 9 SAT and 7 UNSAT
16 variables, 68 clauses: 15 SAT and 1 UNSAT
18 variables, 77 clauses: 3 SAT and 1 UNSAT
20 variables, 85 clauses: 3 SAT and 1 UNSAT
```

Planted-but-answer-blind SAT instrumentation also produced independently verified
terminal witnesses at:

```text
32 variables, 136 clauses: first passage approximately 3.456
64 variables, 272 clauses: first passage approximately 1.436
128 variables, 544 clauses: first passage approximately 1.318.
```

The planted witness was retained only by the external generator for certification. The
native phase flow received only the public clauses and produced its own independently
verified terminal assignment.

## What this establishes

The checkpoint establishes that the previously observed seed-38 obstruction belonged
to the selector-min direction, not to the exact clause truth channel or the `S^1`
carrier itself.

The exact product direction is local, public, polynomial, presentation-robust on the
current controls, and semantically aligned with the clause violation polynomial.

## What remains unproved

Finite campaigns do not establish:

```text
a formula-uniform polynomial deadline
polynomial native trajectory length on every SAT formula
polynomial memory and precision bounds on every formula
total deterministic UNSAT certification
complete reversible environment restoration
standard-model polynomial simulation for the full boundary
P = NP.
```

The exact remaining theorem is:

```text
Every satisfiable public 3-CNF reaches a robust terminal phase section from the
answer-blind public seed within one formula-uniform polynomial native trajectory length.
```
