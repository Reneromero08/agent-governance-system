# Topological Rank Latch

## Exact Projector

Let `Q_F` be the orthogonal projector onto the open solution relation of a public
three-clause formula `F`:

```text
Q_F |x> = |x>  if x satisfies F
Q_F |x> = 0    otherwise.
```

`Q_F` has a compact factorized description as the product of commuting local clause
projectors. Its rank is the number of satisfying assignments.

## Phase Loop

Define the phase-oracle loop:

```text
U_F(theta) = I + (exp(i theta) - 1) Q_F
```

for `theta` from zero through one complete turn.

On the satisfying subspace, `U_F(theta)` has eigenvalue `exp(i theta)`. On its
orthogonal complement, the eigenvalue is one. Therefore:

```text
det U_F(theta) = exp(i theta * rank(Q_F)).
```

The determinant-line winding is:

```text
W_F = (1 / 2*pi*i) integral d log det U_F(theta)
    = rank(Q_F)
    = number of satisfying assignments.
```

Consequently:

```text
F is satisfiable if and only if W_F is nonzero.
```

The presence index `1[W_F != 0]` is idempotent under duplicate derivations and is a
complete topological SAT invariant.

## Native Inverse

Because `Q_F` is a projector:

```text
U_F(theta)^-1 = U_F(-theta).
```

The public phase program therefore has an exact program-derived inverse.

## Why This Does Not Complete The Proof

The operator has a polynomial symbolic description, but the current reference carrier
materializes one phase coordinate for every boundary assignment. The determinant line
of a `2^n`-dimensional operator can be enacted by a completely filled fermionic mode
space only by using `2^n` modes or equivalent unresolved resources.

A normalized trace readout also fails the unique-witness gate:

```text
Tr(U_F(pi)) / 2^n = 1 - 2 * rank(Q_F) / 2^n.
```

With one satisfying assignment, the displacement from the UNSAT value is `2 / 2^n`.
Thus a normalized trace sensor carries an exponential precision or sample requirement.

The remaining target is not the topological invariant. The invariant is complete. The
remaining target is a polynomial-resource native determinant-line or nonzero-rank
sensor that does not materialize the Hilbert basis, fill exponentially many modes, or
resolve an exponentially small normalized signal.

## Current Result

```text
TOPOLOGICAL_DETERMINANT_WINDING_COMPLETE_FOR_REFERENCE_RELATION
POLYNOMIAL_NATIVE_DETERMINANT_LINE_SENSOR_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
