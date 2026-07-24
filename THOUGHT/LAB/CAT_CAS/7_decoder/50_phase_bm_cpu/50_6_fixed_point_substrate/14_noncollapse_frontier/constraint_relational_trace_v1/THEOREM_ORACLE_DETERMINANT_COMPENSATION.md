# Reversible Oracle Determinant Compensation

## Setup

Let `V_F` be a reversible public circuit that computes the truth value of a formula into
one designated output qubit, using any finite number of additional work qubits. Let
`P(theta)` apply phase `exp(i theta)` when that output qubit is one. The ordinary
compute-phase-uncompute oracle is:

```text
O_F(theta) = V_F^-1 P(theta) V_F.
```

## Full-Space Determinant

Conjugation preserves determinant:

```text
det O_F(theta) = det P(theta).
```

Across the complete Hilbert space, exactly half of all basis states have the designated
output qubit equal to one. If the full space has dimension `D`, then:

```text
det O_F(theta) = exp(i theta D/2).
```

Its winding is `D/2`, independent of `F`.

Therefore the determinant of a polynomial reversible implementation does not expose
satisfiability. The dirty ancillary sectors supply a formula-independent topological
background.

## Clean-Sector Determinant

Restrict the oracle to the subspace where every work and output qubit begins and ends in
the declared clean state. The induced assignment-only operation is:

```text
U_F(theta) = I + (exp(i theta) - 1) Q_F.
```

Its determinant winding is the rank of the exact solution projector:

```text
W_clean(F) = rank Q_F = #SAT(F).
```

The complementary sectors carry:

```text
W_dirty(F) = D/2 - #SAT(F),
```

so that:

```text
W_clean(F) + W_dirty(F) = D/2.
```

## Consequence

A full-circuit determinant sensor is a null for SAT. A valid topological rank latch must
measure the determinant line of the clean invariant subspace itself, not the determinant
of the complete reversible dilation.

This sharpens the unresolved native primitive:

```text
prepare or identify the clean relational sector
-> measure its determinant winding or nonzero rank
-> exclude compensating dirty-sector leakage
-> restore the complete carrier
```

The clean sector still has dimension `2^n`. Selecting it with clean ancillas is easy;
measuring its determinant line without filling or traversing the assignment basis is
not established.

## Current Result

```text
FULL_REVERSIBLE_ORACLE_DETERMINANT_FORMULA_INDEPENDENT
CLEAN_SUBSPACE_DETERMINANT_RETAINS_SAT_INDEX
POLYNOMIAL_RESTRICTED_DETERMINANT_SENSOR_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
