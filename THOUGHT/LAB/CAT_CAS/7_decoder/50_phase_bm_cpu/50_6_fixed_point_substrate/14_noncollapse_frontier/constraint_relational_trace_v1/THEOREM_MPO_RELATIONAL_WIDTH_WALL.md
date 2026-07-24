# MPO and Residual-Relation Width Wall

## Historical MPO Projection Audit

The historical CAT_CAS MPO winding experiment uses bond index:

```text
alpha = (control state, current symbol)
```

with reported bond dimension:

```text
chi = number_of_states * alphabet_size.
```

A bounded Turing-machine configuration also requires:

```text
control state
head position
the complete tape word
```

For `S` control states, alphabet size `A`, and `L` tape cells, the bounded configuration
count is:

```text
S * L * A^L.
```

The `(state, current symbol)` projection has only `S*A` coordinates. It is not injective
on machine configurations and cannot preserve arbitrary computation histories. Its
winding is an invariant of a finite projected transition graph.

Therefore the historical MPO result cannot supply the clean determinant-line sensor for
arbitrary SAT.

## Exact Residual-Relation Width

Fix a variable order. After a prefix assignment, the unresolved object is the residual
Boolean relation on the unassigned boundary. Two prefixes may share one carrier state
only when they induce exactly the same residual relation.

The exact width at a cut is the number of inequivalent residual open relations created
by all prefix assignments.

This is a representation-independent semantic quantity for deterministic one-pass
carriers at that cut. An exact MPO, OBDD, or transfer-state representation following the
same order requires at least that many distinguishable bond states.

## Equality Calibration

Consider:

```text
EQ_k = AND_i (x_i iff y_i).
```

Each equivalence is expressed through two padded local three-literal relations.

Under grouped order:

```text
x_1, ..., x_k, y_1, ..., y_k
```

the cut after all `x` variables has `2^k` different residual relations. Each prefix
specifies one different required `y` word.

Under interleaved order:

```text
x_1, y_1, ..., x_k, y_k
```

the maximum exact width is at most three: false sink plus the two pending values of the
current pair.

## Consequence

A small bond dimension can be a property of presentation and variable order rather than
a property of the relation itself. Any proposed compact determinant carrier must pass:

```text
clause order gauge
variable renaming gauge
variable-order audit
held-out relation families
exact residual-relation comparison
```

The equality calibration is not an all-orders lower bound for arbitrary SAT. It is a
killer control against silently choosing a favorable representation and calling the
result a universal relational compression theorem.

## Current Result

```text
MPO_CONTROL_SYMBOL_PROJECTION_NOT_AN_EXACT_CONFIGURATION_CARRIER
EXACT_RESIDUAL_OPEN_RELATION_WIDTH_MEASURED
UNIVERSAL_POLYNOMIAL_BOND_DIMENSION_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
