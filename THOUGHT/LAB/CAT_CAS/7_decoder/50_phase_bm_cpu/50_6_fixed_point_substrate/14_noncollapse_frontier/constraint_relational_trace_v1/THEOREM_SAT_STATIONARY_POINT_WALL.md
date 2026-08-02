# Satisfiable Stationary-Point Wall

## Status

```text
SAT_NON_SOLUTION_STATIONARY_POINT_ESTABLISHED
ARBITRARY_INITIAL_STATE_GLOBAL_CONVERGENCE_FALSIFIED
PUBLIC_SEED_SPECIFIC_THEOREM_REQUIRED
P_EQUALS_NP_NOT_PROVEN
```

## Construction

Consider the satisfiable public relation

```text
(x OR y OR y)
AND
(~x OR ~y OR ~y).
```

Its satisfying assignments are exactly the two assignments with `x != y`.

Place the phase carrier at the unresolved symmetric point

```text
c_x = c_y = 0
s_x = s_y = 1.
```

Set short memory to its fixed boundary value one, long memory to its public cap,
and every selector simplex to equal weights.

At this state:

- both clause-product violations equal `1/2`;
- total phase energy equals `1`;
- the threshold assignment is not a witness;
- positive and negative exact clause gradients cancel variable by variable;
- the boundary-release term vanishes because every cosine is zero;
- rigidity is suppressed because short memory is one;
- memory derivatives vanish at their boundary values;
- every selector derivative vanishes because its local costs are equal.

Therefore the derivative of the complete carrier state is exactly zero even though the
formula is satisfiable and the current boundary assignment is not a witness.

## Consequence

The native vector field does **not** converge to a solution from every possible carrier
state. No proof may claim arbitrary-initial-state global convergence, and no strict
global Lyapunov function can have only satisfying sections as stationary states for the
current carrier.

The answer-blind public low-discrepancy seed is not on this exact symmetry manifold and
has a nonzero derivative. Thus this counterexample does not refute convergence from the
declared public seed.

It sharpens the missing theorem to:

```text
For every satisfiable public 3-CNF, the declared public seed reaches a robust terminal
phase section within one formula-uniform polynomial native trajectory length, under all
declared presentation gauges.
```

## Remaining obligations

This theorem does not establish:

- convergence from the public seed on every satisfiable formula;
- a formula-uniform polynomial deadline;
- polynomial trajectory, memory, precision, or restoration resources;
- a deterministic total UNSAT boundary;
- `P = NP`.
