# Satisfiable Stationary-Point Wall

## Status

```text
SAT_NON_SOLUTION_STATIONARY_POINT_ESTABLISHED
ARBITRARY_INITIAL_STATE_GLOBAL_CONVERGENCE_FALSIFIED
STATIONARY_OBSTRUCTION_PHASE_SADDLE_ESTABLISHED
PUBLIC_SEED_ALREADY_WITNESS_ON_OBSTRUCTION_FORMULA
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
z_x = z_y = 1.
```

Set short memory to its fixed boundary value one, long memory to its public cap `L`,
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

## Exact local phase stability

Let `rho` be the public boundary-release rate. In the cosine tangent chart at the
positive-sine stationary point, the exact two-variable phase block is

```text
A = [[-rho,       -2L],
     [ -2L, -2L - 2rho]].
```

Its trace and determinant are

```text
tr(A)  = -2L - 3rho

det(A) = 2rho L + 2rho^2 - 4L^2
       = -2(2L+rho)(L-rho).
```

The declared carrier has `L > rho`, so

```text
det(A) < 0.
```

Hence the phase block has one strictly positive and one strictly negative real
eigenvalue. The full linearization retains those eigenvalues: at the memory boundaries,
the memory equations have no first-order phase dependence, and when short memory equals
one the exact-product phase force is selector-independent. After reordering coordinates,
the linearization is block triangular with the phase block on its diagonal.

The stationary obstruction is therefore a phase saddle. It has no open attracting basin.
The equal-cost selector simplex still contributes center directions, so this statement
does not by itself characterize the complete global stable manifold.

## Exact public-seed separation

For two variables, the declared low-discrepancy seed has threshold signs

```text
first sorted variable  -> TRUE
second sorted variable -> FALSE.
```

Therefore its threshold assignment already satisfies the relation because the two
variables have opposite values.

The same result holds for both possible variable-renaming order gauges:

```text
x -> a, y -> b
x -> b, y -> a.
```

Clause order, literal order, and semantic duplicate clauses do not change this terminal
fact. For this obstruction formula, the public seed is already on a satisfying terminal
section at time zero.

Thus the stationary point remains a valid counterexample to arbitrary-state convergence,
but it is not a public-seed counterexample for its own formula.

## Consequence

The native vector field does **not** converge to a solution from every possible carrier
state. No proof may claim arbitrary-initial-state global convergence, and no strict
global Lyapunov function can have only satisfying sections as stationary states for the
current carrier.

At the same time, this particular obstruction cannot carry the remaining public-seed
burden:

```text
its public seed already verifies a witness;
its stationary point is a phase saddle rather than an open attractor.
```

The missing theorem remains:

```text
For every satisfiable public 3-CNF, the declared public seed reaches a robust terminal
phase section within one formula-uniform polynomial deadline, under every declared
presentation gauge.
```

## Remaining obligations

This theorem does not establish:

- convergence from the public seed on every satisfiable formula;
- exclusion of the global stable manifold of every non-solution saddle or center set;
- a formula-uniform polynomial deadline;
- deterministic polynomial simulation and robust terminal margin;
- cotangent or environmental restoration for the stronger CAT_CAS lifecycle;
- `P = NP`.
