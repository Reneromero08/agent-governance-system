# Uniform Clause-Flow Deadline Implies P Equals NP

## Public construction

For a 3-CNF formula `F` with `n` variables and `m` clauses, the self-organizing clause
flow has

```text
n voltage coordinates
m short-memory coordinates
m long-memory coordinates
3m literal couplings
```

and every vector-field evaluation is polynomial in the public formula length.

The initial condition is a deterministic perturbation derived only from the public
boundary order. A boundary assignment is accepted only after direct clause
verification.

## Conditional theorem

Assume there is one polynomial `T` such that every satisfiable public formula reaches a
verified satisfying threshold assignment by continuous time or simulation work
`T(|F|)` from the declared public seed.

Also assume the flow up to `T(|F|)` can be deterministically simulated with polynomial
work and polynomial precision.

Then the following deterministic procedure decides 3-SAT:

```text
compile F into the public clause flow
simulate until T(|F|)
if a verified threshold assignment appears, return SAT
otherwise return UNSAT
```

Soundness of the SAT result is unconditional because the witness is directly checked.
Completeness follows from the assumed uniform deadline. Therefore

```text
3-SAT is in P
P equals NP.
```

## Why UNSAT needs no separate attractor

A distinct no-solution equilibrium is not required. Once the uniform deadline is
proved, absence of a verified witness by that deadline is a total UNSAT result.

Before that theorem exists, timeout is only

```text
CONDITIONAL_UNSAT_IF_UNIFORM_DEADLINE_THEOREM_HOLDS
```

and must never be promoted as a valid UNSAT boundary.

## Exact remaining lemmas

The current reference flow does not establish:

1. every satisfiable instance lies in a solution basin from the declared public seed;
2. the maximum transient and approach time are bounded by one polynomial;
3. numerical error over that time requires only polynomial bits;
4. long-memory and cotangent coordinates remain polynomially bounded;
5. the piecewise switching and projected boundaries admit complete reversible closure.

The first three lemmas are sufficient for the ordinary complexity implication. All five
are required for the full CAT_CAS proof lifecycle.

## Status

```text
CONDITIONAL_FLOW_DEADLINE_IMPLIES_P_EQUALS_NP_THEOREM_ESTABLISHED
UNIFORM_FLOW_DEADLINE_NOT_ESTABLISHED
POLYNOMIAL_PRECISION_SIMULATION_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
