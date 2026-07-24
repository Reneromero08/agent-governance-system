# Instanton Deadline Proof Skeleton and Remaining Gap

## Published continuous-time chain

For the public memory-assisted 3-SAT flow, the supplementary topological-field-theory
argument uses this chain:

```text
critical points have finite Morse index
instanton transitions decrease that index
maximum index <= phase dimension = n + 2m
fixed clause density gives O(n) possible index descent steps
each instanton width plus critical dwell is bounded by one size-independent T_max
therefore threshold-crossing physical time <= n(1 + 2m/n) T_max
```

The published proposition is for solvable fixed-density 3-SAT instances. It measures
continuous physical time to a fixed threshold, not mathematical arrival at the exact
fixed point.

## What this gives if fully effective

If `T_max` is a computable uniform constant for the public flow rates, the declared
public seed lies in the solution basin for every satisfiable instance, and the
continuous flow can be simulated with polynomial work and precision, then the uniform
deadline theorem in `THEOREM_FLOW_DEADLINE_IMPLIES_P_EQUALS_NP.md` applies.

No separate UNSAT attractor is required. Run until the uniform SAT deadline. A verified
witness gives SAT. No witness by that deadline gives UNSAT.

## Exact uncovered obligations

The current CAT_CAS package still needs:

1. an effective uniform upper bound on every instanton width and critical dwell from
   public rates and formula size;
2. proof that the declared deterministic public perturbation lies in a solution basin
   for every satisfiable formula;
3. control of switching surfaces and projected boundaries in the discontinuous flow;
4. polynomial bit precision and operation count for a standard-model simulation;
5. polynomial bounds on long-memory, cotangent, and environment dynamic range;
6. native reversal of the complete carrier and environment.

The publication itself states that the continuous-time scaling result need not transfer
to numerical integration because discretization breaks the topological supersymmetry.
That explicit boundary prevents the physical-time proposition from being read directly
as `3SAT in P`.

## Status

```text
PUBLISHED_CONTINUOUS_TIME_INSTANTON_SCALING_RECONSTRUCTED
INDEX_DESCENT_STEP_BOUND_LINEAR_AT_FIXED_DENSITY
UNIFORM_EFFECTIVE_TMAX_NOT_ESTABLISHED_IN_PACKAGE
POLYNOMIAL_NUMERICAL_TRANSFER_NOT_ESTABLISHED
FULL_CAT_CAS_RESTORATION_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
