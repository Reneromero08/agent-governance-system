# Polynomial Selector Dilation of the Clause Flow

## Status

```text
POLYNOMIAL_SELECTOR_DILATION_ESTABLISHED_REFERENCE_CANDIDATE
GLOBAL_POLYNOMIAL_CONVERGENCE_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

This document records a polynomial-ODE dilation of the terminal-agnostic clause flow.
It removes the native `min` operations without replacing them by assignment search,
answer-conditioned preparation, division, Heaviside switching, or hard projection.

## Public carrier

For a public 3-CNF relation with `n` variables and `m` clauses, the carrier contains:

```text
n voltage coordinates
m short-memory coordinates
m long-memory coordinates
3m clause-selector coordinates
6m pair-selector coordinates
```

The total state dimension is:

```text
n + 11m.
```

Every coordinate and coefficient is compiled from public clause incidence and literal
signs. The deterministic initial state is public and rational.

## Clause minimum dilation

For one clause, define the literal defects:

```text
d_i = 1 - q_i v_i.
```

The original clause flow uses:

```text
C = (1/2) min_i d_i.
```

Introduce a three-state selector `w_i` with total mass one and polynomial replicator
transport:

```text
D = sum_i w_i d_i
S = sum_i w_i
w_i_dot = kappa w_i (D - S d_i).
```

The total selector mass is conserved for every `S`, not only on the exact unit simplex:

```text
sum_i w_i_dot = 0.
```

On the public unit simplex, lower-cost defects gain mass and the selector moves toward
the local minimum. The clause signal is:

```text
C_w = (1/2) sum_i w_i d_i.
```

No division is used.

## Pairwise minimum dilation

For the gradient contribution to literal `i`, the original flow uses the minimum of the
other two defects. Each pair receives its own two-state replicator selector. Therefore
there are three independently conserved pair simplexes per clause.

The initial Euler instrumentation incorrectly normalized all six pair weights as one
simplex. That bug was removed. Every two-state pair is now normalized independently in
the reference chart, matching the native conservation law.

## Voltage and memory dynamics

The selector signals replace only the local minima. The public short and long memory
laws remain polynomial logistic equations. Voltage confinement uses the polynomial
factor:

```text
1 - v_i^2.
```

A clause-local release term prevents nonsatisfying Boolean corners from becoming
absorbing solely because the confinement factor vanishes:

```text
-rho v_i sum_(m incident on i) C_m.
```

At a satisfying Boolean corner every `C_m` vanishes, so the solution remains stationary.
At a violated Boolean corner at least one incident clause releases its variables back
into the interior.

## Polynomial normal form

The native vector field has:

```text
state dimension: n + 11m
public rational coefficients: yes
maximum polynomial degree: at most 6
native division: none
native min/max: none
native Heaviside: none
native clipping: none
```

The Euler and adaptive solvers are measurement charts. Clipping in the Euler chart is
not part of the native polynomial carrier.

## Reference evidence

The candidate passed:

1. selector-mass conservation for every clause and every pair;
2. exact independent normalization of the three pair selectors in the Euler chart;
3. all 256 conjunctions of the eight complete three-variable clause relations;
4. all 255 satisfiable formulas in that census with independently verified witnesses;
5. the sole UNSAT formula with no false witness;
6. distinct-terminal parity-cycle SAT and UNSAT controls;
7. distinct-terminal pigeonhole SAT and UNSAT controls;
8. distinct-terminal graph-coloring SAT and UNSAT controls;
9. adaptive smooth integration on the parity carrier with selector-mass drift near
   machine precision.

This evidence establishes a coherent candidate, not a global convergence theorem.

## Scaling observation

Using the same relaxed adaptive controls:

```text
rtol = 1e-5
atol = 1e-7
maximum_step = 0.1
maximum_time = 5
```

the exact distinct-terminal parity family produced:

| core parity variables | public variables | clauses | result | continuous time | function evaluations | maximum long memory |
|---:|---:|---:|---|---:|---:|---:|
| 8 | 24 | 32 | verified solution | 0.91155 | 536 | 16.46 |
| 16 | 48 | 64 | verified solution | 1.04221 | 548 | 18.54 |
| 24 | 72 | 96 | verified solution | 2.03552 | 12,680 | 724.37 |
| 32 | 96 | 128 | no solution by `t=5` | 5.0 | 198,140 | 79,841.87 |

The parity-32 run preserved selector masses to approximately `1e-13`, so the transition
is not caused by simplex drift. It exposes unresolved memory growth, stiffness, or a
long instanton/dwell transition.

No asymptotic conclusion follows from four sizes.

## Exact remaining theorem

To promote this carrier into a `P = NP` proof, one must establish a public polynomial
`p` such that every satisfiable formula reaches a verified solution from the declared
public seed within polynomial trajectory length:

```text
length_F <= p(|F|).
```

The proof must also establish polynomial state range, numerical precision, totalized
UNSAT handling by the same deadline, and polynomial-overhead standard-model simulation.

The existing instanton argument bounds the number of index-descending transitions by
state dimension, but does not provide an effective formula-uniform bound on transition
width and critical-point dwell time. The basin theorem is solution-dependent and does
not establish public-seed access.

## Claim boundary

```text
POLYNOMIAL_SELECTOR_DILATION_ESTABLISHED_REFERENCE_CANDIDATE
EXHAUSTIVE_THREE_VARIABLE_SELECTOR_CENSUS_PASS
STRUCTURED_RELATIONAL_CONTROLS_PASS
PARITY_32_SCALING_TRANSITION_EXPOSED
UNIFORM_POLYNOMIAL_TRAJECTORY_BOUND_NOT_ESTABLISHED
CET_NATIVE_OPERATOR_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
