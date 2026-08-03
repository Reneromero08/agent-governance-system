# Odd-Parity Fixed-Deadline Counterexample

## Status

```text
ODD_PARITY_NON_SOLUTION_MIDPOINT_MANIFOLD_NORMALLY_ATTRACTING
FIXED_DEADLINE_THREE_PUBLIC_SEED_COMPLETENESS_FALSIFIED
TRANSIENT_WITNESS_DOES_NOT_ESTABLISH_TERMINAL_COMPLETENESS
ALL_FORMULA_UNIFORM_POLYNOMIAL_DEADLINES_UNRESOLVED
PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Public formula

Consider the public three-variable, four-clause relation

```text
( x1 OR ~x2 OR ~x3)
AND
(~x1 OR  x2 OR ~x3)
AND
(~x1 OR ~x2 OR  x3)
AND
( x1 OR  x2 OR  x3).
```

This is the exact CNF for odd three-bit parity. Its satisfying assignments are

```text
(False, False, True)
(False, True, False)
(True, False, False)
(True, True, True).
```

The tiny reference boundary therefore certifies:

```text
SAT
witness count = 4.
```

The executable formula and custody surface are in:

```text
odd_parity_fixed_deadline_counterexample.py
```

The audit records the semantic digest, presentation digest, exact clause tokens, reference
witness count, public-seed assignment, stationary assignment, solver controls, terminal
assignments, first-passage observations, and terminal margins.

## Public seed is not terminal

For three sorted variables, the declared low-discrepancy seed thresholds to

```text
x1 = True
x2 = False
x3 = True.
```

This assignment has even parity and does not satisfy the formula.

Thus, unlike the earlier two-variable symmetric obstruction, this formula is relevant to
the public-seed terminal theorem.

## Exact non-solution stationary manifold

Place the phase carrier at

```text
c_1 = c_2 = c_3 = 0
z_1 = z_2 = z_3 = 1,
```

set every short memory to one, every long memory to the public cap `L`, and every clause
selector to any point of its three-state simplex.

At this state:

```text
each literal defect = 1
each exact clause-product violation C_j = 1/2
threshold assignment = (False, False, False)
threshold assignment is not a witness.
```

Across the four odd-parity clauses, every signed exact-product phase force cancels
variable by variable. The release term vanishes because all cosines are zero. Memory
coordinates are fixed at their boundary values. Every selector derivative vanishes
because all three costs in every clause equal one.

Therefore the complete reduced carrier derivative is exactly zero.

## Exact transverse stability

Let `rho` be the boundary-release rate. At the midpoint, the exact phase tangent block is

```text
A_phase = -2 rho I_3.
```

For the frozen public value `rho = 10`, the three phase eigenvalues are

```text
-20, -20, -20.
```

Each clause has `C=1/2`, so the memory-boundary normal exponents are

```text
short from one = -beta(1/2-gamma) = -5
long from cap  = -alpha(1/2-delta) = -9/4.
```

All phase and memory normal directions are strictly attracting. Equal-cost selectors
contribute two center directions per clause, for eight selector-center directions in
total. The stationary object is therefore a normally attracting non-solution manifold
transverse to selector centers, not merely a numerical near-equilibrium.

## Public trajectory at the frozen deadline

The branch previously used the fixed terminal deadline

```text
T = 3.
```

The public trajectory was integrated independently with both:

```text
DOP853
Radau
```

using tighter-than-campaign tolerances and maximum step `0.02`.

Both solver families agree on the following sequence:

```text
the public seed is initially not a witness
a satisfying threshold assignment is observed transiently before t=0.3
the trajectory leaves that transient sign section
the phase cosines contract back toward the non-solution midpoint manifold
at T=3, terminal public verification fails.
```

For every recorded solver control:

```text
fixed deadline reached
first passage observed
terminal status = TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED
terminal phase-cosine norm < 1e-8
terminal clause-satisfaction margin < 0.
```

The terminal threshold assignment returns to

```text
(True, False, True),
```

which is not a witness.

## Falsified claim

This formula falsifies the current deadline-three terminal-completeness statement:

```text
Every satisfiable public formula produces a verified satisfying terminal assignment at
T=3 from the declared public seed.
```

It also separates two notions that had been conflated in earlier finite evidence:

```text
first-passage witness observed
!=
terminal witness verified at the declared decision deadline.
```

A transient sign crossing cannot be used as a terminal decision unless the native law
provides a lawful persistent latch or the decision theorem explicitly permits and
implements first-passage extraction without answer-conditioned stopping.

## What is not falsified

This checkpoint does **not** prove that every possible formula-uniform polynomial deadline
fails. One formula and one frozen deadline cannot rule out a different public deadline
law `q(|F|)`.

It does not yet establish:

```text
that the odd-parity trajectory remains in the midpoint basin for all sufficiently large time
that every later terminal time fails
that no lawful persistent witness latch can be constructed
that the current law fails under every presentation gauge
that every polynomial deadline is impossible
P != NP
P = NP.
```

The exact surviving alternatives are:

1. derive and prove a different formula-uniform polynomial terminal deadline;
2. add a public, terminal-agnostic, polynomial, reversible-compatible witness-persistence
   mechanism and requalify the complete campaign;
3. reformulate the decision surface around a lawful first-passage certificate while
   proving polynomial detection, precision, totality, and restoration obligations.

Any such change is a new mechanism checkpoint. The prior deadline-three terminal claim
must remain marked falsified.
