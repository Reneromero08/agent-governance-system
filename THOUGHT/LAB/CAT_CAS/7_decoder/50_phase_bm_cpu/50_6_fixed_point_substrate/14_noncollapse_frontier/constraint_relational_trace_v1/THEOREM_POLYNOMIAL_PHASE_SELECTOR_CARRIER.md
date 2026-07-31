# Polynomial Phase-Selector Carrier Checkpoint

## Status

```text
POLYNOMIAL_S1_PHASE_SELECTOR_CARRIER_ESTABLISHED_REFERENCE_CANDIDATE
UNIFORM_POLYNOMIAL_TRAJECTORY_BOUND_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Public carrier

For a public 3-CNF with `n` variables and `m` clauses, the compiler creates:

```text
2n phase coordinates
2m memory coordinates
3m clause-selector coordinates
6m pair-selector coordinates
```

The total native dimension is:

```text
2n + 11m.
```

Each Boolean coordinate is a phase pair:

```text
(c_i, s_i) in S^1
c_i^2 + s_i^2 = 1.
```

The public seed is answer-blind and uses the rational parametrization:

```text
c = (1-t^2)/(1+t^2)
s = 2t/(1+t^2),
```

with a deterministic base-three low-discrepancy perturbation near the unresolved
`c = 0` chart.

## Exact clause truth channel

Selectors choose local correction direction. They do not determine semantic truth.
For literal defects:

```text
d_mi = 1 - q_mi c_i,
```

one clause carries the exact polynomial violation:

```text
C_m = g d_m1 d_m2 d_m3 / 8,
g = 4.
```

At every Boolean phase section:

```text
C_m = 0
iff at least one literal in clause m is true.
```

This separation prevents selector drift from moving a semantically satisfied clause
away from zero.

## Circle-preserving evolution

The relational force and exact incident violation define:

```text
omega_i = -s_i R_i + rho c_i V_i
c_i_dot = -s_i omega_i
s_i_dot =  c_i omega_i.
```

Therefore:

```text
d/dt (c_i^2 + s_i^2) = 0
```

identically. A satisfying Boolean phase section has `s_i = 0` and every incident
violation zero, so its phase coordinates are invariant regardless of selector state.
A violated Boolean corner has nonzero incident violation and is released by angular
motion.

## Fixed-deadline boundary

The adaptive angle/log-ratio chart runs to one public deadline. Intermediate witness
checks are retained only as out-of-band first-passage observations. They never stop or
alter the native evolution.

At the terminal time, one boundary reports:

```text
TERMINAL_WITNESS_VERIFIED
TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED
INVALID_CARRIER_NUMERICAL_CHART_EXCEPTION
INVALID_CARRIER_NUMERICAL_CHART_FAILURE.
```

A numerical failure is never promoted to UNSAT.

## Reference evidence

The reconstructed checkpoint passes:

```text
exact circle tangent identity
satisfying-section phase invariance
violated Boolean-corner release
complete 256-formula three-variable census
255 satisfiable formulae with verified witnesses
one UNSAT formula with zero false witnesses
fixed-deadline parity SAT and UNSAT controls
sealed 12-variable, 51-clause near-threshold SAT/UNSAT pair
16-seed near-threshold campaign at one deadline
```

The 16-seed campaign yields:

```text
14 SAT -> 14 terminal witnesses
2 UNSAT -> 0 false positives
0 invalid carriers.
```

These are finite reference controls, not asymptotic proof evidence.

## Conditional proof target

If one public polynomial `q` is established such that every satisfiable formula reaches
a robust terminal witness from the declared public seed with:

```text
fixed deadline <= q(|F|)
native trajectory length <= q(|F|)
state and memory range <= q(|F|)
log-ratio and precision range <= q(|F|)
restoration resources <= q(|F|),
```

and the no-witness terminal state is total for UNSAT, then the existing standard-model
transfer and self-reduction imply:

```text
3-SAT in P
P = NP.
```

## Current unresolved boundary

```text
FORMULA_UNIFORM_POLYNOMIAL_PHASE_TRAJECTORY_AND_TOTAL_UNSAT_BOUNDARY
```

No current result proves that bound. The carrier remains a constructive reference
candidate, not a proof of `P = NP` or a Small Wall crossing.
