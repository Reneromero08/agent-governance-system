# Hybrid Witness-Event Precision Wall

## Status

```text
HYBRID_WITNESS_EVENT_PRECISION_WALL_ESTABLISHED
SEMANTIC_GUARD_HAS_ZERO_WITNESS_MARGIN_INFIMUM
FORMULA_ONLY_POSITIVE_MARGIN_BOUND_REJECTED
DYNAMIC_MARGIN_AND_DWELL_THEOREM_REQUIRED
P_EQUALS_NP_NOT_PROVEN
```

## Fixed-formula witness family

Use the exact odd three-bit parity formula. For every

```text
0 < epsilon < 1,
```

the strict phase state

```text
(c_1,c_2,c_3) = (epsilon,-epsilon,-epsilon)
```

has threshold assignment

```text
TFF,
```

which is a verified witness.

For the exact hybrid guard

```text
G_F(c) = min_j max_{r in clause j} q_r c_i,
```

every clause margin at this state is at least `epsilon`, and one clause margin is
exactly `epsilon`. Therefore

```text
G_F(epsilon,-epsilon,-epsilon) = epsilon.
```

As `epsilon` tends to zero through positive values, every state remains a strict
verified witness while the guard margin tends to zero.

Hence, even for one fixed three-variable formula,

```text
inf { G_F(c) | c is a strict threshold witness } = 0.
```

There is no positive lower bound determined by formula semantics alone.

## Consequence

The semantic exactness of the hybrid guard does not imply:

```text
inverse-polynomial event margin
inverse-polynomial witness dwell time
transverse threshold crossing
polynomial-precision event isolation
robustness under numerical simulation.
```

A recorder that fires on `G_F > 0` must know how deeply and how long the declared
public trajectory enters a satisfying orthant. Those facts are properties of the
specific native dynamics, not of SAT semantics or the guard definition.

The exact remaining statement for this lane is therefore dynamic:

```text
For every satisfiable public formula, the declared public-seed trajectory reaches a
time t <= poly(|F|) with

    G_F(c(t)) >= 1/poly(|F|)

and remains in G_F > 0 for at least 1/poly(|F|) time, with polynomially bounded
transversality and state range.
```

Only such a theorem could support polynomial event detection and a stable copy into the
hybrid recorder.

## Claim boundary

This wall does not prove that the current public trajectory has exponentially small
witness margins. The odd-parity trajectory may have a substantial observed transient
margin.

It proves that no resource bound follows merely from:

```text
the formula being satisfiable
the guard being semantically exact
the recorder using O(n) lanes.
```

The bound must be proved for the declared dynamics across all public formulas and
presentation gauges.

Still unresolved:

```text
public-seed witness crossing for every SAT formula
formula-uniform polynomial crossing time
inverse-polynomial guard margin
inverse-polynomial dwell time
hybrid event simulation and false-event exclusion
reversible event-history restoration
P = NP.
```
