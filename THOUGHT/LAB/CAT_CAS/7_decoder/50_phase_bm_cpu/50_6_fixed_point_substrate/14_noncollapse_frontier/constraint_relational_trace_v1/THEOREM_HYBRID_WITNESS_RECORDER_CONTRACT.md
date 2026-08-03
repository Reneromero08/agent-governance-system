# Hybrid Witness-Recorder Semantic Contract

## Status

```text
HYBRID_THRESHOLD_WITNESS_RECORDER_SEMANTIC_CONTRACT_ESTABLISHED
OFF_BOUNDARY_FALSE_POSITIVE_AND_FALSE_NEGATIVE_COUNTS_ZERO
RECORDER_LANE_COUNT_LINEAR
DYNAMIC_RESOURCE_CLOSURE_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Public guard

For a phase state with cosines `c_i`, define each signed literal value by

```text
q_r c_i,
```

where `q_r` is `+1` for a positive literal and `-1` for a negated literal.
For each clause define

```text
M_j(c) = max_{r in clause j} q_r c_i,
```

and define the formula guard

```text
G_F(c) = min_j M_j(c).
```

Away from all variable threshold surfaces `c_i = 0`, the sign assignment is
unambiguous and

```text
G_F(c) > 0
iff every clause has one sign-true literal
iff the public threshold assignment satisfies F.
```

Thus an explicit hybrid event guarded by `G_F(c) > 0` is semantically exact off
threshold boundaries.

Any state with one or more `c_i = 0` fails closed. The event is disabled until the
complete threshold assignment is unambiguous, even if the remaining variables already
make every clause true.

## Exact finite control

For the odd three-bit parity counterexample, all eight strict sign assignments were
audited at equal interior magnitude.

```text
four witness assignments
four non-witness assignments
zero false-positive events
zero false-negative events.
```

At the interior witness

```text
(c_1,c_2,c_3) = (epsilon,-epsilon,-epsilon),
```

the hybrid guard margin is exactly

```text
G_F = epsilon > 0.
```

At the same state, every current exact-product clause channel remains strictly positive.
The hybrid guard therefore detects the transient sign witness that the smooth
exact-product zero channel cannot see.

## Recorder state

The smallest direct recorder needs:

```text
one latched-valid bit
n stored assignment bits.
```

Its discrete lane count is therefore

```text
n + 1,
```

which is linear in the public variable count.

This coordinate count does not establish a lawful implementation. The event must copy
an unambiguous sign assignment into the recorder and preserve it after the phase flow
leaves the witness orthant.

## Mechanism-class boundary

The operations

```text
min
max
strict sign comparison
one-shot latching
```

are not part of the current smooth polynomial carrier. This candidate is explicitly a
hybrid or discontinuous boundary mechanism.

It must not be narrated as a hidden polynomial ODE or a free topological invariant.
The threshold-witness latch wall rejects exact open-orthant-supported analytic
activation, and the topological witness-record wall rejects witness visitation as a
homotopy or winding invariant.

## Remaining resource obligations

The semantic contract does not establish:

```text
an inverse-polynomial witness guard margin
a polynomial witness dwell time
transverse and isolated event crossings
polynomial-precision event detection
deterministic polynomial simulation of the hybrid system
false-event exclusion under finite precision
storage of the witness without answer-conditioned preparation
reversible event-history restoration
complete environment restoration
a total formula-uniform polynomial deadline
P = NP.
```

The exact next theorem needed for this lane is a dynamic margin statement:

```text
Every satisfiable public formula has a public-seed witness crossing by polynomial time
at which G_F is at least inverse polynomial and remains positive for inverse-polynomial
time, with a polynomially bounded transverse crossing.
```

Without that theorem, the hybrid guard is an exact semantic interface only, not a
polynomial decision procedure.
