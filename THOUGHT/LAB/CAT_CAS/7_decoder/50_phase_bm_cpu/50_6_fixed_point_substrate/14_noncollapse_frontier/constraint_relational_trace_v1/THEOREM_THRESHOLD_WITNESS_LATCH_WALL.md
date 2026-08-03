# Threshold-Witness Latch Wall

## Status

```text
INTERIOR_THRESHOLD_WITNESS_NOT_DETECTED_BY_EXACT_PRODUCT_ZERO_CHANNEL
EXACT_ORTHANT_SUPPORTED_POLYNOMIAL_WRITE_ENABLE_IMPOSSIBLE
EXACT_ORTHANT_SUPPORTED_REAL_ANALYTIC_WRITE_ENABLE_IMPOSSIBLE
NAIVE_SMOOTH_THRESHOLD_LATCH_REJECTED
TOPOLOGICAL_HYBRID_OR_BOOLEANIZATION_ROUTES_REMAIN_OPEN
P_EQUALS_NP_NOT_PROVEN
```

## Semantic mismatch exposed by odd parity

The odd-parity fixed-deadline counterexample briefly crosses a satisfying threshold
assignment while remaining close to the unresolved phase midpoint.

Threshold semantics declares a literal true when

```text
q_i c_i > 0.
```

The current exact-product clause channel is

```text
C_j = truth_gain * product_i (1-q_i c_i) / 8.
```

It reaches zero only when at least one literal reaches its exact phase boundary

```text
q_i c_i = 1.
```

These are different conditions away from Boolean phase boundaries.

For the strict interior odd-parity witness

```text
(c_1,c_2,c_3) = (epsilon,-epsilon,-epsilon)
0 < epsilon < 1,
```

the threshold assignment is

```text
(True, False, False),
```

which satisfies all four odd-parity clauses. Nevertheless every exact-product clause
channel remains strictly positive.

At the executable control `epsilon=1/10` and `truth_gain=4`, the four violations are

```text
0.3645
0.5445
0.5445
0.5445.
```

Thus the current native zero channel does not observe the transient sign witness that the
external threshold verifier observes.

## Polynomial support obstruction

Suppose an autonomous latch uses a polynomial write-enable

```text
g(c_1,...,c_n)
```

that is required to satisfy both:

```text
g = 0 on every non-witness sign orthant
g != 0 on at least one witness sign orthant.
```

Every sign orthant is a nonempty open subset of the phase chart. A real polynomial that
vanishes on any nonempty open set is the zero polynomial. Therefore `g` must vanish
everywhere, contradicting the required witness activation.

The same identity principle applies to a real-analytic write-enable on a connected chart.
A vector-valued write-enable does not evade the result because it applies component by
component.

Therefore:

```text
No nonzero polynomial or real-analytic write-enable can be exactly supported only on
threshold-witness orthants.
```

## Scope of the wall

This result rejects the naive persistence mechanism:

```text
smooth polynomial field
+ exact write activation only when external threshold verification passes
+ exact inactivity on every non-witness open orthant.
```

It does not reject every possible persistence mechanism.

The surviving routes are:

1. **Explicit discontinuous or hybrid threshold event**
   Include the event surface, timing precision, false-trigger law, totality, standard-model
   simulation, and restoration cost in the public resource theorem.
2. **Topological crossing record**
   Store an oriented crossing or intersection invariant without requiring an analytic
   function with compact support on witness orthants.
3. **Proved polynomial Booleanization**
   First prove every SAT public trajectory reaches an exact or robust literal phase
   boundary in polynomial time, then use the existing exact-product zero channel.
4. **Different smooth persistence law**
   Construct a smooth mechanism whose correctness does not depend on exact orthant
   support and independently prove it cannot latch a false witness.

## Consequence for PR #49

The odd-parity counterexample cannot be repaired merely by adding a polynomial scalar
whose derivative turns on exactly when the external sign verifier sees a witness. That
write-enable cannot exist in the current smooth polynomial ontology.

The branch must either:

```text
repair the terminal deadline without latching
change the mechanism class and account for the new boundary primitive
or discover a topological/smooth persistence construction with a separate correctness proof.
```

## Remaining obligations

This theorem does not establish:

```text
impossibility of all hybrid witness latches
impossibility of topological crossing records
impossibility of smooth latches without exact orthant support
polynomial Booleanization
formula-uniform polynomial decision time
robust terminal verification
native restoration
P = NP.
```
