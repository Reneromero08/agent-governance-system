# Topological Witness-Record Wall

## Status

```text
PURE_TOPOLOGICAL_WITNESS_VISIT_RECORDER_REJECTED
TRANSIENT_WITNESS_VISIT_IS_NOT_HOMOTOPY_INVARIANT
UNSIGNED_CROSSING_HISTORY_IS_GEOMETRIC_NOT_TOPOLOGICAL
HYBRID_EVENT_RESOURCE_CLOSURE_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Exact separating paths

Use the exact odd-parity three-variable relation from the fixed-deadline
counterexample. Its assignment

```text
FFF
```

is not a witness, while

```text
TFF
```

is a witness.

Represent phase angles in quarter turns:

```text
0 -> angle 0   -> cosine +1
1 -> angle pi/2 -> cosine 0
2 -> angle pi   -> cosine -1.
```

Consider the out-and-back excursion

```text
(2,2,2)
-> (1,2,2)
-> (0,2,2)
-> (1,2,2)
-> (2,2,2).
```

The path starts and ends at `FFF`, crosses into the witness orthant `TFF`, and
then exactly retraces its incoming arc.

Its phase displacement and winding are

```text
net quarter-turn displacement = (0,0,0)
winding vector = (0,0,0).
```

The two oriented crossings of the threshold surface `c_1 = 0` are

```text
entry = +1
exit  = -1
net oriented crossing = 0.
```

The excursion is a path followed by its inverse. It is endpoint-fixed
null-homotopic and therefore has the same homotopy and winding data as the
constant loop at `FFF`, even though only the excursion visits a witness.

## Consequence

No recorder depending only on:

```text
path homotopy class
phase winding
net oriented crossing number
```

can distinguish exact transient witness visitation in general.

Counting entry and exit without orientation would distinguish these two paths,
but unsigned crossing count is not topological. A small deformation can create
or annihilate a crossing pair without changing path endpoints or homotopy
class.

Therefore a persistent record of witness visitation must carry genuine history:

```text
geometric boundary-event history
hybrid or discontinuous state transition
non-topological hysteresis
or another mechanism with equivalent resource accounting.
```

Calling such history a topological invariant would hide the actual event-detection,
precision, storage, and restoration costs.

## Relationship to the latch wall

The threshold-witness latch wall ruled out a nonzero polynomial or real-analytic
write-enable supported exactly on satisfying sign orthants.

This theorem separately rules out the idea that transient witness visitation can be
preserved for free by ordinary path topology or winding.

Together they leave a narrower mechanism space:

1. an explicitly modeled hybrid threshold event;
2. a geometric recorder with non-topological hysteresis and full resource bounds;
3. proved polynomial Booleanization followed by exact boundary recording;
4. a different smooth persistence law whose false-positive exclusion does not rely on
   compact support or path homotopy.

## Claim boundary

This theorem does not prove that every hybrid recorder is inefficient or impossible.
It proves only that the recorder must account for non-topological history and cannot be
obtained from winding or homotopy alone.

It does not establish:

```text
polynomial event-detection precision
polynomial witness-margin or dwell time
reversible event-history restoration
a total polynomial deadline
public-seed completeness
P = NP.
```
