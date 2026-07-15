# Audio Query-Separation Law

Status: `FINITE_QUERY_EQUIVALENCE_CARRIED_FORWARD`

## Theorem

For source-known state `R`, finite enumerable query family `Q`, and deterministic
response law `h`:

```text
V(R) = [h(R,q) for q in Q]
```

can be prepared before source closure. A delayed receiver query can select a coordinate
of `V(R)`. Therefore delayed query selection is necessary custody but is not an
identifiability proof.

## Query Classes

| Class | What delayed selection excludes | What survives |
| --- | --- | --- |
| Closed finite set | Source learning the realized index | Literal all-answer cache |
| Held-out discrete point | Memorized training table cell | Formula, basis, relation representation |
| Continuous/high-resolution phase | Finite literal enumeration at exact precision | Two-parameter sinusoidal generator |
| Public filter bank | Hidden receiver branch choice | Stored complex coefficients |
| Bounded physical query | Nothing without measured carrier capacity | Capacity-limited generator may or may not fit |

Continuous queries do not automatically defeat compact formulas. The reference engine
demonstrates a two-parameter amplitude/phase generator that answers held-out continuous
phase queries exactly.

## Source/Receiver Custody For A Future Prototype

```text
public preparation freezes
source waveform and hash freeze
source drives physical carrier
source route mechanically disconnects
source process closes
independent query parameters commit
receiver applies query
capture freezes
features freeze
analysis begins
```

This sequence prevents post-query source adaptation. It does not by itself exclude a
cache or compressed generator already embedded in the carrier.

## Capacity Separation Required

A future positive identifiability result requires:

```text
B_answer_min > C_prep
B_relation <= C_prep
```

`B_answer_min` must minimize over literal tables, compression, low-rank bases, public
side information, circuits, formulas, seeds, linear filters, and nonlinear filters.
Raw query count times response precision is not a lower bound.

## Frozen Attacks

The offline reference explicitly executes and expects survival of:

```text
finite answer cache
compressed answer generator
query preselection
ordinary analytic-signal DSP
multitone coefficient replay
ordinary linear filter
ordinary polynomial nonlinear filter
file serialization replay
```

It separately kills label-only renaming, spectral-energy-only explanation for the
matched result, file metadata dependence, and a 37 Hz scrambled query.

## Result

```text
finite-query equivalence = established
continuous-query formula attack = survives
query custody law = specified, not physically implemented
capacity separation = not established
physical query-separated carrier = not established
```
