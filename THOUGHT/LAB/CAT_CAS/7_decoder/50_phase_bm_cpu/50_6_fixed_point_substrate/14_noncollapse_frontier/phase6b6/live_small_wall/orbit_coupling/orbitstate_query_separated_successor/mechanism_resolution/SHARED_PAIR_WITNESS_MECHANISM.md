# Shared Pair Witness Mechanism

Status: `NO_EXACT_FAMILY10H_WITNESS_FROZEN`

## Intended Object

```text
R = unordered relation {a, N-a}
P(R) = physical preparation
Q(q) = receiver-only post-source query
Y(P(R), q) = measured observable
Restoration(P(R), q) = declared return or closure operation
```

## Synthetic Nonadditive Pair Mechanism

The reference model defines a pair-level state:

```text
P_syn(R) = pair_incidence_state(a, b)
Y_q(P_syn(R)) = A_q(a) + B_q(b) + I_q(a,b)
```

where `I_q(a,b)` is nonzero only when both branches jointly occupy one relation
state. The branch-local controls make:

```text
J_q = Y_q(a,b) - Y_q(a,empty) - Y_q(empty,b) + Y_q(empty,empty)
J_q = I_q(a,b)
```

An additive branch model has `J_q = 0`. This is the correct mathematical shape for a
future nonseparability gate.

This synthetic mechanism is not a Family 10h witness. It is deterministic in
`(R,q)` and is exactly reproducible by a finite answer cache on a closed query set.

## Required Family 10h Physical Instantiation

A real witness would need to freeze all of:

```text
physical pages
cache lines
sets or banks
cores
ownership/coherence states
source operations
source lifetime
source-death condition
handoff
query operation
measurement window
restoration operation
```

The preparation must be invariant under blinded branch-to-lane permutation. A rule
such as:

```text
min(a,N-a) -> lane A
max(a,N-a) -> lane B
```

fails the value-orientation symmetry law.

## Candidate Physical Sketch And Its Blocker

The prior architecture names:

```text
shared Family 10h ownership-intent topology over paired cache-line sets
```

but it does not identify the persistent physical state that remains after source
closure. It also does not prove that relation mutation can change only pair incidence
while preserving:

```text
same pages or address population
same bank population
same operation count
same source timing envelope
same route/core allocation
same order distribution
same branch-local occupancy
```

Without that exact state and intervention, an observed contrast can still be ordinary
route, bank, address, timing, order, or value-orientation structure.

## Selected Physical Witness Mechanism

None.

The package preserves `SHARED_PAIR_TOPOLOGY_QUERY_SEPARATED_CARRIER` as a research
direction only. It does not freeze it for protocol.
