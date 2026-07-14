# Value Orientation Symmetry Law

Status: `LAW_FROZEN_FOR_FUTURE_PROTOCOL`

## Problem

Candidate-label swaps do not detect value-derived physical orientation. This rule:

```text
min(a,N-a) -> lane A
max(a,N-a) -> lane B
```

can survive label-name swapping while still encoding branch orientation into bank,
route, lane, order, or address.

## Required Blinded Action

The future protocol must blind physical branch-to-lane assignment:

```text
pi: {branch_0, branch_1} -> {lane_A, lane_B}
```

where `pi` is generated independently of branch value and hidden from any code that
could choose route, bank, order, address, or workload from branch value.

## Invariance Law

For relation `R = {a,b}`:

```text
P(R, pi) equivalent P(R, swap(pi))
Y_q(P(R, pi)) equivalent Y_q(P(R, swap(pi)))
J_q(P(R, pi)) equivalent J_q(P(R, swap(pi)))
```

up to the frozen physical equivalence metric.

## Fail-Closed Cases

The protocol is invalid if any of these predict the response:

```text
min/max lane rule
value-derived bank assignment
value-derived route assignment
value-derived order assignment
metadata row-shape dictionary
opaque id that maps to condition through public schedule structure
```

## Current Result

The law is frozen as a future requirement. It is not passed by the current
architecture because no exact blinded physical preparation map exists.

