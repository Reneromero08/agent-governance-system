# Deterministic Quantum Amplifier Wall

## Oracle-State Separation

Prepare the uniform assignment state and apply a phase of pi to every satisfying basis
state. If `k` assignments satisfy a formula over `n` variables, the overlap between the
unmarked reference and marked oracle state is:

```text
<psi_0|psi_F> = 1 - 2k/2^n.
```

For pure states, their trace distance is:

```text
D = sqrt(1 - |1 - 2k/2^n|^2).
```

For a unique witness:

```text
D = O(2^(-n/2)).
```

## Data Processing

Trace distance cannot increase under a deterministic completely positive
trace-preserving quantum channel. Therefore a formula-blind one-shot device placed after
the ordinary phase oracle cannot convert the unique-witness state difference into a
constant classical separation.

This applies to a deterministic EP amplifier if its complete gain, loss, environment,
and readout form one ordinary quantum channel. An apparent constant separation must
come from at least one of:

```text
postselection or conditioning
multiple oracle uses
formula-dependent structure used inside the amplifier
nonlinear or non-CPTP dynamics
non-quantum physical resources
an omitted gain/loss environment
an exponentially sensitive noise floor or control parameter
```

## Scope Boundary

This is not a lower bound against a genuinely formula-native CAT_CAS operation that uses
the complete local relation geometry rather than treating the phase oracle as a black
box. It is a killer control against attaching a universal deterministic amplifier after
an exponentially close pair of oracle states.

If CAT_CAS establishes a nonlinear or nonstandard substrate, its physical resource law
and standard-model simulation theorem remain separate obligations.

## Current Result

```text
TRACE_DISTANCE_CONTRACTIVITY_BLOCKS_CONSTANT_ONE_SHOT_AMPLIFICATION
FORMULA_NATIVE_OR_NONSTANDARD_DYNAMICS_REMAIN_OPEN
P_EQUALS_NP_NOT_PROVEN
```
