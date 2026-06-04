# PHASE2_DEEP_5_RETEST

## Verdict

RETEST_PROTOCOL_READY_WAITING_FOR_BETTER_OBSERVABILITY

## When To Retest

Retest only after one of these changes:

- External waveform capture is available.
- A P4-safe firmware control surface is proven design-ready.
- A new non-destructive measurement path gives better signal than TSC jitter.

Do not rerun the prior software-only sweeps blindly.

## Metrics

Phase definition:

- Extract phase from a dominant marker-correlated waveform component.
- Core5 is the reference.
- Core3/Core4 phase difference is measured relative to Core5.

Kuramoto order parameter:

```text
r = |mean(exp(i theta3), exp(i theta4))|
```

Pass threshold:

- `r >= 0.8` for repeated runs, or a stable Core3/Core4 phase difference with null separation.

Ising energy:

```text
E = -sum(J_ij cos(theta_i - theta_j))
```

Accept only if energy decreases reproducibly under a defined state schedule and beats random/shuffled state schedules.

GOE:

- Build a phase correlation matrix from marker-aligned windows.
- Compute adjacent spacing ratio.
- Compare target range `0.51-0.53` against shuffled null and idle null.

## Required Nulls

- Idle.
- Core3-only.
- Core4-only.
- Core5-only.
- Same workload without shared-line pressure.
- Shared-line workload.
- Shuffled marker labels.
- Repeated captures on separate runs.

## Stop Conditions

Stop immediately if:

- k10temp reaches 60 C.
- P4 readback differs from stock unexpectedly.
- SSH becomes unstable.
- External waveform capture has no marker-aligned timing.

