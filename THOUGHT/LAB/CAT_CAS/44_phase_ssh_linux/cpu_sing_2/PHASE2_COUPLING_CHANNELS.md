# PHASE2_COUPLING_CHANNELS

## Verdict

COUPLING_CHANNELS_EXHAUSTED_SOFTWARE

## Channels Tested

- Power-grid style concurrent compute pressure.
- Shared L3/MESI no-line versus shared-line.
- Atomic contention on a shared cache line.
- Memory bus pressure.
- Multiply-heavy workload.
- Branch-heavy workload.
- Single-core nulls.
- Idle null.

## Evidence

The repeated channel matrix showed no real/null separation:

```text
same_noline k=0.6364 p34=0.0758 corr=-0.0095
same_shared k=0.6361 p34=0.0773 corr=-0.0167
atomic_shared k=0.6383 p34=0.0754 corr=-0.0018
mixed_mul_mem k=0.6281 p34=0.0751 corr=0.0000
branch_shared k=0.6369 p34=0.0806 corr=-0.0177
```

The prior roadmap already identified the passive TSC route as dominated by the fixed 2.67 MHz VRM artifact and a non-reproducible 5.34 MHz component. The active route here did not recover a stronger channel.

## Decision

Authorized software coupling channels tested in this pass are rejected as Phase 2 success routes. No unknown writes were used.

