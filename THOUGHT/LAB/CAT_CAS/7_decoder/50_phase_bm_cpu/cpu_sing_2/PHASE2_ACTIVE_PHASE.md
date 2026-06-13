# PHASE2_ACTIVE_PHASE

## Verdict

ACTIVE_PHASE_NO_LOCK

## Protocol

`phase2_probe.c` ran repeated active readout scenarios with Core2 as sampler, Core3/Core4 as PPU cores, and Core5 as reference. It sampled TSC plus Core3/Core4/Core5 counters and shared-line state.

Scenarios, three repeats each:

- `idle`: Core3 off, Core4 off, Core5 reference.
- `single3`: Core3 only.
- `single4`: Core4 only.
- `same_noline`: Core3/Core4 same LCG workload, no shared-line atomic channel.
- `same_shared`: same workload with shared-line atomic pressure.
- `atomic_shared`: Core3/Core4 atomic shared-line workload.
- `mixed_mul_mem`: Core3 multiply workload, Core4 memory workload.
- `branch_shared`: branch workload with shared-line pressure.

Analyzer metrics:

- Kuramoto-style r from Core3/Core4 phases relative to Core5.
- Core3/Core4 phase concentration r.
- Core3/Core4 delta correlation.
- Shuffled timestamp nulls.

## Results

```text
atomic_shared n=3 k=0.6383 p34=0.0754 corr=-0.0018
branch_shared n=3 k=0.6369 p34=0.0806 corr=-0.0177
mixed_mul_mem n=3 k=0.6281 p34=0.0751 corr=0.0000
same_noline n=3 k=0.6364 p34=0.0758 corr=-0.0095
same_shared n=3 k=0.6361 p34=0.0773 corr=-0.0167
single3 n=3 k=0.6365 p34=0.0749 corr=0.0000
single4 n=3 k=0.6323 p34=0.0747 corr=0.0000
idle n=3 k=1.0000 p34=1.0000 corr=0.0000
```

Null comparison:

```text
coupling real_k_mean 0.6806
coupling shuf_k_mean 0.6806
coupling real_p34_mean 0.1917
coupling shuf_p34_mean 0.1926
coupling real_corr34_mean -0.0057
coupling shuf_corr34_mean -0.0007
```

## Decision

The active shared-line route does not show phase transfer. Real metrics do not separate from shuffled nulls, and Core3/Core4 correlation stays near zero. The idle case is a trivial null from both counters being static and is not accepted as coupling.

