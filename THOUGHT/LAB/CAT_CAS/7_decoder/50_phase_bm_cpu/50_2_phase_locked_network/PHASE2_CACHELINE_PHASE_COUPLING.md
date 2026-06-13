# PHASE2_CACHELINE_PHASE_COUPLING

## Verdict

`CACHELINE_PHASE_COUPLING_REJECTED`

The renewed software-only cacheline coupling route did not produce reproducible Kuramoto/phase behavior.

## Objective

Test whether two core-pinned software oscillators on Cores 3 and 4 show phase concentration when their phase counters are placed on the same cache line, compared with isolated cache lines and cyclic-shift nulls.

This is a no-hardware, no-voltage, no-BIOS route.

## Artifact

- `50_2_phase_locked_network/src/cacheline_phase_coupling.c`

## Command

```powershell
Get-Content -Raw 50_2_phase_locked_network\src\cacheline_phase_coupling.c |
  ssh -o BatchMode=yes -o ConnectTimeout=8 root@192.168.137.100 "cat > /tmp/cacheline_phase_coupling.c && gcc -O2 -pthread /tmp/cacheline_phase_coupling.c -lm -o /tmp/cacheline_phase_coupling && /tmp/cacheline_phase_coupling 8 220"
```

## Protocol

Three modes were run for eight repeats:

| Mode | Meaning |
|---|---|
| `isolated_lines` | Cores 3 and 4 publish counters to separate cache lines. |
| `false_shared_line` | Cores 3 and 4 publish counters to adjacent fields on the same cache line. |
| `atomic_same_line` | Cores 3 and 4 apply atomic pressure to adjacent fields on the same cache line. |

Sampler core:

```text
Core 2
```

Acceptance required same-line modes to show stable real phase concentration above isolated-line and cyclic-shift nulls. Phase concentration was measured as circular resultant length `real_r` over low counter phase modulo 4096. Nulls used deterministic cyclic shifts of the second oscillator stream.

## Result Summary

No mode reached an accepted lock.

| Mode | Repeats | Observed real_r range | Max observed real_r | Null relation |
|---|---:|---:|---:|---|
| `isolated_lines` | 8 | `0.004660-0.032650` | `0.032650` | Same scale as nulls. |
| `false_shared_line` | 8 | `0.010798-0.026811` | `0.026811` | Same scale as nulls; often below max null. |
| `atomic_same_line` | 8 | `0.009917-0.043476` | `0.043476` | Same scale as nulls; not stable and often below max null. |

Strongest single row:

```text
atomic_same_line repeat 3 real_r=0.043476 max_null_r=0.027928
```

That row is far below any useful phase-lock threshold and is not stable across repeats.

Representative null failures:

```text
false_shared_line repeat 0 real_r=0.015048 max_null_r=0.018617
false_shared_line repeat 1 real_r=0.016739 max_null_r=0.027857
false_shared_line repeat 3 real_r=0.025401 max_null_r=0.038117
atomic_same_line repeat 4 real_r=0.018324 max_null_r=0.040335
```

## Interpretation

Same-cache-line pressure created measurable software/cache contention, but not phase lock:

- `real_r` stayed near zero in every mode.
- False sharing did not improve concentration over isolated counters.
- Atomic same-line pressure had occasional small spikes, but they were not stable and did not exceed nulls consistently.
- This does not produce accepted Kuramoto, Ising, or CPU-sings evidence.

## Route Impact

Route 5 advances with one more rejected software-only coupling path:

`CACHELINE_PHASE_COUPLING_REJECTED`

This does not prove a true wall by itself. It closes this specific scheduler/cacheline passive-coupling attempt.

## Safety

- No MSR writes.
- No voltage writes.
- No BIOS flash.
- No P0-P3 modification.
- No external measurement or Tier 3 route.
