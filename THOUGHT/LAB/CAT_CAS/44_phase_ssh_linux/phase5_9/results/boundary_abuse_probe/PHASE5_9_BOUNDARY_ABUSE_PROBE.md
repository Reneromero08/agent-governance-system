# Phase 5.9 Boundary Abuse Probe

Verdict: `CARRIER_SATURATION_EDGE_ADVANCED`

Objective: creatively push the carrier/failure boundary with software substrate abuse while avoiding voltage writes, BIOS flash, and destructive hardware actions.

## Metrics

- Runs analyzed: 12
- Restoration failures: 0
- r(boundary_thickness, cycle_cv): 0.729327
- r(boundary_thickness, spike_rate): -0.060804
- r(boundary_thickness, p99_p50): 0.602214
- Max thickness run: ABUSE_syscall_R1 = 21113.361835
- Max/quiet thickness ratio: 3.938315

## Rows

| Run | Mode | Thickness | CV | Spike rate | p99/p50 |
|-----|------|-----------|----|------------|--------|
| ABUSE_branch_R1 | branch | 2739.835229 | 0.240194 | 0.000075 | 1.691635 |
| ABUSE_branch_R2 | branch | 15342.913080 | 0.685403 | 0.000475 | 3.994340 |
| ABUSE_cache_R1 | cache | 613.486208 | 0.173548 | 0.000125 | 1.438043 |
| ABUSE_cache_R2 | cache | 15291.262707 | 0.441471 | 0.046050 | 2.981939 |
| ABUSE_mixed_R1 | mixed | 9525.147429 | 0.539916 | 0.000050 | 3.012107 |
| ABUSE_mixed_R2 | mixed | 11848.295251 | 0.440255 | 0.000125 | 2.008347 |
| ABUSE_pagefault_R1 | pagefault | 3097.800206 | 0.278482 | 0.000125 | 1.698202 |
| ABUSE_pagefault_R2 | pagefault | 51.130032 | 0.183157 | 0.000125 | 1.092580 |
| ABUSE_quiet_R1 | quiet | 1481.626999 | 0.156000 | 0.007975 | 1.340145 |
| ABUSE_quiet_R2 | quiet | 9240.401318 | 0.292902 | 0.000100 | 1.502784 |
| ABUSE_syscall_R1 | syscall | 21113.361835 | 0.336673 | 0.000025 | 1.670308 |
| ABUSE_syscall_R2 | syscall | 944.575306 | 0.102581 | 0.060975 | 1.415656 |
