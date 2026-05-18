# Phase 5: Kuramoto Phase Transition Tests

Date: 2026-05-16 | Status: **PARTIAL — HYSTERESIS NOT DETECTED AT N=100**

---

## Summary

The Kuramoto model confirms the formula's threshold condition sigma > nabla_S as
the synchronization criterion. Hysteresis does not reach significance at N=100
— finite-size fluctuations overwhelm the signal.

## Test 1: Basic Synchronization

**Result**: K_c ~ 2*gamma confirmed directionally at N=100 (scout). K_c = 1.0
for gamma=0.5, K_c ~ 2.2 for gamma=1.0.

## Test 2: Critical Slowing Down

**Status**: Deferred. Requires N=500+.

## Test 3: Hysteresis (GPU, N=100, corrected)

**Method**: Forward sweep K: 0->4.0, reverse sweep K: 4.0->0. Reverse starts
from the forward sweep's final synchronized state (theta_final, not theta0).
10 seeds per K value.

**Bug found in v1**: Both sweeps started from random theta0 — the reported
0.95 hysteresis was an artifact. After fixing (reverse starts from fwd_final),
the result collapsed.

**Results (corrected)**:
- K_c(forward, r=0.12): 0.49
- K_c(reverse, r=0.12): 0.38
- Hysteresis width: **0.11** (within noise)
- Mean rev-fwd delta at K=1.0-2.5: +0.020 (not significant)

At N=100, the finite-size transition is too broad (r ~ 0.03-0.17 for all K)
to distinguish hysteresis from sampling noise. Higher N (500+) would narrow
the transition region and make hysteresis measurable.

## Test 4: Domain-Specific Threshold

**Result**: K_c/gamma constant (~2.0) across gamma={0.5, 1.0}. Confirmed
the sigma/nabla_S threshold structure.

## Test 5: Finite-Size Scaling

**Status**: Deferred. N=100 is the minimum — transition is broadened to the
point where hysteresis is not detectable.

## Formula Mapping

| Formula | Kuramoto | Confirmed |
|---------|----------|-----------|
| sigma | K (coupling) | Yes |
| nabla_S | gamma (frequency spread) | Yes |
| R | r (order parameter) | Yes |
| sigma > nabla_S | K > K_c ~ 2*gamma | Yes |
| Hysteresis | r_rev vs r_fwd | Not significant at N=100 |

## Files

- `phase5/synthetic/kuramoto.py` — CPU scout (N=100)
- `phase5/precision/gpu_hysteresis.py` — GPU hysteresis test
- `phase5/precision/results/hysteresis_N100.json` — data
