# EXP50 PHASE 5.1-5.5 - PROXY HARDENING

**Status:** `PHASE5_3_PINNED_TIMING_HARDENED_PROXY__PHASE5_4_REFERENCE_TO_MULTICHANNEL_PROXY_MEASURED__PHASE5_5_NOISE_JITTER_SHUFFLE_NULL_MEASURED__RESTORATION_INTACT__RANK1_PROXY_PARTIAL__NOISE_TEMPORAL_STRUCTURE_NOT_SEPARATED_FROM_SHUFFLE`

## Scope

This run pushes Phase 5.3-5.5 inside current Phenom software observability.
It is a proxy hardening run, not a physical-limit proof. It does not write
voltage, clocks, firmware, or MSRs.

## Results

| Gate | Result |
|---|---|
| 5.3 pinned timing | restore rate `1.000000`, all-core median reverse/forward `0.997884` |
| 5.4 reference-to-channel proxy | abs correlation floor `0.720324`, sign agreement `1.000000`, rank-1 explained energy `0.556533` |
| 5.5 noise jitter shuffle null | real median `0.389339`, shuffled median `0.389339`, delta `0.000000` |

## Interpretation

5.3 is hardened as a pinned timing proxy across cores. 5.4 gains a measured
many-channel software proxy for one reference coordinate. 5.5 gains a
shuffle-null cap for noise-only jitter: if the median delta is near zero, the
current software-visible noise ordering is not separated from the shuffled
null.

## Claim Boundary

This does not close physical 5.1, 5.2, 5.4, or 5.5. Those still require the
external/physical artifacts listed in `PHASE5_1_5_LIVE_RUNBOOK.md`.

## Artifacts

- `50_5_1_limit_violations/results/proxy_hardening/phase5_1_5_proxy_hardening_summary.json`
- `50_5_1_limit_violations/src/phase5_1_5_proxy_hardening.py`
