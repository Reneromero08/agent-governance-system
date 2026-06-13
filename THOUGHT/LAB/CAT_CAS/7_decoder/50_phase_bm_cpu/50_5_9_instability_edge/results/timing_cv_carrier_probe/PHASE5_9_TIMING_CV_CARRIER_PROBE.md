# Phase 5.9 Timing-CV Carrier Probe

Verdict: `TIMING_CV_CARRIER_CONFIRMED`

Objective: test whether boundary thickness tracks sustained timing CV under controlled P-state and worker-mode variation.

This is not a failure-edge claim. It tests the carrier thread left open by Phase 5.9C.

## Metrics

- Runs analyzed: 18
- Restoration failures: 0
- r(boundary_thickness, cycle_cv): 0.584572
- r(boundary_thickness, spike_rate): -0.053230
- r(boundary_thickness, p99_p50): 0.554754

## Acceptance

- `TIMING_CV_CARRIER_CONFIRMED`: |r_cv| >= 0.5 and stronger than spike-rate correlation.
- `TIMING_CV_CARRIER_CANDIDATE`: |r_cv| >= 0.3.
- `SPIKE_ARTIFACT_DOMINANT`: spike-rate correlation dominates.
- `NO_TIMING_CV_CARRIER`: timing-CV relationship does not reproduce.

## Rows

| Run | Thickness | CV | Spike rate | p99/p50 |
|-----|-----------|----|------------|--------|
| CVCARRIER_P0_cache_R1 | 54.550588 | 0.177653 | 0.000167 | 1.025539 |
| CVCARRIER_P0_cache_R2 | 2079.538328 | 0.241863 | 0.000067 | 1.675041 |
| CVCARRIER_P0_mixed_R1 | 3981.786121 | 0.277397 | 0.001800 | 1.672285 |
| CVCARRIER_P0_mixed_R2 | 2615.047758 | 0.257321 | 0.000267 | 1.674338 |
| CVCARRIER_P0_none_R1 | 1751.871303 | 0.207010 | 0.000100 | 1.670361 |
| CVCARRIER_P0_none_R2 | 19.512559 | 0.047738 | 0.000667 | 1.008024 |
| CVCARRIER_P2_cache_R1 | 60.013661 | 0.092251 | 0.000100 | 1.013449 |
| CVCARRIER_P2_cache_R2 | 1204.930938 | 0.127644 | 0.036533 | 1.696448 |
| CVCARRIER_P2_mixed_R1 | 4358.749084 | 0.228578 | 0.000067 | 1.671962 |
| CVCARRIER_P2_mixed_R2 | 36.387011 | 0.061762 | 0.000167 | 1.005372 |
| CVCARRIER_P2_none_R1 | 1987.827595 | 0.141861 | 0.004200 | 1.417775 |
| CVCARRIER_P2_none_R2 | 7.579009 | 0.019213 | 0.000100 | 1.003557 |
| CVCARRIER_P4_cache_R1 | 2216.111674 | 0.132649 | 0.034000 | 1.670252 |
| CVCARRIER_P4_cache_R2 | 3313.153284 | 0.296522 | 0.002267 | 1.684367 |
| CVCARRIER_P4_mixed_R1 | 303.680257 | 0.062051 | 0.003867 | 1.009529 |
| CVCARRIER_P4_mixed_R2 | 10504.216966 | 0.241066 | 0.000067 | 1.698165 |
| CVCARRIER_P4_none_R1 | 3174.386427 | 0.114707 | 0.086067 | 1.415495 |
| CVCARRIER_P4_none_R2 | 15158.371786 | 0.260484 | 0.000200 | 1.675008 |
