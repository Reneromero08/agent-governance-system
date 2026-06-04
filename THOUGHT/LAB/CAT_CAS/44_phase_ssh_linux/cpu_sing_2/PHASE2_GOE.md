# PHASE2_GOE

## Verdict

GOE_NOT_OBSERVED

## Method

The analyzer formed compact route vectors from correlation and phase metrics, then computed a spacing-ratio proxy across the route matrix. This is not a publication-grade GOE pipeline, but it is sufficient to triage whether the current Phase 2 measurements are near the target.

Targets:

- Poisson-like: approximately `0.39`.
- GOE-like: approximately `0.51-0.53`.

## Results

```text
coupling matrix spacing ratio: 0.2906
detuning matrix spacing ratio: 0.1662
```

## Decision

No GOE-like spacing ratio was observed. The data do not support a non-Poisson coupled-oscillator claim.

