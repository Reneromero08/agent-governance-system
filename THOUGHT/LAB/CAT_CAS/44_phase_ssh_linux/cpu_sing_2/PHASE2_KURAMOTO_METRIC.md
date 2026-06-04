# PHASE2_KURAMOTO_METRIC

## Verdict

KURAMOTO_LOCK_NOT_OBSERVED

## Definition

Phase was defined operationally as the angle of each sampled core counter modulo 1024, with Core5 as the reference. For each sample:

```text
theta35 = phase(Core3) - phase(Core5)
theta45 = phase(Core4) - phase(Core5)
r = |mean(exp(i theta35), exp(i theta45))|
```

Core3/Core4 direct phase concentration used:

```text
phase34_r = |mean(exp(i(phase(Core3)-phase(Core4))))|
```

## Result

The apparent Kuramoto r stayed near `0.63-0.64` across active conditions, but shuffled nulls matched it. Direct Core3/Core4 phase concentration stayed near `0.075-0.093`, far below the requested lock threshold.

```text
coupling real_k_mean 0.6806
coupling shuf_k_mean 0.6806
detune real_k_mean 0.6372
detune shuf_k_mean 0.6372
```

## Decision

No reproducible order parameter `r >= 0.8` was found. Stable phase difference relative to Core5 was not demonstrated.

