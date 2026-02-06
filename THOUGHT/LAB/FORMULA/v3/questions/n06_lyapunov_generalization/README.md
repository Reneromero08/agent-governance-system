# N6: Is the Positive Lyapunov-R Correlation General?

## Why This Question Matters

Q52 predicted R would negatively correlate with Lyapunov exponent (more chaos = less coherence). The opposite was found: positive correlation. R (participation ratio) increases with chaos. If this generalizes, R measures effective dimensionality of attractors -- a novel characterization nobody else has published linking embedding statistics to dynamical systems.

## Hypothesis

**H0:** R (participation ratio) positively correlates with Lyapunov exponent across a broad class of dynamical systems.

**H1:** The positive correlation is specific to logistic map / Henon attractor and does not generalize.

## Pre-Registered Test Design

### Systems (minimum 6, across 3 classes)

**Classical chaotic systems:**
| System | Parameters | Lyapunov (known) |
|--------|-----------|-----------------|
| Logistic map | r = 2.5 to 4.0 (Q52 replication) | Analytically known |
| Henon attractor | a = 0.1 to 1.4 (Q52 replication) | Numerically known |
| Lorenz attractor | rho = 0 to 200 | Numerically known |
| Rossler attractor | a = 0.1 to 0.4 | Numerically known |

**Delayed systems:**
| System | Parameters | Lyapunov |
|--------|-----------|----------|
| Mackey-Glass | tau = 5 to 30 | Computed from time series |

**Real-world time series:**
| System | Source | Lyapunov |
|--------|--------|----------|
| Weather (temperature) | NOAA daily temperatures | Estimated from data |
| Heartbeat (RR intervals) | PhysioNet MIT-BIH | Estimated from data |
| Stock prices (S&P 500) | yfinance | Estimated from data |

### Procedure

1. For each system, generate/obtain time series at multiple parameter values
2. Embed time series in R^d using delay embedding (Takens' theorem)
3. Compute R (participation ratio of eigenspectrum of delay-embedded covariance matrix)
4. Compute Lyapunov exponent (Wolf et al. algorithm for real data)
5. Correlate R with Lyapunov across parameter sweep

### Success Criteria

- **Generalizes:** Positive r > 0.5 (p < 0.01) for >= 5/8 systems
- **Partial:** Positive correlation for 3-4/8 systems
- **Does not generalize:** Positive correlation for < 3/8 systems

### Implications

- If general: participation ratio is a universal dimensionality probe for dynamical systems
- If partial: the relationship is system-class-dependent (interesting to characterize)
- If not general: Q52's finding was specific to 1D/2D maps

## Dependencies

- None. Can run immediately. Q52 provides replication targets.

## Related

- v2/Q52 (Chaos theory -- the falsification that generated this question)
- v2/Q28 (Attractors -- related dynamical systems analysis)
- N4 (Architecture invariants -- if R measures dimensionality, it may explain convergence)
