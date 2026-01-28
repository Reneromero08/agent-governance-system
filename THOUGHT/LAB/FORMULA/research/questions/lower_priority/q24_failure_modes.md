# Question 24: Failure modes (R: 1280)

**STATUS: RESOLVED**

## Question
When the formula says "gate CLOSED," what's the optimal response? Wait for more context? Change observation strategy? Accept uncertainty? Escalate to human review?

## Experiment Summary

**Data**: SPY market data (3 years, 751 data points)
**Low-R Periods Found**: 17 periods where R < 0.8

### Strategies Tested

| Strategy | Description | R Improvement | Success Rate |
|----------|-------------|---------------|--------------|
| WAIT | Collect more context over time | -34.07% | 0.00% |
| CHANGE_FEATURES | Use different observation windows | +79.96% | 60.00% |
| ACCEPT_UNCERTAINTY | Proceed with low confidence | 0% | 94.12% |
| ESCALATE | Defer to human review | -8.20% | 76.47% |

### Key Findings

1. **WAIT is counterproductive**: Simply waiting makes R *worse* on average (-34%). The market doesn't "settle" - uncertainty compounds.

2. **CHANGE_FEATURES is most effective for R improvement**: Using different feature windows (especially longer 50-day windows) improves R by ~80% on average. When the gate is closed, try observing from a different timescale.

3. **ACCEPT_UNCERTAINTY has highest success rate**: 94% of the time, proceeding despite low R led to acceptable outcomes (volatility within 1.5x expected). This suggests low R doesn't always indicate danger.

4. **ESCALATE provides safety margin**: 76% success rate at avoiding bad outcomes. Valuable when R is "dangerously" low (below 0.3).

### Closure Reasons

All 17 low-R periods had closure reason: `BOTH` (high sigma AND low E). This suggests market disagreement periods tend to have both dispersed observations and low mean agreement.

### Market Regime Distribution
- Sideways: 82.4%
- Bull: 11.8%
- Bear: 5.9%

Most low-R periods occur during sideways/ranging markets, not during strong trends.

## Recommendations

### Decision Tree for Closed Gate:

```
R < threshold (gate CLOSED)
    |
    +-- R < 0.3 (dangerously low)?
    |       |
    |       +-- Yes: ESCALATE to human review
    |       |
    |       +-- No: Continue below
    |
    +-- Is action time-sensitive?
            |
            +-- No: CHANGE_FEATURES (try longer window)
            |
            +-- Yes: ACCEPT_UNCERTAINTY (proceed with caution)
```

### Strategy by Closure Reason:

| Closure Reason | Best Strategy | Rationale |
|----------------|---------------|-----------|
| HIGH_SIGMA | CHANGE_FEATURES | Different timescale may show consensus |
| LOW_E | CHANGE_FEATURES | Different features may reveal agreement |
| BOTH | CHANGE_FEATURES (+80% improvement) | Most effective empirically |
| INSUFFICIENT_DATA | WAIT (gather more observations) | Only case where waiting helps |

## Verdict

**Original Hypothesis**: "Waiting improves subsequent R by >20%"
**Result**: NOT SUPPORTED (-34% average, 0% success)

**Alternative Finding**: CHANGE_FEATURES strategy improves R by +80% on average with 60% success rate. When gate closes, change your observation strategy rather than waiting.

**Practical Implication**: Low R is often a signal to observe differently, not to wait. The system should automatically try alternative feature windows before blocking an action.

## Test Artifacts

- Test: `experiments/open_questions/q24/test_q24_failure_modes.py`
- Results: `experiments/open_questions/q24/q24_test_results.json`
- Date: 2026-01-27
