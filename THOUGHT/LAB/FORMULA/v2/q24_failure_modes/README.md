# Q24: R Failure Modes Are Characterizable

## Hypothesis

When the R-gate closes (R falls below threshold), the failure modes are characterizable and the optimal response strategy is determinable. Specifically:

1. Low-R periods can be classified by closure reason (high sigma, low E, both, insufficient data).
2. Different closure reasons require different response strategies (wait, change features, accept uncertainty, escalate).
3. The optimal response strategy for each failure mode can be empirically determined and the results generalize across domains.

The original v1 hypothesis also tested: "Waiting improves subsequent R by >20%" -- which was decisively falsified.

## v1 Evidence Summary

Tested on 3 years of SPY market data (751 data points, 17 low-R periods where R < 0.8):

| Strategy | R Improvement | Success Rate | Best For |
|----------|---------------|--------------|----------|
| WAIT | -34.07% | 0.00% (0/17) | Nothing (counterproductive) |
| CHANGE_FEATURES | +79.96% | 60.00% | HIGH_SIGMA, LOW_E, BOTH |
| ACCEPT_UNCERTAINTY | 0% | 94.12% (16/17) | Time-sensitive actions |
| ESCALATE | -8.20% | 76.47% | R < 0.3 (dangerously low) |

Key findings:
- Waiting is counterproductive (-34% average, 0% success)
- Changing observation window (10-day -> 50-day) improves R by ~80%
- Accepting uncertainty and proceeding works 94% of the time
- All 17 closure reasons were "BOTH" (high sigma AND low E)
- 82.4% of low-R periods occurred in sideways markets

Pre-registered hypothesis "Waiting improves R by >20%" was decisively falsified.

## v1 Methodology Problems

The Phase 6B verification found sound methodology with significant limitations:

1. **n=17 low-R periods is marginal.** All conclusions rest on 17 events. Wilson confidence interval for ACCEPT (16/17): [73%, 99%]. One more failure drops success from 94% to 88%. Statistical power is limited.

2. **Single domain only (SPY financial data).** All findings are from one financial instrument. The CHANGE_FEATURES strategy (try longer window) is specific to time-series data. Would it work for text, genomics, or multi-agent systems? Unknown.

3. **All closure reasons are "BOTH."** The strategy-by-closure-reason decision tree (recommending different strategies for HIGH_SIGMA vs. LOW_E vs. BOTH) is entirely untested for HIGH_SIGMA-only and LOW_E-only scenarios.

4. **CHANGE_FEATURES improvement may be a smoothing artifact.** Switching from 10-day to 50-day window reduces variance by construction (temporal averaging). The R "improvement" reflects mathematical smoothing, not discovery of genuine signal quality.

5. **ACCEPT_UNCERTAINTY success confounded with market regime.** 82.4% of low-R periods are in sideways markets, which are characteristically calm. The 94% success rate may reflect "sideways markets are safe" rather than "low R is not dangerous."

6. **Failure modes coverage is narrow.** Major categories not covered: gate flapping (rapid R oscillation), systematic R bias, numerical instability, adversarial manipulation, correlated failures across systems.

Verdict recommended downgrade from RESOLVED to PARTIAL, R from 1280 to 900-1000.

## v2 Test Plan

### Phase 1: Multi-Domain Replication

Replicate the failure mode analysis across at least 3 genuinely different domains:

1. **Financial time-series** (SPY replication + additional instruments: QQQ, TLT, GLD)
2. **Text coherence monitoring** -- track R over time for document streams (news articles, Wikipedia edit streams)
3. **Multi-agent system outputs** -- monitor R across agent outputs in a multi-turn dialogue

For each domain:
- Identify all low-R periods (using domain-calibrated threshold from Q22)
- Classify closure reasons (HIGH_SIGMA, LOW_E, BOTH, INSUFFICIENT_DATA)
- Test all 4 strategies (WAIT, CHANGE_FEATURES, ACCEPT_UNCERTAINTY, ESCALATE)
- Report success rates with confidence intervals

### Phase 2: Strategy Effectiveness Analysis

For each domain and strategy:
1. Compute effect sizes (R improvement or outcome quality)
2. Compute 95% Wilson confidence intervals on success rates
3. Require n >= 30 low-R events per domain for meaningful conclusions
4. Compare strategies against "always proceed" and "always block" baselines

### Phase 3: Closure Reason Taxonomy Validation

1. Engineer scenarios that produce each closure reason in isolation:
   - HIGH_SIGMA only (diverse but meaningful inputs)
   - LOW_E only (consistent but low-similarity inputs)
   - BOTH (chaotic inputs)
   - INSUFFICIENT_DATA (very few observations)
2. Test whether the recommended strategy differs by closure reason
3. Validate the decision tree empirically (not just on SPY)

### Phase 4: Additional Failure Modes

Test failure modes not covered in v1:
1. **Gate flapping:** What happens when R oscillates rapidly around the threshold?
2. **Systematic bias:** What if R is consistently high for bad content (false positive)?
3. **Adversarial manipulation:** Can an adversary manipulate R to keep the gate open?
4. **Numerical edge cases:** What happens as sigma -> 0 or E -> 0?
5. **Correlated failures:** When multiple R-gated systems fail simultaneously

## Required Data

- **SPY, QQQ, TLT, GLD** (via yfinance) -- financial time-series, 5+ years each
- **Wikipedia Recent Changes API** -- real-time edit stream for text coherence monitoring
- **RealNews / CC-News** -- news article streams for topical coherence
- **Multi-turn dialogue datasets** (ShareGPT, WildChat) -- for multi-agent monitoring simulation
- **Synthetic adversarial data** -- for adversarial manipulation testing (Phase 4.3)

## Pre-Registered Criteria

For multi-domain replication (Phase 1-2):
- **Success (confirm):** The WAIT-is-counterproductive finding replicates (WAIT success < 20%) on at least 2 of 3 domains, AND at least one alternative strategy (CHANGE_FEATURES or ACCEPT_UNCERTAINTY) shows success > 50% on 2 of 3 domains
- **Failure (falsify):** WAIT shows success > 50% on any domain (overturning v1 finding), OR no alternative strategy achieves success > 50% on any domain (strategies are domain-specific with no general pattern)
- **Inconclusive:** WAIT is counterproductive on 1 domain but not others, or strategy effectiveness is highly domain-dependent with no clear pattern

For closure reason taxonomy (Phase 3):
- **Success:** Different closure reasons respond differently to strategies (interaction effect p < 0.05 in 2-way ANOVA: closure_reason x strategy)
- **Failure:** No interaction effect -- all closure reasons respond identically to all strategies
- **Inconclusive:** Interaction effect is marginal (0.05 < p < 0.10) or sample sizes are insufficient

## Baseline Comparisons

1. **Always proceed** (ignore low R, take action anyway)
2. **Always block** (refuse all actions during low-R periods)
3. **Random strategy selection** (pick strategy uniformly at random)
4. **Domain-specific heuristic** (e.g., for finance: wait for volatility to drop; for text: request more context)
5. **Bare E monitoring** (use E alone instead of R for gate decisions)

## Salvageable from v1

- **The core SPY analysis** is sound and reproducible. The finding that WAIT is counterproductive while CHANGE_FEATURES helps is genuine for financial time-series. This serves as the replication target for v2.
- **The decision tree structure** (R < 0.3 -> ESCALATE; time-sensitive -> ACCEPT; otherwise -> CHANGE_FEATURES) is a reasonable starting framework even if it needs multi-domain validation.
- **Test code** at `v1/questions/lower_q24_1280/tests/` and **results** at `v1/questions/lower_q24_1280/results/q24_test_results.json` -- directly reusable for SPY replication
- **Audit reports** at `v1/questions/lower_q24_1280/reports/` (DEEP_AUDIT_Q24, VERIFY_Q24) confirm reproducibility
- **The honest falsification of the WAIT hypothesis** is a clean negative result worth preserving
