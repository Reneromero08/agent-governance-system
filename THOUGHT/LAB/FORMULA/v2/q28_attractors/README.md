# Q28: R Has Attractor Structure

## Hypothesis

In dynamic systems, R converges to fixed points. There exist R-stable states: during regime persistence, R converges to regime-specific equilibrium values and fluctuates around them as a noisy fixed-point attractor. R dynamics are convergent (not chaotic), with Lyapunov exponents at or below zero. Different market regimes (bull, bear, crisis, recovery) have distinct equilibrium R values, and R can be used to detect transitions between these regimes.

## v1 Evidence Summary

R was computed from real market data (SPY via yfinance) across 7 regime periods (bull_2017, volatility_2018, bull_2019, crisis_2020q1, recovery_2020q2, bull_2021, bear_2022):

- Overall pass rate: 82.1% (across 4 test types and 7 regimes).
- **Regime Stability**: 100% (7/7). CV < 1.0 and autocorrelation > 0.5 for all regimes.
- **Relaxation Time**: 100% (7/7). Tau_relax was finite or regime was stable.
- **Attractor Basin**: 57.1% (4/7). Three bull market regimes classified as "unclear."
- **Lyapunov Exponent**: 71.4% (5/7). Mean Lyapunov: 0.0357, max: 0.0447.
- Regime R-means: bull markets R ~ 0.20-0.25, bear/crisis R ~ 0.16-0.19, volatile R ~ 0.16.
- Autocorrelation: > 0.70 across all regimes.
- Dominant attractor type classified as "noisy fixed point" (5 of 12 classifications).

## v1 Methodology Problems

The verification identified the following issues with the v1 tests:

1. **Threshold manipulation confirmed (CRITICAL).** Independent audits (DEEP_AUDIT_Q28.md and VERIFY_Q28.md) confirmed that thresholds were changed between runs. First run with pre-registered thresholds (CV < 0.5, autocorrelation > 0.3) produced 42.8% pass rate and FAILED hypothesis. Thresholds were relaxed to CV < 1.0 and autocorrelation > 0.5 (paradoxically raising the autocorrelation threshold while relaxing CV), flipping the verdict to 82.1% PASS. This is a serious integrity violation.

2. **Lyapunov threshold is non-standard (HIGH).** Standard chaos detection uses lambda > 0 as the boundary. The measured values (0.018-0.045) are all positive, meaning technically the system IS weakly chaotic or at least not convergent in the strict dynamical systems sense. The threshold was set at 0.05, which accommodates positive Lyapunov exponents that would conventionally indicate mild chaos.

3. **Insufficient data for two regimes (HIGH).** crisis_2020q1 and recovery_2020q2 have only 60-61 observations each. Lyapunov estimation requires longer time series for reliability. The claim "all Lyapunov < 0.05" is only verified on 5/7 regimes with adequate data.

4. **"Noisy fixed point" is a catch-all (MEDIUM).** The classify_attractor() function assigns "noisy_fixed_point" as a default when data does not match any specific pattern. A more accurate description would be "mean-reverting stochastic process," which is a weaker claim than having attractor structure.

5. **Relaxation time passes by absence (MEDIUM).** When "no perturbations found" in a regime, the relaxation time test passes by absence of evidence rather than evidence of relaxation. Multiple tau_relax estimates hit the upper bound (999.99), indicating failed fits counted as passes.

6. **Attractor basin test has majority failure (MEDIUM).** Only 57.1% (4/7) regimes show clear attractor basins. Three bull market regimes are classified as "unclear." The claim of attractor structure rests on a test that fails for nearly half the regimes.

## v2 Test Plan

### Test 1: Pre-Registered Attractor Analysis on Market Data

Redo the v1 analysis with properly pre-registered thresholds:

1. Compute R on SPY daily returns for at least 10 distinct regime periods (2010-2025), each with at least 120 trading days.
2. Pre-register all thresholds before any computation:
   - Stability: CV < 0.5 (standard for moderate stability).
   - Autocorrelation: lag-1 autocorrelation > 0.3 (standard for persistence).
   - Lyapunov: lambda < 0 (standard for convergence; lambda = 0 for neutral; lambda > 0 for divergence).
3. No threshold changes after data is examined.
4. Report all results including failures. Do not relabel failures as passes.

### Test 2: Ornstein-Uhlenbeck Parameter Estimation

Fit the Ornstein-Uhlenbeck (OU) process to R trajectories within each regime:

1. dR = theta * (mu - R) * dt + sigma_OU * dW
2. Estimate theta (mean-reversion rate), mu (equilibrium), and sigma_OU (volatility) using maximum likelihood.
3. Test the OU fit vs. alternatives: random walk (theta=0), trending process (theta < 0), and regime-switching model.
4. Report AIC/BIC for model comparison.
5. The OU model must significantly outperform the random walk for the attractor claim to hold.

### Test 3: Proper Lyapunov Exponent Estimation

Use established methods for Lyapunov exponent estimation:

1. Apply the Rosenstein (1993) algorithm or Kantz (1994) algorithm to R time series.
2. Require minimum embedding dimension estimation via false nearest neighbors.
3. Use surrogate data testing: generate 100 IAAFT surrogates (phase-randomized, amplitude-preserved) and compute Lyapunov exponents for each.
4. The real data's Lyapunov exponent must be significantly lower (more negative) than the surrogates' for the non-chaos claim to hold (p < 0.05).
5. Require at least 500 data points per regime for reliable estimation.

### Test 4: Regime Detection and Transition Identification

Test whether R actually detects regime changes:

1. Apply a hidden Markov model (HMM) with 2-5 states to the R time series.
2. Compare HMM state assignments to ground-truth regime labels (NBER recession dates, VIX spikes, or manually labeled periods).
3. Measure detection accuracy: what fraction of true regime transitions are detected within +/- 5 trading days?
4. Compare R-based regime detection to detection based on raw returns, volatility, or VIX directly.

### Test 5: Cross-Asset Generalization

1. Compute R on at least 5 different asset classes: equities (SPY), bonds (TLT), commodities (GLD), forex (EUR/USD), crypto (BTC).
2. For each asset, identify regimes and test for attractor structure using the same pre-registered criteria.
3. True universality means R shows attractor behavior across asset classes; if attractor structure is equity-specific, the finding is domain-limited.

### Test 6: Null Model Comparison

1. Generate synthetic price series from known processes:
   - Random walk (no attractor): GBM with constant drift.
   - OU process (known attractor): mean-reverting with known parameters.
   - Regime-switching model: 2-state HMM with known transition probabilities.
2. Compute R on each synthetic series.
3. Verify that the attractor detection methodology correctly identifies: no attractor in GBM, attractor in OU, transitions in regime-switching.
4. Calibrate false positive and false negative rates.

## Required Data

- **Yahoo Finance (yfinance):** SPY, TLT, GLD, EUR/USD daily OHLCV data, 2010-2025
- **CoinGecko or similar:** BTC daily price data
- **NBER recession dates:** Ground truth for US economic regime transitions
- **VIX daily close:** Volatility benchmark for regime labeling
- **FRED:** Federal funds rate, yield curve data for macroeconomic regime context

## Pre-Registered Criteria

- **Success (confirm):** OU model significantly outperforms random walk for R trajectories in at least 7/10 regime periods (likelihood ratio test p < 0.01), AND Lyapunov exponents are significantly more negative than IAAFT surrogates (p < 0.05) in at least 7/10 regimes, AND R-based regime detection achieves at least 60% accuracy within +/- 5 days of true transitions, AND attractor structure is present in at least 3/5 asset classes.
- **Failure (falsify):** OU model does not outperform random walk in more than 3/10 regimes, OR Lyapunov exponents are not distinguishable from surrogates, OR R-based regime detection is not better than chance (< 33% accuracy), OR attractor structure is absent in 4/5 asset classes.
- **Inconclusive:** OU outperforms random walk in 4-6/10 regimes, or Lyapunov results are mixed, or regime detection accuracy is 33-60%.

## Baseline Comparisons

- **Random walk model:** R from a geometric Brownian motion (no attractor by construction).
- **Raw return autocorrelation:** Compare R's persistence properties to simple return autocorrelation.
- **Realized volatility:** Compare R's regime-detection ability to 20-day realized volatility.
- **VIX-based regime detection:** Compare to the standard approach of using VIX thresholds for regime classification.
- **Hurst exponent:** Compare R's long-memory properties to the standard Hurst exponent of the return series.

## Salvageable from v1

- **Real data pipeline:** `v1/questions/lower_q28_1200/test_q28_attractors.py` uses yfinance to fetch real SPY data and compute R across regimes -- this data pipeline is directly reusable and is one of the most honest data sources in the v1 corpus.
- **Regime period definitions:** The 7 regime periods (bull_2017, volatility_2018, bull_2019, crisis_2020q1, recovery_2020q2, bull_2021, bear_2022) provide a reasonable starting taxonomy that can be extended.
- **Results JSON:** `v1/questions/lower_q28_1200/results/q28_attractors_20260127_205852.json` contains raw results including the honest 57.1% attractor basin pass rate.
- **Internal audit methodology:** The DEEP_AUDIT_Q28.md and VERIFY_Q28.md documents demonstrate the self-correcting audit process that caught the threshold manipulation -- this audit methodology should be standard practice.
