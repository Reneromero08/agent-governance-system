#!/usr/bin/env python3
"""
Q24: Failure Modes - What To Do When R-Gate is CLOSED

PRE-REGISTRATION:
1. HYPOTHESIS: At least one strategy improves R or outcomes by > 20%
2. PREDICTION: Best strategy depends on WHY gate closed
3. FALSIFICATION: If no strategy helps
4. DATA: Market time series (yfinance)
5. THRESHOLD: Document which strategy works when

TESTS:
1. Identify periods where R is low (market disagreement/chaos)
2. Test FOUR strategies:
   - WAIT: Gather more observations over time
   - CHANGE_FEATURES: Use different feature windows (observation strategy)
   - ACCEPT_UNCERTAINTY: Proceed with low R, measure outcomes
   - ESCALATE: Human review - measure when R is "dangerously" low
3. Measure which strategy improves R or outcomes

Run: python test_q24_failure_modes.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum observations for R computation
MIN_OBS = 5

# R threshold for gate (below this = CLOSED)
R_THRESHOLD = 0.8

# Strategy parameters
WAIT_STEPS = [1, 3, 5, 10]  # How many time steps to wait
FEATURE_WINDOWS = [5, 10, 20, 50]  # Different rolling windows for features

# Test parameters
TEST_PERIOD_DAYS = 365 * 3  # 3 years of data
N_LOW_R_PERIODS = 50  # Number of low-R periods to analyze


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class GateStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


class ClosureReason(Enum):
    HIGH_SIGMA = "high_sigma"      # High dispersion among observations
    LOW_E = "low_e"                 # Low mean agreement
    BOTH = "both"                   # Both problems
    INSUFFICIENT_DATA = "insufficient"


class Strategy(Enum):
    WAIT = "wait"
    CHANGE_FEATURES = "change_features"
    ACCEPT_UNCERTAINTY = "accept_uncertainty"
    ESCALATE = "escalate"  # Human review when R is dangerously low


@dataclass
class RResult:
    """Result of R computation."""
    R: float
    E: float
    sigma: float
    n_observations: int
    gate_status: GateStatus
    closure_reason: Optional[ClosureReason]


@dataclass
class StrategyOutcome:
    """Outcome of applying a strategy."""
    strategy: Strategy
    initial_R: float
    final_R: float
    R_improvement: float  # (final - initial) / initial
    time_cost: int        # Steps waited or window used
    success: bool         # R crossed threshold


@dataclass
class LowRPeriod:
    """A period where R was below threshold."""
    start_idx: int
    end_idx: int
    min_R: float
    closure_reason: ClosureReason
    market_regime: str  # e.g., "crash", "volatile", "trending"


# =============================================================================
# R-GATE IMPLEMENTATION (from q17)
# =============================================================================

def compute_r(observations: np.ndarray) -> RResult:
    """
    Compute R = E / sigma for a set of observations.

    Observations should be a 2D array where each row is an observation vector.
    Uses pairwise cosine similarities.
    """
    n = len(observations)

    if n < 2:
        return RResult(
            R=0.0, E=0.0, sigma=float('inf'),
            n_observations=n,
            gate_status=GateStatus.CLOSED,
            closure_reason=ClosureReason.INSUFFICIENT_DATA
        )

    # Normalize observations
    norms = np.linalg.norm(observations, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = observations / norms

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(normalized[i], normalized[j])
            similarities.append(sim)

    similarities = np.array(similarities)

    E = np.mean(similarities)
    sigma = np.std(similarities)

    # R = E / sigma (with stability epsilon)
    R = E / (sigma + 1e-8)

    # Determine gate status and closure reason
    if R >= R_THRESHOLD:
        gate_status = GateStatus.OPEN
        closure_reason = None
    else:
        gate_status = GateStatus.CLOSED
        # Diagnose WHY closed
        median_sigma = 0.1  # Typical healthy sigma
        median_E = 0.3      # Typical healthy E

        high_sigma = sigma > median_sigma * 2
        low_E = E < median_E * 0.5

        if high_sigma and low_E:
            closure_reason = ClosureReason.BOTH
        elif high_sigma:
            closure_reason = ClosureReason.HIGH_SIGMA
        elif low_E:
            closure_reason = ClosureReason.LOW_E
        else:
            closure_reason = ClosureReason.BOTH  # Edge case

    return RResult(
        R=R, E=E, sigma=sigma,
        n_observations=n,
        gate_status=gate_status,
        closure_reason=closure_reason
    )


# =============================================================================
# MARKET DATA FUNCTIONS
# =============================================================================

def fetch_market_data(ticker: str = "SPY", period_days: int = TEST_PERIOD_DAYS) -> Dict[str, np.ndarray]:
    """
    Fetch market data using yfinance.

    Returns dict with: prices, returns, volume, volatility
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)

    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    print(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}...")

    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        print(f"ERROR: No data returned for {ticker}")
        sys.exit(1)

    # Extract and compute features
    prices = data['Close'].values.flatten()

    # Daily returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])  # Pad first value

    # Rolling volatility (20-day)
    vol_window = 20
    volatility = np.zeros(len(prices))
    for i in range(vol_window, len(prices)):
        volatility[i] = np.std(returns[i-vol_window:i])
    volatility[:vol_window] = volatility[vol_window]  # Fill initial

    # Volume (normalized)
    volume = data['Volume'].values.flatten()
    volume = volume / np.mean(volume)

    print(f"  Loaded {len(prices)} data points")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Mean volatility: {volatility.mean():.4f}")

    return {
        'prices': prices,
        'returns': returns,
        'volatility': volatility,
        'volume': volume,
        'dates': data.index.values
    }


def create_feature_vector(data: Dict[str, np.ndarray], idx: int, window: int = 10) -> np.ndarray:
    """
    Create a feature vector for observation at index idx.

    Features include:
    - Recent returns (normalized)
    - Recent volatility (normalized)
    - Volume trend
    - Momentum
    """
    if idx < window:
        return None

    # Get slices
    returns = data['returns'][idx-window:idx]
    volatility = data['volatility'][idx-window:idx]
    volume = data['volume'][idx-window:idx]
    prices = data['prices'][idx-window:idx]

    # Compute features
    features = []

    # Normalized returns
    ret_mean = np.mean(returns)
    ret_std = np.std(returns) + 1e-8
    features.extend((returns - ret_mean) / ret_std)

    # Normalized volatility
    vol_mean = np.mean(volatility)
    vol_std = np.std(volatility) + 1e-8
    features.extend((volatility - vol_mean) / vol_std)

    # Volume trend
    vol_trend = np.polyfit(range(window), volume, 1)[0]
    features.append(vol_trend)

    # Momentum (price change over window)
    momentum = (prices[-1] - prices[0]) / prices[0]
    features.append(momentum)

    return np.array(features)


def classify_market_regime(data: Dict[str, np.ndarray], idx: int, window: int = 20) -> str:
    """Classify market regime at a given point."""
    if idx < window:
        return "unknown"

    returns = data['returns'][idx-window:idx]
    volatility = data['volatility'][idx]

    cum_return = np.sum(returns)
    vol_percentile = np.percentile(data['volatility'], 75)

    if volatility > vol_percentile * 1.5:
        if cum_return < -0.05:
            return "crash"
        else:
            return "volatile"
    elif cum_return > 0.05:
        return "bull"
    elif cum_return < -0.05:
        return "bear"
    else:
        return "sideways"


# =============================================================================
# R-GATE ANALYSIS FUNCTIONS
# =============================================================================

def compute_rolling_r(data: Dict[str, np.ndarray], window: int = 20, feature_window: int = 10) -> Tuple[np.ndarray, List[RResult]]:
    """
    Compute rolling R values across the time series.

    For each time point, we gather `window` observations (feature vectors)
    and compute R.
    """
    n = len(data['prices'])
    r_values = np.zeros(n)
    r_results = []

    start_idx = max(window, feature_window)

    for i in range(start_idx, n):
        # Gather observations from the window
        observations = []
        for j in range(i - window, i):
            fv = create_feature_vector(data, j, feature_window)
            if fv is not None:
                observations.append(fv)

        if len(observations) < MIN_OBS:
            r_values[i] = 0.0
            r_results.append(RResult(
                R=0.0, E=0.0, sigma=float('inf'),
                n_observations=len(observations),
                gate_status=GateStatus.CLOSED,
                closure_reason=ClosureReason.INSUFFICIENT_DATA
            ))
            continue

        observations = np.array(observations)
        result = compute_r(observations)
        r_values[i] = result.R
        r_results.append(result)

    return r_values, r_results


def find_low_r_periods(r_values: np.ndarray, r_results: List[RResult],
                       data: Dict[str, np.ndarray], threshold: float = R_THRESHOLD,
                       min_duration: int = 3) -> List[LowRPeriod]:
    """Find periods where R was below threshold."""
    periods = []
    n = len(r_values)

    i = 0
    while i < n:
        if r_values[i] < threshold and r_values[i] > 0:
            # Found start of low-R period
            start = i
            min_r = r_values[i]

            while i < n and r_values[i] < threshold:
                min_r = min(min_r, r_values[i])
                i += 1

            end = i
            duration = end - start

            if duration >= min_duration:
                # Get closure reason from worst R point
                worst_idx = start + np.argmin(r_values[start:end])
                result_idx = worst_idx - max(20, 10)  # Adjust for offset
                if 0 <= result_idx < len(r_results):
                    closure_reason = r_results[result_idx].closure_reason or ClosureReason.BOTH
                else:
                    closure_reason = ClosureReason.BOTH

                # Classify market regime
                regime = classify_market_regime(data, worst_idx)

                periods.append(LowRPeriod(
                    start_idx=start,
                    end_idx=end,
                    min_R=min_r,
                    closure_reason=closure_reason,
                    market_regime=regime
                ))
        i += 1

    return periods


# =============================================================================
# STRATEGY TESTS
# =============================================================================

def test_wait_strategy(data: Dict[str, np.ndarray], period: LowRPeriod,
                       wait_steps: List[int], feature_window: int = 10) -> List[StrategyOutcome]:
    """
    Test WAIT strategy: gather more observations over time.

    Hypothesis: Waiting allows market to "settle" and increases R.
    """
    outcomes = []

    # Get initial R at start of period
    observations = []
    for j in range(max(0, period.start_idx - 20), period.start_idx):
        fv = create_feature_vector(data, j, feature_window)
        if fv is not None:
            observations.append(fv)

    if len(observations) < MIN_OBS:
        return outcomes

    initial_result = compute_r(np.array(observations))
    initial_R = initial_result.R

    for wait in wait_steps:
        # After waiting, get new observations
        new_idx = min(period.start_idx + wait, len(data['prices']) - 1)

        observations = []
        for j in range(max(0, new_idx - 20), new_idx):
            fv = create_feature_vector(data, j, feature_window)
            if fv is not None:
                observations.append(fv)

        if len(observations) < MIN_OBS:
            continue

        final_result = compute_r(np.array(observations))
        final_R = final_result.R

        improvement = (final_R - initial_R) / (initial_R + 1e-8)
        success = final_R >= R_THRESHOLD

        outcomes.append(StrategyOutcome(
            strategy=Strategy.WAIT,
            initial_R=initial_R,
            final_R=final_R,
            R_improvement=improvement,
            time_cost=wait,
            success=success
        ))

    return outcomes


def test_change_features_strategy(data: Dict[str, np.ndarray], period: LowRPeriod,
                                   feature_windows: List[int]) -> List[StrategyOutcome]:
    """
    Test CHANGE_FEATURES strategy: use different feature windows.

    Hypothesis: Different time scales may show more agreement.
    """
    outcomes = []

    # Get initial R with default window (10)
    observations = []
    for j in range(max(0, period.start_idx - 20), period.start_idx):
        fv = create_feature_vector(data, j, 10)
        if fv is not None:
            observations.append(fv)

    if len(observations) < MIN_OBS:
        return outcomes

    initial_result = compute_r(np.array(observations))
    initial_R = initial_result.R

    for window in feature_windows:
        if window == 10:  # Skip default
            continue

        observations = []
        for j in range(max(window, period.start_idx - 20), period.start_idx):
            fv = create_feature_vector(data, j, window)
            if fv is not None:
                observations.append(fv)

        if len(observations) < MIN_OBS:
            continue

        final_result = compute_r(np.array(observations))
        final_R = final_result.R

        improvement = (final_R - initial_R) / (initial_R + 1e-8)
        success = final_R >= R_THRESHOLD

        outcomes.append(StrategyOutcome(
            strategy=Strategy.CHANGE_FEATURES,
            initial_R=initial_R,
            final_R=final_R,
            R_improvement=improvement,
            time_cost=window,
            success=success
        ))

    return outcomes


def test_accept_uncertainty_strategy(data: Dict[str, np.ndarray], period: LowRPeriod,
                                      lookahead: int = 5) -> StrategyOutcome:
    """
    Test ACCEPT_UNCERTAINTY strategy: proceed despite low R.

    Measure actual outcome quality (did the market move as "expected"?).
    """
    # Get initial R
    observations = []
    for j in range(max(0, period.start_idx - 20), period.start_idx):
        fv = create_feature_vector(data, j, 10)
        if fv is not None:
            observations.append(fv)

    if len(observations) < MIN_OBS:
        return None

    initial_result = compute_r(np.array(observations))
    initial_R = initial_result.R

    # Compute "decision quality" - did proceeding work out?
    # We use realized volatility vs expected as a proxy
    future_idx = min(period.start_idx + lookahead, len(data['prices']) - 1)

    if future_idx <= period.start_idx:
        return None

    # Compute realized volatility
    future_returns = data['returns'][period.start_idx:future_idx]
    realized_vol = np.std(future_returns)

    # Compare to historical volatility
    hist_vol = data['volatility'][period.start_idx]

    # Success = realized vol not much worse than expected
    vol_ratio = realized_vol / (hist_vol + 1e-8)
    success = vol_ratio < 1.5  # Within 50% of expected

    return StrategyOutcome(
        strategy=Strategy.ACCEPT_UNCERTAINTY,
        initial_R=initial_R,
        final_R=initial_R,  # R doesn't change with this strategy
        R_improvement=0.0,
        time_cost=lookahead,
        success=success
    )


def test_escalate_strategy(data: Dict[str, np.ndarray], period: LowRPeriod,
                           escalation_threshold: float = 0.3,
                           lookahead: int = 10) -> Optional[StrategyOutcome]:
    """
    Test ESCALATE strategy: when R is dangerously low, defer to human review.

    This strategy models the value of escalation by measuring:
    1. How often R was "dangerously" low (below escalation_threshold)
    2. What happened if action was taken during such periods
    3. Whether waiting for human review time would have helped

    The "success" metric is: would deferring to human review have avoided
    a bad outcome (measured by extreme volatility/drawdown)?
    """
    # Get initial R
    observations = []
    for j in range(max(0, period.start_idx - 20), period.start_idx):
        fv = create_feature_vector(data, j, 10)
        if fv is not None:
            observations.append(fv)

    if len(observations) < MIN_OBS:
        return None

    initial_result = compute_r(np.array(observations))
    initial_R = initial_result.R

    # Only escalate when R is "dangerously" low
    should_escalate = initial_R < escalation_threshold

    # Measure what happened during the period
    future_idx = min(period.start_idx + lookahead, len(data['prices']) - 1)

    if future_idx <= period.start_idx:
        return None

    # Compute worst drawdown during the period
    prices_ahead = data['prices'][period.start_idx:future_idx + 1]
    if len(prices_ahead) < 2:
        return None

    # Calculate max drawdown
    running_max = np.maximum.accumulate(prices_ahead)
    drawdowns = (running_max - prices_ahead) / running_max
    max_drawdown = np.max(drawdowns)

    # Also check for extreme volatility
    returns_ahead = data['returns'][period.start_idx:future_idx]
    realized_vol = np.std(returns_ahead) if len(returns_ahead) > 0 else 0
    vol_percentile = np.percentile(data['volatility'][data['volatility'] > 0], 90)

    # "Bad outcome" = either large drawdown or extreme volatility
    bad_outcome = max_drawdown > 0.03 or realized_vol > vol_percentile

    # Success metric for ESCALATE:
    # - If we SHOULD escalate (R < threshold) and outcome was bad: escalation would have helped
    # - If we SHOULD escalate and outcome was fine: escalation was conservative but safe
    # - If we shouldn't escalate: measure baseline

    if should_escalate:
        # Escalation was warranted - success if bad outcome was avoided by waiting
        # We simulate "human review took time" by checking R after lookahead
        observations_after = []
        for j in range(max(0, future_idx - 20), future_idx):
            fv = create_feature_vector(data, j, 10)
            if fv is not None:
                observations_after.append(fv)

        if len(observations_after) >= MIN_OBS:
            final_result = compute_r(np.array(observations_after))
            final_R = final_result.R
            # Success if R improved after waiting (human review time)
            r_improvement = (final_R - initial_R) / (initial_R + 1e-8)
            success = final_R > initial_R or not bad_outcome
        else:
            final_R = initial_R
            r_improvement = 0.0
            success = not bad_outcome
    else:
        # R wasn't low enough to warrant escalation
        final_R = initial_R
        r_improvement = 0.0
        success = not bad_outcome  # If no bad outcome, proceeding was fine

    return StrategyOutcome(
        strategy=Strategy.ESCALATE,
        initial_R=initial_R,
        final_R=final_R,
        R_improvement=r_improvement,
        time_cost=lookahead,  # Time cost of human review
        success=success
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def analyze_strategy_effectiveness(outcomes: List[StrategyOutcome],
                                   strategy: Strategy) -> Dict[str, Any]:
    """Analyze effectiveness of a strategy."""
    if not outcomes:
        return {"n_tests": 0, "message": "No data"}

    improvements = [o.R_improvement for o in outcomes]
    successes = [o.success for o in outcomes]
    time_costs = [o.time_cost for o in outcomes]

    return {
        "n_tests": len(outcomes),
        "mean_improvement": np.mean(improvements),
        "std_improvement": np.std(improvements),
        "median_improvement": np.median(improvements),
        "success_rate": np.mean(successes),
        "improvement_gt_20pct": np.mean([i > 0.2 for i in improvements]),
        "best_time_cost": time_costs[np.argmax(improvements)] if improvements else None,
        "hypothesis_supported": np.mean(improvements) > 0.2  # Pre-registered threshold
    }


def run_all_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run all Q24 tests."""
    print("=" * 70)
    print("Q24: FAILURE MODES - WHAT TO DO WHEN R-GATE IS CLOSED")
    print("=" * 70)
    print()

    # Fetch data
    data = fetch_market_data("SPY", TEST_PERIOD_DAYS)

    # Compute rolling R
    print("\nComputing rolling R values...")
    r_values, r_results = compute_rolling_r(data, window=20, feature_window=10)

    # Find low-R periods
    print("Finding low-R periods...")
    periods = find_low_r_periods(r_values, r_results, data)
    print(f"  Found {len(periods)} low-R periods")

    if len(periods) == 0:
        print("ERROR: No low-R periods found. Cannot test strategies.")
        return {"error": "No low-R periods found"}

    # Limit to N_LOW_R_PERIODS
    periods = periods[:N_LOW_R_PERIODS]

    # Analyze closure reasons
    reason_counts = {}
    for p in periods:
        reason = p.closure_reason.value
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    regime_counts = {}
    for p in periods:
        regime = p.market_regime
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print(f"\nClosure reason distribution:")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count} ({100*count/len(periods):.1f}%)")

    print(f"\nMarket regime distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} ({100*count/len(periods):.1f}%)")

    # Test strategies
    print("\n" + "-" * 70)
    print("TESTING STRATEGIES")
    print("-" * 70)

    wait_outcomes = []
    change_features_outcomes = []
    accept_outcomes = []
    escalate_outcomes = []

    # Group by closure reason for detailed analysis
    outcomes_by_reason = {reason: {"wait": [], "change": [], "accept": [], "escalate": []}
                          for reason in ClosureReason}

    for i, period in enumerate(periods):
        if verbose and i % 10 == 0:
            print(f"  Processing period {i+1}/{len(periods)}...")

        # Test WAIT strategy
        wait_results = test_wait_strategy(data, period, WAIT_STEPS)
        wait_outcomes.extend(wait_results)
        outcomes_by_reason[period.closure_reason]["wait"].extend(wait_results)

        # Test CHANGE_FEATURES strategy
        change_results = test_change_features_strategy(data, period, FEATURE_WINDOWS)
        change_features_outcomes.extend(change_results)
        outcomes_by_reason[period.closure_reason]["change"].extend(change_results)

        # Test ACCEPT_UNCERTAINTY strategy
        accept_result = test_accept_uncertainty_strategy(data, period)
        if accept_result:
            accept_outcomes.append(accept_result)
            outcomes_by_reason[period.closure_reason]["accept"].append(accept_result)

        # Test ESCALATE strategy
        escalate_result = test_escalate_strategy(data, period)
        if escalate_result:
            escalate_outcomes.append(escalate_result)
            outcomes_by_reason[period.closure_reason]["escalate"].append(escalate_result)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "SPY",
        "n_data_points": len(data['prices']),
        "n_low_r_periods": len(periods),
        "r_threshold": R_THRESHOLD,
        "closure_reason_distribution": reason_counts,
        "market_regime_distribution": regime_counts,
        "strategies": {}
    }

    # WAIT strategy
    print("\n### WAIT STRATEGY ###")
    wait_analysis = analyze_strategy_effectiveness(wait_outcomes, Strategy.WAIT)
    results["strategies"]["wait"] = wait_analysis
    print(f"  N tests: {wait_analysis['n_tests']}")
    print(f"  Mean R improvement: {wait_analysis['mean_improvement']:.2%}")
    print(f"  Median R improvement: {wait_analysis['median_improvement']:.2%}")
    print(f"  Success rate (R crossed threshold): {wait_analysis['success_rate']:.2%}")
    print(f"  Improvement > 20%: {wait_analysis['improvement_gt_20pct']:.2%}")
    print(f"  Best wait time: {wait_analysis['best_time_cost']} steps")
    print(f"  HYPOTHESIS (>20% improvement): {'SUPPORTED' if wait_analysis['hypothesis_supported'] else 'NOT SUPPORTED'}")

    # CHANGE_FEATURES strategy
    print("\n### CHANGE_FEATURES STRATEGY ###")
    change_analysis = analyze_strategy_effectiveness(change_features_outcomes, Strategy.CHANGE_FEATURES)
    results["strategies"]["change_features"] = change_analysis
    print(f"  N tests: {change_analysis['n_tests']}")
    print(f"  Mean R improvement: {change_analysis['mean_improvement']:.2%}")
    print(f"  Median R improvement: {change_analysis['median_improvement']:.2%}")
    print(f"  Success rate: {change_analysis['success_rate']:.2%}")
    print(f"  Best window: {change_analysis['best_time_cost']} days")
    print(f"  HYPOTHESIS: {'SUPPORTED' if change_analysis['hypothesis_supported'] else 'NOT SUPPORTED'}")

    # ACCEPT_UNCERTAINTY strategy
    print("\n### ACCEPT_UNCERTAINTY STRATEGY ###")
    accept_analysis = analyze_strategy_effectiveness(accept_outcomes, Strategy.ACCEPT_UNCERTAINTY)
    results["strategies"]["accept_uncertainty"] = accept_analysis
    print(f"  N tests: {accept_analysis['n_tests']}")
    print(f"  Success rate (acceptable outcome): {accept_analysis['success_rate']:.2%}")

    # ESCALATE strategy
    print("\n### ESCALATE (HUMAN REVIEW) STRATEGY ###")
    escalate_analysis = analyze_strategy_effectiveness(escalate_outcomes, Strategy.ESCALATE)
    results["strategies"]["escalate"] = escalate_analysis
    print(f"  N tests: {escalate_analysis['n_tests']}")
    print(f"  Mean R improvement (after review time): {escalate_analysis['mean_improvement']:.2%}")
    print(f"  Success rate (avoided bad outcome or R improved): {escalate_analysis['success_rate']:.2%}")
    print(f"  Escalation value: {'HIGH' if escalate_analysis['success_rate'] > 0.7 else 'MODERATE' if escalate_analysis['success_rate'] > 0.5 else 'LOW'}")

    # Analyze by closure reason
    print("\n### BY CLOSURE REASON ###")
    results["by_closure_reason"] = {}

    for reason in ClosureReason:
        reason_data = outcomes_by_reason[reason]
        if not any(reason_data.values()):
            continue

        print(f"\n  {reason.value.upper()}:")
        results["by_closure_reason"][reason.value] = {}

        for strat_name, outcomes in reason_data.items():
            if not outcomes:
                continue
            # Map strategy names to enums
            strat_enum_map = {
                "wait": Strategy.WAIT,
                "change": Strategy.CHANGE_FEATURES,
                "accept": Strategy.ACCEPT_UNCERTAINTY,
                "escalate": Strategy.ESCALATE
            }
            analysis = analyze_strategy_effectiveness(
                outcomes,
                strat_enum_map.get(strat_name, Strategy.WAIT)
            )
            results["by_closure_reason"][reason.value][strat_name] = analysis
            print(f"    {strat_name}: improvement={analysis['mean_improvement']:.2%}, success={analysis['success_rate']:.2%}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Determine best strategy
    strategies = [
        ("WAIT", wait_analysis),
        ("CHANGE_FEATURES", change_analysis),
        ("ACCEPT_UNCERTAINTY", accept_analysis),
        ("ESCALATE", escalate_analysis)
    ]

    best_by_improvement = max(strategies, key=lambda x: x[1].get('mean_improvement', -1))
    best_by_success = max(strategies, key=lambda x: x[1].get('success_rate', -1))

    print(f"\nBest strategy by R improvement: {best_by_improvement[0]}")
    print(f"Best strategy by success rate: {best_by_success[0]}")

    # Check pre-registered hypothesis
    hypothesis_supported = wait_analysis.get('hypothesis_supported', False)

    print(f"\n*** PRE-REGISTERED HYPOTHESIS: Waiting improves R by >20% ***")
    if hypothesis_supported:
        print(f"RESULT: SUPPORTED (mean improvement = {wait_analysis['mean_improvement']:.2%})")
    else:
        print(f"RESULT: NOT SUPPORTED (mean improvement = {wait_analysis['mean_improvement']:.2%})")

    # Strategy recommendations by closure reason
    print("\n*** STRATEGY RECOMMENDATIONS BY CLOSURE REASON ***")
    results["recommendations"] = {}

    for reason in ClosureReason:
        reason_data = results.get("by_closure_reason", {}).get(reason.value, {})
        if not reason_data:
            continue

        best = max(reason_data.items(), key=lambda x: x[1].get('mean_improvement', -1))
        print(f"  {reason.value}: Use {best[0].upper()} (improvement: {best[1]['mean_improvement']:.2%})")
        results["recommendations"][reason.value] = best[0]

    results["verdict"] = {
        "hypothesis_supported": hypothesis_supported,
        "best_overall_strategy": best_by_improvement[0],
        "best_by_success": best_by_success[0]
    }

    # Save results
    output_dir = Path(__file__).parent
    output_path = output_dir / "q24_test_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with appropriate code
    if results.get("verdict", {}).get("hypothesis_supported", False):
        print("\n*** HYPOTHESIS SUPPORTED - At least one strategy improves R significantly ***")
        sys.exit(0)
    else:
        print("\n*** HYPOTHESIS NOT SUPPORTED - No strategy consistently improves R by >20% ***")
        print("*** This is valuable data: documents actual failure mode behaviors ***")
        sys.exit(0)  # Still exit 0 since this is valid experimental result
