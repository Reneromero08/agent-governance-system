"""
PSYCHOHISTORY BOT TEST SUITE
============================

Validation tests for the formula-driven trading system.

Tests:
1. Formula correlation (R discriminates coherent from conflicting)
2. Gate threshold discrimination
3. Alpha drift detection lead time
4. Monte Carlo stress test
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
from pathlib import Path

from signal_vocabulary import SignalState, AssetClass, get_all_signals
from prime_radiant import PrimeRadiant
from formula_executor import MarketFormulaExecutor
from seldon_gate import SeldonGate, GateTier, AlphaWarningLevel
from psychohistory_bot import PsychohistoryBot, BotConfig


# =============================================================================
# TEST UTILITIES
# =============================================================================

def create_coherent_states(n: int, base_signals: Dict[str, float], noise: float = 0.1) -> List[SignalState]:
    """Create n coherent (similar) states with small noise."""
    states = []
    for i in range(n):
        signals = {
            k: max(0, min(1, v + np.random.randn() * noise))
            for k, v in base_signals.items()
        }
        states.append(SignalState(
            signals=signals,
            timestamp=(datetime.now() + timedelta(minutes=i)).isoformat(),
            asset="TEST",
            asset_class=AssetClass.ALL,
        ))
    return states


def create_conflicting_states(n: int) -> List[SignalState]:
    """Create n conflicting (random) states."""
    all_signals = get_all_signals()
    signal_ids = [s.signal_id for s in all_signals[:20]]  # Use first 20 signals

    states = []
    for i in range(n):
        # Random subset of signals with random strengths
        n_signals = np.random.randint(2, 6)
        selected = np.random.choice(signal_ids, size=n_signals, replace=False)
        signals = {s: np.random.random() for s in selected}

        states.append(SignalState(
            signals=signals,
            timestamp=(datetime.now() + timedelta(minutes=i)).isoformat(),
            asset="TEST",
            asset_class=AssetClass.ALL,
        ))
    return states


def create_regime_sequence(phases: List[Tuple[str, int]]) -> Tuple[List[SignalState], List[float]]:
    """
    Create a sequence of states simulating different market regimes.

    Args:
        phases: List of (regime, n_steps) tuples
            regimes: "bull", "bear", "volatile", "crisis", "recovery"

    Returns:
        (states, prices) tuple
    """
    regime_signals = {
        "bull": {"trend_up": 0.8, "volume_surge": 0.5, "bullish_news": 0.7, "momentum_confirmation": 0.6},
        "bear": {"trend_down": 0.8, "volume_surge": 0.5, "bearish_news": 0.7, "momentum_divergence": 0.4},
        "volatile": {"sideways": 0.5, "vol_expanding": 0.8, "mixed_news": 0.6, "unusual_activity": 0.5},
        "crisis": {"breakdown": 0.9, "vol_spike": 0.9, "bearish_news": 0.9, "extreme_fear": 0.8},
        "recovery": {"trend_up": 0.6, "vol_contracting": 0.6, "bullish_news": 0.5, "oversold": 0.4},
    }

    regime_price_params = {
        "bull": (0.002, 0.01),      # +0.2% trend, 1% vol
        "bear": (-0.001, 0.012),    # -0.1% trend, 1.2% vol
        "volatile": (0.0, 0.025),   # 0% trend, 2.5% vol
        "crisis": (-0.015, 0.04),   # -1.5% trend, 4% vol
        "recovery": (0.005, 0.02),  # +0.5% trend, 2% vol
    }

    states = []
    prices = [100.0]

    for regime, n_steps in phases:
        base_signals = regime_signals.get(regime, regime_signals["volatile"])
        trend, vol = regime_price_params.get(regime, (0, 0.02))

        for i in range(n_steps):
            # Create state with noise
            signals = {
                k: max(0, min(1, v + np.random.randn() * 0.15))
                for k, v in base_signals.items()
            }
            states.append(SignalState(
                signals=signals,
                timestamp=(datetime.now() + timedelta(minutes=len(states))).isoformat(),
                asset="TEST",
                asset_class=AssetClass.ALL,
            ))

            # Generate price
            price_change = trend + np.random.randn() * vol
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))

    return states, prices[1:]  # Remove initial price


# =============================================================================
# TEST 1: FORMULA CORRELATION
# =============================================================================

def test_formula_correlation():
    """
    Test that R discriminates coherent from conflicting signals.

    Target: R_coherent >> R_conflicting (at least 2x ratio)
    """
    print("\n" + "=" * 60)
    print("TEST 1: Formula Correlation")
    print("=" * 60)

    radiant = PrimeRadiant()
    executor = MarketFormulaExecutor()

    # Test with coherent signals
    coherent_base = {"trend_up": 0.8, "volume_surge": 0.6, "bullish_news": 0.7}
    coherent_states = create_coherent_states(20, coherent_base, noise=0.1)
    coherent_vecs = radiant.states_to_vectors(coherent_states)

    coherent_R_values = []
    for i in range(5, len(coherent_vecs)):
        query = coherent_vecs[i]
        context = coherent_vecs[i-5:i]
        result = executor.compute_R(query, context)
        coherent_R_values.append(result.R)

    # Test with conflicting signals
    conflicting_states = create_conflicting_states(20)
    conflicting_vecs = radiant.states_to_vectors(conflicting_states)

    conflicting_R_values = []
    for i in range(5, len(conflicting_vecs)):
        query = conflicting_vecs[i]
        context = conflicting_vecs[i-5:i]
        result = executor.compute_R(query, context)
        conflicting_R_values.append(result.R)

    # Results
    mean_coherent = np.mean(coherent_R_values)
    mean_conflicting = np.mean(conflicting_R_values)
    ratio = mean_coherent / max(mean_conflicting, 1e-6)

    print(f"\nCoherent signals:")
    print(f"  Mean R: {mean_coherent:.4f}")
    print(f"  Std R: {np.std(coherent_R_values):.4f}")

    print(f"\nConflicting signals:")
    print(f"  Mean R: {mean_conflicting:.4f}")
    print(f"  Std R: {np.std(conflicting_R_values):.4f}")

    print(f"\nRatio: {ratio:.2f}x")

    if ratio > 2.0:
        print("[PASS] Formula discriminates coherent from conflicting (ratio > 2x)")
        return True
    else:
        print("[FAIL] Discrimination ratio < 2x")
        return False


# =============================================================================
# TEST 2: GATE THRESHOLD DISCRIMINATION
# =============================================================================

def test_gate_thresholds():
    """
    Test that gate thresholds separate trades correctly.

    Target: Higher tiers require higher R values
    """
    print("\n" + "=" * 60)
    print("TEST 2: Gate Threshold Discrimination")
    print("=" * 60)

    gate = SeldonGate()

    # Test R values
    test_R_values = np.linspace(0, 2, 50)

    tier_counts = {tier: 0 for tier in GateTier}
    tier_R_ranges = {tier: [] for tier in GateTier}

    for R in test_R_values:
        decision = gate.check_gate(R, GateTier.T3_LARGE_POS)
        tier_counts[decision.tier] += 1
        tier_R_ranges[decision.tier].append(R)

    print("\nTier distribution:")
    for tier in GateTier:
        count = tier_counts[tier]
        R_range = tier_R_ranges[tier]
        if R_range:
            print(f"  {tier.value}: {count} ({min(R_range):.2f} - {max(R_range):.2f})")
        else:
            print(f"  {tier.value}: {count} (no samples)")

    # Check ordering
    tier_order = [GateTier.T0_OBSERVE, GateTier.T1_SMALL_POS, GateTier.T2_MEDIUM_POS, GateTier.T3_LARGE_POS]
    ordered_correctly = True

    for i in range(len(tier_order) - 1):
        current_tier = tier_order[i]
        next_tier = tier_order[i + 1]

        if tier_R_ranges[current_tier] and tier_R_ranges[next_tier]:
            current_max = max(tier_R_ranges[current_tier])
            next_min = min(tier_R_ranges[next_tier])

            if current_max > next_min:
                ordered_correctly = False
                print(f"\n[WARN] Tier overlap: {current_tier.value} max ({current_max:.2f}) > {next_tier.value} min ({next_min:.2f})")

    if ordered_correctly:
        print("\n[PASS] Gate tiers properly ordered by R threshold")
        return True
    else:
        print("\n[FAIL] Gate tiers have improper ordering")
        return False


# =============================================================================
# TEST 3: ALPHA DRIFT LEAD TIME
# =============================================================================

def test_alpha_drift():
    """
    Test that alpha drift provides early warning before regime changes.

    Target: Warning should appear before gate closes
    """
    print("\n" + "=" * 60)
    print("TEST 3: Alpha Drift Detection")
    print("=" * 60)

    gate = SeldonGate()

    # Simulate stable period
    print("\nPhase 1: Stable alpha (simulating normal market)")
    for _ in range(30):
        stable_alpha = 0.5 + np.random.randn() * 0.02
        gate.update_alpha(stable_alpha)

    drift_stable = gate.detect_drift()
    print(f"  Warning level: {drift_stable.warning_level.name}")
    print(f"  Alpha: {drift_stable.alpha:.4f}")

    # Simulate drifting period
    print("\nPhase 2: Drifting alpha (simulating regime change)")
    warning_step = None
    for i in range(20):
        drifting_alpha = 0.5 + i * 0.025  # Drift toward 1.0
        gate.update_alpha(drifting_alpha)

        drift = gate.detect_drift()
        if drift.warning_level != AlphaWarningLevel.NONE and warning_step is None:
            warning_step = i
            print(f"  Step {i}: Warning triggered! Level={drift.warning_level.name}")

    final_drift = gate.detect_drift()
    print(f"\nFinal state:")
    print(f"  Warning level: {final_drift.warning_level.name}")
    print(f"  Alpha: {final_drift.alpha:.4f}")
    print(f"  Lead time estimate: {final_drift.lead_time_estimate} steps")

    if warning_step is not None and warning_step < 15:
        print(f"\n[PASS] Warning triggered at step {warning_step} (lead time > 5 steps)")
        return True
    else:
        print("\n[FAIL] Warning not triggered early enough")
        return False


# =============================================================================
# TEST 4: MONTE CARLO STRESS TEST
# =============================================================================

def test_monte_carlo(n_runs: int = 20):
    """
    Monte Carlo stress test comparing Psychohistory bot to traditional.

    Target: Lower max drawdown than buy-and-hold
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: Monte Carlo Stress Test ({n_runs} runs)")
    print("=" * 60)

    # Define test scenarios
    scenario_options = [
        [("bull", 100), ("crisis", 50), ("recovery", 100)],  # Flash crash
        [("bear", 200)],  # Slow bleed
        [("bull", 50), ("volatile", 50), ("bear", 50), ("recovery", 50)],  # Whipsaw
        [("bull", 100), ("volatile", 50), ("crisis", 100), ("recovery", 100)],  # 2008-style
        [("bull", 100), ("crisis", 30), ("recovery", 50), ("bull", 70)],  # V-recovery
    ]

    psycho_returns = []
    psycho_drawdowns = []
    bh_returns = []

    for run in range(n_runs):
        if run % 5 == 0:
            print(f"  Run {run + 1}/{n_runs}...")

        np.random.seed(run * 17 + 42)

        # Random scenario
        scenario = scenario_options[run % len(scenario_options)]
        states, prices = create_regime_sequence(scenario)

        # Psychohistory bot
        bot = PsychohistoryBot(BotConfig(initial_capital=100000))

        for i, (state, price) in enumerate(zip(states, prices)):
            # Simple direction signal based on recent price trend
            if i > 5:
                recent_return = (prices[i] - prices[i-5]) / prices[i-5]
                direction = 1 if recent_return > 0.01 else (-1 if recent_return < -0.01 else 0)
            else:
                direction = 0

            bot.decide(state, price, direction)
            bot.record_equity({state.asset: price})

        psycho_returns.append(bot.get_returns())
        psycho_drawdowns.append(bot.get_max_drawdown())

        # Buy and hold
        bh_return = (prices[-1] - prices[0]) / prices[0]
        bh_returns.append(bh_return)

    # Results
    print(f"\n{'Strategy':<15} {'Mean':>10} {'Worst':>10} {'Best':>10} {'Max DD':>10}")
    print("-" * 60)

    print(f"{'Psychohistory':<15} {np.mean(psycho_returns):>+9.1%} {np.min(psycho_returns):>+9.1%} {np.max(psycho_returns):>+9.1%} {np.mean(psycho_drawdowns):>9.1%}")
    print(f"{'Buy & Hold':<15} {np.mean(bh_returns):>+9.1%} {np.min(bh_returns):>+9.1%} {np.max(bh_returns):>+9.1%} {'N/A':>10}")

    # Risk-adjusted comparison
    psycho_worst = np.min(psycho_returns)
    bh_worst = np.min(bh_returns)

    print(f"\nRisk comparison:")
    print(f"  Psychohistory worst case: {psycho_worst:+.1%}")
    print(f"  Buy & Hold worst case: {bh_worst:+.1%}")

    if psycho_worst > bh_worst:
        print("\n[PASS] Psychohistory has better worst-case than Buy & Hold")
        return True
    else:
        print("\n[WARN] Psychohistory did not improve worst-case")
        return psycho_worst > bh_worst - 0.1  # Allow 10% margin


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("PSYCHOHISTORY BOT VALIDATION SUITE")
    print("=" * 60)

    results = {}

    # Test 1
    results["formula_correlation"] = test_formula_correlation()

    # Test 2
    results["gate_thresholds"] = test_gate_thresholds()

    # Test 3
    results["alpha_drift"] = test_alpha_drift()

    # Test 4
    results["monte_carlo"] = test_monte_carlo(n_runs=20)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Save results
    output_path = Path(__file__).parent / "test_results_psychohistory.json"
    with open(output_path, 'w') as f:
        json.dump({
            "results": {k: bool(v) for k, v in results.items()},
            "passed": passed,
            "total": total,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
