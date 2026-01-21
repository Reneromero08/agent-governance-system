"""
FAST STRESS TEST
================

Faster version that loads model once and uses efficient simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_sources import MockMarketScenario, MarketTick

# Add compass path
LAB_PATH = Path(__file__).parent.parent
COMPASS_PATH = LAB_PATH / "CAT_CHAT" / "compass_agi"
sys.path.insert(0, str(COMPASS_PATH))


class FastParadigmBot:
    """
    Fast version of market bot that shares a single detector instance.
    """

    def __init__(
        self,
        detector,  # Shared detector instance
        initial_capital: float = 100000,
        use_paradigm: bool = True,  # If False, acts as traditional bot
    ):
        self.detector = detector
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.use_paradigm = use_paradigm

        self.position_size = 0.0
        self.entry_price = 0.0
        self.trades = []

        # Regime-based position limits
        self.position_limits = {
            "STABLE": 1.0,
            "TRANSITIONAL": 0.5,
            "SHIFT": 0.2,
        }

    def reset(self):
        """Reset for new run."""
        self.capital = self.initial_capital
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trades = []

    def detect_regime(self, headlines: List[str]) -> Tuple[str, float]:
        """Detect regime from headlines."""
        if not self.use_paradigm or self.detector is None:
            return "STABLE", 0.0

        result = self.detector.detect_shift(headlines)
        return result['shift_type'], result['shift_score']

    def compute_signal(self, prices: List[float]) -> str:
        """Simple momentum signal."""
        if len(prices) < 10:
            return "HOLD"

        sma_short = np.mean(prices[-5:])
        sma_long = np.mean(prices[-10:])

        if sma_short > sma_long * 1.01:
            return "BUY"
        elif sma_short < sma_long * 0.99:
            return "SELL"
        return "HOLD"

    def process_tick(self, price: float, headlines: List[str], prices: List[float]) -> Optional[Dict]:
        """Process a tick and return trade if any."""
        if len(prices) < 5:
            return None

        # Detect regime
        regime, shift_score = self.detect_regime(headlines)

        # Get position limit based on regime
        max_position = self.position_limits.get(regime, 1.0)

        # Get technical signal
        signal = self.compute_signal(prices)

        trade = None

        # Trading logic
        if self.use_paradigm:
            # Paradigm-aware: respect regime limits
            if regime == "SHIFT":
                # Shift: close positions, stay out
                if self.position_size > 0:
                    pnl = (price - self.entry_price) * self.position_size
                    self.capital += pnl
                    trade = {"action": "SELL", "price": price, "pnl": pnl, "reason": "SHIFT_EXIT"}
                    self.position_size = 0
            elif signal == "BUY" and self.position_size == 0:
                # Open position up to limit
                size = (self.capital * max_position * 0.5) / price
                self.position_size = size
                self.entry_price = price
                trade = {"action": "BUY", "price": price, "size": size}
            elif signal == "SELL" and self.position_size > 0:
                pnl = (price - self.entry_price) * self.position_size
                self.capital += pnl
                trade = {"action": "SELL", "price": price, "pnl": pnl, "reason": "SIGNAL"}
                self.position_size = 0
        else:
            # Traditional: ignore regime, follow signals
            if signal == "BUY" and self.position_size == 0:
                size = (self.capital * 0.5) / price
                self.position_size = size
                self.entry_price = price
                trade = {"action": "BUY", "price": price, "size": size}
            elif signal == "SELL" and self.position_size > 0:
                pnl = (price - self.entry_price) * self.position_size
                self.capital += pnl
                trade = {"action": "SELL", "price": price, "pnl": pnl}
                self.position_size = 0

        if trade:
            self.trades.append(trade)

        return trade

    def get_equity(self, current_price: float) -> float:
        """Get current equity including unrealized P&L."""
        unrealized = (current_price - self.entry_price) * self.position_size if self.position_size > 0 else 0
        return self.capital + unrealized


def run_backtest(
    bot: FastParadigmBot,
    ticks: List[MarketTick],
) -> Tuple[List[float], List[Dict]]:
    """Run backtest, return equity curve and trades."""
    bot.reset()
    equity_curve = [bot.initial_capital]
    prices_history = []

    for tick in ticks:
        prices_history.append(tick.price)
        if len(prices_history) > 50:
            prices_history = prices_history[-50:]

        bot.process_tick(tick.price, tick.headlines, prices_history)
        equity_curve.append(bot.get_equity(tick.price))

    return equity_curve, bot.trades


def compute_max_drawdown(equity: List[float]) -> float:
    """Compute max drawdown."""
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.max(dd))


def main():
    print("=" * 70)
    print("FAST STRESS TEST - Can You Trust This Bot?")
    print("=" * 70)

    # Load detector ONCE
    print("\nLoading paradigm detector (one-time)...")
    try:
        from realtime_paradigm_detector import ParadigmShiftDetector
        detector = ParadigmShiftDetector()
        print("Detector loaded.")
    except Exception as e:
        print(f"Detector failed: {e}")
        detector = None

    initial_capital = 100000
    generator = MockMarketScenario(seed=42)

    # =========================================================================
    # PART 1: STRESS SCENARIOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: STRESS SCENARIOS (2000 ticks each)")
    print("=" * 70)

    scenarios = {
        "flash_crash": [("stable_bull", 700), ("crisis", 200), ("recovery", 1100)],
        "slow_bleed": [("stable_bear", 2000)],
        "whipsaw": [("stable_bull", 200), ("stable_bear", 200)] * 5,
        "2008_style": [("stable_bull", 500), ("early_warning", 200), ("crisis", 500), ("recovery", 800)],
        "covid_style": [("stable_bull", 800), ("crisis", 200), ("recovery", 400), ("stable_bull", 600)],
    }

    stress_results = {}
    print(f"\n{'Scenario':<15} {'Paradigm':>12} {'Traditional':>12} {'Buy&Hold':>12} {'Winner':>12}")
    print("-" * 65)

    for name, phases in scenarios.items():
        # Generate data
        ticks = generator.generate_multi_phase(phases, start_price=100)

        # Paradigm-aware bot
        bot_paradigm = FastParadigmBot(detector, initial_capital, use_paradigm=True)
        eq_paradigm, _ = run_backtest(bot_paradigm, ticks)
        ret_paradigm = (eq_paradigm[-1] - initial_capital) / initial_capital
        dd_paradigm = compute_max_drawdown(eq_paradigm)

        # Traditional bot
        bot_trad = FastParadigmBot(detector, initial_capital, use_paradigm=False)
        eq_trad, _ = run_backtest(bot_trad, ticks)
        ret_trad = (eq_trad[-1] - initial_capital) / initial_capital

        # Buy and hold
        ret_bh = (ticks[-1].price - ticks[0].price) / ticks[0].price

        # Winner
        returns = {"Paradigm": ret_paradigm, "Traditional": ret_trad, "BuyHold": ret_bh}
        winner = max(returns, key=returns.get)

        stress_results[name] = {
            "paradigm": ret_paradigm,
            "traditional": ret_trad,
            "buyhold": ret_bh,
            "paradigm_dd": dd_paradigm,
            "winner": winner,
        }

        print(f"{name:<15} {ret_paradigm:>+11.2%} {ret_trad:>+11.2%} {ret_bh:>+11.2%} {winner:>12}")

    # =========================================================================
    # PART 2: MONTE CARLO (faster with shared detector)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: MONTE CARLO SIMULATION (50 runs x 1000 ticks)")
    print("=" * 70)

    n_runs = 50
    mc_paradigm = []
    mc_traditional = []
    mc_buyhold = []

    for run in range(n_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{n_runs}...")

        # Generate random market
        seed = run * 17 + 123
        gen = MockMarketScenario(seed=seed)

        # Random phase sequence
        np.random.seed(seed)
        phases = []
        for _ in range(10):
            phase = np.random.choice(["stable_bull", "stable_bear", "volatile", "early_warning", "crisis", "recovery"],
                                      p=[0.35, 0.15, 0.20, 0.15, 0.05, 0.10])
            duration = np.random.randint(50, 200)
            phases.append((phase, duration))

        ticks = gen.generate_multi_phase(phases, start_price=100)

        # Test all strategies
        bot_p = FastParadigmBot(detector, initial_capital, use_paradigm=True)
        eq_p, _ = run_backtest(bot_p, ticks)
        mc_paradigm.append((eq_p[-1] - initial_capital) / initial_capital)

        bot_t = FastParadigmBot(detector, initial_capital, use_paradigm=False)
        eq_t, _ = run_backtest(bot_t, ticks)
        mc_traditional.append((eq_t[-1] - initial_capital) / initial_capital)

        mc_buyhold.append((ticks[-1].price - ticks[0].price) / ticks[0].price)

    # Analyze Monte Carlo
    print("\n--- MONTE CARLO RESULTS ---")
    print(f"{'Strategy':<15} {'Mean':>10} {'Std':>10} {'Worst':>10} {'Best':>10} {'P(Loss)':>10}")
    print("-" * 70)

    for name, data in [("Paradigm", mc_paradigm), ("Traditional", mc_traditional), ("Buy&Hold", mc_buyhold)]:
        mean = np.mean(data)
        std = np.std(data)
        worst = np.min(data)
        best = np.max(data)
        p_loss = np.mean([1 if r < 0 else 0 for r in data])
        print(f"{name:<15} {mean:>+9.2%} {std:>9.2%} {worst:>+9.2%} {best:>+9.2%} {p_loss:>9.1%}")

    # =========================================================================
    # PART 3: FINAL VERDICT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    paradigm_mean = np.mean(mc_paradigm)
    paradigm_worst = np.min(mc_paradigm)
    paradigm_p_loss = np.mean([1 if r < 0 else 0 for r in mc_paradigm])
    trad_mean = np.mean(mc_traditional)

    # Count stress test wins
    stress_wins = sum(1 for r in stress_results.values() if r["winner"] == "Paradigm")

    print(f"\nParadigm Bot Performance:")
    print(f"  Mean return: {paradigm_mean:+.2%}")
    print(f"  Worst case: {paradigm_worst:+.2%}")
    print(f"  Probability of loss: {paradigm_p_loss:.1%}")
    print(f"  Stress test wins: {stress_wins}/{len(stress_results)}")
    print(f"  Advantage over traditional: {paradigm_mean - trad_mean:+.2%}")

    # Risk assessment
    print("\n--- RISK ASSESSMENT ---")
    if paradigm_worst > -0.30:
        print("[OK] Worst case loss < 30%")
    else:
        print("[WARN] Worst case loss >= 30%")

    if paradigm_p_loss < 0.40:
        print("[OK] Probability of loss < 40%")
    else:
        print("[WARN] Probability of loss >= 40%")

    if paradigm_mean > trad_mean:
        print("[OK] Outperforms traditional on average")
    else:
        print("[WARN] Underperforms traditional on average")

    if stress_wins >= 3:
        print("[OK] Wins most stress scenarios")
    else:
        print("[WARN] Loses most stress scenarios")

    # Final recommendation
    print("\n--- RECOMMENDATION ---")
    score = 0
    if paradigm_worst > -0.30: score += 1
    if paradigm_p_loss < 0.40: score += 1
    if paradigm_mean > trad_mean: score += 1
    if stress_wins >= 3: score += 1

    if score >= 3:
        print("CAUTIOUSLY PROCEED with small position sizes")
        print("- Start with paper trading")
        print("- Use strict position limits (20% max)")
        print("- Always have stop losses")
    else:
        print("DO NOT DEPLOY without further development")
        print("- Paradigm detection needs improvement")
        print("- Consider additional safety mechanisms")

    print("\n--- CRITICAL REMINDERS ---")
    print("1. This is SYNTHETIC data - real markets differ")
    print("2. Options have additional risks (Greeks)")
    print("3. Never trade more than you can afford to lose")
    print("4. Paper trade first, then small positions")

    # Save results
    output = {
        "stress_tests": stress_results,
        "monte_carlo": {
            "paradigm": {"mean": paradigm_mean, "worst": paradigm_worst, "p_loss": paradigm_p_loss},
            "traditional": {"mean": trad_mean},
        },
        "score": score,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent / "stress_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
