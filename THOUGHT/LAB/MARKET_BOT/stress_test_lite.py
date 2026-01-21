"""
STRESS TEST LITE
================

Ultra-fast stress test using rule-based regime detection.

This tests the TRADING STRATEGY, not the paradigm detector.
(Paradigm detector accuracy was validated separately on historical data)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path


# =============================================================================
# MOCK REGIME DETECTOR (rule-based, no ML)
# =============================================================================

class MockRegimeDetector:
    """
    Rule-based regime detection for fast testing.

    Uses keyword matching instead of embeddings.
    This simulates what the real detector does, just faster.
    """

    SHIFT_KEYWORDS = [
        "crisis", "crash", "freefall", "unprecedented", "transformation",
        "changing", "disruption", "breaking", "collapse", "death",
        "revolution", "upheaval", "earthquake", "panic", "catastroph",
    ]

    STABLE_KEYWORDS = [
        "steady", "reliable", "trusted", "loyal", "faithful",
        "consistent", "established", "guidance", "stable", "growth",
    ]

    WARNING_KEYWORDS = [
        "unusual", "questions", "cracks", "uncertainty", "caution",
        "warning", "concern", "volatile", "revision",
    ]

    def detect_shift(self, headlines: List[str]) -> Dict[str, Any]:
        """Detect regime from headlines using keywords."""
        text = " ".join(headlines).lower()

        shift_score = sum(1 for kw in self.SHIFT_KEYWORDS if kw in text)
        stable_score = sum(1 for kw in self.STABLE_KEYWORDS if kw in text)
        warning_score = sum(1 for kw in self.WARNING_KEYWORDS if kw in text)

        if shift_score >= 2:
            return {"shift_type": "SHIFT", "shift_score": 0.2}
        elif warning_score >= 2:
            return {"shift_type": "TRANSITIONAL", "shift_score": 0.05}
        elif stable_score >= 2:
            return {"shift_type": "STABLE", "shift_score": -0.1}
        else:
            return {"shift_type": "STABLE", "shift_score": 0.0}


# =============================================================================
# MARKET DATA GENERATOR
# =============================================================================

class FastMarketGenerator:
    """Generate market data quickly."""

    HEADLINES = {
        "stable_bull": [
            "Loyal shareholders rewarded with steady dividends",
            "Trusted institutions maintain guidance",
            "Consistent growth in key metrics",
            "Established companies deliver reliable returns",
        ],
        "stable_bear": [
            "Markets drift lower on steady selling",
            "Gradual rotation continues",
            "Patient investors wait for opportunities",
        ],
        "volatile": [
            "Markets show mixed signals",
            "Conflicting data keeps traders cautious",
        ],
        "early_warning": [
            "Unusual volatility in credit markets",
            "Questions raised about valuations",
            "Warning signs emerge in leading indicators",
            "Cracks appearing in market structure",
        ],
        "crisis": [
            "Markets in freefall as crisis deepens",
            "Unprecedented disruption across sectors",
            "Everything is changing rapidly",
            "Complete transformation underway",
            "Panic selling accelerates",
        ],
        "recovery": [
            "Markets stabilizing after turmoil",
            "Cautious optimism returns",
            "Trust slowly rebuilding",
        ],
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_phase(self, phase: str, n_ticks: int, start_price: float) -> Tuple[List[float], List[List[str]]]:
        """Generate price and headlines for a phase."""
        # Phase parameters
        params = {
            "stable_bull": (0.0015, 0.01),
            "stable_bear": (-0.001, 0.01),
            "volatile": (0.0, 0.025),
            "early_warning": (-0.001, 0.018),
            "crisis": (-0.012, 0.035),
            "recovery": (0.004, 0.02),
        }

        trend, vol = params.get(phase, (0, 0.01))
        headlines_pool = self.HEADLINES.get(phase, ["Market update"])

        prices = [start_price]
        all_headlines = []

        for _ in range(n_ticks):
            change = trend + self.rng.randn() * vol
            prices.append(max(prices[-1] * (1 + change), 1.0))

            # Sample headlines
            n_headlines = self.rng.randint(2, 5)
            headlines = list(self.rng.choice(headlines_pool, size=min(n_headlines, len(headlines_pool)), replace=True))
            all_headlines.append(headlines)

        return prices[1:], all_headlines

    def generate_multi_phase(self, phases: List[Tuple[str, int]], start_price: float = 100) -> Tuple[List[float], List[List[str]]]:
        """Generate multi-phase market."""
        all_prices = []
        all_headlines = []
        price = start_price

        for phase, n_ticks in phases:
            prices, headlines = self.generate_phase(phase, n_ticks, price)
            all_prices.extend(prices)
            all_headlines.extend(headlines)
            price = prices[-1] if prices else price

        return all_prices, all_headlines


# =============================================================================
# TRADING BOT
# =============================================================================

class FastTradingBot:
    """Fast trading bot for stress testing."""

    def __init__(self, initial_capital: float = 100000, use_paradigm: bool = True):
        self.initial_capital = initial_capital
        self.use_paradigm = use_paradigm
        self.detector = MockRegimeDetector()
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = [self.initial_capital]

    def get_regime(self, headlines: List[str]) -> str:
        if not self.use_paradigm:
            return "STABLE"
        result = self.detector.detect_shift(headlines)
        return result["shift_type"]

    def get_signal(self, prices: List[float]) -> str:
        if len(prices) < 10:
            return "HOLD"
        sma5 = np.mean(prices[-5:])
        sma10 = np.mean(prices[-10:])
        if sma5 > sma10 * 1.005:
            return "BUY"
        elif sma5 < sma10 * 0.995:
            return "SELL"
        return "HOLD"

    def step(self, price: float, headlines: List[str], prices: List[float]):
        """Process one tick."""
        regime = self.get_regime(headlines)
        signal = self.get_signal(prices)

        # Position limits by regime
        max_pos = {"STABLE": 1.0, "TRANSITIONAL": 0.5, "SHIFT": 0.2}[regime]

        # Trading logic
        if self.use_paradigm and regime == "SHIFT" and self.position > 0:
            # Exit on SHIFT
            pnl = (price - self.entry_price) * self.position
            self.capital += pnl
            self.trades.append({"type": "EXIT_SHIFT", "pnl": pnl})
            self.position = 0

        elif signal == "BUY" and self.position == 0:
            size = (self.capital * max_pos * 0.5) / price
            self.position = size
            self.entry_price = price
            self.trades.append({"type": "BUY", "price": price})

        elif signal == "SELL" and self.position > 0:
            pnl = (price - self.entry_price) * self.position
            self.capital += pnl
            self.trades.append({"type": "SELL", "pnl": pnl})
            self.position = 0

        # Update equity
        unrealized = (price - self.entry_price) * self.position if self.position > 0 else 0
        self.equity_curve.append(self.capital + unrealized)

    def run(self, prices: List[float], headlines: List[List[str]]) -> Tuple[float, float, int]:
        """Run through all data. Returns (return, max_dd, n_trades)."""
        self.reset()
        price_history = []

        for i, (price, hdl) in enumerate(zip(prices, headlines)):
            price_history.append(price)
            if len(price_history) > 50:
                price_history = price_history[-50:]
            self.step(price, hdl, price_history)

        final_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        max_dd = self._compute_max_dd()
        return final_return, max_dd, len(self.trades)

    def _compute_max_dd(self) -> float:
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(np.max(dd))


# =============================================================================
# MAIN STRESS TEST
# =============================================================================

def main():
    print("=" * 70)
    print("STRESS TEST LITE - Fast Trading Strategy Validation")
    print("=" * 70)
    print("\nThis tests the REGIME-AWARE TRADING STRATEGY")
    print("(Paradigm detector accuracy validated separately on historical data)\n")

    # =========================================================================
    # PART 1: STRESS SCENARIOS
    # =========================================================================
    print("=" * 70)
    print("PART 1: STRESS SCENARIOS")
    print("=" * 70)

    scenarios = {
        "flash_crash": [("stable_bull", 500), ("crisis", 150), ("recovery", 350)],
        "slow_bleed": [("stable_bear", 1000)],
        "whipsaw": [("stable_bull", 100), ("stable_bear", 100)] * 5,
        "2008_style": [("stable_bull", 300), ("early_warning", 150), ("crisis", 300), ("recovery", 250)],
        "covid_style": [("stable_bull", 400), ("crisis", 100), ("recovery", 200), ("stable_bull", 300)],
        "long_bull": [("stable_bull", 1000)],
        "double_dip": [("stable_bull", 200), ("crisis", 150), ("recovery", 200), ("crisis", 150), ("recovery", 300)],
    }

    print(f"\n{'Scenario':<15} {'Paradigm':>10} {'Trad':>10} {'B&H':>10} {'Best':>12} {'Para DD':>10}")
    print("-" * 75)

    stress_results = {}
    gen = FastMarketGenerator(seed=42)

    for name, phases in scenarios.items():
        prices, headlines = gen.generate_multi_phase(phases, start_price=100)

        # Paradigm-aware
        bot_p = FastTradingBot(use_paradigm=True)
        ret_p, dd_p, trades_p = bot_p.run(prices, headlines)

        # Traditional
        bot_t = FastTradingBot(use_paradigm=False)
        ret_t, dd_t, trades_t = bot_t.run(prices, headlines)

        # Buy and hold
        ret_bh = (prices[-1] - prices[0]) / prices[0]

        best = max([("Paradigm", ret_p), ("Trad", ret_t), ("B&H", ret_bh)], key=lambda x: x[1])[0]

        stress_results[name] = {"paradigm": ret_p, "trad": ret_t, "bh": ret_bh, "dd": dd_p}
        print(f"{name:<15} {ret_p:>+9.1%} {ret_t:>+9.1%} {ret_bh:>+9.1%} {best:>12} {dd_p:>9.1%}")

    # =========================================================================
    # PART 2: MONTE CARLO
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: MONTE CARLO (200 random markets)")
    print("=" * 70)

    n_runs = 200
    mc_paradigm = []
    mc_trad = []
    mc_bh = []

    phase_options = ["stable_bull", "stable_bear", "volatile", "early_warning", "crisis", "recovery"]
    phase_probs = [0.35, 0.15, 0.20, 0.12, 0.08, 0.10]

    for run in range(n_runs):
        seed = run * 17 + 999
        gen = FastMarketGenerator(seed=seed)
        np.random.seed(seed)

        # Random phase sequence
        phases = []
        for _ in range(np.random.randint(5, 15)):
            phase = np.random.choice(phase_options, p=phase_probs)
            duration = np.random.randint(30, 200)
            phases.append((phase, duration))

        prices, headlines = gen.generate_multi_phase(phases, start_price=100)

        bot_p = FastTradingBot(use_paradigm=True)
        ret_p, _, _ = bot_p.run(prices, headlines)
        mc_paradigm.append(ret_p)

        bot_t = FastTradingBot(use_paradigm=False)
        ret_t, _, _ = bot_t.run(prices, headlines)
        mc_trad.append(ret_t)

        mc_bh.append((prices[-1] - prices[0]) / prices[0])

    # Analyze
    print(f"\n{'Strategy':<15} {'Mean':>10} {'Median':>10} {'Worst':>10} {'Best':>10} {'P(Loss)':>10}")
    print("-" * 70)

    for name, data in [("Paradigm", mc_paradigm), ("Traditional", mc_trad), ("Buy&Hold", mc_bh)]:
        mean = np.mean(data)
        median = np.median(data)
        worst = np.min(data)
        best = np.max(data)
        p_loss = np.mean([1 if r < 0 else 0 for r in data])
        print(f"{name:<15} {mean:>+9.1%} {median:>+9.1%} {worst:>+9.1%} {best:>+9.1%} {p_loss:>9.1%}")

    # =========================================================================
    # PART 3: DISTRIBUTION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: RETURN DISTRIBUTION")
    print("=" * 70)

    # Percentiles
    print(f"\n{'Percentile':<15} {'Paradigm':>12} {'Traditional':>12} {'Advantage':>12}")
    print("-" * 55)
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        p_val = np.percentile(mc_paradigm, pct)
        t_val = np.percentile(mc_trad, pct)
        print(f"{pct}%{'':<13} {p_val:>+11.1%} {t_val:>+11.1%} {p_val - t_val:>+11.1%}")

    # =========================================================================
    # PART 4: VERDICT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    paradigm_mean = np.mean(mc_paradigm)
    paradigm_worst = np.min(mc_paradigm)
    paradigm_p_loss = np.mean([1 if r < 0 else 0 for r in mc_paradigm])
    trad_mean = np.mean(mc_trad)

    stress_wins = sum(1 for r in stress_results.values() if r["paradigm"] > r["trad"])

    print(f"\nParadigm Strategy Results:")
    print(f"  Mean return: {paradigm_mean:+.1%}")
    print(f"  Worst case: {paradigm_worst:+.1%}")
    print(f"  P(loss): {paradigm_p_loss:.1%}")
    print(f"  vs Traditional: {paradigm_mean - trad_mean:+.1%}")
    print(f"  Stress wins: {stress_wins}/{len(stress_results)}")

    # Scoring
    score = 0
    checks = []

    if paradigm_worst > -0.40:
        score += 1
        checks.append("[PASS] Worst case > -40%")
    else:
        checks.append("[FAIL] Worst case <= -40%")

    if paradigm_p_loss < 0.45:
        score += 1
        checks.append("[PASS] P(loss) < 45%")
    else:
        checks.append("[FAIL] P(loss) >= 45%")

    if paradigm_mean > trad_mean:
        score += 1
        checks.append("[PASS] Beats traditional on average")
    else:
        checks.append("[FAIL] Underperforms traditional")

    if stress_wins >= len(stress_results) // 2:
        score += 1
        checks.append("[PASS] Wins majority of stress tests")
    else:
        checks.append("[FAIL] Loses majority of stress tests")

    print("\nRisk Checks:")
    for check in checks:
        print(f"  {check}")

    print(f"\nScore: {score}/4")

    if score >= 3:
        print("\n>>> STRATEGY SHOWS PROMISE <<<")
        print("Proceed with caution:")
        print("  1. Paper trade first")
        print("  2. Start with 10% of intended capital")
        print("  3. Use strict stop losses")
        print("  4. Monitor regime detection accuracy")
    else:
        print("\n>>> NEEDS MORE WORK <<<")
        print("Do not deploy. Consider:")
        print("  1. Improving regime detection thresholds")
        print("  2. Adding more safety mechanisms")
        print("  3. Testing on real historical data")

    # Save results
    output = {
        "stress_tests": stress_results,
        "monte_carlo": {
            "paradigm": {"mean": paradigm_mean, "worst": paradigm_worst, "p_loss": paradigm_p_loss},
            "traditional": {"mean": trad_mean},
            "n_runs": n_runs,
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
