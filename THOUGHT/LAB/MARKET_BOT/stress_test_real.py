"""
REAL STRESS TEST
================

Uses the actual cached paradigm detector (not mock keywords).
Tests the full system end-to-end.
"""

import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import json
from pathlib import Path
import time

# Local imports
from cached_detector import CachedParadigmDetector, EScoreDetector, CACHE_DIR


# =============================================================================
# REALISTIC HEADLINE GENERATOR
# =============================================================================

class RealisticHeadlineGenerator:
    """
    Generate headlines that don't use obvious keywords.
    Tests if the SEMANTIC detector works, not just keyword matching.
    """

    # These avoid the obvious shift/stable keywords
    STABLE_HEADLINES = [
        "S&P 500 ends week higher on consistent corporate results",
        "Federal Reserve maintains current policy stance",
        "Employment numbers align with analyst forecasts",
        "Consumer spending remains on expected trajectory",
        "Market participants show measured optimism",
        "Tech sector delivers anticipated quarterly performance",
        "Bond yields hold within established range",
        "Economic indicators suggest continuation of trends",
        "Retail sales data meets consensus estimates",
        "Manufacturing index reflects ongoing expansion",
    ]

    WARNING_HEADLINES = [
        "Volatility measures tick higher amid mixed signals",
        "Analysts note divergence in sector performance",
        "Credit spreads show signs of widening",
        "Institutional flows suggest positioning shifts",
        "Options market implies increased hedging activity",
        "Correlation patterns deviate from historical norms",
        "Market breadth narrows as leadership concentrates",
        "Sentiment surveys show growing divergence",
        "Liquidity conditions tighten in some segments",
        "Forward guidance becomes less certain",
    ]

    CRISIS_HEADLINES = [
        "Markets experience significant dislocation",
        "Correlation structures break down across assets",
        "Liquidity evaporates in key market segments",
        "Risk models fail to capture current dynamics",
        "Unprecedented moves in volatility indices",
        "Cross-asset contagion accelerates",
        "Margin calls force rapid deleveraging",
        "Central banks convene emergency meetings",
        "Circuit breakers triggered multiple times",
        "Institutional investors rush for exits",
    ]

    RECOVERY_HEADLINES = [
        "Markets find footing after recent turbulence",
        "Buyers emerge at lower levels",
        "Volatility begins to normalize",
        "Credit markets show signs of stabilization",
        "Policy response provides support",
        "Risk appetite slowly returns",
        "Correlation patterns begin to restore",
        "Volume normalizes from extreme levels",
        "Sentiment indicators turn less negative",
        "Market structure healing underway",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, phase: str, n: int = 5) -> List[str]:
        """Generate n headlines for a phase."""
        pools = {
            "stable_bull": self.STABLE_HEADLINES,
            "stable_bear": self.STABLE_HEADLINES,  # Same headlines, different price
            "volatile": self.WARNING_HEADLINES,
            "early_warning": self.WARNING_HEADLINES,
            "crisis": self.CRISIS_HEADLINES,
            "recovery": self.RECOVERY_HEADLINES,
        }
        pool = pools.get(phase, self.STABLE_HEADLINES)
        return list(self.rng.choice(pool, size=min(n, len(pool)), replace=False))


# =============================================================================
# TRADING BOT WITH REAL DETECTOR
# =============================================================================

class RealTradingBot:
    """Trading bot using real cached detector."""

    def __init__(self, detector: CachedParadigmDetector, initial_capital: float = 100000, use_paradigm: bool = True):
        self.detector = detector
        self.initial_capital = initial_capital
        self.use_paradigm = use_paradigm
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.regime_history = []

    def get_regime(self, headlines: List[str]) -> Tuple[str, float]:
        if not self.use_paradigm:
            return "STABLE", 0.0
        result = self.detector.detect_shift(headlines)
        return result["shift_type"], result["shift_score"]

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
        regime, shift_score = self.get_regime(headlines)
        self.regime_history.append(regime)

        signal = self.get_signal(prices)
        max_pos = {"STABLE": 1.0, "TRANSITIONAL": 0.5, "SHIFT": 0.2}[regime]

        if self.use_paradigm and regime == "SHIFT" and self.position > 0:
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

        unrealized = (price - self.entry_price) * self.position if self.position > 0 else 0
        self.equity_curve.append(self.capital + unrealized)

    def run(self, prices: List[float], headlines_list: List[List[str]]) -> Dict:
        self.reset()
        price_history = []

        for price, headlines in zip(prices, headlines_list):
            price_history.append(price)
            if len(price_history) > 50:
                price_history = price_history[-50:]
            self.step(price, headlines, price_history)

        final_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        max_dd = self._compute_max_dd()

        return {
            "return": final_return,
            "max_dd": max_dd,
            "trades": len(self.trades),
            "regime_distribution": {
                "STABLE": self.regime_history.count("STABLE") / len(self.regime_history) if self.regime_history else 0,
                "SHIFT": self.regime_history.count("SHIFT") / len(self.regime_history) if self.regime_history else 0,
                "TRANSITIONAL": self.regime_history.count("TRANSITIONAL") / len(self.regime_history) if self.regime_history else 0,
            }
        }

    def _compute_max_dd(self) -> float:
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(np.max(dd))


# =============================================================================
# MARKET DATA GENERATOR
# =============================================================================

def generate_prices(n: int, trend: float, vol: float, start: float, seed: int) -> List[float]:
    """Generate price series."""
    rng = np.random.RandomState(seed)
    prices = [start]
    for _ in range(n - 1):
        change = trend + rng.randn() * vol
        prices.append(max(prices[-1] * (1 + change), 1.0))
    return prices


def generate_scenario(phases: List[Tuple[str, int]], seed: int = 42) -> Tuple[List[float], List[List[str]]]:
    """Generate prices and headlines for a scenario."""
    headline_gen = RealisticHeadlineGenerator(seed=seed)

    params = {
        "stable_bull": (0.0015, 0.01),
        "stable_bear": (-0.001, 0.01),
        "volatile": (0.0, 0.025),
        "early_warning": (-0.001, 0.018),
        "crisis": (-0.012, 0.035),
        "recovery": (0.004, 0.02),
    }

    all_prices = []
    all_headlines = []
    price = 100.0

    for phase, n_ticks in phases:
        trend, vol = params.get(phase, (0, 0.01))
        phase_prices = generate_prices(n_ticks, trend, vol, price, seed)
        price = phase_prices[-1]

        for _ in range(n_ticks):
            all_prices.append(phase_prices[min(len(phase_prices) - 1, _)])
            all_headlines.append(headline_gen.generate(phase))

        seed += 1

    return all_prices, all_headlines


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 70)
    print("REAL STRESS TEST - Full System with Cached Detector")
    print("=" * 70)

    # Initialize detector (uses cache if exists)
    print("\nLoading detector...")
    t0 = time.time()
    detector = CachedParadigmDetector()
    print(f"Detector ready in {time.time() - t0:.2f}s")

    # =========================================================================
    # PART 1: STRESS SCENARIOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: STRESS SCENARIOS (500 ticks each)")
    print("=" * 70)

    scenarios = {
        "flash_crash": [("stable_bull", 200), ("crisis", 50), ("recovery", 250)],
        "slow_bleed": [("stable_bear", 500)],
        "2008_style": [("stable_bull", 150), ("early_warning", 75), ("crisis", 125), ("recovery", 150)],
        "covid_style": [("stable_bull", 200), ("crisis", 50), ("recovery", 100), ("stable_bull", 150)],
        "double_dip": [("stable_bull", 100), ("crisis", 75), ("recovery", 100), ("crisis", 75), ("recovery", 150)],
    }

    print(f"\n{'Scenario':<15} {'Paradigm':>10} {'Trad':>10} {'B&H':>10} {'Regimes Detected'}")
    print("-" * 70)

    stress_results = {}

    for name, phases in scenarios.items():
        prices, headlines = generate_scenario(phases, seed=hash(name) % 10000)

        # Paradigm-aware bot
        bot_p = RealTradingBot(detector, use_paradigm=True)
        result_p = bot_p.run(prices, headlines)

        # Traditional bot
        bot_t = RealTradingBot(detector, use_paradigm=False)
        result_t = bot_t.run(prices, headlines)

        # Buy and hold
        ret_bh = (prices[-1] - prices[0]) / prices[0]

        # Format regime distribution
        regimes = result_p["regime_distribution"]
        regime_str = f"ST:{regimes['STABLE']:.0%} SH:{regimes['SHIFT']:.0%} TR:{regimes['TRANSITIONAL']:.0%}"

        stress_results[name] = {
            "paradigm": result_p["return"],
            "traditional": result_t["return"],
            "buyhold": ret_bh,
            "regimes": regimes,
        }

        print(f"{name:<15} {result_p['return']:>+9.1%} {result_t['return']:>+9.1%} {ret_bh:>+9.1%} {regime_str}")

    # =========================================================================
    # PART 2: MONTE CARLO (fewer runs but real detector)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: MONTE CARLO (30 runs, real detector)")
    print("=" * 70)

    n_runs = 30
    mc_paradigm = []
    mc_trad = []
    mc_bh = []

    phase_options = ["stable_bull", "stable_bear", "volatile", "early_warning", "crisis", "recovery"]
    phase_probs = [0.35, 0.15, 0.20, 0.12, 0.08, 0.10]

    for run in range(n_runs):
        if run % 5 == 0:
            print(f"Run {run + 1}/{n_runs}...")

        seed = run * 17 + 999
        np.random.seed(seed)

        # Random phases
        phases = []
        for _ in range(np.random.randint(4, 8)):
            phase = np.random.choice(phase_options, p=phase_probs)
            duration = np.random.randint(40, 120)
            phases.append((phase, duration))

        prices, headlines = generate_scenario(phases, seed=seed)

        bot_p = RealTradingBot(detector, use_paradigm=True)
        result_p = bot_p.run(prices, headlines)
        mc_paradigm.append(result_p["return"])

        bot_t = RealTradingBot(detector, use_paradigm=False)
        result_t = bot_t.run(prices, headlines)
        mc_trad.append(result_t["return"])

        mc_bh.append((prices[-1] - prices[0]) / prices[0])

    # Results
    print(f"\n{'Strategy':<15} {'Mean':>10} {'Worst':>10} {'Best':>10} {'P(Loss)':>10}")
    print("-" * 60)

    for name, data in [("Paradigm", mc_paradigm), ("Traditional", mc_trad), ("Buy&Hold", mc_bh)]:
        print(f"{name:<15} {np.mean(data):>+9.1%} {np.min(data):>+9.1%} {np.max(data):>+9.1%} {np.mean([1 if r < 0 else 0 for r in data]):>9.1%}")

    # =========================================================================
    # PART 3: VERDICT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT (REAL DETECTOR)")
    print("=" * 70)

    paradigm_mean = np.mean(mc_paradigm)
    paradigm_worst = np.min(mc_paradigm)
    paradigm_p_loss = np.mean([1 if r < 0 else 0 for r in mc_paradigm])
    trad_mean = np.mean(mc_trad)

    print(f"\nParadigm Strategy (real detector):")
    print(f"  Mean return: {paradigm_mean:+.1%}")
    print(f"  Worst case: {paradigm_worst:+.1%}")
    print(f"  P(loss): {paradigm_p_loss:.1%}")
    print(f"  vs Traditional: {paradigm_mean - trad_mean:+.1%}")

    # Stress test wins
    stress_wins = sum(1 for r in stress_results.values() if r["paradigm"] > r["traditional"])
    print(f"  Stress wins: {stress_wins}/{len(stress_results)}")

    # Save results
    detector.save_caches()

    output = {
        "stress_tests": stress_results,
        "monte_carlo": {
            "n_runs": n_runs,
            "paradigm": {"mean": paradigm_mean, "worst": paradigm_worst, "p_loss": paradigm_p_loss},
            "traditional": {"mean": trad_mean},
        },
        "detector": "CachedParadigmDetector (real embeddings)",
        "timestamp": datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent / "stress_test_real_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
