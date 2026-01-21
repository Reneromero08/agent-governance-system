"""
STRESS TEST
===========

Rigorous testing of the Catalytic Market Bot.

This runs thousands of ticks across multiple market cycles to answer:
"Can I trust this bot with real money in a ralph loop?"

Tests:
1. Multi-year simulation with realistic market cycles
2. Monte Carlo: 100+ runs with different seeds
3. Worst-case scenarios: flash crashes, whipsaws, V-recoveries
4. Risk metrics: max drawdown, Sharpe, Sortino, win rate
5. Comparison vs baselines: buy-and-hold, always-defensive, random
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from market_bot import CatalyticMarketBot, Regime, Action
from data_sources import MockMarketScenario, MarketTick, RealtimeDataSimulator


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    worst_trade: float
    best_trade: float
    time_in_market: float  # % of time with position
    regime_accuracy: float  # How often regime matched outcome


def compute_risk_metrics(
    equity_curve: List[float],
    trades: List[Dict],
    initial_capital: float,
    n_days: int,
) -> RiskMetrics:
    """Compute comprehensive risk metrics."""
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Total and annualized return
    total_return = (equity[-1] - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = np.max(drawdown)

    # Sharpe ratio (assuming 0 risk-free rate)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns)
        sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    else:
        sortino = sharpe  # No downside = use sharpe

    # Trade statistics
    trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        avg_trade_pnl = np.mean(trade_pnls)
        worst_trade = min(trade_pnls)
        best_trade = max(trade_pnls)
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_pnl = 0
        worst_trade = 0
        best_trade = 0

    return RiskMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trades),
        avg_trade_pnl=avg_trade_pnl,
        worst_trade=worst_trade,
        best_trade=best_trade,
        time_in_market=0,  # TODO
        regime_accuracy=0,  # TODO
    )


class MarketGenerator:
    """Generate realistic multi-year market data."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.scenario_gen = MockMarketScenario(seed)

    def generate_multi_year(
        self,
        years: int = 5,
        ticks_per_day: int = 78,  # 6.5 hours * 12 (5-min bars)
        start_price: float = 100.0,
    ) -> Tuple[List[MarketTick], List[str]]:
        """
        Generate multi-year market data with realistic cycles.

        Returns ticks and a list of phase labels for analysis.
        """
        trading_days_per_year = 252
        total_days = years * trading_days_per_year
        total_ticks = total_days * ticks_per_day

        # Market cycle probabilities (realistic distribution)
        # Most time is stable, with occasional crises
        phase_probs = {
            "stable_bull": 0.35,
            "stable_bear": 0.15,
            "volatile": 0.25,
            "early_warning": 0.10,
            "crisis": 0.05,
            "recovery": 0.10,
        }

        # Generate phase sequence
        phases = []
        phase_labels = []
        remaining_ticks = total_ticks

        while remaining_ticks > 0:
            # Sample phase
            phase = self.rng.choice(
                list(phase_probs.keys()),
                p=list(phase_probs.values())
            )

            # Phase duration (geometric distribution)
            if phase == "crisis":
                # Crises are shorter but intense
                duration = int(self.rng.geometric(0.02) * ticks_per_day)
                duration = min(duration, 60 * ticks_per_day)  # Max 60 days
            elif phase in ["stable_bull", "stable_bear"]:
                # Trends last longer
                duration = int(self.rng.geometric(0.005) * ticks_per_day)
                duration = min(duration, 180 * ticks_per_day)  # Max 180 days
            else:
                duration = int(self.rng.geometric(0.01) * ticks_per_day)
                duration = min(duration, 90 * ticks_per_day)

            duration = min(duration, remaining_ticks)
            phases.append((phase, duration))
            phase_labels.extend([phase] * duration)
            remaining_ticks -= duration

        # Generate ticks
        ticks = self.scenario_gen.generate_multi_phase(phases, start_price)

        return ticks[:total_ticks], phase_labels[:total_ticks]

    def generate_stress_scenario(self, scenario_type: str, n_ticks: int = 1000) -> List[MarketTick]:
        """Generate specific stress scenarios."""
        if scenario_type == "flash_crash":
            # Normal -> sudden 20% drop -> immediate recovery
            return self.scenario_gen.generate_multi_phase([
                ("stable_bull", n_ticks // 3),
                ("crisis", n_ticks // 6),  # Sharp drop
                ("recovery", n_ticks // 2),  # V-recovery
            ])

        elif scenario_type == "slow_bleed":
            # Gradual decline over long period
            return self.scenario_gen.generate_multi_phase([
                ("stable_bear", n_ticks),
            ])

        elif scenario_type == "whipsaw":
            # Rapid regime changes (worst for trend-followers)
            phases = []
            for _ in range(20):
                phases.append((self.rng.choice(["stable_bull", "stable_bear", "volatile"]), n_ticks // 20))
            return self.scenario_gen.generate_multi_phase(phases)

        elif scenario_type == "2008_style":
            # Long bull -> warning -> severe crisis -> slow recovery
            return self.scenario_gen.generate_multi_phase([
                ("stable_bull", n_ticks // 4),
                ("early_warning", n_ticks // 8),
                ("crisis", n_ticks // 4),
                ("recovery", n_ticks // 4),
                ("stable_bull", n_ticks // 8),
            ])

        elif scenario_type == "covid_style":
            # Normal -> sudden crisis -> V-shaped recovery
            return self.scenario_gen.generate_multi_phase([
                ("stable_bull", n_ticks // 3),
                ("crisis", n_ticks // 10),
                ("recovery", n_ticks // 5),
                ("stable_bull", n_ticks // 3),
            ])

        else:
            raise ValueError(f"Unknown scenario: {scenario_type}")


def run_backtest(
    bot: CatalyticMarketBot,
    ticks: List[MarketTick],
    verbose: bool = False,
) -> Tuple[List[float], List[Dict], List[str]]:
    """
    Run backtest and return equity curve, trades, and decisions.
    """
    simulator = RealtimeDataSimulator(ticks)
    equity_curve = [bot.capital]
    all_trades = []
    decisions = []

    for i, tick in enumerate(simulator):
        history = simulator.get_history(lookback=20)
        if len(history) < 5:
            equity_curve.append(bot.capital)
            continue

        decision = bot.process_tick(
            price=tick.price,
            headlines=tick.headlines,
            prices_history=history,
        )
        decisions.append(decision.regime.value)

        trade = bot.execute_decision(decision, tick.price)
        if trade:
            all_trades.append(trade)

        # Update equity
        current_equity = bot.capital
        if bot.position:
            unrealized = (tick.price - bot.position.entry_price) * bot.position.size
            current_equity += unrealized
        equity_curve.append(current_equity)

        if verbose and i % 1000 == 0:
            print(f"Tick {i:6d} | ${tick.price:7.2f} | Equity: ${current_equity:10.2f}")

    return equity_curve, all_trades, decisions


def run_monte_carlo(
    n_runs: int = 100,
    years: int = 3,
    initial_capital: float = 100000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with different random seeds.
    """
    results = {
        "paradigm_aware": [],
        "buy_and_hold": [],
        "always_defensive": [],
    }

    for run in range(n_runs):
        seed = run * 17 + 42  # Different seed each run

        # Generate market data
        generator = MarketGenerator(seed=seed)
        ticks, phases = generator.generate_multi_year(years=years, ticks_per_day=12)  # Hourly for speed

        if verbose and run % 10 == 0:
            print(f"Run {run + 1}/{n_runs}...")

        # Strategy 1: Paradigm-aware bot
        bot_aware = CatalyticMarketBot(initial_capital=initial_capital)
        equity_aware, trades_aware, _ = run_backtest(bot_aware, ticks)
        results["paradigm_aware"].append({
            "final_equity": equity_aware[-1],
            "return": (equity_aware[-1] - initial_capital) / initial_capital,
            "max_dd": compute_max_drawdown(equity_aware),
            "trades": len(trades_aware),
        })

        # Strategy 2: Buy and hold
        start_price = ticks[0].price if ticks else 100
        end_price = ticks[-1].price if ticks else 100
        bh_return = (end_price - start_price) / start_price
        bh_equity = initial_capital * (1 + bh_return)
        # Compute B&H drawdown from price series
        prices = [t.price for t in ticks]
        bh_equity_curve = [initial_capital * p / start_price for p in prices]
        results["buy_and_hold"].append({
            "final_equity": bh_equity,
            "return": bh_return,
            "max_dd": compute_max_drawdown(bh_equity_curve),
            "trades": 1,
        })

        # Strategy 3: Always defensive (20% position max)
        bot_defensive = CatalyticMarketBot(initial_capital=initial_capital)
        bot_defensive.regime_params[Regime.STABLE]["max_position"] = 0.2
        bot_defensive.regime_params[Regime.TRANSITIONAL]["max_position"] = 0.1
        bot_defensive.regime_params[Regime.SHIFT]["max_position"] = 0.0
        equity_def, trades_def, _ = run_backtest(bot_defensive, ticks)
        results["always_defensive"].append({
            "final_equity": equity_def[-1],
            "return": (equity_def[-1] - initial_capital) / initial_capital,
            "max_dd": compute_max_drawdown(equity_def),
            "trades": len(trades_def),
        })

    return results


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown from equity curve."""
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return float(np.max(drawdown))


def analyze_monte_carlo(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Analyze Monte Carlo results."""
    analysis = {}

    for strategy, runs in results.items():
        returns = [r["return"] for r in runs]
        drawdowns = [r["max_dd"] for r in runs]

        analysis[strategy] = {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "median_return": np.median(returns),
            "worst_return": np.min(returns),
            "best_return": np.max(returns),
            "mean_max_dd": np.mean(drawdowns),
            "worst_max_dd": np.max(drawdowns),
            "prob_loss": np.mean([1 if r < 0 else 0 for r in returns]),
            "prob_20pct_loss": np.mean([1 if r < -0.20 else 0 for r in returns]),
            "sharpe_approx": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        }

    return analysis


def run_stress_tests(initial_capital: float = 100000) -> Dict[str, Dict]:
    """Run all stress test scenarios."""
    scenarios = ["flash_crash", "slow_bleed", "whipsaw", "2008_style", "covid_style"]
    results = {}

    generator = MarketGenerator(seed=42)

    for scenario in scenarios:
        print(f"\nStress Test: {scenario}")
        ticks = generator.generate_stress_scenario(scenario, n_ticks=2000)

        # Paradigm-aware
        bot = CatalyticMarketBot(initial_capital=initial_capital)
        equity, trades, decisions = run_backtest(bot, ticks)

        # Buy and hold comparison
        start_price = ticks[0].price
        end_price = ticks[-1].price
        bh_return = (end_price - start_price) / start_price

        paradigm_return = (equity[-1] - initial_capital) / initial_capital
        paradigm_dd = compute_max_drawdown(equity)

        results[scenario] = {
            "paradigm_return": paradigm_return,
            "paradigm_max_dd": paradigm_dd,
            "bh_return": bh_return,
            "paradigm_advantage": paradigm_return - bh_return,
            "trades": len(trades),
            "regime_distribution": {
                "STABLE": decisions.count("STABLE") / len(decisions) if decisions else 0,
                "SHIFT": decisions.count("SHIFT") / len(decisions) if decisions else 0,
                "TRANSITIONAL": decisions.count("TRANSITIONAL") / len(decisions) if decisions else 0,
            },
        }

        print(f"  Paradigm: {paradigm_return:+.2%} (DD: {paradigm_dd:.2%})")
        print(f"  Buy&Hold: {bh_return:+.2%}")
        print(f"  Advantage: {results[scenario]['paradigm_advantage']:+.2%}")

    return results


def main():
    print("=" * 70)
    print("CATALYTIC MARKET BOT - COMPREHENSIVE STRESS TEST")
    print("=" * 70)
    print("\nThis test will take several minutes...")
    print("Testing if you can trust this bot with real money.\n")

    # 1. Stress Test Scenarios
    print("\n" + "=" * 70)
    print("PART 1: STRESS TEST SCENARIOS")
    print("=" * 70)
    stress_results = run_stress_tests()

    # 2. Monte Carlo Simulation
    print("\n" + "=" * 70)
    print("PART 2: MONTE CARLO SIMULATION (100 runs x 3 years)")
    print("=" * 70)
    mc_results = run_monte_carlo(n_runs=100, years=3, verbose=True)
    mc_analysis = analyze_monte_carlo(mc_results)

    # 3. Summary Report
    print("\n" + "=" * 70)
    print("FINAL REPORT: CAN YOU TRUST THIS BOT?")
    print("=" * 70)

    print("\n--- STRESS TEST RESULTS ---")
    print(f"{'Scenario':<15} {'Paradigm':>12} {'Buy&Hold':>12} {'Advantage':>12}")
    print("-" * 55)
    for scenario, data in stress_results.items():
        print(f"{scenario:<15} {data['paradigm_return']:>+11.2%} {data['bh_return']:>+11.2%} {data['paradigm_advantage']:>+11.2%}")

    print("\n--- MONTE CARLO RESULTS (100 runs x 3 years) ---")
    print(f"{'Strategy':<20} {'Mean Return':>12} {'Worst':>12} {'Max DD':>12} {'P(Loss)':>10}")
    print("-" * 70)
    for strategy, stats in mc_analysis.items():
        print(f"{strategy:<20} {stats['mean_return']:>+11.2%} {stats['worst_return']:>+11.2%} {stats['mean_max_dd']:>11.2%} {stats['prob_loss']:>9.1%}")

    # Risk Assessment
    print("\n--- RISK ASSESSMENT ---")
    paradigm_stats = mc_analysis["paradigm_aware"]
    bh_stats = mc_analysis["buy_and_hold"]

    # Key risk metrics
    prob_ruin = paradigm_stats["prob_20pct_loss"]
    worst_case = paradigm_stats["worst_return"]
    mean_dd = paradigm_stats["mean_max_dd"]

    print(f"Probability of >20% loss: {prob_ruin:.1%}")
    print(f"Worst case return: {worst_case:+.2%}")
    print(f"Average max drawdown: {mean_dd:.2%}")
    print(f"Sharpe ratio (approx): {paradigm_stats['sharpe_approx']:.2f}")

    # Verdict
    print("\n--- VERDICT ---")
    if prob_ruin < 0.10 and worst_case > -0.40 and paradigm_stats["mean_return"] > bh_stats["mean_return"]:
        print("CAUTIOUSLY OPTIMISTIC")
        print("- Low probability of severe loss")
        print("- Outperforms buy-and-hold on average")
        print("- But: This is backtested on SYNTHETIC data")
        print("- Real markets have correlations and patterns not captured here")
    else:
        print("CAUTION ADVISED")
        print("- Risk metrics suggest potential for significant losses")
        print("- Further testing with real market data recommended")

    print("\n--- CRITICAL WARNINGS ---")
    print("1. This is SYNTHETIC data - real markets are different")
    print("2. Paradigm detection has latency - real shifts may be missed")
    print("3. Options have additional risks (gamma, theta, vega)")
    print("4. Always use position limits and stop losses")
    print("5. NEVER trade more than you can afford to lose")

    # Save results
    output = {
        "stress_tests": stress_results,
        "monte_carlo": mc_analysis,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent / "stress_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
