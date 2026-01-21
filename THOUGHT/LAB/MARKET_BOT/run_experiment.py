"""
RUN EXPERIMENT
==============

Demo runner for the Catalytic Market Bot experiment.

This runs the bot through various market scenarios and shows
how paradigm-aware context affects decision making.
"""

import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict
import sys
from pathlib import Path

# Add local imports
sys.path.insert(0, str(Path(__file__).parent))

from market_bot import CatalyticMarketBot, Regime, Action
from data_sources import MockMarketScenario, MarketTick, RealtimeDataSimulator


def run_single_scenario(bot: CatalyticMarketBot, ticks: List[MarketTick], verbose: bool = True) -> Dict:
    """Run bot through a single scenario."""
    simulator = RealtimeDataSimulator(ticks)

    results = {
        "decisions": [],
        "trades": [],
        "regime_changes": [],
        "final_status": None,
    }

    last_regime = None

    for i, tick in enumerate(simulator):
        # Get price history
        history = simulator.get_history(lookback=20)
        if len(history) < 5:
            # Need minimum history
            continue

        # Process tick
        decision = bot.process_tick(
            price=tick.price,
            headlines=tick.headlines,
            prices_history=history,
        )

        # Track regime changes
        if last_regime != decision.regime:
            results["regime_changes"].append({
                "tick": i,
                "from": last_regime.value if last_regime else "START",
                "to": decision.regime.value,
                "price": tick.price,
            })
            last_regime = decision.regime

            if verbose:
                print(f"\n>>> REGIME CHANGE: {results['regime_changes'][-1]['from']} -> {decision.regime.value} @ ${tick.price:.2f}")

        # Execute decision
        trade = bot.execute_decision(decision, tick.price)

        results["decisions"].append({
            "tick": i,
            "regime": decision.regime.value,
            "action": decision.action.value,
            "confidence": decision.confidence,
        })

        if trade:
            results["trades"].append(trade)
            if verbose:
                print(f"  TRADE: {trade['action']} @ ${trade['price']:.2f}")

        # Periodic status
        if verbose and i % 20 == 0:
            status = bot.get_status(tick.price)
            print(f"Tick {i:3d} | ${tick.price:7.2f} | {decision.regime.value:12s} | {decision.action.value:6s} | P&L: ${status['total_pnl']:+8.2f}")

    # Final status
    final_price = ticks[-1].price if ticks else 0
    results["final_status"] = bot.get_status(final_price)

    return results


def run_full_cycle_experiment(verbose: bool = True):
    """
    Run a full market cycle experiment:
    Bull -> Warning -> Crisis -> Recovery

    This demonstrates how the bot adapts to different regimes.
    """
    print("=" * 70)
    print("CATALYTIC MARKET BOT - Full Cycle Experiment")
    print("=" * 70)

    # Generate multi-phase market data
    generator = MockMarketScenario(seed=42)
    ticks = generator.generate_multi_phase([
        ("stable_bull", 30),      # Bull market
        ("early_warning", 15),    # Warning signs
        ("crisis", 25),           # Crisis
        ("recovery", 30),         # Recovery
    ], start_price=100.0)

    print(f"\nGenerated {len(ticks)} market ticks")
    print("Phases: Bull(30) -> Warning(15) -> Crisis(25) -> Recovery(30)")

    # Initialize bot
    bot = CatalyticMarketBot(initial_capital=100000)

    # Run experiment
    print("\n" + "-" * 70)
    print("RUNNING EXPERIMENT")
    print("-" * 70)

    results = run_single_scenario(bot, ticks, verbose=verbose)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    status = results["final_status"]
    print(f"\nFinal Capital: ${status['capital']:.2f}")
    print(f"Total P&L: ${status['total_pnl']:.2f} ({status['total_return_pct']:.2f}%)")
    print(f"Total Trades: {len(results['trades'])}")

    print(f"\nRegime Changes:")
    for change in results["regime_changes"]:
        print(f"  Tick {change['tick']:3d}: {change['from']:12s} -> {change['to']:12s} @ ${change['price']:.2f}")

    # Analyze decisions by regime
    print(f"\nDecisions by Regime:")
    regime_counts = {}
    for d in results["decisions"]:
        regime = d["regime"]
        action = d["action"]
        if regime not in regime_counts:
            regime_counts[regime] = {"total": 0, "actions": {}}
        regime_counts[regime]["total"] += 1
        regime_counts[regime]["actions"][action] = regime_counts[regime]["actions"].get(action, 0) + 1

    for regime, data in regime_counts.items():
        print(f"  {regime}: {data['total']} decisions")
        for action, count in data["actions"].items():
            print(f"    - {action}: {count}")

    return results


def run_comparison_experiment():
    """
    Compare bot performance with and without paradigm detection.

    This shows the value of regime-aware trading.
    """
    print("=" * 70)
    print("COMPARISON: Paradigm-Aware vs Traditional")
    print("=" * 70)

    # Generate crisis scenario
    generator = MockMarketScenario(seed=123)
    ticks = generator.generate_multi_phase([
        ("stable_bull", 20),
        ("crisis", 40),
        ("recovery", 20),
    ], start_price=100.0)

    print(f"\nScenario: Bull(20) -> CRISIS(40) -> Recovery(20)")
    print(f"This tests survival during a sharp market decline.\n")

    # Bot 1: Paradigm-aware (normal)
    print("-" * 35)
    print("BOT A: PARADIGM-AWARE")
    print("-" * 35)
    bot_aware = CatalyticMarketBot(initial_capital=100000)
    results_aware = run_single_scenario(bot_aware, ticks, verbose=False)
    status_aware = results_aware["final_status"]
    print(f"Final P&L: ${status_aware['total_pnl']:.2f} ({status_aware['total_return_pct']:.2f}%)")
    print(f"Trades: {len(results_aware['trades'])}")

    # Bot 2: Always-on (ignores regime - buys dips)
    print("\n" + "-" * 35)
    print("BOT B: TRADITIONAL (buys oversold)")
    print("-" * 35)

    # Simulate traditional bot behavior
    bot_traditional = CatalyticMarketBot(initial_capital=100000)
    # Override regime detection to always return STABLE
    original_detect = bot_traditional.detect_regime
    bot_traditional.detect_regime = lambda x: (Regime.STABLE, 0.0, [("Mock", 0.5)])

    results_traditional = run_single_scenario(bot_traditional, ticks, verbose=False)
    status_traditional = results_traditional["final_status"]
    print(f"Final P&L: ${status_traditional['total_pnl']:.2f} ({status_traditional['total_return_pct']:.2f}%)")
    print(f"Trades: {len(results_traditional['trades'])}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Paradigm-Aware':>15} {'Traditional':>15}")
    print("-" * 55)
    print(f"{'Final P&L':<25} ${status_aware['total_pnl']:>14.2f} ${status_traditional['total_pnl']:>14.2f}")
    print(f"{'Return %':<25} {status_aware['total_return_pct']:>14.2f}% {status_traditional['total_return_pct']:>14.2f}%")
    print(f"{'Trades':<25} {len(results_aware['trades']):>15} {len(results_traditional['trades']):>15}")

    diff = status_aware['total_pnl'] - status_traditional['total_pnl']
    print(f"\n{'Paradigm Advantage':<25} ${diff:>+14.2f}")

    if diff > 0:
        print("\n>>> PARADIGM-AWARE BOT OUTPERFORMED <<<")
        print("The regime detector protected capital during the crisis.")
    else:
        print("\n>>> TRADITIONAL BOT OUTPERFORMED <<<")
        print("(This can happen in V-shaped recoveries)")


def main():
    parser = argparse.ArgumentParser(description="Catalytic Market Bot Experiment")
    parser.add_argument(
        "--scenario",
        choices=["full_cycle", "comparison", "quick"],
        default="full_cycle",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if args.scenario == "full_cycle":
        run_full_cycle_experiment(verbose=not args.quiet)
    elif args.scenario == "comparison":
        run_comparison_experiment()
    elif args.scenario == "quick":
        print("Quick test - minimal output")
        generator = MockMarketScenario(seed=42)
        ticks = generator.generate("crisis", n_ticks=20)
        bot = CatalyticMarketBot(initial_capital=100000)
        results = run_single_scenario(bot, ticks, verbose=False)
        print(f"Result: ${results['final_status']['total_pnl']:.2f}")


if __name__ == "__main__":
    main()
