"""
HISTORICAL BACKTEST
===================

Tests the Psychohistory bot on REAL historical data.

The bot runs through history WITHOUT knowing the future.
We know the crashes happened - the bot doesn't.
We'll see if the formula would have protected capital.

Key events:
- COVID crash 2020 (34% drawdown in 1 month)
- 2008 Financial Crisis (57% drawdown over 17 months)
- Flash Crash 2010 (9% intraday)
- And more...
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from signal_vocabulary import SignalState, AssetClass
from real_data_ingest import RealDataFetcher, MarketData, HISTORICAL_EVENTS
from signal_extractor import SignalExtractor
from prime_radiant import PrimeRadiant
from formula_executor import MarketFormulaExecutor
from seldon_gate import SeldonGate, GateTier, AlphaWarningLevel
from psychohistory_bot import PsychohistoryBot, BotConfig


# =============================================================================
# BACKTEST RESULTS
# =============================================================================

@dataclass
class BacktestResult:
    """Result of a historical backtest."""
    event_name: str
    event_description: str
    symbol: str

    # Bot performance
    bot_return: float
    bot_max_drawdown: float
    bot_trades: int
    bot_time_in_market: float  # Fraction of time with position

    # Buy & Hold performance
    bh_return: float
    bh_max_drawdown: float

    # Key dates
    crash_start: str
    crash_end: str
    bot_exit_date: Optional[str]  # When bot exited (if before crash)
    bot_reentry_date: Optional[str]  # When bot re-entered

    # Gate activity
    gate_closures: int  # Number of times gate closed
    alpha_warnings: int  # Number of alpha drift warnings
    first_warning_date: Optional[str]

    # Verdict
    protected_capital: bool  # Did bot have lower max DD than B&H?
    early_warning: bool  # Did bot warn before crash?


# =============================================================================
# BACKTESTER
# =============================================================================

class HistoricalBacktester:
    """
    Runs Psychohistory bot through historical crises.

    The bot sees ONLY past data at each step.
    It cannot peek at the future.
    """

    def __init__(self):
        """Initialize backtester."""
        self.fetcher = RealDataFetcher()
        self.extractor = SignalExtractor()

    def run_event(
        self,
        event_name: str,
        symbol: str = "SPY",
        initial_capital: float = 100000,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest on a historical event.

        Args:
            event_name: Key from HISTORICAL_EVENTS
            symbol: Ticker symbol
            initial_capital: Starting capital
            verbose: Print progress

        Returns:
            BacktestResult with performance metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"BACKTEST: {event_name}")
            print(f"{'='*60}")

        # Fetch data
        data, event = self.fetcher.fetch_event(event_name, symbol, buffer_days=60)

        if verbose:
            print(f"Event: {event['description']}")
            print(f"Period: {data.start_date} to {data.end_date}")
            print(f"Bars: {len(data.bars)}")
            print(f"Expected drawdown: {event['max_drawdown']:.1%}")

        # Initialize bot with fresh state
        config = BotConfig(
            initial_capital=initial_capital,
            lookback_window=10,
            alpha_window=20,
        )
        bot = PsychohistoryBot(config)

        # Track metrics
        prices = []
        equity_curve = []
        gate_statuses = []
        alpha_warnings_list = []
        positions_held = []

        bot_exit_date = None
        bot_reentry_date = None
        first_warning_date = None
        was_in_position = False

        # Extract signals and run bot through history
        states = self.extractor.extract_sequence(data, start_index=30)

        if verbose:
            print(f"\nRunning {len(states)} steps through history...")

        for i, state in enumerate(states):
            bar_idx = i + 30  # Offset for initial warmup
            price = data.bars[bar_idx].close
            timestamp = data.bars[bar_idx].timestamp
            prices.append(price)

            # Simple direction signal from recent momentum
            if len(prices) > 5:
                recent_return = (prices[-1] - prices[-5]) / prices[-5]
                direction = 1 if recent_return > 0.005 else (-1 if recent_return < -0.005 else 0)
            else:
                direction = 0

            # Bot makes decision (this internally calls observe and gate assessment)
            trade = bot.decide(state, price, direction)
            bot.record_equity({state.asset: price})

            # Get gate result from bot's observe (already called in decide)
            # Use assess directly for tracking
            if len(bot.vector_history) > 0:
                current_vec = bot.vector_history[-1]
                context_vecs = bot.vector_history[-bot.config.lookback_window-1:-1] if len(bot.vector_history) > 1 else []
                formula_result = bot.executor.compute_R(current_vec, context_vecs)
                alpha_result = bot.executor.compute_alpha(bot.vector_history[-bot.config.alpha_window:] if len(bot.vector_history) >= 3 else bot.vector_history)
                gate_result = bot.gate.assess(
                    R=formula_result.R,
                    alpha=alpha_result.alpha,
                    Df=alpha_result.Df,
                    requested_tier=bot.config.default_tier,
                )
                gate_statuses.append(gate_result.gate.status)

                # Track alpha warnings
                if gate_result.drift.warning_level != AlphaWarningLevel.NONE:
                    alpha_warnings_list.append((timestamp, gate_result.drift.warning_level.name))
                    if first_warning_date is None:
                        first_warning_date = timestamp
            else:
                gate_statuses.append("OPEN")

            # Track position changes
            in_position = len(bot.positions) > 0
            positions_held.append(in_position)

            if was_in_position and not in_position:
                bot_exit_date = timestamp
            if not was_in_position and in_position:
                bot_reentry_date = timestamp
            was_in_position = in_position

            # Progress
            if verbose and i % 50 == 0:
                gate_status = gate_statuses[-1] if gate_statuses else "N/A"
                print(f"  Step {i}: {timestamp} | ${price:.2f} | Gate: {gate_status} | Pos: {in_position}")

        # Compute metrics
        bot_return = bot.get_returns()
        bot_max_dd = bot.get_max_drawdown()

        # Buy and hold metrics
        bh_return = (prices[-1] - prices[0]) / prices[0]
        bh_equity = [prices[0]]
        for p in prices[1:]:
            bh_equity.append(p)
        bh_equity = np.array(bh_equity)
        bh_peak = np.maximum.accumulate(bh_equity)
        bh_dd = (bh_peak - bh_equity) / bh_peak
        bh_max_dd = float(np.max(bh_dd))

        # Gate closures
        gate_closures = sum(1 for i in range(1, len(gate_statuses))
                          if gate_statuses[i] == "CLOSED" and gate_statuses[i-1] != "CLOSED")

        # Time in market
        time_in_market = sum(positions_held) / len(positions_held) if positions_held else 0

        # Did bot warn before crash?
        crash_start = event["crash_start"]
        early_warning = first_warning_date is not None and first_warning_date < crash_start

        # Did bot protect capital?
        protected = bot_max_dd < bh_max_dd

        result = BacktestResult(
            event_name=event_name,
            event_description=event["description"],
            symbol=symbol,
            bot_return=bot_return,
            bot_max_drawdown=bot_max_dd,
            bot_trades=len(bot.trades),
            bot_time_in_market=time_in_market,
            bh_return=bh_return,
            bh_max_drawdown=bh_max_dd,
            crash_start=crash_start,
            crash_end=event["crash_end"],
            bot_exit_date=bot_exit_date,
            bot_reentry_date=bot_reentry_date,
            gate_closures=gate_closures,
            alpha_warnings=len(alpha_warnings_list),
            first_warning_date=first_warning_date,
            protected_capital=protected,
            early_warning=early_warning,
        )

        if verbose:
            self._print_result(result)

        return result

    def run_all_events(
        self,
        symbol: str = "SPY",
        initial_capital: float = 100000,
        verbose: bool = True
    ) -> List[BacktestResult]:
        """Run backtest on all historical events."""
        results = []

        for event_name in HISTORICAL_EVENTS.keys():
            try:
                result = self.run_event(
                    event_name, symbol, initial_capital,
                    verbose=verbose
                )
                results.append(result)
            except Exception as e:
                import traceback
                print(f"ERROR on {event_name}: {e}")
                traceback.print_exc()

        return results

    def _print_result(self, result: BacktestResult):
        """Print backtest result summary."""
        print(f"\n{'='*60}")
        print(f"RESULT: {result.event_name}")
        print(f"{'='*60}")

        print(f"\n--- Performance ---")
        print(f"  Bot Return: {result.bot_return:+.1%}")
        print(f"  Bot Max DD: {result.bot_max_drawdown:.1%}")
        print(f"  B&H Return: {result.bh_return:+.1%}")
        print(f"  B&H Max DD: {result.bh_max_drawdown:.1%}")

        print(f"\n--- Activity ---")
        print(f"  Trades: {result.bot_trades}")
        print(f"  Time in market: {result.bot_time_in_market:.1%}")
        print(f"  Gate closures: {result.gate_closures}")
        print(f"  Alpha warnings: {result.alpha_warnings}")

        print(f"\n--- Key Dates ---")
        print(f"  Crash started: {result.crash_start}")
        print(f"  Crash ended: {result.crash_end}")
        if result.first_warning_date:
            print(f"  First warning: {result.first_warning_date}")
        if result.bot_exit_date:
            print(f"  Bot exited: {result.bot_exit_date}")

        print(f"\n--- Verdict ---")
        if result.protected_capital:
            print(f"  [PASS] Protected capital ({result.bot_max_drawdown:.1%} vs {result.bh_max_drawdown:.1%})")
        else:
            print(f"  [FAIL] Did not protect capital ({result.bot_max_drawdown:.1%} vs {result.bh_max_drawdown:.1%})")

        if result.early_warning:
            print(f"  [PASS] Early warning before crash")
        else:
            print(f"  [WARN] No early warning detected")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(results: List[BacktestResult]) -> Dict:
    """Generate summary report from backtest results."""
    if not results:
        return {
            "n_events": 0,
            "n_protected": 0,
            "protection_rate": 0,
            "n_warned": 0,
            "warning_rate": 0,
            "avg_bot_return": 0,
            "avg_bh_return": 0,
            "avg_bot_dd": 0,
            "avg_bh_dd": 0,
            "worst_bot_dd": 0,
            "worst_bh_dd": 0,
            "dd_improvement": 0,
        }

    n_events = len(results)
    n_protected = sum(1 for r in results if r.protected_capital)
    n_warned = sum(1 for r in results if r.early_warning)

    avg_bot_return = np.mean([r.bot_return for r in results])
    avg_bh_return = np.mean([r.bh_return for r in results])
    avg_bot_dd = np.mean([r.bot_max_drawdown for r in results])
    avg_bh_dd = np.mean([r.bh_max_drawdown for r in results])

    worst_bot_dd = max(r.bot_max_drawdown for r in results)
    worst_bh_dd = max(r.bh_max_drawdown for r in results)

    return {
        "n_events": n_events,
        "n_protected": n_protected,
        "protection_rate": n_protected / n_events if n_events > 0 else 0,
        "n_warned": n_warned,
        "warning_rate": n_warned / n_events if n_events > 0 else 0,
        "avg_bot_return": avg_bot_return,
        "avg_bh_return": avg_bh_return,
        "avg_bot_dd": avg_bot_dd,
        "avg_bh_dd": avg_bh_dd,
        "worst_bot_dd": worst_bot_dd,
        "worst_bh_dd": worst_bh_dd,
        "dd_improvement": avg_bh_dd - avg_bot_dd,
    }


def print_summary_report(results: List[BacktestResult]):
    """Print summary report."""
    if not results:
        print("\n" + "=" * 70)
        print("HISTORICAL BACKTEST SUMMARY")
        print("=" * 70)
        print("\n[WARN] No successful backtests to report.")
        return

    summary = generate_summary_report(results)

    print("\n" + "=" * 70)
    print("HISTORICAL BACKTEST SUMMARY")
    print("=" * 70)

    print(f"\n--- Event Results ---")
    print(f"{'Event':<25} {'Bot DD':>10} {'B&H DD':>10} {'Protected':>12} {'Warning':>10}")
    print("-" * 70)

    for r in results:
        protected = "YES" if r.protected_capital else "NO"
        warned = "YES" if r.early_warning else "NO"
        print(f"{r.event_name:<25} {r.bot_max_drawdown:>9.1%} {r.bh_max_drawdown:>9.1%} {protected:>12} {warned:>10}")

    print("\n--- Aggregate Metrics ---")
    print(f"  Events tested: {summary['n_events']}")
    print(f"  Capital protected: {summary['n_protected']}/{summary['n_events']} ({summary['protection_rate']:.0%})")
    print(f"  Early warnings: {summary['n_warned']}/{summary['n_events']} ({summary['warning_rate']:.0%})")

    print(f"\n--- Drawdown Comparison ---")
    print(f"  Avg Bot Max DD: {summary['avg_bot_dd']:.1%}")
    print(f"  Avg B&H Max DD: {summary['avg_bh_dd']:.1%}")
    print(f"  Improvement: {summary['dd_improvement']:+.1%}")

    print(f"\n--- Worst Case ---")
    print(f"  Worst Bot DD: {summary['worst_bot_dd']:.1%}")
    print(f"  Worst B&H DD: {summary['worst_bh_dd']:.1%}")

    print(f"\n--- Verdict ---")
    if summary['protection_rate'] >= 0.7:
        print("  [PASS] Bot protected capital in majority of crises")
    else:
        print("  [WARN] Bot needs improvement")

    if summary['dd_improvement'] > 0.1:
        print(f"  [PASS] Average drawdown improvement of {summary['dd_improvement']:.1%}")
    else:
        print("  [WARN] Minimal drawdown improvement")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run historical backtests on all events."""
    print("=" * 70)
    print("PSYCHOHISTORY BOT - HISTORICAL BACKTEST")
    print("The bot doesn't know the crashes are coming. We do.")
    print("=" * 70)

    backtester = HistoricalBacktester()

    # Run all events
    results = backtester.run_all_events(symbol="SPY", verbose=True)

    # Print summary
    print_summary_report(results)

    # Save results
    output_path = Path(__file__).parent / "historical_backtest_results.json"
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "event_name": r.event_name,
                "event_description": r.event_description,
                "bot_return": r.bot_return,
                "bot_max_drawdown": r.bot_max_drawdown,
                "bh_return": r.bh_return,
                "bh_max_drawdown": r.bh_max_drawdown,
                "protected_capital": r.protected_capital,
                "early_warning": r.early_warning,
                "first_warning_date": r.first_warning_date,
                "crash_start": r.crash_start,
            }
            for r in results
        ],
        "summary": generate_summary_report(results),
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
