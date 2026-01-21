"""
PENNY COMPOUNDER BACKTEST
=========================

Test the formula on historical penny stock data.
Does it actually work? Let's find out.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Need yfinance: pip install yfinance")
    sys.exit(1)

from penny_compounder import PennyAnalyzer, CONFIG


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str
    pnl_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    symbol: str
    period: str
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_return: float
    max_drawdown: float
    trades: List[Trade]


def backtest_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100.0,
    verbose: bool = True
) -> BacktestResult:
    """
    Backtest the penny compounder strategy on a single symbol.

    Strategy:
    - Buy when: R > threshold AND trend > 0 AND momentum > 0
    - Sell when: +10% profit OR -3% loss OR R drops below 0.3
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"BACKTEST: {symbol} ({start_date} to {end_date})")
        print(f"{'='*60}")

    # Fetch data
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date)

    if len(hist) < 60:
        print(f"  [SKIP] Not enough data ({len(hist)} bars)")
        return None

    prices = hist['Close'].tolist()
    volumes = hist['Volume'].tolist()
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]

    if verbose:
        print(f"  Data: {len(prices)} days")
        print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")

    # Initialize
    analyzer = PennyAnalyzer()

    capital = initial_capital
    position = None  # {shares, entry_price, entry_date, entry_idx}
    trades = []
    equity_curve = [capital]

    # Simulate day by day
    for i in range(30, len(prices)):
        current_price = prices[i]
        current_date = dates[i]

        # Build data dict for analysis
        data = {
            "symbol": symbol,
            "prices": prices[:i+1],
            "volumes": volumes[:i+1],
            "dates": dates[:i+1],
            "current_price": current_price,
            "avg_volume": np.mean(volumes[max(0,i-20):i+1]),
        }

        # Analyze
        signal = analyzer.analyze_stock(data)

        if signal is None:
            equity_curve.append(capital + (position["shares"] * current_price if position else 0))
            continue

        # Check exits first
        if position:
            pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
            holding_days = i - position["entry_idx"]

            exit_reason = None

            # Take profit
            if pnl_pct >= CONFIG["take_profit_pct"]:
                exit_reason = "TAKE_PROFIT"
            # Stop loss
            elif pnl_pct <= -CONFIG["stop_loss_pct"]:
                exit_reason = "STOP_LOSS"
            # Trend dying
            elif signal.R < 0.3:
                exit_reason = "TREND_DYING"
            # Max holding time (20 days)
            elif holding_days >= 20:
                exit_reason = "MAX_HOLD"

            if exit_reason:
                # Exit position
                proceeds = position["shares"] * current_price
                capital += proceeds

                trade = Trade(
                    symbol=symbol,
                    entry_date=position["entry_date"],
                    entry_price=position["entry_price"],
                    exit_date=current_date,
                    exit_price=current_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                )
                trades.append(trade)

                if verbose and len(trades) <= 20:
                    status = "WIN" if pnl_pct > 0 else "LOSS"
                    print(f"  {current_date}: EXIT ({exit_reason}) {pnl_pct:+.1%} [{status}]")

                position = None

        # Check entries
        if position is None and signal.action == "BUY":
            # Enter position (use all capital for simplicity)
            shares = capital / current_price
            position = {
                "shares": shares,
                "entry_price": current_price,
                "entry_date": current_date,
                "entry_idx": i,
            }
            capital = 0  # All in position

            if verbose and len(trades) < 20:
                print(f"  {current_date}: BUY @ ${current_price:.2f} (R={signal.R:.1f})")

        # Update equity
        equity = capital + (position["shares"] * current_price if position else 0)
        equity_curve.append(equity)

    # Close any open position at end
    if position:
        final_price = prices[-1]
        pnl_pct = (final_price - position["entry_price"]) / position["entry_price"]
        capital += position["shares"] * final_price

        trade = Trade(
            symbol=symbol,
            entry_date=position["entry_date"],
            entry_price=position["entry_price"],
            exit_date=dates[-1],
            exit_price=final_price,
            exit_reason="END_OF_TEST",
            pnl_pct=pnl_pct,
            holding_days=len(prices) - 1 - position["entry_idx"],
        )
        trades.append(trade)

    # Calculate metrics
    n_trades = len(trades)
    if n_trades == 0:
        if verbose:
            print(f"  [RESULT] No trades generated")
        return BacktestResult(
            symbol=symbol,
            period=f"{start_date} to {end_date}",
            n_trades=0,
            n_wins=0,
            n_losses=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            total_return=0,
            max_drawdown=0,
            trades=[],
        )

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

    total_return = (capital - initial_capital) / initial_capital

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    result = BacktestResult(
        symbol=symbol,
        period=f"{start_date} to {end_date}",
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_return=total_return,
        max_drawdown=max_dd,
        trades=trades,
    )

    if verbose:
        print(f"\n  --- RESULTS ---")
        print(f"  Trades: {n_trades} ({n_wins} wins, {n_losses} losses)")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Win: {avg_win:+.1%}")
        print(f"  Avg Loss: {avg_loss:+.1%}")
        print(f"  Total Return: {total_return:+.1%}")
        print(f"  Max Drawdown: {max_dd:.1%}")
        print(f"  Final Capital: ${capital:.2f} (started ${initial_capital:.2f})")

    return result


def run_full_backtest():
    """Run backtest across multiple symbols and time periods."""

    # Test symbols - stocks that have penny stock history
    symbols = ["SNDL", "AMC", "BB", "NOK", "F", "PLUG", "FCEL"]

    # Test periods
    periods = [
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2023-01-01", "2024-12-31", "2023-2024"),
    ]

    print("=" * 70)
    print("PENNY COMPOUNDER - HISTORICAL BACKTEST")
    print("=" * 70)
    print(f"\nStrategy: Buy on high R + positive trend")
    print(f"Take profit: {CONFIG['take_profit_pct']:.0%}")
    print(f"Stop loss: {CONFIG['stop_loss_pct']:.0%}")
    print(f"Min R threshold: {CONFIG['min_R_threshold']}")

    all_results = []

    for start, end, period_name in periods:
        print(f"\n\n{'#'*70}")
        print(f"# PERIOD: {period_name}")
        print(f"{'#'*70}")

        period_results = []

        for symbol in symbols:
            try:
                result = backtest_symbol(symbol, start, end, verbose=True)
                if result and result.n_trades > 0:
                    period_results.append(result)
                    all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")

        # Period summary
        if period_results:
            print(f"\n{'='*60}")
            print(f"PERIOD SUMMARY: {period_name}")
            print(f"{'='*60}")

            total_trades = sum(r.n_trades for r in period_results)
            total_wins = sum(r.n_wins for r in period_results)
            avg_return = np.mean([r.total_return for r in period_results])
            avg_win_rate = np.mean([r.win_rate for r in period_results])

            print(f"  Symbols tested: {len(period_results)}")
            print(f"  Total trades: {total_trades}")
            print(f"  Overall win rate: {total_wins/total_trades:.1%}" if total_trades > 0 else "  No trades")
            print(f"  Avg return per symbol: {avg_return:+.1%}")

    # Grand summary
    if all_results:
        print(f"\n\n{'#'*70}")
        print("# GRAND SUMMARY")
        print(f"{'#'*70}")

        total_trades = sum(r.n_trades for r in all_results)
        total_wins = sum(r.n_wins for r in all_results)
        total_losses = sum(r.n_losses for r in all_results)

        all_trades = []
        for r in all_results:
            all_trades.extend(r.trades)

        if all_trades:
            wins = [t for t in all_trades if t.pnl_pct > 0]
            losses = [t for t in all_trades if t.pnl_pct <= 0]

            print(f"\n  Total trades across all tests: {total_trades}")
            print(f"  Wins: {total_wins} | Losses: {total_losses}")
            print(f"  Overall Win Rate: {total_wins/total_trades:.1%}")
            print(f"  Average Win: {np.mean([t.pnl_pct for t in wins]):+.1%}" if wins else "")
            print(f"  Average Loss: {np.mean([t.pnl_pct for t in losses]):+.1%}" if losses else "")

            # Expected value per trade
            if wins and losses:
                win_rate = total_wins / total_trades
                avg_win = np.mean([t.pnl_pct for t in wins])
                avg_loss = abs(np.mean([t.pnl_pct for t in losses]))
                ev = win_rate * avg_win - (1 - win_rate) * avg_loss
                print(f"\n  Expected Value per trade: {ev:+.2%}")

                if ev > 0:
                    print(f"\n  >>> POSITIVE EXPECTANCY - Strategy has edge <<<")
                else:
                    print(f"\n  >>> NEGATIVE EXPECTANCY - Strategy loses money <<<")

            # Compound projection
            avg_return = np.mean([r.total_return for r in all_results])
            print(f"\n  Average return per symbol/period: {avg_return:+.1%}")

            # What would $20 become?
            print(f"\n  --- $20 Projection ---")
            starting = 20.0
            for n in [10, 25, 50, 100]:
                # Use actual EV if positive, otherwise show the loss
                if 'ev' in dir() and ev > 0:
                    projected = starting * ((1 + ev) ** n)
                else:
                    projected = starting * ((1 + (avg_return / max(1, total_trades/len(all_results)))) ** n)
                print(f"    After {n} trades: ${projected:.2f}")


def quick_test():
    """Quick test on one symbol."""
    print("Quick test on SNDL (2024)...")
    result = backtest_symbol("SNDL", "2024-01-01", "2024-12-31", verbose=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_full_backtest()
