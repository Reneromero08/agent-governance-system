"""
RESONANCE MOMENTUM BACKTEST
===========================

Tests the KEY INSIGHT: Trade based on dR/dt (R velocity), not just R.

- dR/dt > 0 = coherence BUILDING = momentum strengthening = ENTER
- dR/dt < 0 = coherence FADING = momentum dying = EXIT
- dR/dt ~ 0, R high = STABLE trend = HOLD
- dR/dt ~ 0, R low = NOISE = WAIT

This is different from crash detection (which uses alpha drift).
This is MOMENTUM TRADING using resonance velocity.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Need yfinance: pip install yfinance")
    sys.exit(1)

from signal_vocabulary import SignalState, AssetClass
from prime_radiant import PrimeRadiant
from formula_executor import MarketFormulaExecutor, RVelocityResult


# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy mode: "MOMENTUM" (old) or "PROTECTIVE" (new)
STRATEGY_MODE = "PROTECTIVE"

CONFIG_MOMENTUM = {
    # Entry conditions
    "min_dR_normalized": 0.20,
    "min_R_for_entry": 0.5,
    "min_confidence": 0.4,

    # Exit conditions
    "take_profit_pct": 0.08,
    "stop_loss_pct": 0.06,
    "exit_on_fading": False,

    # Position sizing
    "max_position_pct": 1.0,
}

CONFIG_PROTECTIVE = {
    # INVERTED LOGIC: Stay in market, exit on danger
    # Entry: R stable/rising (safe to be invested)
    "min_R_for_entry": 0.3,           # Low bar - be in market by default
    "require_stable_R": True,         # R not dropping fast

    # Exit: R fading = danger approaching
    "exit_dR_threshold": -0.3,        # Exit when dR goes significantly negative
    "exit_R_drop_pct": 0.20,          # Exit if R drops 20% from recent high

    # Standard stops (backup)
    "take_profit_pct": 0.15,          # Higher target (ride trends longer)
    "stop_loss_pct": 0.05,            # Still have a safety stop
    "trailing_stop_pct": 0.08,        # Trailing stop to lock in gains

    # Position sizing
    "max_position_pct": 1.0,
}

CONFIG = CONFIG_PROTECTIVE if STRATEGY_MODE == "PROTECTIVE" else CONFIG_MOMENTUM


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    entry_R: float
    entry_dR: float
    exit_date: str
    exit_price: float
    exit_R: float
    exit_dR: float
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
    # New metrics for resonance momentum
    avg_entry_dR: float
    avg_exit_dR: float
    avg_hold_time: float


# =============================================================================
# SIGNAL EXTRACTION (same as penny_compounder)
# =============================================================================

def extract_signals(prices: List[float], volumes: List[float]) -> Dict[str, float]:
    """Extract trading signals from price/volume data."""
    signals = {}

    if len(prices) < 10:
        return signals

    # Trend signals
    fast_ma = np.mean(prices[-5:])
    slow_ma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    trend = (fast_ma - slow_ma) / slow_ma

    if trend > 0.03:
        signals["trend_up"] = min(trend * 10, 1.0)
    elif trend < -0.03:
        signals["trend_down"] = min(abs(trend) * 10, 1.0)
    else:
        signals["sideways"] = 1.0 - abs(trend) * 10

    # Momentum
    if len(prices) >= 5:
        momentum = (prices[-1] - prices[-5]) / prices[-5]
        if momentum > 0.05:
            signals["momentum_strong"] = min(momentum * 5, 1.0)
        elif momentum < -0.05:
            signals["momentum_weak"] = min(abs(momentum) * 5, 1.0)

    # Volume signals
    if len(volumes) >= 20:
        vol_ratio = volumes[-1] / np.mean(volumes[-20:])
        if vol_ratio > 2.0:
            signals["volume_surge"] = min(vol_ratio / 5, 1.0)
        elif vol_ratio < 0.5:
            signals["volume_drought"] = 1.0 - vol_ratio

    # Volatility
    if len(prices) >= 10:
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(returns[-10:])
        if volatility > 0.03:
            signals["vol_expanding"] = min(volatility * 20, 1.0)

    return signals


# =============================================================================
# RESONANCE MOMENTUM BACKTESTER
# =============================================================================

def backtest_resonance_momentum(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100.0,
    verbose: bool = True
) -> Optional[BacktestResult]:
    """
    Backtest the RESONANCE MOMENTUM strategy.

    Key difference from old strategy:
    - OLD: Buy when R > threshold (static)
    - NEW: Buy when dR/dt > threshold (dynamic, velocity-based)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESONANCE MOMENTUM BACKTEST: {symbol}")
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

    # Initialize formula components
    radiant = PrimeRadiant()
    executor = MarketFormulaExecutor()

    # Build state vectors for entire history
    vector_history = []
    for i in range(20, len(prices)):
        price_slice = prices[max(0, i-30):i+1]
        vol_slice = volumes[max(0, i-30):i+1]

        signals = extract_signals(price_slice, vol_slice)
        state = SignalState(
            timestamp=dates[i],
            asset=symbol,
            asset_class=AssetClass.STOCKS,
            signals=signals,
        )
        vec = radiant.state_to_vector(state)
        vector_history.append(vec)

    if len(vector_history) < 20:
        print(f"  [SKIP] Not enough vectors ({len(vector_history)})")
        return None

    # Compute R sequence
    R_results = executor.compute_R_sequence(vector_history, lookback=10)
    R_values = [r.R for r in R_results]

    # Compute price changes for trend detection
    price_changes = []
    for i in range(20, len(prices)):
        if i >= 5:
            change = (prices[i] - prices[i-5]) / prices[i-5]
        else:
            change = 0.0
        price_changes.append(change)

    # Trading simulation
    capital = initial_capital
    position = None  # {shares, entry_price, entry_date, entry_idx, entry_R, entry_dR}
    trades = []
    equity_curve = [capital]
    R_history = []

    if verbose:
        print(f"\n  --- Trading Signals ---")

    for i in range(5, len(vector_history)):
        # Map back to original price index
        price_idx = 20 + i
        if price_idx >= len(prices):
            break

        current_price = prices[price_idx]
        current_date = dates[price_idx]
        current_R = R_values[i]

        R_history.append(current_R)

        # Compute R velocity
        velocity_result = executor.compute_R_velocity(
            R_history[-6:],  # Last 6 R values
            price_trend=price_changes[i] if i < len(price_changes) else 0.0,
        )

        # ===== PROTECTIVE STRATEGY =====
        if STRATEGY_MODE == "PROTECTIVE":
            # Track R high watermark for this position
            if position:
                if "R_high" not in position:
                    position["R_high"] = current_R
                else:
                    position["R_high"] = max(position["R_high"], current_R)

                if "price_high" not in position:
                    position["price_high"] = current_price
                else:
                    position["price_high"] = max(position["price_high"], current_price)

            # Check exits first
            if position:
                pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
                holding_days = price_idx - position["entry_idx"]

                exit_reason = None

                # Take profit
                if pnl_pct >= CONFIG["take_profit_pct"]:
                    exit_reason = "TAKE_PROFIT"
                # Stop loss
                elif pnl_pct <= -CONFIG["stop_loss_pct"]:
                    exit_reason = "STOP_LOSS"
                # Trailing stop (lock in gains)
                elif position["price_high"] > position["entry_price"]:
                    trailing_stop = position["price_high"] * (1 - CONFIG["trailing_stop_pct"])
                    if current_price < trailing_stop:
                        exit_reason = "TRAILING_STOP"
                # R FADING - the protective signal
                elif velocity_result.dR_normalized < CONFIG["exit_dR_threshold"]:
                    exit_reason = "R_DANGER"
                # R dropped significantly from high
                elif position["R_high"] > 0 and current_R < position["R_high"] * (1 - CONFIG["exit_R_drop_pct"]):
                    exit_reason = "R_DROP"
                # Max holding time
                elif holding_days >= 20:
                    exit_reason = "MAX_HOLD"

                if exit_reason:
                    proceeds = position["shares"] * current_price
                    capital += proceeds

                    trade = Trade(
                        symbol=symbol,
                        entry_date=position["entry_date"],
                        entry_price=position["entry_price"],
                        entry_R=position["entry_R"],
                        entry_dR=position["entry_dR"],
                        exit_date=current_date,
                        exit_price=current_price,
                        exit_R=current_R,
                        exit_dR=velocity_result.dR,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        holding_days=holding_days,
                    )
                    trades.append(trade)

                    if verbose and len(trades) <= 30:
                        status = "WIN" if pnl_pct > 0 else "LOSS"
                        print(f"  {current_date}: EXIT ({exit_reason}) {pnl_pct:+.1%} [{status}]")
                        print(f"             R: {current_R:.2f} dR: {velocity_result.dR_normalized:+.2f}")

                    position = None

            # Entry: Be in market when R is stable/rising (not dangerous)
            if position is None:
                # Entry condition: R not dropping, price trend positive
                entry_signal = (
                    velocity_result.dR_normalized > -0.2 and  # R not dropping fast
                    current_R > CONFIG["min_R_for_entry"] and
                    price_changes[i] > 0 if i < len(price_changes) else True  # Price trending up
                )

                if entry_signal:
                    shares = capital / current_price
                    position = {
                        "shares": shares,
                        "entry_price": current_price,
                        "entry_date": current_date,
                        "entry_idx": price_idx,
                        "entry_R": current_R,
                        "entry_dR": velocity_result.dR,
                        "R_high": current_R,
                        "price_high": current_price,
                    }
                    capital = 0

                    if verbose and len(trades) < 30:
                        print(f"  {current_date}: BUY @ ${current_price:.2f}")
                        print(f"             R: {current_R:.2f} dR: {velocity_result.dR_normalized:+.2f}")

        # ===== MOMENTUM STRATEGY (original) =====
        else:
            # Check exits first
            if position:
                pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
                holding_days = price_idx - position["entry_idx"]

                exit_reason = None

                # Take profit
                if pnl_pct >= CONFIG["take_profit_pct"]:
                    exit_reason = "TAKE_PROFIT"
                # Stop loss
                elif pnl_pct <= -CONFIG["stop_loss_pct"]:
                    exit_reason = "STOP_LOSS"
                # Resonance fading (THE KEY EXIT SIGNAL)
                elif CONFIG.get("exit_on_fading", False) and velocity_result.momentum_phase == "FADING":
                    exit_reason = "R_FADING"
                # Max holding time
                elif holding_days >= 15:
                    exit_reason = "MAX_HOLD"

                if exit_reason:
                    proceeds = position["shares"] * current_price
                    capital += proceeds

                    trade = Trade(
                        symbol=symbol,
                        entry_date=position["entry_date"],
                        entry_price=position["entry_price"],
                        entry_R=position["entry_R"],
                        entry_dR=position["entry_dR"],
                        exit_date=current_date,
                        exit_price=current_price,
                        exit_R=current_R,
                        exit_dR=velocity_result.dR,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        holding_days=holding_days,
                    )
                    trades.append(trade)

                    if verbose and len(trades) <= 30:
                        status = "WIN" if pnl_pct > 0 else "LOSS"
                        print(f"  {current_date}: EXIT ({exit_reason}) {pnl_pct:+.1%} [{status}]")
                        print(f"             R: {current_R:.2f} dR: {velocity_result.dR_normalized:+.2f}")

                    position = None

            # Check entries (THE KEY CHANGE: Use dR/dt not just R)
            if position is None:
                # Entry condition: R velocity positive + price trending up + minimum R
                entry_signal = (
                    velocity_result.momentum_phase == "BUILDING" and
                    velocity_result.dR_normalized > CONFIG.get("min_dR_normalized", 0.15) and
                    current_R > CONFIG.get("min_R_for_entry", 0.3) and
                    velocity_result.signal == "BUY" and
                    velocity_result.confidence >= CONFIG.get("min_confidence", 0.3)
                )

                if entry_signal:
                    shares = capital / current_price
                    position = {
                        "shares": shares,
                        "entry_price": current_price,
                        "entry_date": current_date,
                        "entry_idx": price_idx,
                        "entry_R": current_R,
                        "entry_dR": velocity_result.dR,
                    }
                    capital = 0

                    if verbose and len(trades) < 30:
                        print(f"  {current_date}: BUY @ ${current_price:.2f}")
                        print(f"             R: {current_R:.2f} dR: {velocity_result.dR_normalized:+.2f} conf: {velocity_result.confidence:.2f}")

        # Update equity
        equity = capital + (position["shares"] * current_price if position else 0)
        equity_curve.append(equity)

    # Close any open position
    if position:
        final_price = prices[-1]
        pnl_pct = (final_price - position["entry_price"]) / position["entry_price"]
        capital += position["shares"] * final_price

        trade = Trade(
            symbol=symbol,
            entry_date=position["entry_date"],
            entry_price=position["entry_price"],
            entry_R=position["entry_R"],
            entry_dR=position["entry_dR"],
            exit_date=dates[-1],
            exit_price=final_price,
            exit_R=R_values[-1] if R_values else 0,
            exit_dR=0,
            exit_reason="END_OF_TEST",
            pnl_pct=pnl_pct,
            holding_days=len(prices) - 1 - position["entry_idx"],
        )
        trades.append(trade)

    # Calculate metrics
    n_trades = len(trades)
    if n_trades == 0:
        if verbose:
            print(f"\n  [RESULT] No trades generated")
        return BacktestResult(
            symbol=symbol,
            period=f"{start_date} to {end_date}",
            n_trades=0, n_wins=0, n_losses=0,
            win_rate=0, avg_win=0, avg_loss=0,
            total_return=0, max_drawdown=0,
            trades=[],
            avg_entry_dR=0, avg_exit_dR=0, avg_hold_time=0,
        )

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades
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

    # Resonance momentum specific metrics
    avg_entry_dR = np.mean([t.entry_dR for t in trades])
    avg_exit_dR = np.mean([t.exit_dR for t in trades])
    avg_hold_time = np.mean([t.holding_days for t in trades])

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
        avg_entry_dR=avg_entry_dR,
        avg_exit_dR=avg_exit_dR,
        avg_hold_time=avg_hold_time,
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
        print(f"\n  --- Resonance Metrics ---")
        print(f"  Avg Entry dR: {avg_entry_dR:+.4f}")
        print(f"  Avg Exit dR: {avg_exit_dR:+.4f}")
        print(f"  Avg Hold Time: {avg_hold_time:.1f} days")

    return result


# =============================================================================
# FULL BACKTEST
# =============================================================================

def run_full_backtest():
    """Run backtest across multiple symbols and periods."""

    # Same symbols as penny backtest for comparison
    symbols = ["SNDL", "AMC", "BB", "NOK", "F", "PLUG", "FCEL"]

    periods = [
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2023-01-01", "2024-12-31", "2023-2024"),
    ]

    print("=" * 70)
    print("RESONANCE MOMENTUM BACKTEST")
    print("=" * 70)
    print(f"\nStrategy Mode: {STRATEGY_MODE}")
    if STRATEGY_MODE == "PROTECTIVE":
        print(f"Logic: Stay invested, exit on R danger signals")
        print(f"Exit: R drop {CONFIG['exit_R_drop_pct']:.0%} | dR < {CONFIG['exit_dR_threshold']} | Trailing {CONFIG['trailing_stop_pct']:.0%}")
    else:
        print(f"Logic: Trade on dR/dt (R velocity), not just R")
        print(f"Entry: dR_normalized > {CONFIG.get('min_dR_normalized', 'N/A')} + BUY signal")
    print(f"Stops: Take profit {CONFIG['take_profit_pct']:.0%} | Stop loss {CONFIG['stop_loss_pct']:.0%}")

    all_results = []

    for start, end, period_name in periods:
        print(f"\n\n{'#'*70}")
        print(f"# PERIOD: {period_name}")
        print(f"{'#'*70}")

        period_results = []

        for symbol in symbols:
            try:
                result = backtest_resonance_momentum(symbol, start, end, verbose=True)
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

            print(f"  Symbols tested: {len(period_results)}")
            print(f"  Total trades: {total_trades}")
            print(f"  Overall win rate: {total_wins/total_trades:.1%}" if total_trades > 0 else "  No trades")
            print(f"  Avg return per symbol: {avg_return:+.1%}")

    # Grand summary
    if all_results:
        print(f"\n\n{'#'*70}")
        print("# GRAND SUMMARY - RESONANCE MOMENTUM")
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

            if wins:
                print(f"  Average Win: {np.mean([t.pnl_pct for t in wins]):+.1%}")
            if losses:
                print(f"  Average Loss: {np.mean([t.pnl_pct for t in losses]):+.1%}")

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

            # Resonance momentum specific
            avg_entry_dR = np.mean([t.entry_dR for t in all_trades])
            avg_exit_dR = np.mean([t.exit_dR for t in all_trades])
            avg_hold = np.mean([t.holding_days for t in all_trades])

            print(f"\n  --- Resonance Metrics ---")
            print(f"  Avg Entry dR: {avg_entry_dR:+.4f}")
            print(f"  Avg Exit dR: {avg_exit_dR:+.4f}")
            print(f"  Avg Hold Time: {avg_hold:.1f} days")

            # Exit reason breakdown
            print(f"\n  --- Exit Reasons ---")
            reasons = {}
            for t in all_trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                pct = count / len(all_trades) * 100
                print(f"    {reason}: {count} ({pct:.1f}%)")

            # Compound projection
            print(f"\n  --- $20 Projection ---")
            starting = 20.0
            for n in [10, 25, 50, 100]:
                if ev > 0:
                    projected = starting * ((1 + ev) ** n)
                else:
                    projected = starting * ((1 + ev) ** n)
                print(f"    After {n} trades: ${projected:.2f}")


def quick_test():
    """Quick test on one symbol."""
    print("Quick test on SNDL (2024)...")
    result = backtest_resonance_momentum("SNDL", "2024-01-01", "2024-12-31", verbose=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_full_backtest()
