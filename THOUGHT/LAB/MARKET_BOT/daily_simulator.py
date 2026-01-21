"""
DAILY SIMULATOR
===============

Run this daily to simulate trading with the Psychohistory bot.
Fetches latest data, makes decisions, tracks paper portfolio.

Usage:
    python daily_simulator.py              # Run for SPY
    python daily_simulator.py AAPL         # Run for specific symbol
    python daily_simulator.py --status     # Show current portfolio
    python daily_simulator.py --reset      # Reset portfolio
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from signal_vocabulary import SignalState, AssetClass
from signal_extractor import SignalExtractor
from real_data_ingest import RealDataFetcher
from psychohistory_bot import PsychohistoryBot, BotConfig

# =============================================================================
# PORTFOLIO PERSISTENCE
# =============================================================================

PORTFOLIO_FILE = Path(__file__).parent / "paper_portfolio.json"


def load_portfolio() -> Dict:
    """Load persisted portfolio state."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {
        "created": datetime.now().isoformat(),
        "initial_capital": 100000.0,
        "capital": 100000.0,
        "positions": {},
        "trades": [],
        "equity_history": [{"date": datetime.now().strftime("%Y-%m-%d"), "equity": 100000.0}],
        "last_run": None,
    }


def save_portfolio(portfolio: Dict):
    """Save portfolio state."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


def reset_portfolio():
    """Reset portfolio to initial state."""
    portfolio = {
        "created": datetime.now().isoformat(),
        "initial_capital": 100000.0,
        "capital": 100000.0,
        "positions": {},
        "trades": [],
        "equity_history": [{"date": datetime.now().strftime("%Y-%m-%d"), "equity": 100000.0}],
        "last_run": None,
    }
    save_portfolio(portfolio)
    print("Portfolio reset to $100,000")
    return portfolio


# =============================================================================
# DAILY SIMULATION
# =============================================================================

def run_daily(symbol: str = "SPY", verbose: bool = True) -> Dict:
    """
    Run daily simulation for a symbol.

    Fetches recent data, extracts signals, gets bot decision.
    """
    portfolio = load_portfolio()
    today = datetime.now().strftime("%Y-%m-%d")

    if verbose:
        print("=" * 60)
        print(f"PSYCHOHISTORY DAILY SIMULATION - {today}")
        print("=" * 60)

    # Check if already run today
    if portfolio.get("last_run") == today:
        if verbose:
            print(f"\n[INFO] Already ran today. Use --force to run again.")
            print_status(portfolio)
        return portfolio

    # Fetch recent data (need ~60 days for indicators)
    if verbose:
        print(f"\nFetching recent data for {symbol}...")

    fetcher = RealDataFetcher()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        data = fetcher.fetch(symbol, start_date, end_date, use_cache=False)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return portfolio

    if len(data.bars) < 50:
        print(f"[ERROR] Not enough data ({len(data.bars)} bars)")
        return portfolio

    if verbose:
        print(f"  Got {len(data.bars)} bars")
        print(f"  Latest: {data.bars[-1].timestamp} @ ${data.bars[-1].close:.2f}")

    # Extract signals
    extractor = SignalExtractor()
    current_state = extractor.extract(data, len(data.bars) - 1, AssetClass.ALL)

    if verbose:
        print(f"\n--- Current Signals ---")
        for sig, strength in sorted(current_state.signals.items(), key=lambda x: -x[1]):
            print(f"  {sig}: {strength:.2f}")

    # Initialize bot with recent history
    config = BotConfig(
        initial_capital=portfolio["capital"],
        lookback_window=10,
        alpha_window=20,
    )
    bot = PsychohistoryBot(config)

    # Feed bot recent history (last 30 bars)
    if verbose:
        print(f"\n--- Warming up bot with recent history ---")

    history_states = extractor.extract_sequence(data, start_index=max(30, len(data.bars) - 40))

    for i, state in enumerate(history_states[:-1]):  # All but last
        bar_idx = max(30, len(data.bars) - 40) + i
        price = data.bars[bar_idx].close
        bot.observe(state, price)

    # Current price and direction
    current_price = data.bars[-1].close

    # Compute direction from recent momentum
    prices = [b.close for b in data.bars[-10:]]
    if len(prices) >= 5:
        recent_return = (prices[-1] - prices[-5]) / prices[-5]
        direction = 1 if recent_return > 0.005 else (-1 if recent_return < -0.005 else 0)
    else:
        direction = 0

    # Get bot decision
    if verbose:
        print(f"\n--- Bot Decision ---")

    trade = bot.decide(current_state, current_price, direction)

    # Get gate assessment for display
    gate_result = bot.gate.assess(
        R=bot.executor.compute_R(
            bot.vector_history[-1],
            bot.vector_history[-11:-1] if len(bot.vector_history) > 1 else []
        ).R,
        alpha=bot.executor.compute_alpha(bot.vector_history[-20:]).alpha,
        Df=bot.executor.compute_alpha(bot.vector_history[-20:]).Df,
        requested_tier=config.default_tier,
    )

    if verbose:
        print(f"  R = {gate_result.gate.R:.4f}")
        print(f"  Gate: {gate_result.gate.status}")
        print(f"  Tier: {gate_result.gate.tier.value}")
        print(f"  Alpha: {gate_result.drift.alpha:.4f}")
        print(f"  Drift Warning: {gate_result.drift.warning_level.name}")
        print(f"  Direction Signal: {direction}")

    # Update portfolio based on trade
    if trade:
        if verbose:
            print(f"\n  >>> TRADE: {trade.action} {symbol} @ ${trade.price:.2f}")
            print(f"      Size: {trade.size:.4f} | Reason: {trade.reason}")

        # Record trade
        portfolio["trades"].append({
            "date": today,
            "symbol": symbol,
            "action": trade.action,
            "price": trade.price,
            "size": trade.size,
            "R": trade.R_value,
            "reason": trade.reason,
        })

        # Update positions
        if trade.action == "BUY":
            portfolio["positions"][symbol] = {
                "size": trade.size,
                "entry_price": trade.price,
                "entry_date": today,
            }
            portfolio["capital"] -= trade.size * trade.price
        elif trade.action in ["CLOSE", "SELL"]:
            if symbol in portfolio["positions"]:
                pos = portfolio["positions"][symbol]
                pnl = (trade.price - pos["entry_price"]) * pos["size"]
                portfolio["capital"] += pos["size"] * trade.price
                del portfolio["positions"][symbol]
                if verbose:
                    print(f"      PnL: ${pnl:+.2f}")
        elif trade.action == "REDUCE":
            if symbol in portfolio["positions"]:
                pos = portfolio["positions"][symbol]
                pnl = (trade.price - pos["entry_price"]) * trade.size
                portfolio["capital"] += trade.size * trade.price
                pos["size"] -= trade.size
                if pos["size"] <= 0:
                    del portfolio["positions"][symbol]
    else:
        if verbose:
            print(f"\n  >>> NO TRADE (holding)")

    # Compute total equity
    equity = portfolio["capital"]
    for sym, pos in portfolio["positions"].items():
        # Use current price for open positions
        if sym == symbol:
            equity += pos["size"] * current_price
        else:
            equity += pos["size"] * pos["entry_price"]  # Approximate

    # Record equity
    portfolio["equity_history"].append({
        "date": today,
        "equity": equity,
    })

    portfolio["last_run"] = today
    save_portfolio(portfolio)

    if verbose:
        print(f"\n--- Portfolio Status ---")
        print_status(portfolio, current_price if symbol in portfolio["positions"] else None, symbol)

    return portfolio


def print_status(portfolio: Dict, current_price: Optional[float] = None, symbol: str = "SPY"):
    """Print portfolio status."""
    print(f"  Cash: ${portfolio['capital']:,.2f}")

    if portfolio["positions"]:
        print(f"  Positions:")
        for sym, pos in portfolio["positions"].items():
            price = current_price if sym == symbol and current_price else pos["entry_price"]
            value = pos["size"] * price
            pnl = (price - pos["entry_price"]) * pos["size"]
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            print(f"    {sym}: {pos['size']:.4f} @ ${pos['entry_price']:.2f} -> ${price:.2f}")
            print(f"         Value: ${value:,.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
    else:
        print(f"  Positions: None (all cash)")

    # Compute total equity
    equity = portfolio["capital"]
    for sym, pos in portfolio["positions"].items():
        price = current_price if sym == symbol and current_price else pos["entry_price"]
        equity += pos["size"] * price

    print(f"  Total Equity: ${equity:,.2f}")

    # Performance
    initial = portfolio["initial_capital"]
    returns = (equity - initial) / initial * 100
    print(f"  Returns: {returns:+.2f}%")

    # Recent trades
    if portfolio["trades"]:
        print(f"\n  Recent Trades:")
        for trade in portfolio["trades"][-5:]:
            print(f"    {trade['date']}: {trade['action']} {trade.get('symbol', 'SPY')} @ ${trade['price']:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = sys.argv[1:]

    if "--status" in args:
        portfolio = load_portfolio()
        print("=" * 60)
        print("PSYCHOHISTORY PORTFOLIO STATUS")
        print("=" * 60)
        print_status(portfolio)
        return

    if "--reset" in args:
        reset_portfolio()
        return

    # Get symbol
    symbol = "SPY"
    for arg in args:
        if not arg.startswith("--"):
            symbol = arg.upper()
            break

    run_daily(symbol, verbose=True)


if __name__ == "__main__":
    main()
