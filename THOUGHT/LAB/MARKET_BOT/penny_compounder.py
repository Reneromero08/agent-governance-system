"""
PENNY STOCK COMPOUNDER
======================

Start small ($5-20), compound gains, level up.

Strategy:
1. Find cheap stocks with strong momentum + high R
2. Trade WITH the trend (not crash prediction)
3. Take 5-15% gains, cut 3% losses
4. Reinvest profits to compound
5. Level up when you hit thresholds

The formula doesn't predict crashes here - it measures
TREND STABILITY. High R = trend is real, ride it.

Usage:
    python penny_compounder.py --scan           # Find opportunities
    python penny_compounder.py --status         # Check portfolio
    python penny_compounder.py --simulate       # Paper trade test
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Try yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARN] yfinance not installed. Run: pip install yfinance")

from signal_vocabulary import SignalState, AssetClass
from signal_extractor import SignalExtractor
from prime_radiant import PrimeRadiant
from formula_executor import MarketFormulaExecutor
from seldon_gate import SeldonGate, AlphaWarningLevel


# =============================================================================
# CONFIGURATION
# =============================================================================

# Penny stock universe - cheap, liquid, volatile
PENNY_UNIVERSE = [
    # Popular penny stocks (update this list periodically)
    "SNDL", "PLTR", "SOFI", "NIO", "RIVN", "LCID", "F", "SNAP",
    "AMC", "BB", "NOK", "CLOV", "WISH", "SKLZ", "RIDE", "NKLA",
    "OPEN", "DKNG", "SPCE", "PLUG", "FCEL", "BLNK", "QS", "GOEV",
    # Add more as needed
]

# Level thresholds - unlock bigger plays as you grow
LEVELS = {
    1: {"min": 0, "max": 50, "max_position_pct": 1.0, "name": "Seedling"},
    2: {"min": 50, "max": 200, "max_position_pct": 0.5, "name": "Sprout"},
    3: {"min": 200, "max": 1000, "max_position_pct": 0.3, "name": "Sapling"},
    4: {"min": 1000, "max": 5000, "max_position_pct": 0.2, "name": "Tree"},
    5: {"min": 5000, "max": 25000, "max_position_pct": 0.15, "name": "Grove"},
    6: {"min": 25000, "max": float('inf'), "max_position_pct": 0.10, "name": "Forest"},
}

# Trading parameters
CONFIG = {
    "take_profit_pct": 0.10,      # Take profits at 10%
    "stop_loss_pct": 0.03,        # Cut losses at 3%
    "min_R_threshold": 1.0,       # Minimum R to consider
    "min_trend_strength": 0.02,   # Minimum trend strength (2%)
    "max_positions": 3,           # Max concurrent positions
    "min_volume": 1000000,        # Minimum daily volume
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockSignal:
    """Signal for a single stock."""
    symbol: str
    price: float
    R: float
    trend_strength: float
    momentum_score: float
    volume_ratio: float
    alpha: float
    drift_warning: str
    score: float  # Combined score for ranking
    action: str   # BUY, HOLD, AVOID


@dataclass
class Position:
    """Tracked position."""
    symbol: str
    entry_price: float
    entry_date: str
    shares: float
    current_price: float = 0.0


@dataclass
class Portfolio:
    """Portfolio state."""
    capital: float = 20.0
    initial_capital: float = 20.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)
    created: str = ""
    last_updated: str = ""

    def get_level(self) -> int:
        """Get current level based on equity."""
        equity = self.get_equity()
        for level, config in sorted(LEVELS.items()):
            if config["min"] <= equity < config["max"]:
                return level
        return 6

    def get_equity(self) -> float:
        """Total equity (cash + positions)."""
        pos_value = sum(p.shares * p.current_price for p in self.positions.values())
        return self.capital + pos_value

    def get_returns_pct(self) -> float:
        """Returns as percentage."""
        return (self.get_equity() - self.initial_capital) / self.initial_capital * 100


# =============================================================================
# PORTFOLIO PERSISTENCE
# =============================================================================

PORTFOLIO_FILE = Path(__file__).parent / "penny_portfolio.json"


def load_portfolio() -> Portfolio:
    """Load portfolio from file."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            positions = {
                k: Position(**v) for k, v in data.get("positions", {}).items()
            }
            return Portfolio(
                capital=data.get("capital", 20.0),
                initial_capital=data.get("initial_capital", 20.0),
                positions=positions,
                trades=data.get("trades", []),
                created=data.get("created", datetime.now().isoformat()),
                last_updated=data.get("last_updated", ""),
            )
    return Portfolio(
        created=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
    )


def save_portfolio(portfolio: Portfolio):
    """Save portfolio to file."""
    portfolio.last_updated = datetime.now().isoformat()
    data = {
        "capital": portfolio.capital,
        "initial_capital": portfolio.initial_capital,
        "positions": {
            k: {
                "symbol": v.symbol,
                "entry_price": v.entry_price,
                "entry_date": v.entry_date,
                "shares": v.shares,
                "current_price": v.current_price,
            }
            for k, v in portfolio.positions.items()
        },
        "trades": portfolio.trades,
        "created": portfolio.created,
        "last_updated": portfolio.last_updated,
    }
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# STOCK ANALYZER
# =============================================================================

class PennyAnalyzer:
    """Analyze penny stocks using the Psychohistory formula."""

    def __init__(self):
        self.radiant = PrimeRadiant()
        self.executor = MarketFormulaExecutor()
        self.gate = SeldonGate()
        self.extractor = SignalExtractor()

    def fetch_stock_data(self, symbol: str, days: int = 60) -> Optional[Dict]:
        """Fetch recent stock data."""
        if not YFINANCE_AVAILABLE:
            return None

        try:
            ticker = yf.Ticker(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            hist = ticker.history(start=start, end=end)

            if len(hist) < 30:
                return None

            return {
                "symbol": symbol,
                "prices": hist['Close'].tolist(),
                "volumes": hist['Volume'].tolist(),
                "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                "current_price": hist['Close'].iloc[-1],
                "avg_volume": hist['Volume'].mean(),
            }
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            return None

    def analyze_stock(self, data: Dict) -> Optional[StockSignal]:
        """Analyze a single stock."""
        prices = data["prices"]
        volumes = data["volumes"]
        symbol = data["symbol"]

        if len(prices) < 30:
            return None

        # Build signal states for recent history
        # Simplified: use price/volume patterns directly
        vector_history = []

        for i in range(20, len(prices)):
            # Extract signals at this point
            price_slice = prices[max(0, i-30):i+1]
            vol_slice = volumes[max(0, i-30):i+1]

            signals = self._extract_signals(price_slice, vol_slice)
            state = SignalState(
                timestamp=data["dates"][i] if i < len(data["dates"]) else "",
                asset=symbol,
                asset_class=AssetClass.STOCKS,
                signals=signals,
            )
            vec = self.radiant.state_to_vector(state)
            vector_history.append(vec)

        if len(vector_history) < 10:
            return None

        # Compute R
        current_vec = vector_history[-1]
        context_vecs = vector_history[-11:-1]
        formula_result = self.executor.compute_R(current_vec, context_vecs)
        R = formula_result.R

        # Compute alpha
        alpha_vecs = vector_history[-20:] if len(vector_history) >= 3 else vector_history
        alpha_result = self.executor.compute_alpha(alpha_vecs)

        # Trend strength
        fast_ma = np.mean(prices[-5:])
        slow_ma = np.mean(prices[-20:])
        trend_strength = (fast_ma - slow_ma) / slow_ma

        # Momentum (rate of change)
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # Volume ratio (current vs average)
        vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0

        # Alpha drift warning
        drift_warning = "NONE"
        if abs(alpha_result.alpha - 0.5) > 0.3:
            drift_warning = "HIGH"
        elif abs(alpha_result.alpha - 0.5) > 0.2:
            drift_warning = "MEDIUM"
        elif abs(alpha_result.alpha - 0.5) > 0.1:
            drift_warning = "LOW"

        # Combined score for ranking
        # High R + positive trend + good volume = good score
        score = 0.0
        if R > CONFIG["min_R_threshold"] and trend_strength > CONFIG["min_trend_strength"]:
            score = R * (1 + trend_strength * 10) * min(vol_ratio, 3.0)

        # Determine action
        action = "AVOID"
        if (R > CONFIG["min_R_threshold"] and
            trend_strength > CONFIG["min_trend_strength"] and
            drift_warning in ["NONE", "LOW"] and
            momentum > 0):
            action = "BUY"
        elif R > 0.5 and trend_strength > 0:
            action = "HOLD"

        return StockSignal(
            symbol=symbol,
            price=data["current_price"],
            R=R,
            trend_strength=trend_strength,
            momentum_score=momentum,
            volume_ratio=vol_ratio,
            alpha=alpha_result.alpha,
            drift_warning=drift_warning,
            score=score,
            action=action,
        )

    def _extract_signals(self, prices: List[float], volumes: List[float]) -> Dict[str, float]:
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

        # RSI proxy
        if len(prices) >= 14:
            gains = [max(prices[i] - prices[i-1], 0) for i in range(1, len(prices))]
            losses = [max(prices[i-1] - prices[i], 0) for i in range(1, len(prices))]
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                if rsi > 70:
                    signals["overbought"] = (rsi - 70) / 30
                elif rsi < 30:
                    signals["oversold"] = (30 - rsi) / 30

        return signals

    def scan_universe(self, symbols: List[str] = None, verbose: bool = True) -> List[StockSignal]:
        """Scan all stocks in universe."""
        symbols = symbols or PENNY_UNIVERSE
        results = []

        if verbose:
            print(f"\nScanning {len(symbols)} stocks...\n")

        for i, symbol in enumerate(symbols):
            if verbose:
                print(f"  [{i+1}/{len(symbols)}] {symbol}...", end=" ")

            data = self.fetch_stock_data(symbol)
            if not data:
                if verbose:
                    print("SKIP (no data)")
                continue

            # Filter by price (must be cheap enough)
            if data["current_price"] > 20:
                if verbose:
                    print(f"SKIP (${data['current_price']:.2f} too expensive)")
                continue

            # Filter by volume
            if data["avg_volume"] < CONFIG["min_volume"]:
                if verbose:
                    print("SKIP (low volume)")
                continue

            signal = self.analyze_stock(data)
            if signal:
                results.append(signal)
                if verbose:
                    print(f"${signal.price:.2f} | R={signal.R:.1f} | {signal.action}")
            else:
                if verbose:
                    print("SKIP (analysis failed)")

        # Sort by score (best first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results


# =============================================================================
# TRADING LOGIC
# =============================================================================

def get_buy_candidates(signals: List[StockSignal], portfolio: Portfolio) -> List[StockSignal]:
    """Get stocks worth buying."""
    level = portfolio.get_level()
    level_config = LEVELS[level]

    # Filter to BUY signals only
    candidates = [s for s in signals if s.action == "BUY"]

    # Filter out stocks we already own
    owned = set(portfolio.positions.keys())
    candidates = [s for s in candidates if s.symbol not in owned]

    # Check if we can afford any
    max_position = portfolio.capital * level_config["max_position_pct"]
    candidates = [s for s in candidates if s.price <= max_position]

    return candidates[:5]  # Top 5


def check_exits(portfolio: Portfolio, analyzer: PennyAnalyzer) -> List[Tuple[str, str, float]]:
    """Check if any positions should be exited."""
    exits = []

    for symbol, position in portfolio.positions.items():
        # Fetch current price
        data = analyzer.fetch_stock_data(symbol, days=10)
        if not data:
            continue

        current_price = data["current_price"]
        position.current_price = current_price

        # Calculate P&L
        pnl_pct = (current_price - position.entry_price) / position.entry_price

        # Take profit
        if pnl_pct >= CONFIG["take_profit_pct"]:
            exits.append((symbol, "TAKE_PROFIT", pnl_pct))
            continue

        # Stop loss
        if pnl_pct <= -CONFIG["stop_loss_pct"]:
            exits.append((symbol, "STOP_LOSS", pnl_pct))
            continue

        # Analyze if trend is dying
        signal = analyzer.analyze_stock(data)
        if signal and signal.R < 0.3:
            exits.append((symbol, "TREND_DYING", pnl_pct))

    return exits


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_scan(verbose: bool = True):
    """Scan for opportunities."""
    portfolio = load_portfolio()
    level = portfolio.get_level()
    level_config = LEVELS[level]

    print("=" * 70)
    print("PENNY COMPOUNDER - OPPORTUNITY SCAN")
    print("=" * 70)
    print(f"\nLevel {level}: {level_config['name']}")
    print(f"Equity: ${portfolio.get_equity():.2f}")
    print(f"Cash: ${portfolio.capital:.2f}")
    print(f"Max position: {level_config['max_position_pct']:.0%} = ${portfolio.capital * level_config['max_position_pct']:.2f}")

    analyzer = PennyAnalyzer()

    # Check exits first
    if portfolio.positions:
        print(f"\n--- Checking {len(portfolio.positions)} open positions ---")
        exits = check_exits(portfolio, analyzer)
        for symbol, reason, pnl in exits:
            print(f"  EXIT: {symbol} ({reason}) - P&L: {pnl:+.1%}")

    # Scan for new opportunities
    print(f"\n--- Scanning for new opportunities ---")
    signals = analyzer.scan_universe(verbose=verbose)

    # Get buy candidates
    candidates = get_buy_candidates(signals, portfolio)

    if candidates:
        print(f"\n--- TOP PICKS ---")
        print(f"{'Symbol':<8} {'Price':>8} {'R':>6} {'Trend':>8} {'Score':>8} {'Action'}")
        print("-" * 60)
        for sig in candidates:
            print(f"{sig.symbol:<8} ${sig.price:>7.2f} {sig.R:>6.1f} {sig.trend_strength:>+7.1%} {sig.score:>8.1f} {sig.action}")

        # Calculate position size
        max_pos = portfolio.capital * level_config["max_position_pct"]
        best = candidates[0]
        shares = int(max_pos / best.price)
        cost = shares * best.price

        print(f"\n>>> SUGGESTED TRADE <<<")
        print(f"    BUY {shares} shares of {best.symbol} @ ${best.price:.2f}")
        print(f"    Cost: ${cost:.2f}")
        print(f"    Take profit: ${best.price * (1 + CONFIG['take_profit_pct']):.2f} (+{CONFIG['take_profit_pct']:.0%})")
        print(f"    Stop loss: ${best.price * (1 - CONFIG['stop_loss_pct']):.2f} (-{CONFIG['stop_loss_pct']:.0%})")
    else:
        print("\n[INFO] No good opportunities right now. Check back later.")

    # Save updated prices
    save_portfolio(portfolio)


def cmd_status():
    """Show portfolio status."""
    portfolio = load_portfolio()
    level = portfolio.get_level()
    level_config = LEVELS[level]

    print("=" * 70)
    print("PENNY COMPOUNDER - PORTFOLIO STATUS")
    print("=" * 70)

    print(f"\n--- Account ---")
    print(f"  Level: {level} ({level_config['name']})")
    print(f"  Initial: ${portfolio.initial_capital:.2f}")
    print(f"  Equity: ${portfolio.get_equity():.2f}")
    print(f"  Cash: ${portfolio.capital:.2f}")
    print(f"  Returns: {portfolio.get_returns_pct():+.1f}%")

    # Progress to next level
    next_level = min(level + 1, 6)
    if next_level != level:
        next_threshold = LEVELS[next_level]["min"]
        current = portfolio.get_equity()
        progress = (current - level_config["min"]) / (next_threshold - level_config["min"]) * 100
        print(f"\n  Progress to Level {next_level}: {progress:.0f}%")
        print(f"  Need: ${next_threshold - current:.2f} more")

    if portfolio.positions:
        print(f"\n--- Open Positions ---")
        print(f"{'Symbol':<8} {'Shares':>8} {'Entry':>10} {'Current':>10} {'P&L':>10}")
        print("-" * 50)

        for symbol, pos in portfolio.positions.items():
            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100 if pos.current_price > 0 else 0
            print(f"{symbol:<8} {pos.shares:>8.2f} ${pos.entry_price:>9.2f} ${pos.current_price:>9.2f} {pnl_pct:>+9.1f}%")
    else:
        print(f"\n  No open positions (all cash)")

    if portfolio.trades:
        print(f"\n--- Recent Trades ---")
        for trade in portfolio.trades[-5:]:
            print(f"  {trade['date']}: {trade['action']} {trade['symbol']} @ ${trade['price']:.2f}")

    # Compound projection
    print(f"\n--- Compound Projection ---")
    equity = portfolio.get_equity()
    win_rate = 0.6  # Assume 60% win rate
    avg_win = CONFIG["take_profit_pct"]
    avg_loss = CONFIG["stop_loss_pct"]
    expected_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss

    projections = [10, 25, 50, 100]
    print(f"  If {win_rate:.0%} win rate, {avg_win:.0%} wins, {avg_loss:.0%} losses:")
    for trades in projections:
        projected = equity * ((1 + expected_per_trade) ** trades)
        print(f"    After {trades} trades: ${projected:,.2f}")


def cmd_record_trade(action: str, symbol: str, shares: float, price: float):
    """Record a trade (manual entry)."""
    portfolio = load_portfolio()

    if action.upper() == "BUY":
        cost = shares * price
        if cost > portfolio.capital:
            print(f"[ERROR] Not enough cash. Have ${portfolio.capital:.2f}, need ${cost:.2f}")
            return

        portfolio.capital -= cost
        portfolio.positions[symbol] = Position(
            symbol=symbol,
            entry_price=price,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            shares=shares,
            current_price=price,
        )
        portfolio.trades.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "action": "BUY",
            "symbol": symbol,
            "shares": shares,
            "price": price,
        })
        print(f"Recorded: BUY {shares} {symbol} @ ${price:.2f}")

    elif action.upper() == "SELL":
        if symbol not in portfolio.positions:
            print(f"[ERROR] No position in {symbol}")
            return

        pos = portfolio.positions[symbol]
        proceeds = shares * price
        portfolio.capital += proceeds

        pnl = (price - pos.entry_price) * shares
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

        if shares >= pos.shares:
            del portfolio.positions[symbol]
        else:
            pos.shares -= shares

        portfolio.trades.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "action": "SELL",
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })
        print(f"Recorded: SELL {shares} {symbol} @ ${price:.2f} (P&L: ${pnl:+.2f}, {pnl_pct:+.1f}%)")

    save_portfolio(portfolio)
    print(f"New cash balance: ${portfolio.capital:.2f}")


def cmd_reset(initial: float = 20.0):
    """Reset portfolio."""
    portfolio = Portfolio(
        capital=initial,
        initial_capital=initial,
        created=datetime.now().isoformat(),
    )
    save_portfolio(portfolio)
    print(f"Portfolio reset to ${initial:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = sys.argv[1:]

    if not args or "--help" in args:
        print("""
PENNY COMPOUNDER - Compound small gains into big money

Commands:
  --scan              Scan for opportunities
  --status            Show portfolio status
  --buy SYM QTY PRC   Record a buy (e.g., --buy SNDL 10 2.50)
  --sell SYM QTY PRC  Record a sell
  --reset [AMT]       Reset portfolio (default $20)

Examples:
  python penny_compounder.py --scan
  python penny_compounder.py --status
  python penny_compounder.py --buy SNDL 10 2.50
  python penny_compounder.py --sell SNDL 10 2.75
  python penny_compounder.py --reset 50
""")
        return

    if "--scan" in args:
        cmd_scan()

    elif "--status" in args:
        cmd_status()

    elif "--buy" in args:
        idx = args.index("--buy")
        if idx + 3 >= len(args):
            print("Usage: --buy SYMBOL SHARES PRICE")
            return
        symbol = args[idx + 1].upper()
        shares = float(args[idx + 2])
        price = float(args[idx + 3])
        cmd_record_trade("BUY", symbol, shares, price)

    elif "--sell" in args:
        idx = args.index("--sell")
        if idx + 3 >= len(args):
            print("Usage: --sell SYMBOL SHARES PRICE")
            return
        symbol = args[idx + 1].upper()
        shares = float(args[idx + 2])
        price = float(args[idx + 3])
        cmd_record_trade("SELL", symbol, shares, price)

    elif "--reset" in args:
        idx = args.index("--reset")
        initial = 20.0
        if idx + 1 < len(args):
            try:
                initial = float(args[idx + 1])
            except ValueError:
                pass
        cmd_reset(initial)

    else:
        print("Unknown command. Use --help for usage.")


if __name__ == "__main__":
    main()
