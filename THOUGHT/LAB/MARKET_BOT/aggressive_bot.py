"""
AGGRESSIVE PSYCHOHISTORY BOT
============================

Same R-gating protection, but smarter entry signals.

The original bot is too conservative - it protects capital but
misses 95% of bull market gains. This version:

1. Uses trend-following for entries (not just 5-day momentum)
2. Position sizes based on R value (higher R = more confident)
3. Only exits on actual danger signals (alpha drift, gate close)
4. Stays in trends longer

The formula still decides WHEN to exit. We just enter more often.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from signal_vocabulary import SignalState, AssetClass
from prime_radiant import PrimeRadiant
from formula_executor import MarketFormulaExecutor
from seldon_gate import SeldonGate, GateTier, AlphaWarningLevel


@dataclass
class AggressiveConfig:
    """Configuration for aggressive bot."""
    initial_capital: float = 100000.0

    # Trend detection
    fast_ma: int = 10           # Fast moving average
    slow_ma: int = 30           # Slow moving average
    trend_threshold: float = 0.01  # 1% above/below MA to confirm trend

    # Position sizing
    base_position: float = 0.15    # Base position size (15% of capital)
    max_position: float = 0.40     # Max position size (40% of capital)
    r_scale_factor: float = 0.1    # How much R affects position size

    # Exit conditions
    stop_loss: float = 0.03        # 3% stop loss
    trailing_stop: float = 0.05    # 5% trailing stop

    # R thresholds
    min_r_entry: float = 0.3       # Minimum R to consider entry
    exit_on_alpha_warning: bool = True  # Exit on alpha drift warning


@dataclass
class TrendSignal:
    """Trend detection result."""
    direction: int      # -1, 0, +1
    strength: float     # 0 to 1
    fast_ma: float
    slow_ma: float
    price_vs_ma: float  # % above/below slow MA
    rsi: float
    breakout: bool
    breakdown: bool


class AggressiveBot:
    """
    Psychohistory bot with aggressive entries.

    Still uses R-gating for protection, but enters more aggressively
    when trend signals are strong.
    """

    def __init__(self, config: Optional[AggressiveConfig] = None):
        self.config = config or AggressiveConfig()

        # Core components (same as conservative bot)
        self.radiant = PrimeRadiant()
        self.executor = MarketFormulaExecutor()
        self.gate = SeldonGate()

        # State tracking
        self.prices: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.volumes: List[float] = []
        self.state_history: List[SignalState] = []
        self.vector_history: List[np.ndarray] = []

        # Position tracking
        self.capital = self.config.initial_capital
        self.position_size = 0.0
        self.entry_price = 0.0
        self.peak_price = 0.0  # For trailing stop
        self.in_position = False

        # Performance
        self.equity_curve = [self.config.initial_capital]
        self.trades: List[Dict] = []

    def compute_trend(self) -> TrendSignal:
        """
        Compute trend signal from price history.

        Uses multiple indicators for confirmation:
        - MA crossover (fast vs slow)
        - Price position relative to MA
        - RSI for momentum confirmation
        - Recent breakout/breakdown
        """
        if len(self.prices) < self.config.slow_ma + 10:
            return TrendSignal(0, 0.0, 0, 0, 0, 50, False, False)

        prices = np.array(self.prices)

        # Moving averages
        fast_ma = np.mean(prices[-self.config.fast_ma:])
        slow_ma = np.mean(prices[-self.config.slow_ma:])

        current_price = prices[-1]
        price_vs_ma = (current_price - slow_ma) / slow_ma

        # RSI
        rsi = self._compute_rsi(prices, 14)

        # Breakout/breakdown detection
        lookback = 20
        recent_high = np.max(self.highs[-lookback:-1]) if len(self.highs) > lookback else self.highs[-1]
        recent_low = np.min(self.lows[-lookback:-1]) if len(self.lows) > lookback else self.lows[-1]

        breakout = current_price > recent_high * 1.01  # 1% above recent high
        breakdown = current_price < recent_low * 0.99  # 1% below recent low

        # Trend direction
        direction = 0
        strength = 0.0

        # Bullish conditions
        bullish_signals = 0
        if fast_ma > slow_ma:
            bullish_signals += 1
        if current_price > slow_ma * (1 + self.config.trend_threshold):
            bullish_signals += 1
        if rsi > 50 and rsi < 70:  # Bullish momentum, not overbought
            bullish_signals += 1
        if breakout:
            bullish_signals += 1

        # Bearish conditions
        bearish_signals = 0
        if fast_ma < slow_ma:
            bearish_signals += 1
        if current_price < slow_ma * (1 - self.config.trend_threshold):
            bearish_signals += 1
        if rsi < 50 and rsi > 30:  # Bearish momentum, not oversold
            bearish_signals += 1
        if breakdown:
            bearish_signals += 1

        if bullish_signals >= 3:
            direction = 1
            strength = min(bullish_signals / 4, 1.0)
        elif bearish_signals >= 3:
            direction = -1
            strength = min(bearish_signals / 4, 1.0)
        else:
            direction = 0
            strength = 0.0

        return TrendSignal(
            direction=direction,
            strength=strength,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            price_vs_ma=price_vs_ma,
            rsi=rsi,
            breakout=breakout,
            breakdown=breakdown,
        )

    def _compute_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Compute RSI."""
        if len(prices) < window + 1:
            return 50.0

        deltas = np.diff(prices[-(window+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def compute_position_size(self, R: float, trend_strength: float) -> float:
        """
        Compute position size based on R value and trend strength.

        Higher R = more confident = larger position
        Stronger trend = larger position
        """
        # Base position
        base = self.config.base_position

        # Scale by R (higher R = more coherent signals = more confident)
        r_adjustment = min(R * self.config.r_scale_factor, 0.15)

        # Scale by trend strength
        trend_adjustment = trend_strength * 0.1

        position_pct = base + r_adjustment + trend_adjustment

        # Cap at max
        position_pct = min(position_pct, self.config.max_position)

        return position_pct

    def update(self, state: SignalState, price: float, high: float, low: float, volume: float) -> Dict:
        """
        Update bot with new market data and get decision.

        Returns dict with:
        - action: "BUY", "SELL", "HOLD"
        - reason: Why
        - metrics: R, alpha, trend, etc.
        """
        # Update history
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.state_history.append(state)

        # Convert state to vector
        vec = self.radiant.state_to_vector(state)
        self.vector_history.append(vec)

        # Limit history
        max_history = 500
        if len(self.prices) > max_history:
            self.prices = self.prices[-max_history:]
            self.highs = self.highs[-max_history:]
            self.lows = self.lows[-max_history:]
            self.volumes = self.volumes[-max_history:]
            self.state_history = self.state_history[-max_history:]
            self.vector_history = self.vector_history[-max_history:]

        # Compute R
        context_vecs = self.vector_history[-11:-1] if len(self.vector_history) > 1 else []
        formula_result = self.executor.compute_R(vec, context_vecs)
        R = formula_result.R

        # Compute alpha
        alpha_vecs = self.vector_history[-20:] if len(self.vector_history) >= 3 else self.vector_history
        alpha_result = self.executor.compute_alpha(alpha_vecs)

        # Gate assessment
        gate_result = self.gate.assess(
            R=R,
            alpha=alpha_result.alpha,
            Df=alpha_result.Df,
            requested_tier=GateTier.T2_MEDIUM_POS,
        )

        # Compute trend
        trend = self.compute_trend()

        # Decision logic
        action = "HOLD"
        reason = ""

        # Update equity
        if self.in_position:
            equity = self.capital + self.position_size * price
            # Update peak for trailing stop
            if price > self.peak_price:
                self.peak_price = price
        else:
            equity = self.capital

        self.equity_curve.append(equity)

        # EXIT CONDITIONS (checked first - protection is priority)
        if self.in_position:
            pnl_pct = (price - self.entry_price) / self.entry_price

            # Stop loss
            if pnl_pct < -self.config.stop_loss:
                action = "SELL"
                reason = f"STOP_LOSS ({pnl_pct:.1%})"

            # Trailing stop
            elif self.peak_price > 0:
                drawdown_from_peak = (self.peak_price - price) / self.peak_price
                if drawdown_from_peak > self.config.trailing_stop:
                    action = "SELL"
                    reason = f"TRAILING_STOP ({drawdown_from_peak:.1%} from peak)"

            # Alpha drift warning
            if self.config.exit_on_alpha_warning:
                if gate_result.drift.warning_level in [AlphaWarningLevel.ALERT, AlphaWarningLevel.CRITICAL]:
                    action = "SELL"
                    reason = f"ALPHA_DRIFT ({gate_result.drift.warning_level.name})"

            # Gate closed
            if gate_result.gate.status == "CLOSED":
                action = "SELL"
                reason = f"GATE_CLOSED (R={R:.2f})"

            # Trend reversal
            if trend.direction == -1 and trend.strength > 0.5:
                action = "SELL"
                reason = f"TREND_REVERSAL (strength={trend.strength:.1f})"

        # ENTRY CONDITIONS (only if not in position and not exiting)
        if not self.in_position and action == "HOLD":
            # Need: gate open, R above threshold, bullish trend
            if gate_result.gate.status == "OPEN":
                if R >= self.config.min_r_entry:
                    if trend.direction == 1 and trend.strength >= 0.5:
                        # Compute position size
                        position_pct = self.compute_position_size(R, trend.strength)
                        position_value = self.capital * position_pct

                        action = "BUY"
                        reason = f"TREND_ENTRY (R={R:.2f}, trend={trend.strength:.1f})"

        # Execute action
        if action == "BUY":
            position_pct = self.compute_position_size(R, trend.strength)
            position_value = self.capital * position_pct
            self.position_size = position_value / price
            self.entry_price = price
            self.peak_price = price
            self.capital -= position_value
            self.in_position = True

            self.trades.append({
                "action": "BUY",
                "price": price,
                "size": self.position_size,
                "value": position_value,
                "R": R,
                "trend_strength": trend.strength,
                "reason": reason,
            })

        elif action == "SELL":
            pnl = (price - self.entry_price) * self.position_size
            self.capital += self.position_size * price

            self.trades.append({
                "action": "SELL",
                "price": price,
                "size": self.position_size,
                "entry_price": self.entry_price,
                "pnl": pnl,
                "pnl_pct": (price - self.entry_price) / self.entry_price,
                "R": R,
                "reason": reason,
            })

            self.position_size = 0.0
            self.entry_price = 0.0
            self.peak_price = 0.0
            self.in_position = False

        return {
            "action": action,
            "reason": reason,
            "R": R,
            "alpha": alpha_result.alpha,
            "gate_status": gate_result.gate.status,
            "drift_warning": gate_result.drift.warning_level.name,
            "trend_direction": trend.direction,
            "trend_strength": trend.strength,
            "rsi": trend.rsi,
            "price_vs_ma": trend.price_vs_ma,
            "in_position": self.in_position,
            "equity": equity,
        }

    def get_returns(self) -> float:
        """Get total returns."""
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown))

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        winning_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("pnl", 0) < 0]

        return {
            "total_return": self.get_returns(),
            "max_drawdown": self.get_max_drawdown(),
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
            "avg_win": np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0,
            "current_equity": self.equity_curve[-1] if self.equity_curve else self.config.initial_capital,
            "in_position": self.in_position,
        }


# =============================================================================
# BACKTEST COMPARISON
# =============================================================================

def backtest_comparison(symbol: str = "SPY", start: str = "2017-01-01", end: str = "2022-12-31"):
    """
    Backtest aggressive bot and compare to conservative + B&H.
    """
    from real_data_ingest import RealDataFetcher
    from signal_extractor import SignalExtractor
    from psychohistory_bot import PsychohistoryBot, BotConfig

    print("=" * 70)
    print(f"AGGRESSIVE VS CONSERVATIVE BACKTEST: {symbol}")
    print(f"Period: {start} to {end}")
    print("=" * 70)

    # Fetch data
    fetcher = RealDataFetcher()
    data = fetcher.fetch(symbol, start, end)
    print(f"Bars: {len(data.bars)}")

    # B&H
    bh_return = (data.bars[-1].close - data.bars[0].close) / data.bars[0].close
    bh_equity = np.array([b.close for b in data.bars])
    bh_peak = np.maximum.accumulate(bh_equity)
    bh_dd = np.max((bh_peak - bh_equity) / bh_peak)

    print(f"\nBuy & Hold: Return={bh_return:.1%}, Max DD={bh_dd:.1%}")

    # Conservative bot
    print("\nRunning Conservative Bot...")
    extractor = SignalExtractor()
    config = BotConfig(initial_capital=100000)
    conservative = PsychohistoryBot(config)

    prices = []
    states = extractor.extract_sequence(data, start_index=30)

    for i, state in enumerate(states):
        bar_idx = i + 30
        price = data.bars[bar_idx].close
        prices.append(price)

        if len(prices) > 5:
            recent_return = (prices[-1] - prices[-5]) / prices[-5]
            direction = 1 if recent_return > 0.005 else (-1 if recent_return < -0.005 else 0)
        else:
            direction = 0

        conservative.decide(state, price, direction)
        conservative.record_equity({state.asset: price})

    cons_return = conservative.get_returns()
    cons_dd = conservative.get_max_drawdown()

    print(f"Conservative: Return={cons_return:.1%}, Max DD={cons_dd:.1%}, Trades={len(conservative.trades)}")

    # Aggressive bot
    print("\nRunning Aggressive Bot...")
    aggressive = AggressiveBot()

    for i, state in enumerate(states):
        bar_idx = i + 30
        bar = data.bars[bar_idx]
        aggressive.update(state, bar.close, bar.high, bar.low, bar.volume)

    agg_stats = aggressive.get_stats()

    print(f"Aggressive: Return={agg_stats['total_return']:.1%}, Max DD={agg_stats['max_drawdown']:.1%}, Trades={agg_stats['total_trades']}")
    print(f"  Win Rate: {agg_stats['win_rate']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Return':>12} {'Max DD':>12} {'Risk-Adj':>12}")
    print("-" * 70)
    print(f"{'Buy & Hold':<20} {bh_return:>11.1%} {bh_dd:>11.1%} {bh_return/bh_dd if bh_dd > 0 else 0:>11.2f}")
    print(f"{'Conservative':<20} {cons_return:>11.1%} {cons_dd:>11.1%} {cons_return/cons_dd if cons_dd > 0 else 0:>11.2f}")
    print(f"{'Aggressive':<20} {agg_stats['total_return']:>11.1%} {agg_stats['max_drawdown']:>11.1%} {agg_stats['total_return']/agg_stats['max_drawdown'] if agg_stats['max_drawdown'] > 0 else 0:>11.2f}")

    return {
        "bh": {"return": bh_return, "max_dd": bh_dd},
        "conservative": {"return": cons_return, "max_dd": cons_dd},
        "aggressive": {"return": agg_stats["total_return"], "max_dd": agg_stats["max_drawdown"]},
    }


if __name__ == "__main__":
    # Test on different periods
    print("\n" + "=" * 70)
    print("TEST 1: Full Period (2017-2022)")
    print("=" * 70)
    backtest_comparison("SPY", "2017-01-01", "2022-12-31")

    print("\n" + "=" * 70)
    print("TEST 2: Bull Market (2019)")
    print("=" * 70)
    backtest_comparison("SPY", "2019-01-01", "2019-12-31")

    print("\n" + "=" * 70)
    print("TEST 3: COVID Crash + Recovery (2020)")
    print("=" * 70)
    backtest_comparison("SPY", "2020-01-01", "2020-12-31")

    print("\n" + "=" * 70)
    print("TEST 4: Bear Market (2022)")
    print("=" * 70)
    backtest_comparison("SPY", "2022-01-01", "2022-12-31")
