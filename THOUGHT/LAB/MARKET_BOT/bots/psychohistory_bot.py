"""
PSYCHOHISTORY BOT
=================

The main integration class for formula-driven trading.

Combines:
- Prime Radiant: Precomputed signal embeddings
- Formula Executor: R computation
- Seldon Gate: Decision gating

This bot does NOT reason about markets.
The formula decides, the bot executes.
"""

import numpy as np
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

# Add parent to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_vocabulary import SignalState, AssetClass, get_signal_by_id
from core import PrimeRadiant, MarketFormulaExecutor, FormulaResult, AlphaResult
from core.seldon_gate import SeldonGate, GateTier, FullGateResult, AlphaWarningLevel


# =============================================================================
# BOT CONFIGURATION
# =============================================================================

@dataclass
class BotConfig:
    """Configuration for Psychohistory bot."""
    initial_capital: float = 100000.0
    lookback_window: int = 10          # States to use as context
    alpha_window: int = 20             # Window for alpha computation
    default_tier: GateTier = GateTier.T2_MEDIUM_POS
    min_R_to_trade: float = 0.5        # Minimum R to consider trading
    max_position_pct: float = 0.30     # Maximum position as % of capital
    stop_loss_pct: float = 0.05        # Stop loss percentage


# =============================================================================
# TRADE RECORDS
# =============================================================================

@dataclass
class Trade:
    """Record of a trade."""
    timestamp: str
    asset: str
    action: str                 # "BUY", "SELL", "CLOSE"
    price: float
    size: float
    R_value: float
    gate_tier: str
    reason: str


@dataclass
class Position:
    """Current position in an asset."""
    asset: str
    size: float
    entry_price: float
    entry_time: str
    current_price: float
    unrealized_pnl: float


# =============================================================================
# PSYCHOHISTORY BOT
# =============================================================================

class PsychohistoryBot:
    """
    Formula-driven trading bot using Psychohistory principles.

    The bot does NOT:
    - Reason about market conditions
    - Use LLM judgment for decisions
    - Learn or adapt weights

    The bot DOES:
    - Convert market state to signal vectors
    - Compute R using validated formula
    - Gate decisions based on thresholds
    - Track positions and P&L
    """

    def __init__(self, config: Optional[BotConfig] = None, rebuild_cache: bool = False):
        """
        Initialize Psychohistory bot.

        Args:
            config: Bot configuration
            rebuild_cache: Whether to rebuild Prime Radiant embeddings
        """
        self.config = config or BotConfig()

        # Initialize components
        print("Initializing Psychohistory Bot...")
        print("  Loading Prime Radiant...")
        self.radiant = PrimeRadiant(rebuild=rebuild_cache)

        print("  Loading Formula Executor...")
        self.executor = MarketFormulaExecutor()

        print("  Loading Seldon Gate...")
        self.gate = SeldonGate()

        # State tracking
        self.state_history: List[SignalState] = []
        self.vector_history: List[np.ndarray] = []

        # Portfolio tracking
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [self.config.initial_capital]

        print("Psychohistory Bot ready.")

    # =========================================================================
    # MARKET STATE INPUT
    # =========================================================================

    def observe(self, state: SignalState, current_price: float) -> FullGateResult:
        """
        Observe market state and compute gate assessment.

        This is the main entry point for market data.

        Args:
            state: Current market signal state
            current_price: Current asset price

        Returns:
            FullGateResult with gate decision and drift warning
        """
        # Convert state to vector
        current_vec = self.radiant.state_to_vector(state)

        # Add to history
        self.state_history.append(state)
        self.vector_history.append(current_vec)

        # Limit history size
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-500:]
            self.vector_history = self.vector_history[-500:]

        # Get context vectors
        lookback = self.config.lookback_window
        context_vecs = self.vector_history[-lookback-1:-1] if len(self.vector_history) > 1 else []

        # Compute R
        formula_result = self.executor.compute_R(current_vec, context_vecs)

        # Compute alpha
        alpha_window = self.config.alpha_window
        alpha_vecs = self.vector_history[-alpha_window:] if len(self.vector_history) >= 3 else self.vector_history
        alpha_result = self.executor.compute_alpha(alpha_vecs)

        # Record in Prime Radiant
        self.radiant.record_regime(
            timestamp=state.timestamp,
            asset=state.asset,
            R_value=formula_result.R,
            alpha=alpha_result.alpha,
            Df=alpha_result.Df,
            gate_status="PENDING",
            regime=formula_result.regime,
            signals=state.signals,
        )

        # Gate assessment
        gate_result = self.gate.assess(
            R=formula_result.R,
            alpha=alpha_result.alpha,
            Df=alpha_result.Df,
            requested_tier=self.config.default_tier,
        )

        # Update position prices
        if state.asset in self.positions:
            pos = self.positions[state.asset]
            pos.current_price = current_price
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size

        return gate_result

    # =========================================================================
    # TRADING ACTIONS
    # =========================================================================

    def decide(
        self,
        state: SignalState,
        current_price: float,
        signal_direction: int = 0,  # -1=bearish, 0=neutral, +1=bullish
    ) -> Optional[Trade]:
        """
        Decide whether to trade based on gate result.

        Args:
            state: Current market signal state
            current_price: Current asset price
            signal_direction: External signal direction (from technical analysis, etc.)

        Returns:
            Trade if action taken, None otherwise
        """
        # Get gate assessment
        gate_result = self.observe(state, current_price)

        # Check if we should exit
        if gate_result.should_exit:
            return self._close_position(state.asset, current_price, gate_result, "GATE_EXIT")

        # Check drift warning
        if gate_result.drift.warning_level == AlphaWarningLevel.ALERT:
            # Reduce position
            if state.asset in self.positions:
                return self._reduce_position(state.asset, current_price, 0.5, gate_result, "DRIFT_REDUCE")

        # Gate is open - consider entry
        if gate_result.gate.status == "OPEN" and signal_direction != 0:
            # Compute position size
            max_position = gate_result.gate.position_limit

            # Adjust for drift
            adjusted_limit = self.gate.adjust_position_limit(
                max_position, gate_result.drift
            )

            if adjusted_limit > 0:
                # Compute size in units
                position_value = self.capital * adjusted_limit
                size = position_value / current_price

                if signal_direction > 0:
                    return self._open_position(
                        state.asset, current_price, size, gate_result, "SIGNAL_BUY"
                    )
                elif signal_direction < 0 and state.asset in self.positions:
                    return self._close_position(
                        state.asset, current_price, gate_result, "SIGNAL_SELL"
                    )

        return None

    def _open_position(
        self,
        asset: str,
        price: float,
        size: float,
        gate_result: FullGateResult,
        reason: str
    ) -> Trade:
        """Open a new position."""
        # Close existing position if any
        if asset in self.positions:
            self._close_position(asset, price, gate_result, "REPLACE")

        # Open new position
        self.positions[asset] = Position(
            asset=asset,
            size=size,
            entry_price=price,
            entry_time=gate_result.timestamp,
            current_price=price,
            unrealized_pnl=0.0,
        )

        trade = Trade(
            timestamp=gate_result.timestamp,
            asset=asset,
            action="BUY",
            price=price,
            size=size,
            R_value=gate_result.gate.R,
            gate_tier=gate_result.gate.tier.value,
            reason=reason,
        )
        self.trades.append(trade)

        return trade

    def _close_position(
        self,
        asset: str,
        price: float,
        gate_result: FullGateResult,
        reason: str
    ) -> Optional[Trade]:
        """Close an existing position."""
        if asset not in self.positions:
            return None

        pos = self.positions[asset]
        pnl = (price - pos.entry_price) * pos.size
        self.capital += pnl

        trade = Trade(
            timestamp=gate_result.timestamp,
            asset=asset,
            action="CLOSE",
            price=price,
            size=pos.size,
            R_value=gate_result.gate.R,
            gate_tier=gate_result.gate.tier.value,
            reason=f"{reason} (PnL: {pnl:+.2f})",
        )
        self.trades.append(trade)

        del self.positions[asset]

        return trade

    def _reduce_position(
        self,
        asset: str,
        price: float,
        reduce_pct: float,
        gate_result: FullGateResult,
        reason: str
    ) -> Optional[Trade]:
        """Reduce position size."""
        if asset not in self.positions:
            return None

        pos = self.positions[asset]
        reduce_size = pos.size * reduce_pct

        pnl = (price - pos.entry_price) * reduce_size
        self.capital += pnl

        pos.size -= reduce_size

        if pos.size < 0.001:
            # Close entirely
            return self._close_position(asset, price, gate_result, reason)

        trade = Trade(
            timestamp=gate_result.timestamp,
            asset=asset,
            action="REDUCE",
            price=price,
            size=reduce_size,
            R_value=gate_result.gate.R,
            gate_tier=gate_result.gate.tier.value,
            reason=f"{reason} (PnL: {pnl:+.2f})",
        )
        self.trades.append(trade)

        return trade

    # =========================================================================
    # PORTFOLIO TRACKING
    # =========================================================================

    def get_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Get current total equity.

        Args:
            current_prices: Dict of asset -> current price
        """
        equity = self.capital

        for asset, pos in self.positions.items():
            if current_prices and asset in current_prices:
                pos.current_price = current_prices[asset]
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.size

            equity += pos.unrealized_pnl

        return equity

    def record_equity(self, current_prices: Optional[Dict[str, float]] = None):
        """Record equity point for curve."""
        equity = self.get_equity(current_prices)
        self.equity_curve.append(equity)

    def get_max_drawdown(self) -> float:
        """Compute maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown))

    def get_returns(self) -> float:
        """Get total returns."""
        if len(self.equity_curve) < 2:
            return 0.0

        return (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]

    # =========================================================================
    # STATE INSPECTION
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get bot statistics."""
        return {
            "capital": self.capital,
            "equity": self.get_equity(),
            "n_positions": len(self.positions),
            "n_trades": len(self.trades),
            "n_states_observed": len(self.state_history),
            "returns": self.get_returns(),
            "max_drawdown": self.get_max_drawdown(),
            "gate_stats": self.gate.get_stats(),
            "radiant_stats": self.radiant.get_stats(),
        }

    def get_position_summary(self) -> List[Dict]:
        """Get summary of current positions."""
        return [
            {
                "asset": pos.asset,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_pct": (pos.current_price - pos.entry_price) / pos.entry_price * 100,
            }
            for pos in self.positions.values()
        ]

    def get_trade_history(self, limit: int = 10) -> List[Dict]:
        """Get recent trade history."""
        return [
            {
                "timestamp": t.timestamp,
                "asset": t.asset,
                "action": t.action,
                "price": t.price,
                "size": t.size,
                "R": t.R_value,
                "tier": t.gate_tier,
                "reason": t.reason,
            }
            for t in self.trades[-limit:]
        ]

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self):
        """Reset bot to initial state."""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.state_history = []
        self.vector_history = []
        self.gate = SeldonGate()


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PSYCHOHISTORY BOT - Demo")
    print("=" * 60)

    # Initialize bot
    bot = PsychohistoryBot()

    # Create sample states
    states = [
        SignalState(
            signals={"trend_up": 0.8, "volume_surge": 0.6, "bullish_news": 0.7},
            timestamp=datetime.now().isoformat(),
            asset="SPY",
            asset_class=AssetClass.ALL,
        ),
        SignalState(
            signals={"trend_up": 0.7, "volume_surge": 0.5, "bullish_news": 0.6},
            timestamp=datetime.now().isoformat(),
            asset="SPY",
            asset_class=AssetClass.ALL,
        ),
        SignalState(
            signals={"sideways": 0.5, "vol_expanding": 0.6, "mixed_news": 0.5},
            timestamp=datetime.now().isoformat(),
            asset="SPY",
            asset_class=AssetClass.ALL,
        ),
        SignalState(
            signals={"trend_down": 0.7, "vol_spike": 0.8, "bearish_news": 0.8},
            timestamp=datetime.now().isoformat(),
            asset="SPY",
            asset_class=AssetClass.ALL,
        ),
    ]

    prices = [450.0, 452.0, 451.0, 445.0]
    directions = [1, 1, 0, -1]  # bullish, bullish, neutral, bearish

    print("\n--- Running simulation ---")
    for i, (state, price, direction) in enumerate(zip(states, prices, directions)):
        print(f"\nStep {i+1}:")
        print(f"  Signals: {list(state.signals.keys())}")
        print(f"  Price: ${price:.2f}")

        trade = bot.decide(state, price, direction)

        if trade:
            print(f"  Trade: {trade.action} | R={trade.R_value:.2f} | {trade.reason}")
        else:
            gate_result = bot.observe(state, price)
            print(f"  No trade | R={gate_result.gate.R:.2f} | Gate={gate_result.gate.status}")

        bot.record_equity({state.asset: price})

    # Final stats
    print("\n--- Final Stats ---")
    stats = bot.get_stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Trade History ---")
    for trade in bot.get_trade_history():
        print(f"  {trade['action']} @ ${trade['price']:.2f} | R={trade['R']:.2f} | {trade['reason']}")

    print("\n--- Psychohistory Bot Demo Complete ---")
