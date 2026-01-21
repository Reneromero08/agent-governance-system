"""
CATALYTIC MARKET BOT
====================

A market analysis bot that uses:
- Compass AGI for paradigm/regime detection
- cat_chat for context-aware reasoning
- Paper trading for safe experimentation

The key insight: Don't predict price, predict WHEN prediction works.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import json
import sys
from pathlib import Path

# Add paths for imports
LAB_PATH = Path(__file__).parent.parent
CAT_CHAT_PATH = LAB_PATH / "CAT_CHAT"
COMPASS_PATH = CAT_CHAT_PATH / "compass_agi"
CATALYTIC_PATH = CAT_CHAT_PATH / "catalytic_chat"

for p in [str(COMPASS_PATH), str(CATALYTIC_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import compass components
try:
    from realtime_paradigm_detector import ParadigmShiftDetector, SHIFT_GEODESICS, STABILITY_GEODESICS
    COMPASS_AVAILABLE = True
except ImportError:
    COMPASS_AVAILABLE = False
    print("Warning: Compass not available, using mock detector")

# Import cat_chat components
try:
    from context_partitioner import ContextPartitioner, ContextItem
    CAT_CHAT_AVAILABLE = True
except ImportError:
    CAT_CHAT_AVAILABLE = False
    print("Warning: cat_chat not available, using simplified context")


# =============================================================================
# Enums and Data Classes
# =============================================================================

class Regime(Enum):
    STABLE = "STABLE"
    TRANSITIONAL = "TRANSITIONAL"
    SHIFT = "SHIFT"


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    HEDGE = "HEDGE"


@dataclass
class MarketState:
    """Current market state snapshot."""
    timestamp: str
    price: float
    prices_history: List[float]
    headlines: List[str]
    regime: Regime
    shift_score: float
    dominant_geodesics: List[Tuple[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "regime": self.regime.value,
            "shift_score": self.shift_score,
            "dominant_geodesics": self.dominant_geodesics[:3],
        }


@dataclass
class BotDecision:
    """Bot's decision with reasoning."""
    action: Action
    confidence: float  # 0-1
    position_size: float  # 0-1 (fraction of capital)
    reasoning: str
    context_used: List[str]  # What context was hydrated
    regime: Regime
    timestamp: str


@dataclass
class Position:
    """Current position."""
    size: float  # Positive = long, negative = short, 0 = flat
    entry_price: float
    entry_time: str
    unrealized_pnl: float = 0.0


# =============================================================================
# Context Items for cat_chat
# =============================================================================

class MarketContextBuilder:
    """Builds context items for cat_chat from market data."""

    def __init__(self):
        self.historical_crises = [
            ContextItem(
                item_id="crisis_2008",
                content="2008 Financial Crisis: Started with subprime mortgage defaults. "
                        "RSI showed oversold for months while prices kept falling. "
                        "Traditional buy signals failed. Recovery took years.",
                tokens=50,
                item_type="historical",
                metadata={"regime": "SHIFT", "year": 2008}
            ),
            ContextItem(
                item_id="crisis_2020",
                content="2020 COVID Crash: Markets fell 35% in weeks. "
                        "Early warnings in January (unusual China news). "
                        "V-shaped recovery surprised most. Regime shift detected early.",
                tokens=50,
                item_type="historical",
                metadata={"regime": "SHIFT", "year": 2020}
            ),
            ContextItem(
                item_id="crisis_dotcom",
                content="2000 Dotcom Bubble: Gradual decline over 2 years. "
                        "Multiple 'buy the dip' failures. Fundamentals ignored for too long. "
                        "Recovery took until 2007.",
                tokens=50,
                item_type="historical",
                metadata={"regime": "SHIFT", "year": 2000}
            ),
        ]

        self.stable_patterns = [
            ContextItem(
                item_id="stable_trend",
                content="Trend Following: In stable regimes, momentum works. "
                        "Buy above 50-day MA, sell below. RSI 30/70 levels reliable. "
                        "Mean reversion works.",
                tokens=40,
                item_type="strategy",
                metadata={"regime": "STABLE"}
            ),
            ContextItem(
                item_id="stable_earnings",
                content="Earnings Season: Stocks typically rise into earnings. "
                        "Beat = gap up, miss = gap down. Predictable patterns.",
                tokens=30,
                item_type="strategy",
                metadata={"regime": "STABLE"}
            ),
        ]

        self.warning_signs = [
            ContextItem(
                item_id="warning_vix",
                content="VIX Spike: When VIX rises above 25, volatility increasing. "
                        "Above 35 = fear. Above 50 = panic. Reduce position sizes.",
                tokens=35,
                item_type="indicator",
                metadata={"regime": "TRANSITIONAL"}
            ),
            ContextItem(
                item_id="warning_yield",
                content="Yield Curve: Inversion predicts recession 12-18 months out. "
                        "Watch 2y/10y spread. Inversion = warning sign.",
                tokens=35,
                item_type="indicator",
                metadata={"regime": "TRANSITIONAL"}
            ),
        ]

    def get_context_for_regime(self, regime: Regime) -> List[ContextItem]:
        """Get appropriate context items based on regime."""
        if regime == Regime.SHIFT:
            # During shift: prioritize historical crisis knowledge
            return self.historical_crises + self.warning_signs
        elif regime == Regime.TRANSITIONAL:
            # During transition: balanced context
            return self.warning_signs + self.historical_crises[:1] + self.stable_patterns[:1]
        else:
            # During stable: focus on strategies
            return self.stable_patterns + self.warning_signs[:1]

    def build_current_state_item(self, state: MarketState) -> ContextItem:
        """Build context item from current market state."""
        content = f"""Current Market State:
- Price: ${state.price:.2f}
- Regime: {state.regime.value}
- Shift Score: {state.shift_score:+.3f}
- Top Geodesics: {', '.join(f'{g[0]}({g[1]:.2f})' for g in state.dominant_geodesics[:3])}
- Recent Headlines: {'; '.join(state.headlines[:3])}
"""
        return ContextItem(
            item_id="current_state",
            content=content,
            tokens=len(content) // 4,
            item_type="current",
            metadata={"timestamp": state.timestamp}
        )


# =============================================================================
# The Market Bot
# =============================================================================

class CatalyticMarketBot:
    """
    A market bot that reasons about positions using paradigm-aware context.

    Key principle: Different regimes require different reasoning.
    - STABLE: "What do the technicals say?"
    - SHIFT: "What does history tell us about crises?"
    - TRANSITIONAL: "What are the warning signs?"
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.5,  # Max 50% of capital in position
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct

        # Position tracking
        self.position: Optional[Position] = None
        self.trade_history: List[Dict] = []
        self.decision_history: List[BotDecision] = []

        # Paradigm detector
        if COMPASS_AVAILABLE:
            self.detector = ParadigmShiftDetector()
        else:
            self.detector = None

        # Context builder
        self.context_builder = MarketContextBuilder()

        # Regime-specific parameters
        self.regime_params = {
            Regime.STABLE: {
                "max_position": 1.0,  # Full position allowed
                "stop_loss_pct": 0.05,  # 5% stop
                "use_technicals": True,
            },
            Regime.TRANSITIONAL: {
                "max_position": 0.5,  # Half position
                "stop_loss_pct": 0.03,  # Tighter stop
                "use_technicals": True,
            },
            Regime.SHIFT: {
                "max_position": 0.2,  # Minimal position
                "stop_loss_pct": 0.02,  # Very tight stop
                "use_technicals": False,  # Don't trust technicals
            },
        }

    def detect_regime(self, headlines: List[str]) -> Tuple[Regime, float, List[Tuple[str, float]]]:
        """Detect current market regime from headlines."""
        if self.detector is None:
            # Mock detector
            return Regime.STABLE, 0.0, [("Mock", 0.5)]

        result = self.detector.detect_shift(headlines)

        shift_type = result['shift_type']
        if shift_type == 'SHIFT':
            regime = Regime.SHIFT
        elif shift_type == 'STABLE':
            regime = Regime.STABLE
        else:
            regime = Regime.TRANSITIONAL

        return regime, result['shift_score'], result['top_geodesics'][:3]

    def compute_technicals(self, prices: List[float]) -> Dict[str, Any]:
        """Compute basic technical indicators."""
        if len(prices) < 5:
            return {"signal": "INSUFFICIENT_DATA"}

        prices = np.array(prices)

        # Simple momentum
        momentum = (prices[-1] - prices[-5]) / prices[-5]

        # Moving average trend
        if len(prices) >= 20:
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices[-20:])
            trend = "UP" if sma_short > sma_long else "DOWN"
        else:
            trend = "NEUTRAL"

        # Simple RSI approximation
        if len(prices) >= 14:
            deltas = np.diff(prices[-15:])
            gains = np.sum(deltas[deltas > 0])
            losses = -np.sum(deltas[deltas < 0])
            if losses == 0:
                rsi = 100
            else:
                rsi = 100 - (100 / (1 + gains / losses))
        else:
            rsi = 50

        # Signal
        if rsi < 30 and trend == "DOWN":
            signal = "OVERSOLD"
        elif rsi > 70 and trend == "UP":
            signal = "OVERBOUGHT"
        elif trend == "UP" and momentum > 0:
            signal = "BULLISH"
        elif trend == "DOWN" and momentum < 0:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "momentum": momentum,
            "trend": trend,
            "rsi": rsi,
        }

    def reason_about_market(self, state: MarketState, technicals: Dict) -> BotDecision:
        """
        Use cat_chat-style reasoning to make a decision.

        This is where paradigm-aware context selection matters:
        - In STABLE: Focus on technicals and strategies
        - In SHIFT: Focus on historical crises and caution
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        params = self.regime_params[state.regime]

        # Get appropriate context for regime
        context_items = self.context_builder.get_context_for_regime(state.regime)
        current_state_item = self.context_builder.build_current_state_item(state)
        context_items = [current_state_item] + context_items

        context_used = [item.item_id for item in context_items]

        # Reasoning based on regime
        if state.regime == Regime.SHIFT:
            # SHIFT: Prioritize safety, ignore technicals
            reasoning_parts = [
                f"SHIFT REGIME DETECTED (score: {state.shift_score:+.3f})",
                f"Dominant geodesic: {state.dominant_geodesics[0][0]} (disruption/change)",
                "Historical context: Crises can extend far beyond 'oversold' levels",
                "Technical signals UNRELIABLE during paradigm shifts",
                "Recommendation: DEFENSIVE posture, reduce exposure",
            ]

            if self.position and self.position.size > 0:
                action = Action.REDUCE
                position_size = params["max_position"]
            else:
                action = Action.HOLD
                position_size = 0.0

            confidence = 0.8  # High confidence in defensive stance

        elif state.regime == Regime.TRANSITIONAL:
            # TRANSITIONAL: Cautious, watch for developments
            reasoning_parts = [
                f"TRANSITIONAL REGIME (score: {state.shift_score:+.3f})",
                f"Market between stable and shift states",
                f"Technical signal: {technicals.get('signal', 'N/A')}",
                "Recommendation: Reduced position size, tighter stops",
            ]

            tech_signal = technicals.get("signal", "NEUTRAL")
            if tech_signal == "BULLISH" and not self.position:
                action = Action.BUY
                position_size = params["max_position"] * 0.5
            elif tech_signal == "BEARISH" and self.position:
                action = Action.REDUCE
                position_size = params["max_position"]
            else:
                action = Action.HOLD
                position_size = params["max_position"] if self.position else 0.0

            confidence = 0.5  # Lower confidence in transitional state

        else:
            # STABLE: Trust technicals, normal trading
            reasoning_parts = [
                f"STABLE REGIME (score: {state.shift_score:+.3f})",
                f"Dominant geodesic: {state.dominant_geodesics[0][0]} (stability)",
                f"Technical signal: {technicals.get('signal', 'N/A')}",
                f"Trend: {technicals.get('trend', 'N/A')}, RSI: {technicals.get('rsi', 0):.1f}",
                "Traditional analysis APPLICABLE",
            ]

            tech_signal = technicals.get("signal", "NEUTRAL")
            if tech_signal in ["BULLISH", "OVERSOLD"]:
                action = Action.BUY
                position_size = params["max_position"]
                reasoning_parts.append(f"BUY signal: {tech_signal}")
            elif tech_signal in ["BEARISH", "OVERBOUGHT"]:
                action = Action.SELL
                position_size = 0.0
                reasoning_parts.append(f"SELL signal: {tech_signal}")
            else:
                action = Action.HOLD
                position_size = params["max_position"] if self.position else 0.0

            confidence = 0.7

        return BotDecision(
            action=action,
            confidence=confidence,
            position_size=position_size,
            reasoning=" | ".join(reasoning_parts),
            context_used=context_used,
            regime=state.regime,
            timestamp=timestamp,
        )

    def process_tick(
        self,
        price: float,
        headlines: List[str],
        prices_history: List[float],
    ) -> BotDecision:
        """
        Process a market tick and return a decision.

        This is the main entry point for the bot.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Detect regime from headlines
        regime, shift_score, geodesics = self.detect_regime(headlines)

        # 2. Build market state
        state = MarketState(
            timestamp=timestamp,
            price=price,
            prices_history=prices_history,
            headlines=headlines,
            regime=regime,
            shift_score=shift_score,
            dominant_geodesics=geodesics,
        )

        # 3. Compute technicals (may be ignored in SHIFT regime)
        technicals = self.compute_technicals(prices_history)

        # 4. Reason about market with paradigm-aware context
        decision = self.reason_about_market(state, technicals)

        # 5. Store decision
        self.decision_history.append(decision)

        return decision

    def execute_decision(self, decision: BotDecision, price: float) -> Optional[Dict]:
        """Execute a decision (paper trading)."""
        trade = None

        if decision.action == Action.BUY and not self.position:
            # Open long position
            size = self.capital * decision.position_size * self.max_position_pct / price
            self.position = Position(
                size=size,
                entry_price=price,
                entry_time=decision.timestamp,
            )
            trade = {
                "action": "BUY",
                "size": size,
                "price": price,
                "timestamp": decision.timestamp,
            }

        elif decision.action == Action.SELL and self.position:
            # Close position
            pnl = (price - self.position.entry_price) * self.position.size
            self.capital += pnl
            trade = {
                "action": "SELL",
                "size": self.position.size,
                "price": price,
                "pnl": pnl,
                "timestamp": decision.timestamp,
            }
            self.position = None

        elif decision.action == Action.REDUCE and self.position:
            # Reduce position
            reduce_pct = 0.5  # Reduce by half
            reduce_size = self.position.size * reduce_pct
            pnl = (price - self.position.entry_price) * reduce_size
            self.capital += pnl
            self.position.size -= reduce_size
            trade = {
                "action": "REDUCE",
                "size": reduce_size,
                "price": price,
                "pnl": pnl,
                "timestamp": decision.timestamp,
            }
            if self.position.size < 0.01:  # Close if too small
                self.position = None

        if trade:
            self.trade_history.append(trade)

        return trade

    def get_status(self, current_price: float) -> Dict[str, Any]:
        """Get current bot status."""
        unrealized_pnl = 0.0
        if self.position:
            unrealized_pnl = (current_price - self.position.entry_price) * self.position.size

        realized_pnl = self.capital - self.initial_capital
        total_pnl = realized_pnl + unrealized_pnl

        return {
            "capital": self.capital,
            "position": {
                "size": self.position.size if self.position else 0,
                "entry_price": self.position.entry_price if self.position else 0,
                "unrealized_pnl": unrealized_pnl,
            } if self.position else None,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / self.initial_capital) * 100,
            "trades": len(self.trade_history),
            "decisions": len(self.decision_history),
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CATALYTIC MARKET BOT - Quick Test")
    print("=" * 70)

    bot = CatalyticMarketBot(initial_capital=100000)

    # Simulate a market scenario
    scenarios = [
        {
            "name": "Stable Bull Market",
            "headlines": [
                "Loyal shareholders see steady returns",
                "Trusted institutions raise guidance",
                "Leadership delivers consistent growth",
            ],
            "prices": [100 + i * 0.5 for i in range(20)],  # Uptrend
        },
        {
            "name": "Early Warning",
            "headlines": [
                "Unusual patterns in credit markets",
                "Questions raised about valuations",
                "Some analysts turn cautious",
            ],
            "prices": [110 - i * 0.3 + np.random.randn() * 0.5 for i in range(20)],
        },
        {
            "name": "Paradigm Shift",
            "headlines": [
                "Everything is changing rapidly",
                "Old assumptions no longer hold",
                "Complete transformation underway",
            ],
            "prices": [105 - i * 1.5 for i in range(20)],  # Sharp decline
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name'].upper()} ---")

        prices = scenario["prices"]
        current_price = prices[-1]

        decision = bot.process_tick(
            price=current_price,
            headlines=scenario["headlines"],
            prices_history=prices,
        )

        print(f"Regime: {decision.regime.value}")
        print(f"Action: {decision.action.value}")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Position Size: {decision.position_size:.0%}")
        print(f"Context Used: {decision.context_used[:3]}")
        print(f"Reasoning: {decision.reasoning[:100]}...")

        # Execute the decision
        trade = bot.execute_decision(decision, current_price)
        if trade:
            print(f"Trade: {trade['action']} @ ${trade['price']:.2f}")

    # Final status
    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    status = bot.get_status(scenarios[-1]["prices"][-1])
    print(f"Capital: ${status['capital']:.2f}")
    print(f"Position: {status['position']}")
    print(f"Total P&L: ${status['total_pnl']:.2f} ({status['total_return_pct']:.2f}%)")
    print(f"Trades: {status['trades']}")
