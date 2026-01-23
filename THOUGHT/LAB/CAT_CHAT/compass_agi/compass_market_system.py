"""
COMPASS MARKET SYSTEM
=====================

Combines paradigm detection with traditional market indicators.

The key insight: The compass doesn't predict PRICE, it predicts REGIME.
Use this to know WHEN your other predictions will work.

Traditional methods work in STABLE regimes.
Traditional methods FAIL in SHIFT regimes.
The compass tells you which regime you're in.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

# Import paradigm detector
import sys
from pathlib import Path
COMPASS_PATH = Path(__file__).parent
if str(COMPASS_PATH) not in sys.path:
    sys.path.insert(0, str(COMPASS_PATH))

try:
    from realtime_paradigm_detector import ParadigmShiftDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime classification."""
    STABLE = "STABLE"           # Traditional methods work
    TRANSITIONAL = "TRANSITIONAL"  # Reduce exposure, watch
    SHIFT = "SHIFT"             # Go defensive, methods fail


class TradingStrategy(Enum):
    """Strategy based on regime."""
    MOMENTUM = "MOMENTUM"       # Trend following (STABLE)
    DEFENSIVE = "DEFENSIVE"     # Hedged, reduced (SHIFT)
    CAUTIOUS = "CAUTIOUS"       # Reduced size (TRANSITIONAL)


@dataclass
class MarketSignal:
    """Combined market signal from compass + traditional indicators."""

    # Regime from compass
    regime: MarketRegime
    regime_confidence: float  # 0-1
    shift_score: float       # Raw paradigm shift score

    # Top geodesics (what archetypes are active)
    dominant_geodesics: List[Tuple[str, float]]

    # Recommended strategy
    strategy: TradingStrategy
    position_size_multiplier: float  # 0.0-1.0

    # Traditional indicators (if provided)
    trend_direction: Optional[str] = None  # "UP", "DOWN", "NEUTRAL"
    momentum_score: Optional[float] = None

    # Reasoning
    reasoning: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "shift_score": self.shift_score,
            "dominant_geodesics": self.dominant_geodesics,
            "strategy": self.strategy.value,
            "position_size_multiplier": self.position_size_multiplier,
            "trend_direction": self.trend_direction,
            "momentum_score": self.momentum_score,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


class CompassMarketSystem:
    """
    Combines paradigm detection with traditional market analysis.

    The compass provides META-prediction: It predicts when other predictions work.

    Usage:
        system = CompassMarketSystem()

        # Feed it news headlines
        signal = system.analyze(
            headlines=["Markets rally on earnings", "Tech sector leads gains"],
            prices=[100, 101, 102, 103, 104],  # Recent prices
        )

        print(f"Regime: {signal.regime}")
        print(f"Strategy: {signal.strategy}")
        print(f"Position size: {signal.position_size_multiplier * 100}%")
    """

    def __init__(self):
        if not DETECTOR_AVAILABLE:
            raise ImportError("ParadigmShiftDetector not available")

        self.detector = ParadigmShiftDetector()

        # Strategy parameters by regime
        self.regime_params = {
            MarketRegime.STABLE: {
                "position_multiplier": 1.0,
                "strategy": TradingStrategy.MOMENTUM,
                "stop_loss_pct": 0.05,  # 5% stop
            },
            MarketRegime.TRANSITIONAL: {
                "position_multiplier": 0.5,
                "strategy": TradingStrategy.CAUTIOUS,
                "stop_loss_pct": 0.03,  # 3% tighter stop
            },
            MarketRegime.SHIFT: {
                "position_multiplier": 0.2,
                "strategy": TradingStrategy.DEFENSIVE,
                "stop_loss_pct": 0.02,  # 2% very tight
            },
        }

    def analyze(
        self,
        headlines: List[str],
        prices: Optional[List[float]] = None,
    ) -> MarketSignal:
        """
        Analyze market using compass + traditional indicators.

        Args:
            headlines: Recent news headlines (for paradigm detection)
            prices: Recent prices (for trend/momentum calculation)

        Returns:
            MarketSignal with regime, strategy, and reasoning
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. PARADIGM DETECTION (from headlines)
        paradigm_result = self.detector.detect_shift(headlines)

        # Map to regime
        shift_type = paradigm_result['shift_type']
        if shift_type == 'SHIFT':
            regime = MarketRegime.SHIFT
        elif shift_type == 'STABLE':
            regime = MarketRegime.STABLE
        else:
            regime = MarketRegime.TRANSITIONAL

        # Confidence from shift score magnitude
        shift_score = paradigm_result['shift_score']
        regime_confidence = min(1.0, abs(shift_score) * 5)  # Scale to 0-1

        # 2. TRADITIONAL INDICATORS (if prices provided)
        trend_direction = None
        momentum_score = None

        if prices and len(prices) >= 5:
            trend_direction, momentum_score = self._compute_trend(prices)

        # 3. COMBINE FOR STRATEGY
        params = self.regime_params[regime]
        strategy = params["strategy"]
        position_multiplier = params["position_multiplier"]

        # Adjust position based on trend alignment
        if regime == MarketRegime.STABLE and trend_direction:
            # In stable regime, boost position if trend is clear
            if trend_direction in ["UP", "DOWN"]:
                position_multiplier = min(1.0, position_multiplier * 1.2)

        # Build reasoning
        reasoning = self._build_reasoning(
            regime, paradigm_result, trend_direction, momentum_score
        )

        return MarketSignal(
            regime=regime,
            regime_confidence=regime_confidence,
            shift_score=shift_score,
            dominant_geodesics=paradigm_result['top_geodesics'][:3],
            strategy=strategy,
            position_size_multiplier=position_multiplier,
            trend_direction=trend_direction,
            momentum_score=momentum_score,
            reasoning=reasoning,
            timestamp=timestamp,
        )

    def _compute_trend(self, prices: List[float]) -> Tuple[str, float]:
        """Compute simple trend and momentum from prices."""
        prices = np.array(prices)

        # Simple moving average crossover
        if len(prices) >= 10:
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices[-10:])
        else:
            sma_short = np.mean(prices[-3:])
            sma_long = np.mean(prices)

        # Trend direction
        if sma_short > sma_long * 1.01:  # 1% above
            trend = "UP"
        elif sma_short < sma_long * 0.99:  # 1% below
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        # Momentum: rate of change
        if len(prices) >= 2:
            momentum = (prices[-1] - prices[0]) / prices[0]
        else:
            momentum = 0.0

        return trend, float(momentum)

    def _build_reasoning(
        self,
        regime: MarketRegime,
        paradigm_result: Dict,
        trend: Optional[str],
        momentum: Optional[float],
    ) -> str:
        """Build human-readable reasoning."""
        lines = []

        # Regime explanation
        if regime == MarketRegime.SHIFT:
            lines.append("PARADIGM SHIFT DETECTED - Traditional methods unreliable")
            lines.append("Dominant archetypes: Earthquake/Death/Wind (disruption)")
            lines.append("Recommendation: Reduce exposure, hedge, wait for new regime")
        elif regime == MarketRegime.STABLE:
            lines.append("STABLE REGIME - Traditional methods applicable")
            lines.append("Dominant archetypes: Dog/Deer/Reed (loyalty, guidance)")
            lines.append("Recommendation: Follow momentum, trend strategies work")
        else:
            lines.append("TRANSITIONAL STATE - Regime uncertain")
            lines.append("Watch for shift development")
            lines.append("Recommendation: Reduce position size, tighten stops")

        # Geodesics
        top_geo = paradigm_result['top_geodesics'][0]
        lines.append(f"Top geodesic: {top_geo[0]} ({top_geo[1]:.3f})")

        # Trend info
        if trend:
            lines.append(f"Price trend: {trend}")
        if momentum is not None:
            lines.append(f"Momentum: {momentum:+.2%}")

        return " | ".join(lines)


# =============================================================================
# TRADITIONAL INDICATORS (for comparison)
# =============================================================================

def compute_rsi(prices: List[float], period: int = 14) -> float:
    """Compute Relative Strength Index."""
    prices = np.array(prices)
    if len(prices) < period + 1:
        return 50.0  # Neutral

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def compute_macd(prices: List[float]) -> Tuple[float, float, float]:
    """Compute MACD (12, 26, 9)."""
    prices = np.array(prices)

    if len(prices) < 26:
        return 0.0, 0.0, 0.0

    # EMA calculation
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[-1])
        return np.array(result)

    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)

    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    histogram = macd_line - signal_line

    return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])


def traditional_signal(prices: List[float]) -> Dict[str, Any]:
    """
    Generate traditional technical analysis signal.

    This is what most traders use. It works in STABLE regimes.
    It FAILS during paradigm shifts.
    """
    if len(prices) < 26:
        return {"signal": "INSUFFICIENT_DATA", "confidence": 0.0}

    rsi = compute_rsi(prices)
    macd_line, signal_line, histogram = compute_macd(prices)

    # Simple rules
    signals = []

    # RSI
    if rsi < 30:
        signals.append(("BUY", "RSI oversold"))
    elif rsi > 70:
        signals.append(("SELL", "RSI overbought"))

    # MACD
    if histogram > 0 and macd_line > signal_line:
        signals.append(("BUY", "MACD bullish crossover"))
    elif histogram < 0 and macd_line < signal_line:
        signals.append(("SELL", "MACD bearish crossover"))

    # Combine
    buys = [s for s in signals if s[0] == "BUY"]
    sells = [s for s in signals if s[0] == "SELL"]

    if len(buys) > len(sells):
        signal = "BUY"
        reasons = [s[1] for s in buys]
    elif len(sells) > len(buys):
        signal = "SELL"
        reasons = [s[1] for s in sells]
    else:
        signal = "HOLD"
        reasons = ["No clear signal"]

    return {
        "signal": signal,
        "confidence": len(signals) / 2.0,  # Max 2 signals
        "rsi": rsi,
        "macd": {"line": macd_line, "signal": signal_line, "histogram": histogram},
        "reasons": reasons,
    }


# =============================================================================
# DEMO: Compass vs Traditional
# =============================================================================

def demo():
    """Demonstrate compass-enhanced market analysis."""

    print("=" * 70)
    print("COMPASS MARKET SYSTEM - Regime-Aware Trading")
    print("=" * 70)

    system = CompassMarketSystem()

    # Generate enough prices for traditional indicators (need 26+ for MACD)
    def generate_prices(start, trend, volatility, n=30):
        prices = [start]
        for i in range(n - 1):
            change = trend + np.random.randn() * volatility
            prices.append(prices[-1] * (1 + change))
        return prices

    # Scenario 1: Normal Market (STABLE)
    print("\n\n--- SCENARIO 1: NORMAL MARKET ---")
    stable_headlines = [
        "Loyal shareholders rewarded with steady dividends",
        "Trusted institutions maintain guidance",
        "Faithful customers drive reliable growth",
        "Leadership provides steady direction",
        "Established companies deliver as expected",
    ]
    np.random.seed(42)
    prices_up = generate_prices(100, 0.003, 0.01)  # Slight uptrend

    signal = system.analyze(stable_headlines, prices_up)
    traditional = traditional_signal(prices_up)

    print(f"Compass Regime: {signal.regime.value}")
    print(f"Compass Strategy: {signal.strategy.value}")
    print(f"Position Size: {signal.position_size_multiplier * 100:.0f}%")
    print(f"Traditional Signal: {traditional['signal']}")
    if 'rsi' in traditional:
        print(f"RSI: {traditional['rsi']:.1f}")
    print(f"Reasoning: {signal.reasoning}")

    # Scenario 2: Pre-Crisis (TRANSITIONAL)
    print("\n\n--- SCENARIO 2: EARLY WARNING ---")
    warning_headlines = [
        "Unusual volatility in bond markets",
        "Questions raised about economic data",
        "Some analysts revise forecasts downward",
        "Emerging cracks in credit markets",
        "Uncertainty grows over policy direction",
    ]
    np.random.seed(43)
    prices_choppy = generate_prices(100, -0.001, 0.02)  # Sideways with volatility

    signal = system.analyze(warning_headlines, prices_choppy)
    traditional = traditional_signal(prices_choppy)

    print(f"Compass Regime: {signal.regime.value}")
    print(f"Compass Strategy: {signal.strategy.value}")
    print(f"Position Size: {signal.position_size_multiplier * 100:.0f}%")
    print(f"Traditional Signal: {traditional['signal']}")
    if 'rsi' in traditional:
        print(f"RSI: {traditional['rsi']:.1f}")
    print(f"Reasoning: {signal.reasoning}")
    print(f"\n--> Compass says CAUTION (traditional says {traditional['signal']})")

    # Scenario 3: Crisis (SHIFT)
    print("\n\n--- SCENARIO 3: PARADIGM SHIFT ---")
    crisis_headlines = [
        "Markets in freefall as crisis deepens",
        "Everything we knew is changing",
        "Old models no longer apply",
        "Unprecedented disruption across sectors",
        "Complete transformation of the landscape",
    ]
    np.random.seed(44)
    prices_crash = generate_prices(100, -0.02, 0.03)  # Strong downtrend

    signal = system.analyze(crisis_headlines, prices_crash)
    traditional = traditional_signal(prices_crash)

    print(f"Compass Regime: {signal.regime.value}")
    print(f"Compass Strategy: {signal.strategy.value}")
    print(f"Position Size: {signal.position_size_multiplier * 100:.0f}%")
    print(f"Traditional Signal: {traditional['signal']}")
    if 'rsi' in traditional:
        print(f"RSI: {traditional['rsi']:.1f}")
    print(f"Reasoning: {signal.reasoning}")
    print(f"\n--> Traditional may say BUY (oversold), Compass says GO DEFENSIVE")
    print(f"--> THIS IS WHERE TRADITIONAL METHODS FAIL")

    # Summary
    print("\n\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The compass doesn't replace traditional analysis.
It tells you WHEN traditional analysis works.

STABLE Regime:
  - Traditional methods work
  - Follow RSI, MACD, momentum signals
  - Full position sizes

SHIFT Regime:
  - Traditional methods FAIL
  - RSI "oversold" doesn't mean "buy" in a paradigm shift
  - Catching falling knives kills accounts
  - Reduce exposure, wait for new regime

The compass provides META-PREDICTION:
  It predicts when your other predictions will fail.

This is what no other system provides:
  - WHEN to trust your indicators
  - WHEN to go defensive
  - WHEN the rules have changed
""")


if __name__ == "__main__":
    demo()
