"""
SIGNAL VOCABULARY
=================

Asset-agnostic market signal definitions for the Psychohistory bot.
Each signal has a description that gets embedded into semantic space.

The formula doesn't care what asset - it measures COHERENCE of signals.
A "trend_up + volume_surge + bullish_news" combination has high E (agreement)
regardless of whether it's AAPL, BTC, or SPY options.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class AssetClass(Enum):
    """Asset classes for signal applicability."""
    ALL = "all"
    STOCKS = "stocks"
    CRYPTO = "crypto"
    OPTIONS = "options"


class SignalCategory(Enum):
    """Signal categories for organization."""
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    STRUCTURE = "structure"
    SENTIMENT = "sentiment"
    GREEKS = "greeks"          # Options only
    ONCHAIN = "onchain"        # Crypto only


@dataclass
class SignalDefinition:
    """Definition of a market signal for embedding."""
    signal_id: str
    category: SignalCategory
    applies_to: AssetClass
    description: str
    bullish_bias: float = 0.0  # -1 to +1, 0 = neutral


# =============================================================================
# UNIVERSAL SIGNALS (Apply to ALL asset classes)
# =============================================================================

TREND_SIGNALS = [
    SignalDefinition(
        signal_id="trend_up",
        category=SignalCategory.TREND,
        applies_to=AssetClass.ALL,
        description="Price making higher highs and higher lows, sustained upward movement, bullish trend confirmed",
        bullish_bias=1.0,
    ),
    SignalDefinition(
        signal_id="trend_down",
        category=SignalCategory.TREND,
        applies_to=AssetClass.ALL,
        description="Price making lower highs and lower lows, sustained downward movement, bearish trend confirmed",
        bullish_bias=-1.0,
    ),
    SignalDefinition(
        signal_id="sideways",
        category=SignalCategory.TREND,
        applies_to=AssetClass.ALL,
        description="Price moving horizontally without clear direction, consolidation phase, range-bound trading",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="breakout",
        category=SignalCategory.TREND,
        applies_to=AssetClass.ALL,
        description="Price breaking above resistance level with increased volume, potential new uptrend beginning",
        bullish_bias=0.8,
    ),
    SignalDefinition(
        signal_id="breakdown",
        category=SignalCategory.TREND,
        applies_to=AssetClass.ALL,
        description="Price breaking below support level with increased volume, potential new downtrend beginning",
        bullish_bias=-0.8,
    ),
]

VOLATILITY_SIGNALS = [
    SignalDefinition(
        signal_id="vol_expanding",
        category=SignalCategory.VOLATILITY,
        applies_to=AssetClass.ALL,
        description="Volatility increasing, larger price swings, market uncertainty growing, potential regime change",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="vol_contracting",
        category=SignalCategory.VOLATILITY,
        applies_to=AssetClass.ALL,
        description="Volatility decreasing, smaller price movements, market calming, consolidation before move",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="vol_spike",
        category=SignalCategory.VOLATILITY,
        applies_to=AssetClass.ALL,
        description="Sudden extreme volatility increase, panic or euphoria, major news event, potential reversal",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="vol_crush",
        category=SignalCategory.VOLATILITY,
        applies_to=AssetClass.ALL,
        description="Rapid volatility decrease, fear subsiding, stability returning, trend continuation likely",
        bullish_bias=0.2,
    ),
]

VOLUME_SIGNALS = [
    SignalDefinition(
        signal_id="volume_surge",
        category=SignalCategory.VOLUME,
        applies_to=AssetClass.ALL,
        description="Trading volume significantly above average, high participation, strong conviction in move",
        bullish_bias=0.0,  # Depends on price direction
    ),
    SignalDefinition(
        signal_id="volume_drought",
        category=SignalCategory.VOLUME,
        applies_to=AssetClass.ALL,
        description="Trading volume significantly below average, low participation, weak conviction, potential reversal",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="unusual_activity",
        category=SignalCategory.VOLUME,
        applies_to=AssetClass.ALL,
        description="Abnormal trading patterns, potential insider activity, large block trades, institutional interest",
        bullish_bias=0.0,
    ),
]

MOMENTUM_SIGNALS = [
    SignalDefinition(
        signal_id="overbought",
        category=SignalCategory.MOMENTUM,
        applies_to=AssetClass.ALL,
        description="RSI above 70, price extended too far too fast, potential pullback imminent, exhaustion signs",
        bullish_bias=-0.5,
    ),
    SignalDefinition(
        signal_id="oversold",
        category=SignalCategory.MOMENTUM,
        applies_to=AssetClass.ALL,
        description="RSI below 30, price fallen too far too fast, potential bounce imminent, capitulation signs",
        bullish_bias=0.5,
    ),
    SignalDefinition(
        signal_id="momentum_divergence",
        category=SignalCategory.MOMENTUM,
        applies_to=AssetClass.ALL,
        description="Price and momentum indicator moving in opposite directions, potential trend reversal warning",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="momentum_confirmation",
        category=SignalCategory.MOMENTUM,
        applies_to=AssetClass.ALL,
        description="Price and momentum indicator moving in same direction, trend strength confirmed",
        bullish_bias=0.0,
    ),
]

STRUCTURE_SIGNALS = [
    SignalDefinition(
        signal_id="support_test",
        category=SignalCategory.STRUCTURE,
        applies_to=AssetClass.ALL,
        description="Price testing key support level, buyers defending, potential bounce point",
        bullish_bias=0.3,
    ),
    SignalDefinition(
        signal_id="resistance_test",
        category=SignalCategory.STRUCTURE,
        applies_to=AssetClass.ALL,
        description="Price testing key resistance level, sellers defending, potential rejection point",
        bullish_bias=-0.3,
    ),
    SignalDefinition(
        signal_id="range_bound",
        category=SignalCategory.STRUCTURE,
        applies_to=AssetClass.ALL,
        description="Price oscillating between support and resistance, no breakout yet, wait for direction",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="new_high",
        category=SignalCategory.STRUCTURE,
        applies_to=AssetClass.ALL,
        description="Price making new highs, no overhead resistance, momentum strong, buyers in control",
        bullish_bias=0.9,
    ),
    SignalDefinition(
        signal_id="new_low",
        category=SignalCategory.STRUCTURE,
        applies_to=AssetClass.ALL,
        description="Price making new lows, no support below, momentum weak, sellers in control",
        bullish_bias=-0.9,
    ),
]

SENTIMENT_SIGNALS = [
    SignalDefinition(
        signal_id="bullish_news",
        category=SignalCategory.SENTIMENT,
        applies_to=AssetClass.ALL,
        description="Positive news headlines, optimistic analyst reports, favorable economic data, upgrade calls",
        bullish_bias=0.7,
    ),
    SignalDefinition(
        signal_id="bearish_news",
        category=SignalCategory.SENTIMENT,
        applies_to=AssetClass.ALL,
        description="Negative news headlines, pessimistic analyst reports, unfavorable economic data, downgrade calls",
        bullish_bias=-0.7,
    ),
    SignalDefinition(
        signal_id="mixed_news",
        category=SignalCategory.SENTIMENT,
        applies_to=AssetClass.ALL,
        description="Conflicting news signals, uncertainty in headlines, market unsure of direction",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="no_news",
        category=SignalCategory.SENTIMENT,
        applies_to=AssetClass.ALL,
        description="No significant news, quiet market, technical factors dominate, low catalyst environment",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="extreme_fear",
        category=SignalCategory.SENTIMENT,
        applies_to=AssetClass.ALL,
        description="Fear and greed index at extreme fear, panic selling, potential capitulation bottom",
        bullish_bias=0.4,  # Contrarian
    ),
    SignalDefinition(
        signal_id="extreme_greed",
        category=SignalCategory.SENTIMENT,
        applies_to=AssetClass.ALL,
        description="Fear and greed index at extreme greed, euphoric buying, potential blow-off top",
        bullish_bias=-0.4,  # Contrarian
    ),
]

# =============================================================================
# OPTIONS-SPECIFIC SIGNALS
# =============================================================================

GREEKS_SIGNALS = [
    SignalDefinition(
        signal_id="delta_heavy_calls",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="High delta call buying, directional bullish bets, expecting significant upside move",
        bullish_bias=0.8,
    ),
    SignalDefinition(
        signal_id="delta_heavy_puts",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="High delta put buying, directional bearish bets, expecting significant downside move",
        bullish_bias=-0.8,
    ),
    SignalDefinition(
        signal_id="gamma_scalp",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="High gamma positions near expiry, market makers hedging, potential for rapid price moves",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="theta_decay",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="Time decay accelerating, premium sellers winning, range-bound expected",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="vega_play",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="Volatility bets increasing, expecting IV expansion, big move anticipated",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="put_call_ratio_high",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="Put to call ratio elevated, bearish positioning, potential contrarian bullish signal",
        bullish_bias=0.3,  # Contrarian
    ),
    SignalDefinition(
        signal_id="put_call_ratio_low",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="Put to call ratio depressed, bullish positioning, potential contrarian bearish signal",
        bullish_bias=-0.3,  # Contrarian
    ),
    SignalDefinition(
        signal_id="iv_elevated",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="Implied volatility above historical average, options expensive, expecting big move or IV crush",
        bullish_bias=0.0,
    ),
    SignalDefinition(
        signal_id="iv_depressed",
        category=SignalCategory.GREEKS,
        applies_to=AssetClass.OPTIONS,
        description="Implied volatility below historical average, options cheap, potential IV expansion ahead",
        bullish_bias=0.0,
    ),
]

# =============================================================================
# CRYPTO-SPECIFIC SIGNALS
# =============================================================================

ONCHAIN_SIGNALS = [
    SignalDefinition(
        signal_id="funding_positive",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Perpetual futures funding rate positive, longs paying shorts, bullish sentiment but crowded",
        bullish_bias=-0.2,  # Contrarian
    ),
    SignalDefinition(
        signal_id="funding_negative",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Perpetual futures funding rate negative, shorts paying longs, bearish sentiment but crowded",
        bullish_bias=0.2,  # Contrarian
    ),
    SignalDefinition(
        signal_id="exchange_inflow",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Large deposits to exchanges, potential selling pressure, holders moving to sell",
        bullish_bias=-0.5,
    ),
    SignalDefinition(
        signal_id="exchange_outflow",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Large withdrawals from exchanges, potential accumulation, holders moving to cold storage",
        bullish_bias=0.5,
    ),
    SignalDefinition(
        signal_id="whale_accumulation",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Large wallet addresses increasing holdings, smart money buying, bullish long-term signal",
        bullish_bias=0.6,
    ),
    SignalDefinition(
        signal_id="whale_distribution",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Large wallet addresses decreasing holdings, smart money selling, bearish long-term signal",
        bullish_bias=-0.6,
    ),
    SignalDefinition(
        signal_id="stablecoin_supply_increase",
        category=SignalCategory.ONCHAIN,
        applies_to=AssetClass.CRYPTO,
        description="Stablecoin supply growing, dry powder available, potential buying pressure building",
        bullish_bias=0.4,
    ),
]


# =============================================================================
# VOCABULARY REGISTRY
# =============================================================================

def get_all_signals() -> List[SignalDefinition]:
    """Get all signal definitions."""
    return (
        TREND_SIGNALS +
        VOLATILITY_SIGNALS +
        VOLUME_SIGNALS +
        MOMENTUM_SIGNALS +
        STRUCTURE_SIGNALS +
        SENTIMENT_SIGNALS +
        GREEKS_SIGNALS +
        ONCHAIN_SIGNALS
    )


def get_signals_by_category(category: SignalCategory) -> List[SignalDefinition]:
    """Get signals filtered by category."""
    return [s for s in get_all_signals() if s.category == category]


def get_signals_by_asset(asset_class: AssetClass) -> List[SignalDefinition]:
    """Get signals applicable to an asset class."""
    all_signals = get_all_signals()
    if asset_class == AssetClass.ALL:
        return all_signals
    return [s for s in all_signals if s.applies_to in (AssetClass.ALL, asset_class)]


def get_signal_by_id(signal_id: str) -> Optional[SignalDefinition]:
    """Get a specific signal by ID."""
    for signal in get_all_signals():
        if signal.signal_id == signal_id:
            return signal
    return None


def get_signal_descriptions() -> Dict[str, str]:
    """Get mapping of signal_id to description for embedding."""
    return {s.signal_id: s.description for s in get_all_signals()}


# =============================================================================
# SIGNAL STATE (Runtime representation)
# =============================================================================

@dataclass
class SignalState:
    """
    Runtime state of market signals.

    Each signal has a strength from 0.0 (not present) to 1.0 (strongly present).
    """
    signals: Dict[str, float]  # {signal_id: strength 0-1}
    timestamp: str
    asset: str
    asset_class: AssetClass

    def to_vector_weights(self) -> Dict[str, float]:
        """Convert to weights for vector computation."""
        return {k: v for k, v in self.signals.items() if v > 0}

    def dominant_signals(self, threshold: float = 0.5) -> List[str]:
        """Get signals above threshold."""
        return [k for k, v in self.signals.items() if v >= threshold]

    def net_bias(self) -> float:
        """Compute net bullish/bearish bias from active signals."""
        total_bias = 0.0
        total_weight = 0.0

        for signal_id, strength in self.signals.items():
            if strength > 0:
                signal_def = get_signal_by_id(signal_id)
                if signal_def:
                    total_bias += signal_def.bullish_bias * strength
                    total_weight += strength

        return total_bias / total_weight if total_weight > 0 else 0.0


# =============================================================================
# DEMO / SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIGNAL VOCABULARY - Psychohistory Market Bot")
    print("=" * 60)

    all_signals = get_all_signals()
    print(f"\nTotal signals defined: {len(all_signals)}")

    print("\n--- By Category ---")
    for cat in SignalCategory:
        signals = get_signals_by_category(cat)
        print(f"  {cat.value}: {len(signals)} signals")

    print("\n--- By Asset Class ---")
    for asset in AssetClass:
        signals = get_signals_by_asset(asset)
        print(f"  {asset.value}: {len(signals)} signals")

    print("\n--- Sample Signals ---")
    for signal in all_signals[:5]:
        print(f"  {signal.signal_id}: {signal.description[:60]}...")

    print("\n--- Signal Descriptions (for embedding) ---")
    descs = get_signal_descriptions()
    print(f"  Ready to embed {len(descs)} signal descriptions")
