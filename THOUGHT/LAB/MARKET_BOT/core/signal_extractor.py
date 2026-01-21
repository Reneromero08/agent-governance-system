"""
SIGNAL EXTRACTOR
================

Converts raw price/volume data into our signal vocabulary.
This is the bridge between market data and the formula executor.

The bot sees SIGNALS, not prices. The formula operates on VECTORS.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent to path for cross-package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .signal_vocabulary import SignalState, AssetClass
from data.real_data_ingest import (
    MarketData, OHLCV,
    compute_sma, compute_rsi, compute_volatility, compute_volume_ratio
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds for signal detection
THRESHOLDS = {
    # Trend detection
    "trend_sma_fast": 5,
    "trend_sma_slow": 20,
    "trend_threshold": 0.005,      # 0.5% above/below SMA
    "breakout_threshold": 0.02,    # 2% above recent high/low

    # Volatility detection
    "vol_window": 20,
    "vol_spike_threshold": 2.0,    # 2x normal volatility
    "vol_crush_threshold": 0.5,    # 0.5x normal volatility

    # RSI thresholds
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_extreme_overbought": 80,
    "rsi_extreme_oversold": 20,

    # Volume thresholds
    "volume_surge_threshold": 2.0,   # 2x average volume
    "volume_drought_threshold": 0.5, # 0.5x average volume

    # Structure detection
    "lookback_high_low": 20,       # Days to look back for highs/lows
    "near_level_threshold": 0.02,  # Within 2% of high/low
}


# =============================================================================
# SIGNAL EXTRACTOR
# =============================================================================

class SignalExtractor:
    """
    Extracts signals from market data.

    Converts OHLCV bars to SignalState objects for the formula executor.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize signal extractor."""
        self.config = {**THRESHOLDS, **(config or {})}

    def extract(
        self,
        data: MarketData,
        index: int,
        asset_class: AssetClass = AssetClass.ALL
    ) -> SignalState:
        """
        Extract signals from market data at a specific index.

        The extractor ONLY sees data up to and including the index.
        It cannot see the future.

        Args:
            data: MarketData with OHLCV bars
            index: Current bar index (0-indexed)
            asset_class: Asset class for signal filtering

        Returns:
            SignalState with detected signals
        """
        if index < 0 or index >= len(data.bars):
            raise ValueError(f"Index {index} out of range [0, {len(data.bars)})")

        # Get historical data up to current point (NO FUTURE DATA)
        bars = data.bars[:index + 1]
        prices = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        # Current bar
        current = bars[-1]

        # Initialize signals dict
        signals: Dict[str, float] = {}

        # Extract each signal category
        self._extract_trend_signals(signals, prices, highs, lows, current)
        self._extract_volatility_signals(signals, prices)
        self._extract_momentum_signals(signals, prices)
        self._extract_volume_signals(signals, volumes)
        self._extract_structure_signals(signals, prices, highs, lows)

        return SignalState(
            signals=signals,
            timestamp=current.timestamp,
            asset=data.symbol,
            asset_class=asset_class,
        )

    def extract_sequence(
        self,
        data: MarketData,
        start_index: int = 20,  # Need some history for indicators
        asset_class: AssetClass = AssetClass.ALL
    ) -> List[SignalState]:
        """
        Extract signals for a sequence of bars.

        Args:
            data: MarketData with OHLCV bars
            start_index: Index to start from (need history)
            asset_class: Asset class

        Returns:
            List of SignalState objects
        """
        states = []
        for i in range(start_index, len(data.bars)):
            states.append(self.extract(data, i, asset_class))
        return states

    # =========================================================================
    # TREND SIGNALS
    # =========================================================================

    def _extract_trend_signals(
        self,
        signals: Dict[str, float],
        prices: List[float],
        highs: List[float],
        lows: List[float],
        current: OHLCV
    ):
        """Extract trend-related signals."""
        if len(prices) < self.config["trend_sma_slow"]:
            return

        # Compute SMAs
        sma_fast = np.mean(prices[-self.config["trend_sma_fast"]:])
        sma_slow = np.mean(prices[-self.config["trend_sma_slow"]:])

        current_price = prices[-1]

        # Trend up: price above both SMAs, fast above slow
        if current_price > sma_fast > sma_slow:
            strength = min((current_price / sma_slow - 1) / 0.05, 1.0)  # Normalize to 5%
            signals["trend_up"] = max(0.3, strength)

        # Trend down: price below both SMAs, fast below slow
        elif current_price < sma_fast < sma_slow:
            strength = min((sma_slow / current_price - 1) / 0.05, 1.0)
            signals["trend_down"] = max(0.3, strength)

        # Sideways: SMAs close together
        else:
            sma_diff = abs(sma_fast - sma_slow) / sma_slow
            if sma_diff < 0.01:
                signals["sideways"] = 0.7

        # Breakout/Breakdown
        lookback = min(self.config["lookback_high_low"], len(highs))
        recent_high = max(highs[-lookback:-1]) if len(highs) > 1 else highs[-1]
        recent_low = min(lows[-lookback:-1]) if len(lows) > 1 else lows[-1]

        if current.close > recent_high * (1 + self.config["breakout_threshold"]):
            signals["breakout"] = 0.8

        if current.close < recent_low * (1 - self.config["breakout_threshold"]):
            signals["breakdown"] = 0.8

    # =========================================================================
    # VOLATILITY SIGNALS
    # =========================================================================

    def _extract_volatility_signals(
        self,
        signals: Dict[str, float],
        prices: List[float]
    ):
        """Extract volatility-related signals."""
        vol_window = self.config["vol_window"]
        required_len = vol_window * 2 + 1  # Need 2 windows + 1 for returns

        if len(prices) < required_len:
            return

        # Compute current and historical volatility using positive indices
        n = len(prices)

        # Current volatility (recent window)
        recent_start = n - vol_window
        returns_recent = [(prices[i] - prices[i-1]) / prices[i-1]
                         for i in range(recent_start, n)]
        vol_current = np.std(returns_recent) if returns_recent else 0.01

        # Historical volatility (previous window)
        hist_start = n - vol_window * 2
        hist_end = n - vol_window
        returns_hist = [(prices[i] - prices[i-1]) / prices[i-1]
                       for i in range(hist_start, hist_end)]
        vol_hist = np.std(returns_hist) if returns_hist else 0.01

        vol_ratio = vol_current / max(vol_hist, 0.001)

        # Volatility expanding
        if vol_ratio > 1.3:
            signals["vol_expanding"] = min(vol_ratio / 2, 1.0)

        # Volatility contracting
        if vol_ratio < 0.7:
            signals["vol_contracting"] = min(1 / vol_ratio / 2, 1.0)

        # Volatility spike
        if vol_ratio > self.config["vol_spike_threshold"]:
            signals["vol_spike"] = min(vol_ratio / 3, 1.0)

        # Volatility crush
        if vol_ratio < self.config["vol_crush_threshold"]:
            signals["vol_crush"] = min(1 / vol_ratio / 3, 1.0)

    # =========================================================================
    # MOMENTUM SIGNALS
    # =========================================================================

    def _extract_momentum_signals(
        self,
        signals: Dict[str, float],
        prices: List[float]
    ):
        """Extract momentum-related signals."""
        if len(prices) < 15:
            return

        # Compute RSI
        rsi_values = compute_rsi(prices, window=14)
        rsi = rsi_values[-1] if rsi_values else 50

        # Overbought
        if rsi > self.config["rsi_overbought"]:
            strength = (rsi - self.config["rsi_overbought"]) / 30
            signals["overbought"] = min(max(0.3, strength), 1.0)

        # Oversold
        if rsi < self.config["rsi_oversold"]:
            strength = (self.config["rsi_oversold"] - rsi) / 30
            signals["oversold"] = min(max(0.3, strength), 1.0)

        # Momentum divergence (simple version: price vs RSI trend)
        if len(prices) >= 10 and len(rsi_values) >= 10:
            price_trend = prices[-1] / prices[-10] - 1
            rsi_trend = rsi_values[-1] - rsi_values[-10]

            # Bearish divergence: price up, RSI down
            if price_trend > 0.02 and rsi_trend < -10:
                signals["momentum_divergence"] = 0.6

            # Bullish divergence: price down, RSI up
            if price_trend < -0.02 and rsi_trend > 10:
                signals["momentum_divergence"] = 0.6

            # Momentum confirmation
            if (price_trend > 0.02 and rsi_trend > 5) or (price_trend < -0.02 and rsi_trend < -5):
                signals["momentum_confirmation"] = 0.5

    # =========================================================================
    # VOLUME SIGNALS
    # =========================================================================

    def _extract_volume_signals(
        self,
        signals: Dict[str, float],
        volumes: List[float]
    ):
        """Extract volume-related signals."""
        if len(volumes) < 21:
            return

        # Volume ratio
        vol_ratios = compute_volume_ratio(volumes, window=20)
        vol_ratio = vol_ratios[-1] if vol_ratios else 1.0

        # Volume surge
        if vol_ratio > self.config["volume_surge_threshold"]:
            signals["volume_surge"] = min(vol_ratio / 3, 1.0)

        # Volume drought
        if vol_ratio < self.config["volume_drought_threshold"]:
            signals["volume_drought"] = min(1 / vol_ratio / 3, 1.0)

        # Unusual activity (very high or sustained high volume)
        if vol_ratio > 3.0:
            signals["unusual_activity"] = 0.8

    # =========================================================================
    # STRUCTURE SIGNALS
    # =========================================================================

    def _extract_structure_signals(
        self,
        signals: Dict[str, float],
        prices: List[float],
        highs: List[float],
        lows: List[float]
    ):
        """Extract structure-related signals."""
        lookback = min(self.config["lookback_high_low"], len(prices) - 1)
        if lookback < 5:
            return

        current_price = prices[-1]
        recent_high = max(highs[-lookback:-1]) if len(highs) > 1 else highs[-1]
        recent_low = min(lows[-lookback:-1]) if len(lows) > 1 else lows[-1]

        threshold = self.config["near_level_threshold"]

        # Support test (near recent low)
        if current_price < recent_low * (1 + threshold):
            signals["support_test"] = 0.7

        # Resistance test (near recent high)
        if current_price > recent_high * (1 - threshold):
            signals["resistance_test"] = 0.7

        # New high
        all_time_high = max(highs)
        if current_price >= all_time_high:
            signals["new_high"] = 0.9

        # New low
        all_time_low = min(lows)
        if current_price <= all_time_low:
            signals["new_low"] = 0.9

        # Range bound
        price_range = (recent_high - recent_low) / recent_low
        if price_range < 0.05:  # Less than 5% range
            signals["range_bound"] = 0.6

    # =========================================================================
    # SENTIMENT SIGNALS (placeholder - would need news data)
    # =========================================================================

    def add_sentiment_signals(
        self,
        signals: Dict[str, float],
        sentiment_score: float  # -1 (bearish) to +1 (bullish)
    ):
        """
        Add sentiment signals from external source.

        For backtesting, we might not have historical sentiment,
        so this is optional.
        """
        if sentiment_score > 0.3:
            signals["bullish_news"] = min(sentiment_score, 1.0)
        elif sentiment_score < -0.3:
            signals["bearish_news"] = min(abs(sentiment_score), 1.0)
        else:
            signals["mixed_news"] = 0.5


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from data.real_data_ingest import RealDataFetcher

    print("=" * 60)
    print("SIGNAL EXTRACTOR - Demo")
    print("=" * 60)

    # Fetch some data
    fetcher = RealDataFetcher()
    data, event = fetcher.fetch_event("covid_crash_2020", symbol="SPY")

    print(f"\nData: {data.symbol} ({len(data.bars)} bars)")
    print(f"Event: {event['description']}")

    # Extract signals
    extractor = SignalExtractor()

    print("\n--- Signal extraction for key dates ---")

    # Find key dates
    crash_start = event["crash_start"]
    crash_end = event["crash_end"]

    for i, bar in enumerate(data.bars):
        if bar.timestamp in [crash_start, crash_end]:
            state = extractor.extract(data, i)
            print(f"\n{bar.timestamp} (${bar.close:.2f}):")
            for signal, strength in sorted(state.signals.items(), key=lambda x: -x[1]):
                print(f"  {signal}: {strength:.2f}")

    # Extract full sequence
    print("\n--- Full sequence extraction ---")
    states = extractor.extract_sequence(data, start_index=30)
    print(f"Extracted {len(states)} signal states")

    # Show signal distribution
    all_signals = {}
    for state in states:
        for signal, strength in state.signals.items():
            if signal not in all_signals:
                all_signals[signal] = []
            all_signals[signal].append(strength)

    print("\n--- Signal frequency ---")
    for signal, values in sorted(all_signals.items(), key=lambda x: -len(x[1])):
        print(f"  {signal}: {len(values)} occurrences (avg strength: {np.mean(values):.2f})")

    print("\n--- Signal Extractor Ready ---")
