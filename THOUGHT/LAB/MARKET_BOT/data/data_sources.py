"""
DATA SOURCES
============

Mock and real data sources for the market bot.
Provides price data and news headlines.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class MarketTick:
    """A single market data point."""
    timestamp: datetime
    price: float
    volume: float
    headlines: List[str]


class MockMarketScenario:
    """
    Generates mock market data for different scenarios.

    Scenarios:
    - stable_bull: Steady uptrend with positive news
    - stable_bear: Steady downtrend with negative news
    - volatile: Choppy market with mixed signals
    - crisis: Sharp decline with paradigm shift language
    - recovery: Post-crisis recovery
    """

    SCENARIOS = {
        "stable_bull": {
            "trend": 0.002,      # 0.2% per tick average
            "volatility": 0.01,  # 1% random noise
            "headline_type": "stable_positive",
        },
        "stable_bear": {
            "trend": -0.001,
            "volatility": 0.01,
            "headline_type": "stable_negative",
        },
        "volatile": {
            "trend": 0.0,
            "volatility": 0.025,
            "headline_type": "mixed",
        },
        "early_warning": {
            "trend": -0.001,
            "volatility": 0.02,
            "headline_type": "warning",
        },
        "crisis": {
            "trend": -0.015,
            "volatility": 0.04,
            "headline_type": "crisis",
        },
        "recovery": {
            "trend": 0.005,
            "volatility": 0.02,
            "headline_type": "recovery",
        },
    }

    HEADLINES = {
        "stable_positive": [
            "Loyal shareholders rewarded with steady dividends",
            "Trusted institutions maintain guidance",
            "Faithful customers drive reliable growth",
            "Leadership provides steady direction",
            "Established companies deliver as expected",
            "Steady progress on quarterly targets",
            "Reliable earnings beat expectations",
            "Consistent growth in key metrics",
        ],
        "stable_negative": [
            "Markets drift lower on light volume",
            "Investors wait for clearer direction",
            "Steady decline in sector interest",
            "Gradual rotation out of growth stocks",
            "Patient selling continues",
        ],
        "mixed": [
            "Markets mixed as traders weigh data",
            "Conflicting signals keep investors cautious",
            "Some sectors rise while others fall",
            "Uncertainty keeps volume low",
            "Analysts divided on outlook",
        ],
        "warning": [
            "Unusual volatility in bond markets",
            "Questions raised about economic data",
            "Some analysts revise forecasts downward",
            "Emerging cracks in credit markets",
            "Uncertainty grows over policy direction",
            "Warning signs appear in leading indicators",
            "Unusual patterns detected in trading",
        ],
        "crisis": [
            "Markets in freefall as crisis deepens",
            "Everything we knew is changing",
            "Old models no longer apply",
            "Unprecedented disruption across sectors",
            "Complete transformation of the landscape",
            "Paradigm shift underway in markets",
            "Death of traditional strategies",
            "Revolutionary change sweeping through",
        ],
        "recovery": [
            "New normal beginning to establish",
            "Markets find footing after turmoil",
            "Investors cautiously return",
            "Signs of stabilization emerge",
            "Adapted strategies showing results",
            "Trust slowly rebuilding",
        ],
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        scenario: str,
        n_ticks: int = 100,
        start_price: float = 100.0,
        start_time: Optional[datetime] = None,
    ) -> List[MarketTick]:
        """Generate mock market data for a scenario."""
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")

        config = self.SCENARIOS[scenario]
        headline_pool = self.HEADLINES[config["headline_type"]]

        if start_time is None:
            start_time = datetime.now()

        ticks = []
        price = start_price

        for i in range(n_ticks):
            # Price evolution
            change = config["trend"] + self.rng.randn() * config["volatility"]
            price = price * (1 + change)
            price = max(price, 1.0)  # Floor at $1

            # Volume (random)
            volume = self.rng.exponential(1000000)

            # Headlines (sample 3-5)
            n_headlines = self.rng.randint(3, 6)
            headlines = list(self.rng.choice(headline_pool, size=min(n_headlines, len(headline_pool)), replace=False))

            # Timestamp
            timestamp = start_time + timedelta(minutes=i * 5)

            ticks.append(MarketTick(
                timestamp=timestamp,
                price=price,
                volume=volume,
                headlines=headlines,
            ))

        return ticks

    def generate_multi_phase(
        self,
        phases: List[Tuple[str, int]],
        start_price: float = 100.0,
    ) -> List[MarketTick]:
        """
        Generate multi-phase scenario.

        Args:
            phases: List of (scenario_name, n_ticks) tuples

        Example:
            generate_multi_phase([
                ("stable_bull", 50),
                ("early_warning", 20),
                ("crisis", 30),
                ("recovery", 40),
            ])
        """
        all_ticks = []
        price = start_price
        time = datetime.now()

        for scenario, n_ticks in phases:
            ticks = self.generate(
                scenario=scenario,
                n_ticks=n_ticks,
                start_price=price,
                start_time=time,
            )
            all_ticks.extend(ticks)

            # Continue from last price/time
            if ticks:
                price = ticks[-1].price
                time = ticks[-1].timestamp + timedelta(minutes=5)

        return all_ticks


class HistoricalDataLoader:
    """
    Loads historical market data from files.

    Expected format: JSON with list of {timestamp, price, volume, headlines}
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"

    def load(self, filename: str) -> List[MarketTick]:
        """Load historical data from file."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        ticks = []
        for item in data:
            ticks.append(MarketTick(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                price=item["price"],
                volume=item.get("volume", 0),
                headlines=item.get("headlines", []),
            ))

        return ticks

    def list_available(self) -> List[str]:
        """List available data files."""
        if not self.data_dir.exists():
            return []
        return [f.name for f in self.data_dir.glob("*.json")]


class RealtimeDataSimulator:
    """
    Simulates real-time data feed from historical or mock data.

    Yields ticks one at a time with optional delays.
    """

    def __init__(self, ticks: List[MarketTick]):
        self.ticks = ticks
        self.index = 0

    def __iter__(self) -> Iterator[MarketTick]:
        return self

    def __next__(self) -> MarketTick:
        if self.index >= len(self.ticks):
            raise StopIteration

        tick = self.ticks[self.index]
        self.index += 1
        return tick

    def reset(self):
        """Reset to beginning."""
        self.index = 0

    def get_history(self, lookback: int = 20) -> List[float]:
        """Get price history up to current point."""
        start = max(0, self.index - lookback)
        return [self.ticks[i].price for i in range(start, self.index)]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA SOURCES - Demo")
    print("=" * 60)

    generator = MockMarketScenario(seed=42)

    # Single scenario
    print("\n--- SINGLE SCENARIO: stable_bull ---")
    ticks = generator.generate("stable_bull", n_ticks=10, start_price=100)
    for tick in ticks[:5]:
        print(f"{tick.timestamp.strftime('%H:%M')} | ${tick.price:.2f} | {tick.headlines[0][:40]}...")

    # Multi-phase scenario
    print("\n--- MULTI-PHASE: Bull -> Warning -> Crisis -> Recovery ---")
    ticks = generator.generate_multi_phase([
        ("stable_bull", 20),
        ("early_warning", 10),
        ("crisis", 15),
        ("recovery", 15),
    ])

    # Show transitions
    checkpoints = [0, 19, 29, 44, 59]
    for i in checkpoints:
        if i < len(ticks):
            tick = ticks[i]
            print(f"Tick {i:3d} | ${tick.price:6.2f} | {tick.headlines[0][:50]}...")

    # Price trajectory
    print("\n--- PRICE TRAJECTORY ---")
    prices = [t.price for t in ticks]
    print(f"Start: ${prices[0]:.2f}")
    print(f"Pre-warning: ${prices[19]:.2f}")
    print(f"Pre-crisis: ${prices[29]:.2f}")
    print(f"Crisis low: ${min(prices[30:45]):.2f}")
    print(f"Recovery: ${prices[-1]:.2f}")
