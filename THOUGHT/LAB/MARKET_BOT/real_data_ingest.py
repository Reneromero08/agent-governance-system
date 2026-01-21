"""
REAL DATA INGESTION
===================

Fetches real historical market data for backtesting.
Uses yfinance for stocks/ETFs (free, goes back decades).

The bot will run through history WITHOUT knowing the future.
We know the outcomes - the bot doesn't.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OHLCV:
    """Single OHLCV bar."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float


@dataclass
class MarketData:
    """Market data for an asset."""
    symbol: str
    bars: List[OHLCV]
    start_date: str
    end_date: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([
            {
                'timestamp': b.timestamp,
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume,
                'adj_close': b.adj_close,
            }
            for b in self.bars
        ])

    def get_prices(self) -> List[float]:
        """Get close prices."""
        return [b.close for b in self.bars]

    def get_returns(self) -> List[float]:
        """Get daily returns."""
        prices = self.get_prices()
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]


# =============================================================================
# HISTORICAL EVENTS
# =============================================================================

# Known crisis events for backtesting
HISTORICAL_EVENTS = {
    "covid_crash_2020": {
        "description": "COVID-19 market crash",
        "start": "2020-01-01",
        "end": "2020-06-30",
        "crash_start": "2020-02-19",  # Market peak
        "crash_end": "2020-03-23",    # Market bottom
        "recovery_end": "2020-06-08", # V-recovery
        "max_drawdown": -0.34,        # ~34% drop
    },
    "financial_crisis_2008": {
        "description": "2008 Financial Crisis",
        "start": "2007-10-01",
        "end": "2009-06-30",
        "crash_start": "2007-10-09",  # Market peak
        "crash_end": "2009-03-09",    # Market bottom
        "recovery_end": "2009-06-30", # Partial recovery
        "max_drawdown": -0.57,        # ~57% drop
    },
    "flash_crash_2010": {
        "description": "Flash Crash of May 6, 2010",
        "start": "2010-04-01",
        "end": "2010-06-30",
        "crash_start": "2010-05-06",
        "crash_end": "2010-05-06",    # Same day
        "recovery_end": "2010-05-07",
        "max_drawdown": -0.09,        # ~9% intraday
    },
    "vix_explosion_2018": {
        "description": "February 2018 VIX explosion",
        "start": "2018-01-01",
        "end": "2018-04-30",
        "crash_start": "2018-01-26",
        "crash_end": "2018-02-09",
        "recovery_end": "2018-04-30",
        "max_drawdown": -0.12,        # ~12% drop
    },
    "china_deval_2015": {
        "description": "China devaluation / Aug 2015 crash",
        "start": "2015-07-01",
        "end": "2015-12-31",
        "crash_start": "2015-08-18",
        "crash_end": "2015-08-25",
        "recovery_end": "2015-11-03",
        "max_drawdown": -0.12,
    },
    "dot_com_crash": {
        "description": "Dot-com bubble burst",
        "start": "2000-01-01",
        "end": "2002-12-31",
        "crash_start": "2000-03-10",  # NASDAQ peak
        "crash_end": "2002-10-09",    # Bottom
        "recovery_end": "2002-12-31",
        "max_drawdown": -0.49,        # ~49% for SPY
    },
    # NOTE: Black Monday 1987 removed - SPY didn't exist until 1993
    # For pre-1993 events, use ^GSPC (S&P 500 index) or DIA alternative
}


# =============================================================================
# DATA FETCHER
# =============================================================================

class RealDataFetcher:
    """
    Fetches real historical market data.

    Uses yfinance for free data access.
    Caches data locally to avoid repeated API calls.
    """

    CACHE_DIR = Path(__file__).parent / "data_cache"

    def __init__(self):
        """Initialize data fetcher."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required. Install: pip install yfinance")

        self.CACHE_DIR.mkdir(exist_ok=True)

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> MarketData:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "QQQ", "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            MarketData with OHLCV bars
        """
        cache_file = self.CACHE_DIR / f"{symbol}_{start_date}_{end_date}.json"

        # Check cache
        if use_cache and cache_file.exists():
            return self._load_cache(cache_file)

        # Fetch from yfinance
        print(f"Fetching {symbol} from {start_date} to {end_date}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to OHLCV bars
        bars = []
        for idx, row in df.iterrows():
            bars.append(OHLCV(
                timestamp=idx.strftime("%Y-%m-%d"),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                adj_close=float(row.get('Adj Close', row['Close'])),
            ))

        data = MarketData(
            symbol=symbol,
            bars=bars,
            start_date=start_date,
            end_date=end_date,
        )

        # Cache it
        self._save_cache(data, cache_file)

        return data

    def fetch_event(
        self,
        event_name: str,
        symbol: str = "SPY",
        buffer_days: int = 30
    ) -> Tuple[MarketData, Dict]:
        """
        Fetch data for a known historical event.

        Args:
            event_name: Key from HISTORICAL_EVENTS
            symbol: Ticker to fetch
            buffer_days: Extra days before/after event

        Returns:
            (MarketData, event_info) tuple
        """
        if event_name not in HISTORICAL_EVENTS:
            raise ValueError(f"Unknown event: {event_name}. Available: {list(HISTORICAL_EVENTS.keys())}")

        event = HISTORICAL_EVENTS[event_name]

        # Add buffer
        start = (datetime.strptime(event["start"], "%Y-%m-%d") - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        end = (datetime.strptime(event["end"], "%Y-%m-%d") + timedelta(days=buffer_days)).strftime("%Y-%m-%d")

        data = self.fetch(symbol, start, end)

        return data, event

    def _save_cache(self, data: MarketData, cache_file: Path):
        """Save data to cache."""
        cache_data = {
            "symbol": data.symbol,
            "start_date": data.start_date,
            "end_date": data.end_date,
            "bars": [
                {
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "adj_close": b.adj_close,
                }
                for b in data.bars
            ]
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    def _load_cache(self, cache_file: Path) -> MarketData:
        """Load data from cache."""
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        bars = [
            OHLCV(**b) for b in cache_data["bars"]
        ]

        return MarketData(
            symbol=cache_data["symbol"],
            bars=bars,
            start_date=cache_data["start_date"],
            end_date=cache_data["end_date"],
        )


# =============================================================================
# TECHNICAL INDICATORS (for signal extraction)
# =============================================================================

def compute_sma(prices: List[float], window: int) -> List[float]:
    """Simple moving average."""
    result = [None] * (window - 1)
    for i in range(window - 1, len(prices)):
        result.append(np.mean(prices[i - window + 1:i + 1]))
    return result


def compute_rsi(prices: List[float], window: int = 14) -> List[float]:
    """Relative Strength Index."""
    if len(prices) < window + 1:
        return [50.0] * len(prices)

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    gains = [max(d, 0) for d in deltas]
    losses = [-min(d, 0) for d in deltas]

    result = [50.0] * window

    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        result.append(rsi)

    # Pad to match original length
    result = [50.0] + result
    return result


def compute_volatility(prices: List[float], window: int = 20) -> List[float]:
    """Rolling volatility (std of returns)."""
    if len(prices) < window + 1:
        return [0.01] * len(prices)

    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    result = [0.01] * window
    for i in range(window, len(returns)):
        vol = np.std(returns[i - window + 1:i + 1])
        result.append(vol)

    result = [0.01] + result
    return result


def compute_volume_ratio(volumes: List[float], window: int = 20) -> List[float]:
    """Volume relative to moving average."""
    if len(volumes) < window:
        return [1.0] * len(volumes)

    sma = compute_sma(volumes, window)

    result = []
    for i, vol in enumerate(volumes):
        if sma[i] is not None and sma[i] > 0:
            result.append(vol / sma[i])
        else:
            result.append(1.0)

    return result


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REAL DATA INGESTION - Demo")
    print("=" * 60)

    fetcher = RealDataFetcher()

    # Fetch COVID crash data
    print("\n--- Fetching COVID Crash Data ---")
    data, event = fetcher.fetch_event("covid_crash_2020", symbol="SPY")

    print(f"Symbol: {data.symbol}")
    print(f"Period: {data.start_date} to {data.end_date}")
    print(f"Bars: {len(data.bars)}")
    print(f"Event: {event['description']}")
    print(f"Expected max drawdown: {event['max_drawdown']:.1%}")

    # Show some data
    print("\n--- Sample Data ---")
    df = data.to_dataframe()
    print(df.head(10))

    # Compute some indicators
    print("\n--- Technical Indicators ---")
    prices = data.get_prices()
    rsi = compute_rsi(prices)
    vol = compute_volatility(prices)

    print(f"RSI range: {min(rsi):.1f} - {max(rsi):.1f}")
    print(f"Volatility range: {min(vol):.4f} - {max(vol):.4f}")

    # List available events
    print("\n--- Available Historical Events ---")
    for name, info in HISTORICAL_EVENTS.items():
        print(f"  {name}: {info['description']} ({info['start']} to {info['end']})")

    print("\n--- Data Ingestion Ready ---")
