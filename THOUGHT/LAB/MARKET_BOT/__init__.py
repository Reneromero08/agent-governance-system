# Psychohistory Market Bot
# Formula-driven trading using R = (E / grad_S) * sigma^Df

__version__ = "1.0.0"

# Re-export main components for easy access
from .core import (
    SignalState,
    AssetClass,
    SignalExtractor,
    PrimeRadiant,
    MarketFormulaExecutor,
    SeldonGate,
    GateTier,
    AlphaWarningLevel,
)

from .data import RealDataFetcher
from .utils import Notifier

__all__ = [
    "SignalState",
    "AssetClass",
    "SignalExtractor",
    "PrimeRadiant",
    "MarketFormulaExecutor",
    "SeldonGate",
    "GateTier",
    "AlphaWarningLevel",
    "RealDataFetcher",
    "Notifier",
]
