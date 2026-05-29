"""
E-Threshold Adapter
===================

E-score threshold management and adaptation for auto-controlled context.

Key Features:
- Default threshold = 0.5 (validated in Q44 experiments)
- Per-session configurable threshold
- E-distribution tracking for threshold analysis
- Future: Auto-tune threshold based on response quality

Phase C.6 of Auto-Controlled Context Loop implementation.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EDistributionStats:
    """Statistics about E-score distribution for threshold analysis."""
    sample_count: int
    E_mean: float
    E_std: float
    E_min: float
    E_max: float
    E_median: float
    E_percentile_25: float
    E_percentile_75: float
    above_threshold_pct: float  # Percentage of items above current threshold

    @classmethod
    def from_scores(cls, E_scores: List[float], threshold: float) -> "EDistributionStats":
        """Compute statistics from a list of E-scores."""
        if not E_scores:
            return cls(
                sample_count=0,
                E_mean=0.0,
                E_std=0.0,
                E_min=0.0,
                E_max=0.0,
                E_median=0.0,
                E_percentile_25=0.0,
                E_percentile_75=0.0,
                above_threshold_pct=0.0,
            )

        arr = np.array(E_scores)
        above_count = np.sum(arr >= threshold)

        return cls(
            sample_count=len(E_scores),
            E_mean=float(np.mean(arr)),
            E_std=float(np.std(arr)) if len(arr) > 1 else 0.0,
            E_min=float(np.min(arr)),
            E_max=float(np.max(arr)),
            E_median=float(np.median(arr)),
            E_percentile_25=float(np.percentile(arr, 25)),
            E_percentile_75=float(np.percentile(arr, 75)),
            above_threshold_pct=float(above_count / len(arr)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "sample_count": self.sample_count,
            "E_mean": self.E_mean,
            "E_std": self.E_std,
            "E_min": self.E_min,
            "E_max": self.E_max,
            "E_median": self.E_median,
            "E_percentile_25": self.E_percentile_25,
            "E_percentile_75": self.E_percentile_75,
            "above_threshold_pct": self.above_threshold_pct,
        }


@dataclass
class ThresholdConfig:
    """Configuration for E-threshold."""
    threshold: float = 0.5  # Default from Q44 validation
    auto_adjust: bool = False  # Future: enable auto-adjustment
    min_threshold: float = 0.1  # Never go below this
    max_threshold: float = 0.9  # Never go above this
    adjustment_rate: float = 0.05  # How much to adjust per iteration

    def validate(self) -> None:
        """Validate threshold configuration."""
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {self.threshold}")
        if not (0.0 <= self.min_threshold < self.max_threshold <= 1.0):
            raise ValueError(
                f"Invalid threshold bounds: min={self.min_threshold}, max={self.max_threshold}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "auto_adjust": self.auto_adjust,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "adjustment_rate": self.adjustment_rate,
        }


@dataclass
class ThresholdHistory:
    """History of threshold changes."""
    timestamp: str
    old_threshold: float
    new_threshold: float
    reason: str
    E_stats: Optional[EDistributionStats] = None


# =============================================================================
# Threshold Adapter
# =============================================================================

class ThresholdAdapter:
    """
    Manages E-score threshold for auto-controlled context.

    Usage:
        adapter = ThresholdAdapter(config=ThresholdConfig(threshold=0.5))

        # Record E-scores from partitioning
        adapter.record_partition(E_scores=[0.7, 0.4, 0.8, 0.3])

        # Get current threshold
        threshold = adapter.get_threshold()

        # Manually adjust if needed
        adapter.adjust_threshold(0.6, reason="Manual tuning")

        # Get analysis
        stats = adapter.get_E_distribution_stats()
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize threshold adapter.

        Args:
            config: ThresholdConfig (default: threshold=0.5)
        """
        self.config = config or ThresholdConfig()
        self.config.validate()

        # History tracking
        self._history: List[ThresholdHistory] = []
        self._E_history: List[float] = []  # All recorded E-scores
        self._partition_count: int = 0

    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.config.threshold

    def adjust_threshold(
        self,
        new_threshold: float,
        reason: str = "Manual adjustment"
    ) -> ThresholdHistory:
        """
        Adjust threshold to a new value.

        Args:
            new_threshold: New threshold value
            reason: Reason for adjustment (for logging)

        Returns:
            ThresholdHistory record
        """
        # Clamp to bounds
        new_threshold = max(
            self.config.min_threshold,
            min(self.config.max_threshold, new_threshold)
        )

        old_threshold = self.config.threshold

        # Record history
        history = ThresholdHistory(
            timestamp=datetime.now(timezone.utc).isoformat(),
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            reason=reason,
            E_stats=self.get_E_distribution_stats() if self._E_history else None,
        )
        self._history.append(history)

        # Apply change
        self.config.threshold = new_threshold

        return history

    def record_partition(self, E_scores: List[float]) -> None:
        """
        Record E-scores from a partition operation.

        Used for threshold analysis and potential auto-adjustment.
        """
        self._E_history.extend(E_scores)
        self._partition_count += 1

        # Limit history size to prevent memory growth
        max_history = 10000
        if len(self._E_history) > max_history:
            self._E_history = self._E_history[-max_history:]

    def get_E_distribution_stats(self) -> Optional[EDistributionStats]:
        """Get statistics about recorded E-scores."""
        if not self._E_history:
            return None
        return EDistributionStats.from_scores(self._E_history, self.config.threshold)

    def suggest_threshold(self) -> Tuple[float, str]:
        """
        Suggest a threshold based on E-score distribution.

        Uses the 25th percentile as a reasonable threshold that would
        include ~75% of items. This is a heuristic, not a guarantee.

        Returns:
            (suggested_threshold, explanation)
        """
        stats = self.get_E_distribution_stats()
        if stats is None or stats.sample_count < 10:
            return self.config.threshold, "Insufficient data for suggestion"

        # Use 25th percentile as suggestion
        suggested = stats.E_percentile_25

        # Clamp to bounds
        suggested = max(
            self.config.min_threshold,
            min(self.config.max_threshold, suggested)
        )

        explanation = (
            f"Based on {stats.sample_count} samples: "
            f"E_mean={stats.E_mean:.3f}, E_std={stats.E_std:.3f}, "
            f"percentile_25={stats.E_percentile_25:.3f}. "
            f"Suggested threshold would include ~75% of items."
        )

        return suggested, explanation

    def get_above_threshold_ratio(self) -> float:
        """Get ratio of E-scores above current threshold."""
        stats = self.get_E_distribution_stats()
        return stats.above_threshold_pct if stats else 0.0

    def get_history(self) -> List[Dict[str, Any]]:
        """Get threshold change history."""
        return [
            {
                "timestamp": h.timestamp,
                "old_threshold": h.old_threshold,
                "new_threshold": h.new_threshold,
                "reason": h.reason,
                "E_stats": h.E_stats.to_dict() if h.E_stats else None,
            }
            for h in self._history
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of threshold state."""
        stats = self.get_E_distribution_stats()
        return {
            "current_threshold": self.config.threshold,
            "config": self.config.to_dict(),
            "partition_count": self._partition_count,
            "E_samples_recorded": len(self._E_history),
            "E_stats": stats.to_dict() if stats else None,
            "threshold_changes": len(self._history),
            "above_threshold_ratio": self.get_above_threshold_ratio(),
        }

    def reset_history(self) -> None:
        """Reset recorded E-scores (keeps threshold and change history)."""
        self._E_history = []
        self._partition_count = 0


# =============================================================================
# Utility Functions
# =============================================================================

def create_threshold_adapter_from_session(
    session_capsule: 'SessionCapsule',
    session_id: str
) -> ThresholdAdapter:
    """
    Create a ThresholdAdapter initialized from session history.

    Loads E-scores from partition events to populate the adapter.
    """
    from .session_capsule import SessionCapsule, EVENT_PARTITION

    adapter = ThresholdAdapter()

    # Load partition events
    events = session_capsule.get_events(session_id, event_type=EVENT_PARTITION)

    for event in events:
        # Extract E-scores from partition event
        # Note: We'd need to store per-item E-scores in the event
        # For now, just use E_mean/min/max to estimate distribution
        payload = event.payload
        E_mean = payload.get("E_mean", 0.5)
        E_min = payload.get("E_min", 0.0)
        E_max = payload.get("E_max", 1.0)

        # Generate synthetic samples from summary stats
        # (This is approximate - ideally we'd store full E-scores)
        if E_max > E_min:
            n_samples = payload.get("items_total", 10)
            synthetic_scores = np.linspace(E_min, E_max, n_samples).tolist()
            adapter.record_partition(synthetic_scores)

    return adapter


def recommended_threshold_for_budget(
    budget_tokens: int,
    avg_item_tokens: int,
    total_items: int
) -> float:
    """
    Recommend a threshold based on budget constraints.

    If budget can only fit N items, suggests a threshold that would
    select approximately N items (assuming uniform E distribution).
    """
    if total_items == 0 or avg_item_tokens == 0:
        return 0.5

    # How many items can fit in budget?
    max_items = budget_tokens // avg_item_tokens

    # What fraction of items should we include?
    include_fraction = min(1.0, max_items / total_items)

    # Threshold that would include this fraction
    # (1 - include_fraction) since E above threshold are included
    threshold = 1.0 - include_fraction

    # Clamp to reasonable range
    return max(0.1, min(0.9, threshold))


if __name__ == "__main__":
    # Quick sanity test
    print("Threshold Adapter - Sanity Test")
    print("=" * 50)

    adapter = ThresholdAdapter(ThresholdConfig(threshold=0.5))

    # Simulate partition E-scores
    np.random.seed(42)
    for _ in range(10):
        E_scores = np.random.beta(2, 5, 20).tolist()  # Skewed low
        adapter.record_partition(E_scores)

    print(f"\nCurrent threshold: {adapter.get_threshold()}")
    print(f"Above threshold ratio: {adapter.get_above_threshold_ratio():.1%}")

    stats = adapter.get_E_distribution_stats()
    if stats:
        print(f"\nE-Score Distribution:")
        print(f"  Mean: {stats.E_mean:.3f}")
        print(f"  Std: {stats.E_std:.3f}")
        print(f"  Median: {stats.E_median:.3f}")
        print(f"  25th percentile: {stats.E_percentile_25:.3f}")
        print(f"  75th percentile: {stats.E_percentile_75:.3f}")

    suggested, explanation = adapter.suggest_threshold()
    print(f"\nSuggested threshold: {suggested:.3f}")
    print(f"  {explanation}")

    # Adjust threshold
    adapter.adjust_threshold(suggested, reason="Following suggestion")
    print(f"\nNew threshold: {adapter.get_threshold()}")
    print(f"New above threshold ratio: {adapter.get_above_threshold_ratio():.1%}")

    print(f"\nSummary: {json.dumps(adapter.get_summary(), indent=2)}")
