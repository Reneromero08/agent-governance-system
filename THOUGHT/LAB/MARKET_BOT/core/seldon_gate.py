"""
SELDON GATE
===========

The decision gate for Psychohistory trading.
Named after Hari Seldon who gated actions based on mathematical predictions.

Implements:
- R-gating (Q17): R >= threshold -> OPEN (trade allowed)
- Alpha drift detection (Q21): 5-12 step early warning
- Position limit enforcement based on tier

All thresholds are from validated FORMULA research, not hand-tuned.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# =============================================================================
# GATE CONFIGURATION (From Q17 Validation)
# =============================================================================

class GateTier(Enum):
    """
    Gate tiers with increasing R requirements.

    From Q17: R_high=57.3 vs R_low=0.69 (83x ratio)
    Thresholds calibrated for meaningful discrimination.
    """
    T0_OBSERVE = "T0_OBSERVE"      # Always allowed - just watch
    T1_SMALL_POS = "T1_SMALL_POS"  # Small position (<=5% portfolio)
    T2_MEDIUM_POS = "T2_MEDIUM_POS"  # Medium position (<=15% portfolio)
    T3_LARGE_POS = "T3_LARGE_POS"  # Large position (<=30% portfolio)


# R thresholds for each tier
GATE_THRESHOLDS = {
    GateTier.T0_OBSERVE: 0.0,      # Always open
    GateTier.T1_SMALL_POS: 0.5,    # Requires modest coherence
    GateTier.T2_MEDIUM_POS: 0.8,   # Requires good coherence
    GateTier.T3_LARGE_POS: 1.0,    # Requires strong coherence
}

# Position limits for each tier
POSITION_LIMITS = {
    GateTier.T0_OBSERVE: 0.0,      # No position
    GateTier.T1_SMALL_POS: 0.05,   # 5% of portfolio
    GateTier.T2_MEDIUM_POS: 0.15,  # 15% of portfolio
    GateTier.T3_LARGE_POS: 0.30,   # 30% of portfolio
}


# =============================================================================
# ALPHA DRIFT CONFIGURATION (From Q21 Validation)
# =============================================================================

class AlphaWarningLevel(Enum):
    """Alpha drift warning levels."""
    NONE = 0       # Alpha near 0.5, healthy
    WATCH = 1      # Minor drift, monitor
    ALERT = 2      # Significant drift, reduce exposure
    CRITICAL = 3   # Major drift, exit positions


# Alpha drift configuration
ALPHA_CONFIG = {
    'CRITICAL_ALPHA': 0.5,           # Healthy baseline (Riemann critical line)
    'DRIFT_SIGMA_THRESHOLD_WATCH': 1.5,   # 1.5 std devs = watch
    'DRIFT_SIGMA_THRESHOLD_ALERT': 2.0,   # 2.0 std devs = alert
    'DRIFT_SIGMA_THRESHOLD_CRITICAL': 3.0,  # 3.0 std devs = critical
    'BASELINE_WINDOW': 10,           # Steps to compute baseline
    'MIN_HISTORY': 5,                # Minimum history for detection
    'CONSERVATION_TARGET': 21.746,   # 8e
    'CONSERVATION_TOLERANCE': 0.10,  # 10% violation = escalate
}


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class GateDecision:
    """Result of gate decision."""
    status: str                    # "OPEN" or "CLOSED"
    tier: GateTier                 # Highest allowed tier
    R: float                       # R value used for decision
    threshold: float               # Threshold for this tier
    position_limit: float          # Max position size (fraction)
    reason: str                    # Explanation


@dataclass
class DriftResult:
    """Result of alpha drift detection."""
    warning_level: AlphaWarningLevel
    alpha: float                   # Current alpha
    baseline_alpha: float          # Baseline alpha
    drift: float                   # Deviation from 0.5
    sigma: float                   # Standard deviation of alpha
    lead_time_estimate: int        # Estimated steps before gate closure
    recommendation: str            # Action recommendation


@dataclass
class FullGateResult:
    """Complete gate assessment."""
    gate: GateDecision
    drift: DriftResult
    conservation_ok: bool          # Is Df*alpha near 8e?
    timestamp: str
    should_exit: bool              # Combined recommendation


# =============================================================================
# SELDON GATE CLASS
# =============================================================================

class SeldonGate:
    """
    The Seldon Gate for Psychohistory trading.

    Combines:
    - R-gating for position sizing
    - Alpha drift for early warning
    - Conservation law monitoring

    All decisions are formula-based, not LLM reasoning.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Seldon Gate.

        Args:
            config: Optional override for ALPHA_CONFIG
        """
        self.config = {**ALPHA_CONFIG, **(config or {})}
        self.alpha_history: List[float] = []
        self.R_history: List[float] = []

    # =========================================================================
    # R-GATING (Q17)
    # =========================================================================

    def check_gate(self, R: float, requested_tier: GateTier = GateTier.T3_LARGE_POS) -> GateDecision:
        """
        Check if gate is open for a given R value.

        From Q17: R_high=57.3 vs R_low=0.69 (83x discrimination)

        Args:
            R: The R value from formula executor
            requested_tier: The tier being requested

        Returns:
            GateDecision with status, allowed tier, and position limit
        """
        threshold = GATE_THRESHOLDS[requested_tier]

        if R >= threshold:
            return GateDecision(
                status="OPEN",
                tier=requested_tier,
                R=R,
                threshold=threshold,
                position_limit=POSITION_LIMITS[requested_tier],
                reason=f"R={R:.2f} >= threshold={threshold:.2f}"
            )
        else:
            # Find highest allowed tier
            allowed_tier = GateTier.T0_OBSERVE
            for tier in GateTier:
                if R >= GATE_THRESHOLDS[tier]:
                    allowed_tier = tier

            return GateDecision(
                status="CLOSED" if allowed_tier == GateTier.T0_OBSERVE else "PARTIAL",
                tier=allowed_tier,
                R=R,
                threshold=threshold,
                position_limit=POSITION_LIMITS[allowed_tier],
                reason=f"R={R:.2f} < threshold={threshold:.2f}, downgraded to {allowed_tier.value}"
            )

    def get_max_tier(self, R: float) -> GateTier:
        """Get the maximum tier allowed for a given R value."""
        max_tier = GateTier.T0_OBSERVE
        for tier in GateTier:
            if R >= GATE_THRESHOLDS[tier]:
                max_tier = tier
        return max_tier

    # =========================================================================
    # ALPHA DRIFT DETECTION (Q21)
    # =========================================================================

    def update_alpha(self, alpha: float):
        """Add alpha value to history."""
        self.alpha_history.append(alpha)
        # Keep bounded history
        if len(self.alpha_history) > 1000:
            self.alpha_history = self.alpha_history[-500:]

    def detect_drift(self, current_alpha: Optional[float] = None) -> DriftResult:
        """
        Detect alpha drift from critical value (0.5).

        From Q21: AUC=0.9955, 5-12 step lead time

        Args:
            current_alpha: Optional current alpha (uses latest from history if None)

        Returns:
            DriftResult with warning level and recommendation
        """
        if current_alpha is not None:
            self.update_alpha(current_alpha)

        if len(self.alpha_history) < self.config['MIN_HISTORY']:
            return DriftResult(
                warning_level=AlphaWarningLevel.NONE,
                alpha=current_alpha or 0.5,
                baseline_alpha=0.5,
                drift=0.0,
                sigma=0.0,
                lead_time_estimate=0,
                recommendation="Insufficient history for drift detection"
            )

        # Get baseline from recent window
        baseline_window = self.config['BASELINE_WINDOW']
        baseline_alphas = self.alpha_history[-baseline_window:]
        baseline_alpha = np.mean(baseline_alphas)
        sigma = np.std(baseline_alphas)

        # Current alpha
        alpha = self.alpha_history[-1] if current_alpha is None else current_alpha

        # Drift from critical value (0.5)
        drift = abs(alpha - self.config['CRITICAL_ALPHA'])

        # Determine warning level based on z-score
        if sigma > 0:
            z_score = drift / sigma
        else:
            z_score = 0

        if z_score >= self.config['DRIFT_SIGMA_THRESHOLD_CRITICAL']:
            warning_level = AlphaWarningLevel.CRITICAL
            lead_time = 2
            recommendation = "CRITICAL: Exit all positions immediately"
        elif z_score >= self.config['DRIFT_SIGMA_THRESHOLD_ALERT']:
            warning_level = AlphaWarningLevel.ALERT
            lead_time = 5
            recommendation = "ALERT: Reduce position sizes, prepare to exit"
        elif z_score >= self.config['DRIFT_SIGMA_THRESHOLD_WATCH']:
            warning_level = AlphaWarningLevel.WATCH
            lead_time = 9
            recommendation = "WATCH: Monitor closely, tighten stops"
        else:
            warning_level = AlphaWarningLevel.NONE
            lead_time = 0
            recommendation = "Alpha stable, normal operations"

        return DriftResult(
            warning_level=warning_level,
            alpha=alpha,
            baseline_alpha=baseline_alpha,
            drift=drift,
            sigma=sigma,
            lead_time_estimate=lead_time,
            recommendation=recommendation
        )

    # =========================================================================
    # CONSERVATION LAW CHECK
    # =========================================================================

    def check_conservation(self, Df: float, alpha: float) -> bool:
        """
        Check if conservation law Df * alpha = 8e is satisfied.

        Violation suggests structural anomaly in market signal space.
        """
        conservation = Df * alpha
        target = self.config['CONSERVATION_TARGET']
        tolerance = self.config['CONSERVATION_TOLERANCE']

        deviation = abs(conservation - target) / target
        return deviation <= tolerance

    # =========================================================================
    # FULL GATE ASSESSMENT
    # =========================================================================

    def assess(
        self,
        R: float,
        alpha: float,
        Df: float,
        requested_tier: GateTier = GateTier.T3_LARGE_POS
    ) -> FullGateResult:
        """
        Full gate assessment combining R-gating, drift detection, and conservation.

        Args:
            R: R value from formula executor
            alpha: Alpha value from formula executor
            Df: Effective dimensionality
            requested_tier: Tier being requested

        Returns:
            FullGateResult with combined assessment
        """
        # Record R
        self.R_history.append(R)
        if len(self.R_history) > 1000:
            self.R_history = self.R_history[-500:]

        # R-gating
        gate = self.check_gate(R, requested_tier)

        # Drift detection
        drift = self.detect_drift(alpha)

        # Conservation check
        conservation_ok = self.check_conservation(Df, alpha)

        # Combined should_exit decision
        should_exit = (
            gate.status == "CLOSED" or
            drift.warning_level == AlphaWarningLevel.CRITICAL or
            (drift.warning_level == AlphaWarningLevel.ALERT and not conservation_ok)
        )

        return FullGateResult(
            gate=gate,
            drift=drift,
            conservation_ok=conservation_ok,
            timestamp=datetime.now().isoformat(),
            should_exit=should_exit
        )

    # =========================================================================
    # POSITION ADJUSTMENT
    # =========================================================================

    def adjust_position_limit(self, base_limit: float, drift_result: DriftResult) -> float:
        """
        Adjust position limit based on drift warning level.

        Higher warning = more conservative position sizing.
        """
        adjustments = {
            AlphaWarningLevel.NONE: 1.0,
            AlphaWarningLevel.WATCH: 0.75,
            AlphaWarningLevel.ALERT: 0.50,
            AlphaWarningLevel.CRITICAL: 0.0,
        }
        return base_limit * adjustments[drift_result.warning_level]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get gate statistics."""
        return {
            "alpha_history_len": len(self.alpha_history),
            "R_history_len": len(self.R_history),
            "mean_alpha": np.mean(self.alpha_history) if self.alpha_history else 0.5,
            "std_alpha": np.std(self.alpha_history) if len(self.alpha_history) > 1 else 0,
            "mean_R": np.mean(self.R_history) if self.R_history else 0,
        }


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SELDON GATE - Psychohistory Market Bot")
    print("=" * 60)

    gate = SeldonGate()

    # Test R-gating
    print("\n--- R-Gating Test (Q17) ---")
    test_R_values = [0.3, 0.5, 0.8, 1.5, 5.0]
    for R in test_R_values:
        decision = gate.check_gate(R, GateTier.T3_LARGE_POS)
        print(f"  R={R:.1f}: {decision.status} | Tier={decision.tier.value} | Limit={decision.position_limit:.0%}")

    # Test alpha drift detection
    print("\n--- Alpha Drift Test (Q21) ---")

    # Simulate stable alpha
    print("  Stable alpha (0.48-0.52):")
    for _ in range(20):
        gate.update_alpha(0.5 + np.random.randn() * 0.02)
    drift = gate.detect_drift()
    print(f"    Warning: {drift.warning_level.name} | Drift: {drift.drift:.4f}")

    # Simulate drifting alpha
    print("  Drifting alpha (toward 0.7):")
    for i in range(10):
        gate.update_alpha(0.5 + i * 0.02)
    drift = gate.detect_drift()
    print(f"    Warning: {drift.warning_level.name} | Drift: {drift.drift:.4f}")
    print(f"    Lead time estimate: {drift.lead_time_estimate} steps")
    print(f"    Recommendation: {drift.recommendation}")

    # Test full assessment
    print("\n--- Full Assessment Test ---")
    result = gate.assess(R=1.2, alpha=0.68, Df=22.0, requested_tier=GateTier.T2_MEDIUM_POS)
    print(f"  Gate: {result.gate.status} ({result.gate.tier.value})")
    print(f"  Drift: {result.drift.warning_level.name}")
    print(f"  Conservation OK: {result.conservation_ok}")
    print(f"  Should Exit: {result.should_exit}")

    # Test conservation law
    print("\n--- Conservation Law Test ---")
    test_cases = [
        (22.0, 0.99),   # Df*alpha = 21.78 (close to 8e)
        (44.0, 0.49),   # Df*alpha = 21.56 (close to 8e)
        (10.0, 0.5),    # Df*alpha = 5.0 (violation)
        (50.0, 0.2),    # Df*alpha = 10.0 (violation)
    ]
    for Df, alpha in test_cases:
        ok = gate.check_conservation(Df, alpha)
        print(f"  Df={Df:.1f}, alpha={alpha:.2f}: Df*alpha={Df*alpha:.2f} -> {'OK' if ok else 'VIOLATION'}")

    print("\n--- Seldon Gate Ready ---")
