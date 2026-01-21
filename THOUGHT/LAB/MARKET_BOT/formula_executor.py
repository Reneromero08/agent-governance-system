"""
FORMULA EXECUTOR
================

Executes R = (E / grad_S) * sigma^Df on market signals.
Adapts q44_core for market signal vectors.

This is the mathematical heart of Psychohistory trading.
No LLM reasoning - pure deterministic computation.

Key formulas from FORMULA research:
- E = mean overlap (quantum inner product, r=0.97-0.99 with P_born)
- grad_S = std(overlaps) (regime uncertainty)
- sigma = sqrt(n) (signal redundancy)
- Df = participation ratio (effective dimensionality)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add FORMULA path for q44_core imports
FORMULA_PATH = Path(__file__).parent.parent / "FORMULA" / "experiments" / "open_questions" / "q44"
sys.path.insert(0, str(FORMULA_PATH))

try:
    from q44_core import (
        normalize,
        compute_E_linear,
        compute_E_variants,
        compute_grad_S,
        compute_sigma,
        compute_Df_from_context,
        compute_born_probability,
        pearson_correlation,
    )
    Q44_AVAILABLE = True
except ImportError:
    Q44_AVAILABLE = False
    # Fallback implementations if q44_core not available
    def normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-10)


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class FormulaResult:
    """Result of R formula computation."""
    R: float                    # The R value
    E: float                    # Mean overlap (essence)
    grad_S: float               # Overlap std (entropy gradient)
    sigma: float                # sqrt(n) redundancy
    Df: float                   # Effective dimensionality
    overlaps: List[float]       # Raw overlap values
    P_born: float               # Born probability for comparison
    regime: str                 # STABLE/TRANSITIONAL/SHIFT


@dataclass
class AlphaResult:
    """Result of alpha/Df computation for conservation law."""
    alpha: float                # Eigenvalue decay exponent
    Df: float                   # Effective dimensionality
    conservation: float         # Df * alpha (should be ~8e = 21.746)
    conservation_error: float   # Deviation from 8e


@dataclass
class RVelocityResult:
    """Result of R velocity (dR/dt) computation - THE KEY INSIGHT."""
    R: float                    # Current R value
    R_prev: float               # Previous R value
    dR: float                   # R velocity (change per step)
    dR_normalized: float        # Normalized velocity (-1 to +1)
    momentum_phase: str         # BUILDING, STABLE, FADING, REVERSING
    signal: str                 # BUY, HOLD, SELL, WAIT
    confidence: float           # 0-1 signal confidence


# =============================================================================
# FORMULA EXECUTOR CLASS
# =============================================================================

class MarketFormulaExecutor:
    """
    Executes R = (E / grad_S) * sigma^Df on market signals.

    Takes signal state vectors (from Prime Radiant) and computes:
    - R value for regime assessment
    - Alpha for drift prediction
    - Conservation law check

    All computations are deterministic NumPy operations.
    """

    # Conservation law target (from Q48-50)
    CONSERVATION_TARGET = 8 * np.e  # ~21.746

    # Regime classification thresholds
    # These can be tuned based on validation
    SHIFT_THRESHOLD = 0.05
    STABLE_THRESHOLD = -0.03

    def __init__(self):
        """Initialize formula executor."""
        self._check_q44_available()

    def _check_q44_available(self):
        """Check if q44_core functions are available."""
        if not Q44_AVAILABLE:
            print("WARNING: q44_core not available, using fallback implementations")

    # =========================================================================
    # CORE R COMPUTATION
    # =========================================================================

    def compute_R(
        self,
        query_vec: np.ndarray,
        context_vecs: List[np.ndarray],
        Df_override: Optional[float] = None,
    ) -> FormulaResult:
        """
        Compute R = (E / grad_S) * sigma^Df

        Args:
            query_vec: Current market state vector (normalized)
            context_vecs: Historical state vectors (context)
            Df_override: Optional Df value (if already computed)

        Returns:
            FormulaResult with all components
        """
        if len(context_vecs) == 0:
            return FormulaResult(
                R=0.0, E=0.0, grad_S=1.0, sigma=1.0, Df=1.0,
                overlaps=[], P_born=0.0, regime="STABLE"
            )

        # Normalize vectors
        psi = normalize(query_vec)
        context_normalized = [normalize(phi) for phi in context_vecs]

        # Compute overlaps (quantum inner products)
        overlaps = [float(np.dot(psi, phi)) for phi in context_normalized]

        # E = mean overlap
        E = np.mean(overlaps)

        # grad_S = std of overlaps
        if len(overlaps) < 2:
            grad_S = 1.0
        else:
            grad_S = float(max(np.std(overlaps, ddof=0), 1e-6))

        # sigma = sqrt(n)
        sigma = float(np.sqrt(len(context_vecs)))

        # Df = effective dimensionality from context
        if Df_override is not None:
            Df = Df_override
        else:
            Df = self._compute_Df(context_normalized)

        # R formula
        R = (E / grad_S) * (sigma ** Df) if grad_S > 0 else 0.0

        # Born probability for comparison
        P_born = self._compute_born(psi, context_normalized)

        # Regime classification
        regime = self._classify_regime(E, grad_S, overlaps)

        return FormulaResult(
            R=float(R),
            E=float(E),
            grad_S=float(grad_S),
            sigma=float(sigma),
            Df=float(Df),
            overlaps=overlaps,
            P_born=float(P_born),
            regime=regime,
        )

    def compute_R_simple(
        self,
        query_vec: np.ndarray,
        context_vecs: List[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Simplified R computation: R = E / grad_S

        Without sigma^Df scaling - useful for quick regime checks.

        Returns: (R_simple, E)
        """
        if len(context_vecs) == 0:
            return 0.0, 0.0

        psi = normalize(query_vec)
        context_normalized = [normalize(phi) for phi in context_vecs]

        overlaps = [float(np.dot(psi, phi)) for phi in context_normalized]
        E = np.mean(overlaps)

        if len(overlaps) < 2:
            grad_S = 1.0
        else:
            grad_S = float(max(np.std(overlaps, ddof=0), 1e-6))

        R_simple = E / grad_S

        return float(R_simple), float(E)

    # =========================================================================
    # ALPHA / CONSERVATION LAW
    # =========================================================================

    def compute_alpha(self, context_vecs: List[np.ndarray]) -> AlphaResult:
        """
        Compute alpha (eigenvalue decay exponent) and conservation law.

        From Q48-50: Df * alpha = 8e (~21.746)
        Alpha near 0.5 = healthy (Riemann critical line)
        Alpha drifting from 0.5 = potential regime change
        """
        if len(context_vecs) < 3:
            return AlphaResult(
                alpha=0.5,
                Df=1.0,
                conservation=0.5,
                conservation_error=abs(0.5 - self.CONSERVATION_TARGET),
            )

        # Stack embeddings
        X = np.array([normalize(v) for v in context_vecs])

        # Gram matrix
        gram = X @ X.T

        # Eigenvalues (sorted descending)
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) < 2:
            return AlphaResult(
                alpha=0.5, Df=1.0,
                conservation=0.5,
                conservation_error=abs(0.5 - self.CONSERVATION_TARGET),
            )

        # Compute Df (participation ratio)
        sum_lambda = np.sum(eigenvalues)
        sum_lambda_sq = np.sum(eigenvalues ** 2)
        Df = float((sum_lambda ** 2) / sum_lambda_sq)

        # Compute alpha (power law decay exponent)
        # lambda_k ~ k^(-alpha)
        k = np.arange(1, len(eigenvalues) + 1)
        log_k = np.log(k + 1)  # +1 to avoid log(0)
        log_lambda = np.log(eigenvalues + 1e-10)

        # Linear regression to find slope = -alpha
        slope, _ = np.polyfit(log_k, log_lambda, 1)
        alpha = float(-slope)

        # Clamp alpha to reasonable range
        alpha = max(0.1, min(2.0, alpha))

        # Conservation law
        conservation = Df * alpha
        conservation_error = abs(conservation - self.CONSERVATION_TARGET)

        return AlphaResult(
            alpha=alpha,
            Df=Df,
            conservation=conservation,
            conservation_error=conservation_error,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _compute_Df(self, context_vecs: List[np.ndarray]) -> float:
        """Compute effective dimensionality from context."""
        if len(context_vecs) < 2:
            return 1.0

        X = np.array(context_vecs)
        gram = X @ X.T

        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) == 0:
            return 1.0

        sum_lambda = np.sum(eigenvalues)
        sum_lambda_sq = np.sum(eigenvalues ** 2)

        return float((sum_lambda ** 2) / sum_lambda_sq)

    def _compute_born(self, psi: np.ndarray, context_vecs: List[np.ndarray]) -> float:
        """Compute Born probability P = |<psi|phi>|^2."""
        if len(context_vecs) == 0:
            return 0.0

        # Context superposition
        phi_sum = np.sum(context_vecs, axis=0)
        phi_context = phi_sum / np.sqrt(len(context_vecs))

        # Born rule
        overlap = np.dot(psi, phi_context)
        P_born = abs(overlap) ** 2

        return float(P_born)

    def _classify_regime(
        self,
        E: float,
        grad_S: float,
        overlaps: List[float]
    ) -> str:
        """
        Classify regime based on formula components.

        STABLE: High E, low grad_S (coherent signals)
        SHIFT: Low E, high grad_S (conflicting signals)
        TRANSITIONAL: In between
        """
        # Shift score = how much signals disagree
        shift_score = grad_S - E

        if shift_score > self.SHIFT_THRESHOLD:
            return "SHIFT"
        elif shift_score < self.STABLE_THRESHOLD:
            return "STABLE"
        else:
            return "TRANSITIONAL"

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def compute_R_sequence(
        self,
        state_vectors: List[np.ndarray],
        lookback: int = 10,
    ) -> List[FormulaResult]:
        """
        Compute R for a sequence of states.

        Each state uses previous `lookback` states as context.
        """
        results = []

        for i, query_vec in enumerate(state_vectors):
            # Get context from previous states
            start_idx = max(0, i - lookback)
            context_vecs = state_vectors[start_idx:i]

            if len(context_vecs) == 0:
                # First state - no context
                results.append(FormulaResult(
                    R=0.0, E=0.0, grad_S=1.0, sigma=1.0, Df=1.0,
                    overlaps=[], P_born=0.0, regime="STABLE"
                ))
            else:
                results.append(self.compute_R(query_vec, context_vecs))

        return results

    def compute_alpha_sequence(
        self,
        state_vectors: List[np.ndarray],
        window: int = 20,
    ) -> List[AlphaResult]:
        """
        Compute alpha for a rolling window through states.
        """
        results = []

        for i in range(len(state_vectors)):
            start_idx = max(0, i - window + 1)
            window_vecs = state_vectors[start_idx:i + 1]

            results.append(self.compute_alpha(window_vecs))

        return results

    # =========================================================================
    # R VELOCITY (dR/dt) - THE KEY INSIGHT
    # =========================================================================

    def compute_R_velocity(
        self,
        R_history: List[float],
        price_trend: float = 0.0,
        smoothing_window: int = 3,
    ) -> RVelocityResult:
        """
        Compute R velocity (dR/dt) - the rate of change of resonance.

        THE KEY INSIGHT:
        - dR/dt > 0 = coherence building = momentum strengthening = ENTRY
        - dR/dt < 0 = coherence breaking = momentum fading = EXIT
        - dR/dt ~ 0, R high = stable trend = HOLD
        - dR/dt ~ 0, R low = noise = WAIT

        Args:
            R_history: List of recent R values (oldest to newest)
            price_trend: Direction of price movement (+1 up, -1 down, 0 flat)
            smoothing_window: Window for smoothing R values

        Returns:
            RVelocityResult with velocity, phase, and trading signal
        """
        if len(R_history) < 2:
            return RVelocityResult(
                R=R_history[-1] if R_history else 0.0,
                R_prev=0.0,
                dR=0.0,
                dR_normalized=0.0,
                momentum_phase="WAIT",
                signal="WAIT",
                confidence=0.0,
            )

        # Smooth R values to reduce noise
        if len(R_history) >= smoothing_window:
            R_smoothed = [
                np.mean(R_history[max(0, i-smoothing_window+1):i+1])
                for i in range(len(R_history))
            ]
        else:
            R_smoothed = R_history

        R_current = R_smoothed[-1]
        R_prev = R_smoothed[-2]

        # Raw velocity
        dR = R_current - R_prev

        # Normalized velocity (scale by typical R range)
        R_std = np.std(R_history) if len(R_history) > 2 else 1.0
        dR_normalized = dR / max(R_std, 0.01)
        dR_normalized = np.clip(dR_normalized, -1.0, 1.0)

        # Classify momentum phase
        # Thresholds tuned for sensitivity
        VELOCITY_THRESHOLD = 0.1  # Significant change
        R_HIGH_THRESHOLD = 0.8    # "High" R (relative to recent)
        R_LOW_THRESHOLD = 0.3     # "Low" R

        # Dynamic thresholds based on recent R range
        R_mean = np.mean(R_history)
        R_high = R_mean + 0.5 * R_std if R_std > 0 else R_HIGH_THRESHOLD
        R_low = R_mean - 0.5 * R_std if R_std > 0 else R_LOW_THRESHOLD

        if dR_normalized > VELOCITY_THRESHOLD:
            momentum_phase = "BUILDING"  # Resonance increasing
        elif dR_normalized < -VELOCITY_THRESHOLD:
            momentum_phase = "FADING"    # Resonance decreasing
        elif R_current > R_high:
            momentum_phase = "STABLE"    # High and stable
        else:
            momentum_phase = "WAITING"   # Low or unclear

        # Generate trading signal based on phase + price trend
        signal = "WAIT"
        confidence = 0.0

        if momentum_phase == "BUILDING":
            if price_trend > 0:
                signal = "BUY"
                confidence = min(abs(dR_normalized) * R_current, 1.0)
            elif price_trend < 0:
                signal = "SHORT"  # Or avoid if no shorting
                confidence = min(abs(dR_normalized) * R_current, 1.0) * 0.7

        elif momentum_phase == "FADING":
            if R_current > R_low:  # Was high, now falling
                signal = "SELL"
                confidence = min(abs(dR_normalized), 1.0)
            else:
                signal = "WAIT"
                confidence = 0.3

        elif momentum_phase == "STABLE":
            if price_trend > 0:
                signal = "HOLD"
                confidence = min(R_current / max(R_high, 0.01), 1.0)
            elif price_trend < 0:
                signal = "SELL"
                confidence = 0.5

        return RVelocityResult(
            R=float(R_current),
            R_prev=float(R_prev),
            dR=float(dR),
            dR_normalized=float(dR_normalized),
            momentum_phase=momentum_phase,
            signal=signal,
            confidence=float(confidence),
        )

    def compute_R_velocity_sequence(
        self,
        state_vectors: List[np.ndarray],
        price_changes: List[float],
        lookback: int = 10,
    ) -> List[RVelocityResult]:
        """
        Compute R velocity for a sequence of states.

        Args:
            state_vectors: List of state vectors
            price_changes: List of price changes (for trend direction)
            lookback: Context window for R computation

        Returns:
            List of RVelocityResult for each step
        """
        # First compute R sequence
        R_results = self.compute_R_sequence(state_vectors, lookback)
        R_values = [r.R for r in R_results]

        # Then compute velocity for each point
        velocity_results = []
        for i in range(len(state_vectors)):
            R_history = R_values[max(0, i-5):i+1]  # Use recent 5 R values

            # Get price trend (positive = up, negative = down)
            if i < len(price_changes):
                price_trend = 1.0 if price_changes[i] > 0.005 else (-1.0 if price_changes[i] < -0.005 else 0.0)
            else:
                price_trend = 0.0

            velocity_results.append(
                self.compute_R_velocity(R_history, price_trend)
            )

        return velocity_results


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FORMULA EXECUTOR - Psychohistory Market Bot")
    print("=" * 60)

    executor = MarketFormulaExecutor()

    # Generate synthetic test data
    print("\n--- Synthetic Test ---")
    np.random.seed(42)

    # Coherent signals (should have high R, STABLE regime)
    coherent_query = np.random.randn(384)
    coherent_context = [coherent_query + np.random.randn(384) * 0.1 for _ in range(5)]

    result_coherent = executor.compute_R(coherent_query, coherent_context)
    print(f"\nCoherent signals:")
    print(f"  R = {result_coherent.R:.4f}")
    print(f"  E = {result_coherent.E:.4f}")
    print(f"  grad_S = {result_coherent.grad_S:.4f}")
    print(f"  Regime = {result_coherent.regime}")

    # Conflicting signals (should have low R, SHIFT regime)
    conflicting_query = np.random.randn(384)
    conflicting_context = [np.random.randn(384) for _ in range(5)]  # Random, uncorrelated

    result_conflicting = executor.compute_R(conflicting_query, conflicting_context)
    print(f"\nConflicting signals:")
    print(f"  R = {result_conflicting.R:.4f}")
    print(f"  E = {result_conflicting.E:.4f}")
    print(f"  grad_S = {result_conflicting.grad_S:.4f}")
    print(f"  Regime = {result_conflicting.regime}")

    # Alpha test
    print("\n--- Alpha / Conservation Law Test ---")
    alpha_result = executor.compute_alpha(coherent_context + conflicting_context)
    print(f"  alpha = {alpha_result.alpha:.4f}")
    print(f"  Df = {alpha_result.Df:.4f}")
    print(f"  Df * alpha = {alpha_result.conservation:.4f}")
    print(f"  Target (8e) = {executor.CONSERVATION_TARGET:.4f}")
    print(f"  Error = {alpha_result.conservation_error:.4f}")

    # R discrimination test
    print("\n--- R Discrimination Test ---")
    print(f"  R_coherent / R_conflicting = {result_coherent.R / max(result_conflicting.R, 1e-6):.2f}x")

    if result_coherent.R > result_conflicting.R * 2:
        print("  [PASS] Formula discriminates coherent from conflicting")
    else:
        print("  [WARN] Discrimination may be weak")

    print("\n--- Formula Executor Ready ---")
