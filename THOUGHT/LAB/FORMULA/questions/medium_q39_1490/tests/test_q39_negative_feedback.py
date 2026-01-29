#!/usr/bin/env python3
"""
Q39 Test 3: Negative Feedback Quantification

Hypothesis: Low M triggers increased evidence gathering.

Protocol:
1. Monitor (M(t), E(t)) pairs during operation
2. Compute correlation: corr(M, dE/dt)
3. Should be NEGATIVE (low M → high dE/dt)

Pass Criteria:
- Correlation < -0.3 (significant negative relationship)
- Effect size increases with deviation magnitude
- Feedback is proportional (not just threshold-based)

Run:
    pytest test_q39_negative_feedback.py -v
"""

import numpy as np
import pytest
from typing import List, Dict, Tuple
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path

from q39_homeostasis_utils import (
    compute_R, compute_M, HomeostasisState,
    compute_feedback_correlation, EPS
)


# =============================================================================
# Active Inference Feedback System
# =============================================================================

class FeedbackSystem:
    """
    System that implements Active Inference with measurable feedback.

    The core loop:
    1. Observe M (meaning field value)
    2. If M < M*, increase evidence gathering rate E
    3. More evidence → M rises toward M*
    4. This IS negative feedback

    We measure: does corr(M, dE/dt) < 0?
    """

    def __init__(self, M_star: float = 0.5, tau: float = 1.732,
                 feedback_gain: float = 0.5, seed: int = 42):
        self.M_star = M_star
        self.tau = tau
        self.feedback_gain = feedback_gain
        self.rng = np.random.default_rng(seed)

        # State
        self.M = M_star
        self.E = 1.0  # Evidence level
        self.R = np.exp(M_star)

        # History
        self.M_history = []
        self.E_history = []
        self.dE_history = []

    def compute_evidence_response(self, M: float) -> float:
        """
        Compute how much evidence to gather based on M.

        Active Inference: low M → high evidence gathering
        This is the core negative feedback mechanism.
        """
        # Error signal: deviation from equilibrium
        error = self.M_star - M

        # Proportional response (negative feedback)
        # When M < M*, error > 0, so dE > 0 (gather more evidence)
        dE = self.feedback_gain * error

        # Add some noise
        dE += self.rng.normal(0, 0.1)

        # Can't have negative evidence gathering
        return max(dE, -0.5)

    def step(self, external_perturbation: float = 0.0):
        """
        Single simulation step.

        Args:
            external_perturbation: External force pushing M away from equilibrium
        """
        # Record current state
        self.M_history.append(self.M)
        self.E_history.append(self.E)

        # Compute evidence response (the feedback signal)
        dE = self.compute_evidence_response(self.M)
        self.dE_history.append(dE)

        # Update evidence level
        self.E = max(0.1, self.E + dE * 0.1)

        # M responds to evidence (higher E → M moves toward M*)
        # This is the "action" part of Active Inference
        dM = 0.2 * (self.E - 1.0) * (self.M_star - self.M) / (abs(self.M_star - self.M) + 0.1)

        # Add external perturbation
        dM += external_perturbation

        # Add noise
        dM += self.rng.normal(0, 0.05)

        # Update M
        self.M = self.M + dM

    def run_episode(self, n_steps: int = 200,
                    perturbation_schedule: List[Tuple[int, float]] = None) -> Dict:
        """
        Run a full episode with optional perturbations.

        Args:
            n_steps: Number of steps
            perturbation_schedule: List of (step, magnitude) tuples

        Returns:
            Dict with histories and correlation analysis
        """
        # Reset
        self.M = self.M_star + self.rng.normal(0, 0.3)
        self.E = 1.0
        self.M_history = []
        self.E_history = []
        self.dE_history = []

        perturbations = dict(perturbation_schedule or [])

        for t in range(n_steps):
            perturb = perturbations.get(t, 0.0)
            self.step(external_perturbation=perturb)

        return self.analyze_feedback()

    def analyze_feedback(self) -> Dict:
        """
        Analyze the feedback relationship between M and dE.

        Returns:
            Dict with correlation, p_value, interpretation
        """
        M = np.array(self.M_history)
        dE = np.array(self.dE_history)

        if len(M) < 10 or len(dE) < 10:
            return {
                'correlation': np.nan,
                'p_value': 1.0,
                'is_negative_feedback': False,
                'error': 'Insufficient data'
            }

        # Align arrays (M at time t, dE computed at time t)
        n = min(len(M), len(dE))
        M = M[:n]
        dE = dE[:n]

        # Compute Pearson correlation
        corr, p_value = pearsonr(M, dE)

        # Also compute Spearman (rank correlation) for robustness
        spearman_corr, spearman_p = spearmanr(M, dE)

        return {
            'pearson_correlation': corr,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'is_negative_feedback': corr < -0.3 and p_value < 0.05,
            'M_history': M.tolist(),
            'dE_history': dE.tolist(),
            'interpretation': 'NEGATIVE_FEEDBACK' if corr < -0.3 else 'INSUFFICIENT'
        }


class ProportionalFeedbackTest:
    """
    Test that feedback is proportional to deviation, not just threshold-based.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def measure_response_curve(self, M_values: np.ndarray,
                                feedback_fn) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measure dE response across range of M values.

        Returns:
            (M_values, dE_values) arrays
        """
        dE_values = []
        for M in M_values:
            # Sample multiple times and average
            samples = [feedback_fn(M) for _ in range(10)]
            dE_values.append(np.mean(samples))
        return M_values, np.array(dE_values)

    def test_proportionality(self, M_values: np.ndarray,
                             dE_values: np.ndarray) -> Dict:
        """
        Test if dE is proportional to deviation from M*.

        Proportional: dE = k(M* - M) means dE vs M should be linear with negative slope.
        """
        # Fit linear model: dE = a*M + b
        coeffs = np.polyfit(M_values, dE_values, 1)
        slope, intercept = coeffs

        # Compute R² for linear fit
        dE_pred = slope * M_values + intercept
        ss_res = np.sum((dE_values - dE_pred) ** 2)
        ss_tot = np.sum((dE_values - np.mean(dE_values)) ** 2)
        R_squared = 1 - ss_res / (ss_tot + EPS)

        return {
            'slope': slope,
            'intercept': intercept,
            'R_squared': R_squared,
            'is_proportional': R_squared > 0.8 and slope < 0,
            'interpretation': 'LINEAR_PROPORTIONAL' if R_squared > 0.8 else 'NONLINEAR'
        }


# =============================================================================
# Tests
# =============================================================================

class TestNegativeFeedback:
    """Test suite for negative feedback quantification."""

    @pytest.fixture
    def system(self):
        return FeedbackSystem(M_star=0.5, seed=42)

    def test_correlation_is_negative(self, system):
        """
        Test that corr(M, dE/dt) < -0.3.
        """
        # Run with perturbations to create variance
        perturbations = [(20, -0.5), (50, 0.5), (80, -0.3), (120, 0.4)]
        result = system.run_episode(n_steps=200, perturbation_schedule=perturbations)

        corr = result['pearson_correlation']
        p_value = result['pearson_p_value']

        assert corr < -0.3, (
            f"Correlation ({corr:.3f}) should be < -0.3 for negative feedback"
        )

        assert p_value < 0.05, (
            f"p-value ({p_value:.4f}) should be < 0.05 for significance"
        )

        print(f"\n✓ Negative feedback confirmed:")
        print(f"  Pearson r = {corr:.3f}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  Spearman ρ = {result['spearman_correlation']:.3f}")

    def test_feedback_proportional_to_deviation(self, system):
        """
        Test that larger deviations produce stronger feedback responses.
        """
        # Create scenarios with different deviation magnitudes
        deviations = [0.1, 0.3, 0.5, 0.8, 1.0]
        response_magnitudes = []

        for dev in deviations:
            # Set M to M* - dev (below equilibrium)
            M_test = system.M_star - dev
            dE = system.compute_evidence_response(M_test)
            response_magnitudes.append(dE)

        # Larger deviations should produce larger responses
        corr, p_value = pearsonr(deviations, response_magnitudes)

        assert corr > 0.8, (
            f"Response should increase with deviation magnitude (r = {corr:.3f})"
        )

        print(f"\n✓ Proportional response confirmed:")
        print(f"  Correlation(deviation, response) = {corr:.3f}")
        print(f"  Deviations: {deviations}")
        print(f"  Responses: {[f'{r:.3f}' for r in response_magnitudes]}")

    def test_bidirectional_feedback(self, system):
        """
        Test that feedback works in both directions:
        - M < M* → dE > 0 (gather evidence)
        - M > M* → dE < 0 (relax evidence gathering)
        """
        # Below equilibrium
        M_low = system.M_star - 0.5
        dE_low = system.compute_evidence_response(M_low)

        # Above equilibrium
        M_high = system.M_star + 0.5
        dE_high = system.compute_evidence_response(M_high)

        # At equilibrium
        M_eq = system.M_star
        dE_eq = system.compute_evidence_response(M_eq)

        assert dE_low > dE_eq, (
            f"Below equilibrium should trigger higher dE: {dE_low:.3f} vs {dE_eq:.3f}"
        )

        assert dE_high < dE_eq, (
            f"Above equilibrium should trigger lower dE: {dE_high:.3f} vs {dE_eq:.3f}"
        )

        print(f"\n✓ Bidirectional feedback confirmed:")
        print(f"  M = {M_low:.2f} (below): dE = {dE_low:.3f}")
        print(f"  M = {M_eq:.2f} (equilibrium): dE = {dE_eq:.3f}")
        print(f"  M = {M_high:.2f} (above): dE = {dE_high:.3f}")

    def test_response_is_linear(self):
        """
        Test that dE response is linear in M (proportional control).
        """
        system = FeedbackSystem(M_star=0.5, seed=42)
        tester = ProportionalFeedbackTest(seed=42)

        M_values = np.linspace(-0.5, 1.5, 20)
        M_values, dE_values = tester.measure_response_curve(
            M_values,
            system.compute_evidence_response
        )

        result = tester.test_proportionality(M_values, dE_values)

        assert result['is_proportional'], (
            f"Response should be linear (R² = {result['R_squared']:.3f}, "
            f"slope = {result['slope']:.3f})"
        )

        print(f"\n✓ Linear response confirmed:")
        print(f"  Slope = {result['slope']:.3f}")
        print(f"  R² = {result['R_squared']:.3f}")


# =============================================================================
# Comprehensive Test Runner
# =============================================================================

def run_comprehensive_test(seed: int = 42) -> dict:
    """
    Run comprehensive negative feedback analysis.
    """
    results = {
        'test_name': 'Q39_NEGATIVE_FEEDBACK',
        'seed': seed,
        'correlation_tests': [],
        'proportionality_test': {},
        'summary': {}
    }

    # Run multiple episodes to get robust statistics
    correlations = []
    for i in range(10):
        system = FeedbackSystem(M_star=0.5, seed=seed + i)
        perturbations = [
            (20 + i*5, -0.5),
            (50 + i*5, 0.5),
            (80 + i*5, -0.3),
            (120 + i*5, 0.4)
        ]
        result = system.run_episode(n_steps=200, perturbation_schedule=perturbations)

        correlations.append(result['pearson_correlation'])
        results['correlation_tests'].append({
            'episode': i,
            'correlation': result['pearson_correlation'],
            'p_value': result['pearson_p_value'],
            'is_negative_feedback': result['is_negative_feedback']
        })

    # Proportionality test
    system = FeedbackSystem(M_star=0.5, seed=seed)
    tester = ProportionalFeedbackTest(seed=seed)
    M_values = np.linspace(-0.5, 1.5, 20)
    M_values, dE_values = tester.measure_response_curve(
        M_values,
        system.compute_evidence_response
    )
    prop_result = tester.test_proportionality(M_values, dE_values)
    results['proportionality_test'] = prop_result

    # Summary
    mean_corr = np.mean(correlations)
    n_negative = sum(1 for c in correlations if c < -0.3)

    results['summary'] = {
        'mean_correlation': float(mean_corr),
        'std_correlation': float(np.std(correlations)),
        'n_negative_feedback': n_negative,
        'n_total': len(correlations),
        'proportionality_R_squared': prop_result['R_squared'],
        'proportionality_slope': prop_result['slope'],
        'PASS': mean_corr < -0.3 and n_negative >= 8 and prop_result['is_proportional']
    }

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Q39 Test 3: Negative Feedback Quantification")
    print("=" * 60)

    results = run_comprehensive_test()

    print(f"\nCorrelation Analysis:")
    print(f"  Mean r = {results['summary']['mean_correlation']:.3f}")
    print(f"  Std r = {results['summary']['std_correlation']:.3f}")
    print(f"  Negative feedback episodes: {results['summary']['n_negative_feedback']}/{results['summary']['n_total']}")

    print(f"\nProportionality Analysis:")
    print(f"  Slope = {results['summary']['proportionality_slope']:.3f}")
    print(f"  R² = {results['summary']['proportionality_R_squared']:.3f}")

    print(f"\n  PASS: {results['summary']['PASS']}")

    # Save results
    output_path = Path(__file__).parent / 'q39_test3_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
