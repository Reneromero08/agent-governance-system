#!/usr/bin/env python3
"""
Q39 Test 4: Catastrophic Failure Boundary

Hypothesis: Beyond basin boundary, system cannot recover.

Protocol:
1. Find perturbation magnitude that causes collapse
2. Map the boundary precisely using sigmoid fitting
3. Show recovery impossible beyond threshold

Pass Criteria:
- Sharp boundary exists (not gradual degradation)
- Sigmoid fit k > 2.0 (steepness parameter indicates phase transition)
- Connects to Q12 phase transition evidence

Run:
    pytest test_q39_catastrophic_boundary.py -v
"""

import numpy as np
import pytest
from typing import List, Dict, Tuple
from scipy.optimize import curve_fit
import json
from pathlib import Path

from q39_homeostasis_utils import (
    compute_R, compute_M, HomeostasisState,
    find_catastrophic_boundary, EPS
)


# =============================================================================
# System with Catastrophic Failure Mode
# =============================================================================

class CatastrophicSystem:
    """
    Homeostatic system that can fail catastrophically.

    Key insight: Beyond a critical perturbation, the system cannot
    recover and falls into a collapsed state (meaning death).

    This connects to Q12 (phase transitions): the boundary between
    recovery and collapse is a phase transition, not a smooth gradient.
    """

    def __init__(self, M_star: float = 0.5, collapse_threshold: float = -1.5,
                 recovery_strength: float = 0.3, seed: int = 42):
        self.M_star = M_star
        self.collapse_threshold = collapse_threshold
        self.recovery_strength = recovery_strength
        self.rng = np.random.default_rng(seed)

        # State
        self.M = M_star
        self.collapsed = False

    def reset(self):
        """Reset to equilibrium state."""
        self.M = self.M_star
        self.collapsed = False

    def step(self, dt: float = 0.1):
        """
        Single time step with potential collapse.

        If M drops below collapse threshold, system enters irreversible
        collapsed state (no recovery possible).
        """
        if self.collapsed:
            # Once collapsed, M decays toward -infinity
            self.M = self.M - 0.1
            return

        # Check for collapse
        if self.M < self.collapse_threshold:
            self.collapsed = True
            return

        # Normal homeostatic dynamics
        # Recovery toward M*
        dM = self.recovery_strength * (self.M_star - self.M)

        # Add noise
        dM += self.rng.normal(0, 0.05)

        self.M = self.M + dM * dt

    def apply_perturbation(self, magnitude: float):
        """Apply sudden perturbation to M."""
        self.M = self.M - magnitude

    def run_recovery_test(self, perturbation: float,
                          recovery_steps: int = 100) -> Dict:
        """
        Test if system can recover from perturbation.

        Returns:
            Dict with recovery success, final M, trajectory
        """
        self.reset()

        # Apply perturbation
        self.apply_perturbation(perturbation)
        M_after_perturb = self.M

        # Let system evolve
        trajectory = [self.M]
        for _ in range(recovery_steps):
            self.step()
            trajectory.append(self.M)

        # Check recovery
        M_final = self.M
        recovered = (
            not self.collapsed and
            abs(M_final - self.M_star) < 0.3
        )

        return {
            'perturbation': perturbation,
            'M_after_perturbation': M_after_perturb,
            'M_final': M_final,
            'recovered': recovered,
            'collapsed': self.collapsed,
            'trajectory': trajectory
        }


# =============================================================================
# Sigmoid Fitting for Phase Transition Detection
# =============================================================================

def sigmoid(x, x0, k):
    """
    Sigmoid function for phase transition fitting.

    P(recovery) = 1 / (1 + exp(k * (x - x0)))

    Args:
        x: Perturbation magnitude
        x0: Critical point (boundary)
        k: Steepness (higher = sharper transition)

    Returns:
        Recovery probability
    """
    return 1 / (1 + np.exp(k * (x - x0)))


def compute_sigmoid_sharpness(k: float, x_range: float) -> float:
    """
    Compute sharpness from sigmoid steepness parameter.

    For a true phase transition, k should be large (sharp transition).
    For gradual degradation, k is small.

    Sharpness scale:
    - k < 1: Gradual (sharpness < 0.3)
    - k = 2: Moderate (sharpness ~ 0.5)
    - k > 4: Sharp (sharpness > 0.7)
    - k > 8: Phase transition (sharpness > 0.9)

    We normalize by the x_range to get a scale-independent measure.
    """
    # Transition width = range where P goes from 0.9 to 0.1
    # For sigmoid: width = 2 * ln(9) / k ~ 4.4 / k
    transition_width = 4.4 / (k + EPS)

    # Sharpness = 1 - (transition_width / x_range), clamped to [0, 1]
    sharpness = 1 - (transition_width / x_range)
    return np.clip(sharpness, 0, 1)


class BoundaryMapper:
    """
    Maps the catastrophic boundary with high precision using sigmoid fitting.
    """

    def __init__(self, system: CatastrophicSystem):
        self.system = system

    def find_boundary(self, mag_range: Tuple[float, float],
                      n_samples: int = 40,
                      n_trials: int = 10) -> Dict:
        """
        Find the boundary between recovery and collapse using sigmoid fitting.

        Uses multiple trials per magnitude for robust statistics,
        then fits a sigmoid to determine the critical point and sharpness.

        Returns:
            Dict with boundary value, sharpness (from sigmoid), recovery curve
        """
        magnitudes = np.linspace(mag_range[0], mag_range[1], n_samples)
        recovery_rates = []

        for mag in magnitudes:
            # Run multiple trials for statistical robustness
            successes = 0
            for trial in range(n_trials):
                self.system.rng = np.random.default_rng(42 + trial * 100)
                result = self.system.run_recovery_test(mag)
                if result['recovered']:
                    successes += 1

            recovery_rates.append(successes / n_trials)

        recovery_rates = np.array(recovery_rates)

        # Fit sigmoid to recovery curve
        try:
            # Initial guess: boundary at midpoint, moderate steepness
            p0 = [(mag_range[0] + mag_range[1]) / 2, 2.0]

            popt, pcov = curve_fit(
                sigmoid,
                magnitudes,
                recovery_rates,
                p0=p0,
                bounds=(
                    [mag_range[0], 0.1],  # Lower bounds: x0 in range, k > 0
                    [mag_range[1], 20.0]  # Upper bounds
                ),
                maxfev=5000
            )

            boundary = popt[0]
            k = popt[1]

            # Compute R^2 for sigmoid fit
            y_pred = sigmoid(magnitudes, boundary, k)
            ss_res = np.sum((recovery_rates - y_pred) ** 2)
            ss_tot = np.sum((recovery_rates - np.mean(recovery_rates)) ** 2)
            r_squared = 1 - ss_res / (ss_tot + EPS)

            # Compute sharpness from sigmoid steepness
            x_range = mag_range[1] - mag_range[0]
            sharpness = compute_sigmoid_sharpness(k, x_range)

            fit_successful = True

        except Exception as e:
            # Fallback to simple boundary detection
            boundary = None
            k = 0
            r_squared = 0
            sharpness = 0
            fit_successful = False

            # Find simple crossing point
            for i in range(len(recovery_rates) - 1):
                if recovery_rates[i] > 0.5 and recovery_rates[i+1] <= 0.5:
                    boundary = magnitudes[i] + (magnitudes[i+1] - magnitudes[i]) * (
                        (recovery_rates[i] - 0.5) / (recovery_rates[i] - recovery_rates[i+1] + EPS)
                    )
                    break

        # Determine transition type
        if sharpness > 0.7:
            transition_type = 'PHASE_TRANSITION'
        elif sharpness > 0.5:
            transition_type = 'SHARP_TRANSITION'
        else:
            transition_type = 'GRADUAL'

        return {
            'boundary': boundary,
            'sigmoid_k': k,
            'sigmoid_r_squared': r_squared,
            'sharpness': sharpness,
            'magnitudes': magnitudes.tolist(),
            'recovery_rates': recovery_rates.tolist(),
            'is_sharp': sharpness > 0.5,
            'is_phase_transition': sharpness > 0.7,
            'transition_type': transition_type,
            'fit_successful': fit_successful
        }

    def refine_boundary(self, rough_boundary: float,
                        precision: float = 0.01) -> float:
        """
        Binary search to refine boundary estimate.
        """
        low = rough_boundary - 0.5
        high = rough_boundary + 0.5

        while high - low > precision:
            mid = (low + high) / 2

            # Test at midpoint
            self.system.rng = np.random.default_rng(42)
            result = self.system.run_recovery_test(mid)

            if result['recovered']:
                low = mid
            else:
                high = mid

        return (low + high) / 2


# =============================================================================
# Tests
# =============================================================================

class TestCatastrophicBoundary:
    """Test suite for catastrophic failure boundary."""

    @pytest.fixture
    def system(self):
        return CatastrophicSystem(
            M_star=0.5,
            collapse_threshold=-1.5,
            seed=42
        )

    @pytest.fixture
    def mapper(self, system):
        return BoundaryMapper(system)

    def test_boundary_exists(self, mapper):
        """
        Test that a clear boundary between recovery and collapse exists.
        """
        result = mapper.find_boundary(mag_range=(0.5, 3.0), n_samples=40, n_trials=10)

        assert result['boundary'] is not None, (
            "No boundary found - system may always recover or always collapse"
        )

        assert result['boundary'] > 0.5, (
            f"Boundary too low ({result['boundary']:.3f}) - system too fragile"
        )

        assert result['boundary'] < 3.0, (
            f"Boundary too high ({result['boundary']:.3f}) - no collapse detected"
        )

        print(f"\n[+] Catastrophic boundary found:")
        print(f"  Boundary = {result['boundary']:.3f}")
        print(f"  Collapse threshold M = {-1.5:.3f}")
        print(f"  Expected boundary ~ {0.5 - (-1.5):.3f} = 2.0")

    def test_boundary_is_sharp(self, mapper):
        """
        Test that the boundary is a sharp phase transition, not gradual degradation.
        """
        result = mapper.find_boundary(mag_range=(0.5, 3.0), n_samples=40, n_trials=10)

        assert result['sharpness'] > 0.5, (
            f"Boundary not sharp enough (sharpness = {result['sharpness']:.3f}). "
            f"Expected phase transition, got gradual degradation."
        )

        print(f"\n[+] Sharp phase transition confirmed:")
        print(f"  Sharpness = {result['sharpness']:.3f}")
        print(f"  Sigmoid k = {result['sigmoid_k']:.3f}")
        print(f"  Sigmoid R^2 = {result['sigmoid_r_squared']:.3f}")
        print(f"  Transition type: {result['transition_type']}")

    def test_below_boundary_recovers(self, system):
        """
        Test that perturbations below boundary lead to recovery.
        """
        # Use a perturbation well below expected boundary (~2.0)
        safe_perturbation = 1.0

        result = system.run_recovery_test(safe_perturbation, recovery_steps=150)

        assert result['recovered'], (
            f"System failed to recover from safe perturbation ({safe_perturbation}). "
            f"Final M = {result['M_final']:.3f}"
        )

        assert not result['collapsed'], (
            f"System collapsed from safe perturbation ({safe_perturbation})"
        )

        print(f"\n[+] Below-boundary recovery confirmed:")
        print(f"  Perturbation = {safe_perturbation}")
        print(f"  M after = {result['M_after_perturbation']:.3f}")
        print(f"  M final = {result['M_final']:.3f}")
        print(f"  Recovered: {result['recovered']}")

    def test_above_boundary_collapses(self, system):
        """
        Test that perturbations above boundary lead to collapse.
        """
        # Use a perturbation well above expected boundary (~2.0)
        dangerous_perturbation = 2.5

        result = system.run_recovery_test(dangerous_perturbation, recovery_steps=150)

        assert result['collapsed'], (
            f"System did not collapse from dangerous perturbation ({dangerous_perturbation}). "
            f"Final M = {result['M_final']:.3f}"
        )

        print(f"\n[+] Above-boundary collapse confirmed:")
        print(f"  Perturbation = {dangerous_perturbation}")
        print(f"  M after = {result['M_after_perturbation']:.3f}")
        print(f"  Collapsed: {result['collapsed']}")

    def test_collapse_is_irreversible(self, system):
        """
        Test that once collapsed, the system cannot recover.
        """
        # Trigger collapse
        system.reset()
        system.M = system.collapse_threshold - 0.1  # Just below threshold
        system.step()  # Should trigger collapse

        assert system.collapsed, "System should have collapsed"

        # Try to "help" the system
        initial_M = system.M
        for _ in range(100):
            system.step()

        # M should continue decreasing (irreversible)
        assert system.M < initial_M, (
            "Collapsed system should not recover"
        )

        print(f"\n[+] Collapse irreversibility confirmed:")
        print(f"  M at collapse = {initial_M:.3f}")
        print(f"  M after 100 steps = {system.M:.3f}")


# =============================================================================
# Comprehensive Test Runner
# =============================================================================

def run_comprehensive_test(seed: int = 42) -> dict:
    """
    Run comprehensive catastrophic boundary analysis with sigmoid fitting.
    """
    results = {
        'test_name': 'Q39_CATASTROPHIC_BOUNDARY',
        'seed': seed,
        'boundary_search': {},
        'recovery_tests': [],
        'summary': {}
    }

    # Create system
    system = CatastrophicSystem(
        M_star=0.5,
        collapse_threshold=-1.5,
        seed=seed
    )
    mapper = BoundaryMapper(system)

    # Find boundary with improved sigmoid fitting
    boundary_result = mapper.find_boundary(
        mag_range=(0.5, 3.5),
        n_samples=50,  # More samples for better fit
        n_trials=15    # More trials for robustness
    )
    results['boundary_search'] = boundary_result

    # Test specific perturbations
    test_magnitudes = [0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
    for mag in test_magnitudes:
        system.rng = np.random.default_rng(seed)
        result = system.run_recovery_test(mag, recovery_steps=150)
        results['recovery_tests'].append({
            'magnitude': mag,
            'recovered': result['recovered'],
            'collapsed': result['collapsed'],
            'M_final': result['M_final']
        })

    # Summary
    n_recovered = sum(1 for r in results['recovery_tests'] if r['recovered'])
    n_collapsed = sum(1 for r in results['recovery_tests'] if r['collapsed'])

    results['summary'] = {
        'boundary': boundary_result['boundary'],
        'sharpness': boundary_result['sharpness'],
        'sigmoid_k': boundary_result['sigmoid_k'],
        'sigmoid_r_squared': boundary_result['sigmoid_r_squared'],
        'n_recovered': n_recovered,
        'n_collapsed': n_collapsed,
        'expected_boundary': 0.5 - (-1.5),  # M_star - collapse_threshold
        'PASS': (
            boundary_result['boundary'] is not None and
            boundary_result['sharpness'] > 0.5 and
            boundary_result['sigmoid_r_squared'] > 0.8 and
            n_recovered > 0 and
            n_collapsed > 0
        )
    }

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Q39 Test 4: Catastrophic Failure Boundary (Sigmoid Fitting)")
    print("=" * 60)

    results = run_comprehensive_test()

    print(f"\nBoundary Analysis:")
    print(f"  Boundary = {results['summary']['boundary']:.3f}")
    print(f"  Expected = {results['summary']['expected_boundary']:.3f}")
    print(f"  Sharpness = {results['summary']['sharpness']:.3f}")
    print(f"  Sigmoid k = {results['summary']['sigmoid_k']:.3f}")
    print(f"  Sigmoid R^2 = {results['summary']['sigmoid_r_squared']:.3f}")

    print(f"\nRecovery Tests:")
    for test in results['recovery_tests']:
        if test['recovered']:
            status = "[+] recovered"
        elif test['collapsed']:
            status = "[-] collapsed"
        else:
            status = "[?] neither"
        print(f"  Magnitude {test['magnitude']:.1f}: {status}")

    print(f"\n  PASS: {results['summary']['PASS']}")

    # Save results
    output_path = Path(__file__).parent / 'q39_test4_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
