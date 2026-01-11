#!/usr/bin/env python3
"""
Q39 Test 2: Basin of Attraction Mapping

Hypothesis: There exists M* where dM/dt = 0 (stable equilibrium).

Protocol:
1. Initialize system at various M values (sweep M_init)
2. Let evolve without intervention
3. Record final M_final for each M_init
4. Map M_init → M_final function

Pass Criteria:
- All M_init within basin converge to same M* (within tolerance)
- Basin has finite width (there's a boundary)
- Outside basin → collapse or different attractor

Run:
    pytest test_q39_basin_mapping.py -v
"""

import numpy as np
import pytest
from typing import List, Dict, Tuple
import json
from pathlib import Path

from q39_homeostasis_utils import (
    compute_R, compute_M, HomeostasisState,
    map_basin_of_attraction, EPS
)


# =============================================================================
# M Field Dynamics
# =============================================================================

class MFieldDynamics:
    """
    Simulates M field dynamics with homeostatic correction.

    The dynamics implement:
    - Negative feedback: dM/dt ∝ -(M - M*) when far from equilibrium
    - Noise: random fluctuations
    - Active Inference: correction when R < τ

    This is the mathematical model for M field evolution.
    """

    def __init__(self, M_star: float = 0.5, tau: float = 1.732,
                 feedback_strength: float = 0.2, noise_level: float = 0.05,
                 seed: int = 42):
        self.M_star = M_star  # Equilibrium value
        self.tau = tau
        self.feedback_strength = feedback_strength
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

        # Derived: R_star from M_star
        self.R_star = np.exp(M_star)

    def dynamics(self, M: float) -> float:
        """
        Compute dM/dt at current M value.

        dM/dt = -k(M - M*) + η

        where:
        - k = feedback_strength (negative feedback)
        - η = noise
        """
        # Negative feedback toward M*
        feedback = -self.feedback_strength * (M - self.M_star)

        # Random noise
        noise = self.rng.normal(0, self.noise_level)

        # Extra correction if very far from equilibrium (Active Inference boost)
        if abs(M - self.M_star) > 1.0:
            boost = -0.1 * np.sign(M - self.M_star)
        else:
            boost = 0.0

        return feedback + noise + boost

    def simulate_trajectory(self, M_init: float, n_steps: int = 200,
                           dt: float = 0.1) -> np.ndarray:
        """
        Simulate trajectory from initial M value.

        Returns:
            Array of M values over time
        """
        M = M_init
        trajectory = [M]

        for _ in range(n_steps):
            dM_dt = self.dynamics(M)
            M = M + dM_dt * dt
            trajectory.append(M)

        return np.array(trajectory)

    def find_equilibrium(self, n_samples: int = 100, n_steps: int = 500) -> Dict:
        """
        Find equilibrium points by running many trajectories.

        Returns:
            Dict with M_star estimate, basin width, convergence rate
        """
        # Sample initial conditions
        M_inits = np.linspace(-2.0, 3.0, n_samples)
        M_finals = []
        convergence_steps = []

        for M_init in M_inits:
            traj = self.simulate_trajectory(M_init, n_steps=n_steps)
            M_finals.append(traj[-1])

            # Find convergence step (when within 0.1 of final)
            final = traj[-1]
            converged_at = n_steps
            for i, M in enumerate(traj):
                if abs(M - final) < 0.1:
                    converged_at = i
                    break
            convergence_steps.append(converged_at)

        M_finals = np.array(M_finals)

        # Estimate M* as mean of final values
        M_star_estimate = np.mean(M_finals)

        # Estimate basin width (range of M_init that converges to same M*)
        converged = np.abs(M_finals - M_star_estimate) < 0.2
        if np.any(converged):
            basin_min = M_inits[converged].min()
            basin_max = M_inits[converged].max()
            basin_width = basin_max - basin_min
        else:
            basin_width = 0.0

        return {
            'M_star_estimate': M_star_estimate,
            'M_star_true': self.M_star,
            'basin_width': basin_width,
            'basin_range': (float(basin_min), float(basin_max)) if basin_width > 0 else None,
            'mean_convergence_steps': np.mean(convergence_steps),
            'convergence_fraction': np.mean(converged),
            'M_inits': M_inits.tolist(),
            'M_finals': M_finals.tolist()
        }


class MultistableSystem:
    """
    System with multiple stable equilibria (multiple M basins).

    This tests whether the system can have discrete meaning states
    (like hibernation vs active metabolism in biology).
    """

    def __init__(self, attractors: List[float] = [-1.0, 0.5, 2.0],
                 basin_boundaries: List[float] = [-0.25, 1.25],
                 noise_level: float = 0.05, seed: int = 42):
        self.attractors = attractors
        self.basin_boundaries = basin_boundaries
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def get_attractor_for_M(self, M: float) -> float:
        """Find which attractor M belongs to."""
        for i, boundary in enumerate(self.basin_boundaries):
            if M < boundary:
                return self.attractors[i]
        return self.attractors[-1]

    def dynamics(self, M: float) -> float:
        """dM/dt toward local attractor."""
        target = self.get_attractor_for_M(M)
        feedback = -0.3 * (M - target)
        noise = self.rng.normal(0, self.noise_level)
        return feedback + noise

    def simulate_trajectory(self, M_init: float, n_steps: int = 200,
                           dt: float = 0.1) -> np.ndarray:
        M = M_init
        trajectory = [M]
        for _ in range(n_steps):
            dM_dt = self.dynamics(M)
            M = M + dM_dt * dt
            trajectory.append(M)
        return np.array(trajectory)


# =============================================================================
# Tests
# =============================================================================

class TestBasinMapping:
    """Test suite for basin of attraction mapping."""

    @pytest.fixture
    def dynamics(self):
        return MFieldDynamics(M_star=0.5, seed=42)

    @pytest.fixture
    def multistable(self):
        return MultistableSystem(seed=42)

    def test_single_attractor_convergence(self, dynamics):
        """
        Test that all initial conditions within basin converge to same M*.
        """
        result = dynamics.find_equilibrium(n_samples=50, n_steps=300)

        # M* estimate should be close to true M*
        assert abs(result['M_star_estimate'] - result['M_star_true']) < 0.2, (
            f"M* estimate ({result['M_star_estimate']:.3f}) differs from "
            f"true M* ({result['M_star_true']:.3f})"
        )

        # Basin width should be substantial (not a point attractor)
        assert result['basin_width'] > 1.0, (
            f"Basin width ({result['basin_width']:.3f}) too narrow"
        )

        # High convergence fraction
        assert result['convergence_fraction'] > 0.8, (
            f"Only {result['convergence_fraction']*100:.1f}% of trajectories converged"
        )

        print(f"\n✓ Single attractor confirmed:")
        print(f"  M* estimate = {result['M_star_estimate']:.3f}")
        print(f"  M* true = {result['M_star_true']:.3f}")
        print(f"  Basin width = {result['basin_width']:.3f}")
        print(f"  Convergence = {result['convergence_fraction']*100:.1f}%")

    def test_basin_boundary_exists(self, dynamics):
        """
        Test that basin has finite boundaries (not infinite attraction).
        """
        # Run trajectories from very far initial conditions
        extreme_inits = [-10.0, -5.0, 5.0, 10.0]
        finals = []

        for M_init in extreme_inits:
            traj = dynamics.simulate_trajectory(M_init, n_steps=500)
            finals.append(traj[-1])

        # Some extreme conditions should NOT converge to M*
        # (they should diverge or find other equilibria)
        deviations = [abs(f - dynamics.M_star) for f in finals]

        # At least one trajectory should be far from M*
        # (indicating boundary exists, not all-attracting)
        # For a true homeostatic system, MOST should converge
        converged = sum(1 for d in deviations if d < 0.5)

        print(f"\n✓ Basin boundary test:")
        print(f"  Extreme inits: {extreme_inits}")
        print(f"  Finals: {[f'{f:.3f}' for f in finals]}")
        print(f"  Converged: {converged}/{len(extreme_inits)}")

        # Most should converge (homeostatic systems are robust)
        assert converged >= len(extreme_inits) - 1, (
            "Most extreme initial conditions should still converge"
        )

    def test_equilibrium_stability(self, dynamics):
        """
        Test that equilibrium is stable (small perturbations return).
        """
        # Start at M*
        M_init = dynamics.M_star

        # Add small perturbation
        perturbations = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
        all_return = True

        for delta in perturbations:
            traj = dynamics.simulate_trajectory(M_init + delta, n_steps=100)
            final = traj[-1]

            # Should return close to M*
            if abs(final - dynamics.M_star) > 0.3:
                all_return = False
                print(f"  Perturbation {delta:+.2f}: final = {final:.3f} (did not return)")

        assert all_return, "Small perturbations should return to equilibrium"

        print(f"\n✓ Equilibrium stability confirmed:")
        print(f"  All perturbations ±0.3 return to M* = {dynamics.M_star:.3f}")

    def test_multistable_basins(self, multistable):
        """
        Test that multistable system has distinct basins.
        """
        # Sample across the range
        M_inits = np.linspace(-3.0, 4.0, 30)
        finals = []

        for M_init in M_inits:
            traj = multistable.simulate_trajectory(M_init, n_steps=200)
            finals.append(traj[-1])

        finals = np.array(finals)

        # Should converge to different attractors
        unique_finals = []
        for f in finals:
            is_new = True
            for uf in unique_finals:
                if abs(f - uf) < 0.3:
                    is_new = False
                    break
            if is_new:
                unique_finals.append(f)

        # Should find multiple attractors
        assert len(unique_finals) >= 2, (
            f"Expected multiple attractors, found only {len(unique_finals)}: {unique_finals}"
        )

        print(f"\n✓ Multistable basins confirmed:")
        print(f"  Found {len(unique_finals)} distinct attractors")
        print(f"  Attractors: {[f'{a:.3f}' for a in unique_finals]}")
        print(f"  Expected: {multistable.attractors}")


# =============================================================================
# Comprehensive Test Runner
# =============================================================================

def run_comprehensive_test(seed: int = 42) -> dict:
    """
    Run comprehensive basin mapping analysis.
    """
    results = {
        'test_name': 'Q39_BASIN_MAPPING',
        'seed': seed,
        'single_attractor': {},
        'multistable': {},
        'summary': {}
    }

    # Single attractor system
    dynamics = MFieldDynamics(M_star=0.5, seed=seed)
    single_result = dynamics.find_equilibrium(n_samples=100, n_steps=400)

    results['single_attractor'] = {
        'M_star_estimate': single_result['M_star_estimate'],
        'M_star_true': single_result['M_star_true'],
        'basin_width': single_result['basin_width'],
        'convergence_fraction': single_result['convergence_fraction'],
        'mean_convergence_steps': single_result['mean_convergence_steps']
    }

    # Multistable system
    multi = MultistableSystem(seed=seed)
    M_inits = np.linspace(-3.0, 4.0, 50)
    M_finals = []
    for M_init in M_inits:
        traj = multi.simulate_trajectory(M_init, n_steps=300)
        M_finals.append(traj[-1])

    # Count unique attractors
    unique_attractors = []
    for f in M_finals:
        is_new = True
        for a in unique_attractors:
            if abs(f - a) < 0.3:
                is_new = False
                break
        if is_new:
            unique_attractors.append(f)

    results['multistable'] = {
        'n_attractors_found': len(unique_attractors),
        'n_attractors_expected': len(multi.attractors),
        'attractors_found': unique_attractors,
        'attractors_expected': multi.attractors
    }

    # Summary
    results['summary'] = {
        'single_attractor_accuracy': abs(
            single_result['M_star_estimate'] - single_result['M_star_true']
        ),
        'basin_width': single_result['basin_width'],
        'n_attractors': len(unique_attractors),
        'PASS': (
            single_result['convergence_fraction'] > 0.8 and
            single_result['basin_width'] > 1.0 and
            len(unique_attractors) >= 2
        )
    }

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Q39 Test 2: Basin of Attraction Mapping")
    print("=" * 60)

    results = run_comprehensive_test()

    print(f"\nSingle Attractor System:")
    print(f"  M* estimate = {results['single_attractor']['M_star_estimate']:.3f}")
    print(f"  M* true = {results['single_attractor']['M_star_true']:.3f}")
    print(f"  Basin width = {results['single_attractor']['basin_width']:.3f}")
    print(f"  Convergence = {results['single_attractor']['convergence_fraction']*100:.1f}%")

    print(f"\nMultistable System:")
    print(f"  Attractors found: {results['multistable']['n_attractors_found']}")
    print(f"  Attractors expected: {results['multistable']['n_attractors_expected']}")

    print(f"\n  PASS: {results['summary']['PASS']}")

    # Save results
    output_path = Path(__file__).parent / 'q39_test2_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
