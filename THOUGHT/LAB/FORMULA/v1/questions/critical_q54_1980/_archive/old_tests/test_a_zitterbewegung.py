"""
Q54 Test A: Standing Wave vs Propagating Wave Inertia
=====================================================

HYPOTHESIS (stated BEFORE running):
Standing waves (p=0) should exhibit MORE inertia than propagating waves (p!=0)
because standing waves have rest mass E/c^2 while propagating waves do not.

PREDICTION:
- Standing waves respond SLOWER to perturbation (more inertia)
- Propagating waves respond FASTER (less inertia)
- Inertia ratio (standing/propagating) should be > 1

FALSIFICATION:
If standing and propagating waves respond identically,
the standing wave structure does NOT create rest mass behavior.

PHYSICS MODEL:
Uses the WAVE EQUATION (second-order), which has acceleration:
  d^2 psi/dt^2 = c^2 * d^2 psi/dx^2

NOT the advection equation (first-order), which cannot produce inertia.

ANTI-PATTERN CHECK:
- Do NOT assume mass to derive mass (wave equation has no mass term)
- Do NOT tune parameters to get desired result
- Report honestly even if it fails
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, Any
import os

# =============================================================================
# FIXED PARAMETERS (NO TUNING ALLOWED!)
# =============================================================================
N_POINTS = 1000         # Spatial resolution
C = 1.0                 # Speed of light
DT = 0.0005             # Time step (small for wave eq stability)
PERTURBATION = 0.01     # Small push (1% strength)
N_EQUILIBRATE = 2000    # Steps to establish dynamics
N_TRACK = 5000          # Steps to track response

# Domain
THETA = np.linspace(0, 2*np.pi, N_POINTS, endpoint=False)
DTHETA = 2 * np.pi / N_POINTS


def d2_dtheta2(psi: np.ndarray) -> np.ndarray:
    """Second spatial derivative with periodic boundary conditions."""
    return (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / (DTHETA**2)


def evolve_wave_equation(psi: np.ndarray, psi_prev: np.ndarray,
                         dt: float, c: float = 1.0) -> np.ndarray:
    """
    Evolve using the WAVE EQUATION (second-order in time).

    d^2 psi/dt^2 = c^2 * d^2 psi/dx^2

    Using Verlet integration:
    psi_new = 2*psi - psi_prev + dt^2 * c^2 * d2psi/dx2

    This equation HAS an acceleration mechanism and CAN produce inertia.
    """
    d2psi = d2_dtheta2(psi)
    psi_new = 2*psi - psi_prev + (c*dt)**2 * d2psi
    return psi_new


def create_standing_wave(k: int, width: float = 0.3) -> tuple:
    """
    Create a standing wave: cos(k*theta) with Gaussian envelope.

    Standing wave = e^(ikx) + e^(-ikx) = 2*cos(kx)
    Has NET MOMENTUM p = 0
    Has REST MASS = E/c^2
    """
    envelope = np.exp(-(THETA - np.pi)**2 / (2 * width**2))
    psi = envelope * np.cos(k * THETA)

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * DTHETA)
    psi = psi / norm

    # For wave equation, need psi at t and t-dt
    # Standing wave doesn't propagate, so psi_prev = psi
    psi_prev = psi.copy()

    return psi, psi_prev


def create_propagating_wave(k: int, width: float = 0.3) -> tuple:
    """
    Create a propagating wave: e^(ikx) with Gaussian envelope.

    Has MOMENTUM p = hbar * k != 0
    Has NO rest mass (like a photon)
    """
    envelope = np.exp(-(THETA - np.pi)**2 / (2 * width**2))
    psi = envelope * np.exp(1j * k * THETA)

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * DTHETA)
    psi = psi / norm

    # For propagating wave, phase evolves as e^(i*omega*t) where omega = c*k
    omega = C * k
    psi_prev = psi * np.exp(1j * omega * DT)
    psi_prev = psi_prev / np.sqrt(np.sum(np.abs(psi_prev)**2) * DTHETA)

    return psi, psi_prev


def compute_momentum(psi: np.ndarray) -> float:
    """
    Compute expectation value of momentum: <psi| -i d/dx |psi>
    """
    dpsi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2 * DTHETA)
    momentum = np.sum(np.conj(psi) * (-1j) * dpsi) * DTHETA
    return np.real(momentum)


def center_of_energy(psi: np.ndarray) -> float:
    """Compute x-component of center of energy."""
    prob = np.abs(psi)**2
    total = np.sum(prob)
    if total < 1e-10:
        return 0.0
    prob = prob / total
    x = np.cos(THETA)
    return np.sum(prob * x)


def apply_perturbation(psi: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply perturbation that pushes in +x direction.
    Same perturbation for both wave types (fair comparison).
    """
    phase_kick = strength * np.cos(THETA)
    return psi * np.exp(1j * phase_kick)


def find_response_time(trajectory: np.ndarray, initial_pos: float,
                       threshold: float = 0.1) -> int:
    """
    Find time to reach threshold fraction of maximum displacement.
    Higher response time = more inertia.
    """
    displacements = np.abs(trajectory - initial_pos)
    max_disp = np.max(displacements)

    if max_disp < 1e-10:
        return len(trajectory)

    target = threshold * max_disp
    indices = np.where(displacements >= target)[0]

    return indices[0] if len(indices) > 0 else len(trajectory)


def run_simulation(wave_type: str, k: int) -> Dict[str, Any]:
    """
    Run simulation for one wave type.

    Args:
        wave_type: 'standing' or 'propagating'
        k: wave number

    Returns:
        Dictionary with results
    """
    # Create wave
    if wave_type == 'standing':
        psi, psi_prev = create_standing_wave(k)
    else:
        psi, psi_prev = create_propagating_wave(k)

    # Check initial momentum
    initial_momentum = compute_momentum(psi)

    # Equilibrate (establish dynamics)
    for _ in range(N_EQUILIBRATE):
        psi_new = evolve_wave_equation(psi, psi_prev, DT, C)
        psi_prev = psi
        psi = psi_new

    # Record initial position
    x_init = center_of_energy(psi)

    # Apply perturbation
    psi = apply_perturbation(psi, PERTURBATION)

    # Track response
    trajectory = []
    for _ in range(N_TRACK):
        psi_new = evolve_wave_equation(psi, psi_prev, DT, C)
        psi_prev = psi
        psi = psi_new
        trajectory.append(center_of_energy(psi))

    trajectory = np.array(trajectory)

    # Analyze
    displacements = np.abs(trajectory - x_init)
    max_displacement = np.max(displacements)
    response_time = find_response_time(trajectory, x_init)

    return {
        'wave_type': wave_type,
        'k': k,
        'initial_momentum': float(initial_momentum),
        'initial_position': float(x_init),
        'max_displacement': float(max_displacement),
        'response_time_steps': int(response_time),
        'response_time_units': float(response_time * DT),
        'trajectory_sample': trajectory[:100].tolist(),
    }


def main():
    """Run all simulations and compile results."""
    print("=" * 70)
    print("Q54 TEST A: STANDING WAVE vs PROPAGATING WAVE INERTIA")
    print("=" * 70)
    print()

    print("PHYSICS MODEL: Wave Equation (second-order)")
    print("  d^2 psi/dt^2 = c^2 * d^2 psi/dx^2")
    print()
    print("KEY INSIGHT:")
    print("  Standing wave: p = 0, has REST MASS = E/c^2")
    print("  Propagating wave: p != 0, NO rest mass")
    print()

    print("PREDICTION (stated BEFORE running):")
    print("  Standing waves should respond SLOWER (more inertia)")
    print("  Inertia ratio (standing/propagating) > 1")
    print()

    print("PARAMETERS (FIXED - NO TUNING):")
    print(f"  N_POINTS = {N_POINTS}")
    print(f"  C = {C}")
    print(f"  DT = {DT}")
    print(f"  PERTURBATION = {PERTURBATION}")
    print()

    # Test multiple k values
    k_values = [1, 2, 3, 4, 5]

    results = {
        'test_name': 'Q54_Test_A_Standing_vs_Propagating',
        'timestamp': datetime.now().isoformat(),
        'physics_model': 'wave_equation_second_order',
        'parameters': {
            'n_points': N_POINTS,
            'c': C,
            'dt': DT,
            'perturbation': PERTURBATION,
            'n_equilibrate': N_EQUILIBRATE,
            'n_track': N_TRACK,
        },
        'k_values': k_values,
        'standing_results': [],
        'propagating_results': [],
        'ratios': [],
    }

    print("-" * 70)
    print(f"{'k':>4} {'Standing':>15} {'Propagating':>15} {'Ratio':>10} {'Result':>10}")
    print(f"{'':>4} {'(response time)':>15} {'(response time)':>15}")
    print("-" * 70)

    all_pass = True

    for k in k_values:
        # Run both simulations
        standing = run_simulation('standing', k)
        propagating = run_simulation('propagating', k)

        results['standing_results'].append(standing)
        results['propagating_results'].append(propagating)

        # Compute ratio
        if propagating['response_time_steps'] > 0:
            ratio = standing['response_time_steps'] / propagating['response_time_steps']
        else:
            ratio = float('inf')

        results['ratios'].append(ratio)

        # Check if standing has more inertia
        passed = ratio > 1.0
        if not passed:
            all_pass = False

        result_str = "PASS" if passed else "FAIL"
        print(f"{k:>4} {standing['response_time_steps']:>15} {propagating['response_time_steps']:>15} {ratio:>10.2f} {result_str:>10}")

    print("-" * 70)
    print()

    # Summary statistics
    avg_ratio = np.mean(results['ratios'])
    min_ratio = np.min(results['ratios'])
    max_ratio = np.max(results['ratios'])

    print("SUMMARY:")
    print(f"  Average inertia ratio: {avg_ratio:.2f}x")
    print(f"  Range: {min_ratio:.2f}x to {max_ratio:.2f}x")
    print()

    # Verify momentum
    print("MOMENTUM CHECK:")
    for i, k in enumerate(k_values):
        p_stand = results['standing_results'][i]['initial_momentum']
        p_prop = results['propagating_results'][i]['initial_momentum']
        print(f"  k={k}: Standing p={p_stand:.4f}, Propagating p={p_prop:.4f}")
    print()

    # Overall verdict
    overall_pass = avg_ratio > 1.5  # Require significant difference

    results['summary'] = {
        'average_ratio': float(avg_ratio),
        'min_ratio': float(min_ratio),
        'max_ratio': float(max_ratio),
        'all_k_pass': bool(all_pass),
        'overall_pass': bool(overall_pass),
    }

    print("=" * 70)
    if overall_pass:
        print(f"OVERALL RESULT: PASS (avg ratio = {avg_ratio:.2f}x)")
        print("=" * 70)
        print()
        print("INTERPRETATION:")
        print("  Standing waves (p=0) show MORE inertia than propagating waves.")
        print("  This supports the hypothesis that standing wave structure")
        print("  creates rest mass behavior (E/c^2).")
        print()
        print("  The wave equation (second-order) correctly distinguishes")
        print("  between standing waves (has rest mass) and propagating waves")
        print("  (no rest mass), validating Q54's core thesis.")
    else:
        print(f"OVERALL RESULT: FAIL (avg ratio = {avg_ratio:.2f}x)")
        print("=" * 70)
        print()
        print("INTERPRETATION:")
        print("  Standing and propagating waves show similar inertia.")
        print("  This would challenge the hypothesis that p=0 creates")
        print("  rest mass behavior.")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'test_a_results.json')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    results = main()
