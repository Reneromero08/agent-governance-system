#!/usr/bin/env python3
"""
Q42 Test 3: Acausal Consensus - Non-Local Agreement

PURPOSE: Test if disconnected observers can agree beyond chance.

If A1 (locality) holds:
- Agreement requires information transfer
- Disconnected observers should have uncorrelated R values
- correlation(R_A, R_B) ≈ 0

If non-locality exists:
- Observers might agree acausally
- correlation(R_A, R_B) > 0 without shared information
- Would suggest "semantic non-locality"

Run: python test_q42_acausal_consensus.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from bell import compute_R_base


# =============================================================================
# OBSERVER SIMULATION
# =============================================================================

class Observer:
    """Simulates an observer interpreting ground truths."""

    def __init__(self, seed: int, bias: float = 0.0, noise: float = 1.0):
        """
        Initialize observer.

        Args:
            seed: Random seed for this observer's interpretations
            bias: Systematic bias in observations
            noise: Observation noise level
        """
        self.seed = seed
        self.bias = bias
        self.noise = noise
        self.rng = np.random.RandomState(seed)

    def interpret(self, truth: float, n_samples: int = 100) -> np.ndarray:
        """
        Interpret a ground truth value.

        Returns noisy observations around the truth.
        """
        observations = truth + self.bias + self.noise * self.rng.randn(n_samples)
        return observations

    def compute_R(self, truth: float, n_samples: int = 100) -> float:
        """Interpret truth and compute R."""
        obs = self.interpret(truth, n_samples)
        return compute_R_base(obs, truth)


def generate_population(
    n_observers: int,
    seed: int,
    bias_std: float = 0.1,
    noise_range: Tuple[float, float] = (0.5, 1.5)
) -> List[Observer]:
    """
    Generate a population of observers.

    Each observer has independent random characteristics.
    """
    rng = np.random.RandomState(seed)

    observers = []
    for i in range(n_observers):
        obs_seed = seed * 1000 + i
        bias = bias_std * rng.randn()
        noise = rng.uniform(*noise_range)
        observers.append(Observer(obs_seed, bias, noise))

    return observers


def compute_population_R(
    population: List[Observer],
    truths: np.ndarray,
    n_samples_per_truth: int = 100
) -> np.ndarray:
    """
    Compute R values for a population across multiple truths.

    Returns:
        (n_truths, n_observers) array of R values
    """
    n_truths = len(truths)
    n_observers = len(population)

    R_matrix = np.zeros((n_truths, n_observers))

    for t, truth in enumerate(truths):
        for o, observer in enumerate(population):
            R_matrix[t, o] = observer.compute_R(truth, n_samples_per_truth)

    return R_matrix


# =============================================================================
# ACAUSAL CONSENSUS TESTS
# =============================================================================

def test_independent_populations() -> Tuple[bool, Dict]:
    """
    Test: Two independent populations should have uncorrelated R values.

    This is the baseline - if they're truly independent, correlation ≈ 0.
    """
    print("\n" + "=" * 70)
    print("TEST 3a: Independent Populations (Control)")
    print("=" * 70)

    # Generate two populations with different seeds
    pop_A = generate_population(n_observers=20, seed=42)
    pop_B = generate_population(n_observers=20, seed=137)

    # Same ground truths
    truths = np.linspace(-2, 2, 50)

    # Compute R for each population
    R_A = compute_population_R(pop_A, truths)
    R_B = compute_population_R(pop_B, truths)

    # Mean R across observers for each truth
    mean_R_A = R_A.mean(axis=1)
    mean_R_B = R_B.mean(axis=1)

    # Correlation between populations
    corr, p_value = pearsonr(mean_R_A, mean_R_B)

    print(f"\nPopulation A: 20 observers, seed=42")
    print(f"Population B: 20 observers, seed=137")
    print(f"Ground truths: 50 values in [-2, 2]")
    print(f"\nMean R correlation between populations:")
    print(f"  Pearson r = {corr:.4f}")
    print(f"  p-value = {p_value:.4e}")

    # For independent populations, expect |r| < 0.3
    passed = abs(corr) < 0.3 or p_value > 0.01

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Independent populations are uncorrelated")

    return passed, {
        'correlation': float(corr),
        'p_value': float(p_value),
        'n_truths': len(truths),
        'n_observers': 20,
        'passed': passed
    }


def test_shared_truth_correlation() -> Tuple[bool, Dict]:
    """
    Test: Populations seeing SAME truths - do they show correlation?

    This tests if shared ground truth produces R correlation
    even without shared information.
    """
    print("\n" + "=" * 70)
    print("TEST 3b: Shared Truth Effect")
    print("=" * 70)

    # Two populations, same truths
    pop_A = generate_population(n_observers=20, seed=42)
    pop_B = generate_population(n_observers=20, seed=137)

    # Easy truths (low noise) vs hard truths (high noise)
    easy_truths = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Low variance
    hard_truths = np.linspace(-5, 5, 50)  # High variance

    R_A_easy = compute_population_R(pop_A, easy_truths)
    R_B_easy = compute_population_R(pop_B, easy_truths)
    R_A_hard = compute_population_R(pop_A, hard_truths)
    R_B_hard = compute_population_R(pop_B, hard_truths)

    # Check if R values correlate across populations
    corr_easy, p_easy = pearsonr(R_A_easy.mean(axis=1), R_B_easy.mean(axis=1))
    corr_hard, p_hard = pearsonr(R_A_hard.mean(axis=1), R_B_hard.mean(axis=1))

    print(f"\nEasy truths (constant):")
    print(f"  Correlation: {corr_easy:.4f} (p={p_easy:.4e})")
    print(f"\nHard truths (varied):")
    print(f"  Correlation: {corr_hard:.4f} (p={p_hard:.4e})")

    # Shared truth should NOT produce acausal correlation
    # (If it does, that's interesting but not "non-local")
    passed = True  # This is exploratory

    return passed, {
        'easy_corr': float(corr_easy),
        'easy_p': float(p_easy),
        'hard_corr': float(corr_hard),
        'hard_p': float(p_hard),
        'passed': passed
    }


def test_acausal_consensus_simulation() -> Tuple[bool, Dict]:
    """
    Test: Simulate scenario where acausal consensus SHOULD be detected.

    If we inject non-local coupling, does correlation appear?
    """
    print("\n" + "=" * 70)
    print("TEST 3c: Acausal Consensus Simulation")
    print("=" * 70)

    # Baseline: independent populations
    pop_A = generate_population(n_observers=20, seed=42)
    pop_B = generate_population(n_observers=20, seed=137)

    truths = np.linspace(-2, 2, 50)

    R_A = compute_population_R(pop_A, truths)
    R_B = compute_population_R(pop_B, truths)

    mean_R_A = R_A.mean(axis=1)
    mean_R_B = R_B.mean(axis=1)

    baseline_corr, _ = pearsonr(mean_R_A, mean_R_B)

    # Inject "non-local coupling" - make B's R depend on A's
    # This simulates what we'd expect if semantic non-locality exists
    coupling_strength = 0.5
    R_B_coupled = (1 - coupling_strength) * mean_R_B + coupling_strength * mean_R_A
    R_B_coupled += 0.1 * np.random.randn(len(truths))  # Add noise

    coupled_corr, coupled_p = pearsonr(mean_R_A, R_B_coupled)

    print(f"\nBaseline (no coupling):")
    print(f"  Correlation: {baseline_corr:.4f}")

    print(f"\nWith simulated non-local coupling (strength={coupling_strength}):")
    print(f"  Correlation: {coupled_corr:.4f} (p={coupled_p:.4e})")

    # The test passes if we can detect the injected coupling
    detected = coupled_corr > 0.3 and coupled_p < 0.001

    print(f"\n{'✓ PASS' if detected else '✗ FAIL'}: Non-local coupling is detectable")

    return detected, {
        'baseline_corr': float(baseline_corr),
        'coupled_corr': float(coupled_corr),
        'coupled_p': float(coupled_p),
        'coupling_strength': coupling_strength,
        'detected': detected
    }


def test_cross_population_spearman() -> Tuple[bool, Dict]:
    """
    Test: Use Spearman correlation (rank-based) for robustness.
    """
    print("\n" + "=" * 70)
    print("TEST 3d: Rank Correlation (Spearman)")
    print("=" * 70)

    pop_A = generate_population(n_observers=30, seed=42)
    pop_B = generate_population(n_observers=30, seed=137)

    truths = np.linspace(-3, 3, 100)

    R_A = compute_population_R(pop_A, truths)
    R_B = compute_population_R(pop_B, truths)

    mean_R_A = R_A.mean(axis=1)
    mean_R_B = R_B.mean(axis=1)

    # Spearman rank correlation
    spearman_r, spearman_p = spearmanr(mean_R_A, mean_R_B)

    print(f"\nSpearman correlation:")
    print(f"  ρ = {spearman_r:.4f}")
    print(f"  p = {spearman_p:.4e}")

    # For independent populations, expect low correlation
    passed = abs(spearman_r) < 0.3 or spearman_p > 0.01

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Rank correlation consistent with independence")

    return passed, {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'passed': passed
    }


def test_bootstrap_null_distribution() -> Tuple[bool, Dict]:
    """
    Test: Bootstrap null distribution for acausal consensus.

    If real correlation exceeds 95% of shuffled correlations,
    we have evidence of acausal consensus.
    """
    print("\n" + "=" * 70)
    print("TEST 3e: Bootstrap Null Distribution")
    print("=" * 70)

    pop_A = generate_population(n_observers=20, seed=42)
    pop_B = generate_population(n_observers=20, seed=137)

    truths = np.linspace(-2, 2, 50)

    R_A = compute_population_R(pop_A, truths).mean(axis=1)
    R_B = compute_population_R(pop_B, truths).mean(axis=1)

    # Observed correlation
    observed_corr, _ = pearsonr(R_A, R_B)

    # Bootstrap null distribution by shuffling
    n_bootstrap = 1000
    null_corrs = []
    for _ in range(n_bootstrap):
        shuffled_B = np.random.permutation(R_B)
        null_corr, _ = pearsonr(R_A, shuffled_B)
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))
    percentile = np.mean(null_corrs < observed_corr) * 100

    print(f"\nObserved correlation: {observed_corr:.4f}")
    print(f"Null distribution (n={n_bootstrap}):")
    print(f"  Mean: {null_corrs.mean():.4f}")
    print(f"  Std: {null_corrs.std():.4f}")
    print(f"  95th percentile: {np.percentile(null_corrs, 95):.4f}")
    print(f"\nObserved is at {percentile:.1f}th percentile")
    print(f"Bootstrap p-value: {p_value:.4f}")

    # If p > 0.05, consistent with null (no acausal consensus)
    passed = p_value > 0.05

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: No significant acausal consensus detected")

    return passed, {
        'observed_corr': float(observed_corr),
        'null_mean': float(null_corrs.mean()),
        'null_std': float(null_corrs.std()),
        'percentile': float(percentile),
        'p_value': float(p_value),
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests() -> Dict:
    """Run all acausal consensus tests."""

    print("\n" + "=" * 70)
    print("Q42 TEST 3: ACAUSAL CONSENSUS - NON-LOCAL AGREEMENT")
    print("=" * 70)
    print("\nHypotheses:")
    print("  H0: Correlation(R_A, R_B) ≈ 0 (locality holds)")
    print("  H1: Correlation(R_A, R_B) > 0.3 (acausal agreement)")

    results = {}

    tests = [
        ('independent', test_independent_populations),
        ('shared_truth', test_shared_truth_correlation),
        ('acausal_sim', test_acausal_consensus_simulation),
        ('spearman', test_cross_population_spearman),
        ('bootstrap', test_bootstrap_null_distribution),
    ]

    for name, test_func in tests:
        passed, data = test_func()
        results[name] = data

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Acausal Consensus Tests")
    print("=" * 70)

    n_passed = sum(1 for r in results.values() if r.get('passed', False))

    # Key finding: Is there evidence of acausal consensus?
    bootstrap_p = results['bootstrap'].get('p_value', 1.0)
    observed_corr = results['bootstrap'].get('observed_corr', 0.0)

    if bootstrap_p < 0.001 and observed_corr > 0.3:
        verdict = "ACAUSAL CONSENSUS DETECTED (H1)"
        h0_status = "REJECTED"
        h1_status = "CONFIRMED"
    elif bootstrap_p < 0.05:
        verdict = "Weak evidence of acausal correlation"
        h0_status = "UNCERTAIN"
        h1_status = "UNCERTAIN"
    else:
        verdict = "NO ACAUSAL CONSENSUS (H0 supported)"
        h0_status = "CONFIRMED"
        h1_status = "REJECTED"

    print(f"\nTests passed: {n_passed}/{len(tests)}")
    print(f"Bootstrap p-value: {bootstrap_p:.4f}")
    print(f"Observed correlation: {observed_corr:.4f}")
    print(f"\nVerdict: {verdict}")

    results['summary'] = {
        'n_passed': n_passed,
        'n_total': len(tests),
        'verdict': verdict,
        'h0_status': h0_status,
        'h1_status': h1_status,
        'bootstrap_p': float(bootstrap_p),
        'observed_corr': float(observed_corr),
        'timestamp': datetime.now().isoformat()
    }

    return results


if __name__ == '__main__':
    results = run_all_tests()

    output_path = Path(__file__).parent / 'q42_test3_results.json'
    with open(output_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")
    sys.exit(0 if results['summary']['h0_status'] == 'CONFIRMED' else 1)
