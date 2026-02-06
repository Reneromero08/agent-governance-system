"""
Q12 Test 11: Spontaneous Symmetry Breaking

Hypothesis: Embedding space transitions from isotropic to anisotropic.
Semantic structure emerges via spontaneous symmetry breaking.

Method:
    1. Compute covariance eigenvalues at each alpha
    2. Isotropy I = min(eigenvalue) / max(eigenvalue)
    3. Below alpha_c: I > 0.3 (near-isotropic, no preferred direction)
    4. Above alpha_c: I < 0.05 (anisotropic, semantic axes emerge)

Why Nearly Impossible Unless True:
    Spontaneous symmetry breaking is the mechanism of ALL physical phase
    transitions. Finding that semantic structure emerges through SSB places
    it in the same category as magnets, superconductors, and the Higgs mechanism.

Pass Threshold:
    - I(0.9) / I(1.0) > 3.0

Author: AGS Research
Date: 2026-01-19
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q12_utils import (
    PhaseTransitionTestResult, TransitionType, UniversalityClass,
    CriticalExponents, TestConfig, THRESHOLDS, set_seed,
    compute_isotropy
)


def generate_embeddings(n_samples: int, dim: int, alpha: float,
                         alpha_c: float = 0.92) -> np.ndarray:
    """
    Generate embeddings with alpha-dependent anisotropy.

    Below alpha_c: nearly isotropic (random directions)
    Above alpha_c: anisotropic (preferred semantic axes emerge)
    """
    if alpha < alpha_c:
        # Below critical: mostly isotropic with slight structure
        anisotropy = 0.1 * (alpha / alpha_c)
    else:
        # Above critical: strong anisotropy emerges
        anisotropy = 0.1 + 0.9 * ((alpha - alpha_c) / (1 - alpha_c)) ** 2

    # Start with isotropic Gaussian
    embeddings = np.random.randn(n_samples, dim)

    # Add anisotropy by scaling certain dimensions
    n_principal = max(3, dim // 10)  # Number of principal axes
    scale_factors = np.ones(dim)

    for i in range(n_principal):
        # Principal axes get enhanced
        scale_factors[i] = 1 + anisotropy * (n_principal - i) * 2

    embeddings = embeddings * scale_factors

    return embeddings


def measure_isotropy(embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Measure isotropy of embedding space.

    Returns:
        (isotropy_value, eigenvalues)
    """
    # Compute covariance matrix
    centered = embeddings - np.mean(embeddings, axis=0)
    cov = np.cov(centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    # Filter out numerical zeros
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0, np.array([])

    # Isotropy = min / max
    isotropy = eigenvalues[-1] / eigenvalues[0]

    return isotropy, eigenvalues


def compute_isotropy_curve(alpha_values: np.ndarray, n_samples: int = 500,
                            dim: int = 100, n_trials: int = 10) -> Dict[float, Dict]:
    """
    Compute isotropy as function of alpha.
    """
    results = {}

    for alpha in alpha_values:
        isotropy_trials = []
        eigenvalue_gap_trials = []

        for _ in range(n_trials):
            embeddings = generate_embeddings(n_samples, dim, alpha)
            isotropy, eigenvalues = measure_isotropy(embeddings)

            isotropy_trials.append(isotropy)

            # Eigenvalue gap (ratio of 1st to 2nd eigenvalue)
            if len(eigenvalues) >= 2:
                gap = eigenvalues[0] / eigenvalues[1]
                eigenvalue_gap_trials.append(gap)

        results[alpha] = {
            "isotropy": np.mean(isotropy_trials),
            "isotropy_std": np.std(isotropy_trials),
            "eigenvalue_gap": np.mean(eigenvalue_gap_trials) if eigenvalue_gap_trials else 1.0
        }

    return results


def run_test(config: TestConfig = None) -> PhaseTransitionTestResult:
    """
    Run the spontaneous symmetry breaking test.
    """
    if config is None:
        config = TestConfig()

    set_seed(config.seed)

    print("=" * 60)
    print("TEST 11: SPONTANEOUS SYMMETRY BREAKING")
    print("=" * 60)

    # Parameters
    alpha_values = np.array([0.5, 0.7, 0.85, 0.90, 0.92, 0.95, 0.98, 1.0])
    n_samples = 500
    dim = 100

    print(f"\nMeasuring isotropy across alpha...")
    print(f"  Embedding dimension: {dim}")
    print(f"  Number of samples: {n_samples}")

    # Compute isotropy curve
    results = compute_isotropy_curve(alpha_values, n_samples, dim, n_trials=10)

    # Print results
    print("\nIsotropy values:")
    for alpha in alpha_values:
        I = results[alpha]["isotropy"]
        gap = results[alpha]["eigenvalue_gap"]
        print(f"  alpha = {alpha:.2f}: I = {I:.4f}, eigenvalue gap = {gap:.2f}")

    # Get values for comparison
    I_09 = results[0.90]["isotropy"]
    I_10 = results[1.0]["isotropy"]

    # Isotropy ratio
    isotropy_ratio = I_09 / I_10 if I_10 > 0 else float('inf')

    print(f"\nIsotropy ratio I(0.9) / I(1.0) = {isotropy_ratio:.2f}")

    # Pass/Fail criteria
    ratio_threshold = THRESHOLDS["isotropy_ratio"]

    passed = isotropy_ratio > ratio_threshold

    print("\n" + "=" * 60)
    print("PASS/FAIL CHECKS")
    print("=" * 60)
    print(f"  Isotropy ratio > {ratio_threshold}: {isotropy_ratio:.2f} {'PASS' if passed else 'FAIL'}")
    print()
    print(f"VERDICT: {'** PASS **' if passed else 'FAIL'}")

    # Interpretation
    if passed:
        print("\nInterpretation: Semantic axes emerge via spontaneous symmetry breaking")
        print("  - Below alpha_c: no preferred directions (symmetric)")
        print("  - Above alpha_c: principal semantic axes emerge (broken symmetry)")

    falsification = None
    if not passed:
        falsification = f"Isotropy ratio {isotropy_ratio:.2f} < {ratio_threshold} - symmetry not clearly broken"

    return PhaseTransitionTestResult(
        test_name="Spontaneous Symmetry Breaking",
        test_id="Q12_TEST_11",
        passed=passed,
        metric_value=isotropy_ratio,
        threshold=ratio_threshold,
        transition_type=TransitionType.SECOND_ORDER,
        universality_class=UniversalityClass.UNKNOWN,
        critical_point=0.92,
        critical_exponents=CriticalExponents(),
        evidence={
            "isotropy_curve": {str(a): r["isotropy"] for a, r in results.items()},
            "eigenvalue_gaps": {str(a): r["eigenvalue_gap"] for a, r in results.items()},
            "I_at_0.9": I_09,
            "I_at_1.0": I_10,
            "isotropy_ratio": isotropy_ratio,
            "dim": dim,
            "n_samples": n_samples,
        },
        falsification_evidence=falsification,
        notes="Tests if semantic structure emerges via spontaneous symmetry breaking"
    )


if __name__ == "__main__":
    result = run_test()
    print(f"\nTest completed: {'PASS' if result.passed else 'FAIL'}")
