"""
conservation_law_detector.py - Test for 8e conservation law at biological scales

The semiotic conservation law states: Df x alpha = 8e (approximately 21.746)

Where:
- Df = effective dimensionality (participation ratio)
- alpha = spectral decay exponent
- 8e = 8 * e (Euler's number) ~ 21.746

This law has been proven for semantic spaces (Q48-50).
Q18 tests if it holds at biological scales.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
import json


# The universal constant
CONSTANT_8E = 8 * np.e  # ~21.746


@dataclass
class ConservationResult:
    """Result of 8e conservation test."""
    scale: str
    Df: float
    alpha: float
    product: float
    deviation_from_8e: float  # As percentage
    passed: bool
    eigenvalues: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "Df": self.Df,
            "alpha": self.alpha,
            "Df_x_alpha": self.product,
            "expected_8e": CONSTANT_8E,
            "deviation_pct": self.deviation_from_8e,
            "passed": self.passed
        }


def compute_eigenvalues(data: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues from data covariance matrix.

    Args:
        data: Data matrix (samples x features)

    Returns:
        Eigenvalues in descending order
    """
    # Center data
    data_centered = data - np.mean(data, axis=0)

    # Compute covariance matrix
    n_samples = data.shape[0]
    cov = np.dot(data_centered.T, data_centered) / (n_samples - 1)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)

    # Sort descending and filter positive
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    return eigenvalues


def compute_Df(eigenvalues: np.ndarray) -> float:
    """
    Compute effective dimensionality (participation ratio).

    Df = (sum(lambda))^2 / sum(lambda^2)

    This measures how many dimensions effectively contribute.
    - If all eigenvalues equal: Df = n (full rank)
    - If one dominates: Df -> 1
    """
    if len(eigenvalues) == 0:
        return 0.0

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)

    if sum_lambda_sq < 1e-20:
        return 0.0

    return (sum_lambda**2) / sum_lambda_sq


def compute_alpha(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral decay exponent.

    Fits: lambda_k ~ k^(-alpha)

    The Riemann critical line value is alpha = 0.5.
    """
    if len(eigenvalues) < 3:
        return 0.5  # Default to Riemann value

    ranks = np.arange(1, len(eigenvalues) + 1)

    # Log-log regression
    log_ranks = np.log(ranks)
    log_eigenvalues = np.log(eigenvalues + 1e-20)

    # Remove infinities
    valid = np.isfinite(log_eigenvalues)
    if valid.sum() < 3:
        return 0.5

    slope, intercept = np.polyfit(log_ranks[valid], log_eigenvalues[valid], 1)

    alpha = -slope
    return max(0.0, alpha)


def test_8e_at_scale(
    data: np.ndarray,
    scale_name: str,
    tolerance: float = 0.15
) -> ConservationResult:
    """
    Test if Df x alpha = 8e at a given scale.

    Args:
        data: Data matrix (samples x features)
        scale_name: Name of the scale ("neural", "molecular", etc.)
        tolerance: Acceptable deviation from 8e (default 15%)

    Returns:
        ConservationResult with Df, alpha, and pass/fail
    """
    eigenvalues = compute_eigenvalues(data)

    if len(eigenvalues) < 3:
        return ConservationResult(
            scale=scale_name,
            Df=0.0,
            alpha=0.0,
            product=0.0,
            deviation_from_8e=100.0,
            passed=False,
            eigenvalues=eigenvalues
        )

    Df = compute_Df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    product = Df * alpha

    deviation = abs(product - CONSTANT_8E) / CONSTANT_8E * 100

    return ConservationResult(
        scale=scale_name,
        Df=Df,
        alpha=alpha,
        product=product,
        deviation_from_8e=deviation,
        passed=deviation <= tolerance * 100,
        eigenvalues=eigenvalues
    )


def test_8e_universality(
    scale_results: List[ConservationResult],
    max_cv: float = 0.15
) -> Dict[str, Any]:
    """
    Test if 8e is universal across all scales.

    Computes coefficient of variation (CV) of Df x alpha across scales.
    If CV < max_cv, the constant is universal.

    Args:
        scale_results: List of ConservationResult from different scales
        max_cv: Maximum acceptable CV (default 15%)

    Returns:
        Dict with universality test results
    """
    products = [r.product for r in scale_results if r.product > 0]

    if len(products) < 2:
        return {
            "cv": float('inf'),
            "mean_product": 0.0,
            "n_scales": len(products),
            "passed": False,
            "reason": "Insufficient scales"
        }

    mean_product = np.mean(products)
    std_product = np.std(products)
    cv = std_product / mean_product if mean_product > 0 else float('inf')

    # Also test deviation from expected 8e
    mean_deviation = np.mean([abs(p - CONSTANT_8E) / CONSTANT_8E for p in products])

    return {
        "cv": cv,
        "mean_product": mean_product,
        "std_product": std_product,
        "expected_8e": CONSTANT_8E,
        "mean_deviation_from_8e_pct": mean_deviation * 100,
        "n_scales": len(products),
        "products_by_scale": {r.scale: r.product for r in scale_results},
        "passed": cv <= max_cv,
        "universality_confirmed": cv <= max_cv and mean_deviation <= 0.15
    }


def riemann_connection_test(eigenvalues: np.ndarray) -> Dict[str, Any]:
    """
    Test connection to Riemann zeta function.

    The Riemann critical line is at Re(s) = 1/2.
    If alpha ~ 0.5, there's a connection.

    Also tests if eigenvalue spacing resembles Riemann zero spacing.
    """
    if len(eigenvalues) < 10:
        return {
            "alpha": 0.0,
            "deviation_from_half": float('inf'),
            "spacing_correlation": 0.0,
            "riemann_connection": False
        }

    alpha = compute_alpha(eigenvalues)
    deviation_from_half = abs(alpha - 0.5)

    # Eigenvalue spacing analysis
    log_eigenvalues = np.log(eigenvalues + 1e-20)
    spacings = np.diff(log_eigenvalues)

    # Expected Riemann zero spacing (GUE random matrix)
    # Mean spacing ~ pi/log(n) for Riemann zeros
    n = len(eigenvalues)
    expected_spacing = np.pi / np.log(n + 1)

    mean_spacing = np.mean(np.abs(spacings))
    spacing_ratio = mean_spacing / expected_spacing if expected_spacing > 0 else 0

    return {
        "alpha": alpha,
        "deviation_from_half": deviation_from_half,
        "mean_spacing": mean_spacing,
        "expected_riemann_spacing": expected_spacing,
        "spacing_ratio": spacing_ratio,
        "riemann_connection": deviation_from_half < 0.1 and 0.5 < spacing_ratio < 2.0
    }


def generate_report(
    scale_results: List[ConservationResult]
) -> Dict[str, Any]:
    """
    Generate comprehensive 8e conservation report.

    Args:
        scale_results: Results from all tested scales

    Returns:
        Complete report dict
    """
    universality = test_8e_universality(scale_results)

    # Per-scale summary
    scale_summary = {}
    for result in scale_results:
        scale_summary[result.scale] = {
            "Df": result.Df,
            "alpha": result.alpha,
            "Df_x_alpha": result.product,
            "deviation_pct": result.deviation_from_8e,
            "passed": result.passed
        }

    # Riemann connection for each scale
    riemann_tests = {}
    for result in scale_results:
        if result.eigenvalues is not None and len(result.eigenvalues) >= 10:
            riemann_tests[result.scale] = riemann_connection_test(result.eigenvalues)

    return {
        "expected_constant": CONSTANT_8E,
        "scales_tested": len(scale_results),
        "scales_passed": sum(1 for r in scale_results if r.passed),
        "universality": universality,
        "scale_details": scale_summary,
        "riemann_connection": riemann_tests,
        "conclusion": (
            "8e CONFIRMED at biological scales"
            if universality["passed"] and universality.get("universality_confirmed", False)
            else "8e NOT confirmed - further investigation needed"
        )
    }
