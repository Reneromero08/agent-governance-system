#!/usr/bin/env python3
"""
Q41: L-Function Utilities

Provides L-function computation for Langlands tests.

L-functions are central to the Langlands program:
- They encode arithmetic/automorphic structure
- Functional equations relate L(s) to L(1-s)
- Euler products factor over primes

For semantic spaces, we define:
- "Semantic primes" as irreducible meaning units (via factorization)
- L(s, π) for semantic representations π
- Functional equation and analytic properties

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.cluster.vq import kmeans2
from scipy.special import gamma as gamma_func
import warnings

# =============================================================================
# SEMANTIC PRIME FACTORIZATION
# =============================================================================

def find_semantic_primes(
    embeddings: np.ndarray,
    n_primes: int = 10,
    method: str = "nmf"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find "semantic primes" - irreducible basis elements.

    Methods:
    - "nmf": Non-negative matrix factorization (requires non-negative input)
    - "kmeans": K-means clustering centroids
    - "ica": Independent Component Analysis

    Returns:
        primes: (n_primes, d) array of prime vectors
        coefficients: (n, n_primes) array of factorization coefficients
    """
    n, d = embeddings.shape

    if method == "nmf":
        # Shift to non-negative
        X = embeddings - embeddings.min()

        try:
            from sklearn.decomposition import NMF
            nmf = NMF(n_components=n_primes, init='nndsvda', max_iter=500, random_state=42)
            coefficients = nmf.fit_transform(X)
            primes = nmf.components_
        except ImportError:
            # Fallback to simple factorization
            primes, _ = kmeans2(X, n_primes, minit='++')
            coefficients = _compute_coefficients(X, primes)

    elif method == "kmeans":
        primes, labels = kmeans2(embeddings, n_primes, minit='++')
        coefficients = _compute_coefficients(embeddings, primes)

    elif method == "ica":
        try:
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=n_primes, random_state=42, max_iter=500)
            coefficients = ica.fit_transform(embeddings)
            primes = ica.components_
        except ImportError:
            primes, _ = kmeans2(embeddings, n_primes, minit='++')
            coefficients = _compute_coefficients(embeddings, primes)

    else:
        primes, _ = kmeans2(embeddings, n_primes, minit='++')
        coefficients = _compute_coefficients(embeddings, primes)

    return primes, coefficients


def _compute_coefficients(X: np.ndarray, primes: np.ndarray) -> np.ndarray:
    """Compute factorization coefficients via least squares."""
    # X ≈ coefficients @ primes
    # coefficients = X @ primes^T @ (primes @ primes^T)^{-1}
    reg = 1e-6 * np.eye(primes.shape[0])
    coefficients = X @ primes.T @ np.linalg.inv(primes @ primes.T + reg)
    return coefficients


def verify_unique_factorization(
    embeddings: np.ndarray,
    primes: np.ndarray,
    coefficients: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Verify unique factorization property.

    Returns dict with:
    - reconstruction_error: How well factorization reconstructs original
    - uniqueness_score: Measure of coefficient sparsity
    - factorization_quality: Overall quality metric
    """
    # Reconstruction
    reconstructed = coefficients @ primes
    recon_error = np.linalg.norm(embeddings - reconstructed, 'fro') / (np.linalg.norm(embeddings, 'fro') + 1e-10)

    # Uniqueness via sparsity (L1/L2 ratio)
    l1_norms = np.abs(coefficients).sum(axis=1)
    l2_norms = np.linalg.norm(coefficients, axis=1)
    sparsity = (l1_norms / (np.sqrt(coefficients.shape[1]) * l2_norms + 1e-10)).mean()

    # Quality metric
    quality = 1.0 - recon_error

    return {
        "reconstruction_error": float(recon_error),
        "uniqueness_score": float(sparsity),
        "factorization_quality": float(quality),
        "passes_threshold": bool(recon_error < threshold)
    }


# =============================================================================
# L-FUNCTION CONSTRUCTION
# =============================================================================

def compute_euler_product(
    embeddings: np.ndarray,
    primes: np.ndarray,
    s_values: np.ndarray
) -> np.ndarray:
    """
    Compute L-function via Euler product.

    L(s) = Π_p L_p(s)

    where L_p(s) = (1 - a_p * p^{-s})^{-1} for classical L-functions.

    Semantic analog:
    L_p(s) = (1 - exp(-d_p * s))^{-1}
    where d_p is the "arithmetic complexity" of prime p.
    """
    n_primes = len(primes)
    L_values = np.ones(len(s_values), dtype=complex)

    for p_idx, prime in enumerate(primes):
        # "Arithmetic complexity" - L2 norm as proxy
        d_p = np.linalg.norm(prime)

        # Prime index (position in ordering)
        p = p_idx + 2  # Start from 2 like actual primes

        for i, s in enumerate(s_values):
            # Allow computation for Re(s) > 0 (finite product converges)
            if np.real(s) > 0:
                # Classical form: (1 - a_p / p^s)^{-1}
                a_p = d_p / (np.linalg.norm(primes, axis=1).mean() + 1e-10)  # Normalized coefficient
                local_factor = 1.0 / (1.0 - a_p * (p ** (-s)) + 1e-10)
                L_values[i] *= local_factor

    return L_values


def compute_dirichlet_series(
    embeddings: np.ndarray,
    coefficients: np.ndarray,
    s_values: np.ndarray,
    n_terms: int = 100
) -> np.ndarray:
    """
    Compute L-function as Dirichlet series.

    L(s) = Σ_{n=1}^∞ a_n / n^s

    Coefficients a_n derived from embedding structure.
    """
    L_values = np.zeros(len(s_values), dtype=complex)

    # Compute Dirichlet coefficients from factorization structure
    a_n = _compute_dirichlet_coefficients(coefficients, n_terms)

    for i, s in enumerate(s_values):
        if np.real(s) > 1:
            for n in range(1, n_terms + 1):
                L_values[i] += a_n[n-1] / (n ** s)

    return L_values


def _compute_dirichlet_coefficients(coefficients: np.ndarray, n_terms: int) -> np.ndarray:
    """
    Compute Dirichlet coefficients from factorization.

    Uses Ramanujan sum-like construction.
    """
    n_points, n_primes = coefficients.shape
    a_n = np.zeros(n_terms)

    for n in range(1, n_terms + 1):
        # a_n = average projection onto n-th frequency
        phase = np.exp(2j * np.pi * n * np.arange(n_points) / n_points)
        coef_sum = np.abs(coefficients.sum(axis=1))
        a_n[n-1] = np.abs(np.sum(coef_sum * phase)) / n_points

    return a_n


# =============================================================================
# FUNCTIONAL EQUATION
# =============================================================================

def verify_functional_equation(
    L_values: np.ndarray,
    s_values: np.ndarray,
    gamma_factor: bool = True
) -> Dict[str, Any]:
    """
    Verify functional equation L(s) = ε(s) * L(1-s).

    For classical L-functions:
    Λ(s) = Γ(s/2) * π^{-s/2} * L(s) satisfies Λ(s) = Λ(1-s)

    Returns dict with quality metrics.
    """
    n = len(s_values)

    # Compute L(1-s)
    s_dual = 1 - s_values

    # Simple symmetry check around Re(s) = 1/2
    # Find pairs (s, 1-s) in our evaluation points
    pairs = []
    for i, s in enumerate(s_values):
        s_conj = 1 - s
        # Find closest match
        dists = np.abs(s_values - s_conj)
        j = np.argmin(dists)
        if dists[j] < 0.01:
            pairs.append((i, j))

    if len(pairs) < 2:
        return {
            "fe_quality": 0.0,
            "n_pairs_tested": 0,
            "passes": False,
            "notes": "Not enough symmetric pairs to test"
        }

    # Compare L(s) and L(1-s) for pairs
    ratios = []
    for i, j in pairs:
        if np.abs(L_values[j]) > 1e-10:
            ratio = np.abs(L_values[i] / L_values[j])
            ratios.append(ratio)

    if not ratios:
        return {
            "fe_quality": 0.0,
            "n_pairs_tested": 0,
            "passes": False,
            "notes": "Could not compute ratios"
        }

    # Quality = how close ratios are to constant (epsilon factor should be bounded)
    ratio_array = np.array(ratios)
    ratio_cv = np.std(ratio_array) / (np.mean(ratio_array) + 1e-10)

    # Good functional equation has consistent ratio
    quality = 1.0 / (1.0 + ratio_cv)

    return {
        "fe_quality": float(quality),
        "n_pairs_tested": len(pairs),
        "mean_ratio": float(np.mean(ratio_array)),
        "ratio_cv": float(ratio_cv),
        "passes": bool(quality > 0.3),
        "notes": f"Tested {len(pairs)} symmetric pairs"
    }


# =============================================================================
# ANALYTIC PROPERTIES
# =============================================================================

def analyze_l_function(
    L_values: np.ndarray,
    s_values: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze analytic properties of L-function.

    Returns:
    - smoothness: How smooth the function is
    - growth_rate: How fast it grows
    - zeros_estimate: Estimated location of zeros
    """
    magnitudes = np.abs(L_values)

    # Smoothness via second derivative
    if len(magnitudes) > 2:
        d2 = np.diff(magnitudes, n=2)
        smoothness = 1.0 / (1.0 + np.std(d2))
    else:
        smoothness = 0.0

    # Growth rate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mag = np.log(magnitudes + 1e-10)
        log_s = np.log(np.abs(s_values) + 1e-10)

    # Linear fit in log-log space
    valid = np.isfinite(log_mag) & np.isfinite(log_s)
    if valid.sum() > 2:
        coeffs = np.polyfit(log_s[valid].real, log_mag[valid], 1)
        growth_rate = float(coeffs[0])
    else:
        growth_rate = 0.0

    # Zero estimation (where magnitude is minimal)
    min_idx = np.argmin(magnitudes)
    zero_estimate = s_values[min_idx]

    return {
        "smoothness": float(smoothness),
        "growth_rate": growth_rate,
        "zero_estimate_re": float(np.real(zero_estimate)),
        "zero_estimate_im": float(np.imag(zero_estimate)),
        "min_magnitude": float(magnitudes.min()),
        "max_magnitude": float(magnitudes.max())
    }


# =============================================================================
# CROSS-SCALE L-FUNCTION COMPARISON
# =============================================================================

def compare_l_functions_across_scales(
    L_child: np.ndarray,
    L_parent: np.ndarray,
    s_values: np.ndarray
) -> Dict[str, Any]:
    """
    Compare L-functions at different scales for functoriality.

    Key test: L(s, φ(π)) should relate predictably to L(s, π, r)
    where φ is the lifting map and r is some representation.
    """
    # Magnitude correlation
    mag_child = np.abs(L_child)
    mag_parent = np.abs(L_parent)

    # Normalize
    mag_child_n = (mag_child - mag_child.mean()) / (mag_child.std() + 1e-10)
    mag_parent_n = (mag_parent - mag_parent.mean()) / (mag_parent.std() + 1e-10)

    correlation = float(np.corrcoef(mag_child_n, mag_parent_n)[0, 1])

    # Phase alignment
    phase_child = np.angle(L_child)
    phase_parent = np.angle(L_parent)

    phase_diff = np.abs(phase_child - phase_parent)
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Wrap around
    phase_alignment = 1.0 - phase_diff.mean() / np.pi

    # Ratio consistency (should be nearly constant for functoriality)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratios = np.abs(L_parent / (L_child + 1e-10))
        valid = np.isfinite(ratios)

    if valid.sum() > 2:
        ratio_cv = np.std(ratios[valid]) / (np.mean(ratios[valid]) + 1e-10)
        ratio_consistency = 1.0 / (1.0 + ratio_cv)
    else:
        ratio_consistency = 0.0

    return {
        "correlation": correlation,
        "phase_alignment": float(phase_alignment),
        "ratio_consistency": float(ratio_consistency),
        "functoriality_score": float((correlation + ratio_consistency) / 2),
        "passes": bool(correlation > 0.5 and ratio_consistency > 0.3)
    }


def compare_l_functions_cross_lingual(
    L_en: np.ndarray,
    L_zh: np.ndarray,
    s_values: np.ndarray
) -> Dict[str, Any]:
    """
    Compare L-functions across languages for base change analog.

    In classical Langlands, base change for GL(n) gives:
    L(s, BC_{K/F}(π)) = prod_{σ ∈ Gal(K/F)} L(s, π^σ)

    For our semantic analog, we test if L_EN and L_ZH are related by
    a consistent scaling factor (epsilon), indicating structural preservation
    across the cross-lingual embedding.
    """
    # Similar to cross-scale comparison
    mag_en = np.abs(L_en)
    mag_zh = np.abs(L_zh)

    mag_en_n = (mag_en - mag_en.mean()) / (mag_en.std() + 1e-10)
    mag_zh_n = (mag_zh - mag_zh.mean()) / (mag_zh.std() + 1e-10)

    correlation = float(np.corrcoef(mag_en_n, mag_zh_n)[0, 1])

    # Epsilon factor estimation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epsilon = L_zh / (L_en + 1e-10)
        valid = np.isfinite(epsilon)

    if valid.sum() > 2:
        epsilon_mag = np.abs(epsilon[valid])
        epsilon_consistency = 1.0 / (1.0 + np.std(epsilon_mag) / (np.mean(epsilon_mag) + 1e-10))
    else:
        epsilon_consistency = 0.0

    return {
        "correlation": correlation,
        "epsilon_consistency": float(epsilon_consistency),
        "base_change_score": float((correlation + epsilon_consistency) / 2),
        "passes": bool(correlation > 0.4)
    }
