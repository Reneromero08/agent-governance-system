"""Core infrastructure for Q40 QEC tests.

Provides error injection, R-gate simulation, and baseline comparison utilities.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add library paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "THOUGHT" / "LAB" / "VECTOR_ELO" / "eigen-alignment"))

from lib.mds import squared_distance_matrix, classical_mds, effective_rank
from lib.procrustes import procrustes_align, cosine_similarity


# =============================================================================
# Constants
# =============================================================================

SEMIOTIC_CONSTANT = 8 * np.e  # Df * alpha = 8e ~ 21.746
DEFAULT_R_THRESHOLD = 1.732   # sqrt(3) from Q23


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ErrorInjectionResult:
    """Result of error injection on embeddings."""
    original: np.ndarray
    corrupted: np.ndarray
    error_type: str
    error_params: Dict
    n_errors: int


@dataclass
class RGateResult:
    """Result of R-gate evaluation."""
    R_value: float
    sigma: float
    passed: bool
    threshold: float


@dataclass
class CodeDistanceResult:
    """Result of code distance determination."""
    t_max: int                    # Maximum correctable errors
    d: int                        # Code distance = 2*t_max + 1
    error_type: str               # Type of errors injected
    gate_pass_rates: List[float]  # Pass rate at each n_errors
    semantic_t_max: int           # t_max for semantic embeddings
    random_t_max: int             # t_max for random embeddings
    p_value: float                # Statistical significance
    is_significant: bool          # semantic > random at p < 0.01


@dataclass
class ThresholdResult:
    """Result of error threshold analysis."""
    epsilon_th: float             # Error threshold
    k_below: float                # Suppression exponent below threshold
    k_above: float                # Amplification exponent above threshold
    fit_r2: float                 # Goodness of fit
    semantic_k: float             # Suppression for semantic
    random_k: float               # Suppression for random
    is_qecc: bool                 # k > 2 indicates QECC


# =============================================================================
# Error Injection Functions
# =============================================================================

def flip_dimension_sign(embedding: np.ndarray, dim_idx: int) -> np.ndarray:
    """Flip the sign of a specific dimension."""
    result = embedding.copy()
    result[dim_idx] = -result[dim_idx]
    return result


def zero_dimension(embedding: np.ndarray, dim_idx: int) -> np.ndarray:
    """Zero out a specific dimension."""
    result = embedding.copy()
    result[dim_idx] = 0.0
    # Re-normalize to unit sphere
    norm = np.linalg.norm(result)
    if norm > 1e-10:
        result = result / norm
    return result


def add_gaussian_noise(embedding: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise and re-normalize."""
    noise = np.random.normal(0, sigma, embedding.shape)
    result = embedding + noise
    norm = np.linalg.norm(result)
    if norm > 1e-10:
        result = result / norm
    return result


def random_direction_perturbation(embedding: np.ndarray, epsilon: float) -> np.ndarray:
    """Perturb in a random direction by epsilon."""
    direction = np.random.randn(len(embedding))
    direction = direction / np.linalg.norm(direction)
    result = embedding + epsilon * direction
    norm = np.linalg.norm(result)
    if norm > 1e-10:
        result = result / norm
    return result


def inject_n_errors(
    embedding: np.ndarray,
    n_errors: int,
    error_type: str,
    **kwargs
) -> ErrorInjectionResult:
    """Inject n errors of specified type into embedding.

    Args:
        embedding: Original embedding vector
        n_errors: Number of errors to inject
        error_type: One of 'dimension_flip', 'dimension_zero',
                    'gaussian_noise', 'random_direction'
        **kwargs: Additional parameters for specific error types
                  - sigma: noise level for gaussian_noise (default 0.05)
                  - epsilon: perturbation size for random_direction (default 0.15)
                  - n_dims_per_error: dims to flip per error for dimension_flip

    Returns:
        ErrorInjectionResult with original and corrupted embeddings
    """
    corrupted = embedding.copy()
    dim = len(embedding)

    if error_type == "dimension_flip":
        # Flip dimensions - can flip multiple dims per "error" for meaningful impact
        n_dims_per_error = kwargs.get('n_dims_per_error', 1)
        total_dims = min(n_errors * n_dims_per_error, dim)
        dims_to_flip = np.random.choice(dim, total_dims, replace=False)
        for d in dims_to_flip:
            corrupted = flip_dimension_sign(corrupted, d)

    elif error_type == "dimension_zero":
        n_dims_per_error = kwargs.get('n_dims_per_error', 1)
        total_dims = min(n_errors * n_dims_per_error, dim)
        dims_to_zero = np.random.choice(dim, total_dims, replace=False)
        for d in dims_to_zero:
            corrupted[d] = 0.0
        # Re-normalize
        norm = np.linalg.norm(corrupted)
        if norm > 1e-10:
            corrupted = corrupted / norm

    elif error_type == "gaussian_noise":
        sigma = kwargs.get('sigma', 0.05)  # Smaller default for gradual degradation
        # Apply noise n_errors times (cumulative)
        for _ in range(n_errors):
            corrupted = add_gaussian_noise(corrupted, sigma)

    elif error_type == "random_direction":
        epsilon = kwargs.get('epsilon', 0.15)  # Moderate default
        for _ in range(n_errors):
            corrupted = random_direction_perturbation(corrupted, epsilon)

    else:
        raise ValueError(f"Unknown error type: {error_type}")

    return ErrorInjectionResult(
        original=embedding,
        corrupted=corrupted,
        error_type=error_type,
        error_params=kwargs,
        n_errors=n_errors
    )


# =============================================================================
# R-Gate Functions
# =============================================================================

def compute_agreement(embeddings: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity (agreement)."""
    n = len(embeddings)
    if n < 2:
        return 1.0

    # Compute all pairwise cosine similarities
    cos_sim = embeddings @ embeddings.T
    # Extract upper triangle (exclude diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    agreements = cos_sim[mask]

    return float(np.mean(agreements))


def compute_dispersion(embeddings: np.ndarray) -> float:
    """Compute dispersion (standard deviation of agreements)."""
    n = len(embeddings)
    if n < 2:
        return 0.0

    cos_sim = embeddings @ embeddings.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    agreements = cos_sim[mask]

    return float(np.std(agreements))


def compute_R(
    embeddings: np.ndarray,
    epsilon: float = 1e-10
) -> Tuple[float, float]:
    """Compute R-value and dispersion for a set of embeddings.

    R = E / sigma where E is agreement and sigma is dispersion.

    Args:
        embeddings: (n, d) array of L2-normalized embeddings
        epsilon: Small value to prevent division by zero

    Returns:
        Tuple of (R_value, sigma)
    """
    E = compute_agreement(embeddings)
    sigma = compute_dispersion(embeddings)

    R = E / (sigma + epsilon)

    return float(R), float(sigma)


def r_gate(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_R_THRESHOLD
) -> RGateResult:
    """Apply R-gate to embeddings.

    Args:
        embeddings: (n, d) array of observations
        threshold: R threshold for gate (default: sqrt(3))

    Returns:
        RGateResult with R-value and pass/fail status
    """
    R, sigma = compute_R(embeddings)
    passed = R > threshold

    return RGateResult(
        R_value=R,
        sigma=sigma,
        passed=passed,
        threshold=threshold
    )


def compute_reference_agreement(
    embeddings: np.ndarray,
    reference: np.ndarray
) -> float:
    """Compute mean cosine similarity between embeddings and reference.

    Args:
        embeddings: (n, d) array of observations
        reference: (d,) reference embedding (e.g., M-field centroid)

    Returns:
        Mean cosine similarity to reference
    """
    # Normalize reference
    ref_norm = reference / (np.linalg.norm(reference) + 1e-10)
    # Compute similarities
    sims = embeddings @ ref_norm
    return float(np.mean(sims))


def compute_m_field_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute M-field centroid (mean direction of observations).

    The M-field centroid represents the "consensus truth" - the central
    tendency of all observations. For valid semantic content, observations
    should cluster around a meaningful centroid.

    Args:
        embeddings: (n, d) L2-normalized observations

    Returns:
        L2-normalized centroid vector
    """
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-10:
        return centroid
    return centroid / norm


def r_gate_with_reference(
    embeddings: np.ndarray,
    reference: np.ndarray,
    threshold: float = DEFAULT_R_THRESHOLD,
    reference_weight: float = 0.5
) -> RGateResult:
    """Apply R-gate with reference comparison.

    This enhanced R-gate measures both:
    1. Internal agreement (observations agree with each other)
    2. Reference alignment (observations agree with ground truth)

    The combined metric detects both random corruption (low internal agreement)
    AND systematic corruption (high internal agreement but low reference alignment).

    Args:
        embeddings: (n, d) array of observations
        reference: (d,) reference embedding (ground truth)
        threshold: R threshold for gate
        reference_weight: Weight for reference alignment (0-1)

    Returns:
        RGateResult with combined R-value
    """
    # Internal agreement
    E_internal = compute_agreement(embeddings)
    sigma = compute_dispersion(embeddings)

    # Reference alignment
    E_reference = compute_reference_agreement(embeddings, reference)

    # Combined agreement: weighted average
    # When reference_weight=0.5, both components equally weighted
    E_combined = (1 - reference_weight) * E_internal + reference_weight * E_reference

    # R-value with combined agreement
    R = E_combined / (sigma + 1e-10)
    passed = R > threshold

    return RGateResult(
        R_value=R,
        sigma=sigma,
        passed=passed,
        threshold=threshold
    )


def compute_manifold_deviation(
    embedding: np.ndarray,
    reference_set: np.ndarray,
    k: int = 5
) -> float:
    """Compute how far an embedding deviates from the local manifold.

    Uses k-NN to estimate local manifold structure. An embedding that
    deviates from its k nearest neighbors in the reference set is
    likely corrupted.

    Args:
        embedding: (d,) query embedding
        reference_set: (n, d) set of valid reference embeddings
        k: Number of nearest neighbors

    Returns:
        Mean cosine distance to k nearest neighbors (0 = on manifold, 1 = far off)
    """
    # Compute similarities to all reference embeddings
    sims = reference_set @ embedding

    # Get k nearest neighbors (highest similarity)
    k = min(k, len(reference_set))
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]

    # Deviation = 1 - mean similarity to k-NN
    return float(1.0 - np.mean(top_k_sims))


# =============================================================================
# Random Baseline Generation
# =============================================================================

def generate_random_embeddings(n: int, dim: int, seed: int = None) -> np.ndarray:
    """Generate random L2-normalized embeddings.

    Args:
        n: Number of embeddings
        dim: Embedding dimension
        seed: Random seed (optional)

    Returns:
        (n, dim) array of L2-normalized random vectors
    """
    if seed is not None:
        np.random.seed(seed)

    embeddings = np.random.randn(n, dim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def compute_effective_dimensionality(embeddings: np.ndarray) -> float:
    """Compute effective dimensionality (participation ratio) of embeddings.

    Args:
        embeddings: (n, d) L2-normalized embeddings

    Returns:
        Effective dimensionality (Df)
    """
    D2 = squared_distance_matrix(embeddings)
    _, eigenvalues, _ = classical_mds(D2)
    return effective_rank(eigenvalues)


def compute_alpha(eigenvalues: np.ndarray) -> float:
    """Compute power law decay exponent alpha where lambda_k ~ k^(-alpha).

    For healthy semantic embeddings, alpha ~ 0.5 (Riemann critical line).
    Alpha drift indicates structural degradation.

    Args:
        eigenvalues: Sorted eigenvalues (descending)

    Returns:
        Alpha exponent (positive value)
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0.5

    k = np.arange(1, len(ev) + 1)
    n_fit = max(5, len(ev) // 2)
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def get_eigenspectrum(embeddings: np.ndarray) -> np.ndarray:
    """Get eigenvalues from covariance matrix.

    Args:
        embeddings: (n, d) embeddings

    Returns:
        Sorted eigenvalues (descending)
    """
    if len(embeddings) < 2:
        return np.array([1.0])
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def compute_compass_health(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """Compute compass health metrics (alpha, Df, Df*alpha).

    The compass health monitors semantic structure via:
    - Alpha: eigenvalue decay exponent (healthy = 0.5)
    - Df: effective dimensionality
    - Df*alpha: should equal 8e = 21.746 (conservation law)

    Args:
        embeddings: (n, d) embeddings

    Returns:
        Tuple of (alpha, Df, Df*alpha)
    """
    eigenvalues = get_eigenspectrum(embeddings)
    alpha = compute_alpha(eigenvalues)

    D2 = squared_distance_matrix(embeddings)
    _, mds_eigenvalues, _ = classical_mds(D2)
    Df = effective_rank(mds_eigenvalues)

    return alpha, Df, alpha * Df


SEMIOTIC_CONSTANT_8E = 8 * np.e  # 21.746


# =============================================================================
# Statistical Utilities
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Input data array
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level

    Returns:
        Tuple of (statistic, lower_bound, upper_bound)
    """
    n = len(data)
    boot_stats = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_stats.append(statistic(data[idx]))

    boot_stats = np.array(boot_stats)
    alpha = 1 - ci

    return (
        statistic(data),
        np.percentile(boot_stats, 100 * alpha / 2),
        np.percentile(boot_stats, 100 * (1 - alpha / 2))
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Args:
        group1: First group
        group2: Second group

    Returns:
        Cohen's d (>0.8 is large effect)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def mann_whitney_u(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Compute Mann-Whitney U test.

    Args:
        group1: First group
        group2: Second group

    Returns:
        Tuple of (U statistic, p-value)
    """
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(group1, group2, alternative='greater')
    return float(stat), float(p)
