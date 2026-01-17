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
                  - sigma: noise level for gaussian_noise
                  - epsilon: perturbation size for random_direction

    Returns:
        ErrorInjectionResult with original and corrupted embeddings
    """
    corrupted = embedding.copy()
    dim = len(embedding)

    if error_type == "dimension_flip":
        # Flip n_errors dimensions (choose randomly)
        dims_to_flip = np.random.choice(dim, min(n_errors, dim), replace=False)
        for d in dims_to_flip:
            corrupted = flip_dimension_sign(corrupted, d)

    elif error_type == "dimension_zero":
        dims_to_zero = np.random.choice(dim, min(n_errors, dim), replace=False)
        for d in dims_to_zero:
            corrupted[d] = 0.0
        # Re-normalize
        norm = np.linalg.norm(corrupted)
        if norm > 1e-10:
            corrupted = corrupted / norm

    elif error_type == "gaussian_noise":
        sigma = kwargs.get('sigma', 0.1)
        # Apply noise n_errors times (cumulative)
        for _ in range(n_errors):
            corrupted = add_gaussian_noise(corrupted, sigma)

    elif error_type == "random_direction":
        epsilon = kwargs.get('epsilon', 0.1)
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
