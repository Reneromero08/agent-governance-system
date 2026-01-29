#!/usr/bin/env python3
"""
Q25: What Determines Sigma? (R: 1260)

PRE-REGISTRATION:
1. HYPOTHESIS: Sigma is predictable from dataset properties (R^2 > 0.7)
2. PREDICTION: sigma = f(dimensionality, entropy, sample_size)
3. FALSIFICATION: If R^2 < 0.5, sigma is irreducibly empirical
4. DATA: Multiple domains - NLP, market, image, graph, and baselines
5. THRESHOLD: Find predictive formula or prove none exists

METHODOLOGY:
1. Generate 10+ synthetic datasets with known, diverse properties
2. For each dataset, compute the optimal sigma via golden-section search
   (maximizing R stability across bootstrap resamples)
3. Measure: dimensionality (Df), entropy (H), sample size (N), domain type
4. Fit regression: sigma ~ properties
5. Report cross-validated R^2

The sigma parameter appears in the formula:
    R = (E / nabla_H) * sigma^Df

Where sigma controls the scale of the "semantic unit" and Df is the fractal
dimension. This experiment tests whether sigma can be derived from data
properties or must remain an empirical fitting parameter.

CRITICAL FIX (v2): Previous version had sigma hitting the upper boundary (5.0)
for all datasets, yielding no variance. This version:
1. Uses a wider sigma range (0.001 to 100)
2. Computes optimal sigma as the value maximizing R consistency (min CV)
3. Uses log-spaced sigma grid for better coverage
4. Adds diverse datasets with truly different optimal sigmas

Author: Claude Opus 4.5
Date: 2026-01-27 (v2)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Pre-registered thresholds
R2_PREDICTABLE_THRESHOLD = 0.7  # Sigma is predictable if R^2 > 0.7
R2_IRREDUCIBLE_THRESHOLD = 0.5  # Sigma is irreducibly empirical if R^2 < 0.5

# Sigma search range (log-spaced for better coverage)
SIGMA_MIN = 0.001
SIGMA_MAX = 100.0
SIGMA_STEPS = 200

# Bootstrap resamples for stability estimation
N_BOOTSTRAP = 50

# Cross-validation folds
N_FOLDS = 5


@dataclass
class DatasetProperties:
    """Properties of a dataset."""
    name: str
    domain: str
    n_samples: int
    n_dimensions: int

    # Computed properties
    entropy: float = 0.0
    effective_dim: float = 0.0  # Participation ratio
    mean_pairwise_distance: float = 0.0
    std_pairwise_distance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    eigenvalue_ratio: float = 0.0  # ratio of top to sum
    mean_norm: float = 0.0
    std_norm: float = 0.0
    intrinsic_scale: float = 0.0  # std of distances / sqrt(dim)

    # Optimal sigma found via grid search
    optimal_sigma: float = 0.0
    optimal_R_cv: float = 0.0  # CV at optimal (lower is better)
    optimal_R_mean: float = 0.0  # Mean R at optimal


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    r2_train: float
    r2_cv: float  # Cross-validated
    coefficients: Dict[str, float]
    formula: str
    residual_std: float
    predictions: List[float]
    actuals: List[float]
    best_features: List[str]


def to_builtin(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj
    return obj


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def compute_entropy(embeddings: np.ndarray) -> float:
    """Compute entropy from covariance eigenspectrum."""
    centered = embeddings - embeddings.mean(axis=0)
    n = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    # Normalize to probability distribution
    probs = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return float(entropy)


def compute_effective_dim(embeddings: np.ndarray) -> float:
    """Compute effective dimensionality (participation ratio)."""
    centered = embeddings - embeddings.mean(axis=0)
    n = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)
    return float(Df)


def compute_eigenvalue_ratio(embeddings: np.ndarray) -> float:
    """Compute ratio of top eigenvalue to sum (concentration)."""
    centered = embeddings - embeddings.mean(axis=0)
    n = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    return float(eigenvalues[0] / np.sum(eigenvalues))


def compute_pairwise_stats(embeddings: np.ndarray, max_pairs: int = 5000) -> Tuple[float, float]:
    """Compute mean and std of pairwise distances."""
    n = len(embeddings)

    # Subsample for efficiency
    if n * (n - 1) // 2 > max_pairs:
        indices = np.random.choice(n, size=int(np.sqrt(max_pairs * 2)), replace=False)
        embeddings = embeddings[indices]
        n = len(embeddings)

    # Compute pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(d)

    if len(distances) == 0:
        return 0.0, 0.0

    distances = np.array(distances)
    return float(np.mean(distances)), float(np.std(distances))


def compute_skewness(data: np.ndarray) -> float:
    """Compute skewness."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3))


def compute_kurtosis(data: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    m4 = np.mean((data - mean) ** 4)
    m2 = np.mean((data - mean) ** 2)
    if m2 < 1e-10:
        return 0.0
    return float((m4 / (m2 ** 2)) - 3)


# =============================================================================
# R COMPUTATION
# =============================================================================

def compute_R_for_sigma(embeddings: np.ndarray, sigma: float) -> float:
    """
    Compute R score from embeddings with given sigma.

    The key insight: sigma determines the "error tolerance" in the evidence function.
    Lower sigma = stricter, higher sigma = more forgiving.

    R = mean(E(z)) where z = error/sigma, E(z) = exp(-0.5 * z^2)

    We want to find sigma that makes R stable across bootstrap resamples.
    """
    if len(embeddings) < 2:
        return 0.0

    # Compute typical error scale: std of pairwise distances to centroid
    centroid = embeddings.mean(axis=0)
    errors = np.linalg.norm(embeddings - centroid, axis=1)

    # Normalized errors
    z = errors / (sigma + 1e-10)

    # Gaussian kernel evidence
    E_values = np.exp(-0.5 * z ** 2)

    return float(np.mean(E_values))


def compute_R_bootstrap(embeddings: np.ndarray, sigma: float,
                        n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float]:
    """
    Compute R mean and CV across bootstrap resamples.

    Returns (mean_R, cv_R) where cv = std/mean.
    Lower CV means more stable sigma.
    """
    n = len(embeddings)
    R_values = []

    for _ in range(n_bootstrap):
        # Bootstrap resample
        indices = np.random.choice(n, size=n, replace=True)
        resampled = embeddings[indices]
        R = compute_R_for_sigma(resampled, sigma)
        R_values.append(R)

    R_values = np.array(R_values)
    mean_R = np.mean(R_values)
    std_R = np.std(R_values)

    if mean_R < 1e-10:
        return 0.0, 1.0

    cv = std_R / mean_R
    return float(mean_R), float(cv)


def find_optimal_sigma(embeddings: np.ndarray,
                       sigma_min: float = SIGMA_MIN,
                       sigma_max: float = SIGMA_MAX,
                       n_steps: int = SIGMA_STEPS,
                       verbose: bool = False) -> Tuple[float, float, float]:
    """
    Find optimal sigma via grid search.

    Optimal sigma minimizes the coefficient of variation of R across
    bootstrap resamples while maintaining a reasonable R value.

    The key insight: sigma should be on the same scale as the typical
    error in the embeddings. Too small = all E near 0 (unstable),
    too large = all E near 1 (no discrimination).

    Returns: (optimal_sigma, cv_at_optimal, mean_R_at_optimal)
    """
    # Log-spaced grid for better coverage
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_steps)

    best_sigma = sigmas[0]
    best_cv = float('inf')
    best_mean_R = 0.0

    # First pass: find the "sweet spot" range
    # Optimal sigma should give R in [0.2, 0.8] range (meaningful discrimination)
    sweet_spot_sigmas = []

    for sigma in sigmas:
        mean_R, cv = compute_R_bootstrap(embeddings, sigma, n_bootstrap=10)  # Quick pass

        if 0.1 < mean_R < 0.95:
            sweet_spot_sigmas.append((sigma, cv, mean_R))

    if len(sweet_spot_sigmas) == 0:
        # Fallback: use the one with R closest to 0.5
        for sigma in sigmas:
            mean_R, cv = compute_R_bootstrap(embeddings, sigma, n_bootstrap=10)
            if abs(mean_R - 0.5) < abs(best_mean_R - 0.5):
                best_sigma = sigma
                best_cv = cv
                best_mean_R = mean_R
    else:
        # Find best CV among sweet spot candidates
        for sigma, cv, mean_R in sweet_spot_sigmas:
            if cv < best_cv:
                best_sigma = sigma
                best_cv = cv
                best_mean_R = mean_R

    # Refine with more bootstrap samples
    _, final_cv = compute_R_bootstrap(embeddings, best_sigma, n_bootstrap=N_BOOTSTRAP)

    if verbose:
        print(f"  Optimal sigma: {best_sigma:.6f}, CV: {final_cv:.4f}, R: {best_mean_R:.4f}")

    return float(best_sigma), float(final_cv), float(best_mean_R)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_nlp_like(n_samples: int, n_dims: int,
                      concentration: float = 0.5,
                      scale: float = 1.0,
                      seed: int = 42) -> np.ndarray:
    """
    Generate NLP-like embeddings with semantic clustering.

    NLP embeddings typically have:
    - High concentration on a few dimensions
    - Clustered structure (semantic groups)
    - Non-uniform distribution
    - Normalized (unit sphere)
    """
    np.random.seed(seed)

    n_clusters = max(3, n_samples // 20)
    cluster_size = n_samples // n_clusters

    embeddings = []

    # Create cluster centers on unit sphere
    centers = np.random.randn(n_clusters, n_dims)
    for i in range(n_clusters):
        centers[i] /= np.linalg.norm(centers[i])

    for i in range(n_clusters):
        n_in_cluster = cluster_size if i < n_clusters - 1 else n_samples - i * cluster_size

        # Generate points around cluster center with concentration parameter
        noise = np.random.randn(n_in_cluster, n_dims) * concentration * scale
        cluster_points = centers[i] * scale + noise

        embeddings.append(cluster_points)

    return np.vstack(embeddings)


def generate_market_like(n_samples: int, n_dims: int,
                         correlation: float = 0.3,
                         volatility: float = 1.0,
                         seed: int = 42) -> np.ndarray:
    """
    Generate market data-like embeddings.

    Market data typically has:
    - Strong correlations between assets
    - Fat-tailed distributions
    - Time-varying volatility
    """
    np.random.seed(seed)

    # Create correlation structure
    cov = np.eye(n_dims)
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            cov[i, j] = correlation * np.random.uniform(0.5, 1.0)
            cov[j, i] = cov[i, j]

    # Make positive definite
    cov += 0.01 * np.eye(n_dims)
    cov *= volatility ** 2

    # Generate correlated data
    mean = np.zeros(n_dims)
    embeddings = np.random.multivariate_normal(mean, cov, n_samples)

    # Add fat tails (student-t-like)
    scale = np.random.gamma(2, 0.5, n_samples)[:, np.newaxis]
    embeddings = embeddings * scale

    return embeddings


def generate_image_like(n_samples: int, n_dims: int,
                        sparsity: float = 0.3,
                        scale: float = 1.0,
                        seed: int = 42) -> np.ndarray:
    """
    Generate image embedding-like data.

    Image embeddings typically have:
    - Sparse activations
    - ReLU-like distributions (many zeros)
    - Varying importance across dimensions
    """
    np.random.seed(seed)

    embeddings = np.random.randn(n_samples, n_dims) * scale

    # Apply ReLU-like sparsity
    mask = np.random.uniform(0, 1, (n_samples, n_dims)) > sparsity
    embeddings = embeddings * mask

    # Make some dimensions more important (like channels)
    importance = np.random.exponential(1, n_dims)
    embeddings = embeddings * importance

    return embeddings


def generate_graph_like(n_samples: int, n_dims: int,
                        connectivity: float = 0.1,
                        scale: float = 1.0,
                        seed: int = 42) -> np.ndarray:
    """
    Generate graph embedding-like data.

    Graph embeddings typically have:
    - Power-law degree distribution reflected in embeddings
    - Community structure
    - Geometric relationships
    """
    np.random.seed(seed)

    n_communities = max(2, n_samples // 30)
    community_sizes = np.random.multinomial(
        n_samples - n_communities,
        np.ones(n_communities) / n_communities
    ) + 1

    embeddings = []

    # Community centers form a simplex
    centers = np.eye(min(n_communities, n_dims), n_dims) * scale * 3
    if n_communities > n_dims:
        extra = np.random.randn(n_communities - n_dims, n_dims) * scale
        centers = np.vstack([centers, extra])

    for i, size in enumerate(community_sizes):
        # Power-law "degree" distribution within community
        degrees = np.random.pareto(2, size) + 1
        degrees = degrees / np.max(degrees)

        # Embed based on degree and community
        for d in degrees:
            point = centers[i] + np.random.randn(n_dims) * (1 - d) * 0.5 * scale
            # Cross-community connections for high-degree nodes
            if d > 0.7:
                other = np.random.randint(n_communities)
                point += centers[other] * connectivity * (d - 0.7) * 3
            embeddings.append(point)

    return np.array(embeddings)


def generate_random_gaussian(n_samples: int, n_dims: int,
                              scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate purely random Gaussian embeddings (baseline)."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_dims) * scale


def generate_uniform(n_samples: int, n_dims: int,
                     scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate uniform random embeddings."""
    np.random.seed(seed)
    return np.random.uniform(-scale, scale, (n_samples, n_dims))


def generate_multimodal(n_samples: int, n_dims: int,
                        n_modes: int = 5, separation: float = 3.0,
                        seed: int = 42) -> np.ndarray:
    """Generate multimodal (mixture of Gaussians) embeddings."""
    np.random.seed(seed)

    mode_weights = np.random.dirichlet(np.ones(n_modes))
    mode_counts = np.random.multinomial(n_samples, mode_weights)

    embeddings = []
    for i, count in enumerate(mode_counts):
        if count == 0:
            continue
        center = np.random.randn(n_dims) * separation
        spread = np.random.uniform(0.1, 1.0)
        mode_points = center + np.random.randn(count, n_dims) * spread
        embeddings.append(mode_points)

    return np.vstack(embeddings)


def generate_low_rank(n_samples: int, n_dims: int,
                      rank: int = 5, scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate low-rank embeddings (intrinsic dimension << ambient)."""
    np.random.seed(seed)

    # Generate in low-dimensional subspace
    latent = np.random.randn(n_samples, rank) * scale

    # Project to high-dimensional space
    projection = np.random.randn(rank, n_dims) / np.sqrt(rank)

    return latent @ projection


def generate_heavy_tailed(n_samples: int, n_dims: int,
                          df: float = 3.0, scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate heavy-tailed (student-t) embeddings."""
    np.random.seed(seed)

    # Multivariate t-distribution via scale mixture
    chi2_samples = np.random.chisquare(df, n_samples)
    normal_samples = np.random.randn(n_samples, n_dims)

    # Scale by sqrt(df / chi2) to get t-distribution
    t_samples = normal_samples / np.sqrt(chi2_samples / df)[:, np.newaxis]

    return t_samples * scale


def generate_anisotropic(n_samples: int, n_dims: int,
                         anisotropy: float = 10.0, seed: int = 42) -> np.ndarray:
    """Generate anisotropic Gaussian (different variance per dimension)."""
    np.random.seed(seed)

    # Variances follow geometric sequence
    variances = np.array([anisotropy ** (-i / n_dims) for i in range(n_dims)])

    embeddings = np.random.randn(n_samples, n_dims)
    embeddings = embeddings * np.sqrt(variances)

    return embeddings


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def analyze_dataset(embeddings: np.ndarray, name: str, domain: str,
                    verbose: bool = True) -> DatasetProperties:
    """Compute all properties of a dataset."""
    n_samples, n_dims = embeddings.shape

    # Compute all properties
    entropy = compute_entropy(embeddings)
    effective_dim = compute_effective_dim(embeddings)
    eigenvalue_ratio = compute_eigenvalue_ratio(embeddings)
    mean_dist, std_dist = compute_pairwise_stats(embeddings)

    # Norms
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))

    # Intrinsic scale (normalized by sqrt(dim))
    intrinsic_scale = std_dist / np.sqrt(n_dims) if n_dims > 0 else 0.0

    # Flatten for skewness/kurtosis
    flat = embeddings.flatten()
    skewness = compute_skewness(flat)
    kurtosis = compute_kurtosis(flat)

    # Find optimal sigma
    optimal_sigma, optimal_cv, optimal_mean_R = find_optimal_sigma(
        embeddings, verbose=verbose
    )

    return DatasetProperties(
        name=name,
        domain=domain,
        n_samples=n_samples,
        n_dimensions=n_dims,
        entropy=entropy,
        effective_dim=effective_dim,
        mean_pairwise_distance=mean_dist,
        std_pairwise_distance=std_dist,
        skewness=skewness,
        kurtosis=kurtosis,
        eigenvalue_ratio=eigenvalue_ratio,
        mean_norm=mean_norm,
        std_norm=std_norm,
        intrinsic_scale=intrinsic_scale,
        optimal_sigma=optimal_sigma,
        optimal_R_cv=optimal_cv,
        optimal_R_mean=optimal_mean_R
    )


# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================

def fit_linear_regression(X: np.ndarray, y: np.ndarray,
                          feature_names: List[str]) -> Tuple[np.ndarray, float]:
    """Fit linear regression and return coefficients and R^2."""
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Fit using pseudo-inverse
    try:
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        coeffs = np.zeros(X_with_intercept.shape[1])

    # Predictions and R^2
    predictions = X_with_intercept @ coeffs
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0

    return coeffs, float(r2)


def cross_validate_regression(X: np.ndarray, y: np.ndarray,
                               n_folds: int = N_FOLDS) -> float:
    """Cross-validated R^2."""
    n = len(y)
    if n < n_folds * 2:
        return 0.0

    fold_size = n // n_folds
    r2_values = []

    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(n)

    for i in range(n_folds):
        # Test fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Fit on train
        X_train_int = np.column_stack([np.ones(len(X_train)), X_train])
        try:
            coeffs = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(X_train_int.shape[1])

        # Predict on test
        X_test_int = np.column_stack([np.ones(len(X_test)), X_test])
        predictions = X_test_int @ coeffs

        # R^2 on test
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
        r2_values.append(max(0.0, r2))  # Clamp negative R^2

    return float(np.mean(r2_values))


def build_feature_matrix(datasets: List[DatasetProperties],
                         features: List[str]) -> np.ndarray:
    """Build feature matrix from dataset properties."""
    X = []
    for ds in datasets:
        row = []
        for f in features:
            if f == "log_n_samples":
                row.append(np.log(ds.n_samples + 1))
            elif f == "log_n_dimensions":
                row.append(np.log(ds.n_dimensions + 1))
            elif f == "entropy":
                row.append(ds.entropy)
            elif f == "effective_dim":
                row.append(ds.effective_dim)
            elif f == "log_effective_dim":
                row.append(np.log(ds.effective_dim + 1))
            elif f == "mean_pairwise_distance":
                row.append(ds.mean_pairwise_distance)
            elif f == "std_pairwise_distance":
                row.append(ds.std_pairwise_distance)
            elif f == "skewness":
                row.append(ds.skewness)
            elif f == "kurtosis":
                row.append(ds.kurtosis)
            elif f == "eigenvalue_ratio":
                row.append(ds.eigenvalue_ratio)
            elif f == "entropy_over_dim":
                row.append(ds.entropy / (np.log(ds.n_dimensions) + 1e-10))
            elif f == "mean_norm":
                row.append(ds.mean_norm)
            elif f == "std_norm":
                row.append(ds.std_norm)
            elif f == "intrinsic_scale":
                row.append(ds.intrinsic_scale)
            elif f == "log_intrinsic_scale":
                row.append(np.log(ds.intrinsic_scale + 1e-6))
            elif f == "log_mean_dist":
                row.append(np.log(ds.mean_pairwise_distance + 1e-6))
            elif f == "log_std_dist":
                row.append(np.log(ds.std_pairwise_distance + 1e-6))
            else:
                row.append(0.0)
        X.append(row)

    return np.array(X)


def analyze_sigma_predictability(datasets: List[DatasetProperties],
                                  verbose: bool = True) -> RegressionResult:
    """
    Main analysis: can sigma be predicted from dataset properties?

    Note: We predict log(sigma) since sigma is positive and spans orders of magnitude.
    """
    # Extract target (log-sigma for better regression)
    y_raw = np.array([ds.optimal_sigma for ds in datasets])

    # Check for zero/near-zero variance
    if np.std(y_raw) < 1e-10:
        print("\nWARNING: No variance in sigma values! All datasets have same optimal sigma.")
        return RegressionResult(
            r2_train=0.0,
            r2_cv=0.0,
            coefficients={"intercept": float(np.mean(y_raw))},
            formula=f"sigma = {np.mean(y_raw):.4f} (constant)",
            residual_std=0.0,
            predictions=y_raw.tolist(),
            actuals=y_raw.tolist(),
            best_features=[]
        )

    # Use log(sigma) as target
    y = np.log(y_raw + 1e-10)

    # Feature sets to try
    feature_sets = [
        # Scale-based (most likely to predict sigma)
        ["log_mean_dist", "log_std_dist"],
        ["intrinsic_scale", "log_intrinsic_scale"],
        ["mean_pairwise_distance", "std_pairwise_distance"],

        # Dimensionality-based
        ["log_n_samples", "log_n_dimensions"],
        ["log_effective_dim", "eigenvalue_ratio"],

        # Spectral properties
        ["entropy", "effective_dim", "eigenvalue_ratio"],

        # Distribution properties
        ["skewness", "kurtosis", "mean_norm", "std_norm"],

        # Combined: scale + structure
        ["log_mean_dist", "log_effective_dim", "eigenvalue_ratio"],
        ["intrinsic_scale", "entropy", "log_n_dimensions"],

        # Kitchen sink
        ["log_n_samples", "log_n_dimensions", "entropy", "log_effective_dim",
         "log_mean_dist", "log_std_dist", "skewness", "kurtosis",
         "eigenvalue_ratio", "intrinsic_scale"]
    ]

    best_r2_cv = -np.inf
    best_features = None
    best_coeffs = None
    best_r2_train = 0.0
    best_X = None

    if verbose:
        print("\n--- Feature Set Comparison ---")
        print(f"{'Features':<55} {'R2_train':>10} {'R2_cv':>10}")
        print("-" * 75)

    for features in feature_sets:
        X = build_feature_matrix(datasets, features)

        # Standardize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-10
        X_scaled = (X - X_mean) / X_std

        coeffs, r2_train = fit_linear_regression(X_scaled, y, features)
        r2_cv = cross_validate_regression(X_scaled, y)

        if verbose:
            feat_str = ", ".join(features[:3]) + ("..." if len(features) > 3 else "")
            print(f"{feat_str:<55} {r2_train:>10.4f} {r2_cv:>10.4f}")

        if r2_cv > best_r2_cv:
            best_r2_cv = r2_cv
            best_r2_train = r2_train
            best_features = features
            best_coeffs = coeffs
            best_X = X_scaled
            best_X_mean = X_mean
            best_X_std = X_std

    # Build coefficient dictionary
    coeff_dict = {"intercept": float(best_coeffs[0])}
    for i, f in enumerate(best_features):
        coeff_dict[f] = float(best_coeffs[i + 1])

    # Build formula string
    formula_parts = [f"log(sigma) = {best_coeffs[0]:.4f}"]
    for i, f in enumerate(best_features):
        sign = "+" if best_coeffs[i + 1] >= 0 else ""
        formula_parts.append(f"{sign}{best_coeffs[i + 1]:.4f}*{f}")
    formula = " ".join(formula_parts)

    # Predictions (in log space, then convert back)
    X_with_int = np.column_stack([np.ones(len(best_X)), best_X])
    log_predictions = X_with_int @ best_coeffs
    predictions = np.exp(log_predictions)

    # Residual std (in log space)
    residuals = y - log_predictions
    residual_std = float(np.std(residuals))

    return RegressionResult(
        r2_train=best_r2_train,
        r2_cv=best_r2_cv,
        coefficients=coeff_dict,
        formula=formula,
        residual_std=residual_std,
        predictions=predictions.tolist(),
        actuals=y_raw.tolist(),
        best_features=best_features
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(verbose: bool = True) -> Dict[str, Any]:
    """Run the full Q25 experiment."""

    if verbose:
        print("=" * 80)
        print("Q25: WHAT DETERMINES SIGMA? (v2)")
        print("=" * 80)
        print("\nPRE-REGISTRATION:")
        print(f"  HYPOTHESIS: Sigma predictable from dataset properties (R^2 > {R2_PREDICTABLE_THRESHOLD})")
        print(f"  FALSIFICATION: If R^2 < {R2_IRREDUCIBLE_THRESHOLD}, sigma is irreducibly empirical")
        print(f"  DATA: 15+ synthetic datasets across multiple domains")
        print(f"  THRESHOLD: Find predictive formula or prove none exists")

    # Define datasets with VARIED scales to get different optimal sigmas
    dataset_configs = [
        # NLP-like with different scales
        ("nlp_tight", "nlp", generate_nlp_like,
         {"n_samples": 300, "n_dims": 128, "concentration": 0.1, "scale": 1.0, "seed": 1}),
        ("nlp_spread", "nlp", generate_nlp_like,
         {"n_samples": 300, "n_dims": 128, "concentration": 0.5, "scale": 3.0, "seed": 2}),
        ("nlp_wide", "nlp", generate_nlp_like,
         {"n_samples": 300, "n_dims": 256, "concentration": 0.8, "scale": 5.0, "seed": 3}),

        # Market-like with different volatilities
        ("market_calm", "market", generate_market_like,
         {"n_samples": 400, "n_dims": 50, "correlation": 0.3, "volatility": 0.5, "seed": 4}),
        ("market_volatile", "market", generate_market_like,
         {"n_samples": 400, "n_dims": 100, "correlation": 0.5, "volatility": 2.0, "seed": 5}),
        ("market_extreme", "market", generate_market_like,
         {"n_samples": 400, "n_dims": 100, "correlation": 0.7, "volatility": 5.0, "seed": 6}),

        # Image-like with different sparsities
        ("image_dense", "image", generate_image_like,
         {"n_samples": 300, "n_dims": 128, "sparsity": 0.1, "scale": 1.0, "seed": 7}),
        ("image_sparse", "image", generate_image_like,
         {"n_samples": 300, "n_dims": 256, "sparsity": 0.6, "scale": 2.0, "seed": 8}),

        # Graph-like with different connectivity
        ("graph_dense", "graph", generate_graph_like,
         {"n_samples": 300, "n_dims": 64, "connectivity": 0.3, "scale": 1.0, "seed": 9}),
        ("graph_sparse", "graph", generate_graph_like,
         {"n_samples": 300, "n_dims": 128, "connectivity": 0.05, "scale": 3.0, "seed": 10}),

        # Baselines with different scales
        ("gaussian_small", "baseline", generate_random_gaussian,
         {"n_samples": 400, "n_dims": 64, "scale": 0.5, "seed": 11}),
        ("gaussian_large", "baseline", generate_random_gaussian,
         {"n_samples": 400, "n_dims": 128, "scale": 5.0, "seed": 12}),
        ("uniform_small", "baseline", generate_uniform,
         {"n_samples": 400, "n_dims": 64, "scale": 0.3, "seed": 13}),
        ("uniform_large", "baseline", generate_uniform,
         {"n_samples": 400, "n_dims": 128, "scale": 3.0, "seed": 14}),

        # Multimodal with different separations
        ("multimodal_close", "structure", generate_multimodal,
         {"n_samples": 400, "n_dims": 64, "n_modes": 5, "separation": 1.0, "seed": 15}),
        ("multimodal_far", "structure", generate_multimodal,
         {"n_samples": 400, "n_dims": 64, "n_modes": 5, "separation": 8.0, "seed": 16}),

        # Low-rank datasets
        ("lowrank_tight", "structure", generate_low_rank,
         {"n_samples": 400, "n_dims": 128, "rank": 3, "scale": 0.5, "seed": 17}),
        ("lowrank_spread", "structure", generate_low_rank,
         {"n_samples": 400, "n_dims": 256, "rank": 10, "scale": 3.0, "seed": 18}),

        # Heavy-tailed distributions
        ("heavy_light", "tails", generate_heavy_tailed,
         {"n_samples": 400, "n_dims": 64, "df": 10.0, "scale": 1.0, "seed": 19}),
        ("heavy_extreme", "tails", generate_heavy_tailed,
         {"n_samples": 400, "n_dims": 128, "df": 2.5, "scale": 2.0, "seed": 20}),

        # Anisotropic
        ("aniso_mild", "structure", generate_anisotropic,
         {"n_samples": 400, "n_dims": 64, "anisotropy": 3.0, "seed": 21}),
        ("aniso_extreme", "structure", generate_anisotropic,
         {"n_samples": 400, "n_dims": 128, "anisotropy": 100.0, "seed": 22}),
    ]

    if verbose:
        print(f"\n--- Analyzing {len(dataset_configs)} datasets ---")

    # Analyze each dataset
    datasets = []
    for name, domain, gen_func, params in dataset_configs:
        if verbose:
            print(f"\n{name} ({domain}):")

        embeddings = gen_func(**params)
        props = analyze_dataset(embeddings, name, domain, verbose=verbose)
        datasets.append(props)

        if verbose:
            print(f"  Shape: {params['n_samples']} x {params.get('n_dims', 'varies')}")
            print(f"  Entropy: {props.entropy:.4f}, Effective dim: {props.effective_dim:.2f}")
            print(f"  Mean pairwise dist: {props.mean_pairwise_distance:.4f}")

    # Show sigma distribution
    sigmas = [ds.optimal_sigma for ds in datasets]
    if verbose:
        print("\n" + "=" * 80)
        print("SIGMA DISTRIBUTION")
        print("=" * 80)
        print(f"  Min: {min(sigmas):.6f}")
        print(f"  Max: {max(sigmas):.6f}")
        print(f"  Mean: {np.mean(sigmas):.6f}")
        print(f"  Std: {np.std(sigmas):.6f}")
        print(f"  Range ratio: {max(sigmas)/min(sigmas):.2f}x")

    if verbose:
        print("\n" + "=" * 80)
        print("REGRESSION ANALYSIS")
        print("=" * 80)

    # Run regression analysis
    regression = analyze_sigma_predictability(datasets, verbose=verbose)

    # Determine verdict
    if regression.r2_cv >= R2_PREDICTABLE_THRESHOLD:
        verdict = "SIGMA_PREDICTABLE"
        verdict_detail = f"R^2_cv = {regression.r2_cv:.4f} >= {R2_PREDICTABLE_THRESHOLD}"
    elif regression.r2_cv < R2_IRREDUCIBLE_THRESHOLD:
        verdict = "SIGMA_IRREDUCIBLY_EMPIRICAL"
        verdict_detail = f"R^2_cv = {regression.r2_cv:.4f} < {R2_IRREDUCIBLE_THRESHOLD}"
    else:
        verdict = "SIGMA_PARTIALLY_PREDICTABLE"
        verdict_detail = f"R^2_cv = {regression.r2_cv:.4f} in [{R2_IRREDUCIBLE_THRESHOLD}, {R2_PREDICTABLE_THRESHOLD})"

    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\nBest R^2 (training): {regression.r2_train:.4f}")
        print(f"Best R^2 (cross-validated): {regression.r2_cv:.4f}")
        print(f"Best features: {regression.best_features}")
        print(f"Residual std (log-space): {regression.residual_std:.4f}")
        print(f"\nBest formula: {regression.formula}")

        print(f"\n--- Predicted vs Actual Sigma ---")
        print(f"{'Dataset':<25} {'Actual':>12} {'Predicted':>12} {'Ratio':>12}")
        print("-" * 61)
        for i, ds in enumerate(datasets):
            ratio = regression.predictions[i] / ds.optimal_sigma if ds.optimal_sigma > 0 else 0
            print(f"{ds.name:<25} {ds.optimal_sigma:>12.6f} {regression.predictions[i]:>12.6f} {ratio:>12.2f}")

    # Domain-specific analysis
    domain_sigmas = {}
    for ds in datasets:
        if ds.domain not in domain_sigmas:
            domain_sigmas[ds.domain] = []
        domain_sigmas[ds.domain].append(ds.optimal_sigma)

    domain_summary = {}
    for domain, sigmas_list in domain_sigmas.items():
        domain_summary[domain] = {
            "mean_sigma": float(np.mean(sigmas_list)),
            "std_sigma": float(np.std(sigmas_list)),
            "n_datasets": len(sigmas_list)
        }

    if verbose:
        print(f"\n--- Domain-Specific Sigma Summary ---")
        for domain, stats in domain_summary.items():
            print(f"  {domain}: mean={stats['mean_sigma']:.6f}, std={stats['std_sigma']:.6f} (n={stats['n_datasets']})")

    # Final verdict
    if verbose:
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(f"\n{verdict}: {verdict_detail}")

        if verdict == "SIGMA_PREDICTABLE":
            print("\nSigma CAN be predicted from dataset properties!")
            print(f"Best predictive features: {regression.best_features}")
            print(f"Formula: {regression.formula}")
        elif verdict == "SIGMA_IRREDUCIBLY_EMPIRICAL":
            print("\nSigma CANNOT be reliably predicted from dataset properties.")
            print("It must be determined empirically for each dataset.")
            print("This supports the view that sigma is a fundamental tuning parameter.")
        else:
            print("\nSigma is PARTIALLY predictable.")
            print("Some variance explained, but significant empirical component remains.")
            print("Practical approach: use prediction as initial guess, then fine-tune.")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "Q25_what_determines_sigma_v2",
        "pre_registration": {
            "hypothesis": f"Sigma predictable (R^2 > {R2_PREDICTABLE_THRESHOLD})",
            "falsification": f"Sigma irreducibly empirical (R^2 < {R2_IRREDUCIBLE_THRESHOLD})",
            "data": f"{len(dataset_configs)} synthetic datasets across {len(set(d[1] for d in dataset_configs))} domains"
        },
        "n_datasets": len(datasets),
        "sigma_distribution": {
            "min": float(min(sigmas)),
            "max": float(max(sigmas)),
            "mean": float(np.mean(sigmas)),
            "std": float(np.std(sigmas)),
            "range_ratio": float(max(sigmas)/min(sigmas)) if min(sigmas) > 0 else None
        },
        "datasets": [],
        "regression": {
            "r2_train": regression.r2_train,
            "r2_cv": regression.r2_cv,
            "best_features": regression.best_features,
            "coefficients": regression.coefficients,
            "formula": regression.formula,
            "residual_std": regression.residual_std,
            "predictions": regression.predictions,
            "actuals": regression.actuals
        },
        "domain_summary": domain_summary,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "passes_hypothesis": regression.r2_cv >= R2_PREDICTABLE_THRESHOLD,
        "falsified": regression.r2_cv < R2_IRREDUCIBLE_THRESHOLD
    }

    for ds in datasets:
        output["datasets"].append({
            "name": ds.name,
            "domain": ds.domain,
            "n_samples": ds.n_samples,
            "n_dimensions": ds.n_dimensions,
            "entropy": ds.entropy,
            "effective_dim": ds.effective_dim,
            "eigenvalue_ratio": ds.eigenvalue_ratio,
            "mean_pairwise_distance": ds.mean_pairwise_distance,
            "std_pairwise_distance": ds.std_pairwise_distance,
            "skewness": ds.skewness,
            "kurtosis": ds.kurtosis,
            "mean_norm": ds.mean_norm,
            "std_norm": ds.std_norm,
            "intrinsic_scale": ds.intrinsic_scale,
            "optimal_sigma": ds.optimal_sigma,
            "optimal_R_cv": ds.optimal_R_cv,
            "optimal_R_mean": ds.optimal_R_mean
        })

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_experiment(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "q25_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
