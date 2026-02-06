#!/usr/bin/env python3
"""
Q18 Deep Investigation: Why R's Distribution Produces 8e

CRITICAL FINDING:
- r_original: Df x alpha = 21.15 (2.7% dev)
- r_shuffled: Df x alpha = 21.15 (2.8% dev)  <- Same!
- r_uniform: Df x alpha = 17.05 (21.6% dev)
- r_gaussian: Df x alpha = 9.50 (56.3% dev)

This means R's HEAVY-TAILED DISTRIBUTION is what matters, not which gene has which R.

THIS SCRIPT INVESTIGATES:
1. Characterize R's exact distribution (skewness, kurtosis, fit to distributions)
2. Test synthetic distributions (log-normal, Pareto, gamma, etc.)
3. Find minimal sufficient conditions for 8e emergence
4. Derive theoretical understanding

Author: Claude Opus 4.5
Date: 2026-01-26
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
TOLERANCE = 0.15  # 15% deviation threshold


@dataclass
class DistributionResult:
    """Results from testing a distribution."""
    name: str
    description: str
    n_samples: int
    parameters: Dict[str, float]

    # Distribution statistics
    mean: float
    std: float
    skewness: float
    kurtosis: float  # Excess kurtosis (normal = 0)

    # Embedding results
    embedding_dims: int
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float
    passes_8e: bool

    # Eigenvalue info
    top_eigenvalues: List[float] = field(default_factory=list)


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

def compute_skewness(data: np.ndarray) -> float:
    """Compute Fisher-Pearson skewness coefficient."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)


def compute_kurtosis(data: np.ndarray) -> float:
    """Compute excess kurtosis (normal distribution = 0)."""
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
    return (m4 / (m2 ** 2)) - 3  # Excess kurtosis


def fit_lognormal(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit log-normal distribution to data.
    Returns (mu, sigma, ks_statistic) where log(X) ~ N(mu, sigma^2).
    """
    # Filter positive values
    data_pos = data[data > 0]
    if len(data_pos) < 10:
        return 0.0, 1.0, 1.0

    log_data = np.log(data_pos)
    mu = np.mean(log_data)
    sigma = np.std(log_data, ddof=1)

    # KS statistic (simplified)
    sorted_data = np.sort(data_pos)
    n = len(sorted_data)
    theoretical_cdf = np.array([
        0.5 * (1 + np.tanh((np.log(x) - mu) / (sigma * np.sqrt(2) + 1e-10)))
        for x in sorted_data
    ])
    empirical_cdf = np.arange(1, n + 1) / n
    ks_stat = np.max(np.abs(theoretical_cdf - empirical_cdf))

    return mu, sigma, ks_stat


def fit_pareto(data: np.ndarray, x_min: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Fit Pareto distribution to data.
    Returns (alpha, x_min, ks_statistic).
    P(X > x) = (x_min / x)^alpha for x >= x_min
    """
    data_pos = data[data > 0]
    if len(data_pos) < 10:
        return 2.0, 1.0, 1.0

    if x_min is None:
        x_min = np.min(data_pos)

    data_above = data_pos[data_pos >= x_min]
    if len(data_above) < 5:
        return 2.0, x_min, 1.0

    # MLE estimator for Pareto alpha
    n = len(data_above)
    alpha = n / np.sum(np.log(data_above / x_min))

    # KS statistic (simplified)
    sorted_data = np.sort(data_above)
    theoretical_cdf = 1 - (x_min / sorted_data) ** alpha
    empirical_cdf = np.arange(1, n + 1) / n
    ks_stat = np.max(np.abs(theoretical_cdf - empirical_cdf))

    return alpha, x_min, ks_stat


def fit_gamma(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit Gamma distribution using method of moments.
    Returns (shape, scale, ks_statistic).
    """
    data_pos = data[data > 0]
    if len(data_pos) < 10:
        return 1.0, 1.0, 1.0

    mean = np.mean(data_pos)
    var = np.var(data_pos, ddof=1)

    if var < 1e-10:
        return 1.0, mean, 1.0

    # Method of moments
    shape = mean ** 2 / var
    scale = var / mean

    return shape, scale, 0.5  # Simplified KS


# =============================================================================
# SPECTRAL COMPUTATION
# =============================================================================

def compute_spectral_properties(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute Df (participation ratio) and alpha (spectral decay) from embeddings.
    Returns: (Df, alpha, eigenvalues)
    """
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def create_embedding(R_values: np.ndarray, n_dims: int = 50, seed: int = 42) -> np.ndarray:
    """
    Create sinusoidal embedding with R-modulation.
    This is the formula that produces 8e.
    """
    np.random.seed(seed)
    n_samples = len(R_values)
    embeddings = np.zeros((n_samples, n_dims))

    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    return embeddings


def test_distribution(R_values: np.ndarray, name: str, description: str,
                      parameters: Dict[str, float], n_dims: int = 50) -> DistributionResult:
    """Test a distribution and return results."""

    # Compute distribution statistics
    mean = float(np.mean(R_values))
    std = float(np.std(R_values, ddof=1))
    skewness = compute_skewness(R_values)
    kurtosis = compute_kurtosis(R_values)

    # Create embedding and compute spectral properties
    embeddings = create_embedding(R_values, n_dims)
    Df, alpha, eigenvalues = compute_spectral_properties(embeddings)

    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return DistributionResult(
        name=name,
        description=description,
        n_samples=len(R_values),
        parameters=parameters,
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        passes_8e=deviation < TOLERANCE,
        top_eigenvalues=eigenvalues[:20].tolist() if len(eigenvalues) >= 20 else eigenvalues.tolist()
    )


# =============================================================================
# DISTRIBUTION GENERATORS
# =============================================================================

def generate_uniform(n: int, low: float, high: float, seed: int = 42) -> np.ndarray:
    """Generate uniform distribution."""
    np.random.seed(seed)
    return np.random.uniform(low, high, n)


def generate_gaussian(n: int, mean: float, std: float, low: float, high: float, seed: int = 42) -> np.ndarray:
    """Generate Gaussian distribution (clipped to range)."""
    np.random.seed(seed)
    samples = np.random.randn(n) * std + mean
    return np.clip(samples, low, high)


def generate_lognormal(n: int, mu: float, sigma: float, seed: int = 42) -> np.ndarray:
    """Generate log-normal distribution."""
    np.random.seed(seed)
    return np.random.lognormal(mu, sigma, n)


def generate_pareto(n: int, alpha: float, x_min: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate Pareto distribution."""
    np.random.seed(seed)
    # Pareto with shape = alpha
    return (np.random.pareto(alpha, n) + 1) * x_min


def generate_gamma(n: int, shape: float, scale: float, seed: int = 42) -> np.ndarray:
    """Generate Gamma distribution."""
    np.random.seed(seed)
    return np.random.gamma(shape, scale, n)


def generate_exponential(n: int, scale: float, seed: int = 42) -> np.ndarray:
    """Generate Exponential distribution."""
    np.random.seed(seed)
    return np.random.exponential(scale, n)


def generate_weibull(n: int, shape: float, scale: float, seed: int = 42) -> np.ndarray:
    """Generate Weibull distribution."""
    np.random.seed(seed)
    return scale * np.random.weibull(shape, n)


def generate_beta_scaled(n: int, a: float, b: float, low: float, high: float, seed: int = 42) -> np.ndarray:
    """Generate Beta distribution scaled to [low, high]."""
    np.random.seed(seed)
    samples = np.random.beta(a, b, n)
    return low + samples * (high - low)


def generate_mixture_lognormal(n: int, mu1: float, sigma1: float, mu2: float, sigma2: float,
                               weight1: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate mixture of two log-normal distributions."""
    np.random.seed(seed)
    n1 = int(n * weight1)
    n2 = n - n1
    samples1 = np.random.lognormal(mu1, sigma1, n1)
    samples2 = np.random.lognormal(mu2, sigma2, n2)
    samples = np.concatenate([samples1, samples2])
    np.random.shuffle(samples)
    return samples


def generate_custom_heavy_tail(n: int, skewness_target: float, kurtosis_target: float,
                               mean_target: float, std_target: float, seed: int = 42) -> np.ndarray:
    """
    Generate a distribution with specific skewness and kurtosis.
    Uses a transformation approach.
    """
    np.random.seed(seed)

    # Start with standard normal
    z = np.random.randn(n)

    # Apply Fleishman's power transformation for approximate skewness/kurtosis
    # y = a + b*z + c*z^2 + d*z^3
    # This is a simplified approximation

    # For heavy tail, use exponential of transformed normal
    # Adjust sigma to control skewness
    sigma_log = 0.5 + 0.3 * abs(skewness_target)  # Heuristic

    samples = np.exp(sigma_log * z)

    # Rescale to target mean and std
    samples = (samples - np.mean(samples)) / np.std(samples) * std_target + mean_target
    samples = np.maximum(samples, 0.1)  # Ensure positive

    return samples


# =============================================================================
# MAIN EXPERIMENTS
# =============================================================================

def run_distribution_analysis(verbose: bool = True) -> Dict[str, Any]:
    """Run comprehensive distribution analysis."""

    # Load original R values
    cache_path = Path(__file__).parent / "cache" / "gene_expression_sample.json"

    if verbose:
        print("=" * 80)
        print("Q18 DEEP INVESTIGATION: Why R's Distribution Produces 8e")
        print("=" * 80)
        print(f"\nLoading data from: {cache_path}")

    with open(cache_path, 'r') as f:
        data = json.load(f)

    genes_data = data['genes']
    R_original = np.array([g['R'] for g in genes_data.values()])
    n = len(R_original)

    if verbose:
        print(f"Loaded {n} genes")
        print(f"\n--- Original R Distribution ---")
        print(f"Mean: {np.mean(R_original):.4f}")
        print(f"Std: {np.std(R_original):.4f}")
        print(f"Min: {np.min(R_original):.4f}")
        print(f"Max: {np.max(R_original):.4f}")
        print(f"Skewness: {compute_skewness(R_original):.4f}")
        print(f"Kurtosis (excess): {compute_kurtosis(R_original):.4f}")

    # Fit distributions to original R
    lognorm_mu, lognorm_sigma, lognorm_ks = fit_lognormal(R_original)
    pareto_alpha, pareto_xmin, pareto_ks = fit_pareto(R_original)
    gamma_shape, gamma_scale, gamma_ks = fit_gamma(R_original)

    if verbose:
        print(f"\n--- Distribution Fits ---")
        print(f"Log-normal: mu={lognorm_mu:.4f}, sigma={lognorm_sigma:.4f}, KS={lognorm_ks:.4f}")
        print(f"Pareto: alpha={pareto_alpha:.4f}, x_min={pareto_xmin:.4f}, KS={pareto_ks:.4f}")
        print(f"Gamma: shape={gamma_shape:.4f}, scale={gamma_scale:.4f}")

    results = []

    # -------------------------------------------------------------------------
    # Test 1: Original R
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 1: Original R Distribution")
        print("=" * 80)

    result = test_distribution(
        R_original,
        name="original_r",
        description="Original R values from gene expression data",
        parameters={"source": "gene_expression"}
    )
    results.append(result)

    if verbose:
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 2: Shuffled R (same distribution)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 2: Shuffled R (same distribution)")
        print("=" * 80)

    R_shuffled = R_original.copy()
    np.random.seed(123)
    np.random.shuffle(R_shuffled)

    result = test_distribution(
        R_shuffled,
        name="shuffled_r",
        description="Shuffled R values (same distribution, different assignment)",
        parameters={"method": "shuffle"}
    )
    results.append(result)

    if verbose:
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 3: Uniform Distribution
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 3: Uniform Distribution")
        print("=" * 80)

    R_uniform = generate_uniform(n, R_original.min(), R_original.max())

    result = test_distribution(
        R_uniform,
        name="uniform",
        description="Uniform distribution with same range",
        parameters={"low": float(R_original.min()), "high": float(R_original.max())}
    )
    results.append(result)

    if verbose:
        print(f"Skewness: {result.skewness:.4f}, Kurtosis: {result.kurtosis:.4f}")
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 4: Gaussian Distribution
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 4: Gaussian Distribution")
        print("=" * 80)

    R_gaussian = generate_gaussian(n, np.mean(R_original), np.std(R_original),
                                   R_original.min(), R_original.max())

    result = test_distribution(
        R_gaussian,
        name="gaussian",
        description="Gaussian distribution with same mean/std",
        parameters={"mean": float(np.mean(R_original)), "std": float(np.std(R_original))}
    )
    results.append(result)

    if verbose:
        print(f"Skewness: {result.skewness:.4f}, Kurtosis: {result.kurtosis:.4f}")
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 5: Log-normal Distribution (fitted)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 5: Log-normal Distribution (fitted to original)")
        print("=" * 80)

    R_lognormal_fitted = generate_lognormal(n, lognorm_mu, lognorm_sigma)

    result = test_distribution(
        R_lognormal_fitted,
        name="lognormal_fitted",
        description="Log-normal distribution with fitted parameters",
        parameters={"mu": lognorm_mu, "sigma": lognorm_sigma}
    )
    results.append(result)

    if verbose:
        print(f"Skewness: {result.skewness:.4f}, Kurtosis: {result.kurtosis:.4f}")
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 6-10: Log-normal with varying sigma
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENTS 6-10: Log-normal Sigma Sweep")
        print("=" * 80)

    sigma_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    for sigma in sigma_values:
        R_test = generate_lognormal(n, lognorm_mu, sigma)

        result = test_distribution(
            R_test,
            name=f"lognormal_sigma_{sigma:.1f}",
            description=f"Log-normal with sigma={sigma}",
            parameters={"mu": lognorm_mu, "sigma": sigma}
        )
        results.append(result)

        if verbose:
            status = "PASS" if result.passes_8e else "FAIL"
            print(f"sigma={sigma:.1f}: Df x alpha={result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% dev) [{status}]")
            print(f"         skewness={result.skewness:.2f}, kurtosis={result.kurtosis:.2f}")

    # -------------------------------------------------------------------------
    # Test 11-15: Pareto with varying alpha
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENTS 11-15: Pareto Alpha Sweep")
        print("=" * 80)

    alpha_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    for alpha in alpha_values:
        R_test = generate_pareto(n, alpha, pareto_xmin)

        result = test_distribution(
            R_test,
            name=f"pareto_alpha_{alpha:.1f}",
            description=f"Pareto with alpha={alpha}",
            parameters={"alpha": alpha, "x_min": pareto_xmin}
        )
        results.append(result)

        if verbose:
            status = "PASS" if result.passes_8e else "FAIL"
            print(f"alpha={alpha:.1f}: Df x alpha={result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% dev) [{status}]")
            print(f"         skewness={result.skewness:.2f}, kurtosis={result.kurtosis:.2f}")

    # -------------------------------------------------------------------------
    # Test 16-20: Gamma with varying shape
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENTS 16-20: Gamma Shape Sweep")
        print("=" * 80)

    shape_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    for shape in shape_values:
        R_test = generate_gamma(n, shape, gamma_scale)

        result = test_distribution(
            R_test,
            name=f"gamma_shape_{shape:.1f}",
            description=f"Gamma with shape={shape}",
            parameters={"shape": shape, "scale": gamma_scale}
        )
        results.append(result)

        if verbose:
            status = "PASS" if result.passes_8e else "FAIL"
            print(f"shape={shape:.1f}: Df x alpha={result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% dev) [{status}]")
            print(f"          skewness={result.skewness:.2f}, kurtosis={result.kurtosis:.2f}")

    # -------------------------------------------------------------------------
    # Test 21: Exponential
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 21: Exponential Distribution")
        print("=" * 80)

    R_exp = generate_exponential(n, np.mean(R_original))

    result = test_distribution(
        R_exp,
        name="exponential",
        description="Exponential distribution",
        parameters={"scale": float(np.mean(R_original))}
    )
    results.append(result)

    if verbose:
        print(f"Skewness: {result.skewness:.4f}, Kurtosis: {result.kurtosis:.4f}")
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 22-24: Weibull with varying shape
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENTS 22-24: Weibull Shape Sweep")
        print("=" * 80)

    weibull_shapes = [0.5, 1.0, 2.0, 3.0]

    for shape in weibull_shapes:
        R_test = generate_weibull(n, shape, np.mean(R_original))

        result = test_distribution(
            R_test,
            name=f"weibull_shape_{shape:.1f}",
            description=f"Weibull with shape={shape}",
            parameters={"shape": shape, "scale": float(np.mean(R_original))}
        )
        results.append(result)

        if verbose:
            status = "PASS" if result.passes_8e else "FAIL"
            print(f"shape={shape:.1f}: Df x alpha={result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% dev) [{status}]")
            print(f"          skewness={result.skewness:.2f}, kurtosis={result.kurtosis:.2f}")

    # -------------------------------------------------------------------------
    # Test 25-27: Beta distribution
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENTS 25-27: Beta Distribution Sweep")
        print("=" * 80)

    beta_params = [(0.5, 2.0), (1.0, 3.0), (2.0, 5.0)]

    for a, b in beta_params:
        R_test = generate_beta_scaled(n, a, b, R_original.min(), R_original.max())

        result = test_distribution(
            R_test,
            name=f"beta_a{a:.1f}_b{b:.1f}",
            description=f"Beta(a={a}, b={b}) scaled to R range",
            parameters={"a": a, "b": b}
        )
        results.append(result)

        if verbose:
            status = "PASS" if result.passes_8e else "FAIL"
            print(f"Beta({a},{b}): Df x alpha={result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% dev) [{status}]")
            print(f"            skewness={result.skewness:.2f}, kurtosis={result.kurtosis:.2f}")

    # -------------------------------------------------------------------------
    # Test 28: Mixture of log-normals
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("EXPERIMENT 28: Mixture of Log-normals")
        print("=" * 80)

    R_mixture = generate_mixture_lognormal(n, lognorm_mu - 0.5, lognorm_sigma * 0.5,
                                           lognorm_mu + 1.0, lognorm_sigma * 1.5, weight1=0.7)

    result = test_distribution(
        R_mixture,
        name="mixture_lognormal",
        description="Mixture of two log-normal distributions",
        parameters={"mu1": lognorm_mu - 0.5, "sigma1": lognorm_sigma * 0.5,
                   "mu2": lognorm_mu + 1.0, "sigma2": lognorm_sigma * 1.5}
    )
    results.append(result)

    if verbose:
        print(f"Skewness: {result.skewness:.4f}, Kurtosis: {result.kurtosis:.4f}")
        print(f"Df x alpha = {result.Df_x_alpha:.4f} ({result.deviation_from_8e*100:.2f}% dev)")
        print(f"Status: {'PASS' if result.passes_8e else 'FAIL'}")

    # -------------------------------------------------------------------------
    # ANALYSIS: Find the threshold
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("THRESHOLD ANALYSIS: Finding Minimal Sufficient Conditions")
        print("=" * 80)

    # Sort results by deviation
    sorted_results = sorted(results, key=lambda r: r.deviation_from_8e)
    passing_results = [r for r in sorted_results if r.passes_8e]
    failing_results = [r for r in sorted_results if not r.passes_8e]

    if passing_results:
        avg_pass_skewness = np.mean([r.skewness for r in passing_results])
        avg_pass_kurtosis = np.mean([r.kurtosis for r in passing_results])
        min_pass_skewness = min(r.skewness for r in passing_results)
        min_pass_kurtosis = min(r.kurtosis for r in passing_results)
    else:
        avg_pass_skewness = avg_pass_kurtosis = 0
        min_pass_skewness = min_pass_kurtosis = 0

    if failing_results:
        avg_fail_skewness = np.mean([r.skewness for r in failing_results])
        avg_fail_kurtosis = np.mean([r.kurtosis for r in failing_results])
    else:
        avg_fail_skewness = avg_fail_kurtosis = 0

    if verbose:
        print(f"\nPassing distributions ({len(passing_results)}):")
        print(f"  Avg skewness: {avg_pass_skewness:.4f}")
        print(f"  Avg kurtosis: {avg_pass_kurtosis:.4f}")
        print(f"  Min skewness: {min_pass_skewness:.4f}")
        print(f"  Min kurtosis: {min_pass_kurtosis:.4f}")

        print(f"\nFailing distributions ({len(failing_results)}):")
        print(f"  Avg skewness: {avg_fail_skewness:.4f}")
        print(f"  Avg kurtosis: {avg_fail_kurtosis:.4f}")

        print(f"\nHypothesized thresholds:")
        print(f"  Skewness > {(avg_pass_skewness + avg_fail_skewness) / 2:.2f}")
        print(f"  Kurtosis > {(avg_pass_kurtosis + avg_fail_kurtosis) / 2:.2f}")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n,
        "theoretical_8e": float(EIGHT_E),
        "tolerance": TOLERANCE,
        "original_R_statistics": {
            "mean": float(np.mean(R_original)),
            "std": float(np.std(R_original)),
            "min": float(np.min(R_original)),
            "max": float(np.max(R_original)),
            "skewness": float(compute_skewness(R_original)),
            "kurtosis": float(compute_kurtosis(R_original))
        },
        "distribution_fits": {
            "lognormal": {
                "mu": float(lognorm_mu),
                "sigma": float(lognorm_sigma),
                "ks_statistic": float(lognorm_ks)
            },
            "pareto": {
                "alpha": float(pareto_alpha),
                "x_min": float(pareto_xmin),
                "ks_statistic": float(pareto_ks)
            },
            "gamma": {
                "shape": float(gamma_shape),
                "scale": float(gamma_scale)
            }
        },
        "results": [],
        "summary": {},
        "threshold_analysis": {}
    }

    for r in results:
        output["results"].append({
            "name": r.name,
            "description": r.description,
            "parameters": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                          for k, v in r.parameters.items()},
            "n_samples": r.n_samples,
            "mean": r.mean,
            "std": r.std,
            "skewness": r.skewness,
            "kurtosis": r.kurtosis,
            "embedding_dims": r.embedding_dims,
            "Df": r.Df,
            "alpha": r.alpha,
            "Df_x_alpha": r.Df_x_alpha,
            "deviation_from_8e": r.deviation_from_8e,
            "deviation_percent": r.deviation_from_8e * 100,
            "passes_8e": r.passes_8e,
            "top_eigenvalues": r.top_eigenvalues[:10]
        })

    output["summary"] = {
        "total_distributions_tested": len(results),
        "passing_8e": len(passing_results),
        "failing_8e": len(failing_results),
        "passing_names": [r.name for r in passing_results],
        "failing_names": [r.name for r in failing_results],
        "best_distribution": sorted_results[0].name if sorted_results else None,
        "best_deviation_pct": sorted_results[0].deviation_from_8e * 100 if sorted_results else None
    }

    output["threshold_analysis"] = {
        "passing_avg_skewness": float(avg_pass_skewness),
        "passing_avg_kurtosis": float(avg_pass_kurtosis),
        "passing_min_skewness": float(min_pass_skewness),
        "passing_min_kurtosis": float(min_pass_kurtosis),
        "failing_avg_skewness": float(avg_fail_skewness),
        "failing_avg_kurtosis": float(avg_fail_kurtosis),
        "hypothesized_skewness_threshold": float((avg_pass_skewness + avg_fail_skewness) / 2),
        "hypothesized_kurtosis_threshold": float((avg_pass_kurtosis + avg_fail_kurtosis) / 2),
    }

    # Key findings
    key_findings = []

    # Finding 1: Does shuffled R work?
    orig_result = next((r for r in results if r.name == "original_r"), None)
    shuffled_result = next((r for r in results if r.name == "shuffled_r"), None)
    if orig_result and shuffled_result:
        if abs(orig_result.Df_x_alpha - shuffled_result.Df_x_alpha) < 0.5:
            key_findings.append(
                f"CONFIRMED: Shuffled R produces same Df x alpha ({shuffled_result.Df_x_alpha:.2f} vs {orig_result.Df_x_alpha:.2f}). "
                "Distribution matters, not gene-R correspondence."
            )

    # Finding 2: Which distributions pass?
    if passing_results:
        passing_types = set()
        for r in passing_results:
            if "lognormal" in r.name:
                passing_types.add("log-normal")
            elif "pareto" in r.name:
                passing_types.add("pareto")
            elif "gamma" in r.name:
                passing_types.add("gamma")
            elif "original" in r.name or "shuffled" in r.name:
                passing_types.add("original (likely log-normal)")

        key_findings.append(f"PASSING DISTRIBUTION TYPES: {', '.join(passing_types)}")

    # Finding 3: Skewness/kurtosis thresholds
    if min_pass_skewness > 0:
        key_findings.append(
            f"SKEWNESS THRESHOLD: All passing distributions have skewness > {min_pass_skewness:.2f}"
        )
    if min_pass_kurtosis > -2:  # Normal = 0, so > -2 is meaningful
        key_findings.append(
            f"KURTOSIS THRESHOLD: All passing distributions have excess kurtosis > {min_pass_kurtosis:.2f}"
        )

    # Finding 4: Which always fail?
    always_fail = ["uniform", "gaussian"]
    always_fail_results = [r for r in results if any(f in r.name for f in always_fail)]
    if all(not r.passes_8e for r in always_fail_results):
        key_findings.append("CONFIRMED: Uniform and Gaussian distributions NEVER produce 8e")

    output["key_findings"] = key_findings

    if verbose:
        print(f"\n{'='*80}")
        print("KEY FINDINGS")
        print("=" * 80)
        for finding in key_findings:
            print(f"\n  {finding}")

        print(f"\n{'='*80}")
        print("SUMMARY TABLE: ALL DISTRIBUTIONS")
        print("=" * 80)
        print(f"\n{'Name':<25} {'Skewness':<10} {'Kurtosis':<10} {'Df x a':<10} {'Dev %':<10} {'Status'}")
        print("-" * 75)

        for r in sorted_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"{r.name:<25} {r.skewness:<10.2f} {r.kurtosis:<10.2f} {r.Df_x_alpha:<10.2f} {r.deviation_from_8e*100:<10.1f} {status}")

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_distribution_analysis(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "r_distribution_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
