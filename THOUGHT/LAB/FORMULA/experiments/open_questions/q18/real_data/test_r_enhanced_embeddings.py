#!/usr/bin/env python3
"""
Q18 Investigation: R as Embedding Dimension

CRITICAL INSIGHT:
- Raw gene expression shows Df x alpha = 1177 (way off from 8e)
- R-based embeddings showed Df x alpha = 21.12 (2.9% deviation from 8e!)

HYPOTHESIS:
R captures a component of meaning that, when combined with other features,
produces 8e-compliant embeddings. This script systematically investigates
EXACTLY what R contributes.

EXPERIMENTS:
1. R-weighted PCA: Weight genes by R before PCA
2. R as additional dimension: Append R to feature vectors
3. R-normalized expression: Divide expression by sigma (like R formula)
4. R-gated embeddings: Use R as attention/gating mechanism
5. R-transformed features: Various mathematical transformations of R
6. Minimal R combination: Find smallest feature set that produces 8e

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
TOLERANCE = 0.15  # 15% deviation threshold for "passing" 8e


@dataclass
class ExperimentResult:
    """Results from a single R-enhancement experiment."""
    name: str
    description: str
    hypothesis: str
    n_samples: int
    embedding_dims: int
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float
    passes_8e: bool
    eigenvalues: List[float]
    r_contribution: str  # Description of how R was used
    key_insight: str = ""


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


def load_gene_expression_data(filepath: str) -> Dict[str, Any]:
    """Load gene expression data from JSON cache."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_spectral_properties(cov_matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute Df (participation ratio) and alpha (spectral decay) from covariance matrix.
    Returns: (Df, alpha, eigenvalues)
    """
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio = (sum(lambda))^2 / sum(lambda^2)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent (power law fit: lambda_k ~ k^(-alpha))
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def compute_from_embeddings(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compute spectral properties from embedding matrix directly."""
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)
    return compute_spectral_properties(cov)


def make_result(name: str, description: str, hypothesis: str, embeddings: np.ndarray,
                r_contribution: str, key_insight: str = "") -> ExperimentResult:
    """Create ExperimentResult from embeddings."""
    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return ExperimentResult(
        name=name,
        description=description,
        hypothesis=hypothesis,
        n_samples=embeddings.shape[0],
        embedding_dims=embeddings.shape[1],
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        passes_8e=deviation < TOLERANCE,
        eigenvalues=eigenvalues[:20].tolist(),
        r_contribution=r_contribution,
        key_insight=key_insight
    )


# =============================================================================
# EXPERIMENT 1: R-Weighted PCA
# =============================================================================

def experiment_r_weighted_pca(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> ExperimentResult:
    """
    Weight genes by R before PCA dimensionality reduction.

    Hypothesis: R weights emphasize stable genes, creating structured embeddings.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    # Extract features
    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    # Create feature matrix: [mean, std, log(mean), log(std)]
    features = np.column_stack([
        means / means.max(),
        stds / stds.max(),
        np.log(means + 0.01) / np.log(means.max() + 0.01),
        np.log(stds + 0.01) / np.log(stds.max() + 0.01),
    ])

    # Weight rows by R (stable genes have more influence)
    R_weights = R_values / R_values.max()
    weighted_features = features * R_weights[:, np.newaxis]

    # Expand to n_dims via random projection
    random_proj = np.random.randn(features.shape[1], n_dims) / np.sqrt(features.shape[1])
    embeddings = weighted_features @ random_proj

    return make_result(
        name="r_weighted_pca",
        description="PCA with rows weighted by R values",
        hypothesis="R weights emphasize stable genes, creating structured embeddings",
        embeddings=embeddings,
        r_contribution="R used as row weights (multiplicative scaling)",
    )


# =============================================================================
# EXPERIMENT 2: R as Additional Dimension
# =============================================================================

def experiment_r_as_dimension(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> ExperimentResult:
    """
    Append R directly as an additional dimension to feature vectors.

    Hypothesis: R provides a distinct axis of variation that complements other features.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    # Create base features (without R)
    base_features = np.column_stack([
        means / means.max(),
        stds / stds.max(),
    ])

    # Expand to n_dims-1 via random projection
    random_proj = np.random.randn(base_features.shape[1], n_dims - 1) / np.sqrt(base_features.shape[1])
    expanded = base_features @ random_proj

    # Append R as final dimension
    R_normalized = R_values / R_values.max()
    embeddings = np.column_stack([expanded, R_normalized])

    return make_result(
        name="r_as_dimension",
        description="R appended as an additional embedding dimension",
        hypothesis="R provides a distinct axis of variation that complements other features",
        embeddings=embeddings,
        r_contribution="R added as explicit dimension (column)",
    )


# =============================================================================
# EXPERIMENT 3: R-Normalized Expression (mean/sigma structure)
# =============================================================================

def experiment_r_normalized_features(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> ExperimentResult:
    """
    Create features that mirror R's structure (mean/std normalization).

    Hypothesis: R = mean/std captures signal-to-noise; similar normalizations may help.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    # Create R-like normalized features
    features = np.column_stack([
        means / (stds + 1e-10),  # This IS R
        np.sqrt(means) / (stds + 1e-10),  # sqrt-R variant
        np.log(means + 1) / (stds + 1e-10),  # log-R variant
        means / (stds**2 + 1e-10),  # Precision-weighted mean
    ])

    # Normalize
    for i in range(features.shape[1]):
        col_std = np.std(features[:, i])
        if col_std > 1e-10:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / col_std

    # Expand to n_dims
    random_proj = np.random.randn(features.shape[1], n_dims) / np.sqrt(features.shape[1])
    embeddings = features @ random_proj

    return make_result(
        name="r_normalized_features",
        description="Features with R-like mean/std normalization structure",
        hypothesis="R = mean/std captures signal-to-noise; similar normalizations may help",
        embeddings=embeddings,
        r_contribution="R structure (mean/std ratio) applied to feature construction",
    )


# =============================================================================
# EXPERIMENT 4: R-Gated Embeddings (Attention Mechanism)
# =============================================================================

def experiment_r_gated_embeddings(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> ExperimentResult:
    """
    Use R as an attention/gating mechanism for feature selection.

    Hypothesis: R acts like attention weights, selecting informative features.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    # Create base embeddings
    base_embeddings = np.zeros((n_genes, n_dims))
    for i in range(n_genes):
        np.random.seed(i + seed)
        base_embeddings[i] = np.random.randn(n_dims)

    # R-gating: sigmoid(R) as attention weights
    # High R = high confidence = attended
    R_attention = 1.0 / (1.0 + np.exp(-R_values / R_values.mean()))  # Sigmoid gating

    # Apply gating: scale embeddings by attention
    embeddings = base_embeddings * R_attention[:, np.newaxis]

    return make_result(
        name="r_gated_attention",
        description="R used as attention weights (sigmoid gating)",
        hypothesis="R acts like attention weights, selecting informative features",
        embeddings=embeddings,
        r_contribution="R as multiplicative attention gate (sigmoid normalization)",
    )


# =============================================================================
# EXPERIMENT 5: R-Transformed Features (Mathematical Transformations)
# =============================================================================

def experiment_r_transforms(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> ExperimentResult:
    """
    Various mathematical transformations of R.

    Hypothesis: Specific transformations of R unlock 8e-compliant structure.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    R_values = np.array([genes_data[g]['R'] for g in gene_ids])

    # Multiple R transformations
    features = np.column_stack([
        R_values / R_values.max(),                    # Linear
        np.log(R_values + 0.01),                      # Log
        np.sqrt(R_values),                            # Square root
        R_values ** 2 / (R_values ** 2).max(),        # Quadratic
        np.exp(-R_values / R_values.mean()),          # Exponential decay
        1.0 / (R_values + 0.1),                       # Inverse
        np.sin(R_values / R_values.max() * np.pi),    # Sinusoidal
        np.tanh(R_values / R_values.mean()),          # Tanh
    ])

    # Normalize
    for i in range(features.shape[1]):
        col_std = np.std(features[:, i])
        if col_std > 1e-10:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / col_std

    # Expand to n_dims
    random_proj = np.random.randn(features.shape[1], n_dims) / np.sqrt(features.shape[1])
    embeddings = features @ random_proj

    return make_result(
        name="r_transforms",
        description="Multiple mathematical transformations of R",
        hypothesis="Specific transformations of R unlock 8e-compliant structure",
        embeddings=embeddings,
        r_contribution="R through log, sqrt, exp, sin, tanh transformations",
    )


# =============================================================================
# EXPERIMENT 6: R-Modulated Sinusoidal (The Successful Method - Analyzed)
# =============================================================================

def experiment_sinusoidal_analysis(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> List[ExperimentResult]:
    """
    Analyze WHAT makes the sinusoidal R embedding work.
    Test each component separately.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)
    R_values = np.array([genes_data[g]['R'] for g in gene_ids])

    results = []

    # Component A: Just the sinusoidal base (no R modulation)
    embeddings_a = np.zeros((n_genes, n_dims))
    for i in range(n_genes):
        # Use gene index instead of R for base position
        embeddings_a[i] = np.sin(np.arange(n_dims) * i / (n_genes / 10.0))

    results.append(make_result(
        name="sin_base_only",
        description="Sinusoidal base without R modulation",
        hypothesis="The sinusoidal structure alone may produce 8e",
        embeddings=embeddings_a,
        r_contribution="NONE - R not used",
    ))

    # Component B: Just the R-modulated noise (no sinusoidal base)
    embeddings_b = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        embeddings_b[i] = scale * direction

    results.append(make_result(
        name="r_noise_only",
        description="R-modulated noise without sinusoidal base",
        hypothesis="R-scaled random directions may produce 8e",
        embeddings=embeddings_b,
        r_contribution="R as inverse scale for random directions",
    ))

    # Component C: Full sinusoidal + R noise (the working method)
    embeddings_c = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_c[i] = base_pos + scale * direction

    results.append(make_result(
        name="sin_r_full",
        description="Full sinusoidal base + R-modulated noise",
        hypothesis="Combination of structured base and R-scaled noise produces 8e",
        embeddings=embeddings_c,
        r_contribution="R in both sinusoidal base frequency and noise scaling",
    ))

    # Component D: R in base only (fixed noise)
    embeddings_d = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        direction = np.random.randn(n_dims) * 0.1  # Fixed noise scale
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_d[i] = base_pos + direction

    results.append(make_result(
        name="sin_r_base_fixed_noise",
        description="R-modulated sinusoidal base with fixed noise",
        hypothesis="R in sinusoidal base alone may be sufficient for 8e",
        embeddings=embeddings_d,
        r_contribution="R only in sinusoidal frequency (not noise scaling)",
    ))

    # Component E: Fixed base + R noise
    embeddings_e = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * 0.5)  # Fixed base
        embeddings_e[i] = base_pos + scale * direction

    results.append(make_result(
        name="fixed_base_r_noise",
        description="Fixed sinusoidal base with R-modulated noise",
        hypothesis="R in noise scaling alone may be sufficient for 8e",
        embeddings=embeddings_e,
        r_contribution="R only in noise scaling (not sinusoidal base)",
    ))

    return results


# =============================================================================
# EXPERIMENT 7: Minimal R Combination Search
# =============================================================================

def experiment_minimal_r_combination(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> List[ExperimentResult]:
    """
    Find the MINIMAL feature set that produces 8e when R is included.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    results = []

    # Test: Just R alone (1D)
    embeddings_r_only = R_values.reshape(-1, 1)
    # Need to expand for meaningful spectral analysis
    random_proj = np.random.randn(1, n_dims)
    embeddings_r_expanded = embeddings_r_only @ random_proj

    results.append(make_result(
        name="r_only_expanded",
        description="Only R, randomly projected to n_dims",
        hypothesis="R alone contains the essential structure",
        embeddings=embeddings_r_expanded,
        r_contribution="R is the only input feature",
    ))

    # Test: R + mean (2 features)
    features_2 = np.column_stack([
        R_values / R_values.max(),
        means / means.max(),
    ])
    random_proj = np.random.randn(2, n_dims) / np.sqrt(2)
    embeddings_2 = features_2 @ random_proj

    results.append(make_result(
        name="r_plus_mean",
        description="R + mean expression (2 features)",
        hypothesis="R + mean together capture enough structure",
        embeddings=embeddings_2,
        r_contribution="R as one of two features",
    ))

    # Test: R + std (2 features)
    features_3 = np.column_stack([
        R_values / R_values.max(),
        stds / stds.max(),
    ])
    random_proj = np.random.randn(2, n_dims) / np.sqrt(2)
    embeddings_3 = features_3 @ random_proj

    results.append(make_result(
        name="r_plus_std",
        description="R + std (2 features)",
        hypothesis="R + std together capture enough structure",
        embeddings=embeddings_3,
        r_contribution="R as one of two features",
    ))

    # Test: R + log(R) (structural redundancy)
    features_4 = np.column_stack([
        R_values / R_values.max(),
        np.log(R_values + 0.01) / np.log(R_values.max() + 0.01),
    ])
    random_proj = np.random.randn(2, n_dims) / np.sqrt(2)
    embeddings_4 = features_4 @ random_proj

    results.append(make_result(
        name="r_plus_logr",
        description="R + log(R) (redundant R information)",
        hypothesis="Multiple R representations may improve 8e compliance",
        embeddings=embeddings_4,
        r_contribution="R in two different transformations",
    ))

    # Test: mean + std WITHOUT R (control)
    features_5 = np.column_stack([
        means / means.max(),
        stds / stds.max(),
    ])
    random_proj = np.random.randn(2, n_dims) / np.sqrt(2)
    embeddings_5 = features_5 @ random_proj

    results.append(make_result(
        name="mean_std_no_r",
        description="mean + std WITHOUT R (control)",
        hypothesis="Without R, 8e should not emerge",
        embeddings=embeddings_5,
        r_contribution="NONE - R not used (control)",
    ))

    return results


# =============================================================================
# EXPERIMENT 8: R Distribution Preservation
# =============================================================================

def experiment_r_distribution_preservation(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> List[ExperimentResult]:
    """
    Test if preserving R's DISTRIBUTION is key to 8e emergence.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)
    R_values = np.array([genes_data[g]['R'] for g in gene_ids])

    results = []

    # Original R distribution - sinusoidal embedding
    embeddings_orig = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_orig[i] = base_pos + scale * direction

    results.append(make_result(
        name="r_original_distribution",
        description="Original R distribution (baseline)",
        hypothesis="Original R distribution produces 8e",
        embeddings=embeddings_orig,
        r_contribution="Original R values",
    ))

    # Shuffled R - breaks R-gene correspondence
    R_shuffled = R_values.copy()
    np.random.shuffle(R_shuffled)
    embeddings_shuffled = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_shuffled):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_shuffled[i] = base_pos + scale * direction

    results.append(make_result(
        name="r_shuffled",
        description="Shuffled R values (same distribution, broken correspondence)",
        hypothesis="R's correspondence to genes matters, not just distribution",
        embeddings=embeddings_shuffled,
        r_contribution="Shuffled R values (same distribution)",
    ))

    # Uniform R - completely different distribution
    R_uniform = np.linspace(R_values.min(), R_values.max(), n_genes)
    embeddings_uniform = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_uniform):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_uniform[i] = base_pos + scale * direction

    results.append(make_result(
        name="r_uniform",
        description="Uniform R distribution (same range)",
        hypothesis="R's non-uniform distribution is key",
        embeddings=embeddings_uniform,
        r_contribution="Uniformly distributed R values",
    ))

    # Gaussian R - normal distribution with same mean/std
    R_gaussian = np.random.randn(n_genes) * R_values.std() + R_values.mean()
    R_gaussian = np.clip(R_gaussian, R_values.min(), R_values.max())
    embeddings_gaussian = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_gaussian):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_gaussian[i] = base_pos + scale * direction

    results.append(make_result(
        name="r_gaussian",
        description="Gaussian R distribution (same mean/std)",
        hypothesis="R's specific distribution shape matters",
        embeddings=embeddings_gaussian,
        r_contribution="Gaussian-distributed R values",
    ))

    return results


# =============================================================================
# EXPERIMENT 9: R as Spectral Parameter
# =============================================================================

def experiment_r_spectral_role(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> List[ExperimentResult]:
    """
    Test R's role in shaping the eigenvalue spectrum directly.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)
    R_values = np.array([genes_data[g]['R'] for g in gene_ids])

    results = []

    # Create embeddings where R DIRECTLY determines eigenvalue contributions
    # Hypothesis: R encodes the "eigenvalue weight" of each sample

    # Method A: R as eigenvalue weights
    base_embeddings = np.random.randn(n_genes, n_dims)
    # Weight each gene's contribution to covariance by R
    R_weights = np.sqrt(R_values / R_values.max())  # sqrt for variance-like scaling
    weighted_embeddings = base_embeddings * R_weights[:, np.newaxis]

    results.append(make_result(
        name="r_eigenvalue_weights",
        description="R as direct eigenvalue contribution weights",
        hypothesis="R determines how much each gene contributes to spectral structure",
        embeddings=weighted_embeddings,
        r_contribution="R as sqrt-scaled row weights (eigenvalue contribution)",
    ))

    # Method B: R to construct power-law spectrum
    # Create embeddings that produce eigenvalues following R's sorted distribution
    R_sorted = np.sort(R_values)[::-1]

    # Use SVD to create embeddings with controlled spectrum
    U = np.random.randn(n_genes, n_dims)
    U, _ = np.linalg.qr(U)  # Orthonormal
    V = np.random.randn(n_dims, n_dims)
    V, _ = np.linalg.qr(V)  # Orthonormal

    # Singular values from R (normalized)
    S = np.zeros(n_dims)
    S[:min(n_dims, len(R_sorted))] = R_sorted[:min(n_dims, len(R_sorted))] / R_sorted[0]

    embeddings_controlled = U[:, :n_dims] @ np.diag(S) @ V

    results.append(make_result(
        name="r_controlled_spectrum",
        description="Embeddings with spectrum directly from R values",
        hypothesis="R's distribution IS the eigenvalue distribution",
        embeddings=embeddings_controlled,
        r_contribution="R values used as singular values",
    ))

    return results


# =============================================================================
# EXPERIMENT 10: The Sigma^(-1) Insight
# =============================================================================

def experiment_sigma_inverse(genes_data: Dict, n_dims: int = 50, seed: int = 42) -> List[ExperimentResult]:
    """
    R = E/sigma, so 1/sigma is a key component.
    Test if 1/sigma alone (without E) produces 8e.
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    results = []

    # 1/sigma alone
    sigma_inv = 1.0 / (stds + 1e-10)
    embeddings_sigma_inv = np.zeros((n_genes, n_dims))
    for i, s_inv in enumerate(sigma_inv):
        np.random.seed(i + seed)
        scale = 1.0 / (s_inv / sigma_inv.max() + 0.1)  # Scale inversely
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * s_inv / (sigma_inv.max() / 10.0))
        embeddings_sigma_inv[i] = base_pos + scale * direction

    results.append(make_result(
        name="sigma_inverse_only",
        description="Using 1/sigma instead of R",
        hypothesis="1/sigma (precision) may be the key component of R",
        embeddings=embeddings_sigma_inv,
        r_contribution="1/sigma used instead of R (testing if precision alone works)",
    ))

    # E alone (mean/max approximation)
    E_approx = means / means.max()  # Treating normalized mean as E
    embeddings_e_only = np.zeros((n_genes, n_dims))
    for i, e in enumerate(E_approx):
        np.random.seed(i + seed)
        scale = 1.0 / (e + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * e * 10.0)
        embeddings_e_only[i] = base_pos + scale * direction

    results.append(make_result(
        name="e_component_only",
        description="Using E approximation (normalized mean) instead of R",
        hypothesis="E (agreement) may be the key component of R",
        embeddings=embeddings_e_only,
        r_contribution="E (normalized mean) used instead of R",
    ))

    # E * sigma^(-1) = R (confirming the formula works)
    R_reconstructed = E_approx * sigma_inv / sigma_inv.max()  # Scale to match
    embeddings_r_reconstructed = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_reconstructed):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings_r_reconstructed[i] = base_pos + scale * direction

    results.append(make_result(
        name="r_reconstructed",
        description="R reconstructed from E * 1/sigma",
        hypothesis="Reconstructed R should match original R behavior",
        embeddings=embeddings_r_reconstructed,
        r_contribution="R reconstructed from components (E * 1/sigma)",
    ))

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_experiments(verbose: bool = True) -> Dict[str, Any]:
    """Run all R-enhanced embedding experiments."""

    cache_path = Path(__file__).parent / "cache" / "gene_expression_sample.json"

    if verbose:
        print("=" * 80)
        print("Q18 INVESTIGATION: R AS EMBEDDING DIMENSION")
        print("Understanding WHAT R Contributes to 8e Emergence")
        print("=" * 80)
        print(f"\nLoading data from: {cache_path}")

    data = load_gene_expression_data(str(cache_path))
    genes_data = data['genes']
    R_values = np.array([g['R'] for g in genes_data.values()])
    n_genes = len(R_values)

    if verbose:
        print(f"Loaded {n_genes} genes")
        print(f"R range: [{R_values.min():.2f}, {R_values.max():.2f}]")
        print(f"R mean: {R_values.mean():.2f}, std: {R_values.std():.2f}")
        print(f"\nTheoretical 8e: {EIGHT_E:.4f}")
        print(f"Tolerance: {TOLERANCE*100:.0f}%")

    all_results = []

    # Run each experiment group
    experiments = [
        ("1. R-Weighted PCA", lambda: [experiment_r_weighted_pca(genes_data)]),
        ("2. R as Dimension", lambda: [experiment_r_as_dimension(genes_data)]),
        ("3. R-Normalized Features", lambda: [experiment_r_normalized_features(genes_data)]),
        ("4. R-Gated Attention", lambda: [experiment_r_gated_embeddings(genes_data)]),
        ("5. R Transforms", lambda: [experiment_r_transforms(genes_data)]),
        ("6. Sinusoidal Analysis", lambda: experiment_sinusoidal_analysis(genes_data)),
        ("7. Minimal R Combination", lambda: experiment_minimal_r_combination(genes_data)),
        ("8. R Distribution Tests", lambda: experiment_r_distribution_preservation(genes_data)),
        ("9. R Spectral Role", lambda: experiment_r_spectral_role(genes_data)),
        ("10. Sigma^(-1) Insight", lambda: experiment_sigma_inverse(genes_data)),
    ]

    for exp_name, exp_func in experiments:
        if verbose:
            print(f"\n{'='*80}")
            print(f"EXPERIMENT: {exp_name}")
            print("=" * 80)

        results = exp_func()
        all_results.extend(results)

        if verbose:
            for r in results:
                status = "PASS" if r.passes_8e else "FAIL"
                print(f"\n  {r.name}:")
                print(f"    Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_from_8e*100:.1f}% dev) [{status}]")
                print(f"    R contribution: {r.r_contribution}")

    # Analyze results
    passing = [r for r in all_results if r.passes_8e]
    failing = [r for r in all_results if not r.passes_8e]

    # Sort by deviation
    all_results_sorted = sorted(all_results, key=lambda x: x.deviation_from_8e)

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: TOP 10 METHODS BY 8e COMPLIANCE")
        print("=" * 80)
        print(f"\n{'Rank':<6} {'Method':<35} {'Df x alpha':<12} {'Dev %':<10} {'Status'}")
        print("-" * 75)

        for i, r in enumerate(all_results_sorted[:10], 1):
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"{i:<6} {r.name:<35} {r.Df_x_alpha:<12.2f} {r.deviation_from_8e*100:<10.1f} {status}")

    # Key insights
    if verbose:
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)

        # What R contributions pass?
        print("\n[PASSING METHODS - How R is used:]")
        for r in passing:
            print(f"  - {r.name}: {r.r_contribution}")

        print(f"\n[STATISTICS]")
        print(f"  Total experiments: {len(all_results)}")
        print(f"  Passing 8e: {len(passing)}")
        print(f"  Failing 8e: {len(failing)}")
        print(f"  Best method: {all_results_sorted[0].name} ({all_results_sorted[0].deviation_from_8e*100:.1f}%)")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_genes": n_genes,
        "theoretical_8e": float(EIGHT_E),
        "tolerance": TOLERANCE,
        "R_statistics": {
            "mean": float(R_values.mean()),
            "std": float(R_values.std()),
            "min": float(R_values.min()),
            "max": float(R_values.max())
        },
        "results": [],
        "summary": {},
        "key_findings": []
    }

    for r in all_results:
        output["results"].append({
            "name": r.name,
            "description": r.description,
            "hypothesis": r.hypothesis,
            "n_samples": r.n_samples,
            "embedding_dims": r.embedding_dims,
            "Df": r.Df,
            "alpha": r.alpha,
            "Df_x_alpha": r.Df_x_alpha,
            "deviation_from_8e": r.deviation_from_8e,
            "deviation_percent": r.deviation_from_8e * 100,
            "passes_8e": r.passes_8e,
            "r_contribution": r.r_contribution,
            "top_eigenvalues": r.eigenvalues[:10]
        })

    output["summary"] = {
        "total_experiments": len(all_results),
        "passing_8e": len(passing),
        "failing_8e": len(failing),
        "passing_methods": [r.name for r in passing],
        "best_method": all_results_sorted[0].name if all_results else None,
        "best_deviation_pct": all_results_sorted[0].deviation_from_8e * 100 if all_results else None
    }

    # Extract key findings
    key_findings = []

    # Finding 1: Does sinusoidal structure alone work?
    sin_base = next((r for r in all_results if r.name == "sin_base_only"), None)
    sin_r_full = next((r for r in all_results if r.name == "sin_r_full"), None)
    if sin_base and sin_r_full:
        if sin_base.passes_8e:
            key_findings.append("FINDING: Sinusoidal structure ALONE produces 8e (R is optional)")
        elif sin_r_full.passes_8e:
            key_findings.append("FINDING: R modulation is NECESSARY for sinusoidal embeddings to produce 8e")

    # Finding 2: Does shuffled R still work?
    r_orig = next((r for r in all_results if r.name == "r_original_distribution"), None)
    r_shuffled = next((r for r in all_results if r.name == "r_shuffled"), None)
    if r_orig and r_shuffled:
        if r_shuffled.passes_8e and r_orig.passes_8e:
            key_findings.append("FINDING: Shuffled R still produces 8e - distribution matters, not correspondence")
        elif r_orig.passes_8e and not r_shuffled.passes_8e:
            key_findings.append("FINDING: R-gene correspondence is CRUCIAL - shuffling breaks 8e")

    # Finding 3: Is 1/sigma the key?
    sigma_inv = next((r for r in all_results if r.name == "sigma_inverse_only"), None)
    if sigma_inv:
        if sigma_inv.passes_8e:
            key_findings.append("FINDING: 1/sigma (precision) ALONE produces 8e - R's key component identified!")
        else:
            key_findings.append("FINDING: 1/sigma alone is NOT sufficient - R's E component also matters")

    # Finding 4: Minimal combination
    r_plus_mean = next((r for r in all_results if r.name == "r_plus_mean"), None)
    mean_std_no_r = next((r for r in all_results if r.name == "mean_std_no_r"), None)
    if r_plus_mean and mean_std_no_r:
        if r_plus_mean.passes_8e and not mean_std_no_r.passes_8e:
            key_findings.append("FINDING: R is the essential ingredient - adding R to mean produces 8e, but mean+std without R fails")

    output["key_findings"] = key_findings

    if verbose and key_findings:
        print("\n" + "=" * 80)
        print("CRITICAL DISCOVERIES")
        print("=" * 80)
        for finding in key_findings:
            print(f"\n  {finding}")

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_all_experiments(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "r_enhanced_embeddings_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
