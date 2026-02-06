#!/usr/bin/env python3
"""
Q18 Investigation: 8e Detection in Biological Embeddings

CRITICAL INSIGHT FROM Q18 TESTING:
- Raw gene expression data: Df x alpha = 1177 (5316% deviation from 8e)
- R-based structured embedding: Df x alpha = 21.12 (2.9% deviation!)

This suggests 8e is a property of REPRESENTATION STRUCTURE, not raw data.

This script systematically tests different embedding approaches to understand:
1. WHY structured embeddings produce 8e
2. WHAT properties of embeddings are necessary for 8e to emerge
3. HOW to detect "novel information" via 8e deviation

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746


@dataclass
class EmbeddingResult:
    """Results from a single embedding approach."""
    name: str
    description: str
    n_samples: int
    embedding_dims: int
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float
    eigenvalues: List[float]
    passes_8e: bool  # Within 15% of 8e


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
    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio
    # PR = (sum(lambda))^2 / sum(lambda^2)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent
    # Fit power law: lambda_k ~ k^(-alpha)
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    # Linear regression for slope
    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def compute_from_embeddings(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compute spectral properties from embedding matrix directly."""
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)
    # Compute covariance
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)
    return compute_spectral_properties(cov)


# =============================================================================
# EMBEDDING STRATEGIES
# =============================================================================

def embedding_raw_R_values(R_values: np.ndarray) -> EmbeddingResult:
    """
    Method 1: Raw R values as 1D 'embedding'
    EXPECTED: FAIL - No structure
    """
    # Treat sorted R values as pseudo-eigenvalues
    R_sorted = np.sort(R_values)[::-1]
    R_sorted = R_sorted[R_sorted > 0]

    # Df from R values
    sum_R = np.sum(R_sorted)
    sum_R_sq = np.sum(R_sorted ** 2)
    Df_R = (sum_R ** 2) / (sum_R_sq + 1e-10)

    # alpha from R value decay
    k = np.arange(1, len(R_sorted) + 1)
    log_k = np.log(k)
    log_R = np.log(R_sorted + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_R) - np.sum(log_k) * np.sum(log_R))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha_R = -slope

    product = Df_R * alpha_R
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="raw_R_values",
        description="Raw R values treated as pseudo-spectrum (no embedding)",
        n_samples=len(R_values),
        embedding_dims=1,
        Df=Df_R,
        alpha=alpha_R,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=R_sorted[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_sinusoidal_R(R_values: np.ndarray, n_dims: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 2: Sinusoidal embedding based on R (the successful method)

    Key insight: This creates STRUCTURE through:
    - Base position: sinusoidal function of R * dimension_index
    - Noise: scaled inversely by R (high R = clustered, low R = spread)

    EXPECTED: PASS - ~2.9% deviation
    """
    np.random.seed(seed)
    n_genes = len(R_values)
    embeddings = np.zeros((n_genes, n_dims))

    for i, r in enumerate(R_values):
        np.random.seed(i + seed)

        # Scale factor based on R (inverse relationship)
        scale = 1.0 / (r + 0.1)

        # Random direction modulated by R
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Position: base + R-modulated spread
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="sinusoidal_R_embedding",
        description="Sinusoidal base + R-modulated noise (original successful method)",
        n_samples=n_genes,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_pca_expression(genes_data: Dict[str, Dict], n_components: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 3: PCA of gene expression statistics

    Creates embeddings from [mean, std, R, log(R), mean/std, mean*R, ...]

    EXPECTED: UNCERTAIN - depends on feature correlation structure
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    # Create feature matrix
    features = []
    for gene_id in gene_ids:
        g = genes_data[gene_id]
        R = g['R']
        mean_expr = g['mean_expr']
        std_expr = g['std_expr']

        # Multiple derived features
        f = [
            R,
            np.log(R + 0.01),
            mean_expr,
            std_expr,
            mean_expr / (std_expr + 1e-10),
            R * mean_expr,
            np.sqrt(R),
            R ** 2,
            np.sin(R),
            np.cos(R),
        ]
        features.append(f)

    features = np.array(features)

    # Normalize
    for i in range(features.shape[1]):
        col_std = np.std(features[:, i])
        if col_std > 1e-10:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / col_std

    # Extend to n_components via random projection if needed
    if features.shape[1] < n_components:
        random_proj = np.random.randn(features.shape[1], n_components) / np.sqrt(features.shape[1])
        features = features @ random_proj

    Df, alpha, eigenvalues = compute_from_embeddings(features)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="pca_expression_features",
        description="PCA of derived expression features (R, log(R), mean, std, etc.)",
        n_samples=n_genes,
        embedding_dims=features.shape[1],
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_network_graph(R_values: np.ndarray, n_dims: int = 50, n_neighbors: int = 20, seed: int = 42) -> EmbeddingResult:
    """
    Method 4: Gene-gene similarity network embedding

    Creates a network where genes are connected based on R-similarity,
    then embeds using spectral graph embedding.

    EXPECTED: UNCERTAIN - network structure may or may not produce 8e
    """
    np.random.seed(seed)
    n_genes = len(R_values)

    # Build adjacency matrix based on R-similarity
    # Two genes are connected if their R values are similar
    adj = np.zeros((n_genes, n_genes))

    for i in range(n_genes):
        # Find k nearest neighbors by R-distance
        distances = np.abs(R_values - R_values[i])
        neighbors = np.argsort(distances)[:n_neighbors + 1]

        for j in neighbors:
            if i != j:
                # Weight by R-similarity (closer R = stronger connection)
                similarity = np.exp(-np.abs(R_values[i] - R_values[j]) / (np.std(R_values) + 1e-10))
                adj[i, j] = similarity
                adj[j, i] = similarity

    # Compute Laplacian
    degree = np.sum(adj, axis=1)
    D = np.diag(degree)
    L = D - adj

    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Spectral embedding: use eigenvectors of Laplacian
    eigenvalues_L, eigenvectors = np.linalg.eigh(L_norm)

    # Take first n_dims non-trivial eigenvectors as embedding
    embeddings = eigenvectors[:, 1:min(n_dims + 1, n_genes)]

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="network_spectral_embedding",
        description="Spectral embedding of R-similarity network (Laplacian eigenvectors)",
        n_samples=n_genes,
        embedding_dims=embeddings.shape[1],
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_cluster_based(R_values: np.ndarray, n_clusters: int = 8, n_dims: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 5: Cluster-based embedding (8 clusters for Peircean octants)

    Hypothesis: If 8e comes from 8 octants, clustering into 8 groups
    and creating structured embeddings might recover 8e.

    EXPECTED: INTERESTING - Tests the octant hypothesis directly
    """
    np.random.seed(seed)
    n_genes = len(R_values)

    # Simple k-means style clustering on R values
    # Sort R values and divide into n_clusters
    sorted_indices = np.argsort(R_values)
    cluster_size = n_genes // n_clusters
    cluster_labels = np.zeros(n_genes, dtype=int)

    for c in range(n_clusters):
        start = c * cluster_size
        end = (c + 1) * cluster_size if c < n_clusters - 1 else n_genes
        cluster_labels[sorted_indices[start:end]] = c

    # Create embeddings: each gene gets position based on cluster + R-offset
    embeddings = np.zeros((n_genes, n_dims))

    # Cluster centers in embedding space (arranged on simplex)
    centers = np.zeros((n_clusters, n_dims))
    for c in range(n_clusters):
        # Spread clusters in embedding space
        angle = 2 * np.pi * c / n_clusters
        centers[c, :min(2, n_dims)] = [np.cos(angle), np.sin(angle)]
        if n_dims > 2:
            centers[c, 2:] = np.random.randn(n_dims - 2) * 0.1

    for i in range(n_genes):
        c = cluster_labels[i]
        r = R_values[i]

        # Position = cluster center + R-modulated offset
        offset = np.random.randn(n_dims) / (r + 0.1)
        embeddings[i] = centers[c] + offset

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="cluster_8_octant_embedding",
        description=f"8-cluster embedding (testing Peircean octant hypothesis)",
        n_samples=n_genes,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_random_baseline(n_samples: int, n_dims: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 6: Pure random embeddings (baseline)

    Random matrices should produce Df x alpha ~ 14.5, NOT 8e.
    This is the negative control.

    EXPECTED: ~14.5 (NOT 8e) - confirms 8e requires structure
    """
    np.random.seed(seed)
    embeddings = np.random.randn(n_samples, n_dims)

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="random_baseline",
        description="Pure random embeddings (negative control - should NOT show 8e)",
        n_samples=n_samples,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_dimensionality_sweep(R_values: np.ndarray, dims_list: List[int], seed: int = 42) -> List[EmbeddingResult]:
    """
    Method 7: Test how dimensionality affects 8e emergence

    Hypothesis: 8e might emerge as dimensionality increases (more structure)
    """
    results = []
    for n_dims in dims_list:
        result = embedding_sinusoidal_R(R_values, n_dims=n_dims, seed=seed)
        result.name = f"sinusoidal_R_{n_dims}D"
        result.description = f"Sinusoidal R embedding at {n_dims} dimensions"
        results.append(result)
    return results


def embedding_amino_acid_property(genes_data: Dict[str, Dict], n_dims: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 8: Protein-like embedding using amino acid property analogy

    Treats R as a composite 'property' like amino acid hydrophobicity,
    and creates embeddings similar to how protein language models work.

    EXPECTED: UNCERTAIN - tests if biological property encoding produces 8e
    """
    np.random.seed(seed)
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    # Extract properties
    R_values = np.array([genes_data[g]['R'] for g in gene_ids])
    means = np.array([genes_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([genes_data[g]['std_expr'] for g in gene_ids])

    # Create "amino acid property"-like encoding
    # Analogous to how ESM-2 encodes proteins via learned embeddings

    # Property 1: Stability (R)
    # Property 2: Activity (mean)
    # Property 3: Variability (std)
    # Property 4-N: Derived/interaction terms

    properties = np.column_stack([
        R_values / R_values.max(),  # Normalized stability
        means / means.max(),  # Normalized activity
        stds / stds.max(),  # Normalized variability
        R_values * means / (R_values.max() * means.max()),  # Interaction
        np.log(R_values + 1) / np.log(R_values.max() + 1),  # Log stability
    ])

    # Expand to n_dims using learned-like transformation
    # This mimics how neural networks create embeddings from features
    W1 = np.random.randn(properties.shape[1], n_dims * 2) * 0.1
    hidden = np.tanh(properties @ W1)  # Nonlinearity like in neural nets

    W2 = np.random.randn(n_dims * 2, n_dims) * 0.1
    embeddings = hidden @ W2

    # Add position encoding (like transformers)
    position_encoding = np.zeros((n_genes, n_dims))
    for i in range(n_genes):
        for d in range(n_dims):
            if d % 2 == 0:
                position_encoding[i, d] = np.sin(R_values[i] / (10000 ** (d / n_dims)))
            else:
                position_encoding[i, d] = np.cos(R_values[i] / (10000 ** ((d - 1) / n_dims)))

    embeddings = embeddings + 0.1 * position_encoding

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="amino_acid_property_analog",
        description="Protein-like property encoding (mimics ESM-2 structure)",
        n_samples=n_genes,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_fourier_R(R_values: np.ndarray, n_dims: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 9: Fourier embedding of R values

    Uses Fourier basis functions to create embeddings.
    This is commonly used in neural networks (positional encoding).

    EXPECTED: INTERESTING - Fourier basis is mathematically principled
    """
    np.random.seed(seed)
    n_genes = len(R_values)
    embeddings = np.zeros((n_genes, n_dims))

    # Normalize R to [0, 2*pi] range
    R_norm = 2 * np.pi * (R_values - R_values.min()) / (R_values.max() - R_values.min() + 1e-10)

    for i, r in enumerate(R_norm):
        for d in range(n_dims):
            freq = (d // 2) + 1
            if d % 2 == 0:
                embeddings[i, d] = np.sin(freq * r)
            else:
                embeddings[i, d] = np.cos(freq * r)

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="fourier_R_embedding",
        description="Fourier basis embedding of R values (sin/cos at multiple frequencies)",
        n_samples=n_genes,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


def embedding_gaussian_mixture(R_values: np.ndarray, n_components: int = 8, n_dims: int = 50, seed: int = 42) -> EmbeddingResult:
    """
    Method 10: Gaussian mixture embedding

    Models R distribution as mixture of Gaussians, embeds based on
    posterior probabilities for each component.

    EXPECTED: INTERESTING - GMM is a common generative model
    """
    np.random.seed(seed)
    n_genes = len(R_values)

    # Fit simple GMM (using k-means approximation)
    # Sort and create component boundaries
    R_sorted = np.sort(R_values)
    boundaries = [R_sorted[min(i * n_genes // n_components, n_genes - 1)] for i in range(n_components)]
    boundaries.append(R_sorted[-1] + 1)

    # Compute component means and stds
    component_means = []
    component_stds = []
    for c in range(n_components):
        mask = (R_values >= boundaries[c]) & (R_values < boundaries[c + 1])
        if np.sum(mask) > 0:
            component_means.append(np.mean(R_values[mask]))
            component_stds.append(max(np.std(R_values[mask]), 0.1))
        else:
            component_means.append(boundaries[c])
            component_stds.append(1.0)

    component_means = np.array(component_means)
    component_stds = np.array(component_stds)

    # Compute soft assignments (posterior probabilities)
    posteriors = np.zeros((n_genes, n_components))
    for i, r in enumerate(R_values):
        for c in range(n_components):
            posteriors[i, c] = np.exp(-0.5 * ((r - component_means[c]) / component_stds[c]) ** 2)
        posteriors[i] /= posteriors[i].sum() + 1e-10

    # Expand to n_dims
    embeddings = np.zeros((n_genes, n_dims))
    embeddings[:, :n_components] = posteriors

    # Add R-based features in remaining dimensions
    if n_dims > n_components:
        for d in range(n_components, n_dims):
            freq = (d - n_components + 1) * 0.5
            embeddings[:, d] = np.sin(freq * R_values / 10.0) + np.random.randn(n_genes) * 0.1

    Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return EmbeddingResult(
        name="gaussian_mixture_embedding",
        description="GMM posterior probabilities + R-based features",
        n_samples=n_genes,
        embedding_dims=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        passes_8e=deviation < 0.15
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_embedding_tests(verbose: bool = True) -> Dict[str, Any]:
    """
    Run all embedding tests and compare results.
    """
    # Load data
    cache_path = Path(__file__).parent / "cache" / "gene_expression_sample.json"

    if verbose:
        print("=" * 80)
        print("Q18 INVESTIGATION: 8e DETECTION IN BIOLOGICAL EMBEDDINGS")
        print("=" * 80)
        print(f"\nLoading data from: {cache_path}")

    data = load_gene_expression_data(str(cache_path))
    genes_data = data['genes']
    R_values = np.array([g['R'] for g in genes_data.values()])
    n_genes = len(R_values)

    if verbose:
        print(f"Loaded {n_genes} genes")
        print(f"R range: {R_values.min():.2f} - {R_values.max():.2f}")
        print(f"R mean: {R_values.mean():.2f}, std: {R_values.std():.2f}")
        print(f"\nTheoretical 8e: {EIGHT_E:.4f}")
        print(f"Random baseline expected: ~14.5")
        print("\n" + "=" * 80)

    results = []

    # Test 1: Raw R values
    if verbose:
        print("\n[1/10] Testing: Raw R values...")
    result = embedding_raw_R_values(R_values)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 2: Sinusoidal R embedding (the successful one)
    if verbose:
        print("\n[2/10] Testing: Sinusoidal R embedding (50D)...")
    result = embedding_sinusoidal_R(R_values, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 3: PCA of expression features
    if verbose:
        print("\n[3/10] Testing: PCA expression features...")
    result = embedding_pca_expression(genes_data, n_components=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 4: Network spectral embedding
    if verbose:
        print("\n[4/10] Testing: Network spectral embedding...")
    result = embedding_network_graph(R_values, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 5: 8-cluster embedding
    if verbose:
        print("\n[5/10] Testing: 8-cluster octant embedding...")
    result = embedding_cluster_based(R_values, n_clusters=8, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 6: Random baseline
    if verbose:
        print("\n[6/10] Testing: Random baseline (negative control)...")
    result = embedding_random_baseline(n_genes, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 7: Amino acid property analog
    if verbose:
        print("\n[7/10] Testing: Amino acid property analog...")
    result = embedding_amino_acid_property(genes_data, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 8: Fourier R embedding
    if verbose:
        print("\n[8/10] Testing: Fourier R embedding...")
    result = embedding_fourier_R(R_values, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 9: Gaussian mixture embedding
    if verbose:
        print("\n[9/10] Testing: Gaussian mixture embedding (8 components)...")
    result = embedding_gaussian_mixture(R_values, n_components=8, n_dims=50)
    results.append(result)
    if verbose:
        print(f"       Df x alpha = {result.Df_x_alpha:.2f} ({result.deviation_from_8e*100:.1f}% deviation)")

    # Test 10: Dimensionality sweep
    if verbose:
        print("\n[10/10] Testing: Dimensionality sweep...")
    dims_results = embedding_dimensionality_sweep(R_values, [10, 25, 50, 100, 200, 500])
    results.extend(dims_results)
    if verbose:
        for r in dims_results:
            print(f"       {r.embedding_dims}D: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_from_8e*100:.1f}%)")

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: EMBEDDING COMPARISON")
        print("=" * 80)
        print(f"\n{'Method':<40} {'Df':<10} {'alpha':<10} {'Df x alpha':<12} {'Dev %':<10} {'8e?'}")
        print("-" * 95)

        for r in results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"{r.name:<40} {r.Df:<10.2f} {r.alpha:<10.4f} {r.Df_x_alpha:<12.2f} {r.deviation_from_8e*100:<10.1f} {status}")

    # Build output structure
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_genes": n_genes,
        "theoretical_8e": float(EIGHT_E),
        "random_expected": 14.5,
        "R_statistics": {
            "mean": float(R_values.mean()),
            "std": float(R_values.std()),
            "min": float(R_values.min()),
            "max": float(R_values.max())
        },
        "results": [],
        "summary": {}
    }

    for r in results:
        output["results"].append({
            "name": r.name,
            "description": r.description,
            "n_samples": r.n_samples,
            "embedding_dims": r.embedding_dims,
            "Df": r.Df,
            "alpha": r.alpha,
            "Df_x_alpha": r.Df_x_alpha,
            "deviation_from_8e": r.deviation_from_8e,
            "deviation_percent": r.deviation_from_8e * 100,
            "passes_8e": r.passes_8e,
            "top_eigenvalues": r.eigenvalues[:10]
        })

    # Compute summary statistics
    passing_methods = [r for r in results if r.passes_8e]
    failing_methods = [r for r in results if not r.passes_8e]

    output["summary"] = {
        "total_methods": len(results),
        "passing_8e": len(passing_methods),
        "failing_8e": len(failing_methods),
        "passing_methods": [r.name for r in passing_methods],
        "failing_methods": [r.name for r in failing_methods],
        "best_method": min(results, key=lambda x: x.deviation_from_8e).name if results else None,
        "worst_method": max(results, key=lambda x: x.deviation_from_8e).name if results else None
    }

    if verbose:
        print(f"\n{'=' * 80}")
        print("KEY FINDINGS:")
        print("=" * 80)
        print(f"\n  Methods passing 8e (<15% deviation): {len(passing_methods)}/{len(results)}")
        print(f"  Best method: {output['summary']['best_method']}")
        if passing_methods:
            print(f"\n  PASSING METHODS:")
            for r in passing_methods:
                print(f"    - {r.name}: {r.Df_x_alpha:.2f} ({r.deviation_from_8e*100:.1f}% dev)")
        print("=" * 80)

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_all_embedding_tests(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "8e_embeddings_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
