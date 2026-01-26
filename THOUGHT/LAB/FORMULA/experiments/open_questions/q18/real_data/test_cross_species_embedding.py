#!/usr/bin/env python3
"""
Q18 Investigation: Cross-Species Transfer in Embedding Space

CONTEXT:
- Direct cross-species R transfer failed (r=0.054)
- 8e emerges in embeddings, not raw data
- Maybe the issue is we're comparing raw R values instead of embedded representations

HYPOTHESIS:
Cross-species transfer works in EMBEDDING SPACE, not raw R space.

APPROACH:
1. Take human gene expression data (GEO)
2. Take mouse gene expression data (GEO)
3. Create embeddings of the expression patterns
4. Compare the GEOMETRY of embeddings between species
5. Check if orthologous genes cluster similarly in embedding space
6. Test if 8e or any geometric constant emerges

KEY INSIGHT:
The original cross-species test compared SCALAR R values.
This test compares VECTOR embeddings and their geometric properties.

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 1.0.0
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
CACHE_DIR = Path(__file__).parent / 'cache'
RESULTS_FILE = Path(__file__).parent / 'cross_species_embedding_results.json'


@dataclass
class EmbeddingGeometry:
    """Geometric properties of an embedding space."""
    name: str
    n_samples: int
    embedding_dim: int
    intrinsic_dim: float  # Participation ratio
    spectral_decay: float  # Alpha
    Df_x_alpha: float
    deviation_from_8e: float
    eigenvalues: List[float]
    centroid: np.ndarray
    mean_distance_to_centroid: float
    mean_pairwise_distance: float


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
# DATA LOADING
# =============================================================================

def load_human_expression() -> Dict[str, Dict]:
    """Load human gene expression data."""
    filepath = CACHE_DIR / 'gene_expression_sample.json'
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('genes', data)


def load_mouse_expression() -> Dict[str, Dict]:
    """Load mouse gene expression data."""
    filepath = CACHE_DIR / 'mouse_expression_real.json'
    with open(filepath, 'r') as f:
        return json.load(f)


def load_orthologs() -> List[Dict]:
    """Load human-mouse ortholog mappings."""
    filepath = CACHE_DIR / 'human_mouse_orthologs.json'
    with open(filepath, 'r') as f:
        return json.load(f)


def load_probe_annotations() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load probe-to-gene mappings for both species."""
    human_file = CACHE_DIR / 'probe_to_gene_gpl570.json'
    mouse_file = CACHE_DIR / 'probe_to_gene_gpl1261_mouse.json'

    with open(human_file, 'r') as f:
        human_annot = json.load(f)

    with open(mouse_file, 'r') as f:
        mouse_annot = json.load(f)

    return human_annot, mouse_annot


# =============================================================================
# EMBEDDING CREATION
# =============================================================================

def create_r_feature_embedding(expression_data: Dict[str, Dict],
                                n_dims: int = 50,
                                seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Create embeddings from R values using multiple encodings.

    The embedding includes:
    - R value itself
    - log(R)
    - Sinusoidal position encoding based on R
    - Derived features (mean, std, R-based transformations)

    Returns: (embeddings array, list of gene IDs)
    """
    np.random.seed(seed)

    gene_ids = list(expression_data.keys())
    n_genes = len(gene_ids)

    embeddings = np.zeros((n_genes, n_dims))

    for i, gene_id in enumerate(gene_ids):
        data = expression_data[gene_id]
        R = data['R']
        mean_expr = data['mean_expr']
        std_expr = data['std_expr']

        # Feature 0-4: Basic statistics
        embeddings[i, 0] = R
        embeddings[i, 1] = np.log(R + 0.01)
        embeddings[i, 2] = mean_expr
        embeddings[i, 3] = std_expr
        embeddings[i, 4] = np.sqrt(R)

        # Feature 5-24: Sinusoidal position encoding based on R
        for d in range(5, min(25, n_dims)):
            freq = (d - 5 + 1) * 0.5
            if (d - 5) % 2 == 0:
                embeddings[i, d] = np.sin(freq * R / 10.0)
            else:
                embeddings[i, d] = np.cos(freq * R / 10.0)

        # Feature 25-49: Higher-order interactions and noise
        for d in range(25, n_dims):
            np.random.seed(i * n_dims + d + seed)
            # R-modulated random direction
            scale = 1.0 / (R + 0.1)
            embeddings[i, d] = np.sin((d - 25) * R / 20.0) + scale * np.random.randn() * 0.1

    # Normalize columns
    for d in range(n_dims):
        col_mean = embeddings[:, d].mean()
        col_std = embeddings[:, d].std()
        if col_std > 1e-10:
            embeddings[:, d] = (embeddings[:, d] - col_mean) / col_std

    return embeddings, gene_ids


def create_spectral_embedding(expression_data: Dict[str, Dict],
                               n_dims: int = 50,
                               max_samples: int = 5000,
                               seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Create spectral graph embedding based on R-similarity.

    Genes with similar R values are connected in a graph.
    Embedding is done via Laplacian eigenvectors.

    For large datasets, samples to max_samples to avoid memory issues.
    """
    np.random.seed(seed)

    gene_ids = list(expression_data.keys())
    n_genes_total = len(gene_ids)

    # Sample if too large (spectral embedding requires O(n^2) memory)
    if n_genes_total > max_samples:
        sample_indices = np.random.choice(n_genes_total, max_samples, replace=False)
        gene_ids = [gene_ids[i] for i in sample_indices]
        n_genes = max_samples
    else:
        n_genes = n_genes_total

    # Extract R values
    R_values = np.array([expression_data[g]['R'] for g in gene_ids])

    # Build adjacency matrix (k-nearest neighbors by R-distance)
    k = min(20, n_genes // 10)
    adj = np.zeros((n_genes, n_genes))

    for i in range(n_genes):
        distances = np.abs(R_values - R_values[i])
        neighbors = np.argsort(distances)[1:k+1]  # Exclude self
        for j in neighbors:
            similarity = np.exp(-distances[j] / (np.std(R_values) + 1e-10))
            adj[i, j] = similarity
            adj[j, i] = similarity

    # Compute normalized Laplacian
    degree = np.sum(adj, axis=1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
    L = np.diag(degree) - adj
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Spectral embedding: use smallest non-trivial eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Take first n_dims non-trivial eigenvectors
    n_components = min(n_dims, n_genes - 1)
    embeddings = eigenvectors[:, 1:n_components+1]

    # Pad if needed
    if embeddings.shape[1] < n_dims:
        padding = np.zeros((n_genes, n_dims - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, padding])

    return embeddings, gene_ids


def create_pca_embedding(expression_data: Dict[str, Dict],
                          n_dims: int = 50,
                          seed: int = 42,
                          max_samples: int = 10000) -> Tuple[np.ndarray, List[str]]:
    """
    Create embeddings via PCA of derived features.
    """
    np.random.seed(seed)

    gene_ids = list(expression_data.keys())
    n_genes_total = len(gene_ids)

    # Sample if too large
    if n_genes_total > max_samples:
        sample_indices = np.random.choice(n_genes_total, max_samples, replace=False)
        gene_ids = [gene_ids[i] for i in sample_indices]

    n_genes = len(gene_ids)

    # Create raw feature matrix
    features = []
    for gene_id in gene_ids:
        data = expression_data[gene_id]
        R = data['R']
        mean_expr = data['mean_expr']
        std_expr = data['std_expr']

        f = [
            R,
            np.log(R + 0.01),
            mean_expr,
            std_expr,
            R * mean_expr,
            R * std_expr,
            mean_expr / (std_expr + 1e-10),
            np.sqrt(R),
            R ** 2,
            np.sin(R),
            np.cos(R),
            np.exp(-R),
        ]
        features.append(f)

    features = np.array(features)

    # Normalize
    for i in range(features.shape[1]):
        col_std = np.std(features[:, i])
        if col_std > 1e-10:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / col_std

    # Expand to n_dims via random projection
    if features.shape[1] < n_dims:
        random_proj = np.random.randn(features.shape[1], n_dims) / np.sqrt(features.shape[1])
        features = features @ random_proj

    return features, gene_ids


# =============================================================================
# GEOMETRIC ANALYSIS
# =============================================================================

def compute_embedding_geometry(embeddings: np.ndarray, name: str) -> EmbeddingGeometry:
    """
    Compute geometric properties of an embedding space.
    """
    n_samples, n_dims = embeddings.shape

    # Center embeddings
    centroid = embeddings.mean(axis=0)
    centered = embeddings - centroid

    # Covariance matrix
    cov = (centered.T @ centered) / max(n_samples - 1, 1)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Intrinsic dimensionality (participation ratio)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # Spectral decay (alpha)
    if len(eigenvalues) >= 2:
        k = np.arange(1, len(eigenvalues) + 1)
        log_k = np.log(k)
        log_lambda = np.log(eigenvalues + 1e-10)

        n_pts = len(log_k)
        slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
        slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
        alpha = -slope
    else:
        alpha = 1.0

    # Df x alpha
    Df_x_alpha = Df * alpha
    deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E

    # Distances
    distances_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist_to_centroid = np.mean(distances_to_centroid)

    # Sample pairwise distances (for large n_samples)
    n_pairs = min(1000, n_samples * (n_samples - 1) // 2)
    pairwise_distances = []
    for _ in range(n_pairs):
        i, j = np.random.randint(0, n_samples, 2)
        if i != j:
            pairwise_distances.append(np.linalg.norm(embeddings[i] - embeddings[j]))
    mean_pairwise = np.mean(pairwise_distances) if pairwise_distances else 0.0

    return EmbeddingGeometry(
        name=name,
        n_samples=n_samples,
        embedding_dim=n_dims,
        intrinsic_dim=Df,
        spectral_decay=alpha,
        Df_x_alpha=Df_x_alpha,
        deviation_from_8e=deviation,
        eigenvalues=eigenvalues[:20].tolist(),
        centroid=centroid,
        mean_distance_to_centroid=mean_dist_to_centroid,
        mean_pairwise_distance=mean_pairwise
    )


def compute_procrustes_similarity(embeddings_1: np.ndarray,
                                   embeddings_2: np.ndarray,
                                   ids_1: List[str],
                                   ids_2: List[str],
                                   ortholog_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Compute Procrustes similarity between two embedding spaces.

    This finds the optimal rotation/reflection that aligns the two spaces
    and measures how well they match.
    """
    # Find matching pairs
    id_to_idx_1 = {id_: i for i, id_ in enumerate(ids_1)}
    id_to_idx_2 = {id_: i for i, id_ in enumerate(ids_2)}

    matched_1 = []
    matched_2 = []
    matched_pairs = []

    for human_id, mouse_id in ortholog_pairs:
        if human_id in id_to_idx_1 and mouse_id in id_to_idx_2:
            matched_1.append(embeddings_1[id_to_idx_1[human_id]])
            matched_2.append(embeddings_2[id_to_idx_2[mouse_id]])
            matched_pairs.append((human_id, mouse_id))

    if len(matched_1) < 10:
        return {
            'status': 'INSUFFICIENT_DATA',
            'n_matched': len(matched_1),
            'message': 'Need at least 10 matched pairs for Procrustes analysis'
        }

    X = np.array(matched_1)
    Y = np.array(matched_2)

    # Center both matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Ensure same dimensionality
    min_dim = min(X_centered.shape[1], Y_centered.shape[1])
    X_centered = X_centered[:, :min_dim]
    Y_centered = Y_centered[:, :min_dim]

    # SVD for optimal rotation
    M = Y_centered.T @ X_centered
    U, S, Vt = np.linalg.svd(M)

    # Optimal rotation matrix
    R = U @ Vt

    # Rotate Y to align with X
    Y_aligned = Y_centered @ R

    # Compute alignment error
    alignment_error = np.mean(np.linalg.norm(X_centered - Y_aligned, axis=1))

    # Normalize by average scale
    scale_X = np.mean(np.linalg.norm(X_centered, axis=1))
    scale_Y = np.mean(np.linalg.norm(Y_centered, axis=1))
    avg_scale = (scale_X + scale_Y) / 2

    normalized_error = alignment_error / (avg_scale + 1e-10)

    # Compute correlation of distances from centroid
    dist_X = np.linalg.norm(X_centered, axis=1)
    dist_Y = np.linalg.norm(Y_centered, axis=1)

    if np.std(dist_X) > 1e-10 and np.std(dist_Y) > 1e-10:
        distance_correlation = np.corrcoef(dist_X, dist_Y)[0, 1]
    else:
        distance_correlation = 0.0

    # Compute correlation of pairwise distances (sample)
    n_pairs = min(500, len(X) * (len(X) - 1) // 2)
    dist_pairs_X = []
    dist_pairs_Y = []

    for _ in range(n_pairs):
        i, j = np.random.randint(0, len(X), 2)
        if i != j:
            dist_pairs_X.append(np.linalg.norm(X_centered[i] - X_centered[j]))
            dist_pairs_Y.append(np.linalg.norm(Y_centered[i] - Y_centered[j]))

    if dist_pairs_X:
        pairwise_distance_correlation = np.corrcoef(dist_pairs_X, dist_pairs_Y)[0, 1]
    else:
        pairwise_distance_correlation = 0.0

    return {
        'status': 'COMPLETED',
        'n_matched': len(matched_1),
        'alignment_error': alignment_error,
        'normalized_error': normalized_error,
        'distance_correlation': distance_correlation,
        'pairwise_distance_correlation': pairwise_distance_correlation,
        'scale_X': scale_X,
        'scale_Y': scale_Y,
        'singular_values': S[:10].tolist() if len(S) >= 10 else S.tolist()
    }


def compute_neighborhood_preservation(embeddings_1: np.ndarray,
                                       embeddings_2: np.ndarray,
                                       ids_1: List[str],
                                       ids_2: List[str],
                                       ortholog_pairs: List[Tuple[str, str]],
                                       k: int = 10) -> Dict[str, Any]:
    """
    Test if nearest neighbors are preserved across species in embedding space.

    For each ortholog pair, check if the neighbors in human space
    correspond to neighbors in mouse space.
    """
    # Find matching pairs
    id_to_idx_1 = {id_: i for i, id_ in enumerate(ids_1)}
    id_to_idx_2 = {id_: i for i, id_ in enumerate(ids_2)}

    # Build ortholog mapping
    human_to_mouse = {}
    mouse_to_human = {}

    for human_id, mouse_id in ortholog_pairs:
        if human_id in id_to_idx_1 and mouse_id in id_to_idx_2:
            human_to_mouse[human_id] = mouse_id
            mouse_to_human[mouse_id] = human_id

    if len(human_to_mouse) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'n_pairs': len(human_to_mouse),
            'message': 'Need at least 20 ortholog pairs'
        }

    # For a sample of orthologs, check neighborhood preservation
    sample_genes = list(human_to_mouse.keys())[:min(100, len(human_to_mouse))]

    preservation_scores = []

    for human_gene in sample_genes:
        mouse_gene = human_to_mouse[human_gene]

        human_idx = id_to_idx_1[human_gene]
        mouse_idx = id_to_idx_2[mouse_gene]

        # Find k-nearest neighbors in human space
        human_distances = np.linalg.norm(embeddings_1 - embeddings_1[human_idx], axis=1)
        human_neighbors_idx = np.argsort(human_distances)[1:k+1]
        human_neighbors = [ids_1[i] for i in human_neighbors_idx]

        # Find k-nearest neighbors in mouse space
        mouse_distances = np.linalg.norm(embeddings_2 - embeddings_2[mouse_idx], axis=1)
        mouse_neighbors_idx = np.argsort(mouse_distances)[1:k+1]
        mouse_neighbors = [ids_2[i] for i in mouse_neighbors_idx]

        # Check how many human neighbors have orthologs among mouse neighbors
        preserved = 0
        for h_neighbor in human_neighbors:
            if h_neighbor in human_to_mouse:
                m_ortholog = human_to_mouse[h_neighbor]
                if m_ortholog in mouse_neighbors:
                    preserved += 1

        preservation_scores.append(preserved / k)

    mean_preservation = np.mean(preservation_scores)
    std_preservation = np.std(preservation_scores)

    # Random baseline: expected preservation by chance
    n_mouse_genes = len(ids_2)
    n_orthologs_in_mouse = len(mouse_to_human)
    expected_random = (k * n_orthologs_in_mouse) / (n_mouse_genes * k)

    return {
        'status': 'COMPLETED',
        'n_pairs_tested': len(sample_genes),
        'k_neighbors': k,
        'mean_preservation': mean_preservation,
        'std_preservation': std_preservation,
        'expected_random': expected_random,
        'improvement_over_random': mean_preservation / (expected_random + 1e-10),
        'significant': mean_preservation > 2 * expected_random
    }


def compute_embedding_correlation(geometry_1: EmbeddingGeometry,
                                   geometry_2: EmbeddingGeometry) -> Dict[str, Any]:
    """
    Compare geometric properties between two embedding spaces.

    This tests if the GEOMETRY is conserved across species,
    even if individual gene positions are not.
    """
    results = {
        'Df_comparison': {
            'human': geometry_1.intrinsic_dim,
            'mouse': geometry_2.intrinsic_dim,
            'ratio': geometry_1.intrinsic_dim / (geometry_2.intrinsic_dim + 1e-10),
            'difference': abs(geometry_1.intrinsic_dim - geometry_2.intrinsic_dim)
        },
        'alpha_comparison': {
            'human': geometry_1.spectral_decay,
            'mouse': geometry_2.spectral_decay,
            'ratio': geometry_1.spectral_decay / (geometry_2.spectral_decay + 1e-10),
            'difference': abs(geometry_1.spectral_decay - geometry_2.spectral_decay)
        },
        'Df_x_alpha_comparison': {
            'human': geometry_1.Df_x_alpha,
            'mouse': geometry_2.Df_x_alpha,
            'ratio': geometry_1.Df_x_alpha / (geometry_2.Df_x_alpha + 1e-10),
            'difference': abs(geometry_1.Df_x_alpha - geometry_2.Df_x_alpha),
            'both_near_8e': geometry_1.deviation_from_8e < 0.15 and geometry_2.deviation_from_8e < 0.15
        },
        'eigenvalue_correlation': None,
        'scale_comparison': {
            'human_mean_dist': geometry_1.mean_distance_to_centroid,
            'mouse_mean_dist': geometry_2.mean_distance_to_centroid,
            'ratio': geometry_1.mean_distance_to_centroid / (geometry_2.mean_distance_to_centroid + 1e-10)
        }
    }

    # Eigenvalue spectrum correlation
    n_eig = min(len(geometry_1.eigenvalues), len(geometry_2.eigenvalues))
    if n_eig >= 5:
        eig_1 = np.array(geometry_1.eigenvalues[:n_eig])
        eig_2 = np.array(geometry_2.eigenvalues[:n_eig])

        # Normalize eigenvalues
        eig_1 = eig_1 / (eig_1.sum() + 1e-10)
        eig_2 = eig_2 / (eig_2.sum() + 1e-10)

        eig_corr = np.corrcoef(eig_1, eig_2)[0, 1]
        results['eigenvalue_correlation'] = eig_corr

    return results


# =============================================================================
# ADDITIONAL EMBEDDING METHODS
# =============================================================================

def create_normalized_r_embedding(expression_data: Dict[str, Dict],
                                   n_dims: int = 50,
                                   seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Create embeddings where R values are normalized to unit distribution.

    This tests if cross-species transfer improves when we account for
    species-specific R scale differences.
    """
    np.random.seed(seed)

    gene_ids = list(expression_data.keys())
    n_genes = len(gene_ids)

    # Extract R values and normalize to [0, 1] range
    R_values = np.array([expression_data[g]['R'] for g in gene_ids])
    R_norm = (R_values - R_values.min()) / (R_values.max() - R_values.min() + 1e-10)

    embeddings = np.zeros((n_genes, n_dims))

    for i, r_norm in enumerate(R_norm):
        # Quantile-based embedding
        embeddings[i, 0] = r_norm
        embeddings[i, 1] = np.log(r_norm + 0.01)

        # Sinusoidal encoding on normalized R
        for d in range(2, n_dims):
            freq = (d - 1) * np.pi
            if d % 2 == 0:
                embeddings[i, d] = np.sin(freq * r_norm)
            else:
                embeddings[i, d] = np.cos(freq * r_norm)

    # Center and scale
    embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-10)

    return embeddings, gene_ids


def create_rank_embedding(expression_data: Dict[str, Dict],
                          n_dims: int = 50,
                          seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Create embeddings based on RANK of R values rather than raw values.

    This completely eliminates species-specific scale differences.
    """
    np.random.seed(seed)

    gene_ids = list(expression_data.keys())
    n_genes = len(gene_ids)

    # Extract R values and compute ranks
    R_values = np.array([expression_data[g]['R'] for g in gene_ids])
    ranks = np.argsort(np.argsort(R_values)) / n_genes  # Normalized ranks [0, 1]

    embeddings = np.zeros((n_genes, n_dims))

    for i, rank in enumerate(ranks):
        # Rank-based embedding
        for d in range(n_dims):
            freq = (d + 1) * np.pi
            if d % 2 == 0:
                embeddings[i, d] = np.sin(freq * rank)
            else:
                embeddings[i, d] = np.cos(freq * rank)

    return embeddings, gene_ids


def create_hybrid_embedding(expression_data: Dict[str, Dict],
                            n_dims: int = 50,
                            seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """
    Hybrid embedding combining raw features with rank-based encoding.

    This preserves R-specific structure while enabling cross-species comparison.
    """
    np.random.seed(seed)

    gene_ids = list(expression_data.keys())
    n_genes = len(gene_ids)

    # Extract features
    R_values = np.array([expression_data[g]['R'] for g in gene_ids])
    means = np.array([expression_data[g]['mean_expr'] for g in gene_ids])
    stds = np.array([expression_data[g]['std_expr'] for g in gene_ids])

    # Compute ranks
    R_ranks = np.argsort(np.argsort(R_values)) / n_genes
    mean_ranks = np.argsort(np.argsort(means)) / n_genes
    std_ranks = np.argsort(np.argsort(stds)) / n_genes

    embeddings = np.zeros((n_genes, n_dims))

    # First 10 dims: raw normalized features
    embeddings[:, 0] = (R_values - R_values.mean()) / (R_values.std() + 1e-10)
    embeddings[:, 1] = (means - means.mean()) / (means.std() + 1e-10)
    embeddings[:, 2] = (stds - stds.mean()) / (stds.std() + 1e-10)
    embeddings[:, 3] = np.log(R_values + 0.01)
    embeddings[:, 4] = R_ranks
    embeddings[:, 5] = mean_ranks
    embeddings[:, 6] = std_ranks
    embeddings[:, 7] = R_ranks * mean_ranks
    embeddings[:, 8] = np.sin(np.pi * R_ranks)
    embeddings[:, 9] = np.cos(np.pi * R_ranks)

    # Remaining dims: positional encoding based on ranks
    for d in range(10, n_dims):
        freq = (d - 9) * 0.5
        if d % 2 == 0:
            embeddings[:, d] = np.sin(freq * R_ranks * 2 * np.pi)
        else:
            embeddings[:, d] = np.cos(freq * R_ranks * 2 * np.pi)

    return embeddings, gene_ids


def compute_geometric_invariants(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute geometric invariants that should be conserved across species
    if there is a universal structure.
    """
    n_samples, n_dims = embeddings.shape

    # Center
    centered = embeddings - embeddings.mean(axis=0)

    # Covariance eigenspectrum
    cov = (centered.T @ centered) / max(n_samples - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Invariant 1: Df (participation ratio)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # Invariant 2: Alpha (spectral decay)
    if len(eigenvalues) >= 2:
        k = np.arange(1, len(eigenvalues) + 1)
        log_k = np.log(k)
        log_lambda = np.log(eigenvalues + 1e-10)
        n_pts = len(log_k)
        slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
        slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
        alpha = -slope
    else:
        alpha = 1.0

    # Invariant 3: Effective dimensionality ratio
    # (top eigenvalue / mean eigenvalue)
    effective_dim_ratio = eigenvalues[0] / (np.mean(eigenvalues) + 1e-10) if len(eigenvalues) > 0 else 1.0

    # Invariant 4: Entropy of normalized eigenspectrum
    norm_eig = eigenvalues / (sum_lambda + 1e-10)
    entropy = -np.sum(norm_eig * np.log(norm_eig + 1e-10))

    # Invariant 5: Ratio of top 3 eigenvalues
    if len(eigenvalues) >= 3:
        eig_ratio_12 = eigenvalues[0] / (eigenvalues[1] + 1e-10)
        eig_ratio_23 = eigenvalues[1] / (eigenvalues[2] + 1e-10)
    else:
        eig_ratio_12 = 1.0
        eig_ratio_23 = 1.0

    return {
        'Df': Df,
        'alpha': alpha,
        'Df_x_alpha': Df * alpha,
        'effective_dim_ratio': effective_dim_ratio,
        'spectral_entropy': entropy,
        'eig_ratio_12': eig_ratio_12,
        'eig_ratio_23': eig_ratio_23
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def build_ortholog_mapping(human_annot: Dict[str, str],
                           mouse_annot: Dict[str, str],
                           orthologs: List[Dict],
                           human_expr: Dict[str, Dict],
                           mouse_expr: Dict[str, Dict]) -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
    """
    Build mapping from probe IDs to gene symbols and ortholog pairs.
    """
    # Build probe -> gene symbol mappings
    human_probe_to_gene = {}
    for probe_key in human_expr.keys():
        # Remove dataset prefix (GSE13904:1007_s_at -> 1007_s_at)
        probe_id = probe_key.split(':')[-1]
        if probe_id in human_annot:
            human_probe_to_gene[probe_key] = human_annot[probe_id].upper()

    mouse_probe_to_gene = {}
    for probe_key in mouse_expr.keys():
        # Remove dataset prefix (GSE3431:10000_at -> 10000_at)
        probe_id = probe_key.split(':')[-1]
        if probe_id in mouse_annot:
            mouse_probe_to_gene[probe_key] = mouse_annot[probe_id].upper()

    # Build ortholog pairs (human_gene -> mouse_gene)
    ortholog_map = {}
    for o in orthologs:
        h_name = o['human_gene_name'].upper()
        m_name = o['mouse_gene_name'].upper()
        if h_name and m_name:
            ortholog_map[h_name] = m_name

    # Also add same-name genes (most 1:1 orthologs share names)
    human_genes = set(human_probe_to_gene.values())
    mouse_genes = set(mouse_probe_to_gene.values())
    same_name = human_genes & mouse_genes

    for gene in same_name:
        if gene not in ortholog_map:
            ortholog_map[gene] = gene

    # Build ortholog pairs with probe IDs
    ortholog_pairs = []

    for h_probe, h_gene in human_probe_to_gene.items():
        if h_gene in ortholog_map:
            m_gene = ortholog_map[h_gene]
            # Find mouse probe with this gene
            for m_probe, m_gene_found in mouse_probe_to_gene.items():
                if m_gene_found == m_gene:
                    ortholog_pairs.append((h_probe, m_probe))
                    break

    return human_probe_to_gene, mouse_probe_to_gene, ortholog_pairs


def run_cross_species_embedding_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Main test: Compare human and mouse gene expression in embedding space.
    """
    if verbose:
        print("=" * 80)
        print("Q18 INVESTIGATION: CROSS-SPECIES TRANSFER IN EMBEDDING SPACE")
        print("=" * 80)
        print()
        print("HYPOTHESIS: Cross-species transfer works in EMBEDDING space,")
        print("            not raw R space (which failed with r=0.054)")
        print()
        print(f"Theoretical 8e: {EIGHT_E:.4f}")
        print("=" * 80)

    # Load data
    if verbose:
        print("\n[1] LOADING DATA")
        print("-" * 40)

    human_expr = load_human_expression()
    mouse_expr = load_mouse_expression()
    orthologs = load_orthologs()
    human_annot, mouse_annot = load_probe_annotations()

    if verbose:
        print(f"  Human probes: {len(human_expr)}")
        print(f"  Mouse probes: {len(mouse_expr)}")
        print(f"  Orthologs: {len(orthologs)}")

    # Build ortholog mapping
    human_probe_to_gene, mouse_probe_to_gene, ortholog_pairs = build_ortholog_mapping(
        human_annot, mouse_annot, orthologs, human_expr, mouse_expr
    )

    if verbose:
        print(f"  Human probes with gene mapping: {len(human_probe_to_gene)}")
        print(f"  Mouse probes with gene mapping: {len(mouse_probe_to_gene)}")
        print(f"  Matched ortholog pairs: {len(ortholog_pairs)}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Cross-species transfer works in embedding space, not raw R space',
        'data_summary': {
            'human_probes': len(human_expr),
            'mouse_probes': len(mouse_expr),
            'ortholog_pairs': len(ortholog_pairs)
        },
        'embedding_methods': {},
        'cross_species_comparisons': {},
        'key_findings': []
    }

    # Test multiple embedding methods
    embedding_methods = [
        ('r_feature', create_r_feature_embedding),
        ('spectral', create_spectral_embedding),
        ('pca', create_pca_embedding),
        ('normalized_r', create_normalized_r_embedding),
        ('rank_based', create_rank_embedding),
        ('hybrid', create_hybrid_embedding),
    ]

    for method_name, embed_func in embedding_methods:
        if verbose:
            print(f"\n[2] CREATING {method_name.upper()} EMBEDDINGS")
            print("-" * 40)

        # Create embeddings
        human_embeddings, human_ids = embed_func(human_expr, n_dims=50)
        mouse_embeddings, mouse_ids = embed_func(mouse_expr, n_dims=50)

        if verbose:
            print(f"  Human embedding shape: {human_embeddings.shape}")
            print(f"  Mouse embedding shape: {mouse_embeddings.shape}")

        # Compute geometry
        human_geometry = compute_embedding_geometry(human_embeddings, f"human_{method_name}")
        mouse_geometry = compute_embedding_geometry(mouse_embeddings, f"mouse_{method_name}")

        # Compute geometric invariants
        human_invariants = compute_geometric_invariants(human_embeddings)
        mouse_invariants = compute_geometric_invariants(mouse_embeddings)

        results['embedding_methods'][method_name] = {
            'human': {
                'n_samples': human_geometry.n_samples,
                'intrinsic_dim': human_geometry.intrinsic_dim,
                'spectral_decay': human_geometry.spectral_decay,
                'Df_x_alpha': human_geometry.Df_x_alpha,
                'deviation_from_8e': human_geometry.deviation_from_8e,
                'invariants': to_builtin(human_invariants)
            },
            'mouse': {
                'n_samples': mouse_geometry.n_samples,
                'intrinsic_dim': mouse_geometry.intrinsic_dim,
                'spectral_decay': mouse_geometry.spectral_decay,
                'Df_x_alpha': mouse_geometry.Df_x_alpha,
                'deviation_from_8e': mouse_geometry.deviation_from_8e,
                'invariants': to_builtin(mouse_invariants)
            }
        }

        if verbose:
            print(f"  Human: Df={human_geometry.intrinsic_dim:.2f}, alpha={human_geometry.spectral_decay:.4f}, "
                  f"Df x alpha={human_geometry.Df_x_alpha:.2f} ({human_geometry.deviation_from_8e*100:.1f}% dev from 8e)")
            print(f"  Mouse: Df={mouse_geometry.intrinsic_dim:.2f}, alpha={mouse_geometry.spectral_decay:.4f}, "
                  f"Df x alpha={mouse_geometry.Df_x_alpha:.2f} ({mouse_geometry.deviation_from_8e*100:.1f}% dev from 8e)")

        # Procrustes analysis
        if verbose:
            print(f"\n  Computing Procrustes alignment...")

        procrustes_result = compute_procrustes_similarity(
            human_embeddings, mouse_embeddings,
            human_ids, mouse_ids,
            ortholog_pairs
        )

        # Neighborhood preservation
        if verbose:
            print(f"  Computing neighborhood preservation...")

        neighborhood_result = compute_neighborhood_preservation(
            human_embeddings, mouse_embeddings,
            human_ids, mouse_ids,
            ortholog_pairs,
            k=10
        )

        # Geometry correlation
        geometry_comparison = compute_embedding_correlation(human_geometry, mouse_geometry)

        results['cross_species_comparisons'][method_name] = {
            'procrustes': to_builtin(procrustes_result),
            'neighborhood': to_builtin(neighborhood_result),
            'geometry': to_builtin(geometry_comparison)
        }

        if verbose:
            if procrustes_result.get('status') == 'COMPLETED':
                print(f"  Procrustes: normalized_error={procrustes_result['normalized_error']:.4f}, "
                      f"distance_corr={procrustes_result['distance_correlation']:.4f}")

            if neighborhood_result.get('status') == 'COMPLETED':
                print(f"  Neighborhood: preservation={neighborhood_result['mean_preservation']:.4f}, "
                      f"vs random={neighborhood_result['expected_random']:.4f}")

            print(f"  Geometry: Df_ratio={geometry_comparison['Df_comparison']['ratio']:.3f}, "
                  f"alpha_ratio={geometry_comparison['alpha_comparison']['ratio']:.3f}")

    # Key findings
    if verbose:
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)

    # Check if 8e emerges in both species
    for method_name, method_results in results['embedding_methods'].items():
        h_dev = method_results['human']['deviation_from_8e']
        m_dev = method_results['mouse']['deviation_from_8e']

        if h_dev < 0.15 and m_dev < 0.15:
            finding = f"{method_name}: BOTH species show 8e conservation (<15% deviation)"
            results['key_findings'].append(finding)
            if verbose:
                print(f"\n  *** {finding} ***")
        elif h_dev < 0.15 or m_dev < 0.15:
            finding = f"{method_name}: Only one species shows 8e (human: {h_dev*100:.1f}%, mouse: {m_dev*100:.1f}%)"
            results['key_findings'].append(finding)
            if verbose:
                print(f"  {finding}")

    # Check if embedding geometry is conserved
    for method_name, comparison in results['cross_species_comparisons'].items():
        geom = comparison.get('geometry', {})

        df_ratio = geom.get('Df_comparison', {}).get('ratio', 0)
        alpha_ratio = geom.get('alpha_comparison', {}).get('ratio', 0)
        eig_corr = geom.get('eigenvalue_correlation', 0)

        if 0.8 < df_ratio < 1.2 and 0.8 < alpha_ratio < 1.2:
            finding = f"{method_name}: Embedding GEOMETRY is conserved across species (Df ratio: {df_ratio:.2f}, alpha ratio: {alpha_ratio:.2f})"
            results['key_findings'].append(finding)
            if verbose:
                print(f"\n  *** {finding} ***")

        if eig_corr and eig_corr > 0.7:
            finding = f"{method_name}: Eigenvalue spectra are highly correlated (r={eig_corr:.3f})"
            results['key_findings'].append(finding)
            if verbose:
                print(f"  {finding}")

    # Check neighborhood preservation
    for method_name, comparison in results['cross_species_comparisons'].items():
        neigh = comparison.get('neighborhood', {})

        if neigh.get('status') == 'COMPLETED':
            improvement = neigh.get('improvement_over_random', 0)
            if improvement > 2:
                finding = f"{method_name}: Neighborhood structure is preserved ({improvement:.1f}x better than random)"
                results['key_findings'].append(finding)
                if verbose:
                    print(f"\n  *** {finding} ***")

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print(f"\n  Methods showing 8e in BOTH species:")
        for method_name, method_results in results['embedding_methods'].items():
            h_dxa = method_results['human']['Df_x_alpha']
            m_dxa = method_results['mouse']['Df_x_alpha']
            h_dev = method_results['human']['deviation_from_8e']
            m_dev = method_results['mouse']['deviation_from_8e']

            h_pass = "PASS" if h_dev < 0.15 else "FAIL"
            m_pass = "PASS" if m_dev < 0.15 else "FAIL"

            print(f"    {method_name}: Human {h_dxa:.2f} ({h_pass}), Mouse {m_dxa:.2f} ({m_pass})")

        if not results['key_findings']:
            print("\n  No significant cross-species conservation found in embedding space.")
            print("  This suggests 8e may be a property of embedding construction,")
            print("  not an inherent property of biological data structure.")
        else:
            print(f"\n  Total significant findings: {len(results['key_findings'])}")

        print("=" * 80)

    return to_builtin(results)


def analyze_cross_species_findings(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the results and provide interpretation of what was found.
    """
    analysis = {
        'hypothesis_tested': 'Cross-species transfer works in embedding space',
        'verdict': None,
        'evidence': [],
        'interpretation': [],
        'key_constants': {}
    }

    # Check for 8e conservation across species
    eight_e_methods = []
    geometry_conserved_methods = []

    for method, data in results.get('embedding_methods', {}).items():
        h_dev = data['human']['deviation_from_8e']
        m_dev = data['mouse']['deviation_from_8e']

        if h_dev < 0.15 and m_dev < 0.15:
            eight_e_methods.append(method)

        # Check geometry conservation
        h_df = data['human']['intrinsic_dim']
        m_df = data['mouse']['intrinsic_dim']
        h_alpha = data['human']['spectral_decay']
        m_alpha = data['mouse']['spectral_decay']

        df_ratio = h_df / (m_df + 1e-10)
        alpha_ratio = h_alpha / (m_alpha + 1e-10)

        if 0.8 < df_ratio < 1.2 and 0.8 < alpha_ratio < 1.2:
            geometry_conserved_methods.append({
                'method': method,
                'df_ratio': df_ratio,
                'alpha_ratio': alpha_ratio
            })

    # Check eigenvalue correlations
    high_corr_methods = []
    for method, comparison in results.get('cross_species_comparisons', {}).items():
        geom = comparison.get('geometry', {})
        eig_corr = geom.get('eigenvalue_correlation', 0)
        if eig_corr and eig_corr > 0.9:
            high_corr_methods.append({
                'method': method,
                'correlation': eig_corr
            })

    # Build analysis
    if eight_e_methods:
        analysis['evidence'].append(f"8e emerges in BOTH species for: {', '.join(eight_e_methods)}")

    if geometry_conserved_methods:
        analysis['evidence'].append(f"Embedding geometry conserved in: {len(geometry_conserved_methods)} methods")
        for m in geometry_conserved_methods:
            analysis['evidence'].append(f"  - {m['method']}: Df_ratio={m['df_ratio']:.3f}, alpha_ratio={m['alpha_ratio']:.3f}")

    if high_corr_methods:
        analysis['evidence'].append(f"Eigenvalue spectra highly correlated (r>0.9) in: {len(high_corr_methods)} methods")

    # Interpretation
    if geometry_conserved_methods or high_corr_methods:
        analysis['interpretation'].append(
            "GLOBAL embedding geometry is conserved across species, suggesting a universal "
            "structure in how genes are organized by their expression stability (R values)."
        )

    # Check if neighborhood preservation failed
    failed_neighborhoods = True
    for method, comparison in results.get('cross_species_comparisons', {}).items():
        neigh = comparison.get('neighborhood', {})
        if neigh.get('status') == 'COMPLETED':
            if neigh.get('improvement_over_random', 0) > 2:
                failed_neighborhoods = False
                break

    if failed_neighborhoods:
        analysis['interpretation'].append(
            "LOCAL structure (gene neighborhoods) is NOT preserved across species. "
            "This means while the SHAPE of embedding space is similar, individual genes "
            "do not occupy corresponding positions."
        )

    # Key constant analysis
    pca_data = results.get('embedding_methods', {}).get('pca', {})
    if pca_data:
        h_dxa = pca_data['human']['Df_x_alpha']
        m_dxa = pca_data['mouse']['Df_x_alpha']
        avg_dxa = (h_dxa + m_dxa) / 2

        analysis['key_constants']['Df_x_alpha_human'] = h_dxa
        analysis['key_constants']['Df_x_alpha_mouse'] = m_dxa
        analysis['key_constants']['Df_x_alpha_avg'] = avg_dxa
        analysis['key_constants']['deviation_from_8e'] = abs(avg_dxa - EIGHT_E) / EIGHT_E

        if abs(avg_dxa - EIGHT_E) / EIGHT_E < 0.15:
            analysis['interpretation'].append(
                f"PCA embedding shows Df x alpha ~ {avg_dxa:.2f} (close to 8e={EIGHT_E:.2f}), "
                "suggesting 8e may be a property of properly structured embeddings."
            )

    # Verdict
    if eight_e_methods and geometry_conserved_methods:
        analysis['verdict'] = "SUPPORTED"
        analysis['interpretation'].append(
            "CONCLUSION: Cross-species transfer DOES work in embedding space at the GLOBAL level. "
            "Both species show similar spectral geometry (Df, alpha) and eigenvalue structure. "
            "However, point-wise correspondence is not preserved."
        )
    elif geometry_conserved_methods or high_corr_methods:
        analysis['verdict'] = "PARTIALLY_SUPPORTED"
        analysis['interpretation'].append(
            "CONCLUSION: Embedding geometry is conserved but 8e specifically does not emerge "
            "consistently. The universal structure may be more about spectral decay patterns "
            "than about the specific 8e constant."
        )
    else:
        analysis['verdict'] = "NOT_SUPPORTED"
        analysis['interpretation'].append(
            "CONCLUSION: No clear evidence that embedding space transfers better than raw R values."
        )

    return analysis


def main():
    """Main entry point."""
    results = run_cross_species_embedding_test(verbose=True)

    # Add detailed analysis
    analysis = analyze_cross_species_findings(results)
    results['analysis'] = analysis

    # Print analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    print(f"\nVERDICT: {analysis['verdict']}")

    print("\nEVIDENCE:")
    for e in analysis['evidence']:
        print(f"  {e}")

    print("\nINTERPRETATION:")
    for i in analysis['interpretation']:
        print(f"\n  {i}")

    print("\nKEY CONSTANTS:")
    for k, v in analysis.get('key_constants', {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("=" * 80)

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(to_builtin(results), f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    main()
