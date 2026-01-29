#!/usr/bin/env python3
"""
Q18 Real Data: 8e Conservation Law Test with Gene Expression Data

Tests if Df x alpha = 8e holds for real biological gene expression data from GEO.

HYPOTHESIS: 8e (~21.746) should NOT hold for raw biological data.
The 8e conservation law is expected to emerge only in trained embeddings
(neural networks, protein structure models, etc.), not in raw measurements.

Data source: GEO gene expression sample (2,500 genes with R values)

Author: Claude
Date: 2026-01-25
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import sys

# Constants
EIGHT_E = 8 * np.e  # ~21.746


def load_gene_expression_data(filepath: str) -> Dict[str, Any]:
    """Load gene expression data from JSON cache."""
    with open(filepath, 'r') as f:
        return json.load(f)


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


def construct_covariance_matrix_from_R(genes_data: Dict[str, Dict]) -> np.ndarray:
    """
    Construct a covariance-like matrix from R values.

    R = mean/std, so we can reconstruct variance-like quantities.
    We'll create a synthetic covariance structure based on R patterns.
    """
    gene_ids = list(genes_data.keys())
    n_genes = len(gene_ids)

    # Extract R values, means, and stds
    R_values = []
    means = []
    stds = []

    for gene_id in gene_ids:
        R_values.append(genes_data[gene_id]['R'])
        means.append(genes_data[gene_id]['mean_expr'])
        stds.append(genes_data[gene_id]['std_expr'])

    R_values = np.array(R_values)
    means = np.array(means)
    stds = np.array(stds)

    # Create feature matrix: each gene is a "sample" with features
    # Features: [R, mean, std, log(R), mean/R, R*mean]
    # This mimics how gene expression data would create correlations

    features = np.zeros((n_genes, 6))
    features[:, 0] = R_values
    features[:, 1] = means
    features[:, 2] = stds
    features[:, 3] = np.log(R_values + 1e-10)
    features[:, 4] = means / (R_values + 1e-10)
    features[:, 5] = R_values * means

    # Normalize features
    for i in range(features.shape[1]):
        col_std = np.std(features[:, i])
        if col_std > 1e-10:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / col_std

    # Compute covariance matrix
    cov = np.cov(features.T)

    return cov, features, R_values


def construct_R_based_covariance(R_values: np.ndarray, n_dims: int = 50) -> np.ndarray:
    """
    Create a higher-dimensional covariance matrix based on R value patterns.

    This simulates what we might see from gene co-expression networks.
    """
    n_genes = len(R_values)

    # Sort R values for structured analysis
    sorted_idx = np.argsort(R_values)[::-1]
    R_sorted = R_values[sorted_idx]

    # Create synthetic embedding based on R
    # Each gene gets a position in n_dims space based on its R
    embeddings = np.zeros((n_genes, n_dims))

    for i, r in enumerate(R_values):
        # Position in embedding space influenced by R
        # Low R (high variance) = spread out
        # High R (low variance) = clustered
        np.random.seed(i)

        # Scale factor based on R (inverse relationship)
        scale = 1.0 / (r + 0.1)

        # Random direction modulated by R
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Position: base + R-modulated spread
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    # Compute covariance of embeddings
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.dot(centered.T, centered) / max(n_genes - 1, 1)

    return cov, embeddings


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


def compute_direct_R_spectral(R_values: np.ndarray) -> Tuple[float, float]:
    """
    Compute spectral properties directly from R value distribution.

    Treats R values as a 1D "spectrum" and computes equivalent Df and alpha.
    """
    # Sort R values as pseudo-eigenvalues
    R_sorted = np.sort(R_values)[::-1]
    R_sorted = R_sorted[R_sorted > 0]

    if len(R_sorted) < 2:
        return 1.0, 1.0

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

    return Df_R, alpha_R


def run_8e_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Run the 8e conservation law test on real gene expression data.
    """
    # Load data
    cache_path = Path(__file__).parent / "cache" / "gene_expression_sample.json"

    if verbose:
        print("=" * 70)
        print("Q18 REAL DATA: 8e CONSERVATION LAW TEST")
        print("Testing Df x alpha = 8e on Gene Expression Data")
        print("=" * 70)
        print(f"\nLoading data from: {cache_path}")

    data = load_gene_expression_data(str(cache_path))
    genes_data = data['genes']
    n_genes = len(genes_data)

    if verbose:
        print(f"Loaded {n_genes} genes")

        # Show sample data
        sample_genes = list(genes_data.items())[:5]
        print("\nSample gene data:")
        for gene_id, info in sample_genes:
            print(f"  {gene_id}: R={info['R']:.4f}, mean={info['mean_expr']:.4f}, std={info['std_expr']:.4f}")

    # Extract R values
    R_values = np.array([g['R'] for g in genes_data.values()])

    if verbose:
        print(f"\nR value statistics:")
        print(f"  Mean R: {np.mean(R_values):.4f}")
        print(f"  Std R: {np.std(R_values):.4f}")
        print(f"  Min R: {np.min(R_values):.4f}")
        print(f"  Max R: {np.max(R_values):.4f}")
        print(f"  Median R: {np.median(R_values):.4f}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_genes": n_genes,
        "theoretical_8e": float(EIGHT_E),
        "R_statistics": {
            "mean": float(np.mean(R_values)),
            "std": float(np.std(R_values)),
            "min": float(np.min(R_values)),
            "max": float(np.max(R_values)),
            "median": float(np.median(R_values))
        },
        "methods": {}
    }

    # Method 1: Direct R-value spectral analysis
    if verbose:
        print("\n" + "-" * 70)
        print("METHOD 1: Direct R-value Spectral Analysis")
        print("-" * 70)

    Df_direct, alpha_direct = compute_direct_R_spectral(R_values)
    product_direct = Df_direct * alpha_direct
    deviation_direct = abs(product_direct - EIGHT_E) / EIGHT_E

    if verbose:
        print(f"  Df (from R values): {Df_direct:.4f}")
        print(f"  alpha (from R decay): {alpha_direct:.4f}")
        print(f"  Df x alpha: {product_direct:.4f}")
        print(f"  Theoretical 8e: {EIGHT_E:.4f}")
        print(f"  Deviation from 8e: {deviation_direct*100:.1f}%")

    results["methods"]["direct_R"] = {
        "Df": float(Df_direct),
        "alpha": float(alpha_direct),
        "Df_x_alpha": float(product_direct),
        "deviation_from_8e": float(deviation_direct),
        "deviation_percent": float(deviation_direct * 100)
    }

    # Method 2: Small feature covariance (6 features)
    if verbose:
        print("\n" + "-" * 70)
        print("METHOD 2: Feature Covariance Matrix (6 features)")
        print("-" * 70)

    cov_small, features, _ = construct_covariance_matrix_from_R(genes_data)
    Df_small, alpha_small, eigs_small = compute_spectral_properties(cov_small)
    product_small = Df_small * alpha_small
    deviation_small = abs(product_small - EIGHT_E) / EIGHT_E

    if verbose:
        print(f"  Covariance matrix shape: {cov_small.shape}")
        print(f"  Top 3 eigenvalues: {eigs_small[:3]}")
        print(f"  Df: {Df_small:.4f}")
        print(f"  alpha: {alpha_small:.4f}")
        print(f"  Df x alpha: {product_small:.4f}")
        print(f"  Deviation from 8e: {deviation_small*100:.1f}%")

    results["methods"]["feature_covariance"] = {
        "matrix_shape": list(cov_small.shape),
        "top_eigenvalues": [float(e) for e in eigs_small[:10]],
        "Df": float(Df_small),
        "alpha": float(alpha_small),
        "Df_x_alpha": float(product_small),
        "deviation_from_8e": float(deviation_small),
        "deviation_percent": float(deviation_small * 100)
    }

    # Method 3: Higher-dimensional R-based embedding
    if verbose:
        print("\n" + "-" * 70)
        print("METHOD 3: R-based Embedding Covariance (50 dimensions)")
        print("-" * 70)

    cov_embed, embeddings = construct_R_based_covariance(R_values, n_dims=50)
    Df_embed, alpha_embed, eigs_embed = compute_spectral_properties(cov_embed)
    product_embed = Df_embed * alpha_embed
    deviation_embed = abs(product_embed - EIGHT_E) / EIGHT_E

    if verbose:
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Covariance matrix shape: {cov_embed.shape}")
        print(f"  Top 5 eigenvalues: {eigs_embed[:5]}")
        print(f"  Df: {Df_embed:.4f}")
        print(f"  alpha: {alpha_embed:.4f}")
        print(f"  Df x alpha: {product_embed:.4f}")
        print(f"  Deviation from 8e: {deviation_embed*100:.1f}%")

    results["methods"]["R_embedding"] = {
        "embedding_dims": 50,
        "n_samples": n_genes,
        "top_eigenvalues": [float(e) for e in eigs_embed[:10]],
        "Df": float(Df_embed),
        "alpha": float(alpha_embed),
        "Df_x_alpha": float(product_embed),
        "deviation_from_8e": float(deviation_embed),
        "deviation_percent": float(deviation_embed * 100)
    }

    # Method 4: Gene-gene correlation matrix
    if verbose:
        print("\n" + "-" * 70)
        print("METHOD 4: Gene-Gene Correlation (from R ratios)")
        print("-" * 70)

    # Sample subset for computational tractability
    n_sample = min(500, n_genes)
    sample_indices = np.random.choice(n_genes, n_sample, replace=False)
    R_sample = R_values[sample_indices]

    # Create correlation-like matrix based on R similarity
    R_matrix = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(n_sample):
            # Correlation based on R ratio similarity
            r_ratio = min(R_sample[i], R_sample[j]) / (max(R_sample[i], R_sample[j]) + 1e-10)
            R_matrix[i, j] = r_ratio

    Df_corr, alpha_corr, eigs_corr = compute_spectral_properties(R_matrix)
    product_corr = Df_corr * alpha_corr
    deviation_corr = abs(product_corr - EIGHT_E) / EIGHT_E

    if verbose:
        print(f"  Sampled genes: {n_sample}")
        print(f"  Correlation matrix shape: {R_matrix.shape}")
        print(f"  Top 5 eigenvalues: {eigs_corr[:5]}")
        print(f"  Df: {Df_corr:.4f}")
        print(f"  alpha: {alpha_corr:.4f}")
        print(f"  Df x alpha: {product_corr:.4f}")
        print(f"  Deviation from 8e: {deviation_corr*100:.1f}%")

    results["methods"]["gene_correlation"] = {
        "n_sampled": n_sample,
        "top_eigenvalues": [float(e) for e in eigs_corr[:10]],
        "Df": float(Df_corr),
        "alpha": float(alpha_corr),
        "Df_x_alpha": float(product_corr),
        "deviation_from_8e": float(deviation_corr),
        "deviation_percent": float(deviation_corr * 100)
    }

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: 8e CONSERVATION TEST RESULTS")
        print("=" * 70)
        print(f"\nTheoretical 8e value: {EIGHT_E:.4f}")
        print("\nMethod comparison:")
        print("-" * 50)
        print(f"{'Method':<25} {'Df x alpha':<12} {'Deviation':<12}")
        print("-" * 50)

        for method_name, method_data in results["methods"].items():
            print(f"{method_name:<25} {method_data['Df_x_alpha']:<12.4f} {method_data['deviation_percent']:<12.1f}%")

    # Calculate average deviation
    deviations = [m["deviation_from_8e"] for m in results["methods"].values()]
    products = [m["Df_x_alpha"] for m in results["methods"].values()]

    avg_deviation = np.mean(deviations)
    avg_product = np.mean(products)

    results["summary"] = {
        "average_Df_x_alpha": float(avg_product),
        "average_deviation_from_8e": float(avg_deviation),
        "average_deviation_percent": float(avg_deviation * 100),
        "8e_holds": avg_deviation < 0.20,  # Within 20% threshold
        "interpretation": ""
    }

    # Interpretation
    if avg_deviation > 0.50:
        interpretation = (
            "STRONG DEVIATION: Raw biological data shows Df x alpha significantly different from 8e. "
            "This supports the hypothesis that 8e conservation emerges from trained representations, "
            "not raw biological measurements. The deviation suggests biological R values lack the "
            "spectral structure that produces 8e in neural embeddings."
        )
    elif avg_deviation > 0.20:
        interpretation = (
            "MODERATE DEVIATION: Gene expression data shows notable deviation from 8e. "
            "This partial deviation is consistent with biological data having some but not all "
            "of the structural properties needed for 8e conservation."
        )
    else:
        interpretation = (
            "WEAK DEVIATION: Surprisingly, gene expression data approaches 8e conservation. "
            "This could indicate either: (1) biological R values share structural properties with "
            "trained embeddings, or (2) the relationship is more universal than hypothesized."
        )

    results["summary"]["interpretation"] = interpretation

    if verbose:
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        print(f"\nAverage Df x alpha across methods: {avg_product:.4f}")
        print(f"Average deviation from 8e: {avg_deviation*100:.1f}%")
        print(f"\n{interpretation}")
        print("\n" + "=" * 70)

    return results


def main():
    """Main entry point."""
    results = run_8e_test(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "8e_gene_expression_results.json"
    with open(output_path, 'w') as f:
        json.dump(to_builtin(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
