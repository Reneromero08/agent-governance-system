#!/usr/bin/env python3
"""
Verify the original 8e claim using the ACTUAL gene expression data.
Compare with audit results using simulated data.
"""

import json
import numpy as np
from pathlib import Path

EIGHT_E = 8 * np.e

def compute_from_embeddings(embeddings: np.ndarray):
    """Compute Df and alpha from embeddings."""
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

    # alpha: Power-law fit
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


# Load actual gene expression data
cache_path = Path(__file__).parent.parent / "real_data" / "cache" / "gene_expression_sample.json"
with open(cache_path, 'r') as f:
    data = json.load(f)

genes_data = data['genes']
gene_ids = list(genes_data.keys())
n_genes = len(gene_ids)

R_values = np.array([genes_data[g]['R'] for g in gene_ids])

print("="*80)
print("VERIFICATION OF ORIGINAL 8e CLAIM")
print("="*80)
print(f"\nUsing ACTUAL gene expression data: {n_genes} genes")
print(f"R statistics:")
print(f"  Mean: {R_values.mean():.2f}")
print(f"  Std:  {R_values.std():.2f}")
print(f"  Min:  {R_values.min():.2f}")
print(f"  Max:  {R_values.max():.2f}")

# Run the EXACT embedding from the original code
n_dims = 50
seed = 42

print(f"\n" + "-"*80)
print("EXACT REPLICATION of original sin_r_full embedding")
print("-"*80)

embeddings = np.zeros((n_genes, n_dims))
for i, r in enumerate(R_values):
    np.random.seed(i + seed)
    scale = 1.0 / (r + 0.1)
    direction = np.random.randn(n_dims)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    base_pos = np.sin(np.arange(n_dims) * r / 10.0)
    embeddings[i] = base_pos + scale * direction

Df, alpha, eigenvalues = compute_from_embeddings(embeddings)
product = Df * alpha
deviation = abs(product - EIGHT_E) / EIGHT_E * 100

print(f"\nResults:")
print(f"  Df = {Df:.4f}")
print(f"  alpha = {alpha:.4f}")
print(f"  Df x alpha = {product:.4f}")
print(f"  8e = {EIGHT_E:.4f}")
print(f"  Deviation = {deviation:.2f}%")

# Now test with different parameters
print(f"\n" + "-"*80)
print("PARAMETER SENSITIVITY with ACTUAL data")
print("-"*80)

# Test varying frequency scale
print("\nVarying frequency scale (r/X):")
for freq_scale in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    embeddings = np.zeros((n_genes, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / freq_scale)
        embeddings[i] = base_pos + scale * direction

    Df, alpha, _ = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E * 100
    status = "PASS" if deviation < 15 else "FAIL"
    print(f"  r/{freq_scale}: Df x alpha = {product:.2f} ({deviation:.1f}%) [{status}]")

# Test varying dimensions
print("\nVarying dimensions:")
for n_dims_test in [20, 30, 40, 50, 60, 70, 100]:
    embeddings = np.zeros((n_genes, n_dims_test))
    for i, r in enumerate(R_values):
        np.random.seed(i + seed)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims_test)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims_test) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    Df, alpha, _ = compute_from_embeddings(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E * 100
    status = "PASS" if deviation < 15 else "FAIL"
    print(f"  {n_dims_test}D: Df x alpha = {product:.2f} ({deviation:.1f}%) [{status}]")

# Compare with uniform random R in same range
print(f"\n" + "-"*80)
print("COMPARISON: Actual R vs Random R (same range)")
print("-"*80)

n_dims = 50
np.random.seed(seed)
R_uniform = np.random.uniform(R_values.min(), R_values.max(), n_genes)

embeddings_uniform = np.zeros((n_genes, n_dims))
for i, r in enumerate(R_uniform):
    np.random.seed(i + seed)
    scale = 1.0 / (r + 0.1)
    direction = np.random.randn(n_dims)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    base_pos = np.sin(np.arange(n_dims) * r / 10.0)
    embeddings_uniform[i] = base_pos + scale * direction

Df_u, alpha_u, _ = compute_from_embeddings(embeddings_uniform)
product_u = Df_u * alpha_u
deviation_u = abs(product_u - EIGHT_E) / EIGHT_E * 100

print(f"  Actual R:  Df x alpha = {product:.2f} ({deviation:.1f}%)")
print(f"  Uniform R: Df x alpha = {product_u:.2f} ({deviation_u:.1f}%)")

# KEY FINDING: The R distribution
print(f"\n" + "-"*80)
print("CRITICAL ANALYSIS: What makes R special?")
print("-"*80)

# Histogram of R values
print(f"\nR distribution (binned):")
bins = [0, 2, 5, 10, 20, 50, 100]
for i in range(len(bins)-1):
    count = np.sum((R_values >= bins[i]) & (R_values < bins[i+1]))
    pct = count / len(R_values) * 100
    print(f"  R in [{bins[i]:3}, {bins[i+1]:3}): {count:4} genes ({pct:.1f}%)")

print(f"\nKey observation:")
print(f"  R values are right-skewed (log-normal-like)")
print(f"  Most values are small (< 20), few are large")
print(f"  This creates non-uniform frequency distribution in sinusoidal embedding")
