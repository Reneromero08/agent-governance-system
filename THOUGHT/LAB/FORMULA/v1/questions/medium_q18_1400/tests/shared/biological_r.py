"""
biological_r.py - Core R computation for biological scales

R = E / sigma where:
- E = Evidence/Agreement (0 to 1, higher = more consistent)
- sigma = Dispersion (standard deviation or similar)

This module provides R computation functions for different biological data types.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import hashlib
import json


@dataclass
class RResult:
    """Result of R computation at a biological scale."""
    scale: str  # "neural", "molecular", "cellular", "gene"
    R: float
    E: float
    sigma: float
    n_samples: int
    data_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "R": self.R,
            "E": self.E,
            "sigma": self.sigma,
            "n_samples": self.n_samples,
            "data_hash": self.data_hash,
            "metadata": self.metadata
        }


def compute_data_hash(data: np.ndarray) -> str:
    """Compute deterministic hash of data for reproducibility."""
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def compute_neural_R(
    eeg_data: np.ndarray,  # Shape: (trials, channels, timepoints)
    time_window: Optional[Tuple[int, int]] = None
) -> RResult:
    """
    Compute R for neural (EEG) data.

    E = cross-trial consistency (mean pairwise correlation)
    sigma = trial-to-trial variance

    Args:
        eeg_data: EEG data with shape (trials, channels, timepoints)
        time_window: Optional (start, end) indices for time window

    Returns:
        RResult with R value and metadata
    """
    if time_window:
        eeg_data = eeg_data[:, :, time_window[0]:time_window[1]]

    n_trials = eeg_data.shape[0]

    # Flatten each trial to vector
    trial_vectors = eeg_data.reshape(n_trials, -1)

    # E = mean pairwise correlation (cross-trial consistency)
    correlations = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            r = np.corrcoef(trial_vectors[i], trial_vectors[j])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

    E = np.mean(correlations) if correlations else 0.0
    E = (E + 1) / 2  # Map [-1, 1] to [0, 1]

    # sigma = trial-to-trial variance (mean across dimensions)
    sigma = np.mean(np.std(trial_vectors, axis=0)) + 1e-10

    R = E / sigma

    return RResult(
        scale="neural",
        R=R,
        E=E,
        sigma=sigma,
        n_samples=n_trials,
        data_hash=compute_data_hash(eeg_data),
        metadata={
            "n_channels": eeg_data.shape[1],
            "n_timepoints": eeg_data.shape[2] if len(eeg_data.shape) > 2 else 1,
            "time_window": time_window
        }
    )


def compute_molecular_R(
    embeddings: np.ndarray,  # Shape: (samples, features)
    labels: Optional[np.ndarray] = None  # Group labels for consistency
) -> RResult:
    """
    Compute R for molecular (protein) data.

    E = within-group consistency or self-consistency
    sigma = embedding variance

    Args:
        embeddings: Protein embeddings with shape (samples, features)
        labels: Optional group labels for computing within-group consistency

    Returns:
        RResult with R value and metadata
    """
    n_samples, n_features = embeddings.shape

    if labels is not None:
        # E = mean within-group consistency
        unique_labels = np.unique(labels)
        within_corrs = []
        for label in unique_labels:
            group_mask = labels == label
            group_data = embeddings[group_mask]
            if len(group_data) > 1:
                # Mean pairwise correlation within group
                for i in range(len(group_data)):
                    for j in range(i + 1, len(group_data)):
                        r = np.corrcoef(group_data[i], group_data[j])[0, 1]
                        if not np.isnan(r):
                            within_corrs.append(r)
        E = np.mean(within_corrs) if within_corrs else 0.5
        E = (E + 1) / 2
    else:
        # Self-consistency: how clustered are the embeddings?
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        E = 1 / (1 + np.mean(distances))

    # sigma = embedding variance
    sigma = np.mean(np.std(embeddings, axis=0)) + 1e-10

    R = E / sigma

    return RResult(
        scale="molecular",
        R=R,
        E=E,
        sigma=sigma,
        n_samples=n_samples,
        data_hash=compute_data_hash(embeddings),
        metadata={
            "n_features": n_features,
            "has_labels": labels is not None
        }
    )


def compute_cellular_R(
    expression: np.ndarray,  # Shape: (cells, genes)
    condition_labels: Optional[np.ndarray] = None
) -> RResult:
    """
    Compute R for cellular (single-cell) data.

    E = expression consistency across cells
    sigma = cell-to-cell variance

    Args:
        expression: Gene expression matrix (cells x genes)
        condition_labels: Optional condition/perturbation labels

    Returns:
        RResult with R value and metadata
    """
    n_cells, n_genes = expression.shape

    # E = 1 / (1 + CV^2) averaged across genes
    gene_means = np.mean(expression, axis=0) + 1e-10
    gene_stds = np.std(expression, axis=0)
    cvs = gene_stds / gene_means
    E = np.mean(1 / (1 + cvs**2))

    # sigma = mean variance across cells
    sigma = np.mean(np.std(expression, axis=0)) + 1e-10

    R = E / sigma

    return RResult(
        scale="cellular",
        R=R,
        E=E,
        sigma=sigma,
        n_samples=n_cells,
        data_hash=compute_data_hash(expression),
        metadata={
            "n_genes": n_genes,
            "has_conditions": condition_labels is not None
        }
    )


def compute_gene_R(
    expression: np.ndarray,  # Shape: (samples, genes)
    gene_idx: Optional[int] = None
) -> RResult:
    """
    Compute R for gene expression (bulk RNA-seq) data.

    If gene_idx is provided, compute R for that specific gene.
    Otherwise, compute overall R for the expression matrix.

    E = expression consistency across samples
    sigma = log-normalized variance

    Args:
        expression: Expression matrix (samples x genes)
        gene_idx: Optional index of specific gene

    Returns:
        RResult with R value and metadata
    """
    if gene_idx is not None:
        gene_expr = expression[:, gene_idx]
        gene_expr = gene_expr[gene_expr > 0]  # Remove zeros

        if len(gene_expr) < 3:
            return RResult(
                scale="gene",
                R=0.0,
                E=0.0,
                sigma=1.0,
                n_samples=len(gene_expr),
                data_hash="",
                metadata={"gene_idx": gene_idx, "insufficient_data": True}
            )

        cv = np.std(gene_expr) / (np.mean(gene_expr) + 1e-10)
        E = 1.0 / (1.0 + cv**2)
        sigma = np.std(np.log1p(gene_expr)) + 1e-10
        R = E / sigma
        n_samples = len(gene_expr)
    else:
        # Overall matrix R
        log_expr = np.log1p(expression)
        gene_cvs = np.std(expression, axis=0) / (np.mean(expression, axis=0) + 1e-10)
        E = np.mean(1 / (1 + gene_cvs**2))
        sigma = np.mean(np.std(log_expr, axis=0)) + 1e-10
        R = E / sigma
        n_samples = expression.shape[0]

    return RResult(
        scale="gene",
        R=R,
        E=E,
        sigma=sigma,
        n_samples=n_samples,
        data_hash=compute_data_hash(expression),
        metadata={
            "gene_idx": gene_idx,
            "n_genes": expression.shape[1]
        }
    )


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio (effective dimensionality).

    Df = (sum(eigenvalues))^2 / sum(eigenvalues^2)

    This measures how many dimensions effectively contribute to variance.
    """
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    return (np.sum(eigenvalues)**2) / (np.sum(eigenvalues**2) + 1e-10)


def compute_spectral_decay(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral decay exponent (alpha).

    Fits power law: eigenvalue ~ rank^(-alpha)

    Returns alpha (decay exponent).
    """
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 3:
        return 0.5  # Default to Riemann critical line

    ranks = np.arange(1, len(eigenvalues) + 1)

    # Log-log fit
    log_ranks = np.log(ranks)
    log_eigenvalues = np.log(eigenvalues)

    # Linear regression in log-log space
    slope, _ = np.polyfit(log_ranks, log_eigenvalues, 1)

    alpha = -slope  # Power law exponent
    return max(0.0, alpha)  # Ensure non-negative


def compute_8e_product(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Df x alpha for testing 8e conservation law.

    Args:
        data: Data matrix (samples x features)

    Returns:
        (Df, alpha, Df_x_alpha) tuple
    """
    # Compute covariance matrix
    data_centered = data - np.mean(data, axis=0)
    cov = np.cov(data_centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)

    Df = compute_participation_ratio(eigenvalues)
    alpha = compute_spectral_decay(eigenvalues)

    return Df, alpha, Df * alpha


# Expected conservation constant
EXPECTED_8E = 8 * np.e  # ~21.746


def test_8e_conservation(data: np.ndarray, tolerance: float = 0.15) -> dict:
    """
    Test if Df x alpha = 8e holds for this data.

    Args:
        data: Data matrix (samples x features)
        tolerance: Acceptable deviation from 8e (default 15%)

    Returns:
        Dict with Df, alpha, product, deviation, and pass/fail
    """
    Df, alpha, product = compute_8e_product(data)
    deviation = abs(product - EXPECTED_8E) / EXPECTED_8E

    return {
        "Df": Df,
        "alpha": alpha,
        "Df_x_alpha": product,
        "expected_8e": EXPECTED_8E,
        "deviation_pct": deviation * 100,
        "passed": deviation <= tolerance
    }
