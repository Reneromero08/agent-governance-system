"""
v2 Shared Formula Implementation

Single authoritative implementation of R = (E / grad_S) * sigma^Df

All v2 tests MUST use these functions. No alternative E definitions permitted.
"""

import numpy as np
from itertools import combinations


def compute_E(embeddings):
    """
    Compute E = mean pairwise cosine similarity.

    This is the ONE E definition for all v2 work.
    E = (1/C(n,2)) * sum_{i<j} cos(x_i, x_j)

    Args:
        embeddings: np.ndarray of shape (n, d) -- n observation vectors of dimension d

    Returns:
        float: mean pairwise cosine similarity in [-1, 1]
    """
    n = embeddings.shape[0]
    if n < 2:
        return float('nan')

    # Normalize to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    normed = embeddings / norms

    # Compute all pairwise cosine similarities via dot products
    sim_matrix = normed @ normed.T

    # Extract upper triangle (i < j pairs only)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    return float(np.mean(pairwise_sims))


def compute_grad_S(embeddings):
    """
    Compute grad_S = standard deviation of pairwise cosine similarities.

    This measures local dispersion in the similarity space.

    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        float: standard deviation of pairwise cosine similarities (>= 0)
    """
    n = embeddings.shape[0]
    if n < 2:
        return float('nan')

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms

    sim_matrix = normed @ normed.T
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    return float(np.std(pairwise_sims))


def compute_sigma(embeddings):
    """
    Compute sigma = compression ratio from eigenvalue spectrum.

    sigma = (effective dimensionality) / (ambient dimensionality)
    where effective dimensionality is estimated via participation ratio
    of the covariance eigenspectrum.

    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        float: sigma in (0, 1]
    """
    n, d = embeddings.shape
    if n < 2:
        return float('nan')

    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]  # positive only

    if len(eigenvalues) == 0:
        return float('nan')

    # Participation ratio: (sum(lambda))^2 / sum(lambda^2)
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    # Normalize by ambient dimensionality
    sigma = pr / d

    return float(np.clip(sigma, 1e-10, 1.0))


def compute_Df(embeddings):
    """
    Compute Df = fractal dimension from eigenvalue decay.

    Estimated from the power-law exponent of eigenvalue spectrum:
    lambda_k ~ k^(-alpha), then Df = 2/alpha (box-counting relationship).

    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        float: estimated fractal dimension
    """
    n, d = embeddings.shape
    if n < 3:
        return float('nan')

    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)

    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) < 3:
        return float('nan')

    # Fit power law: log(lambda) = -alpha * log(k) + const
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_eig = np.log(eigenvalues)

    # Linear regression
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    result = np.linalg.lstsq(A, log_eig, rcond=None)
    alpha = -result[0][0]  # negative slope = alpha

    if alpha <= 0:
        return float('nan')

    Df = 2.0 / alpha
    return float(Df)


def compute_R_simple(embeddings):
    """
    Compute R_simple = E / grad_S (without fractal scaling).

    This is the minimal version of the formula.

    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        float: R_simple value
    """
    E = compute_E(embeddings)
    grad_S = compute_grad_S(embeddings)

    if np.isnan(E) or np.isnan(grad_S) or grad_S < 1e-10:
        return float('nan')

    return E / grad_S


def compute_R_full(embeddings):
    """
    Compute R_full = (E / grad_S) * sigma^Df (the complete formula).

    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        float: R_full value
    """
    E = compute_E(embeddings)
    grad_S = compute_grad_S(embeddings)
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)

    if any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        return float('nan')

    if grad_S < 1e-10:
        return float('nan')

    R = (E / grad_S) * (sigma ** Df)
    return float(R)


def compute_all(embeddings):
    """
    Compute all formula components and both R forms.

    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        dict with keys: E, grad_S, sigma, Df, R_simple, R_full
    """
    E = compute_E(embeddings)
    grad_S = compute_grad_S(embeddings)
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)

    R_simple = E / grad_S if (not np.isnan(grad_S) and grad_S > 1e-10) else float('nan')

    if any(np.isnan(x) for x in [E, grad_S, sigma, Df]) or grad_S < 1e-10:
        R_full = float('nan')
    else:
        R_full = (E / grad_S) * (sigma ** Df)

    return {
        'E': E,
        'grad_S': grad_S,
        'sigma': sigma,
        'Df': Df,
        'R_simple': R_simple,
        'R_full': R_full,
    }
