"""
Q44: Quantum Born Rule Validation
=================================

Core library for testing whether R = (E / grad_S) * sigma^Df
computes the quantum Born rule probability P(psi->phi) = |<psi|phi>|^2

Dependencies:
- numpy
- sentence-transformers (optional, for real embeddings)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class BornValidationResult:
    """Result of comparing R to Born rule probability."""
    query: str
    context: List[str]
    R: float
    P_born: float
    E: float
    grad_S: float
    sigma: float
    Df: float
    overlaps: List[float]


# =============================================================================
# Embedding Functions
# =============================================================================

def embed_synthetic(text: str, dim: int = 384, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate deterministic synthetic embedding from text hash.
    Used for reproducible testing without model dependencies.
    """
    if seed is not None:
        np.random.seed(seed)
    # Use hash of text as seed for reproducibility
    text_hash = hash(text) % (2**31)
    rng = np.random.RandomState(text_hash)
    vec = rng.randn(dim)
    return vec / np.linalg.norm(vec)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length. Always returns a normalized vector."""
    norm = np.linalg.norm(vec)
    # Always normalize safely - never return unnormalized vector
    return vec / max(norm, 1e-10)


# =============================================================================
# Born Rule Computation
# =============================================================================

def compute_born_probability(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray]
) -> float:
    """
    Compute quantum Born rule: P(psi->phi) = |<psi|phi>|^2

    Args:
        query_vec: Normalized query embedding (psi)
        context_vecs: List of normalized context embeddings

    Returns:
        Born rule probability |<psi|phi_context>|^2
    """
    if len(context_vecs) == 0:
        return 0.0

    # Normalize query
    psi = normalize(query_vec)

    # Context superposition: |phi_context> = (1/sqrt(n)) * sum(|phi_i>)
    # This creates a normalized superposition state
    phi_sum = np.sum(context_vecs, axis=0)
    phi_context = phi_sum / np.sqrt(len(context_vecs))
    phi_context = normalize(phi_context)

    # Born rule: |<psi|phi>|^2
    overlap = np.dot(psi, phi_context)
    P_born = abs(overlap) ** 2

    return float(P_born)


def compute_born_probability_direct(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray]
) -> float:
    """
    Alternative Born rule: average of individual |<psi|phi_i>|^2

    This is the projection onto the mixed state (density matrix formulation).
    """
    if len(context_vecs) == 0:
        return 0.0

    psi = normalize(query_vec)
    P_born = np.mean([abs(np.dot(psi, normalize(phi)))**2 for phi in context_vecs])
    return float(P_born)


# =============================================================================
# R Formula Components
# =============================================================================

def compute_E_linear(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray]
) -> Tuple[float, List[float]]:
    """
    Compute E (Essence) as mean overlap with context vectors.

    E = mean(⟨ψ|φᵢ⟩) - the quantum inner product average.

    For normalized vectors, overlap = cos(angle), so:
    - overlap = 1 means perfect alignment
    - overlap = 0 means orthogonal
    - overlap = -1 means opposite

    Returns:
        (E value, list of raw overlaps)
    """
    if len(context_vecs) == 0:
        return 0.0, []

    psi = normalize(query_vec)
    overlaps = [float(np.dot(psi, normalize(phi))) for phi in context_vecs]

    # E = mean overlap (quantum inner product)
    E_linear = np.mean(overlaps)

    return float(E_linear), overlaps


def compute_E_variants(overlaps: List[float]) -> Dict[str, float]:
    """
    Compute multiple E variants to test which correlates best with Born rule.
    """
    if len(overlaps) == 0:
        return {"linear": 0.0, "squared": 0.0, "abs": 0.0, "gaussian": 0.0}

    return {
        "linear": float(np.mean(overlaps)),
        "squared": float(np.mean([o**2 for o in overlaps])),
        "abs": float(np.mean([abs(o) for o in overlaps])),
        "gaussian": float(np.mean([np.exp(-(1-o)**2/2) for o in overlaps])),
    }


def compute_grad_S(overlaps: List[float]) -> float:
    """
    Compute grad_S (entropy gradient / local curvature).

    grad_S = std(overlaps) - measures local disagreement/spread.

    For uniform context (all similar), grad_S is small.
    For diverse context, grad_S is large.

    Note: For single context (n=1), std is undefined. We return 1.0
    (not 1e-6) to avoid artificially inflating R values.
    """
    if len(overlaps) < 2:
        # Single context: return 1.0 to keep R = E (no scaling)
        return 1.0
    return float(max(np.std(overlaps, ddof=1), 1e-6))


def compute_sigma(n_context: int) -> float:
    """
    Compute sigma (redundancy factor).

    sigma = sqrt(n) where n = number of context items.
    This represents information redundancy scaling.
    """
    return float(np.sqrt(max(n_context, 1)))


def compute_Df_simple(vec: np.ndarray) -> float:
    """
    Compute Df (effective dimensionality) for a single vector.

    Simple approximation: participation ratio of squared components.
    Df = (sum |v_i|^2)^2 / sum |v_i|^4

    For a truly d-dimensional uniform vector: Df = d
    For a 1-hot vector: Df = 1
    """
    v_squared = vec ** 2
    sum_sq = np.sum(v_squared)
    sum_sq_sq = np.sum(v_squared ** 2)

    if sum_sq_sq < 1e-10:
        return 1.0

    return float((sum_sq ** 2) / sum_sq_sq)


def compute_Df_from_context(context_vecs: List[np.ndarray]) -> float:
    """
    Compute Df from context embedding covariance (Q43-style).

    Uses participation ratio of covariance eigenspectrum.
    """
    if len(context_vecs) < 2:
        return 1.0

    # Stack embeddings
    X = np.array(context_vecs)

    # Compute Gram matrix
    gram = X @ X.T

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    # Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    return float((sum_lambda ** 2) / sum_lambda_sq)


# =============================================================================
# Full R Computation
# =============================================================================

def compute_R(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray],
    Df: Optional[float] = None,
    E_type: str = "linear"
) -> BornValidationResult:
    """
    Compute R = (E / grad_S) * sigma^Df

    Args:
        query_vec: Query embedding
        context_vecs: Context embeddings
        Df: Effective dimensionality (if None, computed from context)
        E_type: Which E variant to use ("linear", "squared", "abs", "gaussian")

    Returns:
        BornValidationResult with all components
    """
    if len(context_vecs) == 0:
        return BornValidationResult(
            query="", context=[], R=0.0, P_born=0.0,
            E=0.0, grad_S=1.0, sigma=1.0, Df=1.0, overlaps=[]
        )

    # Normalize
    psi = normalize(query_vec)
    context_normalized = [normalize(phi) for phi in context_vecs]

    # Compute overlaps
    overlaps = [float(np.dot(psi, phi)) for phi in context_normalized]

    # E variants
    E_variants = compute_E_variants(overlaps)
    E = E_variants.get(E_type, E_variants["linear"])

    # grad_S
    grad_S = compute_grad_S(overlaps)

    # sigma
    sigma = compute_sigma(len(context_vecs))

    # Df
    if Df is None:
        Df = compute_Df_from_context(context_normalized)

    # R formula
    R = (E / grad_S) * (sigma ** Df) if grad_S > 0 else 0.0

    # Born probability for comparison
    P_born = compute_born_probability(psi, context_normalized)

    return BornValidationResult(
        query="",
        context=[],
        R=float(R),
        P_born=float(P_born),
        E=float(E),
        grad_S=float(grad_S),
        sigma=float(sigma),
        Df=float(Df),
        overlaps=overlaps
    )


def compute_R_variants(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray],
    Df: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute R with different E formulations to find best Born rule match.
    """
    if len(context_vecs) == 0:
        return {"R_linear": 0.0, "R_squared": 0.0, "R_abs": 0.0, "R_gaussian": 0.0}

    # Shared components
    psi = normalize(query_vec)
    context_normalized = [normalize(phi) for phi in context_vecs]
    overlaps = [float(np.dot(psi, phi)) for phi in context_normalized]
    grad_S = compute_grad_S(overlaps)
    sigma = compute_sigma(len(context_vecs))

    if Df is None:
        Df = compute_Df_from_context(context_normalized)

    E_variants = compute_E_variants(overlaps)

    # Compute R for each E variant
    results = {}
    for name, E in E_variants.items():
        R = (E / grad_S) * (sigma ** Df) if grad_S > 0 else 0.0
        results[f"R_{name}"] = float(R)

    return results


# =============================================================================
# Simplified R for Born Comparison
# =============================================================================

def compute_R_simple(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray]
) -> Tuple[float, float]:
    """
    Simplified R computation focused on Born rule comparison.

    R_simple = E / grad_S (without sigma^Df scaling)

    Returns: (R_simple, P_born)
    """
    if len(context_vecs) == 0:
        return 0.0, 0.0

    psi = normalize(query_vec)
    context_normalized = [normalize(phi) for phi in context_vecs]

    # Overlaps
    overlaps = [float(np.dot(psi, phi)) for phi in context_normalized]

    # E = mean overlap
    E = np.mean(overlaps)

    # grad_S = std of overlaps
    grad_S = max(np.std(overlaps), 1e-6) if len(overlaps) > 1 else 1e-6

    # R_simple
    R_simple = E / grad_S

    # Born probability
    P_born = compute_born_probability(psi, context_normalized)

    return float(R_simple), float(P_born)


def compute_E_squared_R(
    query_vec: np.ndarray,
    context_vecs: List[np.ndarray]
) -> Tuple[float, float]:
    """
    R with E² to match Born rule's |<psi|phi>|^2 structure.

    R_E2 = E² / grad_S

    Returns: (R_E2, P_born)
    """
    if len(context_vecs) == 0:
        return 0.0, 0.0

    psi = normalize(query_vec)
    context_normalized = [normalize(phi) for phi in context_vecs]

    overlaps = [float(np.dot(psi, phi)) for phi in context_normalized]
    E = np.mean(overlaps)
    E_squared = E ** 2
    grad_S = max(np.std(overlaps), 1e-6) if len(overlaps) > 1 else 1e-6

    R_E2 = E_squared / grad_S
    P_born = compute_born_probability(psi, context_normalized)

    return float(R_E2), float(P_born)


# =============================================================================
# Correlation Analysis
# =============================================================================

def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    x_arr = np.array(x)
    y_arr = np.array(y)

    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sqrt(np.sum((x_arr - x_mean)**2) * np.sum((y_arr - y_mean)**2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def normalize_values(values: List[float]) -> List[float]:
    """Normalize values to [0, 1] range for comparison."""
    arr = np.array(values)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-10:
        return [0.5] * len(values)
    return list((arr - min_val) / (max_val - min_val))


if __name__ == "__main__":
    # Quick sanity test
    print("Q44 Core Library - Sanity Test")
    print("=" * 50)

    # Generate synthetic test
    query = embed_synthetic("verify canonical governance", dim=384)
    context = [
        embed_synthetic("verification protocols", dim=384),
        embed_synthetic("canonical rules", dim=384),
        embed_synthetic("governance integrity", dim=384),
    ]

    result = compute_R(query, context)
    print(f"E = {result.E:.4f}")
    print(f"grad_S = {result.grad_S:.4f}")
    print(f"sigma = {result.sigma:.4f}")
    print(f"Df = {result.Df:.4f}")
    print(f"R = {result.R:.4f}")
    print(f"P_born = {result.P_born:.4f}")

    # Check correlation structure
    R_simple, P_born = compute_R_simple(query, context)
    print(f"\nSimplified R = {R_simple:.4f}")
    print(f"P_born = {P_born:.4f}")

    R_E2, P_born = compute_E_squared_R(query, context)
    print(f"R (E²) = {R_E2:.4f}")
