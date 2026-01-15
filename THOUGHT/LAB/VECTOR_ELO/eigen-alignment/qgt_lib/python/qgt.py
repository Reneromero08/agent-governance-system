"""
Quantum Geometric Tensor (QGT) Python Bindings

Provides numpy-based implementations of QGT computations for semantic embeddings,
with optional C library acceleration via ctypes.

Key functions:
- fubini_study_metric(): Compute the Fubini-Study metric tensor
- berry_phase(): Compute Berry phase around closed loops
- participation_ratio(): Effective dimensionality (Df)
- natural_gradient(): QGT-based gradient for compass mode

References:
- Q43: Quantum Geometric Tensor for Semiosphere
- E.X.3.10: QGT Integration
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path
import warnings

# Try to load C library for acceleration
_lib = None
_LIB_PATH = Path(__file__).parent.parent / "build" / "lib" / "libquantum_geometric.so"

def _try_load_lib():
    """Attempt to load the C library via ctypes."""
    global _lib
    if _lib is not None:
        return _lib

    try:
        import ctypes
        if _LIB_PATH.exists():
            _lib = ctypes.CDLL(str(_LIB_PATH))
            return _lib
    except Exception as e:
        warnings.warn(f"Could not load C library: {e}. Using pure Python.")
    return None


# =============================================================================
# Core QGT Functions (Pure Python/NumPy)
# =============================================================================

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit sphere S^(n-1).

    For QGT, we treat normalized embeddings as points on the unit sphere,
    which is the real slice of complex projective space CP^(n-1).

    Args:
        embeddings: (n_samples, dim) array of embeddings

    Returns:
        Normalized embeddings on unit sphere
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    return embeddings / norms


def fubini_study_metric(embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute the Fubini-Study metric tensor for embedding space.

    For normalized real embeddings v on the unit sphere S^(n-1),
    the Fubini-Study metric reduces to the covariance structure.

    The metric g_μν = ⟨∂_μψ|∂_νψ⟩ - ⟨∂_μψ|ψ⟩⟨ψ|∂_νψ⟩

    For our purposes with embeddings:
    - The covariance matrix serves as the metric tensor
    - Eigenvalues give principal curvature directions

    Args:
        embeddings: (n_samples, dim) array of embeddings
        normalize: Whether to normalize embeddings first

    Returns:
        (dim, dim) metric tensor (covariance matrix)
    """
    if normalize:
        embeddings = normalize_embeddings(embeddings)

    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Covariance matrix = Fubini-Study metric in embedding coordinates
    metric = np.cov(centered.T)

    return metric


def participation_ratio(embeddings: np.ndarray, normalize: bool = True) -> float:
    """
    Compute the participation ratio (effective dimensionality Df).

    Df = (Σλ)² / Σλ²

    This IS the effective rank of the Fubini-Study metric.

    For trained BERT: Df ≈ 22 (confirmed by E.X.3.4)
    For random embeddings: Df ≈ n (full dimensionality)

    Args:
        embeddings: (n_samples, dim) array of embeddings
        normalize: Whether to normalize embeddings first

    Returns:
        Participation ratio (effective dimensionality)
    """
    metric = fubini_study_metric(embeddings, normalize=normalize)
    eigenvalues = np.linalg.eigvalsh(metric)

    # Filter small/negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    if sum_lambda_sq < 1e-20:
        return 0.0

    return (sum_lambda ** 2) / sum_lambda_sq


def metric_eigenspectrum(embeddings: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of the Fubini-Study metric.

    The eigenvectors define the principal axes of semantic variation.
    These should correspond to the "compass mode" directions from E.X.

    Args:
        embeddings: (n_samples, dim) array of embeddings
        normalize: Whether to normalize embeddings first

    Returns:
        (eigenvalues, eigenvectors) - sorted by descending eigenvalue
    """
    metric = fubini_study_metric(embeddings, normalize=normalize)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# =============================================================================
# Berry Phase Computation (for closed loops)
# =============================================================================

def berry_connection(path: np.ndarray) -> np.ndarray:
    """
    Compute the Berry connection A along a path of embeddings.

    A_i = Im[⟨ψ_i|ψ_{i+1}⟩] for each step

    For real embeddings, this is always zero, but we compute it
    as a stepping stone to Berry phase.

    Args:
        path: (n_points, dim) array of embeddings along path

    Returns:
        (n_points-1,) array of connection values
    """
    path = normalize_embeddings(path)

    # Compute overlaps between consecutive points
    overlaps = np.sum(path[:-1] * path[1:], axis=1)

    # For real embeddings, overlaps are real, so Im = 0
    # But we can still compute the phase angle
    # θ = arccos(overlap) gives the geodesic angle

    # Clip to valid range for arccos
    overlaps = np.clip(overlaps, -1.0, 1.0)
    angles = np.arccos(overlaps)

    return angles


def berry_phase(path: np.ndarray, closed: bool = True) -> float:
    """
    Compute the Berry phase as winding angle in 2D PCA projection.

    For high-dimensional embeddings, project to 2D via PCA and
    compute the winding angle (total angle swept in complex plane).

    This is more robust than spherical excess for high-D data where
    the (n-2)*pi correction assumes 2D geometry.

    Args:
        path: (n_points, dim) array of embeddings forming a loop
               If closed=True, path[0] should equal path[-1]
        closed: Whether to close the loop (connect last to first)

    Returns:
        Berry phase (winding angle) in radians
    """
    path = normalize_embeddings(path)

    if closed and not np.allclose(path[0], path[-1]):
        # Close the loop
        path = np.vstack([path, path[0:1]])

    # Project to 2D via PCA for winding computation
    centered = path - path.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_2d = centered @ Vt[:2].T
    except:
        return 0.0

    if proj_2d.shape[1] < 2:
        return 0.0

    # Map to complex plane
    z = proj_2d[:, 0] + 1j * proj_2d[:, 1]

    # Handle zero magnitudes
    z = np.where(np.abs(z) > 1e-10, z, 1e-10)

    # Compute winding angle (sum of angle changes)
    phase_diffs = np.angle(z[1:] / z[:-1])
    winding_angle = np.sum(phase_diffs)

    return winding_angle


def holonomy(path: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Parallel transport a vector around a closed loop.

    The holonomy is the rotation of the vector after parallel transport
    around a closed loop. Non-trivial holonomy indicates curvature.

    Args:
        path: (n_points, dim) embeddings forming a closed loop
        vector: (dim,) tangent vector to transport

    Returns:
        Transported vector after completing the loop
    """
    path = normalize_embeddings(path)

    # Ensure path is closed
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0:1]])

    v = vector.copy()
    v = v - np.dot(v, path[0]) * path[0]  # Project to tangent space
    v = v / (np.linalg.norm(v) + 1e-10)

    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]

        # Parallel transport from p1 to p2
        # Project v onto tangent space at p2
        v = v - np.dot(v, p2) * p2

        # Normalize (preserve length)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm

    return v


def holonomy_angle(path: np.ndarray, vector: np.ndarray) -> float:
    """
    Compute the rotation angle from holonomy around a loop.

    Args:
        path: (n_points, dim) embeddings forming a closed loop
        vector: (dim,) tangent vector to transport

    Returns:
        Rotation angle in radians
    """
    v_initial = vector.copy()
    v_initial = v_initial - np.dot(v_initial, path[0]) * path[0]
    v_initial = v_initial / (np.linalg.norm(v_initial) + 1e-10)

    v_final = holonomy(path, vector)

    # Compute angle between initial and final vectors
    dot = np.clip(np.dot(v_initial, v_final), -1.0, 1.0)
    angle = np.arccos(dot)

    return angle


# =============================================================================
# Natural Gradient (for Compass Mode)
# =============================================================================

def natural_gradient(
    embeddings: np.ndarray,
    euclidean_gradient: np.ndarray,
    regularization: float = 1e-6
) -> np.ndarray:
    """
    Compute the natural gradient using the QGT (Fubini-Study metric).

    natural_grad = g^{-1} @ euclidean_grad

    where g is the Fubini-Study metric tensor.

    The natural gradient should correspond to the "compass mode"
    directions identified in E.X.

    Args:
        embeddings: (n_samples, dim) array of embeddings
        euclidean_gradient: (dim,) or (n, dim) Euclidean gradient
        regularization: Regularization for matrix inversion

    Returns:
        Natural gradient (same shape as euclidean_gradient)
    """
    metric = fubini_study_metric(embeddings)

    # Regularize for numerical stability
    metric += regularization * np.eye(metric.shape[0])

    # Invert metric
    try:
        metric_inv = np.linalg.inv(metric)
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse
        metric_inv = np.linalg.pinv(metric)

    # Apply inverse metric to gradient
    if euclidean_gradient.ndim == 1:
        return metric_inv @ euclidean_gradient
    else:
        return euclidean_gradient @ metric_inv.T


def principal_directions(embeddings: np.ndarray, n_components: int = 22) -> np.ndarray:
    """
    Extract the principal directions from the QGT metric.

    These are the eigenvectors corresponding to the largest eigenvalues,
    and should match the "compass mode" directions.

    Args:
        embeddings: (n_samples, dim) array of embeddings
        n_components: Number of principal directions (default: 22 based on Df)

    Returns:
        (n_components, dim) principal direction vectors
    """
    eigenvalues, eigenvectors = metric_eigenspectrum(embeddings)
    return eigenvectors[:, :n_components].T


# =============================================================================
# Word Analogy Loops (for Berry Phase Testing)
# =============================================================================

def create_analogy_loop(
    embeddings: dict,
    words: List[str]
) -> np.ndarray:
    """
    Create a closed loop from word embeddings for Berry phase computation.

    Example:
        words = ["king", "queen", "woman", "man"]
        Creates loop: king → queen → woman → man → king

    Args:
        embeddings: Dict mapping words to embedding vectors
        words: List of words forming the loop (will be closed)

    Returns:
        (n_words + 1, dim) array of embeddings forming closed loop
    """
    loop = [embeddings[w] for w in words]
    loop.append(loop[0])  # Close the loop
    return np.array(loop)


def analogy_berry_phase(
    embeddings: dict,
    analogy: Tuple[str, str, str, str]
) -> float:
    """
    Compute Berry phase for a word analogy loop.

    Standard analogy format: (a, b, c, d) where a:b :: c:d
    Example: ("king", "queen", "man", "woman")

    Creates loop: a → b → d → c → a
    (This traces the parallelogram of the analogy)

    Args:
        embeddings: Dict mapping words to embedding vectors
        analogy: Tuple of (a, b, c, d) for a:b :: c:d

    Returns:
        Berry phase in radians
    """
    a, b, c, d = analogy
    loop = create_analogy_loop(embeddings, [a, b, d, c])
    return berry_phase(loop, closed=True)


# =============================================================================
# Chern Number Estimation
# =============================================================================

def chern_number_estimate(
    embeddings: np.ndarray,
    n_samples: int = 100,
    seed: int = 42
) -> float:
    """
    Estimate the Chern number by Monte Carlo integration of Berry curvature.

    For real embeddings, the local Berry curvature is zero, but we can
    estimate a discrete Chern-like invariant from the topology of the
    embedding distribution.

    This uses random triangulations to estimate the integrated curvature.

    WARNING: This is an approximation. True Chern numbers require
    complex structure or proper fiber bundle formulation.

    Args:
        embeddings: (n_samples, dim) array of embeddings
        n_samples: Number of random triangles to sample
        seed: Random seed

    Returns:
        Estimated Chern number (should be integer-ish if well-defined)
    """
    np.random.seed(seed)
    embeddings = normalize_embeddings(embeddings)
    n_points = len(embeddings)

    total_phase = 0.0

    for _ in range(n_samples):
        # Sample random triangle
        idx = np.random.choice(n_points, size=3, replace=False)
        triangle = embeddings[idx]

        # Compute Berry phase around triangle
        phase = berry_phase(triangle, closed=True)
        total_phase += phase

    # Average phase per triangle, scaled to approximate Chern number
    # Chern = (1/2π) ∫ Ω
    avg_phase = total_phase / n_samples
    chern_estimate = avg_phase / (2 * np.pi)

    return chern_estimate


# =============================================================================
# High-Level Analysis Functions
# =============================================================================

def analyze_qgt_structure(
    embeddings: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Complete QGT analysis of embedding space.

    Computes:
    - Participation ratio (effective dimensionality)
    - Metric eigenspectrum
    - Principal directions
    - Chern number estimate

    Args:
        embeddings: (n_samples, dim) array of embeddings
        verbose: Print results

    Returns:
        Dict with all computed quantities
    """
    results = {}

    # Participation ratio
    pr = participation_ratio(embeddings)
    results['participation_ratio'] = pr
    results['effective_dim'] = pr

    # Eigenspectrum
    eigenvalues, eigenvectors = metric_eigenspectrum(embeddings)
    results['eigenvalues'] = eigenvalues
    results['eigenvectors'] = eigenvectors

    # Top eigenvalues
    results['top_22_eigenvalues'] = eigenvalues[:22]
    results['eigenvalue_ratio'] = eigenvalues[0] / eigenvalues[21] if eigenvalues[21] > 0 else np.inf

    # Chern estimate
    chern = chern_number_estimate(embeddings)
    results['chern_estimate'] = chern

    if verbose:
        print("=" * 60)
        print("QGT Structure Analysis")
        print("=" * 60)
        print(f"Participation Ratio (Df): {pr:.2f}")
        print(f"Effective Dimensionality: {int(round(pr))}")
        print(f"Top 5 Eigenvalues: {eigenvalues[:5]}")
        print(f"L1/L22 ratio: {results['eigenvalue_ratio']:.2f}")
        print(f"Chern Number Estimate: {chern:.4f}")
        print("=" * 60)

    return results


def compare_compass_to_qgt(
    embeddings: np.ndarray,
    compass_directions: np.ndarray,
    n_directions: int = 22
) -> dict:
    """
    Compare compass mode directions to QGT principal directions.

    This tests the hypothesis that natural gradient = compass mode.

    Args:
        embeddings: (n_samples, dim) array of embeddings
        compass_directions: (n, dim) compass mode directions from E.X
        n_directions: Number of directions to compare

    Returns:
        Dict with alignment metrics
    """
    qgt_dirs = principal_directions(embeddings, n_directions)

    # Compute alignment (absolute cosine similarity)
    # For each compass direction, find best matching QGT direction
    alignments = []
    for c in compass_directions[:n_directions]:
        c = c / (np.linalg.norm(c) + 1e-10)
        similarities = np.abs(qgt_dirs @ c)
        alignments.append(np.max(similarities))

    results = {
        'alignments': np.array(alignments),
        'mean_alignment': np.mean(alignments),
        'min_alignment': np.min(alignments),
        'max_alignment': np.max(alignments),
    }

    # Subspace overlap using principal angles
    # This measures how well the two subspaces align
    U1 = qgt_dirs.T  # (dim, n_directions)
    U2 = compass_directions[:n_directions].T
    U2 = U2 / np.linalg.norm(U2, axis=0, keepdims=True)

    # SVD of U1.T @ U2 gives cosines of principal angles
    _, s, _ = np.linalg.svd(U1.T @ U2)
    principal_angles = np.arccos(np.clip(s, 0, 1))

    results['principal_angles'] = principal_angles
    results['subspace_overlap'] = np.mean(s)  # Average cosine

    return results


# =============================================================================
# Convenience Functions
# =============================================================================

def load_and_analyze(
    embeddings_path: str,
    key: str = 'embeddings'
) -> dict:
    """
    Load embeddings from file and run QGT analysis.

    Args:
        embeddings_path: Path to .npy or .npz file
        key: Key for embeddings in .npz file

    Returns:
        QGT analysis results
    """
    path = Path(embeddings_path)

    if path.suffix == '.npy':
        embeddings = np.load(path)
    elif path.suffix == '.npz':
        data = np.load(path)
        embeddings = data[key]
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")

    return analyze_qgt_structure(embeddings)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("QGT Python Bindings - Testing")
    print()

    # Test with random embeddings
    print("Testing with random embeddings (should give Df ~ dim):")
    random_emb = np.random.randn(1000, 768)
    pr = participation_ratio(random_emb)
    print(f"  Random Df: {pr:.1f} (expected ~100)")
    print()

    # Test with low-rank embeddings (simulating trained BERT)
    print("Testing with low-rank embeddings (should give Df ~ 22):")
    # Create embeddings that live in ~22D subspace
    low_rank = np.random.randn(1000, 22) @ np.random.randn(22, 768)
    low_rank += 0.1 * np.random.randn(1000, 768)  # Add noise
    pr = participation_ratio(low_rank)
    print(f"  Low-rank Df: {pr:.1f} (expected ~22)")
    print()

    # Test Berry phase
    print("Testing Berry phase with random loop:")
    loop = np.random.randn(4, 768)
    phase = berry_phase(loop)
    print(f"  Berry phase: {phase:.4f} rad ({np.degrees(phase):.2f} deg)")
    print()

    # Full analysis
    print("Full QGT analysis:")
    analyze_qgt_structure(low_rank)
