#!/usr/bin/env python3
"""
Q38: Noether's Theorem - Conservation Laws for Meaning Field

Core implementation for deriving and validating conservation laws on the
semiosphere. Key insight: concepts follow geodesic motion (great circles)
on the embedding sphere, and the Df ~ 22 principal directions span an
approximately flat subspace where momentum is conserved.

Time Evolution: x(t) follows geodesics on unit sphere S^(d-1)
Action: S = integral of (1/2)|dx/dt|^2 dt = arc length
Conservation: Q_a = v . e_a conserved along geodesics in flat subspace

References:
- Q38: Noether's Theorem - Conservation Laws
- Q43: Quantum Geometric Tensor (QGT infrastructure)
- Q34: Platonic Convergence (Df ~ 22 effective dimensions)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import sys

# Add QGT library path
QGT_PATH = Path(__file__).parent.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
if QGT_PATH.exists():
    sys.path.insert(0, str(QGT_PATH))

try:
    from qgt import (
        normalize_embeddings,
        fubini_study_metric,
        metric_eigenspectrum,
        principal_directions,
        participation_ratio
    )
    QGT_AVAILABLE = True
except ImportError:
    QGT_AVAILABLE = False
    print("[WARN] QGT library not available, using local implementations")


# =============================================================================
# Threshold Constants
# =============================================================================

CV_THRESHOLD_CONSERVED = 0.01  # 1% variation threshold for conservation
CV_THRESHOLD_STRICT = 1e-5    # Machine precision threshold


# =============================================================================
# Core Normalization (fallback if QGT not available)
# =============================================================================

def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit sphere S^(n-1)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return embeddings / norms


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a single vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


# =============================================================================
# Sphere Geodesics (Great Circles)
# =============================================================================

def sphere_geodesic(x0: np.ndarray, v0: np.ndarray, t: float) -> np.ndarray:
    """
    Exact geodesic on unit sphere (great circle).

    x(t) = x0 * cos(|v|*t) + (v/|v|) * sin(|v|*t)

    No Christoffel symbols needed - sphere geometry is analytic.

    Args:
        x0: Starting point on unit sphere (d,)
        v0: Initial tangent velocity (d,) - should be tangent to sphere at x0
        t: Time parameter

    Returns:
        Point on geodesic at time t (d,)
    """
    x0 = _normalize_vector(x0)

    # Project v0 to tangent space at x0 (ensure orthogonality)
    v0 = v0 - np.dot(v0, x0) * x0

    v_norm = np.linalg.norm(v0)
    if v_norm < 1e-10:
        return x0  # No velocity, stay at starting point

    v_hat = v0 / v_norm

    # Geodesic formula: x(t) = x0*cos(speed*t) + v_hat*sin(speed*t)
    x_t = x0 * np.cos(v_norm * t) + v_hat * np.sin(v_norm * t)

    return x_t


def sphere_geodesic_trajectory(
    x0: np.ndarray,
    v0: np.ndarray,
    n_steps: int = 100,
    t_max: float = 1.0
) -> np.ndarray:
    """
    Generate discrete trajectory along geodesic (great circle).

    Args:
        x0: Starting point on unit sphere (d,)
        v0: Initial tangent velocity (d,)
        n_steps: Number of points in trajectory
        t_max: Maximum time parameter

    Returns:
        Trajectory array (n_steps, d)
    """
    t_values = np.linspace(0, t_max, n_steps)
    trajectory = np.array([sphere_geodesic(x0, v0, t) for t in t_values])
    return trajectory


def geodesic_velocity(trajectory: np.ndarray, t_max: float = 1.0) -> np.ndarray:
    """
    Compute velocity (tangent vector) along trajectory.

    Uses finite differences: v[i] = (x[i+1] - x[i-1]) / 2dt

    Args:
        trajectory: (n_steps, d) array of points
        t_max: Maximum time parameter (used to compute dt)

    Returns:
        (n_steps, d) array of velocities (endpoints use one-sided diff)
    """
    n = len(trajectory)
    velocities = np.zeros_like(trajectory)

    if n > 1:
        dt = t_max / (n - 1)
    else:
        dt = 1.0

    # Central difference for interior points
    if n > 2:
        velocities[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2.0 * dt)

    # One-sided for endpoints
    if n > 1:
        velocities[0] = (trajectory[1] - trajectory[0]) / dt
        velocities[-1] = (trajectory[-1] - trajectory[-2]) / dt

    return velocities


# =============================================================================
# Principal Subspace Projection
# =============================================================================

def project_to_principal_subspace(
    embeddings: np.ndarray,
    n_dims: int = 22
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings to Df-dimensional effective subspace.

    Uses QGT eigenvectors (principal directions) to define the subspace.

    Args:
        embeddings: (n_samples, d) array of embeddings
        n_dims: Number of principal dimensions (default: Df ~ 22)

    Returns:
        Tuple of:
        - projected: (n_samples, n_dims) projected embeddings
        - axes: (n_dims, d) principal axes
    """
    if QGT_AVAILABLE:
        emb_norm = normalize_embeddings(embeddings)
        axes = principal_directions(emb_norm, n_components=n_dims)
    else:
        # Fallback: use PCA via SVD
        emb_norm = _normalize_embeddings(embeddings)
        centered = emb_norm - emb_norm.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axes = Vt[:n_dims]  # (n_dims, d)

    # Project: each embedding onto each axis
    projected = emb_norm @ axes.T  # (n_samples, n_dims)

    return projected, axes


def subspace_metric_flatness(
    embeddings: np.ndarray,
    principal_axes: np.ndarray
) -> float:
    """
    Test if principal subspace is approximately Euclidean (flat).

    Computes induced metric in subspace and measures deviation from identity.

    Args:
        embeddings: (n_samples, d) array of embeddings
        principal_axes: (n_dims, d) principal direction vectors

    Returns:
        Frobenius norm of (G - I), where G is induced metric.
        0 = perfectly flat (Euclidean), larger = more curved.
    """
    # Project embeddings to subspace
    if QGT_AVAILABLE:
        emb_norm = normalize_embeddings(embeddings)
    else:
        emb_norm = _normalize_embeddings(embeddings)

    projected = emb_norm @ principal_axes.T  # (n_samples, n_dims)

    # Compute metric in subspace (covariance)
    centered = projected - projected.mean(axis=0)
    n_samples = len(projected)

    if n_samples < 2:
        return 0.0

    G = (centered.T @ centered) / (n_samples - 1)  # (n_dims, n_dims)

    # Normalize to unit trace for fair comparison
    trace_G = np.trace(G)
    if trace_G > 1e-10:
        G_normalized = G / trace_G * len(G)  # Scale so trace = n_dims
    else:
        G_normalized = G

    # Identity matrix (perfectly flat)
    I = np.eye(len(G))

    # Deviation from flatness
    deviation = np.linalg.norm(G_normalized - I, 'fro')

    return deviation


# =============================================================================
# Conservation Law Testing
# =============================================================================

def momentum_projection(
    velocity: np.ndarray,
    principal_axes: np.ndarray
) -> np.ndarray:
    """
    Compute momentum projection Q_a = v . e_a for each principal direction.

    In flat space, these are the conserved momenta (Noether charges).

    Args:
        velocity: (d,) velocity vector
        principal_axes: (n_dims, d) principal direction vectors

    Returns:
        (n_dims,) array of momentum projections
    """
    # Q_a = v . e_a
    return principal_axes @ velocity


def conservation_test(
    trajectory: np.ndarray,
    principal_axes: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Track Q_a along trajectory, measure coefficient of variation.

    CV = std / |mean| is the relative fluctuation.
    CV < CV_THRESHOLD_CONSERVED indicates good conservation.

    Args:
        trajectory: (n_steps, d) array of points along geodesic
        principal_axes: (n_dims, d) principal direction vectors

    Returns:
        Dict mapping direction index to {'mean', 'std', 'cv'}
    """
    # Compute velocities
    velocities = geodesic_velocity(trajectory)

    # Track Q_a at each step
    n_steps = len(trajectory)
    n_dims = len(principal_axes)
    Q = np.zeros((n_steps, n_dims))

    for i, v in enumerate(velocities):
        Q[i] = momentum_projection(v, principal_axes)

    # Skip first and last points (boundary effects from finite difference)
    Q_interior = Q[1:-1] if n_steps > 2 else Q

    # Compute statistics for each direction
    results = {}
    for a in range(n_dims):
        q_a = Q_interior[:, a]
        mean = np.mean(q_a)
        std = np.std(q_a)
        cv = std / (np.abs(mean) + 1e-10)

        results[a] = {
            'mean': float(mean),
            'std': float(std),
            'cv': float(cv)
        }

    return results


def count_conserved_directions(
    conservation_results: Dict[int, Dict[str, float]],
    cv_threshold: float = None
) -> Tuple[int, int]:
    """
    Count how many directions have CV below threshold (conserved).

    Args:
        conservation_results: Output from conservation_test()
        cv_threshold: Maximum CV for "conserved" (default: CV_THRESHOLD_CONSERVED)

    Returns:
        (n_conserved, n_total)
    """
    if cv_threshold is None:
        cv_threshold = CV_THRESHOLD_CONSERVED
    n_conserved = sum(1 for d in conservation_results.values() if d['cv'] < cv_threshold)
    n_total = len(conservation_results)
    return n_conserved, n_total


# =============================================================================
# Action Integral
# =============================================================================

def action_integral(trajectory: np.ndarray) -> float:
    """
    Compute action S = integral of (1/2) |dx/dt|^2 dt.

    For geodesics on a sphere with constant-speed parameterization,
    this equals (1/2) * |v|^2 * T = (1/2) * (arc_length)^2 / T.

    Using discrete approximation: S = sum of (1/2) |delta_x|^2

    Args:
        trajectory: (n_steps, d) array of points

    Returns:
        Action integral (scalar)
    """
    if len(trajectory) < 2:
        return 0.0

    # Compute displacements
    delta = np.diff(trajectory, axis=0)  # (n_steps-1, d)

    # Action = sum of kinetic energies
    kinetic = 0.5 * np.sum(delta ** 2, axis=1)  # (n_steps-1,)

    return float(np.sum(kinetic))


def arc_length(trajectory: np.ndarray) -> float:
    """
    Compute arc length along trajectory.

    Args:
        trajectory: (n_steps, d) array of points

    Returns:
        Total arc length (scalar)
    """
    if len(trajectory) < 2:
        return 0.0

    delta = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(delta, axis=1)

    return float(np.sum(segment_lengths))


# =============================================================================
# Perturbed Trajectory (for action comparison)
# =============================================================================

def perturb_trajectory(
    trajectory: np.ndarray,
    noise_scale: float = 0.1,
    seed: int = None
) -> np.ndarray:
    """
    Create a perturbed version of trajectory (not a geodesic).

    Used to verify that geodesics minimize action. Uses tangent space
    perturbation to preserve geodesic structure naturally via exponential map.

    Args:
        trajectory: (n_steps, d) array of points
        noise_scale: Scale of perturbation
        seed: Random seed

    Returns:
        Perturbed trajectory (same shape, still on sphere)
    """
    if seed is not None:
        np.random.seed(seed)

    perturbed = trajectory.copy()

    for i in range(1, len(trajectory) - 1):
        # Get tangent space at this point
        normal = trajectory[i] / (np.linalg.norm(trajectory[i]) + 1e-15)

        # Random noise
        noise = np.random.randn(len(trajectory[i])) * noise_scale
        # Project to tangent space
        noise = noise - np.dot(noise, normal) * normal

        # Apply via exponential map (small angle approximation)
        angle = np.linalg.norm(noise)
        if angle > 1e-10:
            direction = noise / angle
            perturbed[i] = np.cos(angle) * trajectory[i] + np.sin(angle) * direction

    return perturbed


# =============================================================================
# Random Direction Generator (for negative control)
# =============================================================================

def random_directions(dim: int, n_dirs: int, seed: int = None) -> np.ndarray:
    """
    Generate random unit vectors (not principal directions).

    Used as negative control - these should NOT be conserved.

    Args:
        dim: Embedding dimension
        n_dirs: Number of directions
        seed: Random seed

    Returns:
        (n_dirs, dim) array of random unit vectors
    """
    if seed is not None:
        np.random.seed(seed)

    dirs = np.random.randn(n_dirs, dim)
    dirs = _normalize_embeddings(dirs)

    return dirs


# =============================================================================
# Angular Momentum (Correct Conserved Quantity on Sphere)
# =============================================================================

def angular_momentum_tensor(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute angular momentum tensor L_ij = x_i * v_j - x_j * v_i.

    On a sphere, angular momentum IS conserved (not scalar momentum).

    Args:
        x: Position on sphere (d,)
        v: Velocity (tangent vector) (d,)

    Returns:
        L: (d, d) antisymmetric tensor
    """
    d = len(x)
    L = np.outer(x, v) - np.outer(v, x)
    return L


def angular_momentum_magnitude(x: np.ndarray, v: np.ndarray) -> float:
    """
    Compute magnitude of angular momentum |L| = |x × v|.

    For orthogonal x and v (which they are on sphere tangent),
    |L| = |x| * |v| * sin(90 deg) = |v| (since |x| = 1).

    This should be conserved along geodesics.

    Args:
        x: Position on sphere (d,)
        v: Velocity (d,)

    Returns:
        Magnitude of angular momentum
    """
    x = x / (np.linalg.norm(x) + 1e-15)  # Ensure unit norm
    L = angular_momentum_tensor(x, v)
    # Frobenius norm / sqrt(2) gives the "vector" magnitude
    return np.linalg.norm(L, 'fro') / np.sqrt(2)


def angular_momentum_conservation_test(
    trajectory: np.ndarray
) -> Dict[str, float]:
    """
    Test conservation of angular momentum magnitude along trajectory.

    |L| = |v| should be constant for geodesic motion on sphere.

    Args:
        trajectory: (n_steps, d) array of points

    Returns:
        Dict with 'mean', 'std', 'cv' of |L| along trajectory
    """
    velocities = geodesic_velocity(trajectory)

    # Skip first and last (boundary effects)
    n = len(trajectory)
    if n <= 2:
        return {'mean': 0.0, 'std': 0.0, 'cv': 0.0}

    L_magnitudes = []
    for i in range(1, n-1):
        x = trajectory[i]
        v = velocities[i]
        L_mag = angular_momentum_magnitude(x, v)
        L_magnitudes.append(L_mag)

    L_magnitudes = np.array(L_magnitudes)
    mean_L = np.mean(L_magnitudes)
    std_L = np.std(L_magnitudes)
    cv_L = std_L / (np.abs(mean_L) + 1e-10)

    return {
        'mean': float(mean_L),
        'std': float(std_L),
        'cv': float(cv_L)
    }


def principal_plane_angular_momentum(
    trajectory: np.ndarray,
    principal_axes: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Compute angular momentum in each principal plane (e_a, e_b).

    L_ab = (x · e_a)(v · e_b) - (x · e_b)(v · e_a)

    For rotation in a principal plane, L_ab should be conserved.

    Args:
        trajectory: (n_steps, d) array of points
        principal_axes: (n_dims, d) principal directions

    Returns:
        Dict mapping plane index to conservation stats
    """
    velocities = geodesic_velocity(trajectory)
    n_dims = len(principal_axes)
    n_steps = len(trajectory)

    if n_steps <= 2:
        return {}

    # Track L_ab for first few planes
    n_planes = min(10, n_dims * (n_dims - 1) // 2)
    results = {}

    plane_idx = 0
    for a in range(n_dims):
        for b in range(a + 1, n_dims):
            if plane_idx >= n_planes:
                break

            L_values = []
            for i in range(1, n_steps - 1):
                x, v = trajectory[i], velocities[i]
                x_a = np.dot(x, principal_axes[a])
                x_b = np.dot(x, principal_axes[b])
                v_a = np.dot(v, principal_axes[a])
                v_b = np.dot(v, principal_axes[b])
                L_ab = x_a * v_b - x_b * v_a
                L_values.append(L_ab)

            L_values = np.array(L_values)
            mean_L = np.mean(L_values)
            std_L = np.std(L_values)
            cv_L = std_L / (np.abs(mean_L) + 1e-10)

            results[plane_idx] = {
                'plane': (a, b),
                'mean': float(mean_L),
                'std': float(std_L),
                'cv': float(cv_L)
            }
            plane_idx += 1

        if plane_idx >= n_planes:
            break

    return results


# =============================================================================
# High-Level Analysis
# =============================================================================

def analyze_conservation(
    embeddings: np.ndarray,
    n_trajectories: int = 10,
    n_steps: int = 100,
    n_dims: int = 22,
    seed: int = 42
) -> Dict:
    """
    Full conservation analysis on embedding space.

    1. Extract principal subspace
    2. Generate geodesic trajectories
    3. Test conservation along each
    4. Compare to random directions

    Args:
        embeddings: (n_samples, d) array of embeddings
        n_trajectories: Number of test trajectories
        n_steps: Points per trajectory
        n_dims: Principal subspace dimension
        seed: Random seed

    Returns:
        Dict with full analysis results
    """
    np.random.seed(seed)

    # Step 1: Get principal subspace
    projected, axes = project_to_principal_subspace(embeddings, n_dims)

    # Step 2: Measure subspace flatness
    flatness = subspace_metric_flatness(embeddings, axes)

    # Step 3: Compute participation ratio
    if QGT_AVAILABLE:
        df = participation_ratio(embeddings)
    else:
        # Fallback
        emb_norm = _normalize_embeddings(embeddings)
        cov = np.cov(emb_norm.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        df = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)

    # Step 4: Generate trajectories and test conservation
    emb_norm = _normalize_embeddings(embeddings)
    n_samples = len(emb_norm)

    principal_results = []
    random_results = []

    # Generate random directions for comparison
    rand_dirs = random_directions(embeddings.shape[1], n_dims, seed=seed+1000)

    for i in range(n_trajectories):
        # Pick random start and direction
        idx = np.random.randint(n_samples)
        x0 = emb_norm[idx]

        # Random tangent vector
        v0 = np.random.randn(embeddings.shape[1])
        v0 = v0 - np.dot(v0, x0) * x0  # Make tangent
        v0 = _normalize_vector(v0) * 0.5  # Scale velocity

        # Generate geodesic
        traj = sphere_geodesic_trajectory(x0, v0, n_steps=n_steps, t_max=1.0)

        # Test conservation on principal directions
        cons_principal = conservation_test(traj, axes)
        principal_results.append(cons_principal)

        # Test conservation on random directions
        cons_random = conservation_test(traj, rand_dirs)
        random_results.append(cons_random)

    # Aggregate results
    def aggregate_cv(results_list):
        """Average CV across trajectories for each direction."""
        n_dirs = len(results_list[0])
        cvs = np.zeros(n_dirs)
        for d in range(n_dirs):
            cvs[d] = np.mean([r[d]['cv'] for r in results_list])
        return cvs

    cv_principal = aggregate_cv(principal_results)
    cv_random = aggregate_cv(random_results)

    # Count conserved
    n_conserved_principal = np.sum(cv_principal < CV_THRESHOLD_CONSERVED)
    n_conserved_random = np.sum(cv_random < CV_THRESHOLD_CONSERVED)

    return {
        'participation_ratio': float(df),
        'subspace_flatness': float(flatness),
        'n_dims': n_dims,
        'n_trajectories': n_trajectories,
        'cv_principal_mean': float(np.mean(cv_principal)),
        'cv_principal_std': float(np.std(cv_principal)),
        'cv_random_mean': float(np.mean(cv_random)),
        'cv_random_std': float(np.std(cv_random)),
        'n_conserved_principal': int(n_conserved_principal),
        'n_conserved_random': int(n_conserved_random),
        'conservation_ratio': float(n_conserved_principal / n_dims),
        'cv_principal': cv_principal.tolist(),
        'cv_random': cv_random.tolist(),
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Q38: Noether Conservation Laws - Testing")
    print("=" * 60)
    print()

    # Test 1: Sphere geodesic
    print("Test 1: Sphere geodesic (great circle)")
    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    traj = sphere_geodesic_trajectory(x0, v0, n_steps=50, t_max=np.pi/2)

    # Check all points are on unit sphere
    norms = np.linalg.norm(traj, axis=1)
    print(f"  All points on sphere: {np.allclose(norms, 1.0)}")
    print(f"  Start: {traj[0]}")
    print(f"  End: {traj[-1]} (should be near [0,1,0])")
    print()

    # Test 2: Action integral
    print("Test 2: Action integral")
    action_geodesic = action_integral(traj)
    traj_perturbed = perturb_trajectory(traj, noise_scale=0.2, seed=42)
    action_perturbed = action_integral(traj_perturbed)
    print(f"  Action (geodesic): {action_geodesic:.6f}")
    print(f"  Action (perturbed): {action_perturbed:.6f}")
    print(f"  Geodesic shorter: {action_geodesic < action_perturbed}")
    print()

    # Test 3: Conservation with synthetic low-rank embeddings
    print("Test 3: Conservation with synthetic embeddings")
    # Create embeddings in ~22D subspace (like trained BERT)
    np.random.seed(42)
    latent = np.random.randn(200, 22)
    projection = np.random.randn(22, 768)
    embeddings = latent @ projection
    embeddings += 0.1 * np.random.randn(200, 768)  # Add noise

    results = analyze_conservation(embeddings, n_trajectories=5, n_dims=22, seed=42)

    print(f"  Participation ratio (Df): {results['participation_ratio']:.1f}")
    print(f"  Subspace flatness: {results['subspace_flatness']:.4f}")
    print(f"  CV (principal mean): {results['cv_principal_mean']:.4f}")
    print(f"  CV (random mean): {results['cv_random_mean']:.4f}")
    print(f"  Conserved (principal): {results['n_conserved_principal']}/{results['n_dims']}")
    print(f"  Conserved (random): {results['n_conserved_random']}/{results['n_dims']}")
    print()

    print("=" * 60)
    print("Testing complete.")
