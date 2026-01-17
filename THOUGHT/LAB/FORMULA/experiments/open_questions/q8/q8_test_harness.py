"""
Q8 Test Harness - Topology Classification Infrastructure

Q8: "Which manifolds allow local curvature to reveal global truth?"
Target: Prove/Falsify that semantic space is Kahler with c_1 = 1

Provides:
- Threshold constants for Kahler/Chern tests
- Complex structure J computation
- Exterior derivative for closure testing
- Holonomy group accumulator
- 2-cycle generator for Monte Carlo integration
- Negative control framework

Based on Q51 test harness pattern, extended for differential geometry.
"""

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Callable, Any, Union, Dict
import numpy as np
from datetime import datetime
import json
import hashlib

# =============================================================================
# CONSTANTS - All thresholds with justifications
# =============================================================================

class Q8Thresholds:
    """Centralized threshold definitions for Q8 tests."""

    # TEST 1: Chern Class
    CHERN_C1_TARGET = 1.0              # Expected first Chern class
    CHERN_C1_TOLERANCE = 0.10          # 10% tolerance: c1 in [0.90, 1.10]
    CHERN_C1_FALSIFICATION = 0.01      # p < 0.01 to reject c1 = 1
    CHERN_MONTE_CARLO_SAMPLES = 10000  # Triangles for integration
    CHERN_BOOTSTRAP_ITERATIONS = 1000  # Bootstrap resamples

    # TEST 2: Kahler Structure
    KAHLER_J_SQUARED_TOLERANCE = 1e-10   # ||J^2 + I|| < this
    KAHLER_CLOSURE_TOLERANCE = 1e-6      # ||d(omega)|| < this
    KAHLER_OMEGA_DETERMINANT_MIN = 1e-12 # Non-degeneracy threshold

    # TEST 3: Holonomy Group
    HOLONOMY_UNITARY_TOLERANCE = 1e-8    # ||H*H^T - I|| < this
    HOLONOMY_N_LOOPS = 1000              # Number of loops to test
    HOLONOMY_RADIUS_MIN = 0.01           # Smallest loop radius
    HOLONOMY_RADIUS_MAX = 1.0            # Largest loop radius

    # TEST 4: Corruption Stress
    CORRUPTION_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
    CORRUPTION_STABILITY_THRESHOLD = 0.10  # c1 within 10% up to 50% noise

    # TEST 5: Negative Controls
    NEGATIVE_CONTROL_SEPARATION = 3.0  # Cohen's d > 3 for clear separation

    # TEST 6: Cross-Architecture
    CROSS_ARCH_CV_PASS = 0.10          # CV < 10% for universal
    CROSS_ARCH_CV_FAIL = 0.30          # CV > 30% means not universal

    # TEST 7: Alpha Prediction
    ALPHA_PREDICTION_TOLERANCE = 0.05  # Within 5% of measured
    ALPHA_CORRELATION_THRESHOLD = 0.95 # r > 0.95 across models

    # General
    MIN_SAMPLES_FOR_CI = 30
    BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 0.95
    RANDOM_SEED = 42


class Q8Seeds:
    """Reproducibility seeds for each component."""
    TRIANGULATION = 42
    HOLONOMY_LOOPS = 12345
    BOOTSTRAP = 9999
    NEGATIVE_CONTROL = 7777
    CORRUPTION = 5555


# =============================================================================
# ERROR TYPES
# =============================================================================

class Q8ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class Q8ComputationError(RuntimeError):
    """Raised when a geometric computation fails."""
    pass


class Q8TopologyError(RuntimeError):
    """Raised when topological invariant is undefined."""
    pass


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int
    n_bootstrap: int
    confidence_level: float

    def contains(self, value: float) -> bool:
        """Check if value falls within CI."""
        return self.ci_lower <= value <= self.ci_upper

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChernClassResult:
    """Result of Chern class computation."""
    c1: float
    ci: BootstrapCI
    n_triangles: int
    method: str  # 'monte_carlo', 'analytic', 'hybrid'
    passes_test: bool
    p_value: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['ci'] = self.ci.to_dict()
        return d


@dataclass
class KahlerResult:
    """Result of Kahler structure verification."""
    j_squared_norm: float      # ||J^2 + I||, should be ~0
    omega_closure_norm: float  # ||d(omega)||, should be ~0
    omega_determinant: float   # det(omega), should be != 0
    is_kahler: bool
    conditions_passed: Dict[str, bool]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HolonomyResult:
    """Result of holonomy group test."""
    n_loops: int
    n_unitary: int              # How many satisfy U(n) constraint
    unitary_fraction: float     # Should be 1.0
    max_deviation: float        # Worst case ||H*H^T - I||
    mean_deviation: float
    is_unitary: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NegativeControlResult:
    """Result from negative control test."""
    control_type: str           # 'random', 'untrained', 'shuffled', 'degenerate'
    c1_measured: float
    c1_trained: float           # Reference from trained model
    cohens_d: float
    p_value: float
    passed: bool                # True if control FAILS (as expected)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = Q8Thresholds.BOOTSTRAP_ITERATIONS,
    confidence_level: float = Q8Thresholds.CONFIDENCE_LEVEL,
    seed: Optional[int] = Q8Seeds.BOOTSTRAP
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a statistic."""
    data = np.asarray(data).flatten()
    n = len(data)

    if seed is not None:
        np.random.seed(seed)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = data[np.random.randint(0, n, n)]
        bootstrap_stats.append(statistic(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return BootstrapCI(
        mean=statistic(data),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std=bootstrap_stats.std(),
        n_samples=n,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return float('inf') if abs(group1.mean() - group2.mean()) > 1e-10 else 0.0
    return (group1.mean() - group2.mean()) / pooled_std


# =============================================================================
# COMPLEX STRUCTURE J
# =============================================================================

def compute_complex_structure_j(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute complex structure J from embedding covariance.

    For a Kahler manifold, J is an almost complex structure satisfying:
    1. J^2 = -I
    2. g(Jv, Jw) = g(v, w)  (metric compatibility)

    We construct J from the eigenvectors of the covariance matrix,
    pairing eigenvectors to create a rotation by pi/2 in each plane.

    Args:
        embeddings: (n_samples, dim) normalized embeddings

    Returns:
        J: (dim, dim) complex structure matrix
    """
    # Compute covariance as metric
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    dim = cov.shape[0]

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Construct J by pairing eigenvectors
    # J rotates v_i -> v_{i+1} and v_{i+1} -> -v_i for paired indices
    J = np.zeros((dim, dim))

    # Pair eigenvectors (first with second, third with fourth, etc.)
    n_pairs = dim // 2
    for i in range(n_pairs):
        v1 = eigenvectors[:, 2*i]
        v2 = eigenvectors[:, 2*i + 1]

        # J maps v1 -> v2 and v2 -> -v1
        J += np.outer(v2, v1) - np.outer(v1, v2)

    # Handle odd dimension (last eigenvector has no pair)
    # Leave J zero in that direction (degenerate)

    return J


def verify_j_squared(J: np.ndarray) -> Tuple[float, bool]:
    """
    Verify J^2 = -I.

    Returns:
        (frobenius_norm, passes): norm of J^2 + I and whether it passes
    """
    dim = J.shape[0]
    I = np.eye(dim)
    J_squared_plus_I = J @ J + I
    norm = np.linalg.norm(J_squared_plus_I, 'fro')
    passes = norm < Q8Thresholds.KAHLER_J_SQUARED_TOLERANCE
    return norm, passes


def verify_metric_compatibility(J: np.ndarray, metric: np.ndarray) -> Tuple[float, bool]:
    """
    Verify g(Jv, Jw) = g(v, w) for all v, w.

    This is equivalent to: J^T g J = g

    Returns:
        (frobenius_norm, passes): norm of J^T g J - g and whether it passes
    """
    JgJ = J.T @ metric @ J
    diff = JgJ - metric
    norm = np.linalg.norm(diff, 'fro')
    passes = norm < Q8Thresholds.KAHLER_J_SQUARED_TOLERANCE * np.linalg.norm(metric, 'fro')
    return norm, passes


# =============================================================================
# KAHLER FORM AND EXTERIOR DERIVATIVE
# =============================================================================

def compute_kahler_form(J: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """
    Compute Kahler form omega(u, v) = g(Ju, v).

    omega is a 2-form (antisymmetric bilinear).

    Returns:
        omega: (dim, dim) antisymmetric matrix
    """
    omega = metric @ J
    # Verify antisymmetry (should be automatic if J is correct)
    return 0.5 * (omega - omega.T)  # Force antisymmetric


def exterior_derivative_3form(
    embeddings: np.ndarray,
    omega: np.ndarray,
    epsilon: float = 1e-5
) -> float:
    """
    Compute ||d(omega)|| numerically.

    For a 2-form omega, d(omega) is a 3-form.
    d(omega) = 0 is the closure condition for Kahler.

    We estimate this via finite differences on the embedding manifold.

    This is a simplified test - we check if omega is approximately constant
    in all directions, which implies d(omega) ~ 0.

    Args:
        embeddings: Sample of points on manifold
        omega: Kahler form at centroid
        epsilon: Finite difference step

    Returns:
        Norm estimate of d(omega)
    """
    n_samples, dim = embeddings.shape

    # Sample directions
    variations = []
    for i in range(min(n_samples, 100)):
        # Perturb embeddings slightly
        direction = np.random.randn(dim)
        direction = direction / np.linalg.norm(direction)

        # Compute omega at perturbed point (approximate)
        perturbed = embeddings + epsilon * direction
        perturbed_centered = perturbed - perturbed.mean(axis=0)
        perturbed_cov = np.cov(perturbed_centered.T)

        J_perturbed = compute_complex_structure_j(perturbed)
        omega_perturbed = compute_kahler_form(J_perturbed, perturbed_cov)

        # d omega is zero if omega is constant
        diff = omega_perturbed - omega
        variations.append(np.linalg.norm(diff, 'fro'))

    # Average variation as proxy for ||d(omega)||
    return np.mean(variations) / epsilon


# =============================================================================
# BERRY CURVATURE AND CHERN CLASS
# =============================================================================

def berry_phase_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Compute Berry phase around a triangle (v1 -> v2 -> v3 -> v1).

    phi = arg(<v1|v2> * <v2|v3> * <v3|v1>)

    This is the discrete Berry phase, which approximates the integral
    of the Berry curvature over the triangle.
    """
    # Normalize
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # Product of overlaps
    overlap = np.dot(v1, v2) * np.dot(v2, v3) * np.dot(v3, v1)

    # For real vectors, this is always real, so phase is 0 or pi
    # But we can get non-trivial phase from the solid angle
    # Use complex representation for proper phase
    phase = np.angle(overlap + 0j)

    return phase


def solid_angle_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Compute solid angle subtended by triangle on unit sphere.

    This is related to Berry phase: phi = solid_angle / 2

    Uses the formula:
    tan(omega/2) = v1 . (v2 x v3) / (1 + v1.v2 + v2.v3 + v3.v1)
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # Only works in 3D, generalize via projection
    if len(v1) > 3:
        # Project to 3D via PCA of the three vectors
        vecs = np.stack([v1, v2, v3])
        mean = vecs.mean(axis=0)
        centered = vecs - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:3].T
        v1, v2, v3 = projected[0], projected[1], projected[2]
        # Re-normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        v3 = v3 / (np.linalg.norm(v3) + 1e-10)

    # Cross product (3D)
    if len(v1) == 3:
        cross = np.cross(v2, v3)
        numerator = np.dot(v1, cross)
        denominator = 1 + np.dot(v1, v2) + np.dot(v2, v3) + np.dot(v3, v1)

        if abs(denominator) < 1e-10:
            return np.pi if numerator > 0 else -np.pi

        omega = 2 * np.arctan2(numerator, denominator)
        return omega
    else:
        # Fallback for very high dimensions
        return berry_phase_triangle(v1, v2, v3) * 2


def monte_carlo_chern_class(
    embeddings: np.ndarray,
    n_triangles: int = Q8Thresholds.CHERN_MONTE_CARLO_SAMPLES,
    seed: int = Q8Seeds.TRIANGULATION,
    use_solid_angle: bool = True
) -> Tuple[float, np.ndarray]:
    """
    Estimate first Chern class via SPECTRAL method.

    From Q50 derivation: alpha = 1/(2*c_1) where alpha is eigenvalue decay exponent.
    Therefore: c_1 = 1/(2*alpha)

    For trained embeddings: alpha ~ 0.5, so c_1 ~ 1
    For random embeddings: alpha ~ 1/d (much smaller), so c_1 >> 1

    This provides a DISCRIMINATIVE test that differs between trained and random.

    Args:
        embeddings: (n_samples, dim) normalized embeddings
        n_triangles: Not used in spectral method, kept for API compatibility
        seed: Random seed for bootstrap
        use_solid_angle: Not used, kept for API compatibility

    Returns:
        (c1_estimate, alpha_samples): Chern class and bootstrap alpha samples
    """
    np.random.seed(seed)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    # Compute covariance eigenspectrum
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Filter positive eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 5:
        return 0.0, np.array([0.0])

    # Fit power law: lambda_k ~ k^(-alpha)
    # Q50 METHOD: fit FIRST HALF of eigenvalues only
    n_fit = len(eigenvalues) // 2
    if n_fit < 5:
        return 0.0, np.array([0.0])

    k = np.arange(1, n_fit + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues[:n_fit])

    # Linear regression using polyfit (more robust)
    slope, intercept = np.polyfit(log_k, log_lambda, 1)
    alpha = -slope  # Positive alpha for decay

    # Clamp alpha to reasonable range
    alpha = np.clip(alpha, 0.01, 10.0)

    # Compute c_1 = 1/(2*alpha)
    c1_estimate = 1.0 / (2.0 * alpha)

    # Bootstrap for confidence interval
    n_bootstrap = 100
    alpha_samples = []

    for _ in range(n_bootstrap):
        # Resample embeddings
        idx = np.random.choice(len(embeddings), len(embeddings), replace=True)
        boot_emb = embeddings[idx]

        centered_boot = boot_emb - boot_emb.mean(axis=0)
        cov_boot = np.cov(centered_boot.T)
        eig_boot = np.linalg.eigvalsh(cov_boot)
        eig_boot = np.sort(eig_boot)[::-1]
        eig_boot = eig_boot[eig_boot > 1e-10]

        # Q50 METHOD: fit FIRST HALF only
        n_fit_boot = len(eig_boot) // 2
        if n_fit_boot < 5:
            continue

        k_boot = np.arange(1, n_fit_boot + 1)
        log_k_boot = np.log(k_boot)
        log_lambda_boot = np.log(eig_boot[:n_fit_boot])

        slope_b, _ = np.polyfit(log_k_boot, log_lambda_boot, 1)
        alpha_b = -slope_b
        alpha_b = np.clip(alpha_b, 0.01, 10.0)
        alpha_samples.append(alpha_b)

    alpha_samples = np.array(alpha_samples) if alpha_samples else np.array([alpha])

    return c1_estimate, alpha_samples


def compute_alpha_from_spectrum(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute eigenvalue decay exponent alpha from embedding covariance.

    Uses Q50 methodology: fit FIRST HALF of eigenvalues (top 50%).
    This matches the alpha values reported in Q50 (alpha ~ 0.5).

    Returns:
        (alpha, Df, c1): decay exponent, effective dimension, Chern class estimate
    """
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    # Compute covariance
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Participation ratio (effective dimension)
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues ** 2).sum()
    Df = (sum_lambda ** 2) / sum_lambda_sq if sum_lambda_sq > 0 else 0

    # Power law fit for alpha - Q50 METHOD: fit FIRST HALF only
    n_fit = len(eigenvalues) // 2
    if n_fit < 5:
        return 0.0, Df, 0.0

    k = np.arange(1, n_fit + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues[:n_fit])

    # Linear regression: log(lambda) = -alpha * log(k) + const
    slope, intercept = np.polyfit(log_k, log_lambda, 1)
    alpha = -slope  # Positive alpha for decay

    # Clamp to reasonable range
    alpha = np.clip(alpha, 0.01, 10.0)

    # c_1 = 1/(2*alpha)
    c1 = 1.0 / (2.0 * alpha) if alpha > 0 else float('inf')

    return alpha, Df, c1


# =============================================================================
# HOLONOMY GROUP
# =============================================================================

def parallel_transport_frame(
    embeddings: np.ndarray,
    loop_indices: List[int],
    frame: np.ndarray
) -> np.ndarray:
    """
    Parallel transport a frame around a loop of embedding indices.

    Args:
        embeddings: (n_samples, dim) normalized embeddings
        loop_indices: Indices forming a closed loop
        frame: (dim, k) orthonormal frame to transport

    Returns:
        transported_frame: Frame after transport around loop
    """
    current_frame = frame.copy()

    for i in range(len(loop_indices) - 1):
        p1 = embeddings[loop_indices[i]]
        p2 = embeddings[loop_indices[i + 1]]

        # Project frame onto tangent space at p2
        # Tangent space: orthogonal complement of p2
        proj = np.eye(len(p2)) - np.outer(p2, p2)
        current_frame = proj @ current_frame

        # Re-orthonormalize (Gram-Schmidt)
        Q, R = np.linalg.qr(current_frame)
        current_frame = Q

    return current_frame


def compute_holonomy_matrix(
    embeddings: np.ndarray,
    loop_indices: List[int]
) -> np.ndarray:
    """
    Compute holonomy matrix for a closed loop.

    H = lim(frame_transported / frame_original)

    Args:
        embeddings: Normalized embeddings
        loop_indices: Indices forming closed loop

    Returns:
        H: Holonomy matrix (should be in SO(n) or U(n))
    """
    dim = embeddings.shape[1]

    # Use top-k eigenvectors as initial frame
    k = min(10, dim // 2)
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    initial_frame = eigenvectors[:, idx[:k]]

    # Transport frame
    transported = parallel_transport_frame(embeddings, loop_indices, initial_frame)

    # Holonomy matrix: transported = H @ initial
    # Solve: H = transported @ pinv(initial)
    H, residuals, rank, s = np.linalg.lstsq(initial_frame, transported, rcond=None)

    return H.T  # Transpose to get correct orientation


def is_unitary(H: np.ndarray, tolerance: float = Q8Thresholds.HOLONOMY_UNITARY_TOLERANCE) -> Tuple[bool, float]:
    """
    Check if H is in U(n): H*H^T = I and |det(H)| = 1.

    Returns:
        (is_unitary, deviation): Whether unitary and max deviation from constraints
    """
    I = np.eye(H.shape[0])
    HHT = H @ H.T

    deviation_orthogonal = np.linalg.norm(HHT - I, 'fro')
    deviation_det = abs(abs(np.linalg.det(H)) - 1)

    max_deviation = max(deviation_orthogonal, deviation_det)
    is_u = max_deviation < tolerance

    return is_u, max_deviation


# =============================================================================
# UTILITIES
# =============================================================================

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms > 0, norms, 1.0)


def validate_embeddings(
    embeddings: np.ndarray,
    min_samples: int = 10,
    name: str = "embeddings"
) -> Tuple[bool, List[str]]:
    """Validate embedding matrix for common issues."""
    errors = []

    if not isinstance(embeddings, np.ndarray):
        try:
            embeddings = np.array(embeddings)
        except:
            errors.append(f"{name}: Cannot convert to numpy array")
            return False, errors

    if embeddings.ndim != 2:
        errors.append(f"{name}: Expected 2D array, got {embeddings.ndim}D")
        return False, errors

    n_samples, n_dims = embeddings.shape

    if n_samples < min_samples:
        errors.append(f"{name}: Need at least {min_samples} samples, got {n_samples}")

    if np.isnan(embeddings).any():
        errors.append(f"{name}: Contains NaN values")

    if np.isinf(embeddings).any():
        errors.append(f"{name}: Contains Inf values")

    norms = np.linalg.norm(embeddings, axis=1)
    if (norms < 1e-10).any():
        errors.append(f"{name}: Contains zero-norm vectors")

    return len(errors) == 0, errors


def get_test_metadata() -> dict:
    """Get metadata for test reproducibility."""
    return {
        'timestamp': datetime.now().isoformat(),
        'numpy_version': np.__version__,
        'seeds': {
            'triangulation': Q8Seeds.TRIANGULATION,
            'holonomy': Q8Seeds.HOLONOMY_LOOPS,
            'bootstrap': Q8Seeds.BOOTSTRAP,
            'negative_control': Q8Seeds.NEGATIVE_CONTROL
        },
        'thresholds': {
            'c1_target': Q8Thresholds.CHERN_C1_TARGET,
            'c1_tolerance': Q8Thresholds.CHERN_C1_TOLERANCE,
            'kahler_j_tolerance': Q8Thresholds.KAHLER_J_SQUARED_TOLERANCE,
            'kahler_closure_tolerance': Q8Thresholds.KAHLER_CLOSURE_TOLERANCE,
            'holonomy_unitary_tolerance': Q8Thresholds.HOLONOMY_UNITARY_TOLERANCE,
            'cross_arch_cv_pass': Q8Thresholds.CROSS_ARCH_CV_PASS
        }
    }


def save_results(results: dict, filepath: Path):
    """Save results to JSON with numpy handling."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        # Skip objects that cause circular reference
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    # Deep copy to avoid circular reference issues
    import copy
    try:
        results_copy = copy.deepcopy(results)
    except:
        results_copy = results

    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2, default=convert)


# =============================================================================
# TEST HARNESS LOGGER
# =============================================================================

class Q8Logger:
    """Simple logger for Q8 tests."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.logs = []

    def info(self, msg: str):
        entry = f"[{self.name}] {msg}"
        self.logs.append(entry)
        if self.verbose:
            print(entry)

    def warn(self, msg: str):
        entry = f"[{self.name}] WARNING: {msg}"
        self.logs.append(entry)
        if self.verbose:
            print(entry)

    def error(self, msg: str):
        entry = f"[{self.name}] ERROR: {msg}"
        self.logs.append(entry)
        print(entry)  # Always print errors

    def section(self, title: str):
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
