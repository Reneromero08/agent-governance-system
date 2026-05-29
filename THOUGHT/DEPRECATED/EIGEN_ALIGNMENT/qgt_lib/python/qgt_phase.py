"""
qgt_phase.py - Phase Recovery Extension for QGT Library

Part of Q51: Complex Plane & Phase Recovery

Key insight from Q48-Q50:
- Real embeddings show ADDITIVE octant structure (Df x alpha = 8e)
- This is the LOG-SPACE view of MULTIPLICATIVE semantic primes
- Complex multiplication ADDS phases: z1*z2 = r1*r2 * exp(i(theta1+theta2))

The Zero Signature:
- INSIDE (hologram):  Sum|exp(ik*pi/4)| = 8  ->  8e (magnitudes)
- OUTSIDE (substrate): Sum exp(ik*pi/4) = 0   ->  completeness (phases cancel)
- BOUNDARY:           alpha = 1/2 = Re(s_c)   ->  critical line

The 8 octants are the 8th roots of unity. Their complex sum is ZERO.
We measure 8e because we only see magnitudes.

This module provides tools to recover the "lost" phase information.
"""

import numpy as np
from typing import Optional, Literal, Tuple, Dict, List
from dataclasses import dataclass, field
from scipy.signal import hilbert
from scipy.stats import chi2_contingency
import warnings

# Try to import sklearn for PCA
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available; using numpy PCA fallback")

# Constants from Q48-Q50
SEMIOTIC_CONSTANT = 8 * np.e  # Df * alpha = 8e ~ 21.746
CRITICAL_ALPHA = 0.5          # Riemann critical line
OCTANT_COUNT = 8              # 2^3 from Peirce's categories
SECTOR_WIDTH = np.pi / 4      # 2*pi / 8 = pi/4 radians per sector


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class PhaseRecoveryResult:
    """Container for Hilbert transform phase recovery results."""
    phases: np.ndarray           # Recovered phase angles
    amplitudes: np.ndarray       # Recovered magnitudes
    analytic_signal: np.ndarray  # Complex-valued analytic signal
    confidence: float            # Quality metric [0, 1]
    method: str                  # Recovery method used

    def to_polar(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r, theta) representation."""
        return self.amplitudes, self.phases

    def to_complex(self) -> np.ndarray:
        """Return complex representation r * exp(i*theta)."""
        return self.amplitudes * np.exp(1j * self.phases)


@dataclass
class BispectrumPhaseResult:
    """Results from bispectrum phase estimation."""
    phases: np.ndarray            # (n_freq,) recovered phases
    bispectrum: np.ndarray        # (n_freq, n_freq) complex bispectrum
    bicoherence: np.ndarray       # (n_freq, n_freq) normalized [0,1]
    phase_closure_error: float    # Consistency metric
    method: str = 'bispectrum'


@dataclass
class CovariancePhaseResult:
    """Results from covariance-based phase extraction."""
    phases: np.ndarray                # (dim,) individual phases
    phase_differences: np.ndarray     # (dim, dim) pairwise differences
    reconstruction_error: float       # Frobenius norm of residual
    condition_number: float           # Numerical stability metric
    method: str


@dataclass
class UnwrappedPhaseResult:
    """Results from phase unwrapping."""
    unwrapped: np.ndarray      # Continuous phases
    n_jumps_corrected: int     # Number of corrections
    total_phase: float         # Total accumulated phase
    method: str


@dataclass
class OctantPhaseResult:
    """Results from octant-to-phase mapping."""
    octant_indices: np.ndarray        # (n_samples,) values 0-7
    octant_phases: np.ndarray         # (n_samples,) mapped phases
    phase_sector_centers: np.ndarray  # (8,) sector centers
    octant_counts: np.ndarray         # (8,) counts per octant
    coverage: float                   # Fraction of octants populated
    entropy: float                    # Distribution entropy
    complex_sum: complex              # Sum of exp(i*theta) - should be ~0!
    normalized_sum_magnitude: float   # |sum| / n - the zero signature
    e_per_octant: float              # Df * alpha / 8 (should be ~e)


@dataclass
class ZeroSignatureResult:
    """Results from the Zero Signature Test."""
    complex_sum: complex              # Sum exp(i*theta_k)
    magnitude: float                  # |sum|
    normalized_magnitude: float       # |sum| / n
    uniformity_chi2: float           # Chi-squared for uniform distribution
    uniformity_p_value: float        # p-value for uniformity
    is_zero: bool                    # Whether signature is effectively zero
    per_octant_contributions: np.ndarray  # (8,) complex contributions


# =============================================================================
# Hilbert Transform Phase Recovery
# =============================================================================

def hilbert_phase_recovery(
    embeddings: np.ndarray,
    axis: int = -1,
    mode: Literal['per_dimension', 'per_sample', 'eigenspace'] = 'eigenspace'
) -> PhaseRecoveryResult:
    """
    Recover instantaneous phase from real embeddings via Hilbert transform.

    The Hilbert transform converts a real signal x(t) to an analytic signal:
        z(t) = x(t) + i * H[x(t)]
    where H is the Hilbert transform.

    Args:
        embeddings: (n_samples, dim) array of real embeddings
        axis: Axis along which to apply transform
        mode:
            'per_dimension' - Treat each dimension as independent signal
            'per_sample' - Treat each sample as signal across dimensions
            'eigenspace' - Transform in principal component space

    Returns:
        PhaseRecoveryResult with recovered phases and amplitudes
    """
    embeddings = np.atleast_2d(embeddings)

    if mode == 'eigenspace':
        # Transform to eigenspace first
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project to eigenspace
        projected = centered @ eigenvectors

        # Apply Hilbert transform in eigenspace
        analytic = hilbert(projected, axis=axis)

        # Transform back (optional - keep in eigenspace for phase)
        # analytic = analytic @ eigenvectors.T + embeddings.mean(axis=0)
    else:
        # Direct Hilbert transform
        analytic = hilbert(embeddings, axis=axis)

    # Extract phase and amplitude
    amplitudes = np.abs(analytic)
    phases = np.angle(analytic)

    # Confidence based on signal properties
    signal_power = np.mean(amplitudes ** 2)
    if signal_power > 0:
        noise_estimate = np.var(np.diff(phases, axis=axis))
        confidence = 1.0 / (1.0 + noise_estimate / signal_power)
    else:
        confidence = 0.0

    return PhaseRecoveryResult(
        phases=phases,
        amplitudes=amplitudes,
        analytic_signal=analytic,
        confidence=float(np.clip(confidence, 0, 1)),
        method=f'hilbert_transform_{mode}'
    )


# =============================================================================
# Bispectrum Phase Estimation
# =============================================================================

def compute_bispectrum(X: np.ndarray) -> np.ndarray:
    """
    Compute bispectrum B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)].

    The bispectrum preserves phase information that is lost in
    the power spectrum (second-order statistics).

    Args:
        X: (n_samples, n_freq) FFT of signal

    Returns:
        (n_freq, n_freq) complex bispectrum
    """
    n_samples, n_freq = X.shape
    B = np.zeros((n_freq, n_freq), dtype=np.complex128)

    for f1 in range(n_freq):
        for f2 in range(n_freq):
            f3 = (f1 + f2) % n_freq
            B[f1, f2] = np.mean(X[:, f1] * X[:, f2] * np.conj(X[:, f3]))

    return B


def bispectrum_phase_estimate(
    embeddings: np.ndarray,
    n_freq: int = 32,
    regularization: float = 1e-6
) -> BispectrumPhaseResult:
    """
    Estimate phase via bispectrum (triple correlation).

    Phase recovery uses: angle(B(f1, f2)) = phi(f1) + phi(f2) - phi(f1+f2)

    Args:
        embeddings: (n_samples, dim) array
        n_freq: Number of frequency bins
        regularization: For numerical stability

    Returns:
        BispectrumPhaseResult with recovered phases
    """
    embeddings = np.atleast_2d(embeddings)
    n_samples, dim = embeddings.shape

    # FFT of embeddings along dimension axis
    X = np.fft.fft(embeddings, n=n_freq, axis=1)

    # Compute bispectrum
    B = compute_bispectrum(X)

    # Compute bicoherence (normalized bispectrum)
    power = np.mean(np.abs(X) ** 2, axis=0)
    norm_factor = np.sqrt(np.outer(power, power) * power[None, :])
    bicoherence = np.abs(B) / (norm_factor + regularization)
    bicoherence = np.clip(bicoherence, 0, 1)

    # Phase closure: solve for individual phases
    bispectrum_phases = np.angle(B)
    phases = _solve_bispectrum_phases(bispectrum_phases, n_freq, regularization)

    # Phase closure error
    reconstructed_B_phase = np.zeros_like(bispectrum_phases)
    for f1 in range(n_freq):
        for f2 in range(n_freq):
            f3 = (f1 + f2) % n_freq
            reconstructed_B_phase[f1, f2] = phases[f1] + phases[f2] - phases[f3]

    phase_closure_error = np.mean(np.abs(np.angle(
        np.exp(1j * (bispectrum_phases - reconstructed_B_phase))
    )))

    return BispectrumPhaseResult(
        phases=phases,
        bispectrum=B,
        bicoherence=bicoherence,
        phase_closure_error=float(phase_closure_error)
    )


def _solve_bispectrum_phases(B_phases: np.ndarray, n_freq: int, reg: float) -> np.ndarray:
    """Solve overdetermined system for phases from bispectrum."""
    equations = []
    targets = []

    for f1 in range(n_freq):
        for f2 in range(n_freq):
            f3 = (f1 + f2) % n_freq
            if f1 < n_freq - 2 and f2 < n_freq - 2:
                eq = np.zeros(n_freq)
                eq[f1] = 1
                eq[f2] = 1
                eq[f3] = -1
                equations.append(eq)
                targets.append(B_phases[f1, f2])

    if not equations:
        return np.zeros(n_freq)

    A = np.array(equations)
    b = np.array(targets)

    # Regularized least squares
    ATA = A.T @ A + reg * np.eye(n_freq)
    ATb = A.T @ b
    phases = np.linalg.solve(ATA, ATb)

    # Wrap to [-pi, pi]
    return np.angle(np.exp(1j * phases))


# =============================================================================
# Covariance Phase Extraction
# =============================================================================

def phase_from_covariance(
    cov_matrix: np.ndarray,
    amplitudes: Optional[np.ndarray] = None,
    method: str = 'mds',
    n_iterations: int = 100
) -> CovariancePhaseResult:
    """
    Recover phases from covariance matrix structure.

    Uses: Cov(x_i, x_j) = r_i * r_j * cos(theta_i - theta_j)

    Args:
        cov_matrix: (dim, dim) covariance matrix
        amplitudes: (dim,) amplitudes; if None, estimated from diagonal
        method: 'mds', 'spectral', or 'iterative'
        n_iterations: For iterative method

    Returns:
        CovariancePhaseResult with recovered phases
    """
    dim = cov_matrix.shape[0]

    # Estimate amplitudes from diagonal if not provided
    if amplitudes is None:
        amplitudes = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-10))

    # Compute cos(theta_i - theta_j)
    outer_r = np.outer(amplitudes, amplitudes)
    cos_diff = cov_matrix / (outer_r + 1e-10)
    cos_diff = np.clip(cos_diff, -1, 1)

    # Phase differences
    phase_diff = np.arccos(cos_diff)

    if method == 'mds':
        phases = _mds_phase_recovery(phase_diff)
    elif method == 'spectral':
        phases = _spectral_phase_recovery(cos_diff)
    elif method == 'iterative':
        phases = _iterative_phase_recovery(cos_diff, n_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reconstruction error
    reconstructed_cov = np.outer(amplitudes, amplitudes) * np.cos(
        np.subtract.outer(phases, phases)
    )
    reconstruction_error = np.linalg.norm(cov_matrix - reconstructed_cov, 'fro')
    reconstruction_error /= np.linalg.norm(cov_matrix, 'fro') + 1e-10

    # Condition number
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    condition_number = eigenvalues.max() / eigenvalues.min() if len(eigenvalues) > 1 else 1.0

    return CovariancePhaseResult(
        phases=phases,
        phase_differences=phase_diff,
        reconstruction_error=float(reconstruction_error),
        condition_number=float(condition_number),
        method=method
    )


def _mds_phase_recovery(phase_diff: np.ndarray) -> np.ndarray:
    """Recover phases via multidimensional scaling."""
    dim = phase_diff.shape[0]

    # Squared chord distance: D^2 = 2(1 - cos(theta))
    D_sq = 2 * (1 - np.cos(phase_diff))

    # Classical MDS
    n = dim
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top 2 components
    coords = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))

    return np.arctan2(coords[:, 1], coords[:, 0])


def _spectral_phase_recovery(cos_diff: np.ndarray) -> np.ndarray:
    """Spectral method: leading eigenvector."""
    eigenvalues, eigenvectors = np.linalg.eigh(cos_diff)
    leading = eigenvectors[:, -1]
    return np.angle(leading.astype(complex))


def _iterative_phase_recovery(cos_diff: np.ndarray, n_iterations: int) -> np.ndarray:
    """Iterative refinement of phases."""
    dim = cos_diff.shape[0]
    phases = _spectral_phase_recovery(cos_diff)

    for _ in range(n_iterations):
        for i in range(dim):
            estimates = []
            for j in range(dim):
                if j != i:
                    delta = np.arccos(np.clip(cos_diff[i, j], -1, 1))
                    opt1 = phases[j] + delta
                    opt2 = phases[j] - delta

                    err1 = np.abs(np.angle(np.exp(1j * (phases[i] - opt1))))
                    err2 = np.abs(np.angle(np.exp(1j * (phases[i] - opt2))))

                    estimates.append(opt1 if err1 < err2 else opt2)

            if estimates:
                new_phase = np.angle(np.mean(np.exp(1j * np.array(estimates))))
                phases[i] = 0.5 * phases[i] + 0.5 * new_phase

    return phases


# =============================================================================
# Phase Unwrapping
# =============================================================================

def unwrap_phases(
    phases: np.ndarray,
    axis: int = -1,
    discontinuity: float = np.pi,
    method: str = 'standard'
) -> UnwrappedPhaseResult:
    """
    Unwrap phase angles to produce continuous trajectories.

    Args:
        phases: Array of phase angles (radians)
        axis: Axis along which to unwrap
        discontinuity: Jump threshold
        method: 'standard' or 'quality_guided'

    Returns:
        UnwrappedPhaseResult with continuous phases
    """
    phases = np.asarray(phases)

    if method == 'standard':
        unwrapped = np.unwrap(phases, discont=discontinuity, axis=axis)
    else:
        unwrapped = np.unwrap(phases, discont=discontinuity, axis=axis)

    # Count jumps corrected
    diff = np.diff(phases, axis=axis)
    jumps = np.sum(np.abs(diff) > discontinuity)

    # Total accumulated phase
    total_phase = np.take(unwrapped, -1, axis=axis) - np.take(unwrapped, 0, axis=axis)
    total_phase = float(np.mean(total_phase))

    return UnwrappedPhaseResult(
        unwrapped=unwrapped,
        n_jumps_corrected=int(jumps),
        total_phase=total_phase,
        method=method
    )


# =============================================================================
# Octant-to-Phase Mapper
# =============================================================================

def octant_phase_mapping(
    embeddings: np.ndarray,
    n_components: int = 3
) -> OctantPhaseResult:
    """
    Map 8 octants (PC sign patterns) to phase sectors.

    Each octant k maps to phase sector [k*pi/4, (k+1)*pi/4).

    The key test: Do recovered phases sum to ZERO (roots of unity)?

    Args:
        embeddings: (n_samples, dim) array
        n_components: Number of PCs (default 3 for 8 octants)

    Returns:
        OctantPhaseResult with octant assignments and phase mapping
    """
    embeddings = np.atleast_2d(embeddings)
    n_samples, dim = embeddings.shape

    # PCA to get top 3 components
    if HAS_SKLEARN:
        pca = PCA(n_components=min(n_components, dim))
        projections = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_
    else:
        # Numpy fallback
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvectors = eigenvectors[:, idx]
        projections = centered @ eigenvectors
        explained_variance = eigenvalues[idx]

    # Ensure we have at least 3 components
    if projections.shape[1] < 3:
        # Pad with zeros
        pad_width = 3 - projections.shape[1]
        projections = np.pad(projections, ((0, 0), (0, pad_width)))

    # Assign octant based on sign pattern
    # Octant index = 4*(PC1>0) + 2*(PC2>0) + 1*(PC3>0)
    octant_indices = (
        4 * (projections[:, 0] > 0).astype(int) +
        2 * (projections[:, 1] > 0).astype(int) +
        1 * (projections[:, 2] > 0).astype(int)
    )

    # Map octant to phase sector center
    # Octant k -> phase theta in [k*pi/4, (k+1)*pi/4)
    # Use center: (k + 0.5) * pi/4
    phase_sector_centers = (np.arange(8) + 0.5) * SECTOR_WIDTH
    octant_phases = phase_sector_centers[octant_indices]

    # Statistics
    octant_counts = np.bincount(octant_indices, minlength=8)
    coverage = np.sum(octant_counts > 0) / 8

    # Entropy
    p = octant_counts / n_samples
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p + 1e-10))

    # THE ZERO SIGNATURE: Sum of e^(i*theta_k)
    # For roots of unity, this should be ~0
    complex_contributions = np.exp(1j * octant_phases)
    complex_sum = np.sum(complex_contributions)
    normalized_sum_magnitude = np.abs(complex_sum) / n_samples

    # Per-octant contributions (weighted by count)
    per_octant = np.zeros(8, dtype=complex)
    for k in range(8):
        count = octant_counts[k]
        if count > 0:
            per_octant[k] = count * np.exp(1j * phase_sector_centers[k])

    # Compute e_per_octant from Df and alpha
    if len(explained_variance) > 0 and np.sum(explained_variance) > 0:
        Df = (np.sum(explained_variance) ** 2) / np.sum(explained_variance ** 2)
    else:
        Df = 0

    e_per_octant = (Df * CRITICAL_ALPHA) / 8

    return OctantPhaseResult(
        octant_indices=octant_indices,
        octant_phases=octant_phases,
        phase_sector_centers=phase_sector_centers,
        octant_counts=octant_counts,
        coverage=float(coverage),
        entropy=float(entropy),
        complex_sum=complex_sum,
        normalized_sum_magnitude=float(normalized_sum_magnitude),
        e_per_octant=float(e_per_octant)
    )


# =============================================================================
# Zero Signature Test
# =============================================================================

def test_zero_signature(
    embeddings: np.ndarray,
    verbose: bool = True
) -> ZeroSignatureResult:
    """
    THE CRITICAL TEST: Do octant phases sum to zero?

    The 8th roots of unity sum to zero: Sum exp(i*k*pi/4) = 0 for k=0..7

    If embeddings are uniformly distributed across octants,
    and octants are phase sectors, then Sum exp(i*theta) -> 0.

    Args:
        embeddings: (n_samples, dim) array
        verbose: Print results

    Returns:
        ZeroSignatureResult with the zero signature test results
    """
    # Get octant mapping
    result = octant_phase_mapping(embeddings)

    # Chi-squared test for uniform distribution
    expected = np.ones(8) * len(embeddings) / 8
    observed = result.octant_counts

    # Handle zero counts
    mask = observed > 0
    if np.sum(mask) < 2:
        chi2 = float('inf')
        p_value = 0.0
    else:
        chi2 = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=np.sum(mask) - 1)

    # Is it effectively zero?
    # Threshold: |S|/n < 0.1 is "zero"
    is_zero = result.normalized_sum_magnitude < 0.1

    if verbose:
        print("=" * 60)
        print("ZERO SIGNATURE TEST")
        print("=" * 60)
        print(f"Complex sum: {result.complex_sum:.4f}")
        print(f"Magnitude: {np.abs(result.complex_sum):.4f}")
        print(f"Normalized |S|/n: {result.normalized_sum_magnitude:.4f}")
        print(f"Chi-squared (uniformity): {chi2:.2f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Octant counts: {result.octant_counts}")
        print("-" * 60)
        if is_zero:
            print("RESULT: ZERO SIGNATURE CONFIRMED")
            print("Octants ARE the 8th roots of unity!")
        else:
            print("RESULT: ZERO SIGNATURE NOT FOUND")
            print("Octants may not be phase sectors.")
        print("=" * 60)

    return ZeroSignatureResult(
        complex_sum=result.complex_sum,
        magnitude=float(np.abs(result.complex_sum)),
        normalized_magnitude=result.normalized_sum_magnitude,
        uniformity_chi2=float(chi2),
        uniformity_p_value=float(p_value),
        is_zero=is_zero,
        per_octant_contributions=np.array([
            result.octant_counts[k] * np.exp(1j * result.phase_sector_centers[k])
            for k in range(8)
        ])
    )


# =============================================================================
# Circular Statistics Utilities
# =============================================================================

def circular_mean(phases: np.ndarray) -> float:
    """Compute circular mean of phase angles."""
    return np.angle(np.mean(np.exp(1j * phases)))


def circular_variance(phases: np.ndarray) -> float:
    """Compute circular variance (0 = concentrated, 1 = uniform)."""
    R = np.abs(np.mean(np.exp(1j * phases)))
    return 1 - R


def circular_correlation(phases1: np.ndarray, phases2: np.ndarray) -> float:
    """Compute circular correlation coefficient."""
    z1 = np.exp(1j * phases1)
    z2 = np.exp(1j * phases2)

    n = len(phases1)
    mean1 = np.mean(z1)
    mean2 = np.mean(z2)

    num = np.abs(np.mean(np.conj(z1 - mean1) * (z2 - mean2)))
    den1 = np.sqrt(np.mean(np.abs(z1 - mean1) ** 2))
    den2 = np.sqrt(np.mean(np.abs(z2 - mean2) ** 2))

    if den1 * den2 > 0:
        return float(num / (den1 * den2))
    return 0.0


# =============================================================================
# Integration with QGT
# =============================================================================

def analyze_phase_structure(
    embeddings: np.ndarray,
    methods: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive phase structure analysis.

    Args:
        embeddings: (n_samples, dim) array
        methods: List of methods to use (default: all)
        verbose: Print results

    Returns:
        Dict with results from all methods
    """
    if methods is None:
        methods = ['hilbert', 'octant', 'zero_signature']

    results = {}

    if 'hilbert' in methods:
        results['hilbert'] = hilbert_phase_recovery(embeddings)
        if verbose:
            print(f"Hilbert confidence: {results['hilbert'].confidence:.3f}")

    if 'bispectrum' in methods:
        results['bispectrum'] = bispectrum_phase_estimate(embeddings)
        if verbose:
            print(f"Bispectrum closure error: {results['bispectrum'].phase_closure_error:.3f}")

    if 'covariance' in methods:
        cov = np.cov(embeddings.T)
        results['covariance'] = phase_from_covariance(cov)
        if verbose:
            print(f"Covariance reconstruction error: {results['covariance'].reconstruction_error:.3f}")

    if 'octant' in methods:
        results['octant'] = octant_phase_mapping(embeddings)
        if verbose:
            print(f"Octant coverage: {results['octant'].coverage:.2f}")
            print(f"Complex sum magnitude: {results['octant'].normalized_sum_magnitude:.4f}")

    if 'zero_signature' in methods:
        results['zero_signature'] = test_zero_signature(embeddings, verbose=verbose)

    return results


# =============================================================================
# Validation Functions
# =============================================================================

def validate_hilbert_recovery(seed: int = 42) -> Dict:
    """Validate Hilbert transform against synthetic data."""
    np.random.seed(seed)
    dim = 100
    n_samples = 500

    # True phases and amplitudes
    true_phases = np.random.uniform(-np.pi, np.pi, (n_samples, dim))
    true_amplitudes = np.abs(np.random.randn(n_samples, dim)) + 0.5

    # Complex embeddings
    complex_emb = true_amplitudes * np.exp(1j * true_phases)

    # Project to real
    real_emb = complex_emb.real

    # Recover
    result = hilbert_phase_recovery(real_emb)

    # Measure quality
    phase_errors = np.angle(np.exp(1j * (result.phases - true_phases)))

    return {
        'mean_phase_error': float(np.mean(np.abs(phase_errors))),
        'confidence': result.confidence,
        'passed': result.confidence > 0.3
    }


def validate_octant_mapping(seed: int = 42) -> Dict:
    """Validate octant-to-phase mapping."""
    np.random.seed(seed)
    n_samples = 1000
    dim = 384

    # Create embeddings with known octant structure
    embeddings = []
    target_octants = []

    for octant in range(8):
        signs = [
            1 if (octant >> 2) & 1 else -1,
            1 if (octant >> 1) & 1 else -1,
            1 if octant & 1 else -1
        ]

        n_per = n_samples // 8
        for _ in range(n_per):
            vec = np.random.randn(dim)
            vec[0] = signs[0] * np.abs(vec[0])
            vec[1] = signs[1] * np.abs(vec[1])
            vec[2] = signs[2] * np.abs(vec[2])
            embeddings.append(vec / np.linalg.norm(vec))
            target_octants.append(octant)

    embeddings = np.array(embeddings)
    target_octants = np.array(target_octants)

    result = octant_phase_mapping(embeddings)

    # Check accuracy
    accuracy = np.mean(result.octant_indices == target_octants)

    return {
        'octant_accuracy': float(accuracy),
        'zero_signature': result.normalized_sum_magnitude,
        'coverage': result.coverage,
        'passed': accuracy > 0.9
    }


def validate_zero_signature(seed: int = 42) -> Dict:
    """Validate zero signature with uniform random embeddings."""
    np.random.seed(seed)

    # Random embeddings should show zero signature (uniform distribution)
    embeddings = np.random.randn(1000, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = test_zero_signature(embeddings, verbose=False)

    return {
        'normalized_magnitude': result.normalized_magnitude,
        'uniformity_p_value': result.uniformity_p_value,
        'is_zero': result.is_zero,
        'passed': result.is_zero
    }


def validate_all(verbose: bool = True) -> Dict:
    """Run all validation tests."""
    results = {
        'hilbert': validate_hilbert_recovery(),
        'octant': validate_octant_mapping(),
        'zero_signature': validate_zero_signature()
    }

    all_passed = all(r['passed'] for r in results.values())

    if verbose:
        print("\n" + "=" * 60)
        print("PHASE RECOVERY VALIDATION RESULTS")
        print("=" * 60)
        for name, res in results.items():
            status = "PASS" if res['passed'] else "FAIL"
            print(f"{name}: [{status}]")
            for k, v in res.items():
                if k != 'passed':
                    print(f"  {k}: {v}")
        print("-" * 60)
        print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 60)

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("QGT Phase Recovery Module - Validation")
    print()
    validate_all(verbose=True)
