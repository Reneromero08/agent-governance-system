"""
Q51 Phase Stability Test - Test #5

Tests whether recovered phases are stable under noise perturbations.

CRITICAL LIMITATION (v5):
    This test measures PCA projection stability, which is the same for
    structured and random data. The negative control shows random data
    has the same 58x error ratio as real data. Therefore this test
    CANNOT distinguish structural phases from random artifacts.

    The test is kept for documentation but should not be used as
    evidence for phase structure. Use Tests 6 (Method Consistency)
    and 9 (Semantic Coherence) instead - they use external information
    (multiple methods, semantic labels) to distinguish structure from random.

Original Hypothesis (now known to be untestable with this method):
    If phases are structural (not artifacts), they should be stable under small
    perturbations. Phase error should increase with noise level, following a
    predictable power law, NOT be flat (which would indicate random artifacts).

Why the test fails:
    PCA-based phase recovery imposes a coordinate system on ANY data.
    In this coordinate system, both structured and random data have "phases".
    The stability of these phases under noise is a property of PCA,
    not of the underlying data structure.

Method:
    1. Get embeddings, recover phases via PCA 2D projection
    2. Add Gaussian noise at SNR levels: 40dB, 30dB, 20dB, 10dB, 6dB
    3. Recover phases from noisy embeddings using SAME coordinate system
    4. Measure phase error vs SNR
    5. Fit power law: error ~ 10^(-SNR/alpha)

Pass criteria:
    - Negative control must pass (ratio < 2x for random data)
    - Phase error < 0.3 rad at SNR 20dB
    - Power law R^2 > 0.8
    - Decay trend is "increasing"

Current status: INCONCLUSIVE (negative control fails)
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add paths
SCRIPT_DIR = Path(__file__).parent
QGT_PATH = SCRIPT_DIR.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(QGT_PATH))

# Import test harness
from q51_test_harness import (
    Q51Thresholds,
    Q51Seeds,
    Q51ValidationError,
    ValidationResult,
    BootstrapCI,
    NegativeControlResult,
    validate_embeddings,
    bootstrap_ci,
    cohens_d,
    generate_null_embeddings,
    generate_structured_null,
    compute_result_hash,
    format_ci,
    get_test_metadata,
    Q51Logger,
)

from qgt_phase import (
    octant_phase_mapping,
    circular_correlation,
    circular_variance,
)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

# =============================================================================
# Constants
# =============================================================================

# SNR levels to test (in dB)
SNR_LEVELS = [40, 30, 20, 10, 6]

# Thresholds for phase stability
PHASE_ERROR_AT_20DB_PASS = 0.3  # Max error at 20dB SNR
PHASE_ERROR_AT_20DB_PARTIAL = 0.5
POWER_LAW_ALPHA_PASS = 15.0  # dB per decade of error increase
POWER_LAW_ALPHA_PARTIAL = 8.0

# Models to test
MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-small",
]

# Test corpus
CORPUS = [
    # Domain: emotions
    "feeling happy today", "deeply sad", "very angry", "quite fearful",
    "extremely surprised", "totally disgusted", "feeling content",
    # Domain: objects
    "wooden table", "metal chair", "glass window", "plastic bottle",
    "leather bag", "cotton shirt", "rubber ball",
    # Domain: actions
    "running fast", "walking slowly", "jumping high", "swimming deep",
    "flying above", "climbing up", "falling down",
    # Domain: concepts
    "mathematical proof", "philosophical idea", "scientific method",
    "artistic expression", "musical harmony", "literary narrative",
    # Domain: nature
    "tall mountain", "deep ocean", "wide river", "dense forest",
    "open desert", "green meadow", "rocky cliff",
    # Additional variety
    "morning sunrise", "evening sunset", "midnight darkness",
    "summer heat", "winter cold", "spring bloom", "autumn leaves",
    "ancient history", "modern technology", "future dreams",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SNRTestResult:
    """Result for a single SNR level."""
    snr_db: float
    mean_phase_error: float
    std_phase_error: float
    octant_flip_rate: float  # Fraction of octants that changed
    n_trials: int


@dataclass
class ModelStabilityResult:
    """Phase stability result for a single model."""
    model_name: str
    n_samples: int
    snr_results: List[Dict]
    error_at_20db: float
    power_law_alpha: float  # dB per decade
    power_law_r2: float  # Goodness of fit
    decay_trend: str  # 'decreasing', 'flat', 'increasing'
    status: str
    validation_warnings: Optional[List[str]] = None


@dataclass
class CrossModelStabilityResult:
    """Cross-model aggregation."""
    n_models: int
    mean_error_at_20db: float
    std_error_at_20db: float
    mean_power_law_alpha: float
    models_passing: int
    hypothesis_supported: bool
    verdict: str
    error_20db_ci: Optional[dict] = None
    negative_controls: Optional[List[dict]] = None
    test_metadata: Optional[dict] = None
    result_hash: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_embeddings(
    model_name: str,
    texts: List[str],
    validate: bool = True
) -> Tuple[np.ndarray, ValidationResult]:
    """Get embeddings from model or generate synthetic with validation."""
    embeddings = None
    model_error = None

    if HAS_ST:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings)
        except Exception as e:
            model_error = str(e)
            print(f"  Warning: Could not load {model_name}: {e}")

    # Synthetic fallback
    if embeddings is None:
        np.random.seed(hash(model_name) % 2**32)
        dim = 384
        n = len(texts)
        rank = 22
        components = np.random.randn(rank, dim)
        weights = np.random.randn(n, rank)
        embeddings = weights @ components
        embeddings += 0.1 * np.random.randn(n, dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        embeddings = embeddings / norms

    # Validate
    if validate:
        validation = validate_embeddings(
            embeddings, min_samples=10, name=f"stability_{model_name}"
        )
        if model_error:
            validation.warnings.append(f"Using synthetic fallback: {model_error}")
    else:
        validation = ValidationResult(True, [], [], {'n_samples': len(embeddings)})

    return embeddings, validation


def add_noise_at_snr(embeddings: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise to embeddings at specified SNR (in dB).

    SNR = 10 * log10(signal_power / noise_power)
    noise_power = signal_power / 10^(SNR/10)
    """
    if seed is not None:
        np.random.seed(seed)

    # Signal power (per sample, average across dimensions)
    signal_power = np.mean(embeddings ** 2)

    # Required noise power
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate noise with required power
    noise_std = np.sqrt(noise_power)
    noise = np.random.randn(*embeddings.shape) * noise_std

    # Add noise
    noisy = embeddings + noise

    # Re-normalize to unit sphere
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    noisy = noisy / norms

    return noisy


def compute_phase_error(phases_clean: np.ndarray, phases_noisy: np.ndarray) -> float:
    """Compute mean circular distance between phase vectors."""
    # Wrap phase difference to [-pi, pi]
    diff = phases_noisy - phases_clean
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.mean(np.abs(diff)))


def compute_octant_flip_rate(octants_clean: np.ndarray, octants_noisy: np.ndarray) -> float:
    """Compute fraction of samples that changed octant."""
    return float(np.mean(octants_clean != octants_noisy))


def fit_power_law(snr_levels: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Fit power law: error = A * 10^(-SNR/alpha)

    In log space: log10(error) = log10(A) - SNR/alpha

    Returns (alpha, r2)
    """
    # Filter out zeros or negatives
    mask = errors > 1e-10
    if np.sum(mask) < 2:
        return 0.0, 0.0

    x = snr_levels[mask]
    y = np.log10(errors[mask])

    # Linear fit: y = b + m*x, where m = -1/alpha
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))

    if ss_xx < 1e-10:
        return 0.0, 0.0

    m = ss_xy / ss_xx
    b = y_mean - m * x_mean

    # Alpha: m = -1/alpha => alpha = -1/m
    if abs(m) < 1e-10:
        alpha = float('inf')
    else:
        alpha = -1.0 / m

    # R-squared
    y_pred = b + m * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(alpha), float(r2)


# =============================================================================
# Test Functions
# =============================================================================

def get_continuous_phases(embeddings: np.ndarray) -> np.ndarray:
    """
    Extract CONTINUOUS phases via 2D projection.

    Unlike octant mapping (discrete 8 values), this gives smooth phases
    that can properly measure stability under noise.
    """
    # Center the data
    centered = embeddings - embeddings.mean(axis=0)

    # Get top 2 principal components
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:2]

    # Project to 2D
    proj_2d = centered @ eigenvectors[:, idx]

    # Continuous phase from 2D projection
    phases = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])

    return phases


def test_stability_single_snr(
    embeddings: np.ndarray,
    snr_db: float,
    n_trials: int = 10,
    fixed_eigenvectors: np.ndarray = None,
    clean_mean: np.ndarray = None
) -> SNRTestResult:
    """Test phase stability at a single SNR level."""
    phase_errors = []
    flip_rates = []

    # CRITICAL FIX: Use clean data mean for ALL centering operations
    # Previous bug: noisy data was centered using its own mean, which
    # reintroduces data-dependent transformation.
    if clean_mean is None:
        clean_mean = embeddings.mean(axis=0)

    # Compute FIXED coordinate system from clean data
    # This is crucial: if we recompute eigenvectors from noisy data,
    # the coordinate system rotates and we measure coordinate rotation,
    # not phase instability!
    if fixed_eigenvectors is None:
        centered = embeddings - clean_mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:2]
        fixed_eigenvectors = eigenvectors[:, idx]

    # Get clean phases using fixed coordinates and CLEAN MEAN
    centered_clean = embeddings - clean_mean
    proj_clean = centered_clean @ fixed_eigenvectors
    clean_phases = np.arctan2(proj_clean[:, 1], proj_clean[:, 0])

    # Also get octant for flip rate comparison
    clean_result = octant_phase_mapping(embeddings)
    clean_octants = clean_result.octant_indices

    for trial in range(n_trials):
        # Add noise with different seed each trial
        noisy = add_noise_at_snr(embeddings, snr_db, seed=Q51Seeds.NEGATIVE_CONTROL + trial)

        # Get noisy phases using SAME fixed coordinates AND CLEAN MEAN
        # CRITICAL: Must use clean_mean, not noisy.mean()!
        centered_noisy = noisy - clean_mean
        proj_noisy = centered_noisy @ fixed_eigenvectors
        noisy_phases = np.arctan2(proj_noisy[:, 1], proj_noisy[:, 0])

        # Also get octant for flip rate
        noisy_result = octant_phase_mapping(noisy)
        noisy_octants = noisy_result.octant_indices

        # Compute errors - now comparing phases in SAME coordinate system
        phase_error = compute_phase_error(clean_phases, noisy_phases)
        flip_rate = compute_octant_flip_rate(clean_octants, noisy_octants)

        phase_errors.append(phase_error)
        flip_rates.append(flip_rate)

    return SNRTestResult(
        snr_db=float(snr_db),
        mean_phase_error=float(np.mean(phase_errors)),
        std_phase_error=float(np.std(phase_errors)),
        octant_flip_rate=float(np.mean(flip_rates)),
        n_trials=n_trials
    )


def test_stability_single_model(
    model_name: str,
    corpus: List[str],
    snr_levels: List[float] = SNR_LEVELS,
    n_trials: int = 10,
    verbose: bool = True
) -> ModelStabilityResult:
    """Test phase stability for a single model across SNR levels."""
    logger = Q51Logger(f"stability_{model_name}", verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase Stability Test: {model_name}")
        print(f"{'='*60}")

    # Get embeddings
    embeddings, validation = get_embeddings(model_name, corpus)
    n_samples = len(embeddings)

    if not validation.valid:
        raise Q51ValidationError(f"Embedding validation failed: {validation.errors}")

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # Compute FIXED eigenvectors and CLEAN MEAN from clean data
    # These will be used for all SNR levels to ensure consistent coordinate system
    clean_mean = embeddings.mean(axis=0)
    centered = embeddings - clean_mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:2]
    fixed_eigenvectors = eigenvectors[:, idx]

    # Test each SNR level
    snr_results = []
    for snr_db in snr_levels:
        result = test_stability_single_snr(embeddings, snr_db, n_trials, fixed_eigenvectors, clean_mean)
        snr_results.append(asdict(result))

        if verbose:
            print(f"  SNR {snr_db:2d}dB: error={result.mean_phase_error:.4f} +/- {result.std_phase_error:.4f}, flip={result.octant_flip_rate:.2%}")

    # Extract metrics
    snr_arr = np.array(snr_levels)
    error_arr = np.array([r['mean_phase_error'] for r in snr_results])

    # Find error at 20dB (or interpolate)
    if 20 in snr_levels:
        idx_20 = snr_levels.index(20)
        error_at_20db = error_arr[idx_20]
    else:
        # Interpolate
        error_at_20db = float(np.interp(20, snr_arr[::-1], error_arr[::-1]))

    # Fit power law
    alpha, r2 = fit_power_law(snr_arr, error_arr)

    # Determine trend
    if error_arr[-1] < error_arr[0] * 0.5:
        decay_trend = 'decreasing'  # Error decreases with more noise (bad)
    elif error_arr[-1] > error_arr[0] * 2.0:
        decay_trend = 'increasing'  # Error increases with more noise (good)
    else:
        decay_trend = 'flat'

    # Determine status
    # Key insight: Error should INCREASE with noise (lower SNR)
    # A flat curve means phases are random artifacts
    # CRITICAL FIX: Also check R^2 of power law fit (previously unchecked!)
    R2_THRESHOLD = 0.8  # Power law fit quality threshold
    if (decay_trend == 'increasing' and
        error_at_20db < PHASE_ERROR_AT_20DB_PASS and
        r2 > R2_THRESHOLD):
        status = "PASS"
    elif (decay_trend != 'decreasing' and
          error_at_20db < PHASE_ERROR_AT_20DB_PARTIAL and
          r2 > 0.5):
        status = "PARTIAL"
    else:
        status = "FAIL"

    if verbose:
        print(f"\nError at 20dB: {error_at_20db:.4f} (threshold: < {PHASE_ERROR_AT_20DB_PASS})")
        print(f"Power law R^2: {r2:.3f} (threshold: > {R2_THRESHOLD})")
        print(f"Trend: {decay_trend}")
        print(f"Status: {status}")

    return ModelStabilityResult(
        model_name=model_name,
        n_samples=n_samples,
        snr_results=snr_results,
        error_at_20db=float(error_at_20db),
        power_law_alpha=float(alpha),
        power_law_r2=float(r2),
        decay_trend=decay_trend,
        status=status,
        validation_warnings=validation.warnings if validation.warnings else None
    )


def run_negative_control(verbose: bool = True) -> NegativeControlResult:
    """
    Negative control: Random embeddings should show FLAT error curve.

    Random phases have no structure, so noise doesn't change them systematically.

    CRITICAL FIX (v5): Use RATIO metric, not absolute range.
    - Real data shows 50x error increase (0.002 -> 0.1)
    - Random data should show <2x change (stays flat)
    - Previous absolute threshold (0.1) was catching BOTH because:
      - Real: range = 0.098 (from 0.002 to 0.1)
      - Random: range = 0.1 (from 0.9 to 1.0)
    - Ratio clearly distinguishes them: 50x vs 1.1x
    """
    print("\n  [Negative Control] Random embeddings...")

    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)
    random_emb = generate_null_embeddings(len(CORPUS), 384, seed=Q51Seeds.NEGATIVE_CONTROL)

    errors = []
    for snr_db in [40, 20, 6]:
        result = test_stability_single_snr(random_emb, snr_db, n_trials=5)
        errors.append(result.mean_phase_error)

    # FIXED: Use RATIO metric instead of absolute range
    # This aligns with the main test's trend detection logic:
    # Main test uses error[-1] > error[0] * 2.0 for "increasing" trend
    # Negative control should verify random data does NOT show this
    min_error = min(errors) + 1e-10  # Avoid division by zero
    max_error = max(errors)
    error_ratio = max_error / min_error

    # Random data should show ratio < 2.0 (flat curve)
    # Real data shows ratio > 50 (increasing curve)
    RATIO_THRESHOLD = 2.0
    is_flat = error_ratio < RATIO_THRESHOLD

    if verbose:
        print(f"    Error at 40dB: {errors[0]:.4f}")
        print(f"    Error at 20dB: {errors[1]:.4f}")
        print(f"    Error at 6dB: {errors[2]:.4f}")
        print(f"    Error ratio (max/min): {error_ratio:.2f}x (flat if < {RATIO_THRESHOLD}x)")
        status = "PASS" if is_flat else "FAIL"
        print(f"    Status: {status}")

    return NegativeControlResult(
        name="random_embeddings_flat",
        test_passed=is_flat,
        expected_behavior=f"Error ratio should be < {RATIO_THRESHOLD}x (no structure to preserve)",
        actual_behavior=f"Error ratio = {error_ratio:.2f}x",
        metric_value=error_ratio,
        metric_threshold=RATIO_THRESHOLD,
        notes="Random phases have no structure, so noise shouldn't systematically increase error"
    )


def test_stability_cross_model(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[ModelStabilityResult], CrossModelStabilityResult]:
    """Test phase stability across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 PHASE STABILITY TEST - CROSS-MODEL VALIDATION")
    print("=" * 70)
    print(f"\nTesting {len(models)} models on {len(corpus)} texts")
    print(f"SNR levels: {SNR_LEVELS} dB")
    print("\nKey insight:")
    print("  If phases are STRUCTURAL, error should INCREASE with noise.")
    print("  If phases are ARTIFACTS, error curve will be FLAT.")
    print()

    # Run negative control
    negative_control = run_negative_control(verbose=verbose)

    # Test each model
    results = []
    for model in models:
        try:
            result = test_stability_single_model(model, corpus, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    if not results:
        raise RuntimeError("No models tested successfully")

    # Aggregate
    errors_20db = [r.error_at_20db for r in results]
    alphas = [r.power_law_alpha for r in results]

    mean_error = float(np.mean(errors_20db))
    std_error = float(np.std(errors_20db))
    mean_alpha = float(np.mean([a for a in alphas if not np.isinf(a)]))

    passing = sum(1 for r in results if r.status == "PASS")

    # Bootstrap CI for error at 20dB
    if len(errors_20db) >= 3:
        error_ci = bootstrap_ci(np.array(errors_20db), n_bootstrap=Q51Thresholds.BOOTSTRAP_ITERATIONS)
    else:
        error_ci = BootstrapCI(
            mean=mean_error, ci_lower=min(errors_20db), ci_upper=max(errors_20db),
            std=std_error, n_samples=len(errors_20db), n_bootstrap=0,
            confidence_level=Q51Thresholds.CONFIDENCE_LEVEL
        )

    # Verdict
    # CRITICAL: The negative control MUST pass for this test to be meaningful.
    # If random data shows the same behavior as real data, the test cannot
    # distinguish structured from random phases.
    if not negative_control.test_passed:
        hypothesis_supported = False
        verdict = ("INCONCLUSIVE: Negative control failed. Random data shows same "
                   f"error ratio ({negative_control.metric_value:.1f}x) as real data. "
                   "This test measures PCA projection stability, not phase structure.")
    elif passing == len(results):
        hypothesis_supported = True
        verdict = "CONFIRMED: Phases are stable under noise (structural, not artifacts)"
    elif passing >= len(results) * 0.6:
        hypothesis_supported = True
        verdict = "PARTIAL: Most models show stable phase structure"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: Phases appear to be noise artifacts"

    # Metadata
    metadata = get_test_metadata()

    cross_result = CrossModelStabilityResult(
        n_models=len(results),
        mean_error_at_20db=mean_error,
        std_error_at_20db=std_error,
        mean_power_law_alpha=mean_alpha,
        models_passing=passing,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        error_20db_ci=error_ci.to_dict(),
        negative_controls=[negative_control.to_dict()],
        test_metadata=metadata
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean error at 20dB: {format_ci(error_ci)} rad")
    print(f"Mean power law alpha: {mean_alpha:.1f} dB/decade")
    print(f"Models passing: {passing}/{len(results)}")
    print(f"\n{verdict}")

    return results, cross_result


def save_results(
    results: List[ModelStabilityResult],
    cross_result: CrossModelStabilityResult,
    output_dir: Path
):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_PHASE_STABILITY',
        'hypothesis': 'Phases are stable under noise (structural, not artifacts)',
        'per_model': [asdict(r) for r in results],
        'cross_model': asdict(cross_result),
        'hardening': {
            'snr_levels': SNR_LEVELS,
            'thresholds': {
                'PHASE_ERROR_AT_20DB_PASS': PHASE_ERROR_AT_20DB_PASS,
                'PHASE_ERROR_AT_20DB_PARTIAL': PHASE_ERROR_AT_20DB_PARTIAL,
                'POWER_LAW_ALPHA_PASS': POWER_LAW_ALPHA_PASS,
            },
            'seeds': {
                'NEGATIVE_CONTROL': Q51Seeds.NEGATIVE_CONTROL,
            }
        }
    }

    # Compute hash
    output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q51_phase_stability_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Result hash: {output['result_hash']}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Phase Stability Test."""
    print("\n" + "=" * 70)
    print("Q51 TEST #5: PHASE STABILITY UNDER NOISE")
    print("=" * 70)

    results, cross_result = test_stability_cross_model(
        models=MODELS,
        corpus=CORPUS,
        verbose=True
    )

    # Save results
    output_dir = SCRIPT_DIR / "results"
    save_results(results, cross_result, output_dir)

    print("\n" + "=" * 70)

    return cross_result.hypothesis_supported


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
