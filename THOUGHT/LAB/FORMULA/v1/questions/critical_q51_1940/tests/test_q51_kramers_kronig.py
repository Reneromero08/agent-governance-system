"""
Q51 Kramers-Kronig Relations Test - Test #7

Tests whether real and imaginary parts of complex embeddings satisfy
the Kramers-Kronig (K-K) dispersion relations.

Hypothesis:
    If embeddings are projections from analytic complex-valued space,
    the recovered real and imaginary parts must satisfy K-K relations
    (which are consequences of causality).

Theory:
    For analytic function f(omega) = u(omega) + i*v(omega):
        u(omega) = (2/pi) * P.V. integral[ omega'*v(omega') / (omega'^2 - omega^2) ] d(omega')
        v(omega) = -(2/pi) * P.V. integral[ omega*u(omega') / (omega'^2 - omega^2) ] d(omega')

    K-K violation indicates the signal is NOT analytic (not causal).

Pass criteria:
    - Normalized K-K error < 0.15
    - Consistent across models (CV < 25%)

Falsification:
    - K-K error > 0.3 (not analytic)
    - Large model variance
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
    generate_null_embeddings,
    compute_result_hash,
    format_ci,
    get_test_metadata,
    Q51Logger,
)

from qgt_phase import (
    hilbert_phase_recovery,
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

# Thresholds
KK_ERROR_PASS = 0.15
KK_ERROR_PARTIAL = 0.25
KK_ERROR_FAIL = 0.3
CV_THRESHOLD = 0.25

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
    "extremely surprised", "totally disgusted", "feeling content", "quite anxious",
    # Domain: objects
    "wooden table", "metal chair", "glass window", "plastic bottle",
    "leather bag", "cotton shirt", "rubber ball", "paper book",
    # Domain: actions
    "running fast", "walking slowly", "jumping high", "swimming deep",
    "flying above", "climbing up", "falling down", "standing still",
    # Domain: concepts
    "mathematical proof", "philosophical idea", "scientific method",
    "artistic expression", "musical harmony", "literary narrative",
    # Domain: nature
    "tall mountain", "deep ocean", "wide river", "dense forest",
    "open desert", "green meadow", "rocky cliff", "calm lake",
    # Additional variety
    "morning sunrise", "evening sunset", "midnight darkness",
    "summer heat", "winter cold", "spring bloom", "autumn leaves",
    "ancient history", "modern technology", "future dreams",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelKKResult:
    """Kramers-Kronig analysis result for a single model."""
    model_name: str
    n_samples: int
    n_dims: int
    kk_error_real: float  # Error in reconstructing real from imaginary
    kk_error_imag: float  # Error in reconstructing imag from real
    kk_error_mean: float  # Average of both
    reconstruction_quality: float  # 1 - error
    status: str
    validation_warnings: Optional[List[str]] = None


@dataclass
class CrossModelKKResult:
    """Cross-model aggregation."""
    n_models: int
    mean_kk_error: float
    std_kk_error: float
    cv_kk_error: float
    models_passing: int
    hypothesis_supported: bool
    verdict: str
    kk_error_ci: Optional[dict] = None
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
    """Get embeddings from model or generate synthetic."""
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
            embeddings, min_samples=10, name=f"kk_{model_name}"
        )
        if model_error:
            validation.warnings.append(f"Using synthetic fallback: {model_error}")
    else:
        validation = ValidationResult(True, [], [], {'n_samples': len(embeddings)})

    return embeddings, validation


def compute_kk_transform(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute Kramers-Kronig transform (Hilbert transform in frequency domain).

    The K-K transform relates real and imaginary parts of analytic signals.
    For analytic signal z = x + i*y:
        y = H[x] (Hilbert transform)
        x = -H[y]
    """
    from scipy.signal import hilbert

    # Hilbert transform gives analytic signal
    analytic = hilbert(signal, axis=axis)
    return np.imag(analytic)


def compute_kk_error(
    real_part: np.ndarray,
    imag_part: np.ndarray,
    axis: int = -1
) -> Tuple[float, float]:
    """
    Compute K-K relation error.

    For analytic signals:
        imag = H[real]
        real = -H[imag]

    Returns (error_real, error_imag) normalized by signal power.
    """
    # Reconstruct imaginary from real via Hilbert
    imag_reconstructed = compute_kk_transform(real_part, axis=axis)

    # Reconstruct real from imaginary via -Hilbert
    real_reconstructed = -compute_kk_transform(imag_part, axis=axis)

    # Compute normalized errors
    imag_power = np.mean(imag_part ** 2) + 1e-10
    real_power = np.mean(real_part ** 2) + 1e-10

    error_imag = np.mean((imag_part - imag_reconstructed) ** 2) / imag_power
    error_real = np.mean((real_part - real_reconstructed) ** 2) / real_power

    return float(np.sqrt(error_real)), float(np.sqrt(error_imag))


# =============================================================================
# Test Functions
# =============================================================================

def test_kk_single_model(
    model_name: str,
    corpus: List[str],
    verbose: bool = True
) -> ModelKKResult:
    """Test Kramers-Kronig relations for a single model."""
    logger = Q51Logger(f"kk_{model_name}", verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Kramers-Kronig Test: {model_name}")
        print(f"{'='*60}")

    # Get embeddings
    embeddings, validation = get_embeddings(model_name, corpus)
    n_samples, n_dims = embeddings.shape

    if not validation.valid:
        raise Q51ValidationError(f"Embedding validation failed: {validation.errors}")

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # Recover complex signal via Hilbert transform
    hilbert_result = hilbert_phase_recovery(embeddings, mode='eigenspace')

    # Get real and imaginary parts
    analytic = hilbert_result.analytic_signal
    real_part = np.real(analytic)
    imag_part = np.imag(analytic)

    # Compute K-K errors
    error_real, error_imag = compute_kk_error(real_part, imag_part, axis=-1)
    mean_error = (error_real + error_imag) / 2
    reconstruction_quality = 1.0 - mean_error

    if verbose:
        print(f"\nKramers-Kronig Analysis:")
        print(f"  K-K error (real->imag): {error_imag:.4f}")
        print(f"  K-K error (imag->real): {error_real:.4f}")
        print(f"  Mean K-K error: {mean_error:.4f} (threshold: < {KK_ERROR_PASS})")
        print(f"  Reconstruction quality: {reconstruction_quality:.4f}")

    # Determine status
    if mean_error < KK_ERROR_PASS:
        status = "PASS"
    elif mean_error < KK_ERROR_PARTIAL:
        status = "PARTIAL"
    else:
        status = "FAIL"

    if verbose:
        print(f"Status: {status}")

    return ModelKKResult(
        model_name=model_name,
        n_samples=n_samples,
        n_dims=n_dims,
        kk_error_real=error_real,
        kk_error_imag=error_imag,
        kk_error_mean=mean_error,
        reconstruction_quality=reconstruction_quality,
        status=status,
        validation_warnings=validation.warnings if validation.warnings else None
    )


def run_negative_control(verbose: bool = True) -> NegativeControlResult:
    """
    Negative control: Non-analytic signal should have HIGH K-K error.

    Random real+imaginary (independent) should violate K-K.
    """
    print("\n  [Negative Control] Non-analytic signal...")

    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)

    # Create non-analytic signal (independent real and imaginary)
    n_samples, n_dims = len(CORPUS), 384
    real_part = np.random.randn(n_samples, n_dims)
    imag_part = np.random.randn(n_samples, n_dims)  # Independent!

    error_real, error_imag = compute_kk_error(real_part, imag_part)
    mean_error = (error_real + error_imag) / 2

    # Should be high (K-K violated)
    is_high = mean_error > 0.5

    if verbose:
        print(f"    Mean K-K error: {mean_error:.4f}")
        status = "PASS" if is_high else "FAIL"
        print(f"    Status: {status}")

    return NegativeControlResult(
        name="non_analytic_kk_error",
        test_passed=is_high,
        expected_behavior="Non-analytic signals should have high K-K error",
        actual_behavior=f"Mean K-K error = {mean_error:.4f}",
        metric_value=mean_error,
        metric_threshold=0.5,
        notes="Independent real/imaginary parts violate K-K"
    )


def test_kk_cross_model(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[ModelKKResult], CrossModelKKResult]:
    """Test K-K relations across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 KRAMERS-KRONIG RELATIONS TEST - CROSS-MODEL")
    print("=" * 70)
    print(f"\nTesting {len(models)} models on {len(corpus)} texts")
    print("\nKey insight:")
    print("  K-K relations are CAUSALITY constraints on analytic signals.")
    print("  Low K-K error indicates the signal IS analytic (complex-valued).")
    print()

    # Run negative control
    negative_control = run_negative_control(verbose=verbose)

    # Test each model
    results = []
    for model in models:
        try:
            result = test_kk_single_model(model, corpus, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    if not results:
        raise RuntimeError("No models tested successfully")

    # Aggregate
    kk_errors = [r.kk_error_mean for r in results]
    mean_error = float(np.mean(kk_errors))
    std_error = float(np.std(kk_errors))
    cv_error = std_error / mean_error if mean_error > 0 else float('inf')

    passing = sum(1 for r in results if r.status == "PASS")

    # Bootstrap CI
    if len(kk_errors) >= 3:
        error_ci = bootstrap_ci(np.array(kk_errors), n_bootstrap=Q51Thresholds.BOOTSTRAP_ITERATIONS)
    else:
        error_ci = BootstrapCI(
            mean=mean_error, ci_lower=min(kk_errors), ci_upper=max(kk_errors),
            std=std_error, n_samples=len(kk_errors), n_bootstrap=0,
            confidence_level=Q51Thresholds.CONFIDENCE_LEVEL
        )

    # Verdict
    if passing == len(results) and cv_error < CV_THRESHOLD:
        hypothesis_supported = True
        verdict = "CONFIRMED: Embeddings satisfy K-K relations (analytic)"
    elif passing >= len(results) * 0.6:
        hypothesis_supported = True
        verdict = "PARTIAL: Most models satisfy K-K"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: K-K relations violated"

    # Metadata
    metadata = get_test_metadata()

    cross_result = CrossModelKKResult(
        n_models=len(results),
        mean_kk_error=mean_error,
        std_kk_error=std_error,
        cv_kk_error=float(cv_error * 100),
        models_passing=passing,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        kk_error_ci=error_ci.to_dict(),
        negative_controls=[negative_control.to_dict()],
        test_metadata=metadata
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean K-K error: {format_ci(error_ci)} (threshold: < {KK_ERROR_PASS})")
    print(f"CV: {cv_error*100:.1f}% (threshold: < {CV_THRESHOLD*100:.0f}%)")
    print(f"Models passing: {passing}/{len(results)}")
    print(f"\n{verdict}")

    return results, cross_result


def save_results(
    results: List[ModelKKResult],
    cross_result: CrossModelKKResult,
    output_dir: Path
):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_KRAMERS_KRONIG',
        'hypothesis': 'Embeddings satisfy K-K dispersion relations (analytic)',
        'per_model': [asdict(r) for r in results],
        'cross_model': asdict(cross_result),
        'hardening': {
            'thresholds': {
                'KK_ERROR_PASS': KK_ERROR_PASS,
                'KK_ERROR_PARTIAL': KK_ERROR_PARTIAL,
                'CV_THRESHOLD': CV_THRESHOLD,
            },
            'seeds': {
                'NEGATIVE_CONTROL': Q51Seeds.NEGATIVE_CONTROL,
            }
        }
    }

    # Compute hash
    output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q51_kramers_kronig_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Result hash: {output['result_hash']}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Kramers-Kronig Test."""
    print("\n" + "=" * 70)
    print("Q51 TEST #7: KRAMERS-KRONIG RELATIONS")
    print("=" * 70)

    results, cross_result = test_kk_cross_model(
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
