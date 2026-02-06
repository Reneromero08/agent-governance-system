"""
Q51 Method Consistency Test - Test #6

Tests whether TRULY INDEPENDENT phase recovery methods agree.

CRITICAL REDESIGN v2:
    v1 bug: pc23 SHARED PC2 with pc12! This created spurious correlation
    even for random data (~0.24), making the test unreliable.

    v2 fix: Use NON-OVERLAPPING PC pairs:
      - pc12_angle uses PC1, PC2
      - pc34_angle uses PC3, PC4 (completely independent!)

Hypothesis:
    If phases are real (not artifacts), phases computed from INDEPENDENT
    PC subspaces should still correlate, because both subspaces capture
    aspects of the same underlying semantic structure.

Methods Compared (v2):
    1. pc12_angle: arctan2(PC2, PC1) - angle in PC1-PC2 plane
    2. pc34_angle: arctan2(PC4, PC3) - angle in PC3-PC4 plane (NO SHARED PCs!)
    3. hilbert: Analytic signal on PC3 (independent of pc12)
    4. bispectrum: Frequency domain phase closure (different domain)

Key Test:
    pc12_angle vs pc34_angle correlation > 0.15
    These use COMPLETELY INDEPENDENT eigenvectors. For random data,
    correlation should be ~0. For structured data, expect 0.2-0.5.

Pass criteria:
    - PC12-PC34 correlation > 0.15 (truly independent planes agree)
    - PC12-Hilbert correlation > 0.15 (different algorithms agree)
    - Negative control: random data correlation < 0.15

Falsification:
    - Methods uncorrelated (< 0.15)
    - Negative control fails (random data correlates, indicating methodological flaw)
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import combinations

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
    bispectrum_phase_estimate,
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

# Thresholds
MEAN_CORR_PASS = 0.5
MEAN_CORR_PARTIAL = 0.3
MIN_PAIR_CORR = 0.2  # No pair should be below this

# Methods to compare
# CRITICAL REDESIGN v2:
# - pc12_angle vs pc34_angle: Angles in NON-OVERLAPPING PC planes (truly independent!)
#   Previous pc23 shared PC2 with pc12, creating spurious correlation.
# - hilbert: Analytic signal phase on PC3 (independent of pc12)
# - bispectrum: Frequency domain (different domain entirely)
METHODS = ['pc12_angle', 'pc34_angle', 'hilbert', 'bispectrum']

# Models to test
MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-small",
]

# Test corpus - larger for better statistics
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
    # Domain: time
    "morning sunrise", "evening sunset", "midnight darkness", "noon brightness",
    "summer heat", "winter cold", "spring bloom", "autumn leaves",
    # Domain: abstract
    "ancient history", "modern technology", "future dreams", "eternal truth",
    "infinite space", "finite time", "absolute zero", "relative motion",
    # Domain: social
    "friendly conversation", "heated argument", "quiet agreement", "loud disagreement",
    "warm greeting", "cold farewell", "joyful reunion", "sad departure",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MethodPairCorrelation:
    """Correlation between two phase recovery methods."""
    method1: str
    method2: str
    circular_correlation: float
    n_samples: int


@dataclass
class ModelConsistencyResult:
    """Method consistency result for a single model."""
    model_name: str
    n_samples: int
    pairwise_correlations: List[Dict]
    correlation_matrix: List[List[float]]  # 4x4 matrix
    mean_correlation: float
    min_correlation: float
    max_correlation: float
    methods_connected: bool  # All methods have at least one strong correlation
    status: str
    validation_warnings: Optional[List[str]] = None


@dataclass
class CrossModelConsistencyResult:
    """Cross-model aggregation."""
    n_models: int
    mean_correlation: float
    std_correlation: float
    models_passing: int
    hypothesis_supported: bool
    verdict: str
    correlation_ci: Optional[dict] = None
    method_summary: Optional[Dict] = None  # Per-method average
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
            embeddings, min_samples=10, name=f"consistency_{model_name}"
        )
        if model_error:
            validation.warnings.append(f"Using synthetic fallback: {model_error}")
    else:
        validation = ValidationResult(True, [], [], {'n_samples': len(embeddings)})

    return embeddings, validation


def recover_phases_all_methods(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Recover phases using TRULY INDEPENDENT methods.

    CRITICAL REDESIGN v2:

    Previous bug (v1): pc12 vs pc23 BOTH use PC2, so they're NOT independent!
      - pc12_angle = arctan2(PC2, PC1) - PC2 is in numerator
      - pc23_angle = arctan2(PC3, PC2) - PC2 is in denominator
      Sharing PC2 creates spurious correlation even for random data.

    FIX: Use NON-OVERLAPPING PC pairs:
      1. pc12_angle: arctan2(PC2, PC1) - uses PCs 1,2
      2. pc34_angle: arctan2(PC4, PC3) - uses PCs 3,4 (completely independent!)
      3. hilbert: Hilbert transform on PC3 (independent of pc12)
      4. bispectrum: Frequency domain (different domain)

    For real data: Both PC1-2 and PC3-4 planes capture aspects of the same
    semantic structure, so angles should correlate.

    For random data: PC1-2 and PC3-4 are completely independent eigenvector
    pairs, so correlation should be ~0.

    This is the TRUE independence test.
    """
    n_samples, dim = embeddings.shape
    phases = {}

    # Pre-compute PC projections - need top 4 now
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:4]  # Top 4 PCs
    proj_4d = centered @ eigenvectors[:, idx]
    pc1 = proj_4d[:, 0]
    pc2 = proj_4d[:, 1]
    pc3 = proj_4d[:, 2]
    pc4 = proj_4d[:, 3]

    # 1. PC1-PC2 angle (primary plane)
    phases['pc12_angle'] = np.arctan2(pc2, pc1)

    # 2. PC3-PC4 angle (secondary plane) - TRULY INDEPENDENT (no shared PCs!)
    # If phase structure is real, it should appear consistently across
    # independent PC planes. For random data, these should have ~0 correlation.
    phases['pc34_angle'] = np.arctan2(pc4, pc3)

    # 3. HILBERT: Analytic signal on PC3 (independent of pc12!)
    # Using PC3 instead of PC1 ensures this is independent of the pc12 method.
    # This gives instantaneous phase of the PC3 "signal"
    from scipy.signal import hilbert as scipy_hilbert
    analytic = pc3 + 1j * scipy_hilbert(pc3).imag
    phases['hilbert'] = np.angle(analytic)

    # 4. BISPECTRUM (FREQUENCY DOMAIN - different domain entirely)
    # We report this for completeness but acknowledge it measures different thing
    n_freq = 32
    bispectrum_result = bispectrum_phase_estimate(embeddings, n_freq=n_freq)
    recovered_freq_phases = bispectrum_result.phases  # (n_freq,)

    # FFT of each sample
    X = np.fft.fft(embeddings, n=n_freq, axis=1)  # (n_samples, n_freq)

    # Align FFT to bispectrum-recovered phases by derotating
    weights = np.exp(-np.arange(n_freq) / 5.0)  # Decay weights
    aligned = X * np.exp(-1j * recovered_freq_phases) * weights
    phases['bispectrum'] = np.angle(np.sum(aligned, axis=1))

    return phases


def compute_pairwise_correlations(phases: Dict[str, np.ndarray]) -> List[MethodPairCorrelation]:
    """Compute circular correlations between all pairs of methods."""
    results = []

    method_names = list(phases.keys())
    for m1, m2 in combinations(method_names, 2):
        p1 = phases[m1]
        p2 = phases[m2]

        # Ensure same length
        min_len = min(len(p1), len(p2))
        p1 = p1[:min_len]
        p2 = p2[:min_len]

        corr = circular_correlation(p1, p2)

        results.append(MethodPairCorrelation(
            method1=m1,
            method2=m2,
            circular_correlation=float(corr),
            n_samples=min_len
        ))

    return results


def build_correlation_matrix(correlations: List[MethodPairCorrelation], methods: List[str]) -> np.ndarray:
    """Build 4x4 correlation matrix from pairwise results."""
    n = len(methods)
    matrix = np.eye(n)  # Diagonal = 1

    for corr in correlations:
        i = methods.index(corr.method1)
        j = methods.index(corr.method2)
        matrix[i, j] = corr.circular_correlation
        matrix[j, i] = corr.circular_correlation

    return matrix


# =============================================================================
# Test Functions
# =============================================================================

def test_consistency_single_model(
    model_name: str,
    corpus: List[str],
    verbose: bool = True
) -> ModelConsistencyResult:
    """Test method consistency for a single model."""
    logger = Q51Logger(f"consistency_{model_name}", verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Method Consistency Test: {model_name}")
        print(f"{'='*60}")

    # Get embeddings
    embeddings, validation = get_embeddings(model_name, corpus)
    n_samples = len(embeddings)

    if not validation.valid:
        raise Q51ValidationError(f"Embedding validation failed: {validation.errors}")

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # Recover phases using all methods
    phases = recover_phases_all_methods(embeddings)

    if verbose:
        print(f"\nPhases recovered for methods: {list(phases.keys())}")

    # Compute pairwise correlations
    correlations = compute_pairwise_correlations(phases)

    # Build correlation matrix
    matrix = build_correlation_matrix(correlations, METHODS)

    # Summary statistics
    corr_values = [c.circular_correlation for c in correlations]
    mean_corr = float(np.mean(corr_values))
    min_corr = float(np.min(corr_values))
    max_corr = float(np.max(corr_values))

    # Check if methods are connected (each has at least one correlation > 0.3)
    methods_connected = True
    for i, method in enumerate(METHODS):
        row = matrix[i, :]
        other_corrs = [row[j] for j in range(len(METHODS)) if j != i]
        if max(other_corrs) < MIN_PAIR_CORR:
            methods_connected = False
            break

    if verbose:
        print(f"\nCorrelation matrix:")
        print("         ", "  ".join([f"{m[:7]:>7}" for m in METHODS]))
        for i, m1 in enumerate(METHODS):
            row_str = "  ".join([f"{matrix[i,j]:7.3f}" for j in range(len(METHODS))])
            print(f"{m1[:8]:<8} {row_str}")

        print(f"\nMean correlation: {mean_corr:.4f}")
        print(f"Min correlation: {min_corr:.4f}")
        print(f"Max correlation: {max_corr:.4f}")
        print(f"Methods connected: {methods_connected}")

    # Determine status
    # CRITICAL REDESIGN v2:
    #
    # v1 bug: pc23 SHARED PC2 with pc12, creating spurious correlation!
    # v2 fix: Use NON-OVERLAPPING PC pairs (pc12 vs pc34).
    #
    # Key test: pc12_angle vs pc34_angle correlation
    # - pc12 uses PC1, PC2
    # - pc34 uses PC3, PC4 (completely independent!)
    #
    # For real data: Both planes capture the same semantic structure, so
    # angles should correlate even though the planes are independent.
    #
    # For random data: PC1-2 and PC3-4 are independent eigenvector pairs,
    # so correlation should be ~0.
    #
    # This is the TRUE independence test. Hilbert now uses PC3, so it's
    # also independent of pc12.

    pc12_pc34_corr = 0.0
    pc12_hilbert_corr = 0.0
    pc34_hilbert_corr = 0.0
    for corr in correlations:
        m1, m2 = corr.method1, corr.method2
        if {m1, m2} == {'pc12_angle', 'pc34_angle'}:
            pc12_pc34_corr = corr.circular_correlation
        if {m1, m2} == {'pc12_angle', 'hilbert'}:
            pc12_hilbert_corr = corr.circular_correlation
        if {m1, m2} == {'pc34_angle', 'hilbert'}:
            pc34_hilbert_corr = corr.circular_correlation

    # Spatial methods (excluding bispectrum which is frequency domain)
    spatial_corrs = []
    for corr in correlations:
        if 'bispectrum' not in [corr.method1, corr.method2]:
            spatial_corrs.append(corr.circular_correlation)
    mean_spatial = float(np.mean(spatial_corrs)) if spatial_corrs else 0.0

    # Pass criterion: pc12-pc34 > 0.15 (truly independent planes show consistency)
    # Note: Threshold lowered from 0.2 because these are TRULY independent.
    # For random data, expected correlation is ~0.
    # For real data with structure, expect 0.2-0.5.
    PC12_PC34_THRESH = 0.15  # Truly independent - even weak correlation is meaningful
    PC12_HILBERT_THRESH = 0.15  # Also independent now (hilbert uses PC3)

    if pc12_pc34_corr > PC12_PC34_THRESH and pc12_hilbert_corr > PC12_HILBERT_THRESH:
        status = "PASS"  # Independent methods agree - phase structure is real
    elif pc12_pc34_corr > PC12_PC34_THRESH or pc12_hilbert_corr > PC12_HILBERT_THRESH:
        status = "PARTIAL"  # Some agreement
    else:
        status = "FAIL"

    if verbose:
        print(f"\nPC12-PC34 correlation: {pc12_pc34_corr:.4f} (threshold: > {PC12_PC34_THRESH})")
        print(f"  (TRULY INDEPENDENT: no shared PCs)")
        print(f"PC12-Hilbert correlation: {pc12_hilbert_corr:.4f} (threshold: > {PC12_HILBERT_THRESH})")
        print(f"  (Hilbert now uses PC3, independent of pc12)")
        print(f"PC34-Hilbert correlation: {pc34_hilbert_corr:.4f}")
        print(f"Mean spatial correlation: {mean_spatial:.4f}")
        print(f"Status: {status}")

    return ModelConsistencyResult(
        model_name=model_name,
        n_samples=n_samples,
        pairwise_correlations=[asdict(c) for c in correlations],
        correlation_matrix=matrix.tolist(),
        mean_correlation=mean_corr,
        min_correlation=min_corr,
        max_correlation=max_corr,
        methods_connected=methods_connected,
        status=status,
        validation_warnings=validation.warnings if validation.warnings else None
    )


def run_negative_control(verbose: bool = True) -> NegativeControlResult:
    """
    Negative control: Random embeddings should show LOW correlation between
    TRULY INDEPENDENT methods (PC12 vs PC34).

    v2 update: Now that we use non-overlapping PC pairs, random data should
    show near-zero correlation. The previous version with overlapping PCs
    (pc12 vs pc23 sharing PC2) showed ~0.24 spurious correlation.
    """
    print("\n  [Negative Control] Random embeddings...")

    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)
    random_emb = generate_null_embeddings(len(CORPUS), 384, seed=Q51Seeds.NEGATIVE_CONTROL)

    phases = recover_phases_all_methods(random_emb)
    correlations = compute_pairwise_correlations(phases)

    # Find the KEY correlation: PC12 vs PC34 (truly independent)
    pc12_pc34_corr = 0.0
    for c in correlations:
        if {c.method1, c.method2} == {'pc12_angle', 'pc34_angle'}:
            pc12_pc34_corr = c.circular_correlation
            break

    corr_values = [c.circular_correlation for c in correlations]
    mean_corr = float(np.mean(corr_values))

    # For truly independent methods on random data, correlation should be ~0
    # Threshold: 0.15 (same as pass criteria)
    RANDOM_THRESH = 0.15
    is_low = abs(pc12_pc34_corr) < RANDOM_THRESH

    if verbose:
        print(f"    PC12-PC34 correlation: {pc12_pc34_corr:.4f} (threshold: < {RANDOM_THRESH})")
        print(f"    (Truly independent PC pairs should have ~0 correlation for random data)")
        print(f"    Mean all-pairs correlation: {mean_corr:.4f}")
        status = "PASS" if is_low else "FAIL"
        print(f"    Status: {status}")

    return NegativeControlResult(
        name="random_embeddings_independent_pcs",
        test_passed=is_low,
        expected_behavior=f"PC12-PC34 correlation should be < {RANDOM_THRESH} for random data",
        actual_behavior=f"PC12-PC34 correlation = {pc12_pc34_corr:.4f}",
        metric_value=pc12_pc34_corr,
        metric_threshold=RANDOM_THRESH,
        notes="Truly independent PC pairs (no shared components) should have ~0 correlation for random data"
    )


def test_consistency_cross_model(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[ModelConsistencyResult], CrossModelConsistencyResult]:
    """Test method consistency across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 METHOD CONSISTENCY TEST - CROSS-MODEL VALIDATION")
    print("=" * 70)
    print(f"\nTesting {len(models)} models on {len(corpus)} texts")
    print(f"Methods compared: {METHODS}")
    print("\nKey insight:")
    print("  If phases are STRUCTURAL, different methods should AGREE.")
    print("  If phases are ARTIFACTS, methods will be UNCORRELATED.")
    print()

    # Run negative control
    negative_control = run_negative_control(verbose=verbose)

    # Test each model
    results = []
    for model in models:
        try:
            result = test_consistency_single_model(model, corpus, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    if not results:
        raise RuntimeError("No models tested successfully")

    # Aggregate
    mean_corrs = [r.mean_correlation for r in results]
    mean_corr = float(np.mean(mean_corrs))
    std_corr = float(np.std(mean_corrs))

    passing = sum(1 for r in results if r.status == "PASS")

    # Bootstrap CI
    if len(mean_corrs) >= 3:
        corr_ci = bootstrap_ci(np.array(mean_corrs), n_bootstrap=Q51Thresholds.BOOTSTRAP_ITERATIONS)
    else:
        corr_ci = BootstrapCI(
            mean=mean_corr, ci_lower=min(mean_corrs), ci_upper=max(mean_corrs),
            std=std_corr, n_samples=len(mean_corrs), n_bootstrap=0,
            confidence_level=Q51Thresholds.CONFIDENCE_LEVEL
        )

    # Per-method summary (average correlation for each method)
    method_summary = {}
    for method in METHODS:
        method_corrs = []
        for result in results:
            for pair in result.pairwise_correlations:
                if pair['method1'] == method or pair['method2'] == method:
                    method_corrs.append(pair['circular_correlation'])
        method_summary[method] = float(np.mean(method_corrs)) if method_corrs else 0.0

    # Verdict
    # CRITICAL: Negative control MUST pass for results to be meaningful.
    # If random data shows similar correlation, we can't claim structure.
    if not negative_control.test_passed:
        hypothesis_supported = False
        verdict = (f"INCONCLUSIVE: Negative control failed (random data PC12-PC34 corr = "
                   f"{negative_control.metric_value:.3f}). Cannot distinguish structure from random.")
    elif passing == len(results):
        hypothesis_supported = True
        verdict = "CONFIRMED: All methods agree - phases are structural"
    elif passing >= len(results) * 0.6:
        hypothesis_supported = True
        verdict = "PARTIAL: Most models show method consistency"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: Methods do not agree - phases may be artifacts"

    # Metadata
    metadata = get_test_metadata()

    cross_result = CrossModelConsistencyResult(
        n_models=len(results),
        mean_correlation=mean_corr,
        std_correlation=std_corr,
        models_passing=passing,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        correlation_ci=corr_ci.to_dict(),
        method_summary=method_summary,
        negative_controls=[negative_control.to_dict()],
        test_metadata=metadata
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean correlation: {format_ci(corr_ci)}")
    print(f"Models passing: {passing}/{len(results)}")
    print(f"\nPer-method average correlation:")
    for method, avg in method_summary.items():
        print(f"  {method}: {avg:.3f}")
    print(f"\n{verdict}")

    return results, cross_result


def save_results(
    results: List[ModelConsistencyResult],
    cross_result: CrossModelConsistencyResult,
    output_dir: Path
):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_METHOD_CONSISTENCY',
        'hypothesis': 'Different phase recovery methods should agree',
        'per_model': [asdict(r) for r in results],
        'cross_model': asdict(cross_result),
        'hardening': {
            'methods': METHODS,
            'thresholds': {
                'MEAN_CORR_PASS': MEAN_CORR_PASS,
                'MEAN_CORR_PARTIAL': MEAN_CORR_PARTIAL,
                'MIN_PAIR_CORR': MIN_PAIR_CORR,
            },
            'seeds': {
                'NEGATIVE_CONTROL': Q51Seeds.NEGATIVE_CONTROL,
            }
        }
    }

    # Compute hash
    output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q51_method_consistency_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Result hash: {output['result_hash']}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Method Consistency Test."""
    print("\n" + "=" * 70)
    print("Q51 TEST #6: METHOD CONSISTENCY")
    print("=" * 70)

    results, cross_result = test_consistency_cross_model(
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
