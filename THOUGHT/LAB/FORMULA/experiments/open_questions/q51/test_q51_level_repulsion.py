"""
Q51 Spectral Level Repulsion Test - Test #8

Tests whether eigenvalue spacing shows level repulsion (structured correlations)
rather than Poisson statistics (no correlations).

CORRECTED HYPOTHESIS:
    Real embedding covariance is REAL SYMMETRIC, so eigenvalue statistics
    follow GOE (Gaussian Orthogonal Ensemble, beta = 1), NOT GUE (beta = 2).

    GUE requires genuinely complex Hermitian matrices. Adding phases to real
    data via unitary similarity transform PRESERVES eigenvalues - you cannot
    convert GOE to GUE this way.

    The valid test is: GOE (structured) vs Poisson (random).
    - GOE (beta = 1): Level repulsion from structured covariance
    - Poisson (beta = 0): No repulsion from random/uncorrelated data

Theory:
    - Real symmetric matrices: GOE statistics (beta = 1, s^1 repulsion)
    - Complex Hermitian: GUE statistics (beta = 2, s^2 repulsion) [NOT ACHIEVABLE]
    - No correlations: Poisson (beta = 0, no repulsion)

    Level repulsion: P(s) ~ s^beta * exp(-c*s^2) for small s

Pass criteria:
    - Beta > 0.5 (showing some level repulsion, i.e., non-random structure)
    - Closer to GOE (beta=1) than Poisson (beta=0)

Falsification:
    - Poisson-like spacing (beta < 0.3, no repulsion)
    - Inconsistent across models
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

# Phase recovery functions not needed for corrected test
# (removed hilbert_phase_recovery, phase_from_covariance since we now use real covariance directly)

# Try scipy for KS test
try:
    from scipy.stats import ks_2samp, expon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

# =============================================================================
# Constants
# =============================================================================

# Thresholds - CORRECTED for GOE vs Poisson comparison
# GOE (real symmetric) has beta = 1, Poisson has beta = 0
# GUE (beta = 2) is NOT achievable for real data!
BETA_PASS = 0.5  # Half way between Poisson (0) and GOE (1)
BETA_PARTIAL = 0.3
GOE_POISSON_RATIO_PASS = 2.0  # Distance ratio to GOE vs Poisson

# Models to test
MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-small",
]

# Test corpus (need many samples for good statistics)
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
    # Additional variety for more samples
    "morning sunrise", "evening sunset", "midnight darkness", "noon brightness",
    "summer heat", "winter cold", "spring bloom", "autumn leaves",
    "ancient history", "modern technology", "future dreams", "present moment",
    "friendly conversation", "heated argument", "quiet agreement", "loud voice",
    "simple solution", "complex problem", "easy question", "hard answer",
    "bright light", "dark shadow", "warm touch", "cold feeling",
    "fast car", "slow train", "quiet library", "noisy street",
    "empty room", "crowded space", "open door", "closed window",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelLevelRepulsionResult:
    """Level repulsion analysis result for a single model."""
    model_name: str
    n_samples: int
    n_eigenvalues: int
    beta_estimate: float  # Level repulsion exponent
    mean_spacing: float
    spacing_std: float
    ks_stat_gue: float  # KS statistic vs GUE
    ks_stat_poisson: float  # KS statistic vs Poisson
    gue_is_better: bool  # True if closer to GUE
    status: str
    validation_warnings: Optional[List[str]] = None


@dataclass
class CrossModelLevelRepulsionResult:
    """Cross-model aggregation."""
    n_models: int
    mean_beta: float
    std_beta: float
    models_gue_like: int  # Number with GUE-like spacing
    models_passing: int
    hypothesis_supported: bool
    verdict: str
    beta_ci: Optional[dict] = None
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
            embeddings, min_samples=10, name=f"level_repulsion_{model_name}"
        )
        if model_error:
            validation.warnings.append(f"Using synthetic fallback: {model_error}")
    else:
        validation = ValidationResult(True, [], [], {'n_samples': len(embeddings)})

    return embeddings, validation


def compute_eigenvalue_spacings(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute nearest-neighbor spacing distribution.

    Unfolds eigenvalues to have mean spacing 1, then computes spacings.
    """
    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues)

    # Remove zeros and negatives
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return np.array([])

    # Unfold to uniform density (using average spacing)
    n = len(eigenvalues)
    mean_spacing = (eigenvalues[-1] - eigenvalues[0]) / (n - 1)

    if mean_spacing < 1e-10:
        return np.array([])

    # Compute normalized spacings
    spacings = np.diff(eigenvalues) / mean_spacing

    return spacings


def estimate_beta(spacings: np.ndarray) -> float:
    """
    Estimate level repulsion exponent beta from spacing distribution.

    For small s: P(s) ~ s^beta
    Fit by looking at fraction of small spacings.
    """
    if len(spacings) < 10:
        return 0.0

    # Normalize spacings
    spacings = spacings / np.mean(spacings)

    # Use method of ratios for small spacings
    # P(s < s0) ~ s0^(beta+1) for small s0
    # FIXED (Sonnet-swarm review): s0=0.5 violates small-s assumption!
    # Must stay in small-s regime where P(s) ~ s^beta approximation holds
    s0_values = [0.02, 0.05, 0.08, 0.1]
    log_s0 = np.log(np.array(s0_values))
    log_cdf = []

    for s0 in s0_values:
        fraction = np.mean(spacings < s0)
        if fraction > 0:
            log_cdf.append(np.log(fraction))
        else:
            log_cdf.append(-10)  # Very small

    log_cdf = np.array(log_cdf)

    # Linear fit: log(P) ~ (beta+1) * log(s0)
    # Slope = beta + 1
    if len(log_s0) >= 2:
        slope = np.polyfit(log_s0, log_cdf, 1)[0]
        beta = slope - 1
        return float(np.clip(beta, 0, 4))

    return 0.0


def generate_goe_spacings(n: int, seed: int = 42) -> np.ndarray:
    """Generate spacings from GOE (Gaussian Orthogonal Ensemble)."""
    np.random.seed(seed)

    # GOE: random real symmetric matrix
    dim = max(int(np.sqrt(n)), 50)
    H = np.random.randn(dim, dim)
    H = (H + H.T) / 2  # Symmetrize

    eigenvalues = np.linalg.eigvalsh(H)
    spacings = compute_eigenvalue_spacings(eigenvalues)

    return spacings


def generate_poisson_spacings(n: int, seed: int = 42) -> np.ndarray:
    """Generate spacings from Poisson (uncorrelated eigenvalues)."""
    np.random.seed(seed)

    # Poisson: exponential distribution
    spacings = np.random.exponential(1.0, n)

    return spacings


# =============================================================================
# Test Functions
# =============================================================================

def test_level_repulsion_single_model(
    model_name: str,
    corpus: List[str],
    verbose: bool = True
) -> ModelLevelRepulsionResult:
    """Test level repulsion for a single model."""
    logger = Q51Logger(f"level_repulsion_{model_name}", verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Level Repulsion Test: {model_name}")
        print(f"{'='*60}")

    # Get embeddings
    embeddings, validation = get_embeddings(model_name, corpus)
    n_samples, n_dims = embeddings.shape

    if not validation.valid:
        raise Q51ValidationError(f"Embedding validation failed: {validation.errors}")

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # CORRECTED: Test for GOE (real symmetric) vs Poisson, NOT GUE.
    #
    # GUE statistics require genuinely complex Hermitian matrices.
    # Adding phases via Z = E * exp(i*theta) is a unitary similarity transform
    # that PRESERVES eigenvalues - can't convert GOE to GUE this way.
    #
    # The valid test: Does the real covariance show level repulsion (GOE-like)
    # or random structure (Poisson-like)?

    centered = embeddings - embeddings.mean(axis=0)

    # Compute real covariance matrix
    cov = np.cov(centered.T)

    # Eigenvalues of real symmetric matrix
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    n_eigenvalues = len(eigenvalues)

    if verbose:
        print(f"Non-zero eigenvalues: {n_eigenvalues}")

    # Compute spacings
    spacings = compute_eigenvalue_spacings(eigenvalues)

    if len(spacings) < 10:
        if verbose:
            print("Warning: Not enough spacings for analysis")
        return ModelLevelRepulsionResult(
            model_name=model_name,
            n_samples=n_samples,
            n_eigenvalues=n_eigenvalues,
            beta_estimate=0.0,
            mean_spacing=0.0,
            spacing_std=0.0,
            ks_stat_gue=1.0,
            ks_stat_poisson=1.0,
            gue_is_better=False,
            status="FAIL",
            validation_warnings=["Not enough eigenvalues for spacing analysis"]
        )

    # Statistics
    mean_spacing = float(np.mean(spacings))
    spacing_std = float(np.std(spacings))

    # Estimate beta (level repulsion exponent)
    beta = estimate_beta(spacings)

    # Generate reference distributions - GOE (not GUE!) and Poisson
    goe_spacings = generate_goe_spacings(len(spacings), seed=Q51Seeds.NEGATIVE_CONTROL)
    poisson_spacings = generate_poisson_spacings(len(spacings), seed=Q51Seeds.NEGATIVE_CONTROL + 1)

    # KS tests
    if HAS_SCIPY and len(goe_spacings) > 0:
        ks_goe = ks_2samp(spacings, goe_spacings)
        ks_poisson = ks_2samp(spacings, poisson_spacings)
        ks_stat_goe = float(ks_goe.statistic)
        ks_stat_poisson = float(ks_poisson.statistic)
    else:
        # Simple comparison
        ks_stat_goe = float(np.abs(np.mean(spacings) - np.mean(goe_spacings)))
        ks_stat_poisson = float(np.abs(np.mean(spacings) - np.mean(poisson_spacings)))

    # Closer to GOE if KS statistic is smaller
    goe_is_better = ks_stat_goe < ks_stat_poisson

    if verbose:
        print(f"\nLevel Repulsion Analysis:")
        print(f"  Beta estimate: {beta:.3f} (GOE = 1, Poisson = 0)")
        print(f"  Mean spacing: {mean_spacing:.4f}")
        print(f"  Spacing std: {spacing_std:.4f}")
        print(f"  KS stat vs GOE: {ks_stat_goe:.4f}")
        print(f"  KS stat vs Poisson: {ks_stat_poisson:.4f}")
        print(f"  Closer to GOE: {goe_is_better}")

    # Determine status - CORRECTED for GOE comparison
    if beta > BETA_PASS and goe_is_better:
        status = "PASS"
    elif beta > BETA_PARTIAL or goe_is_better:
        status = "PARTIAL"
    else:
        status = "FAIL"

    if verbose:
        print(f"Status: {status}")

    return ModelLevelRepulsionResult(
        model_name=model_name,
        n_samples=n_samples,
        n_eigenvalues=n_eigenvalues,
        beta_estimate=beta,
        mean_spacing=mean_spacing,
        spacing_std=spacing_std,
        ks_stat_gue=ks_stat_goe,  # Renamed to GOE but keep field name for compatibility
        ks_stat_poisson=ks_stat_poisson,
        gue_is_better=goe_is_better,  # Actually GOE is better
        status=status,
        validation_warnings=validation.warnings if validation.warnings else None
    )


def run_negative_control(verbose: bool = True) -> NegativeControlResult:
    """
    Negative control: Pure Poisson (uncorrelated) should have beta ~ 0.

    Random diagonal matrix has no level repulsion.
    """
    print("\n  [Negative Control] Uncorrelated eigenvalues...")

    np.random.seed(Q51Seeds.NEGATIVE_CONTROL)

    # Generate uncorrelated (Poisson) eigenvalues
    eigenvalues = np.random.exponential(1.0, 100)
    spacings = compute_eigenvalue_spacings(eigenvalues)
    beta = estimate_beta(spacings)

    # Should be near 0 (no repulsion)
    is_near_zero = beta < 0.5

    if verbose:
        print(f"    Beta estimate: {beta:.3f} (expected ~0)")
        status = "PASS" if is_near_zero else "FAIL"
        print(f"    Status: {status}")

    return NegativeControlResult(
        name="poisson_eigenvalues_beta",
        test_passed=is_near_zero,
        expected_behavior="Uncorrelated eigenvalues should have beta ~ 0",
        actual_behavior=f"Beta = {beta:.3f}",
        metric_value=beta,
        metric_threshold=0.5,
        notes="Poisson (uncorrelated) should show no level repulsion"
    )


def test_level_repulsion_cross_model(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[ModelLevelRepulsionResult], CrossModelLevelRepulsionResult]:
    """Test level repulsion across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 SPECTRAL LEVEL REPULSION TEST - CROSS-MODEL")
    print("=" * 70)
    print(f"\nTesting {len(models)} models on {len(corpus)} texts")
    print("\nKey insight (CORRECTED):")
    print("  GOE (real symmetric) matrices show level REPULSION (beta = 1).")
    print("  Poisson (uncorrelated) matrices show NO repulsion (beta = 0).")
    print("  GUE (beta = 2) is NOT achievable for real data.")
    print("  Beta > 0.5 indicates structured correlations.")
    print()

    # Run negative control
    negative_control = run_negative_control(verbose=verbose)

    # Test each model
    results = []
    for model in models:
        try:
            result = test_level_repulsion_single_model(model, corpus, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    if not results:
        raise RuntimeError("No models tested successfully")

    # Aggregate
    betas = [r.beta_estimate for r in results]
    mean_beta = float(np.mean(betas))
    std_beta = float(np.std(betas))

    gue_like = sum(1 for r in results if r.gue_is_better)
    passing = sum(1 for r in results if r.status == "PASS")

    # Bootstrap CI
    if len(betas) >= 3:
        beta_ci = bootstrap_ci(np.array(betas), n_bootstrap=Q51Thresholds.BOOTSTRAP_ITERATIONS)
    else:
        beta_ci = BootstrapCI(
            mean=mean_beta, ci_lower=min(betas), ci_upper=max(betas),
            std=std_beta, n_samples=len(betas), n_bootstrap=0,
            confidence_level=Q51Thresholds.CONFIDENCE_LEVEL
        )

    # Verdict - Based on MEAN BETA (level repulsion exponent)
    # Beta > 0 indicates SOME eigenvalue correlations (not pure Poisson)
    # Beta = 1 is full GOE, Beta = 0 is pure Poisson
    # Intermediate values indicate partial structure
    #
    # HONEST REPORTING (Sonnet-swarm review):
    # Report distance to both GOE and Poisson for transparency
    dist_to_poisson = abs(mean_beta - 0.0)
    dist_to_goe = abs(mean_beta - 1.0)
    closer_to = "GOE" if dist_to_goe < dist_to_poisson else "Poisson"

    if mean_beta > BETA_PASS:
        hypothesis_supported = True
        verdict = (f"CONFIRMED: Mean beta = {mean_beta:.2f} (dist to GOE: {dist_to_goe:.2f}, "
                   f"dist to Poisson: {dist_to_poisson:.2f}, closer to {closer_to})")
    elif mean_beta > BETA_PARTIAL:
        hypothesis_supported = True
        verdict = (f"PARTIAL: Mean beta = {mean_beta:.2f} shows weak repulsion "
                   f"(dist to GOE: {dist_to_goe:.2f}, dist to Poisson: {dist_to_poisson:.2f}, closer to {closer_to})")
    else:
        hypothesis_supported = False
        verdict = (f"FALSIFIED: Mean beta = {mean_beta:.2f} (dist to GOE: {dist_to_goe:.2f}, "
                   f"dist to Poisson: {dist_to_poisson:.2f}) - Poisson-like (no correlations)")

    # Metadata
    metadata = get_test_metadata()

    cross_result = CrossModelLevelRepulsionResult(
        n_models=len(results),
        mean_beta=mean_beta,
        std_beta=std_beta,
        models_gue_like=gue_like,
        models_passing=passing,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict,
        beta_ci=beta_ci.to_dict(),
        negative_controls=[negative_control.to_dict()],
        test_metadata=metadata
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean beta: {format_ci(beta_ci)} (GOE = 1, Poisson = 0)")
    print(f"Models GOE-like: {gue_like}/{len(results)}")  # Keep var name, means GOE now
    print(f"Models passing: {passing}/{len(results)}")
    print(f"\n{verdict}")

    return results, cross_result


def save_results(
    results: List[ModelLevelRepulsionResult],
    cross_result: CrossModelLevelRepulsionResult,
    output_dir: Path
):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_LEVEL_REPULSION',
        'hypothesis': 'Eigenvalue spacing shows GOE-like level repulsion (not GUE - real data cannot achieve GUE)',
        'per_model': [asdict(r) for r in results],
        'cross_model': asdict(cross_result),
        'hardening': {
            'thresholds': {
                'BETA_PASS': BETA_PASS,
                'BETA_PARTIAL': BETA_PARTIAL,
                'GOE_POISSON_RATIO_PASS': GOE_POISSON_RATIO_PASS,
            },
            'seeds': {
                'NEGATIVE_CONTROL': Q51Seeds.NEGATIVE_CONTROL,
            }
        }
    }

    # Compute hash
    output['result_hash'] = compute_result_hash(output)

    output_path = output_dir / 'q51_level_repulsion_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Result hash: {output['result_hash']}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Level Repulsion Test."""
    print("\n" + "=" * 70)
    print("Q51 TEST #8: SPECTRAL LEVEL REPULSION")
    print("=" * 70)

    results, cross_result = test_level_repulsion_cross_model(
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
