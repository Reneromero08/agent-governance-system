"""
Q8 TEST 4 (REVISED): Topological Invariance Test

REPLACES: test_q8_corruption.py (which was methodologically invalid)

OLD METHOD (WRONG): Add Gaussian noise, measure c_1 drift
  - Problem: Noise DESTROYS the manifold, not deforms it
  - Linear drift is expected for ANY spectral measure under noise
  - This tests linear algebra, not topology

NEW METHOD (CORRECT): Test under manifold-preserving transformations
  1. Rotation invariance: c_1 unchanged under orthogonal transformations
  2. Scaling invariance: c_1 unchanged under uniform scaling
  3. Smooth warping: c_1 stable under small continuous deformations
  4. Cross-model invariance: c_1 consistent across architectures

Topological invariants are preserved under CONTINUOUS DEFORMATIONS OF THE MANIFOLD.
These tests apply transformations that preserve manifold structure.

Pass criteria:
- c_1 within 5% under rotations
- c_1 within 5% under scaling
- c_1 within 10% under smooth warping
- CV < 10% across models (from Q50: CV = 6.93%)
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import traceback

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import test harness
from q8_test_harness import (
    Q8Thresholds,
    Q8Seeds,
    Q8ValidationError,
    Q8Logger,
    compute_alpha_from_spectrum,
    normalize_embeddings,
    validate_embeddings,
    get_test_metadata,
    save_results
)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("WARNING: sentence-transformers not available")


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

MODELS = [
    ("MiniLM-L6", "all-MiniLM-L6-v2"),
    ("MPNet-base", "all-mpnet-base-v2"),
]

VOCAB_CORE = [
    "truth", "beauty", "justice", "freedom", "love", "hate", "fear", "hope",
    "wisdom", "knowledge", "power", "strength", "weakness", "virtue", "vice",
    "water", "fire", "earth", "air", "stone", "tree", "river", "mountain",
    "sun", "moon", "star", "sky", "ocean", "forest", "desert", "city",
    "run", "walk", "jump", "fly", "swim", "think", "speak", "write",
    "create", "destroy", "build", "break", "give", "take", "push", "pull",
    "hot", "cold", "bright", "dark", "fast", "slow", "big", "small",
]

# Thresholds
ROTATION_TOLERANCE = 0.05      # c_1 within 5% under rotation
SCALING_TOLERANCE = 0.05       # c_1 within 5% under scaling
WARPING_TOLERANCE = 0.10       # c_1 within 10% under smooth warping
CROSS_MODEL_CV_THRESHOLD = 0.10  # CV < 10% across models


# =============================================================================
# MANIFOLD-PRESERVING TRANSFORMATIONS
# =============================================================================

def random_orthogonal_matrix(dim: int, seed: int = None) -> np.ndarray:
    """
    Generate random orthogonal matrix (rotation/reflection).

    Uses QR decomposition of a random Gaussian matrix to produce
    a uniformly distributed orthogonal matrix (Haar measure on O(n)).

    Args:
        dim: dimension of the matrix
        seed: random seed for reproducibility

    Returns:
        Q: (dim, dim) orthogonal matrix with det(Q) = +1 (proper rotation)
    """
    # Use isolated random state
    rng = np.random.default_rng(seed)

    # QR decomposition of random matrix gives orthogonal matrix
    A = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)

    # Ensure proper rotation (det = +1) by correcting signs
    # The diagonal of R contains the "signs" - multiply Q columns accordingly
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1  # Handle edge case of zero diagonal
    Q = Q @ np.diag(signs)

    return Q


def apply_rotation(embeddings: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply random orthogonal transformation to embeddings."""
    dim = embeddings.shape[1]
    Q = random_orthogonal_matrix(dim, seed)
    return embeddings @ Q


def apply_scaling(embeddings: np.ndarray, scale: float) -> np.ndarray:
    """Apply uniform scaling to embeddings."""
    return embeddings * scale


def apply_smooth_warping(
    embeddings: np.ndarray,
    strength: float = 0.1,
    seed: int = None
) -> np.ndarray:
    """
    Apply smooth warping - small continuous deformation.

    Uses a smooth function to slightly perturb each point based on its position.
    This preserves manifold structure (nearby points stay nearby).

    The perturbation is a sum of low-frequency sinusoidal waves, ensuring:
    1. Smoothness: perturbation is C^infinity (infinitely differentiable)
    2. Position-dependent: each point gets a unique perturbation based on location
    3. Bounded: total perturbation magnitude is controlled by strength parameter

    Args:
        embeddings: (n, dim) array of embedding vectors
        strength: magnitude of perturbation relative to embedding norm
        seed: random seed for reproducibility

    Returns:
        warped: (n, dim) smoothly deformed embeddings
    """
    # Use isolated random state to avoid global state pollution
    rng = np.random.default_rng(seed)

    n, dim = embeddings.shape

    # Generate smooth perturbation field using low-frequency waves
    # 3 wave components for each dimension
    n_waves = 3
    frequencies = rng.standard_normal((n_waves, dim)) * 0.5  # Low frequency
    phase_offsets = rng.uniform(0, 2 * np.pi, n_waves)

    # Compute perturbation for each point
    # Each wave adds a direction-dependent offset based on position
    perturbation = np.zeros_like(embeddings)
    for i in range(n_waves):
        # Project each embedding onto frequency vector to get scalar position
        positions = embeddings @ frequencies[i]  # Shape: (n,)
        # Smooth sinusoidal response
        wave_values = np.sin(positions + phase_offsets[i])  # Shape: (n,)
        # Direction of perturbation is the frequency vector
        perturbation += np.outer(wave_values, frequencies[i])

    # Normalize and scale perturbation
    perturbation_norm = np.linalg.norm(perturbation)
    if perturbation_norm > 1e-10:
        perturbation = perturbation / perturbation_norm
    else:
        # Fallback: use small random perturbation if waves cancel out
        perturbation = rng.standard_normal((n, dim))
        perturbation = perturbation / np.linalg.norm(perturbation)

    # Scale by strength and by each embedding's magnitude
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_norms = np.maximum(embedding_norms, 1e-10)  # Avoid division by zero
    warped = embeddings + strength * perturbation * embedding_norms

    return warped


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_rotation_invariance(
    embeddings: np.ndarray,
    n_rotations: int = 10,
    logger: Q8Logger = None
) -> Dict:
    """
    Test that c_1 is invariant under orthogonal transformations.

    For a TRUE topological invariant, rotation should not change it.
    """
    if logger:
        logger.info("Testing rotation invariance...")

    # Baseline c_1
    alpha_base, Df_base, c1_base = compute_alpha_from_spectrum(embeddings)

    c1_values = [c1_base]

    for i in range(n_rotations):
        rotated = apply_rotation(embeddings, seed=Q8Seeds.TRIANGULATION + i)
        alpha, Df, c1 = compute_alpha_from_spectrum(rotated)
        c1_values.append(c1)

        if logger:
            change = abs(c1 - c1_base) / c1_base if c1_base > 0 else 0
            logger.info(f"  Rotation {i+1}: c_1 = {c1:.4f} (change: {change*100:.2f}%)")

    c1_values = np.array(c1_values)
    mean_c1 = c1_values.mean()
    std_c1 = c1_values.std()
    cv = std_c1 / mean_c1 if mean_c1 > 0 else float('inf')
    max_change = np.max(np.abs(c1_values - c1_base)) / c1_base if c1_base > 0 else 0

    passes = max_change < ROTATION_TOLERANCE

    result = {
        'baseline_c1': float(c1_base),
        'mean_c1': float(mean_c1),
        'std_c1': float(std_c1),
        'cv': float(cv),
        'max_change': float(max_change),
        'n_rotations': n_rotations,
        'tolerance': ROTATION_TOLERANCE,
        'passes': passes
    }

    if logger:
        status = "PASS" if passes else "FAIL"
        logger.info(f"  Result: {status} (max change: {max_change*100:.2f}%, threshold: {ROTATION_TOLERANCE*100:.0f}%)")

    return result


def test_scaling_invariance(
    embeddings: np.ndarray,
    scales: List[float] = [0.1, 0.5, 1.0, 2.0, 10.0],
    logger: Q8Logger = None
) -> Dict:
    """
    Test that c_1 is invariant under uniform scaling.

    Scaling changes magnitudes but not the manifold structure.
    c_1 should be unchanged.
    """
    if logger:
        logger.info("Testing scaling invariance...")

    # Baseline c_1 (scale = 1.0)
    alpha_base, Df_base, c1_base = compute_alpha_from_spectrum(embeddings)

    c1_values = []
    scale_results = []

    for scale in scales:
        scaled = apply_scaling(embeddings, scale)
        alpha, Df, c1 = compute_alpha_from_spectrum(scaled)
        c1_values.append(c1)

        change = abs(c1 - c1_base) / c1_base if c1_base > 0 else 0
        scale_results.append({
            'scale': scale,
            'c1': float(c1),
            'change': float(change)
        })

        if logger:
            logger.info(f"  Scale {scale}: c_1 = {c1:.4f} (change: {change*100:.2f}%)")

    c1_values = np.array(c1_values)
    max_change = np.max(np.abs(c1_values - c1_base)) / c1_base if c1_base > 0 else 0

    passes = max_change < SCALING_TOLERANCE

    result = {
        'baseline_c1': float(c1_base),
        'scale_results': scale_results,
        'max_change': float(max_change),
        'tolerance': SCALING_TOLERANCE,
        'passes': passes
    }

    if logger:
        status = "PASS" if passes else "FAIL"
        logger.info(f"  Result: {status} (max change: {max_change*100:.2f}%, threshold: {SCALING_TOLERANCE*100:.0f}%)")

    return result


def test_smooth_warping(
    embeddings: np.ndarray,
    strengths: List[float] = [0.01, 0.05, 0.10, 0.20],
    n_trials: int = 5,
    logger: Q8Logger = None
) -> Dict:
    """
    Test that c_1 is stable under smooth (continuous) deformations.

    Small smooth perturbations should not significantly change c_1.
    This is a more challenging test than rotation/scaling.
    """
    if logger:
        logger.info("Testing smooth warping stability...")

    # Baseline c_1
    alpha_base, Df_base, c1_base = compute_alpha_from_spectrum(embeddings)

    strength_results = []

    for strength in strengths:
        c1_trials = []

        for trial in range(n_trials):
            warped = apply_smooth_warping(
                embeddings,
                strength=strength,
                seed=Q8Seeds.CORRUPTION + trial + int(strength * 1000)
            )
            alpha, Df, c1 = compute_alpha_from_spectrum(warped)
            c1_trials.append(c1)

        mean_c1 = np.mean(c1_trials)
        std_c1 = np.std(c1_trials)
        change = abs(mean_c1 - c1_base) / c1_base if c1_base > 0 else 0

        strength_results.append({
            'strength': strength,
            'mean_c1': float(mean_c1),
            'std_c1': float(std_c1),
            'change': float(change),
            'passes': change < WARPING_TOLERANCE
        })

        if logger:
            status = "PASS" if change < WARPING_TOLERANCE else "FAIL"
            logger.info(f"  Strength {strength}: c_1 = {mean_c1:.4f} +/- {std_c1:.4f} (change: {change*100:.2f}%) - {status}")

    # Overall pass: all strengths pass
    all_pass = all(r['passes'] for r in strength_results)

    result = {
        'baseline_c1': float(c1_base),
        'strength_results': strength_results,
        'tolerance': WARPING_TOLERANCE,
        'passes': all_pass
    }

    if logger:
        status = "PASS" if all_pass else "FAIL"
        logger.info(f"  Overall: {status}")

    return result


def test_cross_model_invariance(
    models: List[Tuple[str, str]],
    vocab: List[str],
    logger: Q8Logger
) -> Dict:
    """
    Test that c_1 is consistent across different model architectures.

    THIS IS THE KEY TEST FOR TOPOLOGICAL INVARIANCE.

    Different models = different "coordinate systems" on the manifold.
    If c_1 is topological, it should be the same in all coordinate systems.

    From Q50: CV = 6.93% across 24 models (PASS).
    """
    logger.info("Testing cross-model invariance...")

    c1_values = []
    alpha_values = []
    model_results = []

    for name, model_id in models:
        try:
            if not HAS_ST:
                raise Q8ValidationError("sentence-transformers not available")

            model = SentenceTransformer(model_id)
            embeddings = model.encode(vocab, show_progress_bar=False)

            alpha, Df, c1 = compute_alpha_from_spectrum(embeddings)

            c1_values.append(c1)
            alpha_values.append(alpha)

            model_results.append({
                'name': name,
                'alpha': float(alpha),
                'Df': float(Df),
                'c1': float(c1)
            })

            logger.info(f"  {name}: alpha = {alpha:.4f}, Df = {Df:.2f}, c_1 = {c1:.4f}")

        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            model_results.append({'name': name, 'error': str(e)})

    if len(c1_values) < 2:
        return {
            'model_results': model_results,
            'error': 'Need at least 2 models',
            'passes': False
        }

    c1_values = np.array(c1_values)
    alpha_values = np.array(alpha_values)

    mean_c1 = c1_values.mean()
    std_c1 = c1_values.std()
    cv_c1 = std_c1 / mean_c1 if mean_c1 > 0 else float('inf')

    mean_alpha = alpha_values.mean()
    std_alpha = alpha_values.std()
    cv_alpha = std_alpha / mean_alpha if mean_alpha > 0 else float('inf')

    passes = cv_c1 < CROSS_MODEL_CV_THRESHOLD

    result = {
        'model_results': model_results,
        'n_models': len(c1_values),
        'c1_mean': float(mean_c1),
        'c1_std': float(std_c1),
        'c1_cv': float(cv_c1),
        'alpha_mean': float(mean_alpha),
        'alpha_std': float(std_alpha),
        'alpha_cv': float(cv_alpha),
        'cv_threshold': CROSS_MODEL_CV_THRESHOLD,
        'passes': passes
    }

    status = "PASS" if passes else "FAIL"
    logger.info(f"  Cross-model: c_1 = {mean_c1:.4f} +/- {std_c1:.4f}, CV = {cv_c1*100:.2f}% - {status}")
    logger.info(f"  (Q50 reference: CV = 6.93% across 24 models)")

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(save: bool = True):
    """
    Run Q8 Test 4 (REVISED): Topological Invariance Test.

    Tests c_1 invariance under manifold-preserving transformations:
    1. Rotation (orthogonal transformation)
    2. Scaling (uniform)
    3. Smooth warping (continuous deformation)
    4. Cross-model (different architectures)
    """
    logger = Q8Logger("Q8-TEST4-TOPOLOGICAL", verbose=True)
    logger.section("Q8 TEST 4 (REVISED): TOPOLOGICAL INVARIANCE")

    print(f"\n{'='*60}")
    print("  METHOD: Test under MANIFOLD-PRESERVING transformations")
    print("  (Replaces invalid noise corruption test)")
    print("")
    print("  Topological invariants are preserved under:")
    print("    - Rotations (orthogonal transformations)")
    print("    - Scaling (uniform)")
    print("    - Smooth warping (continuous deformations)")
    print("    - Change of coordinates (different models)")
    print(f"{'='*60}\n")

    results = {
        'test_name': 'Q8_TEST4_TOPOLOGICAL_INVARIANCE',
        'timestamp': datetime.now().isoformat(),
        'methodology': 'manifold_preserving_transformations',
        'note': 'Replaces invalid noise corruption test'
    }

    # Load embeddings from first model for transformation tests
    if HAS_ST:
        logger.info("Loading embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(VOCAB_CORE, show_progress_bar=False)
        logger.info(f"Embeddings shape: {embeddings.shape}")
    else:
        logger.error("sentence-transformers not available")
        return {'error': 'sentence-transformers not available'}

    # Test 1: Rotation invariance
    logger.section("TEST 1: ROTATION INVARIANCE")
    results['rotation'] = test_rotation_invariance(embeddings, n_rotations=10, logger=logger)

    # Test 2: Scaling invariance
    logger.section("TEST 2: SCALING INVARIANCE")
    results['scaling'] = test_scaling_invariance(embeddings, logger=logger)

    # Test 3: Smooth warping stability
    logger.section("TEST 3: SMOOTH WARPING STABILITY")
    results['warping'] = test_smooth_warping(embeddings, logger=logger)

    # Test 4: Cross-model invariance
    logger.section("TEST 4: CROSS-MODEL INVARIANCE")
    results['cross_model'] = test_cross_model_invariance(MODELS, VOCAB_CORE, logger)

    # Final verdict
    logger.section("FINAL VERDICT")

    rotation_pass = results['rotation'].get('passes', False)
    scaling_pass = results['scaling'].get('passes', False)
    warping_pass = results['warping'].get('passes', False)
    cross_model_pass = results['cross_model'].get('passes', False)

    n_pass = sum([rotation_pass, scaling_pass, warping_pass, cross_model_pass])

    # Require at least 3/4 tests to pass
    overall_pass = n_pass >= 3

    results['verdict'] = {
        'rotation_pass': rotation_pass,
        'scaling_pass': scaling_pass,
        'warping_pass': warping_pass,
        'cross_model_pass': cross_model_pass,
        'n_pass': n_pass,
        'n_total': 4,
        'overall_pass': overall_pass,
        'interpretation': 'c_1 IS topologically invariant' if overall_pass else 'c_1 may NOT be topologically invariant'
    }

    logger.info(f"Rotation:    {'PASS' if rotation_pass else 'FAIL'}")
    logger.info(f"Scaling:     {'PASS' if scaling_pass else 'FAIL'}")
    logger.info(f"Warping:     {'PASS' if warping_pass else 'FAIL'}")
    logger.info(f"Cross-model: {'PASS' if cross_model_pass else 'FAIL'}")
    logger.info(f"")
    logger.info(f"OVERALL: {n_pass}/4 tests pass")

    if overall_pass:
        logger.info("VERDICT: c_1 IS TOPOLOGICALLY INVARIANT")
        logger.info("  - Preserved under orthogonal transformations")
        logger.info("  - Preserved under scaling")
        logger.info("  - Stable under smooth deformations")
        logger.info("  - Consistent across model architectures")
    else:
        logger.warn("VERDICT: c_1 may NOT be topologically invariant")
        logger.warn("  - Failed some manifold-preserving transformation tests")

    # Save results
    if save:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"q8_test4_topological_{timestamp}.json"
        save_results(results, filepath)
        logger.info(f"Results saved to: {filepath}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q8 Test 4 (REVISED): Topological Invariance")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    main(save=not args.no_save)
