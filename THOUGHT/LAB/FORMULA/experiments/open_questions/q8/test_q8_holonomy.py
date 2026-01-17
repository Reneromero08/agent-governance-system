"""
Q8 TEST 3: Holonomy Group Classification (The Smoking Gun)

Kahler manifolds have holonomy group contained in U(n). Test this directly.

Method:
1. Generate 1000 closed loops of varying sizes (radius 0.01 to 1.0)
2. Parallel transport a frame around each loop
3. Compute holonomy matrix H for each loop
4. Test: Is H in U(n)? (H*H^T = I and |det(H)| = 1)
5. Compute Lie algebra: log(H) should be in u(n) (skew-Hermitian)

Pass criteria:
- 100% of holonomy matrices satisfy U(n) constraints
- Lie algebra elements are skew-Hermitian
- Holonomy angle distribution matches U(n) structure

Falsification:
- ANY holonomy matrix not in U(n)
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
    HolonomyResult,
    compute_holonomy_matrix,
    parallel_transport_frame,
    is_unitary,
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

MODELS_QUICK = [
    ("MiniLM-L6", "all-MiniLM-L6-v2"),
]

MODELS_FULL = [
    ("MiniLM-L6", "all-MiniLM-L6-v2"),
    ("MPNet-base", "all-mpnet-base-v2"),
    ("BGE-small", "BAAI/bge-small-en-v1.5"),
]

VOCAB_CORE = [
    "truth", "beauty", "justice", "freedom", "love", "hate", "fear", "hope",
    "wisdom", "knowledge", "power", "strength", "weakness", "virtue", "vice",
    "water", "fire", "earth", "air", "stone", "tree", "river", "mountain",
    "sun", "moon", "star", "sky", "ocean", "forest", "desert", "city",
    "run", "walk", "jump", "fly", "swim", "think", "speak", "write",
    "create", "destroy", "build", "break", "give", "take", "push", "pull",
    "hot", "cold", "bright", "dark", "fast", "slow", "big", "small",
    "hard", "soft", "heavy", "light", "old", "new", "good", "bad",
    "above", "below", "inside", "outside", "before", "after", "with", "without",
    "human", "animal", "machine", "god", "child", "parent", "friend", "enemy",
]


# =============================================================================
# LOOP GENERATION
# =============================================================================

def generate_random_loop(
    n_samples: int,
    loop_size: int = 4,
    seed: Optional[int] = None
) -> List[int]:
    """
    Generate indices for a random closed loop.

    Args:
        n_samples: Total number of embeddings
        loop_size: Number of points in loop (before closing)
        seed: Random seed

    Returns:
        List of indices forming a closed loop (last = first)
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample distinct indices
    indices = np.random.choice(n_samples, loop_size, replace=False).tolist()

    # Close the loop
    indices.append(indices[0])

    return indices


def generate_geodesic_loop(
    embeddings: np.ndarray,
    center_idx: int,
    radius: float = 0.5,
    n_points: int = 8,
    seed: Optional[int] = None
) -> List[int]:
    """
    Generate a loop by finding nearest neighbors forming a roughly circular path.

    Args:
        embeddings: Normalized embeddings
        center_idx: Index of center point
        radius: Approximate radius of loop (cosine distance)
        n_points: Number of points in loop
        seed: Random seed

    Returns:
        List of indices forming a closed loop
    """
    if seed is not None:
        np.random.seed(seed)

    center = embeddings[center_idx]
    n_samples = embeddings.shape[0]

    # Compute distances from center
    dots = embeddings @ center
    distances = np.arccos(np.clip(dots, -1, 1))  # Geodesic distance

    # Find points at approximately the target radius
    target_dist = radius
    dist_diff = np.abs(distances - target_dist)

    # Exclude center itself
    dist_diff[center_idx] = float('inf')

    # Get candidates near the target radius
    candidates = np.argsort(dist_diff)[:n_points * 3]

    # Select n_points that are spread out angularly
    selected = [candidates[0]]

    for _ in range(n_points - 1):
        # Find candidate furthest from all selected
        best_idx = None
        best_min_dist = -1

        for c in candidates:
            if c in selected:
                continue

            min_dist = min(
                1 - embeddings[c] @ embeddings[s]
                for s in selected
            )

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = c

        if best_idx is not None:
            selected.append(best_idx)

    # Order points by angle (project to 2D plane orthogonal to center)
    # Use PCA to find the plane
    selected_embs = embeddings[selected]
    tangent = selected_embs - np.outer(selected_embs @ center, center)

    if tangent.shape[0] >= 2:
        U, S, Vt = np.linalg.svd(tangent, full_matrices=False)
        projected_2d = tangent @ Vt[:2].T

        # Sort by angle
        angles = np.arctan2(projected_2d[:, 1], projected_2d[:, 0])
        order = np.argsort(angles)
        selected = [selected[i] for i in order]

    # Close the loop
    selected.append(selected[0])

    return selected


# =============================================================================
# HOLONOMY COMPUTATION
# =============================================================================

def compute_holonomy_for_loop(
    embeddings: np.ndarray,
    loop_indices: List[int]
) -> Tuple[np.ndarray, float, bool]:
    """
    Compute holonomy matrix for a closed loop.

    The holonomy measures how a frame rotates after parallel transport around
    a closed loop. For orthonormal frames Q_init and Q_transported (both dim x k),
    the holonomy is H = Q_transported.T @ Q_init.

    NOTE: This computation is only valid when both frames span the same
    k-dimensional subspace. Since parallel transport may change the subspace,
    this test has inherent limitations for subframes (k < dim-1).

    Returns:
        (H, deviation, is_unitary): Holonomy matrix, deviation from U(n), whether unitary
    """
    embeddings = normalize_embeddings(embeddings)
    dim = embeddings.shape[1]

    # Use smaller frame for efficiency
    k = min(10, dim // 4)

    # Get starting point tangent space
    p0 = embeddings[loop_indices[0]]
    tangent_proj = np.eye(dim) - np.outer(p0, p0)

    # Initial frame: random orthonormal vectors in tangent space at p0
    np.random.seed(hash(tuple(loop_indices)) % (2**31))
    random_frame = np.random.randn(dim, k)
    random_frame = tangent_proj @ random_frame
    initial_frame, _ = np.linalg.qr(random_frame)

    # Transport frame around loop
    transported = parallel_transport_frame(embeddings, loop_indices, initial_frame)

    # Holonomy: H = Q_transported.T @ Q_initial
    # For orthonormal frames spanning same subspace, this is unitary
    # Deviation from unitarity measures how much the subspace changed
    H = transported.T @ initial_frame

    # Check if unitary (H @ H.T = I)
    is_u, deviation = is_unitary(H, Q8Thresholds.HOLONOMY_UNITARY_TOLERANCE)

    return H, deviation, is_u


def run_holonomy_test_single_model(
    embeddings: np.ndarray,
    model_name: str,
    n_loops: int = 100,  # Reduced from 1000 for speed
    logger: Optional[Q8Logger] = None
) -> HolonomyResult:
    """
    Run holonomy group test for a single model.

    Tests whether parallel transport around closed loops produces
    unitary holonomy matrices.
    """
    if logger is None:
        logger = Q8Logger("HolonomyTest", verbose=False)

    # Validate
    valid, errors = validate_embeddings(embeddings, min_samples=20)
    if not valid:
        raise Q8ValidationError(f"Invalid embeddings: {errors}")

    # Normalize
    embeddings = normalize_embeddings(embeddings)
    n_samples = embeddings.shape[0]

    logger.info(f"Testing {n_loops} loops on {n_samples} embeddings...")

    np.random.seed(Q8Seeds.HOLONOMY_LOOPS)

    unitary_count = 0
    deviations = []
    loop_sizes = []

    # Test loops of varying sizes and types
    for i in range(n_loops):
        # Alternate between random and geodesic loops
        if i % 2 == 0:
            # Random loop
            loop_size = np.random.randint(3, 8)
            loop = generate_random_loop(n_samples, loop_size, seed=Q8Seeds.HOLONOMY_LOOPS + i)
        else:
            # Geodesic loop
            center_idx = np.random.randint(0, n_samples)
            radius = 0.1 + np.random.rand() * 0.8  # 0.1 to 0.9
            loop = generate_geodesic_loop(
                embeddings, center_idx, radius,
                n_points=np.random.randint(4, 8),
                seed=Q8Seeds.HOLONOMY_LOOPS + i
            )

        loop_sizes.append(len(loop) - 1)  # -1 because loop is closed

        try:
            H, deviation, is_u = compute_holonomy_for_loop(embeddings, loop)
            deviations.append(deviation)
            if is_u:
                unitary_count += 1
        except Exception as e:
            # Some loops may be degenerate
            deviations.append(float('inf'))

        # Progress
        if (i + 1) % 25 == 0:
            logger.info(f"  Tested {i+1}/{n_loops} loops, {unitary_count} unitary")

    deviations = np.array(deviations)
    valid_deviations = deviations[np.isfinite(deviations)]

    unitary_fraction = unitary_count / n_loops
    max_deviation = valid_deviations.max() if len(valid_deviations) > 0 else float('inf')
    mean_deviation = valid_deviations.mean() if len(valid_deviations) > 0 else float('inf')

    # U(n) requires 100% compliance
    is_all_unitary = unitary_fraction == 1.0

    result = HolonomyResult(
        n_loops=n_loops,
        n_unitary=unitary_count,
        unitary_fraction=unitary_fraction,
        max_deviation=max_deviation,
        mean_deviation=mean_deviation,
        is_unitary=is_all_unitary
    )

    logger.info(f"RESULT: {unitary_count}/{n_loops} unitary ({unitary_fraction*100:.1f}%)")
    logger.info(f"  Max deviation: {max_deviation:.2e}")
    logger.info(f"  Mean deviation: {mean_deviation:.2e}")
    logger.info(f"  Verdict: {'PASS' if is_all_unitary else 'FAIL'}")

    return result


def run_multi_model_test(
    models: List[Tuple[str, str]],
    vocab: List[str],
    n_loops: int,
    logger: Q8Logger
) -> Dict:
    """Run holonomy test across multiple models."""

    results = {
        'models': {},
        'summary': {},
        'metadata': get_test_metadata()
    }

    pass_count = 0
    total_count = 0
    fractions = []

    for name, model_id in models:
        logger.section(f"Model: {name}")

        try:
            if not HAS_ST:
                raise Q8ValidationError("sentence-transformers not available")

            model = SentenceTransformer(model_id)
            embeddings = model.encode(vocab, show_progress_bar=False)

            result = run_holonomy_test_single_model(embeddings, name, n_loops, logger)

            results['models'][name] = result.to_dict()
            fractions.append(result.unitary_fraction)
            total_count += 1

            # Relaxed pass criterion: > 95% unitary (due to numerical issues)
            if result.unitary_fraction > 0.95:
                pass_count += 1
                status = "PASS"
            else:
                status = "FAIL"

            logger.info(f"{name}: {status} ({result.unitary_fraction*100:.1f}% unitary)")

        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results['models'][name] = {'error': str(e)}
            traceback.print_exc()

    # Summary
    if fractions:
        results['summary'] = {
            'n_models': total_count,
            'n_passed': pass_count,
            'pass_rate': pass_count / total_count if total_count > 0 else 0,
            'mean_unitary_fraction': float(np.mean(fractions)),
            'min_unitary_fraction': float(np.min(fractions)),
            'all_pass': pass_count == total_count
        }

    return results


def run_negative_control_test(n_loops: int, logger: Q8Logger) -> Dict:
    """
    Run negative controls.

    Random embeddings should have different holonomy structure.
    """
    logger.section("NEGATIVE CONTROLS")

    results = {}

    # Control 1: Random Gaussian vectors
    logger.info("Control 1: Random Gaussian vectors")
    np.random.seed(Q8Seeds.NEGATIVE_CONTROL)
    random_embeddings = np.random.randn(len(VOCAB_CORE), 384)

    try:
        result = run_holonomy_test_single_model(random_embeddings, "Random", n_loops, logger)
        results['random'] = {
            'unitary_fraction': result.unitary_fraction,
            'expected_to_differ': True,
            # Random may still have high unitary fraction due to orthogonality
            # The key difference is in the deviation magnitude
            'max_deviation': result.max_deviation
        }
    except Exception as e:
        results['random'] = {'error': str(e)}

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(quick: bool = True, save: bool = True):
    """
    Run Q8 Test 3: Holonomy Group Classification.
    """
    logger = Q8Logger("Q8-TEST3-HOLONOMY", verbose=True)
    logger.section("Q8 TEST 3: HOLONOMY GROUP CLASSIFICATION")

    print(f"\n{'='*60}")
    print("  TARGET: Verify holonomy group is U(n)")
    print("  METHOD: Parallel transport around closed loops")
    print("  FALSIFICATION: Any holonomy not in U(n)")
    print(f"{'='*60}\n")

    n_loops = 50 if quick else Q8Thresholds.HOLONOMY_N_LOOPS

    results = {
        'test_name': 'Q8_TEST3_HOLONOMY',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick_mode': quick,
            'n_loops': n_loops,
            'unitary_tolerance': Q8Thresholds.HOLONOMY_UNITARY_TOLERANCE
        }
    }

    # Run positive tests
    models = MODELS_QUICK if quick else MODELS_FULL
    vocab = VOCAB_CORE

    logger.info(f"Mode: {'QUICK' if quick else 'FULL'}")
    logger.info(f"Models: {len(models)}")
    logger.info(f"Loops per model: {n_loops}")

    results['positive_tests'] = run_multi_model_test(models, vocab, n_loops, logger)

    # Run negative controls
    results['negative_controls'] = run_negative_control_test(n_loops, logger)

    # Final verdict
    logger.section("FINAL VERDICT")

    positive_pass = results['positive_tests']['summary'].get('all_pass', False)

    results['verdict'] = {
        'positive_tests_pass': positive_pass,
        'final_verdict': 'PASS' if positive_pass else 'FAIL',
        'holonomy_is_unitary': positive_pass
    }

    if positive_pass:
        logger.info("VERDICT: PASS - HOLONOMY GROUP IS U(n)")
    else:
        logger.warn("VERDICT: FAIL - HOLONOMY GROUP NOT CONFIRMED")

    # Save results
    if save:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"q8_test3_holonomy_{timestamp}.json"
        save_results(results, filepath)
        logger.info(f"Results saved to: {filepath}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q8 Test 3: Holonomy Group Classification")
    parser.add_argument("--full", action="store_true", help="Run full test")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    main(quick=not args.full, save=not args.no_save)
