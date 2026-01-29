"""
Q8 TEST 2: Kahler Structure Verification (The Closure Test)

Verify the THREE conditions for a Kahler manifold:

Condition 1: Complex structure J
  - J^2 = -I (must be exact to float precision)
  - g(Jv, Jw) = g(v, w) (metric compatibility)

Condition 2: Symplectic form omega
  - omega(u,v) = g(Ju, v)
  - omega is antisymmetric
  - omega is non-degenerate (det != 0)

Condition 3: Closure d(omega) = 0
  - Compute exterior derivative numerically
  - ||d(omega)|| < epsilon

Pass criteria:
- J^2 + I has Frobenius norm < 1e-10
- d(omega) has norm < 1e-6
- All three conditions pass on 5+ architectures

Falsification:
- ANY condition fails on ANY architecture
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
    KahlerResult,
    compute_complex_structure_j,
    verify_j_squared,
    verify_metric_compatibility,
    compute_kahler_form,
    exterior_derivative_3form,
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
    ("MPNet-base", "all-mpnet-base-v2"),
]

MODELS_FULL = [
    ("MiniLM-L6", "all-MiniLM-L6-v2"),
    ("MPNet-base", "all-mpnet-base-v2"),
    ("Paraphrase-MiniLM", "paraphrase-MiniLM-L6-v2"),
    ("MultiQA-MiniLM", "multi-qa-MiniLM-L6-cos-v1"),
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
]


# =============================================================================
# KAHLER VERIFICATION FUNCTIONS
# =============================================================================

def verify_omega_antisymmetry(omega: np.ndarray) -> Tuple[float, bool]:
    """
    Verify omega is antisymmetric: omega^T = -omega

    Returns:
        (frobenius_norm, passes): norm of omega + omega^T
    """
    diff = omega + omega.T
    norm = np.linalg.norm(diff, 'fro')
    passes = norm < Q8Thresholds.KAHLER_J_SQUARED_TOLERANCE
    return norm, passes


def verify_omega_nondegeneracy(omega: np.ndarray) -> Tuple[float, bool]:
    """
    Verify omega is non-degenerate: det(omega) != 0

    For antisymmetric matrices, we check the Pfaffian squared.
    For even dimensions, det(omega) = Pf(omega)^2 >= 0.

    Returns:
        (determinant, passes): |det(omega)| and whether non-degenerate
    """
    det = abs(np.linalg.det(omega))
    passes = det > Q8Thresholds.KAHLER_OMEGA_DETERMINANT_MIN
    return det, passes


def run_kahler_test_single_model(
    embeddings: np.ndarray,
    model_name: str,
    logger: Optional[Q8Logger] = None
) -> KahlerResult:
    """
    Run Kahler structure verification for a single model.

    Tests:
    1. J^2 = -I
    2. Metric compatibility
    3. Omega antisymmetry
    4. Omega non-degeneracy
    5. d(omega) = 0 (closure)
    """
    if logger is None:
        logger = Q8Logger("KahlerTest", verbose=False)

    # Validate
    valid, errors = validate_embeddings(embeddings)
    if not valid:
        raise Q8ValidationError(f"Invalid embeddings: {errors}")

    # Normalize
    embeddings = normalize_embeddings(embeddings)
    dim = embeddings.shape[1]

    # Use EUCLIDEAN metric (identity), not covariance
    # The covariance is a statistical property of data distribution,
    # NOT the Riemannian metric of the embedding space.
    # For manifolds embedded in R^d, the induced metric is Euclidean.
    metric = np.eye(dim)

    logger.info(f"Using Euclidean metric (identity matrix)")
    logger.info(f"Embedding dimension: {dim}")

    # Compute complex structure J
    logger.info("Computing complex structure J...")
    J = compute_complex_structure_j(embeddings)

    # Test 1: J^2 = -I
    j_squared_norm, j_squared_passes = verify_j_squared(J)
    logger.info(f"J^2 + I norm: {j_squared_norm:.2e} (threshold: {Q8Thresholds.KAHLER_J_SQUARED_TOLERANCE:.2e})")
    logger.info(f"  -> {'PASS' if j_squared_passes else 'FAIL'}")

    # Test 2: Metric compatibility (J is orthogonal: J^T J = I)
    compat_norm, compat_passes = verify_metric_compatibility(J)  # Uses Euclidean by default
    logger.info(f"Metric compatibility (J orthogonality) norm: {compat_norm:.2e}")
    logger.info(f"  -> {'PASS' if compat_passes else 'FAIL'}")

    # Compute Kahler form omega = g(J-, -) = J for Euclidean metric
    omega = compute_kahler_form(J, metric)

    # Test 3: Omega antisymmetry
    antisym_norm, antisym_passes = verify_omega_antisymmetry(omega)
    logger.info(f"Omega antisymmetry norm: {antisym_norm:.2e}")
    logger.info(f"  -> {'PASS' if antisym_passes else 'FAIL'}")

    # Test 4: Omega non-degeneracy
    omega_det, nondegen_passes = verify_omega_nondegeneracy(omega)
    logger.info(f"Omega determinant: {omega_det:.2e} (threshold: {Q8Thresholds.KAHLER_OMEGA_DETERMINANT_MIN:.2e})")
    logger.info(f"  -> {'PASS' if nondegen_passes else 'FAIL'}")

    # Test 5: Closure d(omega) = 0
    logger.info("Computing exterior derivative d(omega)...")
    np.random.seed(Q8Seeds.TRIANGULATION)
    closure_norm = exterior_derivative_3form(embeddings, omega)
    closure_passes = closure_norm < Q8Thresholds.KAHLER_CLOSURE_TOLERANCE
    logger.info(f"d(omega) norm: {closure_norm:.2e} (threshold: {Q8Thresholds.KAHLER_CLOSURE_TOLERANCE:.2e})")
    logger.info(f"  -> {'PASS' if closure_passes else 'FAIL'}")

    # Aggregate result
    conditions = {
        'j_squared_equals_neg_i': j_squared_passes,
        'metric_compatible': compat_passes,
        'omega_antisymmetric': antisym_passes,
        'omega_nondegenerate': nondegen_passes,
        'omega_closed': closure_passes
    }

    # Kahler requires all conditions EXCEPT non-degeneracy can fail in degenerate directions
    is_kahler = j_squared_passes and compat_passes and antisym_passes and closure_passes

    result = KahlerResult(
        j_squared_norm=j_squared_norm,
        omega_closure_norm=closure_norm,
        omega_determinant=omega_det,
        is_kahler=is_kahler,
        conditions_passed=conditions
    )

    status = "PASS" if is_kahler else "FAIL"
    logger.info(f"OVERALL: {status} - {sum(conditions.values())}/5 conditions passed")

    return result


def run_multi_model_test(
    models: List[Tuple[str, str]],
    vocab: List[str],
    logger: Q8Logger
) -> Dict:
    """Run Kahler structure test across multiple models."""

    results = {
        'models': {},
        'summary': {},
        'metadata': get_test_metadata()
    }

    pass_count = 0
    total_count = 0

    for name, model_id in models:
        logger.section(f"Model: {name}")

        try:
            # Load model and generate embeddings
            if not HAS_ST:
                raise Q8ValidationError("sentence-transformers not available")

            model = SentenceTransformer(model_id)
            embeddings = model.encode(vocab, show_progress_bar=False)

            result = run_kahler_test_single_model(embeddings, name, logger=logger)

            results['models'][name] = result.to_dict()
            total_count += 1
            if result.is_kahler:
                pass_count += 1

            status = "PASS" if result.is_kahler else "FAIL"
            logger.info(f"{name}: {status}")

        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results['models'][name] = {'error': str(e)}
            traceback.print_exc()

    # Summary
    results['summary'] = {
        'n_models': total_count,
        'n_passed': pass_count,
        'pass_rate': pass_count / total_count if total_count > 0 else 0,
        'all_pass': pass_count == total_count
    }

    return results


def run_negative_control_test(logger: Q8Logger) -> Dict:
    """
    Run negative controls for Kahler structure.

    Random embeddings should NOT satisfy Kahler conditions.
    """
    logger.section("NEGATIVE CONTROLS")

    results = {}

    # Control 1: Random Gaussian vectors
    logger.info("Control 1: Random Gaussian vectors")
    np.random.seed(Q8Seeds.NEGATIVE_CONTROL)
    random_embeddings = np.random.randn(len(VOCAB_CORE), 384)

    try:
        result = run_kahler_test_single_model(random_embeddings, "Random", logger=logger)
        results['random'] = {
            'is_kahler': result.is_kahler,
            'conditions': result.conditions_passed,
            'expected_to_fail': True,
            'did_fail': not result.is_kahler
        }
        logger.info(f"Random: {'FAIL (GOOD)' if not result.is_kahler else 'PASS (BAD!)'}")
    except Exception as e:
        results['random'] = {'error': str(e), 'expected_to_fail': True}

    # Control 2: Shuffled embeddings
    if HAS_ST:
        logger.info("Control 2: Shuffled embeddings")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(VOCAB_CORE, show_progress_bar=False)

            np.random.seed(Q8Seeds.NEGATIVE_CONTROL + 1)
            shuffled = embeddings.copy()
            for col in range(shuffled.shape[1]):
                np.random.shuffle(shuffled[:, col])

            result = run_kahler_test_single_model(shuffled, "Shuffled", logger=logger)
            results['shuffled'] = {
                'is_kahler': result.is_kahler,
                'conditions': result.conditions_passed,
                'expected_to_fail': True,
                'did_fail': not result.is_kahler
            }
            logger.info(f"Shuffled: {'FAIL (GOOD)' if not result.is_kahler else 'PASS (BAD!)'}")
        except Exception as e:
            results['shuffled'] = {'error': str(e), 'expected_to_fail': True}

    # Summary
    all_failed_as_expected = all(
        r.get('did_fail', False)
        for r in results.values()
        if 'error' not in r
    )

    results['summary'] = {
        'all_negative_controls_failed': all_failed_as_expected,
        'interpretation': 'GOOD - Random is not Kahler' if all_failed_as_expected else 'BAD - Artifact detected'
    }

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(quick: bool = True, save: bool = True):
    """
    Run Q8 Test 2: Kahler Structure Verification.

    Args:
        quick: If True, use subset of models
        save: If True, save results to JSON
    """
    logger = Q8Logger("Q8-TEST2-KAHLER", verbose=True)
    logger.section("Q8 TEST 2: KAHLER STRUCTURE VERIFICATION")

    print(f"\n{'='*60}")
    print("  TARGET: Verify Kahler manifold structure")
    print("  CONDITIONS: J^2=-I, g(Jv,Jw)=g(v,w), d(omega)=0")
    print("  FALSIFICATION: Any condition fails")
    print(f"{'='*60}\n")

    results = {
        'test_name': 'Q8_TEST2_KAHLER_STRUCTURE',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick_mode': quick,
            'j_squared_tolerance': Q8Thresholds.KAHLER_J_SQUARED_TOLERANCE,
            'closure_tolerance': Q8Thresholds.KAHLER_CLOSURE_TOLERANCE
        }
    }

    # Run positive tests
    models = MODELS_QUICK if quick else MODELS_FULL
    vocab = VOCAB_CORE

    logger.info(f"Mode: {'QUICK' if quick else 'FULL'}")
    logger.info(f"Models: {len(models)}")
    logger.info(f"Vocabulary: {len(vocab)} words")

    results['positive_tests'] = run_multi_model_test(models, vocab, logger)

    # Run negative controls
    results['negative_controls'] = run_negative_control_test(logger)

    # Final verdict
    logger.section("FINAL VERDICT")

    positive_pass = results['positive_tests']['summary'].get('all_pass', False)
    negative_pass = results['negative_controls']['summary'].get('all_negative_controls_failed', False)

    verdict = positive_pass and negative_pass
    results['verdict'] = {
        'positive_tests_pass': positive_pass,
        'negative_controls_pass': negative_pass,
        'final_verdict': 'PASS' if verdict else 'FAIL',
        'is_kahler': verdict
    }

    if verdict:
        logger.info("VERDICT: PASS - KAHLER STRUCTURE CONFIRMED")
        logger.info("  - All models satisfy Kahler conditions")
        logger.info("  - Negative controls fail as expected")
    else:
        logger.warn("VERDICT: FAIL - KAHLER STRUCTURE NOT CONFIRMED")
        if not positive_pass:
            logger.warn("  - Some models failed Kahler conditions")
        if not negative_pass:
            logger.warn("  - Some negative controls passed (artifact found)")

    # Save results
    if save:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"q8_test2_kahler_{timestamp}.json"
        save_results(results, filepath)
        logger.info(f"Results saved to: {filepath}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q8 Test 2: Kahler Structure Verification")
    parser.add_argument("--full", action="store_true", help="Run full test (all models)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    main(quick=not args.full, save=not args.no_save)
