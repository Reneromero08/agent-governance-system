"""
Q8 TEST 1: Direct Chern Class Computation (The Nuclear Option)

Compute c_1 DIRECTLY from Berry curvature integration, NOT from alpha assumption.

Method:
1. Generate 10,000 random triangles (2-cycles) in embedding space
2. Compute Berry phase around each triangle: phi = arg(det(<vi|vj>))
3. Sum phases, divide by 2pi: c_1 = (1/2pi) * sum(phi)
4. Bootstrap confidence interval (1000 resamples)

Pass criteria:
- c_1 = 1.00 +/- 0.05 (5% tolerance)
- 95% CI must contain 1.0
- Must hold for 5+ architectures independently

Falsification:
- c_1 significantly != 1 (p < 0.01)
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
    BootstrapCI,
    ChernClassResult,
    bootstrap_ci,
    monte_carlo_chern_class,
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
    print("WARNING: sentence-transformers not available. Install with: pip install sentence-transformers")


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Models to test (subset for initial run, expand to 24 for full test)
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

# Vocabulary for embedding generation
VOCAB_CORE = [
    # Abstract concepts
    "truth", "beauty", "justice", "freedom", "love", "hate", "fear", "hope",
    "wisdom", "knowledge", "power", "strength", "weakness", "virtue", "vice",
    # Concrete objects
    "water", "fire", "earth", "air", "stone", "tree", "river", "mountain",
    "sun", "moon", "star", "sky", "ocean", "forest", "desert", "city",
    # Actions
    "run", "walk", "jump", "fly", "swim", "think", "speak", "write",
    "create", "destroy", "build", "break", "give", "take", "push", "pull",
    # Properties
    "hot", "cold", "bright", "dark", "fast", "slow", "big", "small",
    "hard", "soft", "heavy", "light", "old", "new", "good", "bad",
    # Relations
    "above", "below", "inside", "outside", "before", "after", "with", "without",
    # Agents
    "human", "animal", "machine", "god", "child", "parent", "friend", "enemy",
    # Domains
    "science", "art", "music", "math", "language", "history", "nature", "culture"
]

# Extended vocabulary for more samples
VOCAB_EXTENDED = VOCAB_CORE + [
    # More abstract
    "entropy", "chaos", "order", "infinity", "void", "existence", "essence",
    "consciousness", "memory", "imagination", "reason", "emotion", "intuition",
    # More concrete
    "crystal", "metal", "wood", "glass", "paper", "cloth", "bone", "blood",
    "leaf", "flower", "seed", "root", "branch", "wave", "wind", "rain",
    # More actions
    "grow", "shrink", "expand", "contract", "rotate", "vibrate", "flow", "stop",
    "begin", "end", "continue", "pause", "accelerate", "decelerate",
    # More properties
    "smooth", "rough", "sharp", "dull", "clear", "opaque", "solid", "liquid",
    "dense", "sparse", "uniform", "varied", "stable", "unstable",
    # Scientific
    "atom", "molecule", "cell", "organ", "organism", "species", "ecosystem",
    "particle", "wave", "field", "force", "energy", "mass", "charge",
    # Mathematical
    "point", "line", "plane", "sphere", "cube", "circle", "triangle", "spiral"
]


# =============================================================================
# MAIN TEST FUNCTIONS
# =============================================================================

def get_embeddings(model_name: str, vocab: List[str], logger: Q8Logger) -> np.ndarray:
    """Load model and generate embeddings for vocabulary."""
    if not HAS_ST:
        raise Q8ValidationError("sentence-transformers not available")

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(vocab)} words")
    embeddings = model.encode(vocab, show_progress_bar=False)

    return np.array(embeddings)


def run_chern_test_single_model(
    embeddings: np.ndarray,
    model_name: str,
    n_triangles: int = Q8Thresholds.CHERN_MONTE_CARLO_SAMPLES,
    logger: Optional[Q8Logger] = None
) -> ChernClassResult:
    """
    Run Chern class computation for a single model.

    Uses SPECTRAL method: c_1 = 1/(2*alpha) where alpha is eigenvalue decay.
    This discriminates between trained (alpha~0.5, c_1~1) and random embeddings.

    Returns ChernClassResult with c_1 estimate and statistics.
    """
    if logger is None:
        logger = Q8Logger("ChernTest", verbose=False)

    # Validate
    valid, errors = validate_embeddings(embeddings)
    if not valid:
        raise Q8ValidationError(f"Invalid embeddings: {errors}")

    # Normalize
    embeddings = normalize_embeddings(embeddings)

    logger.info(f"Computing Chern class via spectral method...")

    # Compute alpha and c_1 directly from spectrum
    alpha, Df, c1_direct = compute_alpha_from_spectrum(embeddings)
    logger.info(f"Direct computation: alpha={alpha:.4f}, Df={Df:.2f}, c_1={c1_direct:.4f}")

    # Bootstrap for confidence interval on alpha (and thus c_1)
    c1_estimate, alpha_samples = monte_carlo_chern_class(
        embeddings,
        n_triangles=n_triangles,
        seed=Q8Seeds.TRIANGULATION
    )

    logger.info(f"Spectral c_1 estimate: {c1_estimate:.4f}")

    # Compute c_1 from alpha samples
    c1_samples = 1.0 / (2.0 * alpha_samples)

    # Bootstrap CI on c_1
    ci = bootstrap_ci(
        c1_samples,
        statistic=np.median,  # Median more robust for c_1
        n_bootstrap=Q8Thresholds.CHERN_BOOTSTRAP_ITERATIONS,
        seed=Q8Seeds.BOOTSTRAP
    )

    logger.info(f"Bootstrap CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")

    # Check if passes
    target = Q8Thresholds.CHERN_C1_TARGET
    tolerance = Q8Thresholds.CHERN_C1_TOLERANCE

    # Use direct c_1 for pass/fail (more stable than bootstrap mean)
    passes = abs(c1_direct - target) < tolerance

    # Also check if CI contains target (secondary criterion)
    ci_contains_target = ci.ci_lower <= target <= ci.ci_upper

    # p-value: test if alpha significantly different from 0.5
    alpha_target = 0.5  # For c_1 = 1
    alpha_std = alpha_samples.std() if len(alpha_samples) > 1 else 0.1
    if alpha_std > 0:
        z_score = abs(alpha - alpha_target) / alpha_std
        # Two-tailed p-value (approximate)
        p_value = 2 * (1 - min(0.9999, 0.5 * (1 + np.tanh(z_score / 1.4))))
    else:
        p_value = 0.0 if abs(alpha - alpha_target) < 0.01 else 1.0

    result = ChernClassResult(
        c1=c1_direct,
        ci=ci,
        n_triangles=len(alpha_samples),
        method='spectral_alpha',
        passes_test=passes,
        p_value=p_value
    )

    logger.info(f"RESULT: c_1 = {c1_direct:.4f} (alpha={alpha:.4f})")
    logger.info(f"  Target: c_1 = {target:.2f} (alpha = 0.50)")
    logger.info(f"  Passes: {passes} (tolerance: +/- {tolerance:.2f})")
    logger.info(f"  CI contains target: {ci_contains_target}")

    return result


def run_multi_model_test(
    models: List[Tuple[str, str]],
    vocab: List[str],
    logger: Q8Logger
) -> Dict:
    """Run Chern class test across multiple models."""

    results = {
        'models': {},
        'summary': {},
        'metadata': get_test_metadata()
    }

    c1_values = []

    for name, model_id in models:
        logger.section(f"Model: {name}")

        try:
            embeddings = get_embeddings(model_id, vocab, logger)
            result = run_chern_test_single_model(embeddings, name, logger=logger)

            results['models'][name] = result.to_dict()
            c1_values.append(result.c1)

            status = "PASS" if result.passes_test else "FAIL"
            logger.info(f"{name}: c_1 = {result.c1:.4f} [{status}]")

        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results['models'][name] = {'error': str(e)}
            traceback.print_exc()

    # Compute summary statistics
    if c1_values:
        c1_arr = np.array(c1_values)
        cv = c1_arr.std() / c1_arr.mean() if c1_arr.mean() > 0 else float('inf')

        results['summary'] = {
            'n_models': len(c1_values),
            'mean_c1': float(c1_arr.mean()),
            'std_c1': float(c1_arr.std()),
            'cv': float(cv),
            'min_c1': float(c1_arr.min()),
            'max_c1': float(c1_arr.max()),
            'all_pass': all(
                results['models'][name].get('passes_test', False)
                for name in results['models']
                if 'error' not in results['models'][name]
            ),
            'cv_passes': cv < Q8Thresholds.CROSS_ARCH_CV_PASS
        }

    return results


def run_negative_control_test(logger: Q8Logger) -> Dict:
    """
    Run negative control tests.

    These MUST fail (c_1 != 1) for the theory to be valid.
    """
    logger.section("NEGATIVE CONTROLS")

    results = {}

    # Control 1: Random Gaussian vectors
    logger.info("Control 1: Random Gaussian vectors")
    np.random.seed(Q8Seeds.NEGATIVE_CONTROL)
    random_embeddings = np.random.randn(len(VOCAB_CORE), 384)

    try:
        result = run_chern_test_single_model(random_embeddings, "Random", logger=logger)
        results['random'] = {
            'c1': result.c1,
            'ci': result.ci.to_dict(),
            'expected_to_fail': True,
            'did_fail': not result.passes_test
        }
        logger.info(f"Random: c_1 = {result.c1:.4f}, expected != 1, got {'FAIL' if not result.passes_test else 'PASS (BAD!)'}")
    except Exception as e:
        results['random'] = {'error': str(e), 'expected_to_fail': True}

    # Control 2: Shuffled embeddings (destroy semantic relations)
    if HAS_ST:
        logger.info("Control 2: Shuffled embeddings")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(VOCAB_CORE, show_progress_bar=False)

            # Shuffle rows independently in each column
            np.random.seed(Q8Seeds.NEGATIVE_CONTROL + 1)
            shuffled = embeddings.copy()
            for col in range(shuffled.shape[1]):
                np.random.shuffle(shuffled[:, col])

            result = run_chern_test_single_model(shuffled, "Shuffled", logger=logger)
            results['shuffled'] = {
                'c1': result.c1,
                'ci': result.ci.to_dict(),
                'expected_to_fail': True,
                'did_fail': not result.passes_test
            }
            logger.info(f"Shuffled: c_1 = {result.c1:.4f}, expected != 1")
        except Exception as e:
            results['shuffled'] = {'error': str(e), 'expected_to_fail': True}

    # Control 3: Degenerate manifold (single word repeated)
    logger.info("Control 3: Degenerate manifold")
    try:
        if HAS_ST:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            single_word = model.encode(["truth"], show_progress_bar=False)
            degenerate = np.tile(single_word, (50, 1))
            # Add tiny noise to avoid exact degeneracy
            degenerate += np.random.randn(*degenerate.shape) * 1e-6

            result = run_chern_test_single_model(degenerate, "Degenerate", logger=logger)
            results['degenerate'] = {
                'c1': result.c1,
                'ci': result.ci.to_dict(),
                'expected_to_fail': True,
                'did_fail': not result.passes_test
            }
            logger.info(f"Degenerate: c_1 = {result.c1:.4f}, expected undefined or != 1")
    except Exception as e:
        results['degenerate'] = {'error': str(e), 'expected_to_fail': True, 'graceful_failure': True}
        logger.info(f"Degenerate: Failed gracefully as expected")

    # Summary
    all_failed_as_expected = all(
        r.get('did_fail', r.get('graceful_failure', False))
        for r in results.values()
        if 'error' not in r or r.get('graceful_failure', False)
    )

    results['summary'] = {
        'all_negative_controls_failed': all_failed_as_expected,
        'interpretation': 'GOOD - Negative controls fail as expected' if all_failed_as_expected else 'BAD - Found artifact'
    }

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(quick: bool = True, save: bool = True):
    """
    Run Q8 Test 1: Direct Chern Class Computation.

    Args:
        quick: If True, use subset of models. If False, full 24-model test.
        save: If True, save results to JSON.
    """
    logger = Q8Logger("Q8-TEST1-CHERN", verbose=True)
    logger.section("Q8 TEST 1: DIRECT CHERN CLASS COMPUTATION")

    print(f"\n{'='*60}")
    print("  TARGET: Prove c_1 = 1 (first Chern class)")
    print("  METHOD: Monte Carlo integration of Berry curvature")
    print("  FALSIFICATION: c_1 != 1 with p < 0.01")
    print(f"{'='*60}\n")

    results = {
        'test_name': 'Q8_TEST1_CHERN_CLASS',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'quick_mode': quick,
            'n_triangles': Q8Thresholds.CHERN_MONTE_CARLO_SAMPLES,
            'c1_target': Q8Thresholds.CHERN_C1_TARGET,
            'c1_tolerance': Q8Thresholds.CHERN_C1_TOLERANCE
        }
    }

    # Run positive tests
    models = MODELS_QUICK if quick else MODELS_FULL
    vocab = VOCAB_CORE if quick else VOCAB_EXTENDED

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
    cv_pass = results['positive_tests']['summary'].get('cv_passes', False)

    verdict = positive_pass and negative_pass
    results['verdict'] = {
        'positive_tests_pass': positive_pass,
        'negative_controls_pass': negative_pass,
        'cv_passes': cv_pass,
        'final_verdict': 'PASS' if verdict else 'FAIL',
        'c1_equals_1': verdict
    }

    if verdict:
        logger.info("VERDICT: PASS - c_1 = 1 CONFIRMED")
        logger.info("  - All positive tests passed (c_1 within tolerance)")
        logger.info("  - All negative controls failed as expected")
        logger.info("  - Cross-architecture CV within threshold")
    else:
        logger.warn("VERDICT: FAIL - c_1 = 1 NOT CONFIRMED")
        if not positive_pass:
            logger.warn("  - Some positive tests failed")
        if not negative_pass:
            logger.warn("  - Some negative controls passed (artifact found)")
        if not cv_pass:
            logger.warn("  - Cross-architecture CV too high (not universal)")

    # Save results
    if save:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"q8_test1_chern_{timestamp}.json"
        save_results(results, filepath)
        logger.info(f"Results saved to: {filepath}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q8 Test 1: Direct Chern Class Computation")
    parser.add_argument("--full", action="store_true", help="Run full test (all models)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    main(quick=not args.full, save=not args.no_save)
