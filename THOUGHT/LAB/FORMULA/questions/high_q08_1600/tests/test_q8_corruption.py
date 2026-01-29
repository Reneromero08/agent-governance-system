"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DEPRECATED: DO NOT USE THIS TEST
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Use test_q8_topological_invariance.py instead.

METHODOLOGY ISSUES (why this test was invalidated):

1. WRONG PERTURBATION TYPE: Adding Gaussian noise DESTROYS the manifold
   structure, it does not DEFORM it. This tests noise robustness, not
   topological invariance.

2. WRONG INTERPRETATION: Linear drift under noise is expected for ANY
   spectral measure. It's just linear algebra (eigenvalue perturbation),
   not topology. The test was misinterpreting this as "falsification".

3. WRONG HYPOTHESIS: The cross-model breakthrough at 50% corruption works
   via REDUNDANCY (200+ bits encoding ~3 bits of information), not via
   topological protection. The breakthrough paper explicitly states this.

Topological invariants are preserved under CONTINUOUS DEFORMATIONS of the
manifold (rotations, scaling, smooth warping), NOT under random noise.

The REVISED test (test_q8_topological_invariance.py):
- Tests rotation invariance (orthogonal transformations)
- Tests scaling invariance (uniform scaling)
- Tests smooth warping (continuous deformations)
- Tests cross-model invariance (different architectures = different coordinates)

All these preserve manifold structure and are valid topological tests.

Original (invalid) docstring follows for reference:
--------------------------------------------------------------------------------

Q8 TEST 4: 50% Corruption Stress Test (Topological Robustness)

If c_1 = 1 is TOPOLOGICAL, it must survive massive perturbation.

Inspired by CROSS_MODEL_BREAKTHROUGH: 100% accuracy at 50% corruption.

Method:
1. Take trained embeddings (c_1 measured at baseline)
2. Add Gaussian noise at levels: 0%, 10%, 25%, 50%, 75%, 90%
3. Re-measure c_1 at each level
4. Analyze stability and find critical transition

Pass criteria:
- c_1 remains within 10% of original up to 50% corruption
- Sharp transition at some critical corruption level
- Transition is universal across architectures

Falsification:
- c_1 drifts continuously with noise (not topological)
"""

import warnings
warnings.warn(
    "test_q8_corruption.py is DEPRECATED. Use test_q8_topological_invariance.py instead. "
    "This test has fundamental methodology issues (noise destroys manifolds, not deforms them).",
    DeprecationWarning,
    stacklevel=2
)

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

# Corruption levels to test
CORRUPTION_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]


# =============================================================================
# CORRUPTION FUNCTIONS
# =============================================================================

def add_gaussian_noise(
    embeddings: np.ndarray,
    corruption_level: float,
    seed: int = Q8Seeds.CORRUPTION
) -> np.ndarray:
    """
    Add Gaussian noise to embeddings.

    Args:
        embeddings: Original embeddings
        corruption_level: Fraction of signal energy to replace with noise
                         0.0 = no noise, 1.0 = pure noise
        seed: Random seed

    Returns:
        Corrupted embeddings
    """
    np.random.seed(seed)

    # Compute signal energy
    signal_std = embeddings.std()

    # Generate noise with same variance as signal
    noise = np.random.randn(*embeddings.shape) * signal_std

    # Mix: corrupted = sqrt(1-c)*signal + sqrt(c)*noise
    # This preserves total variance
    alpha = np.sqrt(1 - corruption_level)
    beta = np.sqrt(corruption_level)

    corrupted = alpha * embeddings + beta * noise

    return corrupted


def measure_c1_at_corruption(
    embeddings: np.ndarray,
    corruption_level: float,
    n_triangles: int = 1000,
    logger: Optional[Q8Logger] = None
) -> Tuple[float, float, float]:
    """
    Measure c_1 at a given corruption level.

    Uses spectral method: c_1 = 1/(2*alpha) where alpha is eigenvalue decay.

    Returns:
        (c1, ci_lower, ci_upper): Chern class and confidence interval
    """
    # Add noise
    corrupted = add_gaussian_noise(embeddings, corruption_level)

    # Use compute_alpha_from_spectrum for consistent c_1 measurement
    alpha, Df, c1_direct = compute_alpha_from_spectrum(corrupted)

    # Bootstrap CI via monte_carlo_chern_class
    c1_estimate, alpha_samples = monte_carlo_chern_class(
        corrupted,
        n_triangles=100,  # Reduced for speed in corruption test
        seed=Q8Seeds.TRIANGULATION + int(corruption_level * 1000)
    )

    # Compute CI from alpha samples
    if len(alpha_samples) > 1:
        c1_samples = 1.0 / (2.0 * alpha_samples)
        c1_samples = c1_samples[np.isfinite(c1_samples)]
        if len(c1_samples) > 0:
            ci_lower = np.percentile(c1_samples, 2.5)
            ci_upper = np.percentile(c1_samples, 97.5)
        else:
            ci_lower = c1_direct * 0.9
            ci_upper = c1_direct * 1.1
    else:
        ci_lower = c1_direct * 0.9
        ci_upper = c1_direct * 1.1

    return c1_direct, ci_lower, ci_upper


def run_corruption_test_single_model(
    embeddings: np.ndarray,
    model_name: str,
    corruption_levels: List[float] = CORRUPTION_LEVELS,
    logger: Optional[Q8Logger] = None
) -> Dict:
    """
    Run corruption stress test for a single model.

    Measures c_1 at each corruption level and analyzes stability.
    """
    if logger is None:
        logger = Q8Logger("CorruptionTest", verbose=False)

    # Validate
    valid, errors = validate_embeddings(embeddings)
    if not valid:
        raise Q8ValidationError(f"Invalid embeddings: {errors}")

    logger.info(f"Testing {len(corruption_levels)} corruption levels...")

    results = {
        'levels': [],
        'c1_values': [],
        'ci_lower': [],
        'ci_upper': [],
        'relative_change': []
    }

    baseline_c1 = None

    for level in corruption_levels:
        logger.info(f"  Corruption level: {level*100:.0f}%")

        c1, ci_lower, ci_upper = measure_c1_at_corruption(
            embeddings, level, n_triangles=2000, logger=logger
        )

        if baseline_c1 is None:
            baseline_c1 = c1

        relative_change = abs(c1 - baseline_c1) / baseline_c1 if baseline_c1 > 0 else 0

        results['levels'].append(level)
        results['c1_values'].append(c1)
        results['ci_lower'].append(ci_lower)
        results['ci_upper'].append(ci_upper)
        results['relative_change'].append(relative_change)

        logger.info(f"    c_1 = {c1:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], change = {relative_change*100:.1f}%")

    # Analyze stability
    c1_at_50 = None
    for i, level in enumerate(results['levels']):
        if level == 0.50:
            c1_at_50 = results['c1_values'][i]
            change_at_50 = results['relative_change'][i]
            break

    # Find critical transition (where change exceeds threshold)
    critical_level = None
    threshold = Q8Thresholds.CORRUPTION_STABILITY_THRESHOLD

    for i, change in enumerate(results['relative_change']):
        if change > threshold:
            critical_level = results['levels'][i]
            break

    results['analysis'] = {
        'baseline_c1': baseline_c1,
        'c1_at_50_percent': c1_at_50,
        'change_at_50_percent': change_at_50 if c1_at_50 else None,
        'stable_up_to_50': change_at_50 < threshold if c1_at_50 else False,
        'critical_transition_level': critical_level,
        'passes_test': change_at_50 < threshold if c1_at_50 else False
    }

    status = "PASS" if results['analysis']['passes_test'] else "FAIL"
    logger.info(f"RESULT: {status}")
    logger.info(f"  Baseline c_1: {baseline_c1:.4f}")
    logger.info(f"  c_1 at 50%: {c1_at_50:.4f}" if c1_at_50 else "  c_1 at 50%: N/A")
    logger.info(f"  Change at 50%: {change_at_50*100:.1f}%" if c1_at_50 else "  Change: N/A")
    logger.info(f"  Critical transition: {critical_level*100:.0f}%" if critical_level else "  No critical transition found")

    return results


def run_multi_model_test(
    models: List[Tuple[str, str]],
    vocab: List[str],
    logger: Q8Logger
) -> Dict:
    """Run corruption test across multiple models."""

    results = {
        'models': {},
        'summary': {},
        'metadata': get_test_metadata()
    }

    pass_count = 0
    total_count = 0
    critical_levels = []

    for name, model_id in models:
        logger.section(f"Model: {name}")

        try:
            if not HAS_ST:
                raise Q8ValidationError("sentence-transformers not available")

            model = SentenceTransformer(model_id)
            embeddings = model.encode(vocab, show_progress_bar=False)

            result = run_corruption_test_single_model(embeddings, name, logger=logger)

            results['models'][name] = result
            total_count += 1

            if result['analysis']['passes_test']:
                pass_count += 1

            if result['analysis']['critical_transition_level']:
                critical_levels.append(result['analysis']['critical_transition_level'])

            status = "PASS" if result['analysis']['passes_test'] else "FAIL"
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
        'all_pass': pass_count == total_count,
        'mean_critical_level': float(np.mean(critical_levels)) if critical_levels else None,
        'std_critical_level': float(np.std(critical_levels)) if len(critical_levels) > 1 else None
    }

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(save: bool = True):
    """
    Run Q8 Test 4: Corruption Stress Test.
    """
    logger = Q8Logger("Q8-TEST4-CORRUPTION", verbose=True)
    logger.section("Q8 TEST 4: 50% CORRUPTION STRESS TEST")

    print(f"\n{'='*60}")
    print("  TARGET: c_1 survives 50% corruption (topological)")
    print("  METHOD: Add Gaussian noise, measure c_1 stability")
    print("  FALSIFICATION: c_1 drifts continuously (not topological)")
    print(f"{'='*60}\n")

    results = {
        'test_name': 'Q8_TEST4_CORRUPTION',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'corruption_levels': CORRUPTION_LEVELS,
            'stability_threshold': Q8Thresholds.CORRUPTION_STABILITY_THRESHOLD
        }
    }

    # Run tests
    logger.info(f"Corruption levels: {[f'{l*100:.0f}%' for l in CORRUPTION_LEVELS]}")
    logger.info(f"Stability threshold: {Q8Thresholds.CORRUPTION_STABILITY_THRESHOLD*100:.0f}%")

    results['tests'] = run_multi_model_test(MODELS, VOCAB_CORE, logger)

    # Final verdict
    logger.section("FINAL VERDICT")

    all_pass = results['tests']['summary'].get('all_pass', False)

    results['verdict'] = {
        'all_pass': all_pass,
        'final_verdict': 'PASS' if all_pass else 'FAIL',
        'c1_is_topological': all_pass
    }

    if all_pass:
        logger.info("VERDICT: PASS - c_1 IS TOPOLOGICALLY ROBUST")
        logger.info("  - c_1 stable up to 50% corruption")
        logger.info("  - Sharp transition at critical level")
    else:
        logger.warn("VERDICT: FAIL - c_1 NOT TOPOLOGICALLY ROBUST")
        logger.warn("  - c_1 drifts continuously with noise")

    # Save results
    if save:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"q8_test4_corruption_{timestamp}.json"
        save_results(results, filepath)
        logger.info(f"Results saved to: {filepath}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q8 Test 4: Corruption Stress Test")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()

    main(save=not args.no_save)
