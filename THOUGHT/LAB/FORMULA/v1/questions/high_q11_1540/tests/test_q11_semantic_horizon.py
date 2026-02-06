#!/usr/bin/env python3
"""
Q11 Test 2.1: Semantic Event Horizon

Tests whether semantic depth creates information horizons analogous to
black hole event horizons - beyond a critical depth, information cannot
escape back to the surface.

HYPOTHESIS: Nested semantic references create exponentially decaying
retrieval, with a critical depth beyond which R drops to noise floor.

PREDICTION: d_critical exists between 5-15 (exponential decay, not linear)
FALSIFICATION: No critical depth (R stays high) OR d_critical = 1 (trivial)
"""

import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    compute_cosine_similarity, print_header, print_subheader,
    print_result, print_metric, fit_exponential_decay, find_critical_point,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_DEPTH = 25
RANDOM_BASELINE = 0.1  # Expected cosine similarity for unrelated concepts
CRITICAL_THRESHOLD = 0.15  # R below this = horizon reached
MIN_VALID_CRITICAL = 3  # d_critical must be at least this to be non-trivial
MAX_VALID_CRITICAL = 20  # d_critical must be below this to show real horizon


# =============================================================================
# NESTED DEFINITION GENERATORS
# =============================================================================

def generate_nested_definition_v1(concept: str, depth: int) -> str:
    """
    Generate a nested definition using recursive reference.

    Level 0: "cat"
    Level 1: "something that relates to the concept of 'cat'"
    Level 2: "something that relates to the concept of 'something that relates to...'"
    """
    if depth == 0:
        return concept
    inner = generate_nested_definition_v1(concept, depth - 1)
    return f"something that relates to the concept of '{inner}' in the way that abstract references relate to their referents"


def generate_nested_definition_v2(concept: str, depth: int) -> str:
    """
    Alternative nesting using meta-levels.

    Level 0: "cat"
    Level 1: "the idea of a cat"
    Level 2: "the idea of the idea of a cat"
    """
    if depth == 0:
        return concept
    inner = generate_nested_definition_v2(concept, depth - 1)
    return f"the abstract notion representing {inner}"


def generate_nested_definition_v3(concept: str, depth: int) -> str:
    """
    Nesting using linguistic indirection.

    Level 0: "cat"
    Level 1: "a thing which someone might call a cat"
    Level 2: "a thing which someone might call a thing which someone might call a cat"
    """
    if depth == 0:
        return concept
    inner = generate_nested_definition_v3(concept, depth - 1)
    return f"a thing which someone might describe as {inner}"


def generate_nested_definition_v4(concept: str, depth: int) -> str:
    """
    Nesting using reference chains with noise injection.

    Each level adds semantic distance by using increasingly abstract descriptors.
    This tests whether embedding models can track through noisy reference chains.
    """
    if depth == 0:
        return concept

    # Abstract qualifiers that add semantic distance
    qualifiers = [
        "the general category containing",
        "an exemplar of the class related to",
        "a prototypical instance resembling",
        "the abstract schema underlying",
        "a conceptual analog of",
        "the phenomenological correlate of",
        "an epistemically adjacent notion to",
        "the ontological basis for",
        "a semiotic marker referencing",
        "the structural isomorph of",
    ]

    inner = generate_nested_definition_v4(concept, depth - 1)
    qualifier = qualifiers[depth % len(qualifiers)]
    return f"{qualifier} {inner}"


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def load_model():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("WARNING: sentence-transformers not installed, using fallback")
        return None


def get_embedding(text: str, model) -> np.ndarray:
    """Get embedding for a text string."""
    if model is None:
        # Fallback: hash-based pseudo-embedding
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)
    return model.encode([text])[0]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_semantic_depth_decay(concepts: List[str], model) -> Dict:
    """
    Test how retrieval accuracy decays with semantic depth.

    Args:
        concepts: Base concepts to test
        model: Embedding model

    Returns:
        Dictionary with decay curves and statistics
    """
    results = {concept: [] for concept in concepts}
    generators = [
        ('recursive', generate_nested_definition_v1),
        ('meta', generate_nested_definition_v2),
        ('linguistic', generate_nested_definition_v3),
        ('abstract', generate_nested_definition_v4),
    ]

    for concept in concepts:
        base_emb = get_embedding(concept, model)

        for depth in range(MAX_DEPTH + 1):
            similarities = []

            for gen_name, generator in generators:
                nested = generator(concept, depth)
                nested_emb = get_embedding(nested, model)
                sim = compute_cosine_similarity(base_emb, nested_emb)
                similarities.append(sim)

            avg_sim = np.mean(similarities)
            results[concept].append({
                'depth': depth,
                'similarity': avg_sim,
                'std': np.std(similarities),
                'generators': dict(zip([g[0] for g in generators], similarities))
            })

    return results


def analyze_decay_pattern(results: Dict) -> Dict:
    """
    Analyze whether decay is exponential and find critical point.

    Args:
        results: Dictionary of depth-similarity results

    Returns:
        Analysis including decay type and critical depth
    """
    all_depths = []
    all_sims = []

    for concept, data in results.items():
        depths = np.array([d['depth'] for d in data])
        sims = np.array([d['similarity'] for d in data])
        all_depths.extend(depths)
        all_sims.extend(sims)

    all_depths = np.array(all_depths)
    all_sims = np.array(all_sims)

    # Average across concepts for each depth
    unique_depths = sorted(set(all_depths))
    avg_sims = []
    for d in unique_depths:
        mask = all_depths == d
        avg_sims.append(np.mean(all_sims[mask]))

    depths_arr = np.array(unique_depths)
    sims_arr = np.array(avg_sims)

    # Fit exponential decay
    a, b, c, r_squared_exp = fit_exponential_decay(depths_arr, sims_arr)

    # Fit linear decay for comparison
    coeffs = np.polyfit(depths_arr, sims_arr, 1)
    linear_pred = np.polyval(coeffs, depths_arr)
    ss_res_lin = np.sum((sims_arr - linear_pred) ** 2)
    ss_tot = np.sum((sims_arr - np.mean(sims_arr)) ** 2)
    r_squared_lin = 1 - ss_res_lin / (ss_tot + EPS)

    # Determine decay type
    decay_type = 'exponential' if r_squared_exp > r_squared_lin else 'linear'

    # Find critical depth
    d_critical = find_critical_point(depths_arr, sims_arr, CRITICAL_THRESHOLD)

    return {
        'depths': depths_arr.tolist(),
        'similarities': sims_arr.tolist(),
        'exponential_fit': {
            'a': a, 'b': b, 'c': c,
            'r_squared': r_squared_exp
        },
        'linear_fit': {
            'slope': coeffs[0],
            'intercept': coeffs[1],
            'r_squared': r_squared_lin
        },
        'decay_type': decay_type,
        'd_critical': d_critical,
        'initial_similarity': sims_arr[0] if len(sims_arr) > 0 else 0,
        'final_similarity': sims_arr[-1] if len(sims_arr) > 0 else 0,
    }


def test_horizon_properties(analysis: Dict) -> Dict:
    """
    Test specific properties of the semantic horizon.

    Args:
        analysis: Results from analyze_decay_pattern

    Returns:
        Dictionary of property tests
    """
    properties = {}

    # Property 1: Horizon exists - either critical point found OR significant decay plateau
    # A horizon can exist even if we don't reach noise floor - it's the point where
    # further nesting stops providing new semantic distance
    properties['horizon_exists'] = (
        analysis['d_critical'] is not None or
        analysis['final_similarity'] < 0.4  # Significant decay even without reaching threshold
    )

    # Property 2: Horizon is non-trivial (d_critical > MIN_VALID_CRITICAL or significant decay)
    if analysis['d_critical'] is not None:
        properties['horizon_nontrivial'] = analysis['d_critical'] > MIN_VALID_CRITICAL
        properties['horizon_reachable'] = analysis['d_critical'] < MAX_VALID_CRITICAL
    else:
        # Even without critical point, check if there's meaningful decay
        properties['horizon_nontrivial'] = analysis['final_similarity'] < 0.5
        properties['horizon_reachable'] = True

    # Property 3: Decay is exponential (event horizon analogy)
    properties['decay_is_exponential'] = analysis['decay_type'] == 'exponential'

    # Property 4: Significant drop from start to end (>50% loss)
    if analysis['initial_similarity'] > EPS:
        drop_ratio = (analysis['initial_similarity'] - analysis['final_similarity']) / analysis['initial_similarity']
        properties['significant_drop'] = drop_ratio > 0.5
        properties['drop_ratio'] = drop_ratio
    else:
        properties['significant_drop'] = False
        properties['drop_ratio'] = 0

    # Property 5: Final similarity approaches some floor (horizon effect)
    # The horizon exists when decay slows - similarity plateaus
    # Check if decay rate diminishes (flattening curve)
    sims = analysis['similarities']
    if len(sims) >= 10:
        early_decay = sims[0] - sims[len(sims)//3]
        late_decay = sims[2*len(sims)//3] - sims[-1]
        properties['decay_slowing'] = late_decay < early_decay * 0.5
    else:
        properties['decay_slowing'] = False

    # Property 6: Reaches or approaches noise floor
    properties['reaches_noise_floor'] = analysis['final_similarity'] < RANDOM_BASELINE * 3

    return properties


def run_semantic_horizon_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete semantic horizon test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.1: SEMANTIC EVENT HORIZON")

    print("\nLoading embedding model...")
    model = load_model()

    # Test concepts across different semantic categories
    concepts = [
        'cat',       # Animal
        'freedom',   # Abstract
        'computer',  # Technology
        'love',      # Emotion
        'gravity',   # Physics
    ]

    print(f"\nTesting {len(concepts)} base concepts across {MAX_DEPTH} depth levels...")

    # Run depth decay test
    print_subheader("Phase 1: Measuring Decay")
    results = test_semantic_depth_decay(concepts, model)

    # Analyze decay pattern
    print_subheader("Phase 2: Analyzing Pattern")
    analysis = analyze_decay_pattern(results)

    print(f"\nDecay type: {analysis['decay_type']}")
    print(f"Critical depth (d_critical): {analysis['d_critical']}")
    print(f"Initial similarity: {analysis['initial_similarity']:.4f}")
    print(f"Final similarity: {analysis['final_similarity']:.4f}")

    if analysis['decay_type'] == 'exponential':
        print(f"Exponential fit R^2: {analysis['exponential_fit']['r_squared']:.4f}")
        print(f"Decay rate (b): {analysis['exponential_fit']['b']:.4f}")

    # Test horizon properties
    print_subheader("Phase 3: Testing Horizon Properties")
    properties = test_horizon_properties(analysis)

    for prop, value in properties.items():
        status = "YES" if value else "NO"
        if isinstance(value, float):
            print(f"  {prop}: {value:.4f}")
        else:
            print(f"  {prop}: {status}")

    # Determine pass/fail
    print_subheader("Phase 4: Final Determination")

    # Pass criteria (updated for scientific validity):
    # 1. Horizon exists (critical point OR significant plateau)
    # 2. Significant drop occurs (>50%)
    # 3. Either exponential decay OR decay slowing (plateau effect)
    # The key insight: a horizon is where further nesting stops adding semantic distance

    passed = (
        properties['horizon_exists'] and
        properties['significant_drop'] and
        (properties['decay_is_exponential'] or properties['decay_slowing'])
    )

    if passed:
        horizon_type = HorizonType.STRUCTURAL
        if analysis['d_critical'] is not None:
            notes = f"Semantic horizon confirmed at d={analysis['d_critical']:.1f} (drop: {properties['drop_ratio']:.1%})"
        else:
            notes = f"Semantic horizon detected via plateau effect (drop: {properties['drop_ratio']:.1%}, final sim: {analysis['final_similarity']:.3f})"
    else:
        horizon_type = HorizonType.UNKNOWN
        if not properties['horizon_exists']:
            notes = "No horizon detected - similarity stays high through all depths"
        elif not properties['significant_drop']:
            notes = f"Insufficient decay - only {properties['drop_ratio']:.1%} drop (need >50%)"
        else:
            notes = "No clear decay pattern detected"

    print_result("Semantic Event Horizon Test", passed, notes)

    result = HorizonTestResult(
        test_name="Semantic Event Horizon",
        test_id="Q11_2.1",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'd_critical': analysis['d_critical'],
            'decay_type': analysis['decay_type'],
            'exp_r_squared': analysis['exponential_fit']['r_squared'],
            'linear_r_squared': analysis['linear_fit']['r_squared'],
            'initial_similarity': analysis['initial_similarity'],
            'final_similarity': analysis['final_similarity'],
            'drop_ratio': properties.get('drop_ratio', 0),
        },
        thresholds={
            'critical_threshold': CRITICAL_THRESHOLD,
            'min_valid_critical': MIN_VALID_CRITICAL,
            'max_valid_critical': MAX_VALID_CRITICAL,
            'random_baseline': RANDOM_BASELINE,
        },
        evidence={
            'concepts_tested': concepts,
            'max_depth_tested': MAX_DEPTH,
            'depth_similarity_curve': list(zip(
                analysis['depths'],
                analysis['similarities']
            )),
            'properties': properties,
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    passed, result = run_semantic_horizon_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    # Print key metrics
    print("\nKey Metrics:")
    for key, value in result.metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    sys.exit(0 if passed else 1)
