#!/usr/bin/env python3
"""
Q11 Test 2.11: The Qualia Horizon Test

Tests whether subjective experience creates irreducible information horizons -
whether there is an explanatory gap between objective descriptions and
subjective experience (the "hard problem of consciousness").

HYPOTHESIS: Objective descriptions cannot fully capture subjective experience.
No finite set of third-person descriptions converges to first-person qualia.

PREDICTION: Significant gap remains (>0.2 cosine distance) between objective
descriptions and subjective target
FALSIFICATION: Objective descriptions converge to subjective experience
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    compute_cosine_similarity, get_embeddings,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

QUALIA_GAP_THRESHOLD = 0.2  # Minimum gap for qualia horizon to exist
CONVERGENCE_THRESHOLD = 0.05  # If gap < this, descriptions have "converged"


# =============================================================================
# QUALIA TEST CASES
# =============================================================================

# Each test case has:
# - subjective: The first-person experiential description
# - objective: Third-person descriptions that attempt to capture it

QUALIA_CASES = {
    'redness': {
        'subjective': "the ineffable quality of experiencing redness itself, what it is like to see red",
        'objective': [
            "electromagnetic radiation with wavelength 620-750 nanometers",
            "color that activates L-cones more than M-cones in human retina",
            "color of blood, fire, and ripe tomatoes",
            "RGB value (255, 0, 0) in digital color systems",
            "color associated with danger, passion, and heat in Western culture",
            "complementary color to cyan on the color wheel",
            "primary color in additive color systems",
            "color with longest wavelength in visible spectrum",
            "color that causes increased heart rate in psychological studies",
            "color between orange and infrared in the electromagnetic spectrum",
        ],
    },
    'pain': {
        'subjective': "the felt quality of being in pain, the raw hurt of the experience itself",
        'objective': [
            "activation of nociceptors and pain-signaling neural pathways",
            "adaptive response to tissue damage or potential harm",
            "electrical signals transmitted through C-fibers and A-delta fibers",
            "release of substance P and glutamate at synaptic junctions",
            "activation of the anterior cingulate cortex and insula",
            "functional state that causes avoidance behavior",
            "information about bodily damage encoded in neural firing patterns",
            "experience typically accompanied by elevated cortisol levels",
            "adaptive signal promoting protective behaviors",
            "state measured by pain scales and behavioral responses",
        ],
    },
    'sweetness': {
        'subjective': "the pure sensation of tasting something sweet, the qualitative sweetness itself",
        'objective': [
            "activation of T1R2 and T1R3 taste receptor proteins",
            "binding of sugar molecules to taste bud receptors on the tongue",
            "neural signals transmitted via the chorda tympani nerve",
            "activation of the nucleus tractus solitarius in the brainstem",
            "chemical property of sucrose, fructose, and similar compounds",
            "taste associated with caloric density in evolutionary history",
            "opposite of bitter taste on the basic taste spectrum",
            "sensation typically produced by carbohydrates",
            "taste that activates reward pathways in the brain",
            "chemical detection of simple sugar molecular structures",
        ],
    },
    'fear': {
        'subjective': "the visceral feeling of terror, what it is actually like to be afraid",
        'objective': [
            "amygdala activation and sympathetic nervous system arousal",
            "release of adrenaline and cortisol into the bloodstream",
            "increased heart rate, dilated pupils, and rapid breathing",
            "evolutionary response to perceived threats to survival",
            "activation of fight-or-flight response circuitry",
            "functional state causing avoidance of danger",
            "elevated galvanic skin response measurable by sensors",
            "state associated with specific patterns of brain activity on fMRI",
            "adaptive emotional response promoting survival",
            "neurochemical cascade triggered by threat detection",
        ],
    },
    'blue_sky': {
        'subjective': "the experience of gazing at a clear blue sky, the specific quality of that blue",
        'objective': [
            "perception of light with wavelength approximately 450-495 nanometers",
            "Rayleigh scattering of sunlight by atmospheric molecules",
            "activation of S-cones in the retina more than L or M cones",
            "atmospheric optical phenomenon caused by molecular scattering",
            "color opposite to yellow-orange on the color wheel",
            "RGB color approximately (135, 206, 235) in digital representation",
            "perception associated with clear weather and open spaces",
            "spectral composition dominated by short-wavelength visible light",
            "visual experience of low-aerosol atmospheric conditions",
            "color psychologically associated with calm and openness",
        ],
    },
}


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


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_single_qualia(name: str, case: Dict, model) -> Dict:
    """
    Test explanatory gap for a single qualia case.

    Measures the semantic distance between subjective description
    and the closest/combined objective descriptions.

    Returns:
        Dictionary of test results
    """
    # Get embeddings
    subjective_emb = get_embeddings([case['subjective']], model)[0]
    objective_embs = get_embeddings(case['objective'], model)

    # Distance from each objective description to subjective target
    distances = []
    for i, obj_emb in enumerate(objective_embs):
        sim = compute_cosine_similarity(subjective_emb, obj_emb)
        dist = 1.0 - sim
        distances.append({
            'description': case['objective'][i][:50] + "...",
            'similarity': sim,
            'distance': dist,
        })

    # Sort by distance (closest first)
    distances.sort(key=lambda x: x['distance'])

    # Minimum distance (best single description)
    min_distance = distances[0]['distance']
    best_description = distances[0]['description']

    # Try combining descriptions
    # Average embedding of all objective descriptions
    combined_emb = np.mean(objective_embs, axis=0)
    combined_sim = compute_cosine_similarity(subjective_emb, combined_emb)
    combined_distance = 1.0 - combined_sim

    # Weighted combination (weight by inverse distance)
    weights = np.array([1.0 / (d['distance'] + EPS) for d in distances])
    weights = weights / np.sum(weights)
    weighted_emb = np.sum(objective_embs * weights[:, np.newaxis], axis=0)
    weighted_sim = compute_cosine_similarity(subjective_emb, weighted_emb)
    weighted_distance = 1.0 - weighted_sim

    # Best combined approach
    best_combined_distance = min(combined_distance, weighted_distance)

    # Does gap remain?
    gap_remains = min_distance > QUALIA_GAP_THRESHOLD
    combined_gap_remains = best_combined_distance > QUALIA_GAP_THRESHOLD
    converges = min_distance < CONVERGENCE_THRESHOLD

    return {
        'qualia': name,
        'subjective': case['subjective'][:60] + "...",
        'n_objective': len(case['objective']),
        'min_distance': min_distance,
        'best_description': best_description,
        'combined_distance': combined_distance,
        'weighted_distance': weighted_distance,
        'best_combined': best_combined_distance,
        'gap_remains': gap_remains,
        'combined_gap_remains': combined_gap_remains,
        'converges': converges,
        'all_distances': distances[:3],  # Top 3 only
    }


def test_gap_stability(case: Dict, model) -> Dict:
    """
    Test if the gap is stable under increasing descriptions.

    Add more objective descriptions and see if gap decreases.
    If gap is fundamental, more descriptions won't help.
    """
    subjective_emb = get_embeddings([case['subjective']], model)[0]

    # Test with increasing number of descriptions
    gaps = []
    for n in range(1, len(case['objective']) + 1):
        subset = case['objective'][:n]
        subset_embs = get_embeddings(subset, model)

        # Combined embedding
        combined_emb = np.mean(subset_embs, axis=0)
        sim = compute_cosine_similarity(subjective_emb, combined_emb)
        gap = 1.0 - sim

        gaps.append({
            'n_descriptions': n,
            'gap': gap,
        })

    # Check if gap is decreasing
    gap_values = [g['gap'] for g in gaps]
    if len(gap_values) > 1:
        trend = np.polyfit(range(len(gap_values)), gap_values, 1)[0]
        decreasing = trend < -0.01
    else:
        trend = 0
        decreasing = False

    return {
        'gaps_by_n': gaps,
        'trend': trend,
        'gap_decreasing': decreasing,
        'would_converge': decreasing and gap_values[-1] < QUALIA_GAP_THRESHOLD,
    }


def test_cross_qualia_comparison(results: List[Dict]) -> Dict:
    """
    Compare gaps across different qualia types.

    Check if some qualia have larger gaps than others.
    """
    gaps = [(r['qualia'], r['min_distance']) for r in results]
    gaps.sort(key=lambda x: x[1], reverse=True)

    avg_gap = np.mean([g[1] for g in gaps])
    max_gap = max([g[1] for g in gaps])
    min_gap = min([g[1] for g in gaps])

    return {
        'ranked_gaps': gaps,
        'avg_gap': avg_gap,
        'max_gap': max_gap,
        'min_gap': min_gap,
        'gap_variance': np.var([g[1] for g in gaps]),
        'hardest_qualia': gaps[0][0],
        'easiest_qualia': gaps[-1][0],
    }


def run_qualia_horizon_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete qualia horizon test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.11: QUALIA HORIZON")

    np.random.seed(RANDOM_SEED)

    print("\nLoading embedding model...")
    model = load_model()

    # Test each qualia case
    print_subheader("Phase 1: Testing Individual Qualia")
    qualia_results = []

    for name, case in QUALIA_CASES.items():
        result = test_single_qualia(name, case, model)
        qualia_results.append(result)

        print(f"\n{name.upper()}:")
        print(f"  Min distance (best description): {result['min_distance']:.4f}")
        print(f"  Combined distance: {result['combined_distance']:.4f}")
        print(f"  Best combined: {result['best_combined']:.4f}")
        print(f"  Gap remains: {result['gap_remains']}")
        print(f"  Best description: {result['best_description']}")

    # Gap stability test
    print_subheader("Phase 2: Gap Stability Analysis")

    stability_results = []
    for name, case in QUALIA_CASES.items():
        stability = test_gap_stability(case, model)
        stability_results.append({
            'qualia': name,
            **stability,
        })

        print(f"\n{name}:")
        print(f"  Trend: {stability['trend']:.4f}")
        print(f"  Gap decreasing: {stability['gap_decreasing']}")
        print(f"  Would converge: {stability['would_converge']}")

    # Cross-qualia comparison
    print_subheader("Phase 3: Cross-Qualia Comparison")
    comparison = test_cross_qualia_comparison(qualia_results)

    print(f"\nAverage gap: {comparison['avg_gap']:.4f}")
    print(f"Max gap ({comparison['hardest_qualia']}): {comparison['max_gap']:.4f}")
    print(f"Min gap ({comparison['easiest_qualia']}): {comparison['min_gap']:.4f}")
    print(f"Gap variance: {comparison['gap_variance']:.4f}")

    print("\nRanked by gap (largest first):")
    for name, gap in comparison['ranked_gaps']:
        print(f"  {name}: {gap:.4f}")

    # Determine pass/fail
    print_subheader("Phase 4: Final Determination")

    # Count qualia where gap remains
    gaps_remain = sum(1 for r in qualia_results if r['gap_remains'])
    combined_gaps_remain = sum(1 for r in qualia_results if r['combined_gap_remains'])
    total = len(qualia_results)

    # Check if gaps are stable (not converging)
    stable_gaps = sum(1 for s in stability_results if not s['would_converge'])

    print(f"\nQualia with significant gap: {gaps_remain}/{total}")
    print(f"Qualia with combined gap: {combined_gaps_remain}/{total}")
    print(f"Stable (non-converging) gaps: {stable_gaps}/{total}")

    # Pass if majority of qualia show persistent gap
    passed = (gaps_remain > total / 2) and (stable_gaps > total / 2)

    if passed:
        horizon_type = HorizonType.ONTOLOGICAL  # Possibly absolute
        notes = f"Qualia horizon detected: {gaps_remain}/{total} qualia have gaps, {stable_gaps}/{total} stable"
    else:
        horizon_type = HorizonType.UNKNOWN
        if gaps_remain <= total / 2:
            notes = "Most qualia captured by objective descriptions - gap may be closeable"
        else:
            notes = "Gaps exist but are converging - may be epistemic not ontological"

    print_result("Qualia Horizon Test", passed, notes)

    result = HorizonTestResult(
        test_name="Qualia Horizon",
        test_id="Q11_2.11",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'gaps_remain_count': gaps_remain,
            'combined_gaps_remain_count': combined_gaps_remain,
            'stable_gaps_count': stable_gaps,
            'total_qualia': total,
            'avg_gap': comparison['avg_gap'],
            'max_gap': comparison['max_gap'],
            'min_gap': comparison['min_gap'],
            'hardest_qualia': comparison['hardest_qualia'],
        },
        thresholds={
            'qualia_gap_threshold': QUALIA_GAP_THRESHOLD,
            'convergence_threshold': CONVERGENCE_THRESHOLD,
        },
        evidence={
            'qualia_results': [to_builtin({
                'qualia': r['qualia'],
                'min_distance': r['min_distance'],
                'combined_distance': r['combined_distance'],
                'gap_remains': r['gap_remains'],
            }) for r in qualia_results],
            'stability_results': [to_builtin({
                'qualia': s['qualia'],
                'trend': s['trend'],
                'would_converge': s['would_converge'],
            }) for s in stability_results],
            'comparison': to_builtin(comparison),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_qualia_horizon_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    print("\n" + "-" * 70)
    print("PHILOSOPHICAL IMPLICATION:")
    if passed:
        print("  The explanatory gap persists: objective descriptions cannot")
        print("  fully capture subjective experience. This suggests qualia")
        print("  may constitute an ONTOLOGICAL horizon - not just unknown,")
        print("  but possibly unknowable in third-person terms.")
    else:
        print("  Objective descriptions approach subjective experience.")
        print("  The gap may be epistemic rather than ontological.")
    print("-" * 70)

    sys.exit(0 if passed else 1)
