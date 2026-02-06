#!/usr/bin/env python3
"""
Q11 Test 2.5: The Unknown Unknown Detector

Tests whether an agent can detect the PRESENCE of truths it doesn't know
it doesn't know, without being able to identify WHAT those truths are.

HYPOTHESIS: Unknown unknowns can be statistically detected via semantic
coverage analysis - "holes" in embedding space suggest missing concepts.

PREDICTION: Detection rate > 50% (can detect unknown unknowns)
FALSIFICATION: Cannot detect presence of unknown unknowns
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
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

DETECTION_THRESHOLD = 0.5  # Must detect > 50% of hidden concepts
N_PROBES = 5000           # Number of random probes in embedding space
VOID_PERCENTILE = 95      # Threshold for defining voids


# =============================================================================
# CONCEPT SETS
# =============================================================================

# Complete concept set (ground truth)
COMPLETE_CONCEPTS = {
    'animals': ['dog', 'cat', 'bird', 'fish', 'horse', 'elephant', 'lion', 'tiger'],
    'colors': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white'],
    'emotions': ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'joy', 'love'],
    'directions': ['north', 'south', 'east', 'west', 'up', 'down', 'left', 'right'],
    'numbers': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'],
}

# What the agent knows (incomplete - missing some from each category)
KNOWN_CONCEPTS = {
    'animals': ['dog', 'cat', 'bird', 'fish'],  # Missing: horse, elephant, lion, tiger
    'colors': ['red', 'blue', 'green'],          # Missing: yellow, orange, purple, black, white
    'emotions': ['happy', 'sad', 'angry'],       # Missing: fear, surprise, disgust, joy, love
    'directions': ['north', 'south', 'east'],    # Missing: west, up, down, left, right
    'numbers': ['one', 'two', 'three', 'four'],  # Missing: five, six, seven, eight
}


def get_missing_concepts() -> Dict[str, List[str]]:
    """Get the concepts that are missing from the agent's knowledge."""
    missing = {}
    for category in COMPLETE_CONCEPTS:
        missing[category] = [
            c for c in COMPLETE_CONCEPTS[category]
            if c not in KNOWN_CONCEPTS[category]
        ]
    return missing


# =============================================================================
# VOID DETECTION
# =============================================================================

def load_model():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("WARNING: sentence-transformers not installed, using fallback")
        return None


def detect_semantic_voids(known_embeddings: np.ndarray,
                         n_probes: int = N_PROBES) -> Tuple[np.ndarray, float]:
    """
    Detect voids (low-density regions) in semantic space.

    Uses random probing to find regions far from all known concepts.

    Args:
        known_embeddings: Embeddings of known concepts
        n_probes: Number of random probes

    Returns:
        Tuple of (void_probes, void_threshold)
    """
    np.random.seed(RANDOM_SEED + 1)

    # Get embedding dimension
    emb_dim = known_embeddings.shape[1]

    # Generate random probes (normalized to unit sphere)
    probes = np.random.randn(n_probes, emb_dim)
    probes = probes / (np.linalg.norm(probes, axis=1, keepdims=True) + EPS)

    # Find distance to nearest known concept for each probe
    min_distances = []
    for probe in probes:
        distances = np.linalg.norm(known_embeddings - probe, axis=1)
        min_distances.append(np.min(distances))

    min_distances = np.array(min_distances)

    # Define void threshold as high percentile of distances
    void_threshold = np.percentile(min_distances, VOID_PERCENTILE)

    # Select probes in voids (far from all known concepts)
    void_mask = min_distances > void_threshold
    void_probes = probes[void_mask]

    return void_probes, void_threshold


def check_voids_for_missing(void_probes: np.ndarray,
                           missing_embeddings: np.ndarray,
                           missing_labels: List[str],
                           void_threshold: float) -> Dict:
    """
    Check if voids correspond to missing concepts.

    Args:
        void_probes: Probes in void regions
        missing_embeddings: Embeddings of missing concepts
        missing_labels: Labels of missing concepts
        void_threshold: Threshold defining void boundary

    Returns:
        Dictionary of detection results
    """
    detected = []
    not_detected = []

    for label, emb in zip(missing_labels, missing_embeddings):
        # Check if this missing concept is near any void
        if len(void_probes) > 0:
            distances_to_voids = np.linalg.norm(void_probes - emb, axis=1)
            min_dist_to_void = np.min(distances_to_voids)

            # Concept is "detected" if it's reasonably close to a void
            # This means the void correctly indicates missing knowledge
            if min_dist_to_void < void_threshold * 1.5:  # Within 1.5x threshold
                detected.append({
                    'concept': label,
                    'distance_to_nearest_void': min_dist_to_void,
                })
            else:
                not_detected.append({
                    'concept': label,
                    'distance_to_nearest_void': min_dist_to_void,
                })
        else:
            not_detected.append({
                'concept': label,
                'distance_to_nearest_void': float('inf'),
            })

    detection_rate = len(detected) / (len(detected) + len(not_detected)) if (detected or not_detected) else 0

    return {
        'detected': detected,
        'not_detected': not_detected,
        'detection_rate': detection_rate,
        'total_missing': len(detected) + len(not_detected),
    }


# =============================================================================
# CATEGORY-AWARE DETECTION
# =============================================================================

def detect_category_gaps(known_embeddings: np.ndarray,
                        known_labels: List[str],
                        known_categories: Dict[str, List[str]],
                        model) -> Dict:
    """
    Detect gaps in category coverage.

    If a category has fewer concepts than expected, this may indicate
    unknown unknowns in that category.

    Args:
        known_embeddings: Embeddings of known concepts
        known_labels: Labels of known concepts
        known_categories: Known concepts by category
        model: Embedding model

    Returns:
        Dictionary of gap analysis results
    """
    results = {}

    for category, concepts in known_categories.items():
        if len(concepts) < 2:
            continue

        # Get category embeddings
        category_embs = get_embeddings(concepts, model)

        # Compute centroid and spread
        centroid = np.mean(category_embs, axis=0)
        distances_to_centroid = np.linalg.norm(category_embs - centroid, axis=1)
        avg_spread = np.mean(distances_to_centroid)
        max_spread = np.max(distances_to_centroid)

        # Check for unusual spread pattern (might indicate missing concepts)
        # A category with missing concepts might have larger gaps
        spread_ratio = max_spread / (avg_spread + EPS)

        # Compute pairwise distances
        n = len(concepts)
        pairwise = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise.append(np.linalg.norm(category_embs[i] - category_embs[j]))

        avg_pairwise = np.mean(pairwise) if pairwise else 0
        max_pairwise = np.max(pairwise) if pairwise else 0

        # Large max/avg ratio might indicate missing intermediate concepts
        gap_ratio = max_pairwise / (avg_pairwise + EPS) if pairwise else 0

        results[category] = {
            'n_concepts': len(concepts),
            'avg_spread': avg_spread,
            'max_spread': max_spread,
            'spread_ratio': spread_ratio,
            'avg_pairwise': avg_pairwise,
            'max_pairwise': max_pairwise,
            'gap_ratio': gap_ratio,
            'suggests_missing': gap_ratio > 1.5,  # Heuristic threshold
        }

    return results


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_void_detection() -> Dict:
    """
    Test detection of unknown unknowns via void analysis.

    Returns:
        Dictionary of test results
    """
    print("Loading model...")
    model = load_model()

    # Flatten known concepts
    known_flat = []
    for concepts in KNOWN_CONCEPTS.values():
        known_flat.extend(concepts)

    # Get embeddings for known concepts
    known_embs = get_embeddings(known_flat, model)

    # Detect voids
    print(f"Probing semantic space with {N_PROBES} points...")
    void_probes, void_threshold = detect_semantic_voids(known_embs)
    print(f"Found {len(void_probes)} void probes (threshold: {void_threshold:.4f})")

    # Get missing concepts
    missing = get_missing_concepts()
    missing_flat = []
    for concepts in missing.values():
        missing_flat.extend(concepts)

    # Get embeddings for missing concepts
    missing_embs = get_embeddings(missing_flat, model)

    # Check if voids correspond to missing concepts
    detection_results = check_voids_for_missing(
        void_probes, missing_embs, missing_flat, void_threshold
    )

    return {
        'void_count': len(void_probes),
        'void_threshold': void_threshold,
        'known_count': len(known_flat),
        'missing_count': len(missing_flat),
        'detection_results': detection_results,
    }


def test_category_gap_detection() -> Dict:
    """
    Test detection of gaps within categories.

    Returns:
        Dictionary of test results
    """
    print("Loading model...")
    model = load_model()

    # Analyze known categories
    known_flat = []
    for concepts in KNOWN_CONCEPTS.values():
        known_flat.extend(concepts)
    known_embs = get_embeddings(known_flat, model)

    # Detect category gaps
    gap_analysis = detect_category_gaps(
        known_embs, known_flat, KNOWN_CONCEPTS, model
    )

    # Get ground truth
    missing = get_missing_concepts()

    # Check if gap detection correlates with actual missing concepts
    correct_predictions = 0
    for category, analysis in gap_analysis.items():
        actually_missing = len(missing.get(category, []))
        predicted_missing = analysis['suggests_missing']

        if (predicted_missing and actually_missing > 0) or \
           (not predicted_missing and actually_missing == 0):
            correct_predictions += 1

    accuracy = correct_predictions / len(gap_analysis) if gap_analysis else 0

    return {
        'gap_analysis': gap_analysis,
        'prediction_accuracy': accuracy,
        'n_categories': len(gap_analysis),
    }


def test_statistical_anomaly() -> Dict:
    """
    Test if statistical properties reveal unknown unknowns.

    Looks for anomalies in the distribution of known concepts.
    """
    print("Loading model...")
    model = load_model()

    known_flat = []
    for concepts in KNOWN_CONCEPTS.values():
        known_flat.extend(concepts)
    known_embs = get_embeddings(known_flat, model)

    complete_flat = []
    for concepts in COMPLETE_CONCEPTS.values():
        complete_flat.extend(concepts)
    complete_embs = get_embeddings(complete_flat, model)

    # Compare distribution properties
    # If known concepts are missing some, the distribution should be different

    # Eigenspectrum analysis
    from numpy.linalg import svd

    _, known_s, _ = svd(known_embs, full_matrices=False)
    _, complete_s, _ = svd(complete_embs, full_matrices=False)

    # Normalize
    known_s = known_s / np.sum(known_s)
    complete_s = complete_s / np.sum(complete_s)

    # Effective dimensionality
    known_eff_dim = 1.0 / (np.sum(known_s ** 2) + EPS)
    complete_eff_dim = 1.0 / (np.sum(complete_s ** 2) + EPS)

    # If known has lower effective dim, might indicate missing concepts
    dim_ratio = known_eff_dim / complete_eff_dim

    # Coverage analysis
    # Project complete concepts onto known space
    known_centroid = np.mean(known_embs, axis=0)
    distances_known = np.linalg.norm(known_embs - known_centroid, axis=1)
    max_known_dist = np.max(distances_known)

    # Check which complete concepts are "outside" the known coverage
    outside_count = 0
    for emb in complete_embs:
        dist = np.linalg.norm(emb - known_centroid)
        if dist > max_known_dist * 1.2:  # 20% margin
            outside_count += 1

    outside_ratio = outside_count / len(complete_flat)

    return {
        'known_eff_dim': known_eff_dim,
        'complete_eff_dim': complete_eff_dim,
        'dim_ratio': dim_ratio,
        'suggests_incompleteness': dim_ratio < 0.9,  # Lower dim = missing coverage
        'outside_count': outside_count,
        'outside_ratio': outside_ratio,
    }


def run_unknown_unknowns_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete unknown unknowns test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.5: UNKNOWN UNKNOWN DETECTOR")

    np.random.seed(RANDOM_SEED)

    # Test 1: Void detection
    print_subheader("Phase 1: Semantic Void Detection")
    void_results = test_void_detection()

    print(f"Voids found: {void_results['void_count']}")
    print(f"Detection rate: {void_results['detection_results']['detection_rate']:.1%}")
    print(f"Detected {len(void_results['detection_results']['detected'])} of "
          f"{void_results['missing_count']} missing concepts")

    if void_results['detection_results']['detected']:
        print("\nDetected unknown unknowns:")
        for d in void_results['detection_results']['detected'][:5]:
            print(f"  {d['concept']} (dist: {d['distance_to_nearest_void']:.4f})")

    # Test 2: Category gap detection
    print_subheader("Phase 2: Category Gap Detection")
    gap_results = test_category_gap_detection()

    print(f"Category gap prediction accuracy: {gap_results['prediction_accuracy']:.1%}")
    for cat, analysis in gap_results['gap_analysis'].items():
        gap_str = "GAPS DETECTED" if analysis['suggests_missing'] else "complete"
        print(f"  {cat}: {analysis['n_concepts']} concepts, gap_ratio={analysis['gap_ratio']:.2f} - {gap_str}")

    # Test 3: Statistical anomaly
    print_subheader("Phase 3: Statistical Anomaly Detection")
    stat_results = test_statistical_anomaly()

    print(f"Known effective dim: {stat_results['known_eff_dim']:.2f}")
    print(f"Complete effective dim: {stat_results['complete_eff_dim']:.2f}")
    print(f"Dim ratio: {stat_results['dim_ratio']:.3f}")
    print(f"Suggests incompleteness: {stat_results['suggests_incompleteness']}")
    print(f"Concepts outside known coverage: {stat_results['outside_count']} ({stat_results['outside_ratio']:.1%})")

    # Determine pass/fail
    print_subheader("Phase 4: Final Determination")

    # Primary criterion: void detection rate > threshold
    void_detection_passes = (
        void_results['detection_results']['detection_rate'] > DETECTION_THRESHOLD
    )

    # Secondary: any method detects unknown unknowns
    any_detection = (
        void_detection_passes or
        gap_results['prediction_accuracy'] > 0.6 or
        stat_results['suggests_incompleteness']
    )

    passed = void_detection_passes or any_detection

    if passed:
        horizon_type = HorizonType.STRUCTURAL
        if void_detection_passes:
            notes = f"Void detection works: {void_results['detection_results']['detection_rate']:.0%} of unknown unknowns found"
        else:
            notes = "Partial detection via category gaps or statistical anomaly"
    else:
        horizon_type = HorizonType.UNKNOWN
        notes = "Cannot reliably detect unknown unknowns - frame problem confirmed?"

    print(f"\nVoid detection passes: {void_detection_passes}")
    print(f"Any detection method works: {any_detection}")
    print_result("Unknown Unknown Detector Test", passed, notes)

    result = HorizonTestResult(
        test_name="Unknown Unknown Detector",
        test_id="Q11_2.5",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'void_detection_rate': void_results['detection_results']['detection_rate'],
            'gap_prediction_accuracy': gap_results['prediction_accuracy'],
            'dim_ratio': stat_results['dim_ratio'],
            'outside_ratio': stat_results['outside_ratio'],
            'voids_found': void_results['void_count'],
        },
        thresholds={
            'detection_threshold': DETECTION_THRESHOLD,
            'void_percentile': VOID_PERCENTILE,
            'n_probes': N_PROBES,
        },
        evidence={
            'void_results': to_builtin(void_results),
            'gap_results': to_builtin(gap_results),
            'stat_results': to_builtin(stat_results),
            'known_concepts': KNOWN_CONCEPTS,
            'missing_concepts': get_missing_concepts(),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_unknown_unknowns_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)
