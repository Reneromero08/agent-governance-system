"""
Q23 Phase 2: Hexagonal Geometry Test

Test whether semantic embeddings show hexagonal structure in 2D projections.
If semantic space has hexagonal packing, this explains why sqrt(3) appears.

Tests:
- 2A: Angle distribution in Delaunay triangulation (peak at 60 degrees)
- 2B: Nearest neighbor distance ratios (should be sqrt(3)/2 for hexagonal)
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime
from scipy import stats
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

from q23_utils import (
    SQRT_3, TestResult, cohens_d, bootstrap_ci, cv, save_result,
    compute_delaunay_angles, compute_nn_distance_ratios, run_all_validations,
    get_embeddings, project_to_2d
)

# =============================================================================
# REAL VOCABULARY (larger set for meaningful geometry)
# =============================================================================

# Common English words across different categories
# NOTE: All words are unique - no duplicates in source list
VOCABULARY = [
    # Emotions (20)
    "happy", "sad", "angry", "afraid", "surprised", "disgusted", "joy", "love",
    "hate", "fear", "trust", "anticipation", "content", "excited", "bored", "anxious",
    "calm", "frustrated", "hopeful", "desperate",

    # Animals (20)
    "dog", "cat", "bird", "salmon", "horse", "cow", "pig", "sheep", "lion", "tiger",
    "elephant", "monkey", "snake", "eagle", "dolphin", "whale", "bear", "wolf",
    "rabbit", "deer",

    # Food (20)
    "apple", "bread", "cheese", "water", "coffee", "rice", "meat", "tuna", "egg",
    "milk", "sugar", "salt", "butter", "oil", "flour", "honey", "wine", "beer",
    "soup", "salad",

    # Nature (20)
    "tree", "flower", "grass", "river", "mountain", "ocean", "sun", "moon", "star",
    "cloud", "rain", "snow", "wind", "fire", "earth", "stone", "sand", "forest",
    "desert", "island",

    # Objects (20)
    "book", "table", "chair", "door", "window", "car", "phone", "computer", "pen",
    "paper", "clock", "lamp", "bed", "mirror", "key", "cup", "plate", "knife",
    "bottle", "bag",

    # Actions (20)
    "run", "walk", "eat", "sleep", "talk", "read", "write", "think", "feel", "see",
    "hear", "touch", "smell", "taste", "work", "play", "learn", "teach", "help",
    "create",

    # Concepts (20)
    "time", "space", "life", "death", "truth", "beauty", "power", "freedom", "peace",
    "war", "justice", "hope", "faith", "reason", "wisdom", "courage", "honor",
    "duty", "virtue", "mercy",
]

# Verify no duplicates (should be 140 unique words)
assert len(VOCABULARY) == len(set(VOCABULARY)), "Duplicate words in VOCABULARY!"


@dataclass
class HexagonalResult:
    """Result of hexagonal geometry test."""
    test_name: str
    passed: bool
    angle_peak: float  # Peak location in angle histogram
    angle_peak_strength: float  # How strong the peak is vs uniform
    nn_ratio_mean: float  # Mean nearest neighbor ratio
    nn_ratio_expected: float  # Expected ratio for hexagonal (1.0 for equidistant)
    cohens_d_vs_random: float  # Effect size vs random embeddings
    p_value: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "angle_peak": self.angle_peak,
            "angle_peak_strength": self.angle_peak_strength,
            "nn_ratio_mean": self.nn_ratio_mean,
            "nn_ratio_expected": self.nn_ratio_expected,
            "cohens_d_vs_random": self.cohens_d_vs_random,
            "p_value": self.p_value,
            "details": self.details,
        }


# =============================================================================
# TEST 2A: ANGLE DISTRIBUTION
# =============================================================================

def test_angle_distribution(model_name: str = "all-MiniLM-L6-v2",
                           verbose: bool = True) -> HexagonalResult:
    """
    Test if 2D projection of embeddings shows hexagonal structure.

    Hexagonal packing = equilateral triangles = 60-degree peak in angle distribution.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 2A: ANGLE DISTRIBUTION IN 2D PROJECTION")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Vocabulary size: {len(VOCABULARY)}")

    # Get embeddings
    embeddings = get_embeddings(VOCABULARY, model_name)
    if verbose:
        print(f"Embedding dimension: {embeddings.shape[1]}")

    # Project to 2D
    points_2d = project_to_2d(embeddings)
    if verbose:
        print(f"Projected to 2D: {points_2d.shape}")

    # Compute Delaunay angles
    angles = compute_delaunay_angles(points_2d)
    if verbose:
        print(f"Number of triangles: {len(angles) // 3}")
        print(f"Number of angles: {len(angles)}")

    # Histogram with 36 bins (5-degree resolution)
    hist, bin_edges = np.histogram(angles, bins=36, range=(0, 180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peak
    peak_idx = np.argmax(hist)
    peak_angle = bin_centers[peak_idx]
    peak_count = hist[peak_idx]
    mean_count = np.mean(hist)

    # Peak strength: how many standard deviations above mean
    if np.std(hist) > 0:
        peak_strength = (peak_count - mean_count) / np.std(hist)
    else:
        peak_strength = 0.0

    if verbose:
        print(f"\nAngle distribution:")
        print(f"  Peak at: {peak_angle:.1f} degrees")
        print(f"  Peak count: {peak_count}")
        print(f"  Mean count: {mean_count:.1f}")
        print(f"  Peak strength (z-score): {peak_strength:.2f}")

    # Generate random baseline
    np.random.seed(42)
    random_points = np.random.randn(len(VOCABULARY), 2)
    random_angles = compute_delaunay_angles(random_points)

    random_hist, _ = np.histogram(random_angles, bins=36, range=(0, 180))
    random_peak_idx = np.argmax(random_hist)
    random_peak_angle = bin_centers[random_peak_idx]

    # Compare trained vs random
    d = cohens_d(angles, random_angles)

    if verbose:
        print(f"\nRandom baseline:")
        print(f"  Peak at: {random_peak_angle:.1f} degrees")
        print(f"  Cohen's d (trained vs random): {d:.2f}")

    # Statistical test: is the trained distribution different from random?
    # NOTE: Delaunay angles are NOT uniform even for random points, so we compare
    # trained vs random directly using chi-square, not trained vs uniform.
    # Scale random histogram to match trained sample size
    scale_factor = len(angles) / len(random_angles) if len(random_angles) > 0 else 1.0
    expected_from_random = random_hist * scale_factor + 1e-10  # Add epsilon to avoid div by 0
    chi2, p_value = stats.chisquare(hist, expected_from_random)

    if verbose:
        print(f"\nChi-square test vs random baseline:")
        print(f"  Chi-square: {chi2:.2f}")
        print(f"  p-value: {p_value:.6f}")

    # Check if peak is at 60 degrees (within one bin = 5 degrees)
    is_60_peak = abs(peak_angle - 60) < 7.5  # Within 1.5 bins

    # Pass criteria
    passed = (
        is_60_peak and  # Peak at 60 degrees
        peak_strength > 2.0 and  # Significant peak
        p_value < 0.01  # Non-uniform distribution
    )

    if verbose:
        print(f"\nVerdict:")
        print(f"  Peak at 60 degrees: {'YES' if is_60_peak else 'NO'} ({peak_angle:.1f})")
        print(f"  Peak strength > 2.0: {'YES' if peak_strength > 2.0 else 'NO'} ({peak_strength:.2f})")
        print(f"  p < 0.01: {'YES' if p_value < 0.01 else 'NO'} ({p_value:.6f})")
        print(f"  PASSED: {'YES' if passed else 'NO'}")

    return HexagonalResult(
        test_name="angle_distribution",
        passed=passed,
        angle_peak=float(peak_angle),
        angle_peak_strength=float(peak_strength),
        nn_ratio_mean=0.0,  # Not computed in this test
        nn_ratio_expected=1.0,
        cohens_d_vs_random=float(d),
        p_value=float(p_value),
        details={
            "vocabulary_size": len(VOCABULARY),
            "embedding_dim": embeddings.shape[1],
            "n_triangles": len(angles) // 3,
            "histogram": hist.tolist(),
            "bin_centers": bin_centers.tolist(),
            "is_60_peak": is_60_peak,
            "random_peak_angle": float(random_peak_angle),
        }
    )


# =============================================================================
# TEST 2B: NEAREST NEIGHBOR RATIOS
# =============================================================================

def test_nn_ratios(model_name: str = "all-MiniLM-L6-v2",
                  verbose: bool = True) -> HexagonalResult:
    """
    Test nearest neighbor distance ratios.

    For hexagonal packing, ratio of 2nd-nearest to nearest should be 1.0
    (all 6 neighbors are equidistant).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 2B: NEAREST NEIGHBOR DISTANCE RATIOS")
        print("=" * 60)
        print(f"Model: {model_name}")

    # Get embeddings
    embeddings = get_embeddings(VOCABULARY, model_name)

    # Project to 2D
    points_2d = project_to_2d(embeddings)

    # Compute NN ratios (2nd nearest / nearest)
    ratios = compute_nn_distance_ratios(points_2d, k=2)

    if verbose:
        print(f"\nNearest neighbor ratios (2nd/1st):")
        print(f"  Mean: {np.mean(ratios):.3f}")
        print(f"  Std: {np.std(ratios):.3f}")
        print(f"  Expected for hexagonal: 1.0")

    # Generate random baseline
    np.random.seed(42)
    random_points = np.random.randn(len(VOCABULARY), 2)
    random_ratios = compute_nn_distance_ratios(random_points, k=2)

    if verbose:
        print(f"\nRandom baseline:")
        print(f"  Mean: {np.mean(random_ratios):.3f}")
        print(f"  Std: {np.std(random_ratios):.3f}")

    # Compare
    d = cohens_d(ratios, random_ratios)

    if verbose:
        print(f"\nCohen's d (trained vs random): {d:.2f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(ratios, random_ratios)

    if verbose:
        print(f"t-test p-value: {p_value:.6f}")

    # Check if ratio is close to 1.0 (hexagonal)
    # Actually, for random 2D Poisson, ratio is around 1.3-1.4
    # Hexagonal would be 1.0 (equidistant neighbors)
    mean_ratio = np.mean(ratios)
    expected_hexagonal = 1.0

    # Pass if trained embeddings have ratio closer to 1.0 than random
    closer_to_hex = abs(mean_ratio - expected_hexagonal) < abs(np.mean(random_ratios) - expected_hexagonal)

    passed = closer_to_hex and p_value < 0.05

    if verbose:
        print(f"\nVerdict:")
        print(f"  Trained ratio: {mean_ratio:.3f}")
        print(f"  Random ratio: {np.mean(random_ratios):.3f}")
        print(f"  Closer to hexagonal (1.0): {'TRAINED' if closer_to_hex else 'RANDOM'}")
        print(f"  PASSED: {'YES' if passed else 'NO'}")

    return HexagonalResult(
        test_name="nn_ratios",
        passed=passed,
        angle_peak=0.0,  # Not computed
        angle_peak_strength=0.0,
        nn_ratio_mean=float(mean_ratio),
        nn_ratio_expected=expected_hexagonal,
        cohens_d_vs_random=float(d),
        p_value=float(p_value),
        details={
            "vocabulary_size": len(VOCABULARY),
            "random_ratio_mean": float(np.mean(random_ratios)),
            "closer_to_hexagonal": closer_to_hex,
        }
    )


# =============================================================================
# CROSS-MODEL VALIDATION
# =============================================================================

def run_cross_model_validation(model_names: List[str],
                              verbose: bool = True) -> Dict[str, Any]:
    """Run hexagonal tests across multiple models."""
    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-MODEL VALIDATION: HEXAGONAL GEOMETRY")
        print("=" * 60)

    results = {}
    angle_peaks = []
    passed_count = 0

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*40}")
            print(f"Model: {model_name}")
            print("=" * 40)

        try:
            angle_result = test_angle_distribution(model_name, verbose=False)
            nn_result = test_nn_ratios(model_name, verbose=False)

            results[model_name] = {
                "angle_distribution": angle_result.to_dict(),
                "nn_ratios": nn_result.to_dict(),
            }

            angle_peaks.append(angle_result.angle_peak)

            if angle_result.passed:
                passed_count += 1

            if verbose:
                print(f"  Angle peak: {angle_result.angle_peak:.1f} degrees")
                print(f"  Angle test passed: {angle_result.passed}")
                print(f"  NN ratio: {nn_result.nn_ratio_mean:.3f}")
                print(f"  NN test passed: {nn_result.passed}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")

    # Summary
    if angle_peaks:
        angle_peaks = np.array(angle_peaks)
        results["summary"] = {
            "n_models": len(angle_peaks),
            "angle_peak_mean": float(np.mean(angle_peaks)),
            "angle_peak_std": float(np.std(angle_peaks)),
            "angle_peak_cv": float(cv(angle_peaks)),
            "n_passed": passed_count,
            "pass_rate": passed_count / len(angle_peaks),
        }

        if verbose:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print("=" * 60)
            print(f"Models tested: {len(angle_peaks)}")
            print(f"Mean angle peak: {np.mean(angle_peaks):.1f} degrees")
            print(f"CV: {cv(angle_peaks)*100:.1f}%")
            print(f"Passed: {passed_count}/{len(angle_peaks)}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase 2 hexagonal geometry tests."""
    print("=" * 60)
    print("Q23 PHASE 2: HEXAGONAL GEOMETRY TEST")
    print("=" * 60)
    print("\nTesting if semantic embeddings show hexagonal structure")
    print("If true, this explains why sqrt(3) appears in the formula")
    print()

    # Run validations first
    if not run_all_validations():
        print("\nABORTING: Test validation failed")
        return

    # Run tests on primary model
    angle_result = test_angle_distribution("all-MiniLM-L6-v2")
    nn_result = test_nn_ratios("all-MiniLM-L6-v2")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)

    print(f"\n2A. Angle Distribution:")
    print(f"    Peak at: {angle_result.angle_peak:.1f} degrees (expected: 60)")
    print(f"    Peak strength: {angle_result.angle_peak_strength:.2f}")
    print(f"    PASSED: {angle_result.passed}")

    print(f"\n2B. Nearest Neighbor Ratios:")
    print(f"    Mean ratio: {nn_result.nn_ratio_mean:.3f} (expected: 1.0)")
    print(f"    PASSED: {nn_result.passed}")

    if angle_result.passed and nn_result.passed:
        print("\n*** HEXAGONAL STRUCTURE CONFIRMED ***")
        print("This explains why sqrt(3) appears in the formula.")
    elif angle_result.passed or nn_result.passed:
        print("\n*** PARTIAL HEXAGONAL STRUCTURE ***")
        print("Some evidence for hexagonal geometry, but not conclusive.")
    else:
        print("\n*** NO HEXAGONAL STRUCTURE FOUND ***")
        print("The sqrt(3) connection may be coincidental or from another source.")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "all-MiniLM-L6-v2",
        "vocabulary_size": len(VOCABULARY),
        "angle_distribution": angle_result.to_dict(),
        "nn_ratios": nn_result.to_dict(),
        "hexagonal_confirmed": angle_result.passed and nn_result.passed,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, f"q23_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filepath, 'w') as f:
        # Handle numpy types for JSON serialization
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else str(x))

    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
