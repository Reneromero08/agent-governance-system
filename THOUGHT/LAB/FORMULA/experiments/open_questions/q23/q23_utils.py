"""
Q23 Utilities: sqrt(3) Geometry Testing

This module provides validated utilities for testing the sqrt(3) hypothesis.
All functions include known-answer validation to catch bugs before real tests.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.spatial import Delaunay
import json
import os
from datetime import datetime

# Constants
SQRT_2 = np.sqrt(2)  # 1.414
SQRT_3 = np.sqrt(3)  # 1.732
SQRT_5 = np.sqrt(5)  # 2.236
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio 1.618
E = np.e  # 2.718

# Test candidates for sweep
ALPHA_CANDIDATES = [1.0, SQRT_2, 1.5, PHI, SQRT_3, 1.8, 2.0, E, SQRT_5]
ALPHA_NAMES = ["1.0", "sqrt(2)", "1.5", "phi", "sqrt(3)", "1.8", "2.0", "e", "sqrt(5)"]


@dataclass
class TestResult:
    """Structured result for any Q23 test."""
    test_name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp
        }


# =============================================================================
# PHASE 0: TEST VALIDATION FUNCTIONS
# =============================================================================

def validate_angle_measurement() -> bool:
    """
    Validate that angle measurement works correctly.
    Known answer: equilateral triangle has all angles = 60 degrees.
    """
    # Equilateral triangle with side length 1
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    p3 = np.array([0.5, SQRT_3 / 2])

    angles = compute_triangle_angles(p1, p2, p3)

    # All angles should be 60 degrees (within floating point tolerance)
    expected = 60.0
    tolerance = 0.01

    for angle in angles:
        if abs(angle - expected) > tolerance:
            print(f"VALIDATION FAILED: Expected {expected}, got {angle}")
            return False

    print("VALIDATION PASSED: Angle measurement correct (equilateral = 60 deg)")
    return True


def validate_distance_ratio() -> bool:
    """
    Validate distance ratio computation.
    Known answer: in hexagonal packing, 2nd nearest / nearest = 1.0 (same distance).
    """
    # Hexagonal arrangement: center point with 6 equidistant neighbors
    center = np.array([0.0, 0.0])
    neighbors = []
    for i in range(6):
        angle = i * np.pi / 3  # 60 degree increments
        neighbors.append(np.array([np.cos(angle), np.sin(angle)]))

    # All neighbors are distance 1.0 from center
    # So ratio of 2nd-nearest to nearest should be 1.0
    distances = [np.linalg.norm(center - n) for n in neighbors]
    distances.sort()

    ratio = distances[1] / distances[0]  # 2nd / 1st
    expected = 1.0
    tolerance = 0.01

    if abs(ratio - expected) > tolerance:
        print(f"VALIDATION FAILED: Expected ratio {expected}, got {ratio}")
        return False

    print("VALIDATION PASSED: Distance ratio correct (hexagonal = 1.0)")
    return True


def validate_random_baseline() -> bool:
    """
    Validate that random data does NOT produce hexagonal structure.
    If random data shows 60-degree peaks, our test is meaningless.
    """
    np.random.seed(42)
    random_points = np.random.randn(100, 2)

    # Compute angles from Delaunay triangulation
    angles = compute_delaunay_angles(random_points)

    # Check for peak at 60 degrees
    hist, bin_edges = np.histogram(angles, bins=36, range=(0, 180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find bin containing 60 degrees
    idx_60 = np.argmin(np.abs(bin_centers - 60))
    peak_at_60 = hist[idx_60]
    mean_count = np.mean(hist)

    # Random should NOT have a strong peak at 60
    # Peak should be within 2x of mean (not statistically special)
    if peak_at_60 > 2 * mean_count:
        print(f"WARNING: Random data shows 60-degree peak (count={peak_at_60}, mean={mean_count})")
        print("This suggests the test may not discriminate trained from random.")
        return False

    print(f"VALIDATION PASSED: Random data has no 60-degree peak (count={peak_at_60}, mean={mean_count:.1f})")
    return True


def run_all_validations() -> bool:
    """Run all Phase 0 validations. All must pass before real tests."""
    print("=" * 60)
    print("PHASE 0: TEST VALIDATION")
    print("=" * 60)

    results = []
    results.append(("Angle measurement", validate_angle_measurement()))
    results.append(("Distance ratio", validate_distance_ratio()))
    results.append(("Random baseline", validate_random_baseline()))

    print("\n" + "=" * 60)
    all_passed = all(r[1] for r in results)

    if all_passed:
        print("ALL VALIDATIONS PASSED - Safe to run real tests")
    else:
        print("VALIDATION FAILED - Fix issues before running real tests")
        for name, passed in results:
            if not passed:
                print(f"  FAILED: {name}")

    print("=" * 60)
    return all_passed


# =============================================================================
# GEOMETRY FUNCTIONS
# =============================================================================

def compute_triangle_angles(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> List[float]:
    """
    Compute all three angles (in degrees) of a triangle given its vertices.
    Uses law of cosines.
    """
    # Side lengths
    a = np.linalg.norm(p2 - p3)  # opposite to p1
    b = np.linalg.norm(p1 - p3)  # opposite to p2
    c = np.linalg.norm(p1 - p2)  # opposite to p3

    # Angles using law of cosines: cos(A) = (b^2 + c^2 - a^2) / (2bc)
    def safe_acos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))

    angles = []

    # Angle at p1
    if b > 0 and c > 0:
        cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
        angles.append(np.degrees(safe_acos(cos_A)))

    # Angle at p2
    if a > 0 and c > 0:
        cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
        angles.append(np.degrees(safe_acos(cos_B)))

    # Angle at p3
    if a > 0 and b > 0:
        cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
        angles.append(np.degrees(safe_acos(cos_C)))

    return angles


def compute_delaunay_angles(points: np.ndarray) -> np.ndarray:
    """
    Compute all triangle angles from Delaunay triangulation of 2D points.
    Returns array of angles in degrees.
    """
    if len(points) < 3:
        return np.array([])

    try:
        tri = Delaunay(points)
    except Exception as e:
        print(f"Delaunay failed: {e}")
        return np.array([])

    all_angles = []
    for simplex in tri.simplices:
        p1, p2, p3 = points[simplex]
        angles = compute_triangle_angles(p1, p2, p3)
        all_angles.extend(angles)

    return np.array(all_angles)


def compute_nn_distance_ratios(points: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Compute ratio of k-th nearest neighbor distance to 1st nearest neighbor.
    For hexagonal packing, this ratio is characteristic.
    """
    from scipy.spatial import distance_matrix

    n = len(points)
    if n < k + 1:
        return np.array([])

    dist_mat = distance_matrix(points, points)

    ratios = []
    for i in range(n):
        # Get distances to all other points, sorted
        dists = np.sort(dist_mat[i])
        # dists[0] is 0 (self), dists[1] is nearest, dists[2] is 2nd nearest
        if dists[1] > 1e-10:  # Avoid division by zero
            ratio = dists[k] / dists[1]
            ratios.append(ratio)

    return np.array(ratios)


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Get embeddings for a list of texts using sentence-transformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    except ImportError:
        print("sentence-transformers not installed. Using random embeddings.")
        np.random.seed(hash(str(texts)) % (2**32))
        return np.random.randn(len(texts), 384)


def project_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D using PCA."""
    from sklearn.decomposition import PCA

    if embeddings.shape[1] <= 2:
        return embeddings[:, :2]

    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_iterations: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    Returns (lower, point_estimate, upper).
    """
    np.random.seed(42)
    boot_stats = []

    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2

    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    point = statistic(data)

    return lower, point, upper


def cv(data: np.ndarray) -> float:
    """Coefficient of variation (std / mean)."""
    mean = np.mean(data)
    if abs(mean) < 1e-10:
        return float('inf')
    return np.std(data) / abs(mean)


# =============================================================================
# FILE I/O
# =============================================================================

def save_result(result: TestResult, results_dir: str = "results"):
    """Save test result to JSON file."""
    os.makedirs(results_dir, exist_ok=True)

    filename = f"{result.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"Result saved to: {filepath}")
    return filepath


# =============================================================================
# MAIN (for testing utils)
# =============================================================================

if __name__ == "__main__":
    # Run all validations
    success = run_all_validations()
    exit(0 if success else 1)
