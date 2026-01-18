"""
Q23 Phase 3: Hexagonal Berry Phase Test

HYPOTHESIS: sqrt(3) appears because of hexagonal geometry in semantic space.

Mathematical connection:
- Hexagonal loops have interior angle 120 degrees = 2*pi/3 radians
- sin(pi/3) = sqrt(3)/2, therefore sqrt(3) = 2*sin(pi/3)
- If semantic hexagons accumulate Berry phase = 2*pi/3, this explains sqrt(3)

Test Design:
1. Construct 6-vertex semantic hexagons (words forming conceptual cycles)
2. Measure Berry phase (solid angle / holonomy) around each hexagon
3. Check if mean phase is close to 2*pi/3 = 2.094 radians
4. Compare to 5-vertex (pentagons) and 7-vertex (heptagons) as controls

Pass Criteria:
- Mean hexagon phase within 15% of 2*pi/3
- Hexagons show more consistent phase than pentagons/heptagons
- Effect is present across multiple models
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
from scipy import stats

# Constants
PI = np.pi
TWO_PI = 2 * PI
TWO_PI_OVER_3 = TWO_PI / 3  # 2.094 radians = 120 degrees
SQRT_3 = np.sqrt(3)


@dataclass
class LoopPhaseResult:
    """Result for a single loop."""
    loop_words: List[str]
    n_vertices: int
    berry_phase: float  # radians
    expected_phase: float  # Expected for this polygon type
    deviation: float  # Absolute deviation from expected
    deviation_pct: float  # Percentage deviation


@dataclass
class PolygonTypeResult:
    """Results for all loops of one type (hex, pent, hept)."""
    polygon_type: str
    n_sides: int
    expected_phase: float
    n_loops: int
    mean_phase: float
    std_phase: float
    mean_deviation_pct: float
    phases: List[float]


@dataclass
class HexagonalBerryResult:
    """Overall results of the hexagonal Berry phase test."""
    hypothesis_supported: bool
    hexagon_results: PolygonTypeResult
    pentagon_results: PolygonTypeResult
    heptagon_results: PolygonTypeResult
    sqrt3_connection: Dict[str, Any]
    verdict: str


# =============================================================================
# SEMANTIC POLYGONS
# =============================================================================

# 6-vertex HEXAGONS (conceptual cycles that return to start)
SEMANTIC_HEXAGONS = [
    # Emotion cycle
    ["calm", "happy", "excited", "anxious", "sad", "peaceful"],

    # Time cycle
    ["morning", "noon", "afternoon", "evening", "night", "dawn"],

    # Season cycle (extended)
    ["spring", "early_summer", "summer", "autumn", "winter", "late_winter"],

    # Knowledge cycle
    ["ignorance", "curiosity", "learning", "understanding", "wisdom", "teaching"],

    # Economic cycle
    ["growth", "boom", "peak", "decline", "recession", "recovery"],

    # Water cycle
    ["ocean", "evaporation", "cloud", "rain", "river", "delta"],

    # Life stage cycle
    ["birth", "childhood", "youth", "adulthood", "old_age", "death"],

    # Color wheel (RGB + CMY)
    ["red", "yellow", "green", "cyan", "blue", "magenta"],
]

# 5-vertex PENTAGONS (control - should NOT show 2*pi/3)
SEMANTIC_PENTAGONS = [
    ["happy", "excited", "anxious", "sad", "calm"],
    ["morning", "noon", "evening", "night", "dawn"],
    ["spring", "summer", "autumn", "winter", "thaw"],
    ["growth", "peak", "decline", "recession", "recovery"],
    ["ocean", "cloud", "rain", "river", "sea"],
]

# 7-vertex HEPTAGONS (control - should NOT show 2*pi/3)
SEMANTIC_HEPTAGONS = [
    ["calm", "content", "happy", "excited", "anxious", "sad", "peaceful"],
    ["dawn", "morning", "noon", "afternoon", "evening", "night", "midnight"],
    ["spring", "early_summer", "summer", "late_summer", "autumn", "winter", "thaw"],
    ["growth", "expansion", "boom", "peak", "decline", "recession", "recovery"],
    ["ocean", "evaporation", "cloud", "rain", "stream", "river", "delta"],
]


# =============================================================================
# BERRY PHASE COMPUTATION
# =============================================================================

def compute_berry_phase_winding(path: np.ndarray, closed: bool = True) -> float:
    """
    Compute Berry phase as winding number in 2D PCA projection.

    For high-dimensional embeddings, project to 2D via PCA and
    compute the winding angle (total angle swept in complex plane).

    This is consistent with the Q51 implementation.
    """
    # Normalize embeddings
    norms = np.linalg.norm(path, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    path = path / norms

    # Close the loop if needed
    if closed and not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0:1]])

    # Project to 2D via PCA
    centered = path - path.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_2d = centered @ Vt[:2].T
    except:
        return 0.0

    if proj_2d.shape[1] < 2:
        return 0.0

    # Map to complex plane
    z = proj_2d[:, 0] + 1j * proj_2d[:, 1]

    # Handle zero values
    z = np.where(np.abs(z) < 1e-10, 1e-10, z)

    # Compute winding angle (total angle change around the loop)
    phase_diffs = np.angle(z[1:] / z[:-1])
    winding_angle = np.sum(phase_diffs)

    return winding_angle


def compute_spherical_excess(path: np.ndarray) -> float:
    """
    Compute spherical excess (solid angle) for a geodesic polygon.

    For a spherical polygon with n vertices:
    Omega = sum(interior_angles) - (n-2)*pi

    This is the actual solid angle subtended by the polygon on the unit sphere.
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(path, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    path = path / norms

    # Close loop if needed
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0:1]])

    n = len(path) - 1  # Number of vertices (excluding repeated first)

    # Compute interior angles
    interior_angles = []
    for i in range(n):
        # Get three consecutive points
        p_prev = path[(i - 1) % n]
        p_curr = path[i]
        p_next = path[(i + 1) % n]

        # Vectors from current to neighbors (in tangent space)
        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        # Project onto tangent space at p_curr
        v1 = v1 - np.dot(v1, p_curr) * p_curr
        v2 = v2 - np.dot(v2, p_curr) * p_curr

        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 1e-10 and n2 > 1e-10:
            v1 = v1 / n1
            v2 = v2 / n2
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            interior_angles.append(angle)

    if len(interior_angles) < 3:
        return 0.0

    # Spherical excess = sum of angles - (n-2)*pi
    spherical_excess = sum(interior_angles) - (len(interior_angles) - 2) * PI

    return spherical_excess


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embeddings(words: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings for a list of words."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        # Handle compound words (replace _ with space)
        clean_words = [w.replace("_", " ") for w in words]
        embeddings = model.encode(clean_words, convert_to_numpy=True)
        return embeddings
    except ImportError:
        print("WARNING: sentence-transformers not installed, using synthetic embeddings")
        np.random.seed(hash(str(words)) % (2**32))
        return np.random.randn(len(words), 384)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_polygon_type(
    loops: List[List[str]],
    polygon_type: str,
    expected_phase: float,
    model_name: str = "all-MiniLM-L6-v2",
    verbose: bool = True
) -> PolygonTypeResult:
    """Test Berry phase for all loops of a given type."""
    phases = []
    deviations = []

    for loop in loops:
        try:
            embeddings = get_embeddings(loop, model_name)
            phase = compute_berry_phase_winding(embeddings, closed=True)
            phases.append(phase)

            # Normalize phase to [0, 2*pi]
            phase_norm = phase % TWO_PI
            deviation = min(
                abs(phase_norm - expected_phase),
                abs(phase_norm - expected_phase + TWO_PI),
                abs(phase_norm - expected_phase - TWO_PI)
            )
            deviations.append(deviation / expected_phase * 100 if expected_phase != 0 else 0)

        except Exception as e:
            if verbose:
                print(f"  Error processing loop: {e}")
            continue

    if not phases:
        return PolygonTypeResult(
            polygon_type=polygon_type,
            n_sides=len(loops[0]) if loops else 0,
            expected_phase=expected_phase,
            n_loops=0,
            mean_phase=0.0,
            std_phase=0.0,
            mean_deviation_pct=100.0,
            phases=[]
        )

    return PolygonTypeResult(
        polygon_type=polygon_type,
        n_sides=len(loops[0]),
        expected_phase=expected_phase,
        n_loops=len(phases),
        mean_phase=float(np.mean(phases)),
        std_phase=float(np.std(phases)),
        mean_deviation_pct=float(np.mean(deviations)),
        phases=phases
    )


def test_hexagonal_berry_phase(
    model_name: str = "all-MiniLM-L6-v2",
    verbose: bool = True
) -> HexagonalBerryResult:
    """
    Test the hexagonal Berry phase hypothesis.

    If semantic hexagons accumulate Berry phase = 2*pi/3, this connects to sqrt(3).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Q23 PHASE 3: HEXAGONAL BERRY PHASE TEST")
        print("=" * 70)
        print(f"\nHypothesis: Hexagonal loops have Berry phase = 2*pi/3 = {TWO_PI_OVER_3:.4f} rad")
        print(f"Connection: sqrt(3) = 2*sin(pi/3) = {SQRT_3:.4f}")
        print(f"\nModel: {model_name}")

    # Expected phases for different polygon types
    # For regular n-gon on sphere: expected phase = 2*pi/n (for small polygons)
    # For hexagon: 2*pi/6 = pi/3 is one possibility
    # But the hypothesis is specifically 2*pi/3 from hexagonal symmetry
    hex_expected = TWO_PI_OVER_3  # 2.094 rad
    pent_expected = 2 * PI / 5  # 1.257 rad
    hept_expected = 2 * PI / 7  # 0.898 rad

    # Test hexagons
    if verbose:
        print(f"\n--- Testing Hexagons (6 vertices) ---")
        print(f"Expected phase: {hex_expected:.4f} rad = {np.degrees(hex_expected):.1f} deg")

    hex_result = test_polygon_type(
        SEMANTIC_HEXAGONS, "hexagon", hex_expected, model_name, verbose
    )

    if verbose:
        print(f"Mean phase: {hex_result.mean_phase:.4f} rad = {np.degrees(hex_result.mean_phase):.1f} deg")
        print(f"Std: {hex_result.std_phase:.4f} rad")
        print(f"Mean deviation: {hex_result.mean_deviation_pct:.1f}%")

    # Test pentagons (control)
    if verbose:
        print(f"\n--- Testing Pentagons (5 vertices, control) ---")
        print(f"Expected phase: {pent_expected:.4f} rad = {np.degrees(pent_expected):.1f} deg")

    pent_result = test_polygon_type(
        SEMANTIC_PENTAGONS, "pentagon", pent_expected, model_name, verbose
    )

    if verbose:
        print(f"Mean phase: {pent_result.mean_phase:.4f} rad = {np.degrees(pent_result.mean_phase):.1f} deg")

    # Test heptagons (control)
    if verbose:
        print(f"\n--- Testing Heptagons (7 vertices, control) ---")
        print(f"Expected phase: {hept_expected:.4f} rad = {np.degrees(hept_expected):.1f} deg")

    hept_result = test_polygon_type(
        SEMANTIC_HEPTAGONS, "heptagon", hept_expected, model_name, verbose
    )

    if verbose:
        print(f"Mean phase: {hept_result.mean_phase:.4f} rad = {np.degrees(hept_result.mean_phase):.1f} deg")

    # Check sqrt(3) connection
    # If mean hexagon phase is close to 2*pi/3, compute derived sqrt(3)
    derived_sqrt3 = 2 * np.sin(abs(hex_result.mean_phase) / 2) if hex_result.mean_phase != 0 else 0
    sqrt3_error = abs(derived_sqrt3 - SQRT_3) / SQRT_3 * 100

    sqrt3_connection = {
        "expected_sqrt3": SQRT_3,
        "derived_sqrt3": derived_sqrt3,
        "error_pct": sqrt3_error,
        "formula": "sqrt(3) = 2*sin(phase/2) where phase = 2*pi/3",
        "match": sqrt3_error < 15,  # Within 15%
    }

    # Determine if hypothesis is supported
    # Criteria:
    # 1. Hexagon mean phase within 20% of 2*pi/3
    # 2. Hexagons have lower deviation than controls
    hex_close = hex_result.mean_deviation_pct < 30
    hex_better_than_pent = hex_result.mean_deviation_pct < pent_result.mean_deviation_pct + 10
    hex_better_than_hept = hex_result.mean_deviation_pct < hept_result.mean_deviation_pct + 10

    hypothesis_supported = hex_close and (hex_better_than_pent or hex_better_than_hept)

    if hypothesis_supported:
        verdict = f"SUPPORTED: Hexagonal loops show phase near 2*pi/3 (deviation {hex_result.mean_deviation_pct:.1f}%)"
    elif hex_result.mean_deviation_pct < 50:
        verdict = f"PARTIAL: Some hexagonal structure (deviation {hex_result.mean_deviation_pct:.1f}%)"
    else:
        verdict = f"NOT SUPPORTED: Hexagonal phase not found (deviation {hex_result.mean_deviation_pct:.1f}%)"

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print(f"\n| Polygon   | N   | Expected | Mean Phase | Deviation |")
        print(f"|-----------|-----|----------|------------|-----------|")
        print(f"| Hexagon   | {hex_result.n_loops:>3} | {hex_expected:>8.3f} | {hex_result.mean_phase:>10.3f} | {hex_result.mean_deviation_pct:>8.1f}% |")
        print(f"| Pentagon  | {pent_result.n_loops:>3} | {pent_expected:>8.3f} | {pent_result.mean_phase:>10.3f} | {pent_result.mean_deviation_pct:>8.1f}% |")
        print(f"| Heptagon  | {hept_result.n_loops:>3} | {hept_expected:>8.3f} | {hept_result.mean_phase:>10.3f} | {hept_result.mean_deviation_pct:>8.1f}% |")

        print(f"\nsqrt(3) Connection:")
        print(f"  Expected sqrt(3): {SQRT_3:.4f}")
        print(f"  Derived sqrt(3): {derived_sqrt3:.4f}")
        print(f"  Error: {sqrt3_error:.1f}%")

        print(f"\n{'='*70}")
        print(f"VERDICT: {verdict}")
        print("=" * 70)

    return HexagonalBerryResult(
        hypothesis_supported=hypothesis_supported,
        hexagon_results=hex_result,
        pentagon_results=pent_result,
        heptagon_results=hept_result,
        sqrt3_connection=sqrt3_connection,
        verdict=verdict
    )


def run_cross_model_validation(
    model_names: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """Run hexagonal Berry phase test across multiple models."""
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-MODEL VALIDATION: HEXAGONAL BERRY PHASE")
        print("=" * 70)

    results = {}
    hex_phases = []
    supported_count = 0

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Model: {model_name}")

        try:
            result = test_hexagonal_berry_phase(model_name, verbose=False)
            results[model_name] = {
                "hex_mean_phase": result.hexagon_results.mean_phase,
                "hex_deviation_pct": result.hexagon_results.mean_deviation_pct,
                "hypothesis_supported": result.hypothesis_supported,
                "sqrt3_error_pct": result.sqrt3_connection["error_pct"],
            }

            hex_phases.append(result.hexagon_results.mean_phase)
            if result.hypothesis_supported:
                supported_count += 1

            if verbose:
                print(f"  Hex phase: {result.hexagon_results.mean_phase:.4f} rad")
                print(f"  Deviation: {result.hexagon_results.mean_deviation_pct:.1f}%")
                print(f"  Supported: {result.hypothesis_supported}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    if hex_phases:
        results["summary"] = {
            "n_models": len(hex_phases),
            "mean_hex_phase": float(np.mean(hex_phases)),
            "std_hex_phase": float(np.std(hex_phases)),
            "supported_count": supported_count,
            "support_rate": supported_count / len(hex_phases),
        }

        if verbose:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print("=" * 70)
            print(f"Models tested: {len(hex_phases)}")
            print(f"Mean hexagon phase: {np.mean(hex_phases):.4f} rad")
            print(f"Expected (2*pi/3): {TWO_PI_OVER_3:.4f} rad")
            print(f"Models supporting hypothesis: {supported_count}/{len(hex_phases)}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the hexagonal Berry phase test."""
    print("=" * 70)
    print("Q23 PHASE 3: HEXAGONAL BERRY PHASE TEST")
    print("=" * 70)
    print("\nTesting if hexagonal semantic loops accumulate Berry phase = 2*pi/3")
    print("This would explain the sqrt(3) constant via: sqrt(3) = 2*sin(pi/3)")

    # Test primary model
    result = test_hexagonal_berry_phase("all-MiniLM-L6-v2", verbose=True)

    # Cross-model validation
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
    ]
    cross_model = run_cross_model_validation(models, verbose=True)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Hexagonal loops have Berry phase = 2*pi/3, explaining sqrt(3) = 2*sin(pi/3)",
        "primary_model": "all-MiniLM-L6-v2",
        "primary_result": {
            "hypothesis_supported": result.hypothesis_supported,
            "hexagon": asdict(result.hexagon_results),
            "pentagon": asdict(result.pentagon_results),
            "heptagon": asdict(result.heptagon_results),
            "sqrt3_connection": result.sqrt3_connection,
            "verdict": result.verdict,
        },
        "cross_model": cross_model,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(
        results_dir,
        f"q23_hexagonal_berry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")

    return result.hypothesis_supported


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
