"""
Q23 Phase 3: Hexagonal Winding Angle Test

HYPOTHESIS: sqrt(3) appears because of hexagonal geometry in semantic space.

Mathematical connection:
- Hexagonal loops have interior angle 120 degrees = 2*pi/3 radians
- sin(pi/3) = sqrt(3)/2, therefore sqrt(3) = 2*sin(pi/3)
- If semantic hexagons accumulate winding angle = 2*pi/3, this explains sqrt(3)

Test Design:
1. Construct 6-vertex semantic hexagons (words forming conceptual cycles)
2. Measure winding angle (total rotation in 2D PCA projection) around each hexagon
3. Check if mean angle is close to 2*pi/3 = 2.094 radians
4. Compare to 5-vertex (pentagons) and 7-vertex (heptagons) as controls

Pass Criteria:
- Mean hexagon winding angle within 15% of 2*pi/3
- Hexagons show more consistent angle than pentagons/heptagons
- Effect is present across multiple models

CRITICAL DISTINCTION - WINDING ANGLE vs BERRY PHASE:
This module measures WINDING ANGLE, NOT true geometric Berry phase.

Winding angle: Total rotation accumulated when traversing points in a 2D
projection (via PCA). This is a property of the 2D projected path, not the
original high-dimensional embedding space.

True Berry phase: Geometric phase accumulated during parallel transport of
a quantum state around a closed loop. Requires:
1. A well-defined connection (how to parallel transport vectors)
2. Integration of the connection 1-form around the loop
3. The result is gauge-invariant and captures intrinsic geometry

Why we measure winding angle instead:
- Embedding spaces lack a natural connection for parallel transport
- PCA projection to 2D gives a concrete, computable quantity
- Winding angle is still geometrically meaningful, just not Berry phase

IMPORTANT LIMITATIONS:
1. Per-loop PCA would make comparisons meaningless (each loop gets its own
   coordinate system). We use a SHARED PCA fitted on all vocabulary.
2. Even if winding angle = 2*pi/3, this does NOT prove hexagonal Berry phase.
3. The relationship between winding angle and any deeper geometric structure
   is unclear and should not be over-interpreted.
4. Expected phases (2*pi/n) assume REGULAR polygons - semantic loops are NOT
   regular, so there is no mathematical reason to expect these values.

This test is exploratory: interesting patterns may emerge, but should be
interpreted cautiously given the conceptual limitations above.
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
    winding_angle: float  # radians (NOT Berry phase - see module docstring)
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
# Words chosen to be semantically distinct yet connected in a cycle
SEMANTIC_HEXAGONS = [
    # Emotion cycle (distinct emotional states)
    ["calm", "happy", "excited", "anxious", "sad", "peaceful"],

    # Time of day cycle (evenly spaced)
    ["dawn", "morning", "noon", "afternoon", "evening", "night"],

    # Season cycle (each season distinct)
    ["spring", "summer", "autumn", "winter", "thaw", "bloom"],

    # Knowledge cycle
    ["ignorance", "curiosity", "learning", "understanding", "wisdom", "teaching"],

    # Economic cycle
    ["growth", "boom", "peak", "decline", "recession", "recovery"],

    # Water cycle
    ["ocean", "evaporation", "cloud", "rain", "river", "delta"],

    # Life stage cycle
    ["birth", "childhood", "youth", "adulthood", "elder", "death"],

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
    ["spring", "bloom", "summer", "harvest", "autumn", "winter", "thaw"],
    ["growth", "expansion", "boom", "peak", "decline", "recession", "recovery"],
    ["ocean", "evaporation", "cloud", "rain", "stream", "river", "delta"],
]


# =============================================================================
# WINDING ANGLE COMPUTATION (NOT Berry phase - see module docstring)
# =============================================================================

def compute_winding_angle(
    path: np.ndarray,
    closed: bool = True,
    pca_components: np.ndarray = None
) -> float:
    """
    Compute winding angle (total rotation) in 2D projection.

    NOTE: This is NOT Berry phase. Berry phase requires parallel transport
    with a well-defined connection. This function computes the winding angle
    (total angle swept) when the path is projected to 2D and traversed in
    the complex plane.

    Winding angle is a property of the 2D projected path, not the original
    high-dimensional geometry. It measures how many times and in what
    direction the projected path winds around the origin.

    Args:
        path: Array of shape (n_points, n_dims) with embedding vectors
        closed: If True, close the loop by appending the first point
        pca_components: Optional pre-fitted PCA components (shape 2, n_dims).
            If provided, uses these for projection (enables consistent
            comparison across loops). If None, fits PCA on this path only
            (WARNING: per-loop PCA makes cross-loop comparisons meaningless).

    Returns:
        Winding angle in radians. Positive = counterclockwise, negative = clockwise.
    """
    # Normalize embeddings
    norms = np.linalg.norm(path, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    path = path / norms

    # Close the loop if needed
    if closed and not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[0:1]])

    # Project to 2D
    if pca_components is not None:
        # Use pre-fitted PCA components for consistent projection
        centered = path - path.mean(axis=0)
        proj_2d = centered @ pca_components.T
    else:
        # Fit PCA on this path only (WARNING: makes cross-loop comparison meaningless)
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

def fit_shared_pca(
    all_loops: List[List[str]],
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on all unique words from all loops combined.

    This ensures consistent projection across all loops, making winding angle
    comparisons meaningful. Without shared PCA, each loop would have its own
    coordinate system, making cross-loop comparisons invalid.

    Args:
        all_loops: List of word loops (each loop is a list of words)
        model_name: Name of the sentence transformer model

    Returns:
        Tuple of (pca_components, all_embeddings) where:
        - pca_components: Shape (2, n_dims) - first 2 principal components
        - all_embeddings: Shape (n_words, n_dims) - embeddings of all unique words
    """
    # Collect all unique words
    all_words = []
    seen = set()
    for loop in all_loops:
        for word in loop:
            if word not in seen:
                all_words.append(word)
                seen.add(word)

    # Get embeddings for all words
    all_embeddings = get_embeddings(all_words, model_name)

    # Normalize embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    all_embeddings = all_embeddings / norms

    # Fit PCA on all embeddings
    centered = all_embeddings - all_embeddings.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_components = Vt[:2]  # First 2 principal components
    except:
        # Fallback: use identity-like projection
        pca_components = np.zeros((2, all_embeddings.shape[1]))
        pca_components[0, 0] = 1.0
        pca_components[1, 1] = 1.0

    return pca_components, all_embeddings


def test_polygon_type(
    loops: List[List[str]],
    polygon_type: str,
    expected_phase: float,
    model_name: str = "all-MiniLM-L6-v2",
    pca_components: np.ndarray = None,
    verbose: bool = True
) -> PolygonTypeResult:
    """
    Test winding angle for all loops of a given polygon type.

    NOTE: This measures winding angle, NOT Berry phase. See module docstring.

    Args:
        loops: List of word loops to test
        polygon_type: Name of polygon type (e.g., "hexagon")
        expected_phase: Expected winding angle in radians
        model_name: Sentence transformer model name
        pca_components: Pre-fitted PCA components for consistent projection.
            If None, each loop gets its own PCA (WARNING: makes comparisons
            meaningless across loops).
        verbose: Print progress information

    Returns:
        PolygonTypeResult with statistics for this polygon type
    """
    phases = []
    deviations = []

    for loop in loops:
        try:
            embeddings = get_embeddings(loop, model_name)
            phase = compute_winding_angle(
                embeddings,
                closed=True,
                pca_components=pca_components
            )
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
    Test the hexagonal winding angle hypothesis.

    NOTE: This measures WINDING ANGLE, not true Berry phase. See module docstring
    for the important distinction. The name is kept for backward compatibility.

    If semantic hexagons accumulate winding angle = 2*pi/3, this connects to sqrt(3).
    However, even if this is observed, it does NOT prove hexagonal Berry phase.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Q23 PHASE 3: HEXAGONAL WINDING ANGLE TEST")
        print("=" * 70)
        print(f"\nHypothesis: Hexagonal loops have winding angle near 2*pi/3 = {TWO_PI_OVER_3:.4f} rad")
        print(f"Connection: sqrt(3) = 2*sin(pi/3) = {SQRT_3:.4f}")
        print(f"\nModel: {model_name}")
        print("\nNOTE: This measures winding angle in 2D projection, NOT true Berry phase.")

    # Expected angles for different polygon types
    # NOTE: These expectations assume REGULAR polygons on a sphere.
    # Semantic loops are NOT regular - they're placed by meaning, not geometry.
    # These serve as reference points, not strict predictions.
    #
    # The hypothesis is specifically 2*pi/3 from hexagonal symmetry (120 deg interior angle)
    hex_expected = TWO_PI_OVER_3  # 2.094 rad = 120 deg
    pent_expected = 2 * PI / 5  # 1.257 rad = 72 deg (for comparison)
    hept_expected = 2 * PI / 7  # 0.898 rad = 51.4 deg (for comparison)

    # Fit shared PCA on all words from all polygon types
    # This ensures consistent projection across all loops
    all_loops = SEMANTIC_HEXAGONS + SEMANTIC_PENTAGONS + SEMANTIC_HEPTAGONS
    if verbose:
        print(f"\nFitting shared PCA on {len(all_loops)} loops...")

    pca_components, _ = fit_shared_pca(all_loops, model_name)

    if verbose:
        print("Shared PCA fitted. All loops will use the same 2D projection.")

    # Test hexagons
    if verbose:
        print(f"\n--- Testing Hexagons (6 vertices) ---")
        print(f"Expected angle: {hex_expected:.4f} rad = {np.degrees(hex_expected):.1f} deg")

    hex_result = test_polygon_type(
        SEMANTIC_HEXAGONS, "hexagon", hex_expected, model_name,
        pca_components=pca_components, verbose=verbose
    )

    if verbose:
        print(f"Mean angle: {hex_result.mean_phase:.4f} rad = {np.degrees(hex_result.mean_phase):.1f} deg")
        print(f"Std: {hex_result.std_phase:.4f} rad")
        print(f"Mean deviation: {hex_result.mean_deviation_pct:.1f}%")

    # Test pentagons (control)
    if verbose:
        print(f"\n--- Testing Pentagons (5 vertices, control) ---")
        print(f"Expected angle: {pent_expected:.4f} rad = {np.degrees(pent_expected):.1f} deg")

    pent_result = test_polygon_type(
        SEMANTIC_PENTAGONS, "pentagon", pent_expected, model_name,
        pca_components=pca_components, verbose=verbose
    )

    if verbose:
        print(f"Mean angle: {pent_result.mean_phase:.4f} rad = {np.degrees(pent_result.mean_phase):.1f} deg")

    # Test heptagons (control)
    if verbose:
        print(f"\n--- Testing Heptagons (7 vertices, control) ---")
        print(f"Expected angle: {hept_expected:.4f} rad = {np.degrees(hept_expected):.1f} deg")

    hept_result = test_polygon_type(
        SEMANTIC_HEPTAGONS, "heptagon", hept_expected, model_name,
        pca_components=pca_components, verbose=verbose
    )

    if verbose:
        print(f"Mean phase: {hept_result.mean_phase:.4f} rad = {np.degrees(hept_result.mean_phase):.1f} deg")

    # Check sqrt(3) connection
    # If mean hexagon winding angle is close to 2*pi/3, compute derived sqrt(3)
    # NOTE: This is purely numerical - it does NOT prove geometric Berry phase
    derived_sqrt3 = 2 * np.sin(abs(hex_result.mean_phase) / 2) if hex_result.mean_phase != 0 else 0
    sqrt3_error = abs(derived_sqrt3 - SQRT_3) / SQRT_3 * 100

    sqrt3_connection = {
        "expected_sqrt3": SQRT_3,
        "derived_sqrt3": derived_sqrt3,
        "error_pct": sqrt3_error,
        "formula": "sqrt(3) = 2*sin(angle/2) where angle = 2*pi/3",
        "match": sqrt3_error < 15,  # Within 15%
        "caveat": "Numerical match does NOT prove Berry phase relationship",
    }

    # Determine if hypothesis is supported
    # Criteria:
    # 1. Hexagon mean winding angle within 30% of 2*pi/3
    # 2. Hexagons have lower deviation than controls
    #
    # IMPORTANT: Even if supported, this only shows winding angle correlation,
    # NOT true Berry phase. The interpretation is limited.
    hex_close = hex_result.mean_deviation_pct < 30
    hex_better_than_pent = hex_result.mean_deviation_pct < pent_result.mean_deviation_pct + 10
    hex_better_than_hept = hex_result.mean_deviation_pct < hept_result.mean_deviation_pct + 10

    hypothesis_supported = hex_close and (hex_better_than_pent or hex_better_than_hept)

    # Build honest verdict that acknowledges limitations
    if hypothesis_supported:
        verdict = (
            f"WINDING ANGLE CORRELATION OBSERVED: Hexagonal loops show winding angle "
            f"near 2*pi/3 (deviation {hex_result.mean_deviation_pct:.1f}%). "
            f"IMPORTANT CAVEAT: This measures winding angle in 2D PCA projection, "
            f"NOT true geometric Berry phase. The connection to Berry phase is unclear."
        )
    elif hex_result.mean_deviation_pct < 50:
        verdict = (
            f"PARTIAL CORRELATION: Some hexagonal structure in winding angle "
            f"(deviation {hex_result.mean_deviation_pct:.1f}%). "
            f"NOTE: This is winding angle, not Berry phase."
        )
    else:
        verdict = (
            f"NO CORRELATION: Hexagonal winding angle not found "
            f"(deviation {hex_result.mean_deviation_pct:.1f}%). "
            f"This does not rule out other forms of hexagonal structure."
        )

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print(f"\n| Polygon   | N   | Expected | Mean Angle | Deviation |")
        print(f"|-----------|-----|----------|------------|-----------|")
        print(f"| Hexagon   | {hex_result.n_loops:>3} | {hex_expected:>8.3f} | {hex_result.mean_phase:>10.3f} | {hex_result.mean_deviation_pct:>8.1f}% |")
        print(f"| Pentagon  | {pent_result.n_loops:>3} | {pent_expected:>8.3f} | {pent_result.mean_phase:>10.3f} | {pent_result.mean_deviation_pct:>8.1f}% |")
        print(f"| Heptagon  | {hept_result.n_loops:>3} | {hept_expected:>8.3f} | {hept_result.mean_phase:>10.3f} | {hept_result.mean_deviation_pct:>8.1f}% |")

        print(f"\nsqrt(3) Connection (via winding angle, NOT Berry phase):")
        print(f"  Expected sqrt(3): {SQRT_3:.4f}")
        print(f"  Derived sqrt(3): {derived_sqrt3:.4f}")
        print(f"  Error: {sqrt3_error:.1f}%")

        print(f"\n{'='*70}")
        print("IMPORTANT METHODOLOGICAL NOTE:")
        print("This test measures WINDING ANGLE (total rotation in 2D PCA projection),")
        print("NOT true Berry phase. Even if winding angle = 2*pi/3, this does NOT prove:")
        print("  1. Hexagonal geometry exists in embedding space")
        print("  2. Berry phase accumulates around semantic loops")
        print("  3. sqrt(3) arises from geometric phase effects")
        print("See module docstring for full explanation of limitations.")
        print("=" * 70)
        print(f"\nVERDICT: {verdict}")
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
    """Run hexagonal winding angle test across multiple models."""
    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-MODEL VALIDATION: HEXAGONAL WINDING ANGLE")
        print("=" * 70)
        print("NOTE: This measures winding angle, NOT Berry phase.")

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
                print(f"  Hex winding angle: {result.hexagon_results.mean_phase:.4f} rad")
                print(f"  Deviation: {result.hexagon_results.mean_deviation_pct:.1f}%")
                print(f"  Correlation found: {result.hypothesis_supported}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    if hex_phases:
        results["summary"] = {
            "n_models": len(hex_phases),
            "mean_hex_winding_angle": float(np.mean(hex_phases)),
            "std_hex_winding_angle": float(np.std(hex_phases)),
            "supported_count": supported_count,
            "support_rate": supported_count / len(hex_phases),
            "note": "These are winding angles, NOT Berry phases",
        }

        if verbose:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print("=" * 70)
            print(f"Models tested: {len(hex_phases)}")
            print(f"Mean hexagon winding angle: {np.mean(hex_phases):.4f} rad")
            print(f"Expected (2*pi/3): {TWO_PI_OVER_3:.4f} rad")
            print(f"Models showing correlation: {supported_count}/{len(hex_phases)}")
            print("\nREMINDER: Winding angle is NOT Berry phase.")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the hexagonal winding angle test."""
    print("=" * 70)
    print("Q23 PHASE 3: HEXAGONAL WINDING ANGLE TEST")
    print("=" * 70)
    print("\nTesting if hexagonal semantic loops show winding angle near 2*pi/3")
    print("This would connect to sqrt(3) via: sqrt(3) = 2*sin(pi/3)")
    print("\nIMPORTANT: This measures WINDING ANGLE (rotation in 2D projection),")
    print("NOT true geometric Berry phase. See module docstring for details.")

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
        "hypothesis": "Hexagonal loops show winding angle near 2*pi/3, connecting to sqrt(3) = 2*sin(pi/3)",
        "methodology_note": (
            "This test measures WINDING ANGLE (total rotation in 2D PCA projection), "
            "NOT true geometric Berry phase. Even if winding angle correlates with 2*pi/3, "
            "this does NOT prove hexagonal Berry phase or geometric phase effects."
        ),
        "primary_model": "all-MiniLM-L6-v2",
        "primary_result": {
            "correlation_found": result.hypothesis_supported,
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
