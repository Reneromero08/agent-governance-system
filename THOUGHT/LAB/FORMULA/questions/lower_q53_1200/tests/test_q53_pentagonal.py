#!/usr/bin/env python3
"""
Q53: Pentagonal Phi Geometry - Does Embedding Space Exhibit 5-Fold Symmetry?

PRE-REGISTRATION:
1. HYPOTHESIS: Embedding space has icosahedral (5-fold) symmetry
2. PREDICTIONS:
   - Concept angles cluster at ~72 deg (360/5)
   - 5-fold rotational symmetry in PCA projections
3. FALSIFICATION: If angles uniformly distributed (no phi/72-deg signature)
4. DATA: Multiple embedding models, same corpus
5. THRESHOLD: 5-fold signature detected in 3+ models

BACKGROUND:
The golden ratio phi = (1 + sqrt(5)) / 2 appears in:
- Icosahedral symmetry (20 triangular faces, 5-fold rotational axes)
- Penrose tilings (aperiodic with 5-fold local symmetry)
- Quasi-crystals (5-fold diffraction patterns)

If semantic embedding spaces exhibit similar structure, we'd expect:
- Preferred angles at 72 degrees (360/5) and 144 degrees (2*72)
- Peaks in the angle distribution histogram at phi-related values
- PCA eigenspectrum following phi-related ratios

Run: python test_q53_pentagonal.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Callable
from itertools import combinations
from collections import Counter

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ~1.618
PENTAGON_ANGLE = 72.0  # 360 / 5 degrees
PENTAGON_ANGLE_RAD = np.radians(72.0)

# Key phi-related angles (degrees)
PHI_ANGLES = [
    36.0,   # 180/5
    72.0,   # 360/5 - MAIN SIGNATURE
    108.0,  # 3*36
    144.0,  # 2*72
    180.0,  # Pi
]

# Tolerance for angle matching (degrees)
ANGLE_TOLERANCE = 5.0


# =============================================================================
# TEST CORPUS - Semantic Categories
# =============================================================================

# Words organized by semantic categories for embedding
TEST_CORPUS = {
    "animals": ["dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger", "bear", "wolf"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown"],
    "emotions": ["happy", "sad", "angry", "fear", "love", "hate", "joy", "grief", "hope", "despair"],
    "science": ["physics", "chemistry", "biology", "math", "astronomy", "geology", "medicine", "engineering", "computer", "research"],
    "food": ["bread", "meat", "fruit", "vegetable", "rice", "pasta", "cheese", "milk", "egg", "fish"],
    "nature": ["tree", "flower", "mountain", "river", "ocean", "forest", "desert", "rain", "sun", "moon"],
    "actions": ["run", "walk", "jump", "swim", "fly", "climb", "throw", "catch", "push", "pull"],
    "objects": ["chair", "table", "book", "phone", "car", "house", "door", "window", "key", "clock"],
}


def get_all_words() -> List[str]:
    """Get all unique words from corpus."""
    words = []
    for category_words in TEST_CORPUS.values():
        words.extend(category_words)
    return list(set(words))


# =============================================================================
# EMBEDDING MODELS
# =============================================================================

def create_mock_embedder(dim: int = 384, seed: int = 42) -> Callable:
    """Create deterministic mock embedder for testing."""
    rng = np.random.default_rng(seed)
    cache = {}

    def embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for text in texts:
            if text not in cache:
                # Hash-based deterministic embedding
                h = hash(text) % (2**32)
                local_rng = np.random.default_rng(h)
                vec = local_rng.standard_normal(dim)
                vec = vec / np.linalg.norm(vec)
                cache[text] = vec
            result.append(cache[text])
        return np.array(result)

    return embed


def load_embedding_models() -> Dict[str, Callable]:
    """Load multiple embedding models for cross-validation."""
    models = {}

    # Try sentence-transformers models
    try:
        from sentence_transformers import SentenceTransformer

        model_names = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
        ]

        for name in model_names:
            try:
                model = SentenceTransformer(name)
                models[name] = lambda x, m=model: m.encode(x, normalize_embeddings=True)
                print(f"  [OK] Loaded {name}")
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")

    except ImportError:
        print("  [WARN] sentence-transformers not available")

    # Try gensim Word2Vec if available
    try:
        import gensim.downloader as api
        w2v = api.load("word2vec-google-news-300")

        def w2v_embed(words):
            result = []
            for w in words:
                try:
                    vec = w2v[w]
                    vec = vec / np.linalg.norm(vec)
                    result.append(vec)
                except KeyError:
                    result.append(np.zeros(300))
            return np.array(result)

        models["word2vec-google-news"] = w2v_embed
        print("  [OK] Loaded word2vec-google-news")
    except Exception as e:
        print(f"  [SKIP] word2vec: {e}")

    # Always add mock embedder for baseline comparison
    models["mock-random"] = create_mock_embedder(dim=384, seed=42)
    print("  [OK] Loaded mock-random (baseline)")

    # Add another mock with different seed for variance estimation
    models["mock-random-2"] = create_mock_embedder(dim=384, seed=123)
    print("  [OK] Loaded mock-random-2 (variance baseline)")

    return models


# =============================================================================
# ANGLE COMPUTATION
# =============================================================================

def compute_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in degrees."""
    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

    # Cosine similarity
    cos_sim = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    # Convert to degrees
    angle_rad = np.arccos(cos_sim)
    return np.degrees(angle_rad)


def compute_pairwise_angles(embeddings: np.ndarray) -> np.ndarray:
    """Compute all pairwise angles between embedding vectors."""
    n = embeddings.shape[0]
    angles = []

    for i, j in combinations(range(n), 2):
        angle = compute_angle_degrees(embeddings[i], embeddings[j])
        angles.append(angle)

    return np.array(angles)


def compute_angle_histogram(angles: np.ndarray, bins: int = 36) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram of angles (0-180 degrees)."""
    hist, bin_edges = np.histogram(angles, bins=bins, range=(0, 180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist, bin_centers


# =============================================================================
# PENTAGONAL SYMMETRY TESTS
# =============================================================================

def test_72_degree_clustering(angles: np.ndarray) -> Dict:
    """
    TEST 1: Do angles cluster around 72 degrees (360/5)?

    Measure density at phi-related angles vs uniform expectation.
    """
    # Count angles near each phi-related angle
    phi_counts = {}
    for target in PHI_ANGLES:
        near_target = np.abs(angles - target) < ANGLE_TOLERANCE
        phi_counts[target] = int(np.sum(near_target))

    # Total angles
    total = len(angles)

    # Expected if uniform: (ANGLE_TOLERANCE * 2 / 180) * total for each target
    expected_uniform = (ANGLE_TOLERANCE * 2 / 180) * total

    # Key metric: excess at 72 degrees
    excess_72 = phi_counts[72.0] / (expected_uniform + 1) - 1.0

    # Chi-squared like statistic
    chi_sq_contributions = {}
    for angle, count in phi_counts.items():
        chi_sq_contributions[angle] = ((count - expected_uniform) ** 2) / (expected_uniform + 1)

    # Pass if 72-degree peak is significantly above uniform
    passes = excess_72 > 0.1  # At least 10% excess

    return {
        "test": "72_DEGREE_CLUSTERING",
        "phi_counts": phi_counts,
        "expected_uniform": expected_uniform,
        "excess_72_deg": excess_72,
        "chi_sq_contributions": chi_sq_contributions,
        "total_angles": total,
        "pass": passes
    }


def test_phi_ratio_in_spectrum(embeddings: np.ndarray) -> Dict:
    """
    TEST 2: Does the PCA eigenspectrum show phi-related ratios?

    In icosahedral symmetry, eigenvalue ratios should cluster near phi.
    """
    # Center the data
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Only use positive eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return {
            "test": "PHI_RATIO_SPECTRUM",
            "error": "Not enough positive eigenvalues",
            "pass": False
        }

    # Compute consecutive ratios
    ratios = []
    for i in range(len(eigenvalues) - 1):
        ratio = eigenvalues[i] / (eigenvalues[i+1] + 1e-10)
        ratios.append(float(ratio))

    ratios = np.array(ratios)

    # Count ratios near phi and 1/phi
    near_phi = np.abs(ratios - PHI) < 0.1
    near_inv_phi = np.abs(ratios - 1/PHI) < 0.1
    phi_related = np.sum(near_phi) + np.sum(near_inv_phi)

    # Expected if random: very few (phi is ~1.618, most ratios would be ~1)
    phi_density = phi_related / len(ratios)

    # Pass if at least 5% of ratios are phi-related
    passes = phi_density > 0.05

    return {
        "test": "PHI_RATIO_SPECTRUM",
        "n_eigenvalues": len(eigenvalues),
        "n_ratios": len(ratios),
        "ratios_near_phi": int(np.sum(near_phi)),
        "ratios_near_inv_phi": int(np.sum(near_inv_phi)),
        "phi_density": phi_density,
        "top_5_ratios": [float(r) for r in ratios[:5]],
        "pass": passes
    }


def test_5fold_pca_symmetry(embeddings: np.ndarray) -> Dict:
    """
    TEST 3: Does PCA projection show 5-fold rotational symmetry?

    Project to 2D and check if angle distribution from centroid has 5-fold structure.
    """
    # Center
    centered = embeddings - embeddings.mean(axis=0)

    # PCA to 2D
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top_2 = eigenvectors[:, idx[:2]]

    projected = centered @ top_2

    # Compute angles from centroid (which is at origin after centering)
    angles_from_center = np.arctan2(projected[:, 1], projected[:, 0])
    angles_degrees = np.degrees(angles_from_center) % 360

    # Histogram with 5 bins to test 5-fold symmetry
    hist_5, _ = np.histogram(angles_degrees, bins=5, range=(0, 360))

    # For perfect 5-fold symmetry, all bins should be equal
    # Use coefficient of variation as measure
    cv = np.std(hist_5) / (np.mean(hist_5) + 1e-10)

    # Also check 6-fold for comparison (should be worse if 5-fold is real)
    hist_6, _ = np.histogram(angles_degrees, bins=6, range=(0, 360))
    cv_6 = np.std(hist_6) / (np.mean(hist_6) + 1e-10)

    # Pass if 5-fold CV is lower than 6-fold (5-fold fits better)
    passes = cv < cv_6

    return {
        "test": "5FOLD_PCA_SYMMETRY",
        "hist_5fold": [int(x) for x in hist_5],
        "hist_6fold": [int(x) for x in hist_6],
        "cv_5fold": float(cv),
        "cv_6fold": float(cv_6),
        "5fold_better_than_6fold": cv < cv_6,
        "pass": passes
    }


def test_golden_angle_prevalence(angles: np.ndarray) -> Dict:
    """
    TEST 4: Is the golden angle (~137.5 degrees) prevalent?

    The golden angle is 360 * (1 - 1/phi) = 137.5 degrees.
    It appears in phyllotaxis (leaf arrangements) and may indicate
    optimal packing in embedding space.
    """
    GOLDEN_ANGLE = 360 * (1 - 1/PHI)  # ~137.5 degrees

    # Count angles near golden angle
    near_golden = np.abs(angles - GOLDEN_ANGLE) < ANGLE_TOLERANCE
    golden_count = int(np.sum(near_golden))

    # Expected uniform
    total = len(angles)
    expected = (ANGLE_TOLERANCE * 2 / 180) * total

    excess = golden_count / (expected + 1) - 1.0

    passes = excess > 0.05  # At least 5% excess

    return {
        "test": "GOLDEN_ANGLE_PREVALENCE",
        "golden_angle": GOLDEN_ANGLE,
        "count_near_golden": golden_count,
        "expected_uniform": expected,
        "excess": excess,
        "pass": passes
    }


def test_icosahedral_angles(angles: np.ndarray) -> Dict:
    """
    TEST 5: Do angles match icosahedral geometry?

    Icosahedral angles between vertices:
    - 63.43 degrees (adjacent vertices)
    - 116.57 degrees (non-adjacent on same face)
    - 180 degrees (antipodal)
    """
    ICOSA_ANGLES = [
        63.43,   # arccos(1/sqrt(5))
        116.57,  # 180 - 63.43
        180.0,
    ]

    icosa_counts = {}
    total = len(angles)

    for target in ICOSA_ANGLES:
        near = np.abs(angles - target) < ANGLE_TOLERANCE
        icosa_counts[target] = int(np.sum(near))

    expected = (ANGLE_TOLERANCE * 2 / 180) * total

    total_icosa = sum(icosa_counts.values())
    expected_total = expected * len(ICOSA_ANGLES)

    excess = total_icosa / (expected_total + 1) - 1.0

    passes = excess > 0.1

    return {
        "test": "ICOSAHEDRAL_ANGLES",
        "icosa_angles": ICOSA_ANGLES,
        "counts": icosa_counts,
        "total_icosa": total_icosa,
        "expected": expected_total,
        "excess": excess,
        "pass": passes
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_tests_for_model(
    model_name: str,
    embed_fn: Callable,
    words: List[str]
) -> Dict:
    """Run all tests for a single embedding model."""
    print(f"\n  Testing {model_name}...")

    # Get embeddings
    try:
        embeddings = embed_fn(words)
        if embeddings is None or len(embeddings) == 0:
            return {"model": model_name, "error": "No embeddings returned"}
        embeddings = np.array(embeddings)
    except Exception as e:
        return {"model": model_name, "error": str(e)}

    # Filter out zero vectors
    norms = np.linalg.norm(embeddings, axis=1)
    valid_mask = norms > 1e-10
    embeddings = embeddings[valid_mask]

    if len(embeddings) < 10:
        return {"model": model_name, "error": f"Only {len(embeddings)} valid embeddings"}

    # Compute pairwise angles
    angles = compute_pairwise_angles(embeddings)

    # Run all tests
    results = {
        "model": model_name,
        "n_embeddings": len(embeddings),
        "n_angles": len(angles),
        "angle_mean": float(np.mean(angles)),
        "angle_std": float(np.std(angles)),
        "tests": {}
    }

    # Test 1: 72-degree clustering
    results["tests"]["clustering_72"] = test_72_degree_clustering(angles)

    # Test 2: Phi ratio in spectrum
    results["tests"]["phi_spectrum"] = test_phi_ratio_in_spectrum(embeddings)

    # Test 3: 5-fold PCA symmetry
    results["tests"]["pca_5fold"] = test_5fold_pca_symmetry(embeddings)

    # Test 4: Golden angle
    results["tests"]["golden_angle"] = test_golden_angle_prevalence(angles)

    # Test 5: Icosahedral angles
    results["tests"]["icosahedral"] = test_icosahedral_angles(angles)

    # Summary
    n_passed = sum(1 for t in results["tests"].values() if t.get("pass", False))
    results["tests_passed"] = n_passed
    results["tests_total"] = len(results["tests"])

    return results


def run_all_tests():
    """Run all pentagonal geometry tests."""
    print("=" * 70)
    print("Q53: PENTAGONAL PHI GEOMETRY - 5-FOLD SYMMETRY TEST")
    print("=" * 70)
    print()
    print(f"Golden ratio phi = {PHI:.6f}")
    print(f"Pentagon angle = {PENTAGON_ANGLE} degrees")
    print()

    # Load models
    print("Loading embedding models...")
    models = load_embedding_models()
    print(f"\nLoaded {len(models)} models")

    # Get test corpus
    words = get_all_words()
    print(f"Test corpus: {len(words)} words from {len(TEST_CORPUS)} categories")

    # Run tests for each model
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    all_results = []
    for model_name, embed_fn in models.items():
        result = run_tests_for_model(model_name, embed_fn, words)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<30} | {'Passed':>8} | {'72-deg':>8} | {'Phi-spec':>8} | {'5-fold':>8}")
    print("-" * 70)

    models_with_signal = 0

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<30} | ERROR: {r['error']}")
            continue

        tests = r["tests"]
        passed = r["tests_passed"]
        total = r["tests_total"]

        t72 = "PASS" if tests["clustering_72"]["pass"] else "FAIL"
        tphi = "PASS" if tests["phi_spectrum"]["pass"] else "FAIL"
        t5f = "PASS" if tests["pca_5fold"]["pass"] else "FAIL"

        print(f"{r['model']:<30} | {passed}/{total:>6} | {t72:>8} | {tphi:>8} | {t5f:>8}")

        # Count models with 5-fold signal
        if passed >= 2:  # At least 2 tests pass
            models_with_signal += 1

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    # Threshold: 3+ models show 5-fold signature
    n_trained = len([r for r in all_results if "mock" not in r.get("model", "")])
    n_mock = len([r for r in all_results if "mock" in r.get("model", "")])

    # Check if trained models show more signal than random
    trained_signals = []
    mock_signals = []

    for r in all_results:
        if "error" in r:
            continue
        passed = r.get("tests_passed", 0)
        if "mock" in r.get("model", ""):
            mock_signals.append(passed)
        else:
            trained_signals.append(passed)

    avg_trained = np.mean(trained_signals) if trained_signals else 0
    avg_mock = np.mean(mock_signals) if mock_signals else 0

    print(f"Trained models avg tests passed: {avg_trained:.2f}")
    print(f"Mock/random models avg tests passed: {avg_mock:.2f}")
    print(f"Difference: {avg_trained - avg_mock:.2f}")
    print()

    if avg_trained > avg_mock + 0.5 and len(trained_signals) >= 3:
        verdict = "SUPPORTED"
        explanation = (
            f"Trained embedding models show stronger 5-fold symmetry signal "
            f"({avg_trained:.2f} tests) vs random baselines ({avg_mock:.2f} tests). "
            "This suggests embedding spaces may exhibit phi-related geometric structure."
        )
    elif avg_trained > avg_mock:
        verdict = "WEAK_SIGNAL"
        explanation = (
            f"Trained models show slightly more 5-fold signal ({avg_trained:.2f}) "
            f"than random ({avg_mock:.2f}), but not statistically compelling. "
            "More models and larger corpus needed."
        )
    elif len(trained_signals) < 3:
        verdict = "INSUFFICIENT_DATA"
        explanation = (
            f"Only {len(trained_signals)} trained models available. "
            "Need at least 3 trained models for cross-validation. "
            "Install more sentence-transformer models or gensim."
        )
    else:
        verdict = "FALSIFIED"
        explanation = (
            f"Trained models ({avg_trained:.2f}) show no more 5-fold symmetry "
            f"than random embeddings ({avg_mock:.2f}). "
            "Hypothesis of pentagonal phi geometry not supported."
        )

    print(f"VERDICT: {verdict}")
    print()
    print(explanation)

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Embedding space has icosahedral (5-fold) symmetry",
        "phi": PHI,
        "pentagon_angle": PENTAGON_ANGLE,
        "n_words": len(words),
        "n_models_total": len(models),
        "n_trained_models": n_trained,
        "n_mock_models": n_mock,
        "avg_trained_passed": avg_trained,
        "avg_mock_passed": avg_mock,
        "verdict": verdict,
        "explanation": explanation,
        "model_results": all_results,
    }

    output_path = Path(__file__).parent / "q53_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return verdict != "FALSIFIED"


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
