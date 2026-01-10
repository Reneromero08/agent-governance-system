#!/usr/bin/env python3
"""
E.X.3.7: Boundary Discovery - Try to Break the Spectral Convergence

Tests:
1. Adversarial anchor sets (rare words, nonsense, mixed)
2. Fine-tuned models (sentiment, NLI, QA)
3. Minimal anchor set (how few anchors still work?)
"""

import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("WARNING: sentence-transformers not available")

# ============================================
# ANCHOR SETS
# ============================================

# Standard anchors (baseline)
STANDARD_ANCHORS = [
    "water", "fire", "earth", "air", "light", "dark", "love", "hate",
    "life", "death", "time", "space", "good", "evil", "true", "false",
    "big", "small", "hot", "cold", "fast", "slow", "old", "new",
    "happy", "sad", "rich", "poor", "strong", "weak", "high", "low",
    "king", "queen", "man", "woman", "child", "adult", "friend", "enemy",
    "day", "night", "sun", "moon", "star", "ocean", "mountain", "river",
    "food", "drink", "home", "work", "play", "rest", "war", "peace",
    "science", "art", "music", "dance", "book", "film", "game", "sport"
]

# Adversarial: Rare/archaic words
RARE_ANCHORS = [
    "ennui", "zeitgeist", "schadenfreude", "weltanschauung", "gestalt",
    "quixotic", "sesquipedalian", "defenestration", "pulchritudinous",
    "obstreperous", "sycophant", "magnanimous", "ephemeral", "ubiquitous",
    "serendipity", "eloquent", "paradigm", "nomenclature", "antediluvian",
    "perspicacious", "grandiloquent", "mendacious", "pusillanimous",
    "verisimilitude", "obsequious", "perfunctory", "recalcitrant",
    "sanguine", "truculent", "vicissitude", "supercilious", "laconic",
    "parsimonious", "loquacious", "circumlocution", "peremptory",
    "impecunious", "propinquity", "tergiversation", "callipygian",
    "borborygmus", "syzygy", "tmesis", "zeugma", "litotes", "hendiadys",
    "anaphora", "chiasmus", "hyperbaton", "anadiplosis", "epanalepsis",
    "epistrophe", "symploce", "polysyndeton", "asyndeton", "isocolon",
    "tricolon", "anastrophe", "parenthesis", "aporia", "prosopopoeia",
    "apostrophe", "catachresis", "metonymy", "synecdoche"
]

# Adversarial: Nonsense/made-up words
NONSENSE_ANCHORS = [
    "flurble", "grompf", "snazzlewort", "quibbleflax", "zortnik",
    "blimtron", "crundlewick", "daffernoodle", "ecklefritz", "fumblewump",
    "glarfington", "hozzlepop", "inklestink", "jabberwock", "kerfluffle",
    "lollygag", "mimblewimble", "noodlewhack", "oompaloompa", "pifflepaffle",
    "quagswaggle", "razzmatazz", "snickerdoodle", "thingamabob", "umptysquat",
    "vexillology", "whatchamacallit", "xyzzyspoon", "yabbadabbadoo", "zippitydoo",
    "blorpenschmidt", "cranglewitz", "drizzlefrank", "embuggerance", "flibbertigibbet",
    "gobbledygook", "hullabaloo", "jiggery-pokery", "katzenjammer", "lollygagger",
    "mumbo-jumbo", "namby-pamby", "oopsy-daisy", "poppycock", "riffraff",
    "shilly-shally", "topsy-turvy", "wishy-washy", "yackety-yak", "zigzag",
    "abracadabra", "alakazam", "bibimbap", "cockamamie", "doohickey",
    "fandango", "gewgaw", "higgledy-piggledy", "itsy-bitsy", "jeepers"
]

# Adversarial: Technical jargon
TECHNICAL_ANCHORS = [
    "eigenvalue", "tensor", "manifold", "topology", "homotopy",
    "diffeomorphism", "homeomorphism", "isomorphism", "endomorphism",
    "automorphism", "functor", "morphism", "category", "topos",
    "sheaf", "presheaf", "colimit", "limit", "adjunction", "monad",
    "kernel", "cokernel", "image", "preimage", "quotient",
    "subspace", "hyperplane", "orthogonal", "unitary", "hermitian",
    "symplectic", "kahler", "riemannian", "lorentzian", "minkowski",
    "hilbert", "banach", "frechet", "sobolev", "schwartz",
    "laplacian", "hessian", "jacobian", "gradient", "divergence",
    "curl", "flux", "potential", "gauge", "connection",
    "curvature", "torsion", "holonomy", "geodesic", "parallel",
    "covariant", "contravariant", "invariant", "equivariant", "bilinear",
    "multilinear", "antisymmetric", "symmetric", "skew"
]


def compute_cumulative_variance(embeddings: np.ndarray, k: int = 50) -> np.ndarray:
    """Compute cumulative variance curve from embeddings."""
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # Descending

    # Normalize and cumulate
    eigenvalues = eigenvalues[:k]
    total = np.sum(eigenvalues)
    normalized = eigenvalues / total
    cumulative = np.cumsum(normalized)

    return cumulative


def compute_correlation(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Compute Pearson correlation between two curves."""
    min_len = min(len(curve1), len(curve2))
    return float(np.corrcoef(curve1[:min_len], curve2[:min_len])[0, 1])


def test_adversarial_anchors(models: list) -> dict:
    """Test convergence with adversarial anchor sets."""
    print("\n" + "=" * 60)
    print("TEST 1: ADVERSARIAL ANCHOR SETS")
    print("=" * 60)

    anchor_sets = {
        "standard": STANDARD_ANCHORS,
        "rare": RARE_ANCHORS,
        "nonsense": NONSENSE_ANCHORS,
        "technical": TECHNICAL_ANCHORS,
        "mixed": STANDARD_ANCHORS[:16] + RARE_ANCHORS[:16] + NONSENSE_ANCHORS[:16] + TECHNICAL_ANCHORS[:16]
    }

    results = {}

    for set_name, anchors in anchor_sets.items():
        print(f"\n--- {set_name.upper()} ({len(anchors)} words) ---")

        # Get embeddings from all models
        curves = []
        for model in models:
            try:
                embeddings = model.encode(anchors)
                curve = compute_cumulative_variance(embeddings)
                curves.append(curve)
            except Exception as e:
                print(f"  Error with model: {e}")
                continue

        if len(curves) < 2:
            print(f"  Not enough models succeeded")
            results[set_name] = {"status": "failed", "correlations": []}
            continue

        # Compute pairwise correlations
        correlations = []
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                r = compute_correlation(curves[i], curves[j])
                correlations.append(r)

        mean_r = np.mean(correlations)
        std_r = np.std(correlations)

        print(f"  Mean correlation: {mean_r:.4f} (+/- {std_r:.4f})")
        print(f"  Min: {min(correlations):.4f}, Max: {max(correlations):.4f}")

        results[set_name] = {
            "n_anchors": len(anchors),
            "mean_correlation": float(mean_r),
            "std_correlation": float(std_r),
            "min_correlation": float(min(correlations)),
            "max_correlation": float(max(correlations)),
            "converges": bool(mean_r > 0.9)
        }

    return results


def test_finetuned_models() -> dict:
    """Test convergence with fine-tuned models."""
    print("\n" + "=" * 60)
    print("TEST 2: FINE-TUNED MODELS")
    print("=" * 60)

    # Models fine-tuned for different tasks
    model_configs = {
        "base": "all-MiniLM-L6-v2",
        "paraphrase": "paraphrase-MiniLM-L6-v2",
        "qa": "multi-qa-MiniLM-L6-cos-v1",
        "nli": "all-MiniLM-L6-v2",  # Base model for comparison
        "semantic_search": "msmarco-MiniLM-L6-cos-v5",
    }

    # Try to load additional fine-tuned models
    extra_models = [
        ("sentiment", "nlptown/bert-base-multilingual-uncased-sentiment"),
        ("ner", "dslim/bert-base-NER"),
    ]

    results = {}
    curves = {}

    for task, model_name in model_configs.items():
        print(f"\n--- {task.upper()} ({model_name}) ---")
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(STANDARD_ANCHORS)
            curve = compute_cumulative_variance(embeddings)
            curves[task] = curve
            print(f"  Loaded successfully, dim={embeddings.shape[1]}")
        except Exception as e:
            print(f"  Failed to load: {e}")

    # Compute pairwise correlations
    if len(curves) >= 2:
        correlations = []
        pairs = []
        tasks = list(curves.keys())

        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                r = compute_correlation(curves[tasks[i]], curves[tasks[j]])
                correlations.append(r)
                pairs.append((tasks[i], tasks[j], r))
                print(f"  {tasks[i]} <-> {tasks[j]}: {r:.4f}")

        mean_r = np.mean(correlations)
        print(f"\n  MEAN CORRELATION: {mean_r:.4f}")

        results = {
            "models_tested": list(curves.keys()),
            "pairwise_correlations": {f"{p[0]}_vs_{p[1]}": float(p[2]) for p in pairs},
            "mean_correlation": float(mean_r),
            "converges": bool(mean_r > 0.9)
        }
    else:
        results = {"status": "insufficient_models"}

    return results


def test_minimal_anchors(models: list) -> dict:
    """Test minimum number of anchors needed for convergence."""
    print("\n" + "=" * 60)
    print("TEST 3: MINIMAL ANCHOR SET")
    print("=" * 60)

    # Test different anchor set sizes
    sizes = [4, 8, 12, 16, 24, 32, 48, 64]

    results = {}

    for size in sizes:
        anchors = STANDARD_ANCHORS[:size]
        print(f"\n--- {size} ANCHORS ---")

        curves = []
        for model in models:
            try:
                embeddings = model.encode(anchors)
                if embeddings.shape[0] < 4:
                    continue
                curve = compute_cumulative_variance(embeddings, k=min(size-1, 50))
                curves.append(curve)
            except Exception as e:
                continue

        if len(curves) < 2:
            results[size] = {"status": "failed"}
            continue

        # Compute correlations
        correlations = []
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                r = compute_correlation(curves[i], curves[j])
                correlations.append(r)

        mean_r = np.mean(correlations)
        print(f"  Mean correlation: {mean_r:.4f}")

        results[size] = {
            "mean_correlation": float(mean_r),
            "converges": bool(mean_r > 0.9)
        }

    # Find minimum size that works
    min_working = None
    for size in sizes:
        if size in results and results[size].get("converges", False):
            min_working = size
            break

    print(f"\n  MINIMUM WORKING SIZE: {min_working}")

    return {
        "by_size": results,
        "minimum_working": min_working
    }


def main():
    print("=" * 60)
    print("E.X.3.7: BOUNDARY DISCOVERY")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not ST_AVAILABLE:
        print("ERROR: sentence-transformers required")
        return

    # Load models
    print("\nLoading models...")
    model_names = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
    ]

    models = []
    for name in model_names:
        try:
            model = SentenceTransformer(name)
            models.append(model)
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  Failed: {name} - {e}")

    if len(models) < 2:
        print("ERROR: Need at least 2 models")
        return

    results = {
        "timestamp": datetime.now().isoformat(),
        "models": model_names[:len(models)],
        "tests": {}
    }

    # Run tests
    results["tests"]["adversarial"] = test_adversarial_anchors(models)
    results["tests"]["finetuned"] = test_finetuned_models()
    results["tests"]["minimal"] = test_minimal_anchors(models)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    adv = results["tests"]["adversarial"]
    print("\nAdversarial Anchors:")
    for set_name, data in adv.items():
        if isinstance(data, dict) and "mean_correlation" in data:
            status = "CONVERGES" if data["converges"] else "BREAKS"
            print(f"  {set_name}: r={data['mean_correlation']:.3f} [{status}]")

    ft = results["tests"]["finetuned"]
    if "mean_correlation" in ft:
        status = "CONVERGES" if ft["converges"] else "BREAKS"
        print(f"\nFine-tuned Models: r={ft['mean_correlation']:.3f} [{status}]")

    mn = results["tests"]["minimal"]
    print(f"\nMinimal Anchors: {mn['minimum_working']} words needed")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "q34_boundary_discovery.json")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
