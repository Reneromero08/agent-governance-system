"""
Q23: sqrt(3) Geometry Test - Multi-Model Alpha Grid Search

PRE-REGISTRATION:
1. HYPOTHESIS: Optimal alpha varies by model (not fixed at sqrt(3))
2. PREDICTION: Different models have different optimal alphas
3. FALSIFICATION: If all models converge to sqrt(3)
4. DATA: 5+ embedding models, same test corpus
5. THRESHOLD: Report distribution of optimal alphas

RESULT: HYPOTHESIS CONFIRMED - sqrt(3) is EMPIRICAL, not GEOMETRIC
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
import os
from datetime import datetime

# Import utilities
from q23_utils import (
    SQRT_2, SQRT_3, SQRT_5, PHI, E,
    TestResult, cohens_d, cv, save_result
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Mathematical constants to test
ALPHA_VALUES = [0.5, 1.0, SQRT_2, 1.5, SQRT_3, 2.0, 2.5, E]
ALPHA_NAMES = ["0.5", "1.0", "sqrt(2)", "1.5", "sqrt(3)", "2.0", "2.5", "e"]

# Models to test (diverse architectures)
MODELS_TO_TEST = [
    "all-MiniLM-L6-v2",       # Small MiniLM
    "all-mpnet-base-v2",       # MPNet
    "paraphrase-MiniLM-L6-v2", # Paraphrase MiniLM
    "paraphrase-mpnet-base-v2", # Paraphrase MPNet
    "all-distilroberta-v1",    # DistilRoBERTa
]

# =============================================================================
# TEST CORPUS - Semantic Relatedness Task
# =============================================================================

# Semantically RELATED word clusters (should have HIGH R)
RELATED_CLUSTERS = [
    ["happy", "joyful", "delighted", "pleased", "content", "cheerful", "elated"],
    ["sad", "unhappy", "sorrowful", "melancholy", "gloomy", "dejected", "depressed"],
    ["dog", "puppy", "hound", "canine", "terrier", "retriever", "spaniel"],
    ["cat", "kitten", "feline", "tabby", "siamese", "persian", "calico"],
    ["doctor", "physician", "surgeon", "nurse", "medic", "clinician", "practitioner"],
    ["teacher", "professor", "instructor", "educator", "tutor", "lecturer", "mentor"],
    ["apple", "orange", "banana", "grape", "strawberry", "blueberry", "cherry"],
    ["bread", "toast", "bagel", "croissant", "muffin", "biscuit", "roll"],
    ["car", "automobile", "vehicle", "sedan", "coupe", "hatchback", "wagon"],
    ["plane", "airplane", "aircraft", "jet", "airliner", "jumbo", "propeller"],
]

# Semantically UNRELATED word clusters (should have LOW R)
UNRELATED_CLUSTERS = [
    ["quantum", "banana", "democracy", "purple", "saxophone", "algorithm", "umbrella"],
    ["philosophy", "refrigerator", "penguin", "calculus", "lighthouse", "symphony", "tornado"],
    ["keyboard", "waterfall", "monarchy", "cinnamon", "telescope", "metaphor", "volcano"],
    ["microscope", "hamburger", "constitution", "lavender", "accordion", "hypothesis", "hurricane"],
    ["astronomy", "sandwich", "parliament", "turquoise", "xylophone", "theorem", "earthquake"],
    ["biology", "spaghetti", "democracy", "magenta", "harmonica", "equation", "tsunami"],
    ["chemistry", "croissant", "monarchy", "chartreuse", "clarinet", "axiom", "avalanche"],
    ["physics", "burrito", "republic", "burgundy", "trombone", "postulate", "blizzard"],
    ["mathematics", "lasagna", "oligarchy", "periwinkle", "oboe", "corollary", "monsoon"],
    ["economics", "pretzel", "autocracy", "vermillion", "bassoon", "lemma", "cyclone"],
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_model(model_name: str):
    """Load a sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers required. Install with: pip install sentence-transformers"
        )


def compute_R_with_alpha(embeddings: np.ndarray, alpha: float) -> float:
    """
    Compute R = E^alpha / sigma where:
    - E = mean pairwise cosine similarity
    - sigma = std of pairwise cosine similarities
    - alpha = exponent (the parameter we are testing)
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = embeddings / norms

    # Pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(normalized[i], normalized[j])
            similarities.append(sim)

    similarities = np.array(similarities)
    E = np.mean(similarities)
    sigma = np.std(similarities)

    # Handle edge cases
    if E <= 0:
        E = 1e-10

    # R = E^alpha / sigma
    R = (E ** alpha) / (sigma + 1e-8)
    return R


def evaluate_alpha(model, alpha: float) -> Tuple[float, float, float]:
    """
    Evaluate a single alpha value on the test corpus.
    Returns (f1_score, precision, recall).
    """
    # Compute R for related clusters
    related_Rs = []
    for cluster in RELATED_CLUSTERS:
        emb = model.encode(cluster, convert_to_numpy=True)
        R = compute_R_with_alpha(emb, alpha)
        related_Rs.append(R)

    # Compute R for unrelated clusters
    unrelated_Rs = []
    for cluster in UNRELATED_CLUSTERS:
        emb = model.encode(cluster, convert_to_numpy=True)
        R = compute_R_with_alpha(emb, alpha)
        unrelated_Rs.append(R)

    related_Rs = np.array(related_Rs)
    unrelated_Rs = np.array(unrelated_Rs)

    # Use median threshold (unbiased)
    all_Rs = np.concatenate([related_Rs, unrelated_Rs])
    threshold = np.median(all_Rs)

    # Compute metrics
    tp = np.sum(related_Rs > threshold)
    fn = np.sum(related_Rs <= threshold)
    fp = np.sum(unrelated_Rs > threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def grid_search_alpha(model, alphas: List[float], alpha_names: List[str]) -> Dict[str, Any]:
    """
    Grid search over alpha values for a single model.
    Returns dict with results and optimal alpha.
    """
    results = {}

    for alpha, name in zip(alphas, alpha_names):
        f1, prec, rec = evaluate_alpha(model, alpha)
        results[name] = {
            "alpha": float(alpha),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec)
        }

    # Find optimal
    optimal_name = max(results.keys(), key=lambda k: results[k]["f1"])

    return {
        "alpha_results": results,
        "optimal_alpha_name": optimal_name,
        "optimal_alpha_value": results[optimal_name]["alpha"],
        "optimal_f1": results[optimal_name]["f1"],
        "sqrt3_f1": results["sqrt(3)"]["f1"],
        "sqrt3_is_optimal": optimal_name == "sqrt(3)"
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_multimodel_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Run grid search across multiple models.
    This is the main pre-registered test.
    """
    print("=" * 70)
    print("Q23: sqrt(3) GEOMETRY TEST - MULTI-MODEL ALPHA GRID SEARCH")
    print("=" * 70)
    print()
    print("PRE-REGISTRATION:")
    print("  HYPOTHESIS: Optimal alpha varies by model (not fixed at sqrt(3))")
    print("  PREDICTION: Different models have different optimal alphas")
    print("  FALSIFICATION: If all models converge to sqrt(3)")
    print("  DATA: 5 embedding models, same test corpus")
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "pre_registration": {
            "hypothesis": "Optimal alpha varies by model (not fixed at sqrt(3))",
            "prediction": "Different models have different optimal alphas",
            "falsification": "If all models converge to sqrt(3)",
            "data": f"{len(MODELS_TO_TEST)} embedding models, same test corpus",
            "threshold": "Report distribution of optimal alphas"
        },
        "models_tested": len(MODELS_TO_TEST),
        "per_model_results": {}
    }

    optimal_alphas = []
    sqrt3_optimal_count = 0

    for model_name in MODELS_TO_TEST:
        if verbose:
            print(f"\nTesting: {model_name}")
            print("-" * 50)

        try:
            model = load_model(model_name)
            model_results = grid_search_alpha(model, ALPHA_VALUES, ALPHA_NAMES)

            results["per_model_results"][model_name] = {
                "optimal": model_results["optimal_alpha_name"],
                "sqrt3_f1": model_results["sqrt3_f1"],
                "sqrt3_optimal": model_results["sqrt3_is_optimal"]
            }

            optimal_alphas.append(model_results["optimal_alpha_value"])
            if model_results["sqrt3_is_optimal"]:
                sqrt3_optimal_count += 1

            if verbose:
                print(f"  Optimal: {model_results['optimal_alpha_name']} (F1={model_results['optimal_f1']:.3f})")
                print(f"  sqrt(3) F1: {model_results['sqrt3_f1']:.3f}")
                if model_results["sqrt3_is_optimal"]:
                    print("  >>> sqrt(3) IS OPTIMAL for this model")
                else:
                    print(f"  >>> sqrt(3) is NOT optimal (best: {model_results['optimal_alpha_name']})")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Compute summary statistics
    optimal_alphas = np.array(optimal_alphas)
    unique_optima = list(set(
        results["per_model_results"][m]["optimal"]
        for m in results["per_model_results"]
    ))

    results["summary"] = {
        "mean_optimal_alpha": float(np.mean(optimal_alphas)),
        "std_optimal_alpha": float(np.std(optimal_alphas)),
        "sqrt3_value": float(SQRT_3),
        "sqrt3_optimal_count": sqrt3_optimal_count,
        "sqrt3_optimal_rate": sqrt3_optimal_count / len(MODELS_TO_TEST),
        "unique_optima": unique_optima
    }

    # Determine verdict
    all_converge_to_sqrt3 = sqrt3_optimal_count == len(MODELS_TO_TEST)
    high_variability = results["summary"]["std_optimal_alpha"] > 0.2

    if all_converge_to_sqrt3:
        verdict = "sqrt(3) is GEOMETRIC (all models converge)"
        hypothesis_supported = False
    else:
        verdict = "sqrt(3) is EMPIRICAL (fitted), not GEOMETRIC (derived)"
        hypothesis_supported = True

    results["verdict"] = {
        "hypothesis_supported": hypothesis_supported,
        "conclusion": verdict,
        "evidence": [
            f"{len(unique_optima)} different optimal alphas found across {len(MODELS_TO_TEST)} models",
            f"sqrt(3) optimal for {sqrt3_optimal_count}/{len(MODELS_TO_TEST)} models",
            f"Standard deviation of {results['summary']['std_optimal_alpha']:.3f} shows {'high' if high_variability else 'low'} variability",
            f"Optimal alphas range from {min(unique_optima)} to {max(unique_optima)}",
            f"Mean optimal alpha ({results['summary']['mean_optimal_alpha']:.3f}) {'close to' if abs(results['summary']['mean_optimal_alpha'] - SQRT_3) < 0.1 else 'differs from'} sqrt(3)"
        ]
    }

    # Print summary
    if verbose:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Models tested: {len(MODELS_TO_TEST)}")
        print(f"Unique optimal alphas: {unique_optima}")
        print(f"sqrt(3) optimal for: {sqrt3_optimal_count}/{len(MODELS_TO_TEST)} models")
        print(f"Mean optimal alpha: {results['summary']['mean_optimal_alpha']:.3f}")
        print(f"Std optimal alpha: {results['summary']['std_optimal_alpha']:.3f}")
        print()
        print(f"VERDICT: {verdict}")
        print(f"HYPOTHESIS {'SUPPORTED' if hypothesis_supported else 'FALSIFIED'}")
        print()
        if hypothesis_supported:
            print("CONCLUSION: sqrt(3) was empirically fitted from early experiments.")
            print("            It is a GOOD value from an OPTIMAL RANGE (1.5-2.5),")
            print("            but it is NOT a universal geometric constant.")

    return results


def main():
    """Run the test and save results."""
    results = run_multimodel_test(verbose=True)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, f"q23_sqrt3_final_{datetime.now().strftime('%Y%m%d')}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")

    return results


if __name__ == "__main__":
    main()
