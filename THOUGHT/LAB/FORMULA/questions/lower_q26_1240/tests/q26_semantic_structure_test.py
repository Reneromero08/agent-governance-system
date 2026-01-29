#!/usr/bin/env python3
"""
Q26 Follow-up: Does N_min depend on SEMANTIC STRUCTURE of the data?

The main test found N_min=3 uniformly across models and dimensions.
But that was with semantically coherent text.

CRITICAL QUESTION: Does semantic coherence affect N_min?

We test:
1. Coherent text (topic clusters) - expected: low N_min
2. Random/noisy text (gibberish) - expected: higher N_min?
3. Adversarial text (contradictions) - expected: ???

Author: Claude (rigorous investigation)
Date: 2026-01-27
"""

import numpy as np
import json
from typing import Dict, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


def compute_R(embeddings: np.ndarray) -> float:
    """Compute R from embeddings using centroid-based method."""
    if len(embeddings) < 2:
        return 1.0

    truth_vector = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - truth_vector, axis=1)
    mean_dist = np.mean(distances)

    if mean_dist < 1e-10:
        return float(len(embeddings))

    sigma = mean_dist
    z = distances / sigma
    E = np.mean(np.exp(-0.5 * z**2))
    cv = np.std(distances) / (mean_dist + 1e-10)
    concentration = 1.0 / (1.0 + cv)
    R = float(E * concentration / sigma)

    return max(0.0, min(R, 1000.0))


def test_stability(embeddings: np.ndarray, N: int, n_trials: int = 50) -> float:
    """Return CV at sample size N."""
    if N > len(embeddings):
        N = len(embeddings)

    R_values = []
    for trial in range(n_trials):
        np.random.seed(trial * 1000 + N)
        idx = np.random.choice(len(embeddings), N, replace=False)
        R = compute_R(embeddings[idx])
        R_values.append(R)

    R_mean = np.mean(R_values)
    R_std = np.std(R_values)
    return R_std / (R_mean + 1e-10)


def find_N_min(embeddings: np.ndarray, threshold: float = 0.10) -> int:
    """Find minimum N where CV < threshold."""
    N_candidates = [3, 5, 7, 10, 15, 20, 30, 50]

    for N in N_candidates:
        if N > len(embeddings):
            break
        cv = test_stability(embeddings, N)
        if cv < threshold:
            return N

    return N_candidates[-1]


def generate_coherent_corpus(n: int = 100) -> List[str]:
    """Generate semantically coherent text (topical clusters)."""
    topics = [
        "The sun rises in the east and sets in the west.",
        "Gravity pulls objects toward the earth.",
        "Water flows from high to low elevation.",
        "Plants need sunlight to grow.",
        "Birds fly south for the winter.",
        "Fish live in water and breathe through gills.",
        "Humans need food and water to survive.",
        "The moon orbits the earth.",
        "Stars shine because of nuclear fusion.",
        "Clouds form when water vapor condenses.",
    ]

    extended = []
    prefixes = ["Indeed, ", "Actually, ", "In fact, ", "Clearly, "]

    for text in topics:
        extended.append(text)
        for prefix in prefixes:
            extended.append(prefix + text.lower())

    while len(extended) < n:
        extended.extend(topics)

    return extended[:n]


def generate_random_corpus(n: int = 100) -> List[str]:
    """Generate random/gibberish text (no semantic structure)."""
    import random
    random.seed(42)

    words = [
        "xkcd", "blorp", "zymurgy", "quaffle", "flibbertigibbet",
        "kerfuffle", "bumfuzzle", "lollygag", "snickersnee", "widdershins",
        "cattywampus", "gardyloo", "taradiddle", "collywobbles", "bibble"
    ]

    texts = []
    for i in range(n):
        random.seed(i * 100)
        length = random.randint(5, 15)
        sentence = " ".join(random.choices(words, k=length))
        texts.append(sentence + ".")

    return texts


def generate_diverse_corpus(n: int = 100) -> List[str]:
    """Generate maximally diverse text (different topics)."""
    topics = [
        "Quantum physics describes subatomic behavior.",
        "Shakespeare wrote Romeo and Juliet.",
        "The economy influences stock markets.",
        "Elephants are the largest land animals.",
        "Pizza is a popular Italian food.",
        "Democracy is a form of government.",
        "The Mona Lisa hangs in the Louvre.",
        "Neurons transmit electrical signals.",
        "Basketball was invented in 1891.",
        "Climate change affects global temperatures.",
        "DNA carries genetic information.",
        "Mozart composed classical symphonies.",
        "Volcanoes erupt with molten lava.",
        "Computers process binary data.",
        "Poetry uses rhythmic language.",
    ]

    extended = []
    while len(extended) < n:
        extended.extend(topics)

    return extended[:n]


def generate_contradictory_corpus(n: int = 100) -> List[str]:
    """Generate self-contradicting statements."""
    pairs = [
        ("The sky is blue.", "The sky is not blue."),
        ("Water is wet.", "Water is completely dry."),
        ("Fire is hot.", "Fire is cold as ice."),
        ("Dogs bark.", "Dogs are completely silent."),
        ("Trees are tall.", "Trees are very short."),
        ("Snow is white.", "Snow is pitch black."),
        ("Music is audible.", "Music cannot be heard."),
        ("Books contain words.", "Books contain no words."),
    ]

    texts = []
    for a, b in pairs:
        texts.append(a)
        texts.append(b)

    while len(texts) < n:
        texts.extend([p[0] for p in pairs] + [p[1] for p in pairs])

    return texts[:n]


def run_semantic_structure_test():
    """Compare N_min across different semantic structures."""
    print("=" * 80)
    print("Q26: SEMANTIC STRUCTURE EFFECT ON N_min")
    print("=" * 80)

    if not ST_AVAILABLE:
        print("ERROR: SentenceTransformer required")
        return None

    model_name = "all-MiniLM-L6-v2"
    print(f"\nUsing model: {model_name}")
    model = SentenceTransformer(model_name)

    corpus_types = {
        "coherent": generate_coherent_corpus(100),
        "diverse": generate_diverse_corpus(100),
        "contradictory": generate_contradictory_corpus(100),
        "random_gibberish": generate_random_corpus(100),
    }

    results = {}
    N_candidates = [3, 5, 7, 10, 15, 20, 30, 50]

    print("\n" + "-" * 60)
    print("TESTING DIFFERENT CORPUS TYPES")
    print("-" * 60)

    for corpus_type, texts in corpus_types.items():
        print(f"\n{corpus_type.upper()}:")

        embeddings = model.encode(texts, show_progress_bar=False)

        stability_curve = {}
        for N in N_candidates:
            cv = test_stability(embeddings, N)
            stability_curve[N] = cv

        N_min = find_N_min(embeddings)

        results[corpus_type] = {
            "N_min": N_min,
            "stability_curve": stability_curve,
            "D": embeddings.shape[1]
        }

        print(f"  N_min: {N_min}")
        print(f"  CV at N=3: {stability_curve[3]:.4f}")
        print(f"  CV at N=10: {stability_curve[10]:.4f}")

    # Analysis
    print("\n" + "-" * 60)
    print("ANALYSIS")
    print("-" * 60)

    N_mins = [r["N_min"] for r in results.values()]
    N_min_cv = np.std(N_mins) / (np.mean(N_mins) + 1e-10)

    print(f"\nN_min values: {dict((k, v['N_min']) for k, v in results.items())}")
    print(f"N_min mean: {np.mean(N_mins):.1f}")
    print(f"N_min CV: {N_min_cv:.2%}")

    if N_min_cv < 0.10:
        verdict = "NO_EFFECT"
        explanation = "Semantic structure does NOT affect N_min significantly"
    elif max(N_mins) > 2 * min(N_mins):
        verdict = "SIGNIFICANT_EFFECT"
        explanation = f"N_min varies significantly: {min(N_mins)} to {max(N_mins)}"
    else:
        verdict = "MINOR_EFFECT"
        explanation = f"N_min shows minor variation: {min(N_mins)} to {max(N_mins)}"

    print(f"\nVERDICT: {verdict}")
    print(f"Explanation: {explanation}")

    return {
        "test_id": "Q26_SEMANTIC_STRUCTURE_TEST",
        "model": model_name,
        "results": results,
        "analysis": {
            "N_min_values": N_mins,
            "N_min_mean": float(np.mean(N_mins)),
            "N_min_cv": float(N_min_cv),
            "verdict": verdict,
            "explanation": explanation
        }
    }


if __name__ == "__main__":
    results = run_semantic_structure_test()

    if results:
        output_path = Path(__file__).parent / "q26_semantic_structure_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to: {output_path}")
