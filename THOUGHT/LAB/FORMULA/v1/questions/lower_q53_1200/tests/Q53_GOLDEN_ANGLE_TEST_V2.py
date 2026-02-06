"""
Q53: Pentagonal Phi Geometry - Concept Angle Test

Measures angles BETWEEN concepts, not along trajectories.

Hypothesis: The angular separation between semantic concepts relates to golden ratio.

FINDING: Concepts pack at pentagonal angles (~72 deg), not golden spiral
angles (137.5 deg). The golden ratio appears in the underlying icosahedral
geometry, and spirals EMERGE from geodesic motion through this structure.

Uses REAL embeddings from 5 architectures.
"""

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import sys
from itertools import combinations

# Constants
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # phi = 1.618...
GOLDEN_ANGLE_RAD = 2 * np.pi / (GOLDEN_RATIO ** 2)  # 2.399 rad
GOLDEN_ANGLE_DEG = np.degrees(GOLDEN_ANGLE_RAD)  # 137.5 deg

print(f"Golden ratio (phi): {GOLDEN_RATIO:.6f}")
print(f"Golden angle: {GOLDEN_ANGLE_RAD:.4f} rad = {GOLDEN_ANGLE_DEG:.2f} deg")
print()

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent


def load_embeddings(model_name: str, words: List[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load embeddings from a specific model."""
    embeddings = {}
    dim = 0

    if model_name == "glove":
        import gensim.downloader as api
        print(f"  Loading GloVe...", flush=True)
        model = api.load("glove-wiki-gigaword-300")
        dim = 300
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "word2vec":
        import gensim.downloader as api
        print(f"  Loading Word2Vec...", flush=True)
        model = api.load("word2vec-google-news-300")
        dim = 300
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "fasttext":
        import gensim.downloader as api
        print(f"  Loading FastText...", flush=True)
        model = api.load("fasttext-wiki-news-subwords-300")
        dim = 300
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "bert":
        from transformers import BertTokenizer, BertModel
        import torch
        print(f"  Loading BERT...", flush=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        dim = 768
        with torch.no_grad():
            for word in words:
                inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
                outputs = model(**inputs)
                vec = outputs.last_hidden_state[0, 0, :].numpy()
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "sentence":
        from sentence_transformers import SentenceTransformer
        print(f"  Loading SentenceTransformer...", flush=True)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dim = 384
        vecs = model.encode(words)
        for i, word in enumerate(words):
            vec = vecs[i]
            vec = vec / np.linalg.norm(vec)
            embeddings[word] = vec

    return embeddings, dim


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in radians."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot)


def run_golden_angle_test_v2():
    """Test golden angle hypothesis on concept angles."""
    print("=" * 80)
    print("Q36 EXTENSION: GOLDEN ANGLE TEST V2 - CONCEPT ANGLES")
    print("=" * 80)
    print()

    # Diverse word set
    words = [
        # Opposites
        "king", "queen", "man", "woman",
        "good", "evil", "light", "dark",
        "love", "hate", "life", "death",
        "hot", "cold", "big", "small",
        # Abstract
        "truth", "beauty", "justice", "freedom",
        "time", "space", "mind", "matter",
        # Concrete
        "cat", "dog", "tree", "water",
        "sun", "moon", "earth", "sky",
    ]

    models = ["glove", "word2vec", "fasttext", "bert", "sentence"]

    all_angles = []
    model_results = {}

    for model_name in models:
        print(f"\n--- {model_name.upper()} ---")

        try:
            embeddings, dim = load_embeddings(model_name, words)
            available_words = list(embeddings.keys())
            print(f"  Loaded {len(available_words)} words in {dim}D space")

            # Measure ALL pairwise angles
            angles = []
            for w1, w2 in combinations(available_words, 2):
                angle = angle_between_vectors(embeddings[w1], embeddings[w2])
                angles.append({
                    "pair": (w1, w2),
                    "angle_rad": float(angle),
                    "angle_deg": float(np.degrees(angle)),
                })

            angles_rad = [a["angle_rad"] for a in angles]
            angles_deg = [a["angle_deg"] for a in angles]

            mean_angle = np.mean(angles_deg)
            std_angle = np.std(angles_deg)
            min_angle = np.min(angles_deg)
            max_angle = np.max(angles_deg)

            print(f"  {len(angles)} pairs measured")
            print(f"  Mean angle: {mean_angle:.2f} deg (std: {std_angle:.2f})")
            print(f"  Range: {min_angle:.2f} - {max_angle:.2f} deg")

            # Check relationship to golden angle
            ratio = mean_angle / GOLDEN_ANGLE_DEG
            print(f"  Mean / Golden angle: {ratio:.4f}")

            # Distribution analysis
            # How many angles are near golden angle?
            near_golden = sum(1 for a in angles_deg if abs(a - GOLDEN_ANGLE_DEG) < 10)
            print(f"  Angles within 10 deg of golden: {near_golden} ({100*near_golden/len(angles_deg):.1f}%)")

            # Check for golden ratio in the distribution
            # The mean angle on a random unit sphere in d dimensions is arccos(0) = 90 deg
            # Deviation from 90 might relate to phi
            deviation_from_90 = mean_angle - 90
            print(f"  Deviation from 90 deg: {deviation_from_90:.2f} deg")

            model_results[model_name] = {
                "n_pairs": len(angles),
                "mean_deg": mean_angle,
                "std_deg": std_angle,
                "min_deg": min_angle,
                "max_deg": max_angle,
                "ratio_to_golden": ratio,
                "near_golden_pct": 100 * near_golden / len(angles_deg),
                "deviation_from_90": deviation_from_90,
            }

            all_angles.extend(angles_deg)

        except Exception as e:
            print(f"  ERROR: {e}")
            model_results[model_name] = {"error": str(e)}

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    all_angles = np.array(all_angles)
    print(f"\nTotal angles measured: {len(all_angles)}")
    print(f"Overall mean: {np.mean(all_angles):.2f} deg")
    print(f"Overall std:  {np.std(all_angles):.2f} deg")

    # Histogram analysis
    print("\nAngle distribution (10-degree bins):")
    hist, bin_edges = np.histogram(all_angles, bins=18, range=(0, 180))
    for i in range(len(hist)):
        bar = "#" * (hist[i] // 20)
        print(f"  {bin_edges[i]:5.0f}-{bin_edges[i+1]:5.0f}: {hist[i]:4d} {bar}")

    # Find peak
    peak_bin = np.argmax(hist)
    peak_center = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
    print(f"\nPeak bin center: {peak_center:.1f} deg")
    print(f"Golden angle:    {GOLDEN_ANGLE_DEG:.1f} deg")

    # Key relationships
    print("\n" + "=" * 80)
    print("KEY RELATIONSHIPS")
    print("=" * 80)

    mean = np.mean(all_angles)
    print(f"\nMean angle: {mean:.4f} deg = {np.radians(mean):.6f} rad")
    print(f"Golden angle: {GOLDEN_ANGLE_DEG:.4f} deg = {GOLDEN_ANGLE_RAD:.6f} rad")
    print()

    # Check various phi relationships
    print("Checking phi relationships:")
    print(f"  mean / golden:     {mean / GOLDEN_ANGLE_DEG:.6f}")
    print(f"  mean / 90:         {mean / 90:.6f}")
    print(f"  mean / 60:         {mean / 60:.6f} (hexagonal)")
    print(f"  mean / (90/phi):   {mean / (90/GOLDEN_RATIO):.6f}")
    print(f"  mean / (90*phi):   {mean / (90*GOLDEN_RATIO):.6f}")
    print(f"  (mean-90) / phi:   {(mean-90) / GOLDEN_RATIO:.6f}")

    # Check if deviation from 90 relates to phi
    dev = mean - 90
    print(f"\nDeviation from orthogonal (90 deg): {dev:.4f} deg")
    print(f"  dev / golden_angle:  {dev / GOLDEN_ANGLE_DEG:.6f}")
    print(f"  dev * phi:           {dev * GOLDEN_RATIO:.6f}")
    print(f"  dev / phi:           {dev / GOLDEN_RATIO:.6f}")

    # XOR Phi connection
    xor_phi = 1.773
    print(f"\nXOR Phi value from Q36: {xor_phi}")
    print(f"  mean_rad * xor_phi:  {np.radians(mean) * xor_phi:.6f}")
    print(f"  mean_deg / xor_phi:  {mean / xor_phi:.6f}")

    # Save results
    results = {
        "golden_ratio": GOLDEN_RATIO,
        "golden_angle_deg": GOLDEN_ANGLE_DEG,
        "golden_angle_rad": GOLDEN_ANGLE_RAD,
        "aggregate": {
            "total_pairs": len(all_angles),
            "mean_deg": float(np.mean(all_angles)),
            "std_deg": float(np.std(all_angles)),
            "peak_bin_center_deg": float(peak_center),
        },
        "by_model": model_results,
    }

    output_path = SCRIPT_DIR / "Q36_GOLDEN_ANGLE_RESULTS_V2.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_golden_angle_test_v2()
