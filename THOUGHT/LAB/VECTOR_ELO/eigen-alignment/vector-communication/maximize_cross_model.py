#!/usr/bin/env python3
"""MAXIMIZE Cross-Model Alignment - Large Scale Experiment.

Goal: Get as close to 100% cross-model accuracy as possible.

Testing:
- Anchor sets: 32, 64, 128, 256, 512, 777
- k values: 16, 32, 48, 64, 96, 128
- Models: nomic, MiniLM, MPNet
- Corruption: 0%, 25%, 50%, 75%
- Trials: 50 per configuration

Usage:
    python maximize_cross_model.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.mds import squared_distance_matrix, classical_mds
from large_anchor_generator import (
    ANCHOR_128, ANCHOR_256, ANCHOR_512,
    generate_anchor_set, compute_anchor_hash
)
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_32, STABLE_64


# =============================================================================
# Setup
# =============================================================================

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount('http://', adapter)
    return session


SESSION = create_session()
EMBED_CACHE = {}


def get_embeddings_cached(texts, url, model):
    """Get embeddings with batching and caching."""
    uncached = [t for t in texts if t not in EMBED_CACHE]
    if uncached:
        # Batch in chunks of 100 to avoid timeouts
        for i in range(0, len(uncached), 100):
            batch = uncached[i:i+100]
            response = SESSION.post(url, json={"model": model, "input": batch}, timeout=120)
            data = response.json()
            for j, t in enumerate(batch):
                EMBED_CACHE[t] = np.array(data["data"][j]["embedding"])
    return np.array([EMBED_CACHE[t] for t in texts])


# Large test message set
TEST_MESSAGES = [
    # Technical
    "Explain how transformers work in neural networks",
    "Describe gradient descent optimization in machine learning",
    "What is the attention mechanism in deep learning",
    "How do convolutional neural networks process images",
    "Explain backpropagation in neural network training",
    # Emotional
    "Love is a powerful force that connects all humanity",
    "Fear can paralyze us or motivate us to action",
    "Joy comes from meaningful connections with others",
    "Anger often masks deeper feelings of hurt",
    "Sadness is a natural response to loss",
    # Abstract
    "Time flows like a river toward an unknown sea",
    "Truth emerges from the collision of perspectives",
    "Beauty exists in the eye of the beholder alone",
    "Justice requires balance between mercy and law",
    "Freedom carries responsibility and consequence",
    # Concrete
    "The dog ran across the green park this morning",
    "Water boils at one hundred degrees Celsius",
    "The red car drove down the mountain road",
    "Birds migrate south when winter approaches",
    "Trees lose their leaves in autumn season",
]

DISTRACTORS = [
    "The cat sat on the warm windowsill",
    "Computers process binary information rapidly",
    "Music has power to evoke emotions",
    "Mountains stand tall against the sky",
    "Time heals all wounds eventually",
    "The sun rises in the east daily",
    "Knowledge is power in modern society",
    "Actions speak louder than words",
    "The early bird catches the worm",
    "Practice makes perfect over time",
]


# =============================================================================
# Core Test Functions
# =============================================================================

def compute_alignment_quality(key_a, key_b, k):
    """Compute detailed alignment quality metrics."""
    k_use = min(key_a.k, key_b.k, k)

    X_a = key_a.eigenvectors[:, :k_use] * np.sqrt(key_a.eigenvalues[:k_use])
    X_b = key_b.eigenvectors[:, :k_use] * np.sqrt(key_b.eigenvalues[:k_use])

    # Spectrum correlation
    spec_corr, _ = spearmanr(key_a.eigenvalues[:k_use], key_b.eigenvalues[:k_use])

    # Procrustes
    R, _ = orthogonal_procrustes(X_a, X_b)
    aligned = X_a @ R
    residual = np.linalg.norm(aligned - X_b, 'fro')
    normalized_residual = residual / np.linalg.norm(X_b, 'fro')

    # Per-anchor error
    per_anchor_error = np.linalg.norm(aligned - X_b, axis=1)

    return {
        "spectrum_correlation": float(spec_corr),
        "procrustes_residual": float(residual),
        "normalized_residual": float(normalized_residual),
        "mean_anchor_error": float(np.mean(per_anchor_error)),
        "std_anchor_error": float(np.std(per_anchor_error)),
        "max_anchor_error": float(np.max(per_anchor_error)),
    }


def test_cross_model_accuracy(
    key_a, key_b, embed_a, embed_b,
    candidates, test_messages,
    corruption_levels=[0.0, 0.25, 0.50, 0.75],
    n_trials=50
):
    """Test cross-model accuracy with statistical rigor."""
    pair = key_a.align_with(key_b)
    k = pair.k

    results = {}

    for pct in corruption_levels:
        n_corrupt = int(k * pct)
        if n_corrupt >= k:
            continue

        successes = 0
        total = 0
        confidences = []

        for msg in test_messages:
            for trial in range(n_trials // len(test_messages)):
                vec = pair.encode_a_to_b(msg, embed_a)

                if n_corrupt > 0:
                    np.random.seed(trial + hash(msg) % 10000)
                    indices = np.random.choice(len(vec), n_corrupt, replace=False)
                    vec[indices] = 0.0

                match, conf = pair.decode_at_b(vec, candidates, embed_b)
                confidences.append(conf)
                total += 1

                if match == msg:
                    successes += 1

        accuracy = successes / total if total > 0 else 0
        results[f"{int(pct*100)}%"] = {
            "accuracy": accuracy,
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "n_trials": total,
        }

    return results, pair.procrustes_residual


# =============================================================================
# Main Experiment
# =============================================================================

def run_large_scale_experiment():
    """Run comprehensive cross-model alignment experiment."""
    print("=" * 70)
    print("MAXIMIZE CROSS-MODEL ALIGNMENT - LARGE SCALE EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Load models
    print("\nLoading embedding models...")
    from sentence_transformers import SentenceTransformer

    models = {}

    # Nomic via API
    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )
    models["nomic"] = embed_nomic

    # Local models
    for name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-MiniLM-L12-v2"]:
        model = SentenceTransformer(name)
        models[name] = lambda texts, m=model: m.encode(texts, convert_to_numpy=True)
        print(f"  Loaded {name}")

    # Anchor sets to test
    anchor_sets = {
        "STABLE_32": STABLE_32,
        "STABLE_64": STABLE_64,
        "ANCHOR_128": ANCHOR_128,
        "ANCHOR_256": ANCHOR_256,
        "ANCHOR_512": ANCHOR_512,
        "ANCHOR_777": generate_anchor_set(777),
    }

    # k values to test (must be < n_anchors)
    k_values = [16, 32, 48, 64, 96, 128]

    candidates = TEST_MESSAGES + DISTRACTORS

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": list(models.keys()),
        "experiments": [],
    }

    best_config = None
    best_50_accuracy = 0

    # Test each anchor set
    for anchor_name, anchors in anchor_sets.items():
        print(f"\n{'='*60}")
        print(f"ANCHOR SET: {anchor_name} ({len(anchors)} anchors)")
        print(f"{'='*60}")

        n_anchors = len(anchors)

        # Test each valid k
        for k in k_values:
            if k >= n_anchors:
                continue

            print(f"\n  k = {k}")

            # Create keys for all models
            keys = {}
            for model_name, embed_fn in models.items():
                try:
                    keys[model_name] = AlignmentKey.create(
                        model_name, embed_fn, anchors=anchors, k=k
                    )
                except Exception as e:
                    print(f"    Error creating key for {model_name}: {e}")
                    continue

            if len(keys) < 2:
                continue

            # Test all pairs
            model_names = list(keys.keys())
            for i, name_a in enumerate(model_names):
                for name_b in model_names[i+1:]:
                    key_a = keys[name_a]
                    key_b = keys[name_b]
                    embed_a = models[name_a]
                    embed_b = models[name_b]

                    # Alignment quality
                    quality = compute_alignment_quality(key_a, key_b, k)

                    # Cross-model accuracy
                    accuracy, residual = test_cross_model_accuracy(
                        key_a, key_b, embed_a, embed_b,
                        candidates, TEST_MESSAGES[:10],  # Use subset for speed
                        n_trials=50
                    )

                    exp_result = {
                        "anchor_set": anchor_name,
                        "n_anchors": n_anchors,
                        "k": k,
                        "model_a": name_a,
                        "model_b": name_b,
                        "quality": quality,
                        "accuracy": accuracy,
                    }
                    all_results["experiments"].append(exp_result)

                    # Check if best at 50% corruption
                    acc_50 = accuracy.get("50%", {}).get("accuracy", 0)
                    if acc_50 > best_50_accuracy:
                        best_50_accuracy = acc_50
                        best_config = exp_result

                    print(f"    {name_a} -> {name_b}: "
                          f"res={quality['normalized_residual']:.3f}, "
                          f"0%={accuracy['0%']['accuracy']*100:.0f}%, "
                          f"50%={acc_50*100:.0f}%")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Group by anchor set
    print("\nBest results by anchor set:")
    for anchor_name in anchor_sets.keys():
        anchor_exps = [e for e in all_results["experiments"] if e["anchor_set"] == anchor_name]
        if anchor_exps:
            best = max(anchor_exps, key=lambda x: x["accuracy"].get("50%", {}).get("accuracy", 0))
            acc_50 = best["accuracy"].get("50%", {}).get("accuracy", 0)
            print(f"  {anchor_name}: 50% corruption = {acc_50*100:.1f}% "
                  f"(k={best['k']}, {best['model_a']}->{best['model_b']})")

    # Overall best
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    if best_config:
        print(f"  Anchor set: {best_config['anchor_set']}")
        print(f"  k: {best_config['k']}")
        print(f"  Models: {best_config['model_a']} -> {best_config['model_b']}")
        print(f"  Normalized residual: {best_config['quality']['normalized_residual']:.4f}")
        print(f"  Accuracy at 0%: {best_config['accuracy']['0%']['accuracy']*100:.1f}%")
        print(f"  Accuracy at 50%: {best_config['accuracy']['50%']['accuracy']*100:.1f}%")
        print(f"  Accuracy at 75%: {best_config['accuracy'].get('75%', {}).get('accuracy', 0)*100:.1f}%")

    all_results["best_config"] = best_config
    all_results["summary"] = {
        "best_50_accuracy": best_50_accuracy,
        "total_experiments": len(all_results["experiments"]),
    }

    return all_results


if __name__ == "__main__":
    results = run_large_scale_experiment()

    # Save results
    output_path = Path(__file__).parent / "maximize_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
