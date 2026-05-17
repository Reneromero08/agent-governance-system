#!/usr/bin/env python3
"""Investigate eigenvalue spectrum at different layers.

The middle-mean showed alpha=10.7 which is extremely high (fast decay).
Let's look at the raw eigenvalue distributions.
"""

import numpy as np
import torch
import sys
from pathlib import Path
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_ANCHORS = [
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    "bird", "fish", "rock", "flower", "sun", "moon", "star", "cloud",
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    "hope", "pain", "peace", "war", "life", "death", "mind", "soul",
    "run", "walk", "think", "speak", "eat", "sleep", "fly", "swim",
    "jump", "fall", "grow", "die", "know", "feel", "see", "hear",
    "big", "small", "hot", "cold", "fast", "slow", "good", "bad",
    "new", "old", "true", "false", "light", "dark", "soft", "hard",
]


def power_law(k, alpha, C):
    return C * np.power(k, -alpha)


def analyze_spectrum(embeddings, name):
    """Analyze eigenvalue spectrum."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero

    print(f"\n{name}:")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Eigenvalues > 1e-10: {len(eigenvalues)}")
    print(f"  Top 5: {eigenvalues[:5]}")
    print(f"  Ratio top1/top2: {eigenvalues[0]/eigenvalues[1]:.2f}")
    print(f"  Ratio top1/top5: {eigenvalues[0]/eigenvalues[4]:.2f}")

    # Variance explained by top-k
    total = eigenvalues.sum()
    for k in [1, 5, 10, 20]:
        explained = eigenvalues[:k].sum() / total * 100
        print(f"  Variance in top-{k}: {explained:.1f}%")

    # Fit power law on top eigenvalues
    n_fit = min(20, len(eigenvalues) // 2)
    ev_fit = eigenvalues[:n_fit]
    k_fit = np.arange(1, n_fit + 1)

    try:
        popt, _ = curve_fit(power_law, k_fit, ev_fit, p0=[0.5, ev_fit[0]], maxfev=5000)
        alpha = popt[0]
        print(f"  Fitted alpha: {alpha:.4f}")
    except:
        print("  Fitting failed")

    return eigenvalues


def main():
    print("=" * 60)
    print("EIGENVALUE SPECTRUM ANALYSIS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Extract embeddings at different layers
    def extract_embeddings(layer_idx, pooling="mean"):
        embeddings = []
        for text in TEST_ANCHORS:
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[layer_idx]

            if pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1)
                vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            else:  # last
                vec = hidden[:, -1, :]

            embeddings.append(vec[0].cpu().float().numpy())

        return np.array(embeddings)

    # Test different layers and pooling strategies
    print("\n" + "=" * 60)
    print("LAYER 0 (Token Embeddings)")
    analyze_spectrum(extract_embeddings(0, "mean"), "Layer 0, mean-pool")

    print("\n" + "=" * 60)
    print(f"LAYER {num_layers//4} (Early)")
    analyze_spectrum(extract_embeddings(num_layers//4, "mean"), f"Layer {num_layers//4}, mean-pool")
    analyze_spectrum(extract_embeddings(num_layers//4, "last"), f"Layer {num_layers//4}, last-token")

    print("\n" + "=" * 60)
    print(f"LAYER {num_layers//2} (Middle)")
    analyze_spectrum(extract_embeddings(num_layers//2, "mean"), f"Layer {num_layers//2}, mean-pool")
    analyze_spectrum(extract_embeddings(num_layers//2, "last"), f"Layer {num_layers//2}, last-token")

    print("\n" + "=" * 60)
    print(f"LAYER {3*num_layers//4} (Late)")
    analyze_spectrum(extract_embeddings(3*num_layers//4, "mean"), f"Layer {3*num_layers//4}, mean-pool")
    analyze_spectrum(extract_embeddings(3*num_layers//4, "last"), f"Layer {3*num_layers//4}, last-token")

    print("\n" + "=" * 60)
    print(f"LAYER {num_layers} (Final)")
    analyze_spectrum(extract_embeddings(-1, "mean"), f"Layer {num_layers}, mean-pool")
    analyze_spectrum(extract_embeddings(-1, "last"), f"Layer {num_layers}, last-token")

    # Compare L2-normalized versions
    print("\n" + "=" * 60)
    print("WITH L2 NORMALIZATION")

    def extract_normalized(layer_idx, pooling):
        emb = extract_embeddings(layer_idx, pooling)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-8)

    analyze_spectrum(extract_normalized(num_layers//2, "mean"), f"Layer {num_layers//2}, mean-pool, L2-norm")
    analyze_spectrum(extract_normalized(-1, "last"), f"Layer {num_layers}, last-token, L2-norm")


if __name__ == "__main__":
    main()
