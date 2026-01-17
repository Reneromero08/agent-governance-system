#!/usr/bin/env python3
"""Verify: Does Middle-Mean restore the correct semantic topology?

Q8 proved: Semantic manifold has c_1 = 1, alpha = 0.5
Hypothesis: Ollama embeddings (final layer) have alpha != 0.5
            Middle-Mean embeddings have alpha ~ 0.5

If true, this proves Middle-Mean extracts from the semantic manifold.
"""

import numpy as np
import torch
import requests
import sys
from pathlib import Path
from scipy.optimize import curve_fit

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

# Check for transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# =============================================================================
# TEST ANCHORS (enough for eigenvalue spectrum)
# =============================================================================

TEST_ANCHORS = [
    # Concrete nouns
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    "bird", "fish", "rock", "flower", "sun", "moon", "star", "cloud",
    # Abstract concepts
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    "hope", "pain", "peace", "war", "life", "death", "mind", "soul",
    # Actions
    "run", "walk", "think", "speak", "eat", "sleep", "fly", "swim",
    "jump", "fall", "grow", "die", "love", "hate", "know", "feel",
    # Properties
    "big", "small", "hot", "cold", "fast", "slow", "good", "bad",
    "new", "old", "true", "false", "light", "dark", "soft", "hard",
]


def power_law(k, alpha, C):
    """Power law: lambda_k = C * k^(-alpha)"""
    return C * np.power(k, -alpha)


def compute_alpha(embeddings):
    """Compute eigenvalue decay exponent alpha from embeddings.

    For semantic manifold with c_1 = 1: alpha = 1/(2*c_1) = 0.5
    """
    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Use top eigenvalues for fitting (avoid noise floor)
    n_fit = min(30, len(eigenvalues) // 2)
    eigenvalues = eigenvalues[:n_fit]
    eigenvalues = eigenvalues[eigenvalues > 0]  # Positive only

    if len(eigenvalues) < 5:
        return None, None

    # Fit power law: lambda_k ~ k^(-alpha)
    k = np.arange(1, len(eigenvalues) + 1)

    try:
        popt, _ = curve_fit(power_law, k, eigenvalues, p0=[0.5, eigenvalues[0]], maxfev=5000)
        alpha = popt[0]
        c1 = 1 / (2 * alpha) if alpha > 0 else None
        return alpha, c1
    except:
        return None, None


def embed_ollama(texts, model="qwen2.5:7b"):
    """Get embeddings from Ollama (final layer, last token)."""
    results = []
    for t in texts:
        try:
            resp = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": model, "input": t},
                timeout=30
            )
            results.append(np.array(resp.json()["embeddings"][0]))
        except Exception as e:
            print(f"  Ollama error for '{t}': {e}")
            return None
    return np.array(results)


class MiddleMeanEmbedder:
    """Extract middle-layer, mean-pooled embeddings."""

    def __init__(self, model_name, layer_fraction=0.5, device="cuda"):
        print(f"  Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.model.eval()

        num_layers = self.model.config.num_hidden_layers
        self.target_layer = int(num_layers * layer_fraction)
        print(f"  {num_layers} layers, extracting from layer {self.target_layer}")

        self.device = device

    def embed(self, texts):
        """Get middle-layer, mean-pooled embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[self.target_layer]
            mask = inputs["attention_mask"].unsqueeze(-1)
            masked = hidden * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1)

            embeddings.append(pooled[0].cpu().float().numpy())

        return np.array(embeddings)


def test_topology():
    """Compare alpha for Ollama vs Middle-Mean embeddings."""
    print("=" * 60)
    print("TOPOLOGY VERIFICATION TEST")
    print("Q8 predicts: Semantic manifold has alpha = 0.5 (c_1 = 1)")
    print("=" * 60)

    results = {}

    # Test 1: Ollama embeddings
    print("\n[1] Testing Ollama embeddings (final layer, last token)...")
    ollama_emb = embed_ollama(TEST_ANCHORS)
    if ollama_emb is not None:
        alpha_ollama, c1_ollama = compute_alpha(ollama_emb)
        if alpha_ollama:
            print(f"    Ollama: alpha = {alpha_ollama:.4f}, c_1 = {c1_ollama:.4f}")
            print(f"    Deviation from 0.5: {abs(alpha_ollama - 0.5):.4f}")
            results["ollama"] = {"alpha": alpha_ollama, "c1": c1_ollama}
        else:
            print("    Failed to compute alpha")
    else:
        print("    Ollama not available")

    # Test 2: Middle-Mean embeddings
    if HAS_TRANSFORMERS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[2] Testing Middle-Mean embeddings (device: {device})...")

        try:
            embedder = MiddleMeanEmbedder("Qwen/Qwen2.5-0.5B", layer_fraction=0.5, device=device)
            middle_emb = embedder.embed(TEST_ANCHORS)

            alpha_middle, c1_middle = compute_alpha(middle_emb)
            if alpha_middle:
                print(f"    Middle-Mean: alpha = {alpha_middle:.4f}, c_1 = {c1_middle:.4f}")
                print(f"    Deviation from 0.5: {abs(alpha_middle - 0.5):.4f}")
                results["middle_mean"] = {"alpha": alpha_middle, "c1": c1_middle}
            else:
                print("    Failed to compute alpha")

            # Also test last-token for comparison
            print("\n[3] Testing Last-Token (same model, final layer)...")

            def embed_last_token(texts):
                embeddings = []
                for text in texts:
                    inputs = embedder.tokenizer(text, return_tensors="pt", padding=True)
                    if device == "cuda":
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = embedder.model(**inputs, output_hidden_states=True)

                    # Final layer, last token
                    hidden = outputs.hidden_states[-1]
                    last_token = hidden[:, -1, :]
                    embeddings.append(last_token[0].cpu().float().numpy())

                return np.array(embeddings)

            last_emb = embed_last_token(TEST_ANCHORS)
            alpha_last, c1_last = compute_alpha(last_emb)
            if alpha_last:
                print(f"    Last-Token: alpha = {alpha_last:.4f}, c_1 = {c1_last:.4f}")
                print(f"    Deviation from 0.5: {abs(alpha_last - 0.5):.4f}")
                results["last_token"] = {"alpha": alpha_last, "c1": c1_last}

        except Exception as e:
            print(f"    Error: {e}")
    else:
        print("\n[2] Skipping Middle-Mean (transformers not installed)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nQ8 Reference: alpha = 0.5, c_1 = 1.0 (semantic manifold)")
    print()

    for name, data in results.items():
        deviation = abs(data["alpha"] - 0.5)
        on_manifold = "YES" if deviation < 0.15 else "NO"
        print(f"{name:15s}: alpha={data['alpha']:.4f}, c_1={data['c1']:.4f}, on_manifold={on_manifold}")

    print()
    if "middle_mean" in results and "last_token" in results:
        mm_dev = abs(results["middle_mean"]["alpha"] - 0.5)
        lt_dev = abs(results["last_token"]["alpha"] - 0.5)

        if mm_dev < lt_dev:
            print("RESULT: Middle-Mean is CLOSER to semantic manifold (alpha=0.5)")
            print(f"        Middle-Mean deviation: {mm_dev:.4f}")
            print(f"        Last-Token deviation:  {lt_dev:.4f}")
        else:
            print("RESULT: Last-Token is closer (unexpected)")
            print("        Need to investigate layer selection")


if __name__ == "__main__":
    test_topology()
