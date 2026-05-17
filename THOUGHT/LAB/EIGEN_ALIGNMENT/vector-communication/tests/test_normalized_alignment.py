#!/usr/bin/env python3
"""Test: Does L2-normalized middle-mean improve cross-model alignment?

Finding from eigenvalue analysis:
- Raw hidden states have 99.6% variance in ONE direction (residual stream)
- L2 normalization fixes this: alpha -> 0.65-0.80 (closer to Q8's 0.5)

Test: Compare alignment accuracy with normalized vs unnormalized embeddings.
"""

import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey

ANCHORS = [
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    "bird", "fish", "rock", "flower", "sun", "moon", "star", "cloud",
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    "run", "walk", "think", "speak", "eat", "sleep", "fly", "swim",
    "big", "small", "hot", "cold", "fast", "slow", "good", "bad",
]


class LayerEmbedder:
    """Configurable layer/pooling/normalization embedder."""

    def __init__(self, model_name, layer_fraction=0.5, pooling="mean", normalize=True, device="cuda"):
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
        self.target_layer = int(num_layers * layer_fraction) if layer_fraction < 1.0 else -1
        self.pooling = pooling
        self.normalize = normalize
        self.device = device
        self.name = f"L{self.target_layer}_{pooling}{'_norm' if normalize else ''}"

        print(f"  Config: layer={self.target_layer}, pooling={pooling}, normalize={normalize}")

    def embed(self, texts):
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

            if self.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1)
                vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                vec = hidden[:, -1, :]

            vec = vec[0].cpu().float().numpy()

            if self.normalize:
                vec = vec / (np.linalg.norm(vec) + 1e-8)

            embeddings.append(vec)

        return np.array(embeddings)


def test_config(model_a, model_b, name):
    """Test alignment accuracy for a given configuration."""
    print(f"\n--- {name} ---")

    key_a = AlignmentKey.create("model_a", model_a.embed, anchors=ANCHORS, k=32)
    key_b = AlignmentKey.create("model_b", model_b.embed, anchors=ANCHORS, k=32)
    pair = key_a.align_with(key_b)

    print(f"Procrustes residual: {pair.procrustes_residual:.4f}")
    print(f"Spectrum correlation: {pair.spectrum_correlation:.4f}")

    # Test sentences
    test_sentences = [
        "The dog runs fast",
        "Birds fly through clouds",
        "Water flows downhill",
        "Love conquers hate",
        "Time passes quickly",
        "The sun rises early",
    ]

    distractors = [
        ["The cat sleeps quietly", "Fish swim deep", "Trees grow tall"],
        ["Dogs bark loudly", "Rocks sit still", "Fire burns bright"],
        ["Ice melts slowly", "Stars shine above", "Wind blows strong"],
        ["Fear grips tightly", "Joy fills hearts", "Peace calms minds"],
        ["Space extends far", "Truth reveals all", "Ideas spread wide"],
        ["The moon sets late", "Stars fade slowly", "Clouds drift away"],
    ]

    correct = 0
    for i, msg in enumerate(test_sentences):
        candidates = [msg] + distractors[i]
        np.random.shuffle(candidates)

        vec = pair.encode_a_to_b(msg, model_a.embed)
        match, score = pair.decode_at_b(vec, candidates, model_b.embed)

        ok = match == msg
        if ok:
            correct += 1

    accuracy = correct / len(test_sentences)
    print(f"Sentence accuracy: {correct}/{len(test_sentences)} ({accuracy*100:.0f}%)")

    return {
        "residual": pair.procrustes_residual,
        "spectrum": pair.spectrum_correlation,
        "accuracy": accuracy,
    }


def main():
    print("=" * 60)
    print("NORMALIZED ALIGNMENT TEST")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load two different models
    print("\nLoading Qwen/Qwen2.5-0.5B...")
    print("\nLoading Qwen/Qwen2.5-1.5B...")

    results = {}

    # Test 1: Final layer, last token, normalized (baseline - like Ollama)
    print("\n" + "=" * 60)
    print("CONFIG 1: Final layer, last token, normalized")
    a1 = LayerEmbedder("Qwen/Qwen2.5-0.5B", layer_fraction=1.0, pooling="last", normalize=True, device=device)
    b1 = LayerEmbedder("Qwen/Qwen2.5-1.5B", layer_fraction=1.0, pooling="last", normalize=True, device=device)
    results["final_last_norm"] = test_config(a1, b1, "Final/Last/Norm")
    del a1, b1
    torch.cuda.empty_cache()

    # Test 2: Middle layer, mean pooled, normalized
    print("\n" + "=" * 60)
    print("CONFIG 2: Middle layer, mean pooled, normalized")
    a2 = LayerEmbedder("Qwen/Qwen2.5-0.5B", layer_fraction=0.5, pooling="mean", normalize=True, device=device)
    b2 = LayerEmbedder("Qwen/Qwen2.5-1.5B", layer_fraction=0.5, pooling="mean", normalize=True, device=device)
    results["middle_mean_norm"] = test_config(a2, b2, "Middle/Mean/Norm")
    del a2, b2
    torch.cuda.empty_cache()

    # Test 3: Middle layer, last token, normalized
    print("\n" + "=" * 60)
    print("CONFIG 3: Middle layer, last token, normalized")
    a3 = LayerEmbedder("Qwen/Qwen2.5-0.5B", layer_fraction=0.5, pooling="last", normalize=True, device=device)
    b3 = LayerEmbedder("Qwen/Qwen2.5-1.5B", layer_fraction=0.5, pooling="last", normalize=True, device=device)
    results["middle_last_norm"] = test_config(a3, b3, "Middle/Last/Norm")
    del a3, b3
    torch.cuda.empty_cache()

    # Test 4: Final layer, mean pooled, normalized
    print("\n" + "=" * 60)
    print("CONFIG 4: Final layer, mean pooled, normalized")
    a4 = LayerEmbedder("Qwen/Qwen2.5-0.5B", layer_fraction=1.0, pooling="mean", normalize=True, device=device)
    b4 = LayerEmbedder("Qwen/Qwen2.5-1.5B", layer_fraction=1.0, pooling="mean", normalize=True, device=device)
    results["final_mean_norm"] = test_config(a4, b4, "Final/Mean/Norm")
    del a4, b4
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<20} {'Residual':>10} {'Spectrum':>10} {'Accuracy':>10}")
    print("-" * 52)

    best_acc = 0
    best_config = None
    for name, data in results.items():
        print(f"{name:<20} {data['residual']:>10.4f} {data['spectrum']:>10.4f} {data['accuracy']*100:>9.0f}%")
        if data["accuracy"] > best_acc:
            best_acc = data["accuracy"]
            best_config = name

    print(f"\nBest config: {best_config} ({best_acc*100:.0f}% accuracy)")


if __name__ == "__main__":
    main()
