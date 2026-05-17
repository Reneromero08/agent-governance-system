#!/usr/bin/env python3
"""Middle-Mean LLM Alignment Test.

The Problem:
- Ollama's /api/embed returns LAST TOKEN of FINAL LAYER
- This is the "mouth" (prediction head), not the "brain" (understanding)
- "The dog runs fast" -> embedding of "fast", not the sentence

The Fix:
1. Extract from MIDDLE LAYER (layer 14 of 28) - where semantics live
2. MEAN POOL across all tokens - not just last token
3. Use PHRASE ANCHORS - not just single words

This requires transformers library to access hidden states.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey

# Check if transformers is available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers library not installed")
    print("Run: pip install transformers accelerate")
    sys.exit(1)

# =============================================================================
# PHRASE ANCHORS (50% words, 50% phrases)
# =============================================================================

WORD_ANCHORS = [
    "dog", "cat", "water", "fire", "love", "hate", "run", "walk",
    "think", "speak", "fast", "slow", "big", "small", "hot", "cold",
    "happy", "sad", "light", "dark", "up", "down", "in", "out",
    "good", "bad", "new", "old", "true", "false", "yes", "no",
]

PHRASE_ANCHORS = [
    "the quick brown fox",
    "hello world",
    "I think therefore I am",
    "to be or not to be",
    "all that glitters is not gold",
    "the sky is blue",
    "water flows downhill",
    "birds fly south",
    "the sun rises",
    "time flies fast",
    "knowledge is power",
    "actions speak louder",
    "practice makes perfect",
    "seeing is believing",
    "silence is golden",
    "money talks loudly",
]

MIXED_ANCHORS = WORD_ANCHORS + PHRASE_ANCHORS


class MiddleMeanEmbedder:
    """Extract middle-layer, mean-pooled embeddings from LLM."""

    def __init__(self, model_name: str, layer_fraction: float = 0.5, device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B")
            layer_fraction: Which layer to extract (0.5 = middle)
            device: "cuda" or "cpu"
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.model.eval()

        # Calculate target layer
        num_layers = self.model.config.num_hidden_layers
        self.target_layer = int(num_layers * layer_fraction)
        print(f"Model has {num_layers} layers, extracting from layer {self.target_layer}")

        self.device = device
        self.embed_dim = self.model.config.hidden_size

    def embed(self, texts):
        """Get middle-layer, mean-pooled embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Extract middle layer hidden states
            hidden_states = outputs.hidden_states[self.target_layer]  # [batch, seq, hidden]

            # Mean pool across sequence (excluding padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch, seq, 1]
            masked_hidden = hidden_states * attention_mask
            mean_pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)  # [batch, hidden]

            embeddings.append(mean_pooled[0].cpu().numpy())

        return np.array(embeddings)


def test_middle_mean():
    """Test if middle-mean fixes the alignment problem."""
    print("=" * 60)
    print("MIDDLE-MEAN LLM ALIGNMENT TEST")
    print("Extracting middle layer with mean pooling")
    print("=" * 60)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if device == "cpu":
        print("WARNING: CPU mode will be slow. Consider using smaller models.")

    # Load models
    # Using smaller models for testing - replace with full 7B if you have GPU memory
    try:
        print("\n[1] Loading Qwen model...")
        qwen = MiddleMeanEmbedder("Qwen/Qwen2.5-1.5B", layer_fraction=0.5, device=device)

        print("\n[2] Loading alternative model...")
        # Try a different small model for comparison
        other = MiddleMeanEmbedder("microsoft/phi-2", layer_fraction=0.5, device=device)
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("\nTrying Ollama fallback with phrase anchors only...")
        test_phrase_anchors_only()
        return

    # Test basic similarity
    print("\n[3] Testing similarity behavior...")
    v_dog = qwen.embed("dog")[0]
    v_the_dog = qwen.embed("the dog")[0]
    v_dog_runs = qwen.embed("The dog runs fast")[0]

    sim1 = np.dot(v_dog, v_the_dog) / (np.linalg.norm(v_dog) * np.linalg.norm(v_the_dog))
    sim2 = np.dot(v_dog, v_dog_runs) / (np.linalg.norm(v_dog) * np.linalg.norm(v_dog_runs))

    print(f'    "dog" vs "the dog": {sim1:.4f}')
    print(f'    "dog" vs "The dog runs fast": {sim2:.4f}')

    if sim1 > 0.7:
        print("    SUCCESS: Mean pooling preserves word meaning in phrases!")
    else:
        print("    Still low - may need different layer")

    # Create alignment keys with phrase anchors
    print("\n[4] Creating alignment keys with mixed anchors...")

    def embed_qwen(texts):
        return qwen.embed(texts)

    def embed_other(texts):
        return other.embed(texts)

    key_qwen = AlignmentKey.create("qwen-middle", embed_qwen, anchors=MIXED_ANCHORS, k=32)
    key_other = AlignmentKey.create("other-middle", embed_other, anchors=MIXED_ANCHORS, k=32)

    pair = key_qwen.align_with(key_other)
    print(f"    Procrustes residual: {pair.procrustes_residual:.4f}")
    print(f"    Spectrum correlation: {pair.spectrum_correlation:.4f}")

    # Test sentence communication
    print("\n[5] Testing sentence-level communication...")

    test_sentences = [
        "The quick brown fox jumps",
        "Neural networks learn patterns",
        "Water flows downhill naturally",
        "Music brings people together",
    ]

    distractors = [
        ["The slow blue cat crawls", "A tree grows in soil", "Cars need fuel"],
        ["Cats sleep all day", "Fire burns brightly", "Books contain words"],
        ["Ice cream melts fast", "Birds fly through sky", "Mountains stand tall"],
        ["Silence is golden", "Stars shine at night", "Coffee wakes people"],
    ]

    correct = 0
    for i, msg in enumerate(test_sentences):
        candidates = [msg] + distractors[i]
        np.random.shuffle(candidates)

        vec = pair.encode_a_to_b(msg, embed_qwen)
        match, score = pair.decode_at_b(vec, candidates, embed_other)

        ok = match == msg
        if ok:
            correct += 1
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] '{msg[:30]}' -> '{match[:30]}' (conf={score:.3f})")

    print(f"\n    ACCURACY: {correct}/{len(test_sentences)} ({100*correct/len(test_sentences):.0f}%)")

    if correct >= 3:
        print("\n*** MIDDLE-MEAN STRATEGY WORKS! ***")
    else:
        print("\nNeed to tune layer selection or try different approach.")


def test_phrase_anchors_only():
    """Fallback: Test with phrase anchors using Ollama (no middle-layer access)."""
    import requests

    print("\n" + "=" * 60)
    print("PHRASE ANCHORS TEST (Ollama fallback)")
    print("Testing if phrase anchors improve alignment")
    print("=" * 60)

    def embed_qwen(texts):
        results = []
        for t in (texts if isinstance(texts, list) else [texts]):
            resp = requests.post("http://localhost:11434/api/embed",
                               json={"model": "qwen2.5:7b", "input": t})
            results.append(np.array(resp.json()["embeddings"][0]))
        return np.array(results)

    def embed_mistral(texts):
        results = []
        for t in (texts if isinstance(texts, list) else [texts]):
            resp = requests.post("http://localhost:11434/api/embed",
                               json={"model": "mistral:7b", "input": t})
            results.append(np.array(resp.json()["embeddings"][0]))
        return np.array(results)

    print("\n[1] Testing with WORD-ONLY anchors...")
    key_q_words = AlignmentKey.create("qwen-words", embed_qwen, anchors=WORD_ANCHORS, k=24)
    key_m_words = AlignmentKey.create("mistral-words", embed_mistral, anchors=WORD_ANCHORS, k=24)
    pair_words = key_q_words.align_with(key_m_words)
    print(f"    Residual: {pair_words.procrustes_residual:.4f}")

    print("\n[2] Testing with MIXED anchors (words + phrases)...")
    key_q_mixed = AlignmentKey.create("qwen-mixed", embed_qwen, anchors=MIXED_ANCHORS, k=32)
    key_m_mixed = AlignmentKey.create("mistral-mixed", embed_mistral, anchors=MIXED_ANCHORS, k=32)
    pair_mixed = key_q_mixed.align_with(key_m_mixed)
    print(f"    Residual: {pair_mixed.procrustes_residual:.4f}")

    # Compare sentence accuracy
    test_sentences = [
        "The quick brown fox jumps",
        "Neural networks learn patterns",
        "Water flows downhill naturally",
        "Music brings people together",
    ]

    distractors = [
        ["The slow blue cat crawls", "A tree grows in soil", "Cars need fuel"],
        ["Cats sleep all day", "Fire burns brightly", "Books contain words"],
        ["Ice cream melts fast", "Birds fly through sky", "Mountains stand tall"],
        ["Silence is golden", "Stars shine at night", "Coffee wakes people"],
    ]

    print("\n[3] Sentence accuracy with WORD anchors:")
    correct_words = 0
    for i, msg in enumerate(test_sentences):
        candidates = [msg] + distractors[i]
        vec = pair_words.encode_a_to_b(msg, embed_qwen)
        match, score = pair_words.decode_at_b(vec, candidates, embed_mistral)
        if match == msg:
            correct_words += 1
        print(f"    {msg[:30]:30s} -> {match[:30]:30s} ({'OK' if match == msg else 'FAIL'})")
    print(f"    Accuracy: {correct_words}/{len(test_sentences)}")

    print("\n[4] Sentence accuracy with MIXED anchors:")
    correct_mixed = 0
    for i, msg in enumerate(test_sentences):
        candidates = [msg] + distractors[i]
        vec = pair_mixed.encode_a_to_b(msg, embed_qwen)
        match, score = pair_mixed.decode_at_b(vec, candidates, embed_mistral)
        if match == msg:
            correct_mixed += 1
        print(f"    {msg[:30]:30s} -> {match[:30]:30s} ({'OK' if match == msg else 'FAIL'})")
    print(f"    Accuracy: {correct_mixed}/{len(test_sentences)}")

    if correct_mixed > correct_words:
        print(f"\n*** PHRASE ANCHORS HELP: {correct_words} -> {correct_mixed} ***")
    else:
        print("\nPhrase anchors didn't help - need middle-layer access.")


if __name__ == "__main__":
    if HAS_TRANSFORMERS:
        # Try full middle-mean approach first
        try:
            test_middle_mean()
        except Exception as e:
            print(f"\nMiddle-mean failed: {e}")
            print("Falling back to phrase anchors only...")
            test_phrase_anchors_only()
    else:
        test_phrase_anchors_only()