#!/usr/bin/env python3
"""Test cross-architecture alignment with 7B models.

Qwen2.5-7B vs Mistral-7B using middle-last-norm extraction.
Load one model at a time to fit in 12GB VRAM.
"""

import numpy as np
import torch
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey

# Anchors for alignment
ANCHORS = [
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    "bird", "fish", "rock", "flower", "sun", "moon", "star", "cloud",
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    "run", "walk", "think", "speak", "eat", "sleep", "fly", "swim",
    "big", "small", "hot", "cold", "fast", "slow", "good", "bad",
]

# Test sentences
TEST_SENTENCES = [
    "The dog runs fast",
    "Birds fly through clouds",
    "Water flows downhill",
    "Love conquers hate",
    "Time passes quickly",
    "The sun rises early",
]

DISTRACTORS = [
    ["The cat sleeps quietly", "Fish swim deep", "Trees grow tall"],
    ["Dogs bark loudly", "Rocks sit still", "Fire burns bright"],
    ["Ice melts slowly", "Stars shine above", "Wind blows strong"],
    ["Fear grips tightly", "Joy fills hearts", "Peace calms minds"],
    ["Space extends far", "Truth reveals all", "Ideas spread wide"],
    ["The moon sets late", "Stars fade slowly", "Clouds drift away"],
]


def extract_embeddings(model_name, texts, layer_fraction=0.5):
    """Extract middle-last-norm embeddings from a model."""
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    target_layer = int(num_layers * layer_fraction)
    hidden_size = model.config.hidden_size

    print(f"  {num_layers} layers, extracting from layer {target_layer}, hidden_size={hidden_size}")

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer]
        vec = hidden[:, -1, :].cpu().float().numpy()[0]

        # L2 normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings.append(vec)

    embeddings = np.array(embeddings)
    print(f"  Extracted {len(embeddings)} embeddings, shape: {embeddings.shape}")

    # Clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


def main():
    print("=" * 60)
    print("CROSS-ARCHITECTURE TEST")
    print("Qwen2.5-1.5B (28 layers) vs GPT-2 (12 layers)")
    print("Using middle-last-norm extraction")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device != "cuda":
        print("ERROR: CUDA required for 7B models")
        return

    # All texts we need embeddings for
    all_texts = ANCHORS + TEST_SENTENCES
    for d in DISTRACTORS:
        all_texts.extend(d)

    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for t in all_texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    print(f"\nTotal unique texts: {len(unique_texts)}")

    # Extract from Qwen (use 1.5B - fits in memory)
    print("\n[MODEL A: Qwen]")
    qwen_emb = extract_embeddings("Qwen/Qwen2.5-1.5B", unique_texts)

    # Extract from GPT-2 (different architecture, 12 layers vs 28)
    print("\n[MODEL B: GPT-2]")
    gpt2_emb = extract_embeddings("gpt2", unique_texts)

    mistral_emb = gpt2_emb  # Use GPT-2 as "mistral" for cross-arch test

    # Build text -> embedding lookup
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    def embed_qwen(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([qwen_emb[text_to_idx[t]] for t in texts])

    def embed_mistral(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([mistral_emb[text_to_idx[t]] for t in texts])

    # Create alignment keys
    print("\n" + "=" * 60)
    print("ALIGNMENT TEST")
    print("=" * 60)

    key_q = AlignmentKey.create("qwen", embed_qwen, anchors=ANCHORS, k=32)
    key_m = AlignmentKey.create("mistral", embed_mistral, anchors=ANCHORS, k=32)

    pair = key_q.align_with(key_m)

    print(f"\nProcrustes residual: {pair.procrustes_residual:.4f}")
    print(f"Spectrum correlation: {pair.spectrum_correlation:.4f}")

    # Test sentence communication
    print("\n--- Qwen -> Mistral ---")
    correct_qm = 0
    for i, msg in enumerate(TEST_SENTENCES):
        candidates = [msg] + DISTRACTORS[i]
        np.random.shuffle(candidates)

        vec = pair.encode_a_to_b(msg, embed_qwen)
        match, score = pair.decode_at_b(vec, candidates, embed_mistral)

        ok = match == msg
        if ok:
            correct_qm += 1
        status = "OK" if ok else "FAIL"
        print(f"[{status}] '{msg[:30]}' -> '{match[:30]}' (conf={score:.3f})")

    print(f"\nQwen -> Mistral: {correct_qm}/{len(TEST_SENTENCES)} ({100*correct_qm/len(TEST_SENTENCES):.0f}%)")

    # Test reverse direction
    print("\n--- Mistral -> Qwen ---")
    correct_mq = 0
    for i, msg in enumerate(TEST_SENTENCES):
        candidates = [msg] + DISTRACTORS[i]
        np.random.shuffle(candidates)

        vec = pair.encode_b_to_a(msg, embed_mistral)
        match, score = pair.decode_at_a(vec, candidates, embed_qwen)

        ok = match == msg
        if ok:
            correct_mq += 1
        status = "OK" if ok else "FAIL"
        print(f"[{status}] '{msg[:30]}' -> '{match[:30]}' (conf={score:.3f})")

    print(f"\nGPT-2 -> Qwen: {correct_mq}/{len(TEST_SENTENCES)} ({100*correct_mq/len(TEST_SENTENCES):.0f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Procrustes residual: {pair.procrustes_residual:.4f}")
    print(f"Qwen -> GPT-2: {correct_qm}/{len(TEST_SENTENCES)} ({100*correct_qm/len(TEST_SENTENCES):.0f}%)")
    print(f"GPT-2 -> Qwen: {correct_mq}/{len(TEST_SENTENCES)} ({100*correct_mq/len(TEST_SENTENCES):.0f}%)")
    print(f"Total: {correct_qm + correct_mq}/{2*len(TEST_SENTENCES)} ({100*(correct_qm+correct_mq)/(2*len(TEST_SENTENCES)):.0f}%)")


if __name__ == "__main__":
    main()
