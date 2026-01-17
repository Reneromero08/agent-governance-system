#!/usr/bin/env python3
"""Test if higher k helps cross-architecture alignment.

Previous result: k=32 gave asymmetric results (33% vs 83%)
Hypothesis: Higher k preserves more structure, improves alignment
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

# Same setup as before
ANCHORS = [
    "dog", "cat", "tree", "house", "car", "book", "water", "food",
    "bird", "fish", "rock", "flower", "sun", "moon", "star", "cloud",
    "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
    "run", "walk", "think", "speak", "eat", "sleep", "fly", "swim",
    "big", "small", "hot", "cold", "fast", "slow", "good", "bad",
]

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
    """Extract middle-last-norm embeddings."""
    print(f"  Loading {model_name}...")

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

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer]
        vec = hidden[:, -1, :].cpu().float().numpy()[0]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings.append(vec)

    embeddings = np.array(embeddings)
    print(f"  Shape: {embeddings.shape}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


def test_with_k(qwen_emb, gpt2_emb, unique_texts, k):
    """Test alignment with specific k."""
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    def embed_qwen(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([qwen_emb[text_to_idx[t]] for t in texts])

    def embed_gpt2(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([gpt2_emb[text_to_idx[t]] for t in texts])

    key_q = AlignmentKey.create("qwen", embed_qwen, anchors=ANCHORS, k=k)
    key_g = AlignmentKey.create("gpt2", embed_gpt2, anchors=ANCHORS, k=k)
    pair = key_q.align_with(key_g)

    # Qwen -> GPT-2
    correct_qg = 0
    for i, msg in enumerate(TEST_SENTENCES):
        candidates = [msg] + DISTRACTORS[i]
        vec = pair.encode_a_to_b(msg, embed_qwen)
        match, _ = pair.decode_at_b(vec, candidates, embed_gpt2)
        if match == msg:
            correct_qg += 1

    # GPT-2 -> Qwen
    correct_gq = 0
    for i, msg in enumerate(TEST_SENTENCES):
        candidates = [msg] + DISTRACTORS[i]
        vec = pair.encode_b_to_a(msg, embed_gpt2)
        match, _ = pair.decode_at_a(vec, candidates, embed_qwen)
        if match == msg:
            correct_gq += 1

    return {
        "k": k,
        "residual": pair.procrustes_residual,
        "qwen_to_gpt2": correct_qg / len(TEST_SENTENCES),
        "gpt2_to_qwen": correct_gq / len(TEST_SENTENCES),
        "total": (correct_qg + correct_gq) / (2 * len(TEST_SENTENCES)),
    }


def main():
    print("=" * 60)
    print("K-SWEEP FOR CROSS-ARCHITECTURE ALIGNMENT")
    print("Qwen2.5-1.5B vs GPT-2")
    print("=" * 60)

    # Gather all texts
    all_texts = ANCHORS + TEST_SENTENCES
    for d in DISTRACTORS:
        all_texts.extend(d)
    seen = set()
    unique_texts = [t for t in all_texts if not (t in seen or seen.add(t))]

    print(f"\nExtracting embeddings for {len(unique_texts)} texts...")
    qwen_emb = extract_embeddings("Qwen/Qwen2.5-1.5B", unique_texts)
    gpt2_emb = extract_embeddings("gpt2", unique_texts)

    # Test different k values
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    k_values = [8, 16, 24, 32]  # Limited by anchor count (40)
    results = []

    for k in k_values:
        print(f"\nTesting k={k}...")
        r = test_with_k(qwen_emb, gpt2_emb, unique_texts, k)
        results.append(r)
        print(f"  Residual: {r['residual']:.4f}")
        print(f"  Qwen->GPT2: {r['qwen_to_gpt2']*100:.0f}%")
        print(f"  GPT2->Qwen: {r['gpt2_to_qwen']*100:.0f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'k':>4} {'Residual':>10} {'Q->G':>8} {'G->Q':>8} {'Total':>8}")
    print("-" * 42)
    for r in results:
        print(f"{r['k']:>4} {r['residual']:>10.4f} {r['qwen_to_gpt2']*100:>7.0f}% {r['gpt2_to_qwen']*100:>7.0f}% {r['total']*100:>7.0f}%")


if __name__ == "__main__":
    main()
