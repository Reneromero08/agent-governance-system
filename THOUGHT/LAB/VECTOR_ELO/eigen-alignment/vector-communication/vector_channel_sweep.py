"""Sweep k and anchor count to find optimal parameters."""

import numpy as np
from sentence_transformers import SentenceTransformer

from lib.mds import squared_distance_matrix, classical_mds, effective_rank
from lib.procrustes import procrustes_align, out_of_sample_mds, cosine_similarity


def run_channel_test(model_a, model_b, anchors, k, test_messages, candidates):
    """Run a single channel test and return accuracy."""

    def embed_a(texts):
        return model_a.encode(texts)

    def embed_b(texts):
        return model_b.encode(texts)

    # Create endpoints
    emb_a = embed_a(anchors)
    emb_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    D2_a = squared_distance_matrix(emb_a)
    X_a, eig_a, vec_a = classical_mds(D2_a, k=k)

    emb_b = embed_b(anchors)
    emb_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    D2_b = squared_distance_matrix(emb_b)
    X_b, eig_b, vec_b = classical_mds(D2_b, k=k)

    # Align
    actual_k = min(len(eig_a), len(eig_b))
    R, residual = procrustes_align(X_a[:, :actual_k], X_b[:, :actual_k])

    # Test A -> B
    correct = 0
    for msg in test_messages:
        # Send from A
        v = embed_a([msg])[0]
        v = v / np.linalg.norm(v)
        d2 = 2 * (1 - emb_a @ v)
        y = out_of_sample_mds(d2.reshape(1, -1), D2_a, vec_a, eig_a)[0]
        y_rot = y[:actual_k] @ R[:actual_k, :actual_k]

        # Receive at B
        best_match = None
        best_score = -999
        for cand in candidates:
            c = embed_b([cand])[0]
            c = c / np.linalg.norm(c)
            d2_c = 2 * (1 - emb_b @ c)
            y_c = out_of_sample_mds(d2_c.reshape(1, -1), D2_b, vec_b, eig_b)[0]
            sim = cosine_similarity(y_c[:actual_k], y_rot)
            if sim > best_score:
                best_score = sim
                best_match = cand

        if best_match == msg:
            correct += 1

    return correct / len(test_messages), actual_k


def main():
    print("Loading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')
    model_b = SentenceTransformer('all-mpnet-base-v2')

    # Extended anchor sets
    anchors_small = [
        "dog", "cat", "tree", "house", "car", "book", "water", "food",
        "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
        "run", "walk", "think", "speak", "create", "destroy", "give", "take",
        "big", "small", "fast", "slow", "hot", "cold", "good", "bad",
    ]  # 32

    anchors_medium = anchors_small + [
        "above", "below", "inside", "outside", "before", "after", "with", "without",
        "one", "many", "all", "none", "more", "less", "equal", "different",
        "science", "art", "music", "math", "language", "nature", "technology", "society",
        "question", "answer", "problem", "solution", "cause", "effect", "begin", "end",
    ]  # 64

    anchors_large = anchors_medium + [
        "person", "animal", "plant", "machine", "building", "road", "mountain", "river",
        "happy", "sad", "angry", "calm", "excited", "bored", "curious", "confused",
        "see", "hear", "touch", "smell", "taste", "feel", "know", "believe",
        "red", "blue", "green", "white", "black", "bright", "dark", "clear",
        "north", "south", "east", "west", "up", "down", "left", "right",
        "day", "night", "morning", "evening", "spring", "summer", "autumn", "winter",
        "mother", "father", "child", "friend", "enemy", "leader", "worker", "teacher",
        "earth", "fire", "air", "metal", "stone", "wood", "glass", "paper",
    ]  # 128

    # Test messages
    test_messages = [
        "The quick brown fox jumps over the lazy dog",
        "I love programming and building things",
        "The weather is cold and rainy today",
        "Mathematics is the language of the universe",
        "She walked slowly through the quiet forest",
        "The coffee was hot and delicious this morning",
        "Scientists discovered a new species of butterfly",
        "He played the piano beautifully at the concert",
    ]

    candidates = test_messages + [
        "The cat sat on the mat",
        "I hate doing boring tasks",
        "The sun is hot and bright today",
        "Poetry is the soul of humanity",
        "He ran quickly across the busy street",
        "Cooking dinner for the family",
        "Reading a book by the fireplace",
        "The ocean waves crash on the shore",
        "A bird sang sweetly in the tree",
        "The computer processed data rapidly",
        "Children laughed in the playground",
        "Stars twinkled in the night sky",
    ]

    print("\n" + "=" * 70)
    print("PARAMETER SWEEP: Finding optimal k and anchor count")
    print("=" * 70)

    results = []
    for anchors, anchor_name in [
        (anchors_small, "32"),
        (anchors_medium, "64"),
        (anchors_large, "128")
    ]:
        for k in [8, 16, 32, 48, 64]:
            if k > len(anchors) - 1:
                continue

            acc, actual_k = run_channel_test(
                model_a, model_b, anchors, k, test_messages, candidates
            )

            compression = 384 / actual_k
            results.append({
                'anchors': anchor_name,
                'k': actual_k,
                'accuracy': acc,
                'compression': compression
            })

            print(f"  Anchors={anchor_name:>3}, k={actual_k:>2}: accuracy={acc*100:5.1f}%, compression={compression:5.1f}x")

    print("\n" + "=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\nTop 5 by accuracy:")
    for r in results[:5]:
        print(f"  Anchors={r['anchors']}, k={r['k']}: {r['accuracy']*100:.1f}% ({r['compression']:.1f}x compression)")

    # Find best accuracy-compression tradeoff
    print("\nBest accuracy at each compression level:")
    for target_comp in [10, 20, 30, 50]:
        best = max(
            [r for r in results if r['compression'] >= target_comp * 0.8],
            key=lambda x: x['accuracy'],
            default=None
        )
        if best:
            print(f"  {target_comp}x+ compression: {best['accuracy']*100:.1f}% (anchors={best['anchors']}, k={best['k']})")


if __name__ == "__main__":
    main()
