"""
Comprehensive Test Suite for Vector Communication Protocol

Tests:
1. Multi-model compatibility matrix
2. Scale tests (candidate pool size)
3. Adversarial tests (paraphrases, near-duplicates)
4. Dimensionality stress tests
5. Noise robustness
6. Random baseline comparison
"""

import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from scipy.stats import spearmanr

# Will be imported after we know the path structure
import sys
sys.path.insert(0, '..')

from lib.mds import squared_distance_matrix, classical_mds, effective_rank
from lib.procrustes import procrustes_align, out_of_sample_mds, cosine_similarity


@dataclass
class TestResult:
    test_name: str
    passed: bool
    accuracy: float
    details: Dict


def get_anchors_128():
    """Standard 128-word anchor set."""
    return [
        "dog", "cat", "tree", "house", "car", "book", "water", "food",
        "love", "hate", "fear", "joy", "time", "space", "truth", "idea",
        "run", "walk", "think", "speak", "create", "destroy", "give", "take",
        "big", "small", "fast", "slow", "hot", "cold", "good", "bad",
        "above", "below", "inside", "outside", "before", "after", "with", "without",
        "one", "many", "all", "none", "more", "less", "equal", "different",
        "science", "art", "music", "math", "language", "nature", "technology", "society",
        "question", "answer", "problem", "solution", "cause", "effect", "begin", "end",
        "person", "animal", "plant", "machine", "building", "road", "mountain", "river",
        "happy", "sad", "angry", "calm", "excited", "bored", "curious", "confused",
        "see", "hear", "touch", "smell", "taste", "feel", "know", "believe",
        "red", "blue", "green", "white", "black", "bright", "dark", "clear",
        "north", "south", "east", "west", "up", "down", "left", "right",
        "day", "night", "morning", "evening", "spring", "summer", "autumn", "winter",
        "mother", "father", "child", "friend", "enemy", "leader", "worker", "teacher",
        "earth", "fire", "air", "metal", "stone", "wood", "glass", "paper",
    ]


class VectorChannelTest:
    """Lightweight channel for testing."""

    def __init__(self, model_a, model_b, anchors, k=48):
        self.model_a = model_a
        self.model_b = model_b
        self.anchors = anchors
        self.k = k
        self._bootstrap()

    def _bootstrap(self):
        self.emb_a = self.model_a.encode(self.anchors)
        self.emb_a = self.emb_a / np.linalg.norm(self.emb_a, axis=1, keepdims=True)

        self.emb_b = self.model_b.encode(self.anchors)
        self.emb_b = self.emb_b / np.linalg.norm(self.emb_b, axis=1, keepdims=True)

        self.D2_a = squared_distance_matrix(self.emb_a)
        self.D2_b = squared_distance_matrix(self.emb_b)

        self.X_a, self.eig_a, self.vec_a = classical_mds(self.D2_a, k=self.k)
        self.X_b, self.eig_b, self.vec_b = classical_mds(self.D2_b, k=self.k)

        self.actual_k = min(len(self.eig_a), len(self.eig_b))
        self.spectrum_corr, _ = spearmanr(self.eig_a[:self.actual_k], self.eig_b[:self.actual_k])

        self.R_a_to_b, _ = procrustes_align(self.X_a[:, :self.actual_k], self.X_b[:, :self.actual_k])
        self.R_b_to_a, _ = procrustes_align(self.X_b[:, :self.actual_k], self.X_a[:, :self.actual_k])

    def send_a_to_b(self, text):
        v = self.model_a.encode([text])[0]
        v = v / np.linalg.norm(v)
        d2 = 2 * (1 - self.emb_a @ v)
        y = out_of_sample_mds(d2.reshape(1, -1), self.D2_a, self.vec_a, self.eig_a)[0]
        return y[:self.actual_k] @ self.R_a_to_b[:self.actual_k, :self.actual_k]

    def send_b_to_a(self, text):
        v = self.model_b.encode([text])[0]
        v = v / np.linalg.norm(v)
        d2 = 2 * (1 - self.emb_b @ v)
        y = out_of_sample_mds(d2.reshape(1, -1), self.D2_b, self.vec_b, self.eig_b)[0]
        return y[:self.actual_k] @ self.R_b_to_a[:self.actual_k, :self.actual_k]

    def match_in_b(self, vector, candidates):
        best, best_score = None, -999
        for cand in candidates:
            v = self.model_b.encode([cand])[0]
            v = v / np.linalg.norm(v)
            d2 = 2 * (1 - self.emb_b @ v)
            y = out_of_sample_mds(d2.reshape(1, -1), self.D2_b, self.vec_b, self.eig_b)[0]
            sim = cosine_similarity(y[:self.actual_k], vector)
            if sim > best_score:
                best_score, best = sim, cand
        return best, best_score

    def match_in_a(self, vector, candidates):
        best, best_score = None, -999
        for cand in candidates:
            v = self.model_a.encode([cand])[0]
            v = v / np.linalg.norm(v)
            d2 = 2 * (1 - self.emb_a @ v)
            y = out_of_sample_mds(d2.reshape(1, -1), self.D2_a, self.vec_a, self.eig_a)[0]
            sim = cosine_similarity(y[:self.actual_k], vector)
            if sim > best_score:
                best_score, best = sim, cand
        return best, best_score


# =============================================================================
# TEST 1: Multi-Model Compatibility Matrix
# =============================================================================

def test_model_compatibility_matrix(models: Dict[str, any]) -> TestResult:
    """Test all pairs of models for communication compatibility."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Compatibility Matrix")
    print("=" * 60)

    anchors = get_anchors_128()
    model_names = list(models.keys())
    n = len(model_names)

    results = {}
    matrix = np.zeros((n, n))

    test_messages = [
        "The quick brown fox jumps",
        "I love building things",
        "Mathematics is beautiful",
    ]

    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if i == j:
                matrix[i, j] = 1.0
                continue

            print(f"  Testing {name_a} <-> {name_b}...", end=" ")

            channel = VectorChannelTest(models[name_a], models[name_b], anchors, k=48)

            # Test A -> B
            correct = 0
            for msg in test_messages:
                vec = channel.send_a_to_b(msg)
                match, _ = channel.match_in_b(vec, test_messages)
                if match == msg:
                    correct += 1

            acc = correct / len(test_messages)
            matrix[i, j] = acc
            results[f"{name_a}->{name_b}"] = {
                "accuracy": acc,
                "spectrum_correlation": channel.spectrum_corr
            }

            print(f"{acc*100:.0f}%")

    print("\nCompatibility Matrix:")
    print("     ", "  ".join(f"{n[:6]:>6}" for n in model_names))
    for i, name in enumerate(model_names):
        row = "  ".join(f"{matrix[i,j]*100:5.0f}%" for j in range(n))
        print(f"{name[:5]:>5} {row}")

    all_pairs_pass = all(v["accuracy"] >= 0.66 for v in results.values())

    return TestResult(
        test_name="model_compatibility_matrix",
        passed=all_pairs_pass,
        accuracy=np.mean([v["accuracy"] for v in results.values()]),
        details={"matrix": matrix.tolist(), "pairs": results}
    )


# =============================================================================
# TEST 2: Scale Test (Candidate Pool Size)
# =============================================================================

def test_scale(model_a, model_b, pool_sizes=[10, 20, 50, 100]) -> TestResult:
    """Test accuracy as candidate pool grows."""
    print("\n" + "=" * 60)
    print("TEST 2: Scale Test (Candidate Pool Size)")
    print("=" * 60)

    anchors = get_anchors_128()
    channel = VectorChannelTest(model_a, model_b, anchors, k=48)

    # Generate a large pool of sentences
    base_messages = [
        "The quick brown fox jumps over the lazy dog",
        "I love programming and building things",
        "Mathematics is the language of the universe",
        "She walked slowly through the quiet forest",
        "The coffee was hot and delicious",
    ]

    # Distractors
    distractors = [
        "The cat sat on the mat",
        "He ran across the street",
        "Science explains nature",
        "Birds fly in the sky",
        "The ocean is deep and blue",
        "Music fills the room with joy",
        "Children play in the park",
        "Books contain knowledge",
        "The sun sets in the west",
        "Rain falls on the roof",
        "Computers process information",
        "Trees grow in the forest",
        "Fish swim in the water",
        "Mountains rise above clouds",
        "The wind blows through trees",
        "Stars shine at night",
        "Dogs bark at strangers",
        "Flowers bloom in spring",
        "Ice melts in summer heat",
        "Snow covers the ground",
        "Rivers flow to the sea",
        "Bridges connect two sides",
        "Roads lead to cities",
        "Trains carry passengers",
        "Planes fly through clouds",
        "Ships sail across oceans",
        "Clocks measure time passing",
        "Mirrors reflect light rays",
        "Windows let sunlight in",
        "Doors open and close",
        # Continue generating more...
    ] * 3  # Repeat to get enough

    results = {}
    for pool_size in pool_sizes:
        # Create candidate pool
        candidates = base_messages + distractors[:pool_size - len(base_messages)]
        candidates = candidates[:pool_size]

        correct = 0
        for msg in base_messages:
            if msg not in candidates:
                continue
            vec = channel.send_a_to_b(msg)
            match, score = channel.match_in_b(vec, candidates)
            if match == msg:
                correct += 1

        acc = correct / len(base_messages)
        results[pool_size] = acc
        print(f"  Pool size {pool_size:>3}: {acc*100:.0f}% accuracy")

    return TestResult(
        test_name="scale_test",
        passed=results.get(100, 0) >= 0.8,
        accuracy=results.get(100, 0),
        details={"pool_size_results": results}
    )


# =============================================================================
# TEST 3: Adversarial Test (Paraphrases)
# =============================================================================

def test_paraphrase_discrimination(model_a, model_b) -> TestResult:
    """Test discrimination between paraphrases."""
    print("\n" + "=" * 60)
    print("TEST 3: Paraphrase Discrimination")
    print("=" * 60)

    anchors = get_anchors_128()
    channel = VectorChannelTest(model_a, model_b, anchors, k=48)

    # Original + paraphrase pairs
    test_pairs = [
        ("The dog ran quickly", "The canine sprinted fast"),
        ("I enjoy reading books", "I like perusing literature"),
        ("The weather is nice today", "It's pleasant outside now"),
        ("She speaks three languages", "She is trilingual"),
        ("The car is expensive", "The vehicle costs a lot"),
    ]

    results = []
    for original, paraphrase in test_pairs:
        candidates = [original, paraphrase]

        # Send original, should match original not paraphrase
        vec = channel.send_a_to_b(original)
        match, score = channel.match_in_b(vec, candidates)

        correct = match == original
        results.append({
            "original": original,
            "paraphrase": paraphrase,
            "matched": match,
            "correct": correct,
            "score": score
        })

        status = "OK" if correct else "FAIL"
        print(f"  '{original[:30]}...' -> '{match[:30]}...' [{status}]")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\n  Paraphrase discrimination: {accuracy*100:.0f}%")

    return TestResult(
        test_name="paraphrase_discrimination",
        passed=accuracy >= 0.6,
        accuracy=accuracy,
        details={"pairs": results}
    )


# =============================================================================
# TEST 4: Dimensionality Stress Test
# =============================================================================

def test_dimensionality_stress(model_a, model_b) -> TestResult:
    """Test accuracy at different k values."""
    print("\n" + "=" * 60)
    print("TEST 4: Dimensionality Stress Test")
    print("=" * 60)

    anchors = get_anchors_128()

    test_messages = [
        "The quick brown fox jumps",
        "I love building things",
        "Mathematics is beautiful",
        "The weather is cold today",
    ]
    candidates = test_messages + [
        "The cat sat on the mat",
        "He ran across the street",
        "Science explains nature",
        "It's hot outside",
    ]

    results = {}
    for k in [4, 8, 16, 24, 32, 48, 64]:
        channel = VectorChannelTest(model_a, model_b, anchors, k=k)

        correct = 0
        for msg in test_messages:
            vec = channel.send_a_to_b(msg)
            match, _ = channel.match_in_b(vec, candidates)
            if match == msg:
                correct += 1

        acc = correct / len(test_messages)
        results[k] = acc
        compression = 384 / channel.actual_k
        print(f"  k={k:>2} (actual={channel.actual_k:>2}): {acc*100:>5.1f}% accuracy, {compression:.1f}x compression")

    return TestResult(
        test_name="dimensionality_stress",
        passed=results.get(32, 0) >= 0.75,
        accuracy=results.get(48, 0),
        details={"k_results": results}
    )


# =============================================================================
# TEST 5: Noise Robustness
# =============================================================================

def test_noise_robustness(model_a, model_b) -> TestResult:
    """Test robustness to noise in transmitted vectors."""
    print("\n" + "=" * 60)
    print("TEST 5: Noise Robustness")
    print("=" * 60)

    anchors = get_anchors_128()
    channel = VectorChannelTest(model_a, model_b, anchors, k=48)

    test_messages = [
        "The quick brown fox jumps",
        "I love building things",
        "Mathematics is beautiful",
    ]
    candidates = test_messages + ["The cat sat", "He ran fast", "Science rules"]

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = {}

    for noise_std in noise_levels:
        correct = 0
        for msg in test_messages:
            vec = channel.send_a_to_b(msg)

            # Add Gaussian noise
            if noise_std > 0:
                noise = np.random.randn(len(vec)) * noise_std
                vec = vec + noise

            match, _ = channel.match_in_b(vec, candidates)
            if match == msg:
                correct += 1

        acc = correct / len(test_messages)
        results[noise_std] = acc
        print(f"  Noise std={noise_std:.2f}: {acc*100:.0f}% accuracy")

    return TestResult(
        test_name="noise_robustness",
        passed=results.get(0.1, 0) >= 0.66,
        accuracy=results.get(0.0, 0),
        details={"noise_results": results}
    )


# =============================================================================
# TEST 6: Random Baseline
# =============================================================================

def test_random_baseline(model_a, model_b) -> TestResult:
    """Compare trained models to random projections."""
    print("\n" + "=" * 60)
    print("TEST 6: Random Baseline Comparison")
    print("=" * 60)

    anchors = get_anchors_128()

    # Trained channel
    channel_trained = VectorChannelTest(model_a, model_b, anchors, k=48)

    test_messages = [
        "The quick brown fox jumps",
        "I love building things",
        "Mathematics is beautiful",
    ]
    candidates = test_messages + ["The cat sat", "He ran fast", "Science rules"]

    # Test trained
    correct_trained = 0
    for msg in test_messages:
        vec = channel_trained.send_a_to_b(msg)
        match, _ = channel_trained.match_in_b(vec, candidates)
        if match == msg:
            correct_trained += 1

    acc_trained = correct_trained / len(test_messages)

    # Random baseline: random projections
    dim_a = model_a.get_sentence_embedding_dimension()
    dim_b = model_b.get_sentence_embedding_dimension()
    k = 48

    # Random projection matrices
    np.random.seed(42)
    P_a = np.random.randn(dim_a, k)
    P_b = np.random.randn(dim_b, k)

    # Project and align
    def random_project(model, text, P):
        v = model.encode([text])[0]
        v = v / np.linalg.norm(v)
        return v @ P

    anchor_a = np.array([random_project(model_a, a, P_a) for a in anchors])
    anchor_b = np.array([random_project(model_b, a, P_b) for a in anchors])

    R_random, _ = procrustes_align(anchor_a, anchor_b)

    correct_random = 0
    for msg in test_messages:
        vec = random_project(model_a, msg, P_a) @ R_random

        best, best_score = None, -999
        for cand in candidates:
            y = random_project(model_b, cand, P_b)
            sim = cosine_similarity(y, vec)
            if sim > best_score:
                best_score, best = sim, cand

        if best == msg:
            correct_random += 1

    acc_random = correct_random / len(test_messages)

    print(f"  Trained models: {acc_trained*100:.0f}%")
    print(f"  Random projections: {acc_random*100:.0f}%")
    print(f"  Advantage: +{(acc_trained - acc_random)*100:.0f}%")

    return TestResult(
        test_name="random_baseline",
        passed=acc_trained > acc_random,
        accuracy=acc_trained,
        details={
            "trained_accuracy": acc_trained,
            "random_accuracy": acc_random,
            "advantage": acc_trained - acc_random
        }
    )


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests and generate report."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        return

    print("=" * 60)
    print("VECTOR COMMUNICATION TEST SUITE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load models
    print("\nLoading models...")
    models = {
        "MiniLM": SentenceTransformer('all-MiniLM-L6-v2'),
        "MPNet": SentenceTransformer('all-mpnet-base-v2'),
    }

    # Check if more models available
    try:
        models["paraphrase"] = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("  Loaded 3 models")
    except:
        print("  Loaded 2 models")

    model_a = models["MiniLM"]
    model_b = models["MPNet"]

    results = []

    # Run tests
    results.append(test_model_compatibility_matrix(models))
    results.append(test_scale(model_a, model_b))
    results.append(test_paraphrase_discrimination(model_a, model_b))
    results.append(test_dimensionality_stress(model_a, model_b))
    results.append(test_noise_robustness(model_a, model_b))
    results.append(test_random_baseline(model_a, model_b))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}: {r.accuracy*100:.1f}%")

    print(f"\n  Total: {passed}/{total} tests passed")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total
        },
        "tests": [asdict(r) for r in results]
    }

    with open("test_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to test_results.json")

    return results


if __name__ == "__main__":
    run_all_tests()
