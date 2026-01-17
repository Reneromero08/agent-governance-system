#!/usr/bin/env python3
"""Native LLM-to-LLM Vector Communication via AlignmentKey.

This test proves that LLMs can communicate through vectors using
their NATIVE embeddings (not a separate embedding model).

Key insight from research:
- Higher residual is overcome with MORE dimensions
- ANCHOR_512 with k=256 achieved 100% at 50% corruption
- Even with residual ~17, communication works!

So for native LLM embeddings (residual ~5.3), we should succeed
with large anchors and high k.
"""

import numpy as np
import requests
import sys
from pathlib import Path

# Add project root (6 levels up from tests/ subdirectory)
PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey

# Import large_anchor_generator from lib directory
import importlib.util
spec = importlib.util.spec_from_file_location("large_anchor_generator",
    Path(__file__).parent.parent / "lib" / "large_anchor_generator.py")
lag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lag)
ANCHOR_512 = lag.ANCHOR_512
ANCHOR_256 = lag.ANCHOR_256

# Generate ANCHOR_777 (all available words)
ANCHOR_777 = lag.generate_anchor_set(777)

OLLAMA_URL = "http://localhost:11434"
SESSION = requests.Session()

# =============================================================================
# NATIVE LLM EMBEDDING FUNCTIONS
# =============================================================================

def embed_qwen_native(texts):
    """Get qwen's NATIVE internal embeddings (3584D)."""
    results = []
    for t in texts:
        resp = SESSION.post(f"{OLLAMA_URL}/api/embed",
                           json={"model": "qwen2.5:7b", "input": t})
        emb = np.array(resp.json()["embeddings"][0])
        results.append(emb)
    return np.array(results)


def embed_mistral_native(texts):
    """Get mistral's NATIVE internal embeddings (4096D)."""
    results = []
    for t in texts:
        resp = SESSION.post(f"{OLLAMA_URL}/api/embed",
                           json={"model": "mistral:7b", "input": t})
        emb = np.array(resp.json()["embeddings"][0])
        results.append(emb)
    return np.array(results)


def test_cca_alignment():
    """Test CCA-based alignment for native LLM communication."""
    from sklearn.cross_decomposition import CCA

    print("=" * 60)
    print("CCA-BASED NATIVE LLM ALIGNMENT")
    print("Learning direct mapping between native spaces")
    print("=" * 60)

    # Use 256 anchors for training CCA
    train_anchors = ANCHOR_256

    print(f"\n[1] Embedding {len(train_anchors)} anchors with both models...")

    # Embed anchors with both models
    X_qwen = embed_qwen_native(train_anchors)
    X_mistral = embed_mistral_native(train_anchors)

    print(f"    qwen embeddings: {X_qwen.shape}")
    print(f"    mistral embeddings: {X_mistral.shape}")

    # Normalize
    X_qwen = X_qwen / np.linalg.norm(X_qwen, axis=1, keepdims=True)
    X_mistral = X_mistral / np.linalg.norm(X_mistral, axis=1, keepdims=True)

    # Fit CCA with max components
    n_components = min(128, len(train_anchors))
    print(f"\n[2] Fitting CCA with {n_components} components...")
    cca = CCA(n_components=n_components)
    cca.fit(X_qwen, X_mistral)

    # Get the projected anchors
    X_qwen_cca, X_mistral_cca = cca.transform(X_qwen, X_mistral)

    # Check anchor alignment quality
    sims = []
    for i in range(len(train_anchors)):
        cos_sim = np.dot(X_qwen_cca[i], X_mistral_cca[i]) / (
            np.linalg.norm(X_qwen_cca[i]) * np.linalg.norm(X_mistral_cca[i]) + 1e-8)
        sims.append(cos_sim)
    print(f"    Mean anchor alignment: {np.mean(sims):.4f}")
    print(f"    Min anchor alignment: {np.min(sims):.4f}")

    # Test communication
    print("\n[3] Testing cross-LLM communication via CCA...")

    test_messages = [
        "The quick brown fox jumps",
        "Neural networks learn patterns",
        "Water flows downhill naturally",
        "Music brings people together",
        "Dogs are loyal companions",
        "The moon orbits Earth",
        "Children play in parks",
        "Scientists discover new things",
    ]

    distractors = [
        ["The slow blue cat crawls", "A tree grows in soil", "Cars need gasoline fuel"],
        ["Cats sleep all day long", "Fire burns hot and bright", "Books contain many words"],
        ["Ice cream melts quickly", "Birds fly through sky", "Mountains stand tall forever"],
        ["Silence can be golden", "Stars shine at night", "Coffee helps people wake"],
        ["Cats are independent pets", "Fish swim in water", "Birds sing in trees"],
        ["Stars twinkle at night", "The sun shines brightly", "Planets rotate slowly"],
        ["Adults work in offices", "Birds build their nests", "Flowers bloom in spring"],
        ["Artists create paintings", "Teachers educate students", "Engineers build bridges"],
    ]

    def encode_to_cca(text, embed_fn, is_qwen=True):
        """Encode text to CCA space."""
        vec = embed_fn([text])[0]
        vec = vec / np.linalg.norm(vec)
        if is_qwen:
            # Transform qwen to CCA space
            cca_vec = cca.transform(vec.reshape(1, -1), np.zeros((1, X_mistral.shape[1])))[0][0]
        else:
            # Transform mistral to CCA space
            cca_vec = cca.transform(np.zeros((1, X_qwen.shape[1])), vec.reshape(1, -1))[1][0]
        return cca_vec

    def decode_from_cca(vec, candidates, embed_fn, is_qwen=True):
        """Decode CCA vector to best candidate."""
        best_match = None
        best_score = -float('inf')
        for cand in candidates:
            cand_vec = encode_to_cca(cand, embed_fn, is_qwen)
            sim = np.dot(vec, cand_vec) / (np.linalg.norm(vec) * np.linalg.norm(cand_vec) + 1e-8)
            if sim > best_score:
                best_score = sim
                best_match = cand
        return best_match, best_score

    print("\n    qwen -> mistral (via CCA):")
    correct_qm = 0
    for i, msg in enumerate(test_messages):
        candidates = [msg] + distractors[i]
        np.random.shuffle(candidates)

        # Encode at qwen, decode at mistral
        vec = encode_to_cca(msg, embed_qwen_native, is_qwen=True)
        match, score = decode_from_cca(vec, candidates, embed_mistral_native, is_qwen=False)

        ok = match == msg
        if ok:
            correct_qm += 1
        status = "OK" if ok else "FAIL"
        print(f"      [{status}] '{msg[:30]}...' -> '{match[:30]}...' (conf={score:.3f})")

    print(f"\n    mistral -> qwen (via CCA):")
    correct_mq = 0
    for i, msg in enumerate(test_messages):
        candidates = [msg] + distractors[i]
        np.random.shuffle(candidates)

        # Encode at mistral, decode at qwen
        vec = encode_to_cca(msg, embed_mistral_native, is_qwen=False)
        match, score = decode_from_cca(vec, candidates, embed_qwen_native, is_qwen=True)

        ok = match == msg
        if ok:
            correct_mq += 1
        status = "OK" if ok else "FAIL"
        print(f"      [{status}] '{msg[:30]}...' -> '{match[:30]}...' (conf={score:.3f})")

    total = len(test_messages)
    print("\n" + "=" * 60)
    print(f"CCA RESULTS: qwen->mistral {correct_qm}/{total} ({100*correct_qm/total:.0f}%)")
    print(f"             mistral->qwen {correct_mq}/{total} ({100*correct_mq/total:.0f}%)")
    print("=" * 60)

    return correct_qm == total and correct_mq == total


def test_native_communication():
    """Test native LLM-to-LLM communication using AlignmentKey."""
    print("=" * 60)
    print("NATIVE LLM-TO-LLM VECTOR COMMUNICATION")
    print("Using AlignmentKey with large anchors + high k")
    print("=" * 60)

    # Verify models are available
    print("\n[1] Checking LLM native embedding dimensions...")
    try:
        test_q = embed_qwen_native(["test"])[0]
        test_m = embed_mistral_native(["test"])[0]
        print(f"    qwen2.5:7b native: {len(test_q)}D")
        print(f"    mistral:7b native: {len(test_m)}D")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure Ollama is running with qwen2.5:7b and mistral:7b")
        return False

    # Use ANCHOR_512 - better balance of coverage and residual
    # ANCHOR_777 has higher residual which hurts out-of-sample
    anchors = ANCHOR_512
    k = 256  # High k for maximum redundancy

    print(f"\n[2] Creating AlignmentKeys with {len(anchors)} anchors, k={k}...")
    print("    This embeds all anchors with each model's native embeddings...")

    # Create alignment keys using NATIVE LLM embeddings
    print("    Creating qwen key...")
    key_qwen = AlignmentKey.create("qwen-native", embed_qwen_native, anchors=anchors, k=k)
    print(f"    qwen key: {key_qwen.k} dimensions")

    print("    Creating mistral key...")
    key_mistral = AlignmentKey.create("mistral-native", embed_mistral_native, anchors=anchors, k=k)
    print(f"    mistral key: {key_mistral.k} dimensions")

    # Align the two keys
    print("\n[3] Aligning native LLM spaces with Procrustes...")
    pair = key_qwen.align_with(key_mistral)

    print(f"    Spectrum correlation: {pair.spectrum_correlation:.4f}")
    print(f"    Procrustes residual (a->b): {pair.procrustes_residual:.4f}")

    # Check anchor alignment quality in both directions
    print("\n    Checking anchor alignment quality...")
    test_anchors = ["dog", "cat", "water", "fire", "think", "run"]

    print("    Direction a->b (qwen->mistral):")
    for anchor in test_anchors:
        vec_q = key_qwen.encode(anchor, embed_qwen_native)[:pair.k]
        vec_m = key_mistral.encode(anchor, embed_mistral_native)[:pair.k]
        vec_q_rot = vec_q @ pair.R_a_to_b
        cos_sim = np.dot(vec_q_rot, vec_m) / (np.linalg.norm(vec_q_rot) * np.linalg.norm(vec_m) + 1e-8)
        print(f"      {anchor:10s}: cos_sim={cos_sim:.3f}")

    print("    Direction b->a (mistral->qwen):")
    for anchor in test_anchors:
        vec_q = key_qwen.encode(anchor, embed_qwen_native)[:pair.k]
        vec_m = key_mistral.encode(anchor, embed_mistral_native)[:pair.k]
        vec_m_rot = vec_m @ pair.R_b_to_a
        cos_sim = np.dot(vec_m_rot, vec_q) / (np.linalg.norm(vec_m_rot) * np.linalg.norm(vec_q) + 1e-8)
        print(f"      {anchor:10s}: cos_sim={cos_sim:.3f}")

    # Test communication
    print("\n[4] Testing cross-LLM communication...")

    test_messages = [
        "The quick brown fox jumps",
        "Neural networks learn patterns",
        "Water flows downhill naturally",
        "Music brings people together",
        "Dogs are loyal companions",
        "The moon orbits Earth",
        "Children play in parks",
        "Scientists discover new things",
    ]

    # Use MORE DISTINCT distractors (different domains entirely)
    distractors = [
        ["Computers need electricity", "Mountains are very tall", "Doctors heal patients"],
        ["The ocean is vast", "Coffee wakes people up", "Cars need fuel"],
        ["Books contain knowledge", "Music has rhythm", "Artists paint pictures"],
        ["Fish swim in water", "Trees grow in forests", "Stars shine at night"],
        ["Mathematics uses numbers", "History studies the past", "Chemistry mixes substances"],
        ["Flowers bloom in spring", "Ice is frozen water", "Wind moves the air"],
        ["Computers process data", "Engineers build bridges", "Lawyers study law"],
        ["Music has melody", "Food provides energy", "Sleep restores the body"],
    ]

    print("\n    qwen -> mistral:")
    correct_qm = 0
    for i, msg in enumerate(test_messages):
        candidates = [msg] + distractors[i]
        np.random.shuffle(candidates)

        # Encode at qwen, decode at mistral
        vec = pair.encode_a_to_b(msg, embed_qwen_native)
        match, score = pair.decode_at_b(vec, candidates, embed_mistral_native)

        ok = match == msg
        if ok:
            correct_qm += 1
        status = "OK" if ok else "FAIL"
        print(f"      [{status}] '{msg[:30]}...' -> '{match[:30]}...' (conf={score:.3f})")

    print(f"\n    mistral -> qwen:")
    correct_mq = 0
    for i, msg in enumerate(test_messages):
        candidates = [msg] + distractors[i]
        np.random.shuffle(candidates)

        # Encode at mistral, decode at qwen
        vec = pair.encode_b_to_a(msg, embed_mistral_native)
        match, score = pair.decode_at_a(vec, candidates, embed_qwen_native)

        ok = match == msg
        if ok:
            correct_mq += 1
        status = "OK" if ok else "FAIL"
        print(f"      [{status}] '{msg[:30]}...' -> '{match[:30]}...' (conf={score:.3f})")

    total = len(test_messages)
    print("\n" + "=" * 60)
    print(f"RESULTS: qwen->mistral {correct_qm}/{total} ({100*correct_qm/total:.0f}%)")
    print(f"         mistral->qwen {correct_mq}/{total} ({100*correct_mq/total:.0f}%)")
    print("=" * 60)

    if correct_qm == total and correct_mq == total:
        print("\n*** NATIVE LLM COMMUNICATION WORKS! ***")
        print("The AlignmentKey successfully translates between native LLM spaces.")
        return True
    else:
        # If not 100%, try with corruption test
        print("\n[5] Testing with corruption (to see robustness)...")
        test_corruption(pair)
        return False


def test_corruption(pair):
    """Test corruption tolerance of native LLM alignment."""
    msg = "The quick brown fox jumps"
    candidates = [
        msg,
        "The slow blue cat crawls",
        "A tree grows in soil",
        "Cars need gasoline fuel",
    ]

    corruption_levels = [0.0, 0.1, 0.25, 0.5]

    print(f"\n    Testing: '{msg}'")
    print(f"    Candidates: {len(candidates)}")

    for corruption in corruption_levels:
        correct = 0
        trials = 10

        for _ in range(trials):
            # Encode at qwen
            vec = pair.encode_a_to_b(msg, embed_qwen_native)

            # Apply corruption
            if corruption > 0:
                mask = np.random.random(len(vec)) < corruption
                vec[mask] = np.random.randn(mask.sum())

            # Decode at mistral
            match, score = pair.decode_at_b(vec, candidates, embed_mistral_native)

            if match == msg:
                correct += 1

        print(f"    {int(corruption*100):3d}% corruption: {correct}/{trials} ({100*correct/trials:.0f}%)")


def test_word_level():
    """Test communication with single words (should work if alignment is correct)."""
    print("=" * 60)
    print("WORD-LEVEL NATIVE LLM COMMUNICATION TEST")
    print("Testing with single words from anchor vocabulary")
    print("=" * 60)

    # Use AlignmentKey with good parameters
    anchors = ANCHOR_512
    k = 256

    print(f"\n[1] Creating keys with {len(anchors)} anchors, k={k}...")
    key_qwen = AlignmentKey.create("qwen-native", embed_qwen_native, anchors=anchors, k=k)
    key_mistral = AlignmentKey.create("mistral-native", embed_mistral_native, anchors=anchors, k=k)

    pair = key_qwen.align_with(key_mistral)
    print(f"    Procrustes residual: {pair.procrustes_residual:.4f}")

    # Test with held-out words (not in anchor set)
    held_out_words = [
        "programming", "algorithm", "database", "network", "security",
        "quantum", "gravity", "chemistry", "biology", "physics",
    ]

    # Create semantic distractors for each word
    word_distractors = {
        "programming": ["coding", "software", "hardware", "typing"],
        "algorithm": ["procedure", "formula", "equation", "method"],
        "database": ["storage", "memory", "archive", "library"],
        "network": ["connection", "system", "web", "link"],
        "security": ["safety", "protection", "privacy", "defense"],
        "quantum": ["atomic", "particle", "nuclear", "electron"],
        "gravity": ["force", "weight", "mass", "pressure"],
        "chemistry": ["physics", "biology", "science", "medicine"],
        "biology": ["chemistry", "anatomy", "ecology", "genetics"],
        "physics": ["chemistry", "mechanics", "dynamics", "energy"],
    }

    print("\n[2] Testing held-out words (qwen -> mistral):")
    correct = 0
    for word in held_out_words:
        candidates = [word] + word_distractors[word]
        np.random.shuffle(candidates)

        vec = pair.encode_a_to_b(word, embed_qwen_native)
        match, score = pair.decode_at_b(vec, candidates, embed_mistral_native)

        ok = match == word
        if ok:
            correct += 1
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {word:15s} -> {match:15s} (conf={score:.3f})")

    print(f"\n    WORD-LEVEL ACCURACY: {correct}/{len(held_out_words)} ({100*correct/len(held_out_words):.0f}%)")

    # Also test some anchor words (should work perfectly)
    print("\n[3] Testing anchor words (should be high accuracy):")
    test_anchors = ["dog", "cat", "water", "fire", "love", "hate", "run", "walk"]
    correct_anchor = 0
    for word in test_anchors:
        # Use similar anchors as distractors
        if word == "dog":
            distractors = ["cat", "wolf", "bird"]
        elif word == "cat":
            distractors = ["dog", "mouse", "rabbit"]
        elif word == "water":
            distractors = ["fire", "earth", "air"]
        elif word == "fire":
            distractors = ["water", "ice", "heat"]
        elif word == "love":
            distractors = ["hate", "fear", "joy"]
        elif word == "hate":
            distractors = ["love", "anger", "fear"]
        elif word == "run":
            distractors = ["walk", "jump", "fly"]
        else:  # walk
            distractors = ["run", "stand", "sit"]

        candidates = [word] + distractors
        np.random.shuffle(candidates)

        vec = pair.encode_a_to_b(word, embed_qwen_native)
        match, score = pair.decode_at_b(vec, candidates, embed_mistral_native)

        ok = match == word
        if ok:
            correct_anchor += 1
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {word:15s} -> {match:15s} (conf={score:.3f})")

    print(f"\n    ANCHOR-LEVEL ACCURACY: {correct_anchor}/{len(test_anchors)} ({100*correct_anchor/len(test_anchors):.0f}%)")

    return correct == len(held_out_words)


if __name__ == "__main__":
    # Test word-level first
    test_word_level()

    print("\n\n")

    # Test CCA alignment
    test_cca_alignment()

    print("\n\n")

    # Then test AlignmentKey with Procrustes
    test_native_communication()
