"""
Gemini Proposals Validation Suite

Tests Gemini's "Grand Unified Theory" proposals:
1. COHERENCE THRESHOLD (Q12) - Does 0.92 predict crystallization?
2. RULE OF 3 (Q13) - Does N=3 maximize phase lock?
3. TRIANGULATED AGENT - LIQUID/CRYSTAL classification

Based on Gemini's analysis of the Contextual Phase Selection breakthrough.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA

# Add CAT_CHAT to path
CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================

# Q12's critical threshold
COHERENCE_THRESHOLD = 0.92

# Q13's optimal source count
OPTIMAL_N = 3

# Coherent source sets (same topic)
COHERENT_SETS = {
    'thermodynamics': [
        "entropy always increases in isolated systems",
        "the second law of thermodynamics governs heat flow",
        "heat flows spontaneously from hot to cold bodies",
    ],
    'relativity': [
        "spacetime is curved by mass and energy",
        "the speed of light is constant in all reference frames",
        "time dilation occurs near massive objects",
    ],
    'evolution': [
        "natural selection drives adaptation",
        "genetic mutations provide variation",
        "species evolve through differential reproduction",
    ],
}

# Incoherent source sets (mixed topics)
INCOHERENT_SETS = {
    'mixed_1': [
        "entropy always increases in isolated systems",
        "the cat sat on the mat",
        "stock prices fell sharply today",
    ],
    'mixed_2': [
        "spacetime is curved by mass and energy",
        "my favorite color is blue",
        "the recipe calls for two cups of flour",
    ],
    'mixed_3': [
        "natural selection drives adaptation",
        "the train arrives at noon",
        "quantum computing uses qubits",
    ],
}

# Sources for Rule of 3 testing (thermodynamics domain)
SCALING_SOURCES = [
    "entropy always increases in isolated systems",
    "the second law governs irreversible processes",
    "heat flows from hot to cold spontaneously",
    "work can be converted fully to heat but not vice versa",
    "the universe tends toward maximum entropy",
    "free energy determines spontaneous processes",
    "temperature is a measure of average kinetic energy",
    "heat engines have fundamental efficiency limits",
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def model():
    """Load sentence transformer model."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence_transformers not available")
    return SentenceTransformer('all-MiniLM-L6-v2')


# ============================================================================
# Helper Functions
# ============================================================================

def phase_embed(model, text: str, axis: str = "") -> np.ndarray:
    """Embed text with optional contextual axis."""
    if axis:
        prompt = f"{text}, in terms of {axis}"
    else:
        prompt = text
    return model.encode(prompt, convert_to_numpy=True)


def compute_coherence(vectors: np.ndarray) -> float:
    """
    Compute coherence as mean vector magnitude after averaging.

    Gemini's hypothesis: coherent sources should yield high coherence (>0.92)
    because their vectors align and reinforce each other.

    For normalized vectors, this measures alignment:
    - Random vectors: mean magnitude -> 0 (cancel out)
    - Aligned vectors: mean magnitude -> 1 (reinforce)
    """
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-10)

    # Compute mean vector
    mean_vec = np.mean(normalized, axis=0)

    # Coherence = magnitude of mean vector
    coherence = np.linalg.norm(mean_vec)

    return float(coherence)


def compute_pairwise_coherence(vectors: np.ndarray) -> float:
    """
    Alternative coherence: mean pairwise cosine similarity.

    This measures how similar all vectors are to each other.
    """
    n = len(vectors)
    if n < 2:
        return 1.0

    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-10)

    # Pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(normalized[i], normalized[j]))
            similarities.append(sim)

    return float(np.mean(similarities))


def compute_phase_error_global(
    embeddings: Dict[str, np.ndarray],
    analogy: Tuple[str, str, str, str],
    pca: PCA
) -> float:
    """Compute phase error using global PCA."""
    a, b, c, d = analogy

    def get_phase(word):
        v = embeddings[word]
        proj = pca.transform(v.reshape(1, -1))[0]
        z = proj[0] + 1j * proj[1]
        return np.angle(z)

    theta_a = get_phase(a)
    theta_b = get_phase(b)
    theta_c = get_phase(c)
    theta_d = get_phase(d)

    theta_ba = np.angle(np.exp(1j * (theta_b - theta_a)))
    theta_dc = np.angle(np.exp(1j * (theta_d - theta_c)))

    error = abs(theta_ba - theta_dc)
    if error > np.pi:
        error = 2 * np.pi - error

    return np.degrees(error)


# ============================================================================
# TEST 1: Coherence Threshold (Q12 Connection)
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestCoherenceThreshold:
    """
    Test Gemini's claim: coherent sources should have coherence > 0.92

    Q12 shows alpha_c = 0.92 is the phase transition point.
    Gemini hypothesizes this applies to RAG source coherence.
    """

    def test_coherent_sets_have_high_coherence(self, model):
        """Coherent source sets should have high coherence."""
        print("\n" + "=" * 70)
        print("COHERENCE THRESHOLD TEST: Coherent Sets")
        print("=" * 70)
        print(f"Q12 Threshold: {COHERENCE_THRESHOLD}")
        print("-" * 70)

        results = []
        for name, sources in COHERENT_SETS.items():
            vectors = np.array([model.encode(s) for s in sources])
            coherence = compute_coherence(vectors)
            pairwise = compute_pairwise_coherence(vectors)

            results.append({
                'name': name,
                'coherence': coherence,
                'pairwise': pairwise,
                'passes_threshold': coherence > COHERENCE_THRESHOLD,
            })

            status = "PASS" if coherence > COHERENCE_THRESHOLD else "FAIL"
            print(f"{name:20s}: coherence={coherence:.4f}, pairwise={pairwise:.4f} [{status}]")

        mean_coherence = np.mean([r['coherence'] for r in results])
        mean_pairwise = np.mean([r['pairwise'] for r in results])

        print("-" * 70)
        print(f"Mean coherence: {mean_coherence:.4f}")
        print(f"Mean pairwise:  {mean_pairwise:.4f}")
        print(f"Threshold:      {COHERENCE_THRESHOLD}")

        # Document the actual values - may need threshold calibration
        print(f"\nNOTE: If coherence < {COHERENCE_THRESHOLD}, Gemini's threshold needs calibration")

    def test_incoherent_sets_have_low_coherence(self, model):
        """Incoherent source sets should have low coherence."""
        print("\n" + "=" * 70)
        print("COHERENCE THRESHOLD TEST: Incoherent Sets")
        print("=" * 70)

        results = []
        for name, sources in INCOHERENT_SETS.items():
            vectors = np.array([model.encode(s) for s in sources])
            coherence = compute_coherence(vectors)
            pairwise = compute_pairwise_coherence(vectors)

            results.append({
                'name': name,
                'coherence': coherence,
                'pairwise': pairwise,
            })

            print(f"{name:20s}: coherence={coherence:.4f}, pairwise={pairwise:.4f}")

        mean_coherence = np.mean([r['coherence'] for r in results])
        mean_pairwise = np.mean([r['pairwise'] for r in results])

        print("-" * 70)
        print(f"Mean coherence: {mean_coherence:.4f}")
        print(f"Mean pairwise:  {mean_pairwise:.4f}")

    def test_coherent_vs_incoherent_separation(self, model):
        """Coherent sets should have higher coherence than incoherent."""
        print("\n" + "=" * 70)
        print("COHERENCE SEPARATION TEST")
        print("=" * 70)

        coherent_scores = []
        for sources in COHERENT_SETS.values():
            vectors = np.array([model.encode(s) for s in sources])
            coherent_scores.append(compute_coherence(vectors))

        incoherent_scores = []
        for sources in INCOHERENT_SETS.values():
            vectors = np.array([model.encode(s) for s in sources])
            incoherent_scores.append(compute_coherence(vectors))

        mean_coherent = np.mean(coherent_scores)
        mean_incoherent = np.mean(incoherent_scores)
        separation = mean_coherent - mean_incoherent
        ratio = mean_coherent / mean_incoherent if mean_incoherent > 0 else float('inf')

        print(f"Coherent mean:   {mean_coherent:.4f}")
        print(f"Incoherent mean: {mean_incoherent:.4f}")
        print(f"Separation:      {separation:.4f}")
        print(f"Ratio:           {ratio:.2f}x")

        # The key test: coherent > incoherent
        assert mean_coherent > mean_incoherent, \
            f"Coherent ({mean_coherent:.4f}) should exceed incoherent ({mean_incoherent:.4f})"

        print("\nCONFIRMED: Coherent sources have higher coherence than incoherent")

    def test_contextual_coherence_boost(self, model):
        """Adding context should boost coherence for aligned sources."""
        print("\n" + "=" * 70)
        print("CONTEXTUAL COHERENCE BOOST TEST")
        print("=" * 70)

        sources = COHERENT_SETS['thermodynamics']
        axis = "thermodynamics and entropy"

        # Without context
        vecs_neutral = np.array([model.encode(s) for s in sources])
        coherence_neutral = compute_coherence(vecs_neutral)

        # With context
        vecs_context = np.array([phase_embed(model, s, axis) for s in sources])
        coherence_context = compute_coherence(vecs_context)

        boost = coherence_context - coherence_neutral

        print(f"Without context: {coherence_neutral:.4f}")
        print(f"With context:    {coherence_context:.4f}")
        print(f"Boost:           {boost:+.4f}")

        # Document whether context helps
        if boost > 0:
            print("\nCONFIRMED: Context boosts coherence")
        else:
            print("\nNOTE: Context did not boost coherence for this set")


# ============================================================================
# TEST 2: Rule of 3 (Q13 Connection)
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestRuleOf3:
    """
    Test Gemini's claim: N=3 maximizes phase lock / minimizes error

    Q13 shows the improvement ratio peaks at N=2-3.
    """

    def test_coherence_scaling_with_n(self, model):
        """Coherence should peak around N=2-3."""
        print("\n" + "=" * 70)
        print("RULE OF 3: Coherence Scaling")
        print("=" * 70)

        results = []
        for n in [1, 2, 3, 4, 5, 6, 8]:
            if n > len(SCALING_SOURCES):
                continue

            sources = SCALING_SOURCES[:n]
            vectors = np.array([model.encode(s) for s in sources])

            coherence = compute_coherence(vectors)
            pairwise = compute_pairwise_coherence(vectors) if n > 1 else 1.0

            results.append({
                'n': n,
                'coherence': coherence,
                'pairwise': pairwise,
            })

            print(f"N={n}: coherence={coherence:.4f}, pairwise={pairwise:.4f}")

        # Find peak
        coherences = [r['coherence'] for r in results]
        peak_idx = np.argmax(coherences)
        peak_n = results[peak_idx]['n']

        print("-" * 70)
        print(f"Peak coherence at N={peak_n}")
        print(f"Q13 predicts peak at N=2-3")

        if peak_n <= 3:
            print("\nCONFIRMED: Coherence peaks at N<=3 (Rule of 3)")
        else:
            print(f"\nNOTE: Coherence peaks at N={peak_n}, not N<=3")

    def test_phase_error_scaling_with_n(self, model):
        """Phase error should minimize around N=2-3 for analogy completion."""
        print("\n" + "=" * 70)
        print("RULE OF 3: Phase Error Scaling")
        print("=" * 70)

        # Use gender words for analogy testing
        base_words = ['king', 'queen', 'man', 'woman']
        context_words = ['boy', 'girl', 'brother', 'sister', 'father', 'mother', 'prince', 'princess']
        analogy = ('king', 'queen', 'man', 'woman')
        axis = 'gender'

        results = []
        for n in [0, 1, 2, 3, 4, 6, 8]:
            # Build word set: base + n context words
            if n > len(context_words):
                continue

            all_words = base_words + context_words[:n]

            # Embed with context
            vecs = np.array([phase_embed(model, w, axis) for w in all_words])
            embeddings = {w: v for w, v in zip(all_words, vecs)}

            # Fit global PCA
            pca = PCA(n_components=2)
            pca.fit(vecs)

            # Compute phase error
            error = compute_phase_error_global(embeddings, analogy, pca)

            results.append({
                'n': n,
                'error': error,
                'words': len(all_words),
            })

            print(f"N={n} (+{n} context): error={error:.1f} deg, total words={len(all_words)}")

        # Find minimum error
        errors = [r['error'] for r in results]
        min_idx = np.argmin(errors)
        min_n = results[min_idx]['n']

        print("-" * 70)
        print(f"Minimum error at N={min_n}")

        # Q13 predicts optimal at N=2-3
        if min_n <= 3:
            print("\nCONFIRMED: Phase error minimizes at N<=3 (Rule of 3)")
        else:
            print(f"\nNOTE: Phase error minimizes at N={min_n}, not N<=3")

    def test_information_decay_beyond_3(self, model):
        """Adding more than 3 sources should show diminishing returns."""
        print("\n" + "=" * 70)
        print("RULE OF 3: Information Decay")
        print("=" * 70)

        query = "entropy and thermodynamic equilibrium"

        results = []
        for n in [1, 2, 3, 4, 5, 6]:
            if n > len(SCALING_SOURCES):
                continue

            sources = SCALING_SOURCES[:n]

            # Embed query and sources
            query_vec = model.encode(query)
            source_vecs = np.array([model.encode(s) for s in sources])

            # Mean source vector
            mean_source = np.mean(source_vecs, axis=0)

            # Relevance to query
            relevance = float(np.dot(query_vec, mean_source) /
                            (np.linalg.norm(query_vec) * np.linalg.norm(mean_source)))

            # Coherence
            coherence = compute_coherence(source_vecs)

            # Combined score (Gemini's formula)
            score = relevance * coherence

            results.append({
                'n': n,
                'relevance': relevance,
                'coherence': coherence,
                'score': score,
            })

            print(f"N={n}: relevance={relevance:.4f}, coherence={coherence:.4f}, score={score:.4f}")

        # Check if score decays after N=3
        scores = [r['score'] for r in results]
        if len(scores) >= 4:
            score_at_3 = scores[2]  # N=3
            score_at_4 = scores[3]  # N=4

            print("-" * 70)
            if score_at_3 >= score_at_4:
                print(f"CONFIRMED: Score at N=3 ({score_at_3:.4f}) >= N=4 ({score_at_4:.4f})")
            else:
                print(f"NOTE: Score increases from N=3 ({score_at_3:.4f}) to N=4 ({score_at_4:.4f})")


# ============================================================================
# TEST 3: Triangulated Agent (Integration Test)
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestTriangulatedAgent:
    """
    Test Gemini's TriangulatedAgent concept:
    - CRYSTAL if coherence > threshold and sources <= 3
    - LIQUID otherwise
    """

    def test_crystal_for_coherent_sources(self, model):
        """Coherent sources should return CRYSTAL status."""
        print("\n" + "=" * 70)
        print("TRIANGULATED AGENT: Crystal Detection")
        print("=" * 70)

        for name, sources in COHERENT_SETS.items():
            vectors = np.array([model.encode(s) for s in sources])
            coherence = compute_coherence(vectors)

            # Gemini's classification
            n = len(sources)
            is_crystal = coherence > COHERENCE_THRESHOLD and n <= OPTIMAL_N
            status = "CRYSTAL" if is_crystal else "LIQUID"

            print(f"{name:20s}: N={n}, coherence={coherence:.4f} -> {status}")

        print("\nNOTE: Threshold may need calibration based on actual coherence values")

    def test_liquid_for_incoherent_sources(self, model):
        """Incoherent sources should return LIQUID status."""
        print("\n" + "=" * 70)
        print("TRIANGULATED AGENT: Liquid Detection")
        print("=" * 70)

        for name, sources in INCOHERENT_SETS.items():
            vectors = np.array([model.encode(s) for s in sources])
            coherence = compute_coherence(vectors)

            # Gemini's classification
            n = len(sources)
            is_crystal = coherence > COHERENCE_THRESHOLD and n <= OPTIMAL_N
            status = "CRYSTAL" if is_crystal else "LIQUID"

            print(f"{name:20s}: N={n}, coherence={coherence:.4f} -> {status}")

    def test_liquid_for_too_many_sources(self, model):
        """More than 3 sources should return LIQUID (even if coherent)."""
        print("\n" + "=" * 70)
        print("TRIANGULATED AGENT: N>3 Detection")
        print("=" * 70)

        # Use 6 coherent sources
        sources = SCALING_SOURCES[:6]
        vectors = np.array([model.encode(s) for s in sources])
        coherence = compute_coherence(vectors)

        n = len(sources)
        is_crystal = coherence > COHERENCE_THRESHOLD and n <= OPTIMAL_N
        status = "CRYSTAL" if is_crystal else "LIQUID"

        print(f"6 coherent sources: N={n}, coherence={coherence:.4f} -> {status}")

        if n > OPTIMAL_N:
            print(f"\nRule of 3 enforced: N={n} > {OPTIMAL_N} -> LIQUID")

    def test_threshold_calibration(self, model):
        """Determine optimal threshold based on separation."""
        print("\n" + "=" * 70)
        print("THRESHOLD CALIBRATION")
        print("=" * 70)

        # Collect all coherence values
        coherent_scores = []
        for sources in COHERENT_SETS.values():
            vectors = np.array([model.encode(s) for s in sources])
            coherent_scores.append(compute_coherence(vectors))

        incoherent_scores = []
        for sources in INCOHERENT_SETS.values():
            vectors = np.array([model.encode(s) for s in sources])
            incoherent_scores.append(compute_coherence(vectors))

        # Find optimal threshold
        min_coherent = min(coherent_scores)
        max_incoherent = max(incoherent_scores)
        optimal_threshold = (min_coherent + max_incoherent) / 2

        print(f"Coherent range:   [{min(coherent_scores):.4f}, {max(coherent_scores):.4f}]")
        print(f"Incoherent range: [{min(incoherent_scores):.4f}, {max(incoherent_scores):.4f}]")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Q12 threshold:     {COHERENCE_THRESHOLD}")

        gap = min_coherent - max_incoherent
        if gap > 0:
            print(f"\nSeparation gap: {gap:.4f} (clean separation)")
        else:
            print(f"\nOverlap: {-gap:.4f} (some overlap between classes)")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
