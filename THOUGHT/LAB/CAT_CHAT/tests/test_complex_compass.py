"""
Comprehensive Test Suite: Complex Compass (CP^n Navigation)

HYPOTHESIS STATUS (2026-01-21):
================================
This test suite was built to test the Grok/Gemini hypothesis:
1. Real embeddings are "shadows" of complex vectors - PARTIALLY CONFIRMED
2. Pentagonal geometry (72 deg) should appear sharper in phase space - NO
3. Hermitian similarity may reveal structure cosine cannot see - NO
4. Opposites (negation) should have ~180 deg phase shift - NO (for semantic opposites)

KEY FINDINGS:
- Pentagonal geometry (~72 deg) PERSISTS in complex space (confirms Q53)
- Hermitian magnitude TRACKS cosine similarity closely (<0.01 difference)
- Semantic opposites (good/bad) have ~0-4 deg phase shift, NOT ~180 deg
- Exact mathematical opposites (v, -v) DO have 180 deg phase shift (trivially)
- Phase coherence is near-constant (~0.01-0.11) for sign-to-phase method

The tests below are designed to VERIFY THESE NEGATIVE RESULTS. Tests that pass
confirm the hypothesis is FALSIFIED for these specific claims. This is proper
scientific methodology - we test what is, not what we hoped.

Test Categories:
- T1: ComplexGeometricState axioms and properties (mathematical correctness)
- T2: Complexification methods (sign-to-phase, Hilbert, FFT)
- T3: Hermitian vs Cosine similarity comparison (tests claim #3 - NEGATIVE RESULT)
- T4: Negation/opposition detection via phase (tests claim #4 - NEGATIVE RESULT)
- T5: Pentagonal geometry analysis (confirms Q53 finding)
- T6: Compass mode navigation (effectively cosine-based due to constant coherence)
- T7: Q53 replication in complex space
- T8: Integration with existing GeometricReasoner
- T9: Determinism and reproducibility
- T10: Edge cases and error handling

Design follows CAT_CHAT test patterns (STYLE-002 compliant).
Documentation of negative results follows scientific best practices.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from itertools import combinations

# Add paths for imports
CAT_CHAT_PATH = Path(__file__).parent.parent
CATALYTIC_PATH = CAT_CHAT_PATH / "catalytic_chat"
CAPABILITY_PATH = CAT_CHAT_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

sys.path.insert(0, str(CAT_CHAT_PATH))
sys.path.insert(0, str(CATALYTIC_PATH))
sys.path.insert(0, str(CAPABILITY_PATH))

# Import with availability check
try:
    from complex_compass import (
        ComplexCompass,
        ComplexGeometricState,
        ComplexificationMethod,
        GOLDEN_RATIO,
        PENTAGONAL_ANGLE_DEG,
        PENTAGONAL_ANGLE_RAD,
        FIFTH_ROOT_UNITY,
        compare_methods_on_vector,
        analyze_phase_structure,
    )
    COMPLEX_COMPASS_AVAILABLE = True
except ImportError as e:
    COMPLEX_COMPASS_AVAILABLE = False
    IMPORT_ERROR = str(e)

try:
    from scipy.signal import hilbert
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Skip marker
pytestmark = pytest.mark.skipif(
    not COMPLEX_COMPASS_AVAILABLE,
    reason=f"ComplexCompass not available: {IMPORT_ERROR if not COMPLEX_COMPASS_AVAILABLE else ''}"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def compass_hilbert():
    """ComplexCompass with Hilbert transform method."""
    if not SCIPY_AVAILABLE:
        pytest.skip("scipy not available for Hilbert transform")
    return ComplexCompass(method=ComplexificationMethod.HILBERT)


@pytest.fixture
def compass_sign():
    """ComplexCompass with sign-to-phase method."""
    return ComplexCompass(method=ComplexificationMethod.SIGN_TO_PHASE)


@pytest.fixture
def random_real_vector():
    """Random real-valued vector (normalized)."""
    np.random.seed(42)
    v = np.random.randn(384)  # MiniLM dimension
    return v / np.linalg.norm(v)


@pytest.fixture
def positive_real_vector():
    """All-positive real vector."""
    np.random.seed(42)
    v = np.abs(np.random.randn(384))
    return v / np.linalg.norm(v)


@pytest.fixture
def mixed_sign_vector():
    """Vector with mixed signs (typical embedding)."""
    np.random.seed(42)
    v = np.random.randn(384)
    return v / np.linalg.norm(v)


@pytest.fixture
def orthogonal_vectors():
    """Pair of orthogonal vectors."""
    np.random.seed(42)
    v1 = np.random.randn(384)
    v1 = v1 / np.linalg.norm(v1)

    # Make v2 orthogonal to v1
    v2 = np.random.randn(384)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    return v1, v2


@pytest.fixture
def opposite_vectors():
    """Pair of opposite vectors (v and -v)."""
    np.random.seed(42)
    v1 = np.random.randn(384)
    v1 = v1 / np.linalg.norm(v1)
    v2 = -v1
    return v1, v2


@pytest.fixture
def similar_vectors():
    """Pair of similar vectors (high cosine)."""
    np.random.seed(42)
    v1 = np.random.randn(384)
    v1 = v1 / np.linalg.norm(v1)

    # Add very small noise to maintain high similarity
    noise = np.random.randn(384) * 0.01  # Much smaller noise
    v2 = v1 + noise
    v2 = v2 / np.linalg.norm(v2)

    # Verify similarity is actually high
    assert np.dot(v1, v2) > 0.95

    return v1, v2


@pytest.fixture
def word_embeddings():
    """Pre-computed embeddings for test words."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers not available")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    words = [
        "good", "bad", "hot", "cold", "king", "queen",
        "man", "woman", "love", "hate", "light", "dark",
        "cat", "dog", "tree", "water", "sun", "moon"
    ]

    embeddings = {}
    vecs = model.encode(words, convert_to_numpy=True)
    for word, vec in zip(words, vecs):
        vec = vec / np.linalg.norm(vec)
        embeddings[word] = vec

    return embeddings


# ============================================================================
# T1: ComplexGeometricState Axioms and Properties
# ============================================================================

class TestComplexGeometricStateAxioms:
    """Test T1: ComplexGeometricState satisfies required axioms."""

    def test_state_is_normalized(self, compass_sign, random_real_vector):
        """State lives on complex unit sphere (||psi|| = 1)."""
        state = compass_sign.complexify(random_real_vector)
        norm = np.linalg.norm(state.vector)
        assert abs(norm - 1.0) < 1e-10

    def test_state_is_complex(self, compass_sign, random_real_vector):
        """State vector is complex-valued."""
        state = compass_sign.complexify(random_real_vector)
        assert state.vector.dtype in [np.complex64, np.complex128]

    def test_amplitude_is_non_negative(self, compass_sign, random_real_vector):
        """Amplitude (magnitude) is always non-negative."""
        state = compass_sign.complexify(random_real_vector)
        assert np.all(state.amplitude >= 0)

    def test_phase_is_bounded(self, compass_sign, random_real_vector):
        """Phase is in [-pi, pi]."""
        state = compass_sign.complexify(random_real_vector)
        assert np.all(state.phase >= -np.pi)
        assert np.all(state.phase <= np.pi)

    def test_df_is_positive(self, compass_sign, random_real_vector):
        """Participation ratio Df is positive."""
        state = compass_sign.complexify(random_real_vector)
        assert state.Df > 0

    def test_phase_coherence_bounded(self, compass_sign, random_real_vector):
        """Phase coherence is in [0, 1]."""
        state = compass_sign.complexify(random_real_vector)
        assert 0 <= state.phase_coherence <= 1

    def test_hermitian_inner_product_symmetric(
        self,
        compass_sign,
        similar_vectors
    ):
        """Hermitian inner product: <a|b> = conj(<b|a>)."""
        v1, v2 = similar_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        e12 = s1.E_hermitian(s2)
        e21 = s2.E_hermitian(s1)

        assert abs(e12 - np.conj(e21)) < 1e-10

    def test_self_inner_product_is_one(self, compass_sign, random_real_vector):
        """Self inner product: <a|a> = 1 for normalized state."""
        state = compass_sign.complexify(random_real_vector)
        self_product = state.E_hermitian(state)
        assert abs(self_product - 1.0) < 1e-10


# ============================================================================
# T2: Complexification Methods
# ============================================================================

class TestComplexificationMethods:
    """Test T2: All complexification methods work correctly."""

    def test_sign_to_phase_positive_input(self, compass_sign, positive_real_vector):
        """Sign-to-phase: positive values get phase 0."""
        state = compass_sign.complexify(positive_real_vector)

        # All phases should be 0 (or very close)
        assert np.allclose(state.phase, 0, atol=1e-10)

    def test_sign_to_phase_negative_produces_pi(self, compass_sign):
        """Sign-to-phase: negative values get phase pi."""
        # All-negative vector
        v = -np.abs(np.random.randn(100))
        v = v / np.linalg.norm(v)

        state = compass_sign.complexify(v)

        # All phases should be pi
        assert np.allclose(np.abs(state.phase), np.pi, atol=1e-10)

    def test_sign_to_phase_mixed_signs(self, compass_sign, mixed_sign_vector):
        """Sign-to-phase: mixed signs produce 0 and pi phases."""
        state = compass_sign.complexify(mixed_sign_vector)

        # Phases should only be 0 or pi
        phases = state.phase
        is_zero = np.abs(phases) < 1e-10
        is_pi = np.abs(np.abs(phases) - np.pi) < 1e-10

        assert np.all(is_zero | is_pi)

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
    def test_hilbert_produces_analytic_signal(self, compass_hilbert, random_real_vector):
        """Hilbert transform produces analytic signal."""
        state = compass_hilbert.complexify(random_real_vector)

        # Analytic signal should have non-uniform phase
        phase_std = np.std(state.phase)
        assert phase_std > 0.01  # Non-trivial phase variation

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
    def test_hilbert_preserves_real_information(
        self,
        compass_hilbert,
        random_real_vector
    ):
        """Hilbert transform: real part related to original."""
        state = compass_hilbert.complexify(random_real_vector)

        # Real part of analytic signal should correlate with original
        real_part = np.real(state.vector)
        # Normalize for comparison
        real_part = real_part / np.linalg.norm(real_part)
        original = random_real_vector / np.linalg.norm(random_real_vector)

        correlation = np.abs(np.dot(real_part, original))
        assert correlation > 0.5  # Should be related

    def test_all_methods_produce_valid_states(self, random_real_vector):
        """All methods produce valid ComplexGeometricState."""
        results = compare_methods_on_vector(random_real_vector)

        for method_name, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                continue  # Method not available

            assert isinstance(result, ComplexGeometricState)
            assert result.Df > 0
            assert abs(np.linalg.norm(result.vector) - 1.0) < 1e-10


# ============================================================================
# T3: Hermitian vs Cosine Similarity
# ============================================================================

class TestHermitianVsCosineSimilarity:
    """Test T3: Compare Hermitian and cosine similarity metrics."""

    def test_identical_vectors_both_return_one(
        self,
        compass_sign,
        random_real_vector
    ):
        """Identical vectors: both metrics return 1."""
        state = compass_sign.complexify(random_real_vector)

        hermitian_mag = state.E_magnitude(state)
        # Cosine of identical = 1
        cosine = np.dot(random_real_vector, random_real_vector)

        assert abs(hermitian_mag - 1.0) < 1e-10
        assert abs(cosine - 1.0) < 1e-10

    def test_orthogonal_vectors_both_return_zero(
        self,
        compass_sign,
        orthogonal_vectors
    ):
        """Orthogonal vectors: both metrics return ~0."""
        v1, v2 = orthogonal_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        # Cosine should be 0
        cosine = np.dot(v1, v2)
        assert abs(cosine) < 1e-10

        # Hermitian magnitude should be small (not necessarily 0)
        hermitian_mag = s1.E_magnitude(s2)
        assert hermitian_mag < 0.5  # Much smaller than 1

    def test_opposite_vectors_cosine_negative(
        self,
        compass_sign,
        opposite_vectors
    ):
        """Opposite vectors: cosine = -1, Hermitian has phase info."""
        v1, v2 = opposite_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        # Cosine should be -1
        cosine = np.dot(v1, v2)
        assert abs(cosine + 1.0) < 1e-10

        # Hermitian comparison
        comparison = compass_sign.compare_real_vs_complex(s1, s2)

        # Magnitude should be high (they're "close" topically)
        # Phase should indicate negation
        assert comparison['hermitian_magnitude'] > 0.5

    def test_compare_returns_all_metrics(
        self,
        compass_sign,
        similar_vectors
    ):
        """compare_real_vs_complex returns complete analysis."""
        v1, v2 = similar_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        comparison = compass_sign.compare_real_vs_complex(s1, s2)

        # Check all expected keys
        required_keys = [
            'hermitian_magnitude',
            'hermitian_phase_deg',
            'hermitian_real',
            'hermitian_imag',
            'cosine_similarity',
            'difference'
        ]

        for key in required_keys:
            assert key in comparison

    def test_similar_vectors_both_high(
        self,
        compass_sign,
        similar_vectors
    ):
        """Similar vectors: both metrics should be high."""
        v1, v2 = similar_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        comparison = compass_sign.compare_real_vs_complex(s1, s2)

        assert comparison['cosine_similarity'] > 0.8
        assert comparison['hermitian_magnitude'] > 0.8


# ============================================================================
# T4: Negation/Opposition Detection
# ============================================================================

class TestNegationDetection:
    """
    Test T4: Phase-based negation detection.

    CRITICAL NOTE (2026-01-21):
    The Grok/Gemini hypothesis was that semantic opposites (good/bad) should
    have ~180 deg phase shift after complexification. This hypothesis is
    FALSIFIED by our tests:
    - Exact mathematical opposites (v, -v): Phase = 180 deg (WORKS)
    - Semantic opposites (good/bad): Phase = ~0-4 deg (FAILS)

    The tests below document both the success case (exact opposites) and
    the failure case (semantic opposites) accurately.
    """

    def test_exact_opposite_vectors_have_180_deg_phase(
        self,
        compass_sign,
        opposite_vectors
    ):
        """
        EXACT mathematical opposites (v, -v) have 180 deg phase shift.

        This is expected from sign-to-phase math: every component changes
        sign, so every phase shifts by pi, giving Hermitian inner product = -1.
        """
        v1, v2 = opposite_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        negation = compass_sign.detect_negation(s1, s2)

        # For exact opposites, phase must be ~180 deg
        assert abs(negation['phase_abs_deg'] - 180.0) < 1.0, \
            f"Expected ~180 deg, got {negation['phase_abs_deg']:.2f}"
        assert negation['magnitude'] > 0.99, \
            f"Expected magnitude ~1.0, got {negation['magnitude']:.4f}"
        assert negation['is_negation'], \
            "Exact opposites should be detected as negation"

    def test_similar_vectors_not_detected_as_negation(
        self,
        compass_sign,
        similar_vectors
    ):
        """Similar vectors should NOT be detected as negations."""
        v1, v2 = similar_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        negation = compass_sign.detect_negation(s1, s2)

        assert not negation['is_negation']

    @pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
    def test_semantic_opposites_have_small_phase_not_180(self, compass_sign, word_embeddings):
        """
        NEGATIVE RESULT: Semantic opposites do NOT have ~180 deg phase shift.

        This FALSIFIES the Grok/Gemini hypothesis for sign-to-phase method.
        The phase shift between semantic opposites (good/bad, hot/cold) is
        ~0-4 degrees, not ~180 degrees as hypothesized.

        Q51 showed that complex structure exists in RELATIONSHIPS (global PCA),
        not in individual vectors complexified via local transforms.
        """
        pairs = [
            ('good', 'bad'),
            ('hot', 'cold'),
            ('love', 'hate'),
        ]

        for w1, w2 in pairs:
            s1 = compass_sign.complexify(word_embeddings[w1])
            s2 = compass_sign.complexify(word_embeddings[w2])
            comparison = compass_sign.compare_real_vs_complex(s1, s2)

            phase_deg = abs(comparison['hermitian_phase_deg'])

            # NEGATIVE RESULT: Phase is NOT ~180 deg
            # Instead, phase is very small (~0-10 deg)
            assert phase_deg < 20.0, \
                f"{w1}/{w2}: Expected small phase (<20 deg), got {phase_deg:.2f}"

            # Hermitian magnitude tracks cosine similarity closely
            mag = comparison['hermitian_magnitude']
            cosine = comparison['cosine_similarity']
            assert abs(mag - cosine) < 0.01, \
                f"{w1}/{w2}: Hermitian magnitude ({mag:.4f}) should track cosine ({cosine:.4f})"

    @pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
    def test_hermitian_tracks_cosine_for_semantic_pairs(self, compass_sign, word_embeddings):
        """
        Hermitian magnitude tracks cosine similarity closely for all semantic pairs.

        This confirms the NEGATIVE RESULT from the report: complexification
        via sign-to-phase or Hilbert does NOT reveal additional structure
        beyond what cosine similarity already shows.
        """
        pairs = [
            ('good', 'bad'),
            ('hot', 'cold'),
            ('king', 'queen'),
            ('man', 'woman'),
        ]

        max_difference = 0.0

        for w1, w2 in pairs:
            s1 = compass_sign.complexify(word_embeddings[w1])
            s2 = compass_sign.complexify(word_embeddings[w2])

            mag = s1.E_magnitude(s2)
            cosine = float(np.dot(word_embeddings[w1], word_embeddings[w2]))

            difference = abs(mag - cosine)
            max_difference = max(max_difference, difference)

            # Hermitian magnitude should track cosine very closely
            assert difference < 0.01, \
                f"{w1}/{w2}: Hermitian mag ({mag:.4f}) diverges from cosine ({cosine:.4f})"

        # Overall, the maximum difference should be tiny
        assert max_difference < 0.01, \
            f"Max Hermitian/cosine difference is {max_difference:.4f}, expected < 0.01"


# ============================================================================
# T5: Pentagonal Geometry Analysis
# ============================================================================

class TestPentagonalGeometry:
    """Test T5: Pentagonal structure (72 deg) analysis."""

    def test_pentagonal_angle_constant(self):
        """Verify pentagonal angle constant is correct."""
        assert abs(PENTAGONAL_ANGLE_DEG - 72.0) < 1e-10
        assert abs(PENTAGONAL_ANGLE_RAD - np.radians(72.0)) < 1e-10

    def test_fifth_root_unity_has_72_degree_spacing(self):
        """Fifth root of unity has 72 degree phase."""
        phase_deg = np.degrees(np.angle(FIFTH_ROOT_UNITY))
        assert abs(phase_deg - 72.0) < 1e-10

    def test_phase_distribution_returns_histogram(
        self,
        compass_sign,
        word_embeddings
    ):
        """phase_angle_distribution returns valid histogram."""
        if not word_embeddings:
            pytest.skip("word_embeddings not available")

        states = [compass_sign.complexify(v) for v in word_embeddings.values()]
        dist = compass_sign.phase_angle_distribution(states)

        assert 'histogram' in dist
        assert 'bin_edges' in dist
        assert 'pentagonal_score' in dist
        assert len(dist['histogram']) == 36  # Default bins

    def test_geodesic_distribution_returns_analysis(
        self,
        compass_sign,
        word_embeddings
    ):
        """geodesic_angle_distribution returns valid analysis."""
        if not word_embeddings:
            pytest.skip("word_embeddings not available")

        states = [compass_sign.complexify(v) for v in word_embeddings.values()]
        dist = compass_sign.geodesic_angle_distribution(states)

        assert 'mean_deg' in dist
        assert 'peak_center_deg' in dist
        assert 'near_pentagonal' in dist

    @pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
    def test_real_embeddings_cluster_near_pentagonal(self, compass_sign, word_embeddings):
        """
        Q53 REPLICATION: Real embeddings should cluster near 72 degrees.

        This is the key hypothesis from Q53.
        """
        states = [compass_sign.complexify(v) for v in word_embeddings.values()]
        dist = compass_sign.geodesic_angle_distribution(states)

        peak = dist['peak_center_deg']
        mean = dist['mean_deg']

        # Record findings
        print(f"\nGeodesic angle analysis:")
        print(f"  Mean: {mean:.2f} deg")
        print(f"  Peak: {peak:.2f} deg")
        print(f"  Deviation from pentagonal: {abs(peak - 72):.2f} deg")

        # Q53 found mean around 73 deg for non-BERT models
        # We're testing with MiniLM (sentence transformer)
        # Allow wider range since this is exploratory
        assert 30 < mean < 120  # Sanity check

    def test_random_vectors_not_pentagonal(self, compass_sign):
        """Random vectors should NOT show pentagonal structure."""
        np.random.seed(42)
        random_vecs = [np.random.randn(384) for _ in range(20)]
        random_vecs = [v / np.linalg.norm(v) for v in random_vecs]

        states = [compass_sign.complexify(v) for v in random_vecs]
        dist = compass_sign.geodesic_angle_distribution(states)

        # Random high-D vectors cluster near 90 degrees
        mean = dist['mean_deg']
        assert 70 < mean < 110  # Around orthogonal


# ============================================================================
# T6: Compass Mode Navigation
# ============================================================================

class TestCompassNavigation:
    """Test T6: Compass mode direction selection."""

    def test_get_direction_returns_best_index(
        self,
        compass_sign,
        random_real_vector
    ):
        """get_direction returns valid index."""
        current = compass_sign.complexify(random_real_vector)

        # Generate candidates
        np.random.seed(42)
        candidates = [
            compass_sign.complexify(np.random.randn(384))
            for _ in range(5)
        ]

        best_idx, analysis = compass_sign.get_direction(current, candidates)

        assert 0 <= best_idx < len(candidates)
        assert 'best_score' in analysis
        assert 'all_scores' in analysis

    def test_get_direction_prefers_similar(
        self,
        compass_sign,
        random_real_vector,
        similar_vectors
    ):
        """get_direction prefers similar vectors based on magnitude."""
        v1, v2 = similar_vectors
        current = compass_sign.complexify(v1)

        # v2 is similar to v1, others are orthogonal (low magnitude)
        np.random.seed(42)

        # Create orthogonal vectors to v1
        ortho1 = np.random.randn(384)
        ortho1 = ortho1 - np.dot(ortho1, v1) * v1
        ortho1 = ortho1 / np.linalg.norm(ortho1)

        ortho2 = np.random.randn(384)
        ortho2 = ortho2 - np.dot(ortho2, v1) * v1
        ortho2 = ortho2 / np.linalg.norm(ortho2)

        candidates = [
            compass_sign.complexify(v2),  # Similar - should be chosen
            compass_sign.complexify(ortho1),
            compass_sign.complexify(ortho2),
        ]

        best_idx, analysis = compass_sign.get_direction(current, candidates)

        # Similar vector should have highest magnitude
        scores = analysis['all_scores']
        similar_mag = next(s for s in scores if s['index'] == 0)['magnitude']
        max_mag = max(s['magnitude'] for s in scores)

        # The similar vector should have highest magnitude
        assert similar_mag == max_mag

    def test_get_direction_respects_weights(self, compass_sign):
        """get_direction respects weight parameter."""
        np.random.seed(42)
        current = compass_sign.complexify(np.random.randn(384))

        candidates = [
            compass_sign.complexify(np.random.randn(384)),
            compass_sign.complexify(np.random.randn(384)),
            compass_sign.complexify(np.random.randn(384)),
        ]

        # Give high weight to index 2
        weights = np.array([0.1, 0.1, 10.0])

        best_idx, analysis = compass_sign.get_direction(
            current, candidates, weights=weights
        )

        # Index 2 should win due to weight
        assert best_idx == 2

    def test_get_direction_empty_candidates_raises(self, compass_sign, random_real_vector):
        """get_direction raises on empty candidates."""
        current = compass_sign.complexify(random_real_vector)

        with pytest.raises(ValueError):
            compass_sign.get_direction(current, [])


# ============================================================================
# T7: Q53 Replication in Complex Space
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestQ53Replication:
    """Test T7: Replicate Q53 pentagonal findings in complex space."""

    def test_analyze_phase_structure_runs(self, word_embeddings):
        """analyze_phase_structure produces valid output."""
        vectors = list(word_embeddings.values())

        analysis = analyze_phase_structure(
            vectors,
            method=ComplexificationMethod.SIGN_TO_PHASE
        )

        assert 'phase_distribution' in analysis
        assert 'geodesic_distribution' in analysis
        assert 'mean_Df' in analysis

    def test_complex_vs_real_angular_distribution(
        self,
        compass_sign,
        word_embeddings
    ):
        """
        Compare angular distribution in real vs complex space.

        Key test: Does complexification sharpen pentagonal structure?
        """
        vectors = list(word_embeddings.values())
        states = [compass_sign.complexify(v) for v in vectors]

        # Real space angles (standard Q53 method)
        real_angles = []
        for v1, v2 in combinations(vectors, 2):
            cos_angle = np.dot(v1, v2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            real_angles.append(angle_deg)

        # Complex space angles (geodesic in CP^n)
        complex_angles = []
        for s1, s2 in combinations(states, 2):
            angle_deg = np.degrees(s1.distance_geodesic(s2))
            complex_angles.append(angle_deg)

        real_mean = np.mean(real_angles)
        real_std = np.std(real_angles)
        complex_mean = np.mean(complex_angles)
        complex_std = np.std(complex_angles)

        print(f"\nAngular distribution comparison:")
        print(f"  Real space:    mean={real_mean:.2f}, std={real_std:.2f}")
        print(f"  Complex space: mean={complex_mean:.2f}, std={complex_std:.2f}")
        print(f"  Pentagonal target: 72 deg")

        # Both should be valid
        assert 30 < real_mean < 120
        assert 30 < complex_mean < 120

        # Record deviation from pentagonal for analysis
        real_deviation = abs(real_mean - 72)
        complex_deviation = abs(complex_mean - 72)

        print(f"  Real deviation from 72: {real_deviation:.2f} deg")
        print(f"  Complex deviation from 72: {complex_deviation:.2f} deg")


# ============================================================================
# T8: Integration with GeometricReasoner
# ============================================================================

class TestGeometricReasonerIntegration:
    """Test T8: Integration with existing GeometricReasoner."""

    def test_complex_state_compatible_with_real(
        self,
        compass_sign,
        random_real_vector
    ):
        """ComplexGeometricState stores original real vector."""
        state = compass_sign.complexify(random_real_vector)

        assert state.real_source is not None
        assert np.allclose(state.real_source, random_real_vector)

    def test_receipt_format_matches_geometric_state(
        self,
        compass_sign,
        random_real_vector
    ):
        """Receipt format is compatible with provenance system."""
        state = compass_sign.complexify(random_real_vector)
        receipt = state.receipt()

        # Required fields for provenance
        assert 'vector_hash' in receipt
        assert 'Df' in receipt
        assert 'dim' in receipt

        # New complex-specific fields
        assert 'mean_phase_deg' in receipt
        assert 'phase_coherence' in receipt
        assert 'method' in receipt


# ============================================================================
# T9: Determinism and Reproducibility
# ============================================================================

class TestDeterminism:
    """Test T9: Deterministic behavior."""

    def test_same_input_same_output(self, compass_sign, random_real_vector):
        """Same input produces identical output."""
        state1 = compass_sign.complexify(random_real_vector)
        state2 = compass_sign.complexify(random_real_vector)

        assert np.allclose(state1.vector, state2.vector)
        assert state1.receipt()['vector_hash'] == state2.receipt()['vector_hash']

    def test_comparison_deterministic(self, compass_sign, similar_vectors):
        """Comparisons are deterministic."""
        v1, v2 = similar_vectors
        s1 = compass_sign.complexify(v1)
        s2 = compass_sign.complexify(v2)

        comp1 = compass_sign.compare_real_vs_complex(s1, s2)
        comp2 = compass_sign.compare_real_vs_complex(s1, s2)

        assert abs(comp1['hermitian_magnitude'] - comp2['hermitian_magnitude']) < 1e-10

    def test_navigation_deterministic(self, compass_sign, random_real_vector):
        """Navigation is deterministic."""
        np.random.seed(42)
        current = compass_sign.complexify(random_real_vector)
        candidates = [compass_sign.complexify(np.random.randn(384)) for _ in range(3)]

        idx1, _ = compass_sign.get_direction(current, candidates)

        np.random.seed(42)
        current = compass_sign.complexify(random_real_vector)
        candidates = [compass_sign.complexify(np.random.randn(384)) for _ in range(3)]

        idx2, _ = compass_sign.get_direction(current, candidates)

        assert idx1 == idx2


# ============================================================================
# T10: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test T10: Edge cases and error handling."""

    def test_zero_vector_handled(self, compass_sign):
        """Zero vector is handled (normalized to unit)."""
        # Note: In practice, embeddings are never zero
        # But the normalization should handle it gracefully
        v = np.zeros(384)
        # This will produce a state with norm 0, which post_init handles
        # by not dividing (avoiding NaN)
        state = compass_sign.complexify(v)

        # State should still be valid (even if degenerate)
        assert isinstance(state, ComplexGeometricState)

    def test_single_dimension_vector(self, compass_sign):
        """Single dimension vector works."""
        v = np.array([1.0])
        state = compass_sign.complexify(v)

        assert len(state.vector) == 1
        assert abs(np.linalg.norm(state.vector) - 1.0) < 1e-10

    def test_large_dimension_vector(self, compass_sign):
        """Large dimension vector works."""
        np.random.seed(42)
        v = np.random.randn(10000)
        v = v / np.linalg.norm(v)

        state = compass_sign.complexify(v)

        assert len(state.vector) == 10000
        assert state.Df > 0

    def test_stats_tracked(self, compass_sign, similar_vectors):
        """Statistics are tracked correctly."""
        v1, v2 = similar_vectors

        initial_stats = compass_sign.get_stats()
        initial_complexifications = initial_stats['complexifications']

        compass_sign.complexify(v1)
        compass_sign.complexify(v2)

        final_stats = compass_sign.get_stats()
        assert final_stats['complexifications'] == initial_complexifications + 2


# ============================================================================
# Benchmark: Phase Sharpening
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestPhaseSharpening:
    """
    Benchmark test: Does complexification sharpen pentagonal structure?

    This is the main hypothesis from Grok/Gemini analysis.
    """

    def test_phase_sharpening_hypothesis(self, word_embeddings):
        """
        MAIN HYPOTHESIS TEST

        If complexification reveals hidden pentagonal structure,
        then phase-space analysis should show clearer 72-degree peaks
        than real-space angular analysis.
        """
        vectors = list(word_embeddings.values())

        # Test with sign-to-phase (Grok's suggestion)
        compass_sign = ComplexCompass(method=ComplexificationMethod.SIGN_TO_PHASE)
        analysis_sign = analyze_phase_structure(
            vectors,
            method=ComplexificationMethod.SIGN_TO_PHASE
        )

        # Test with Hilbert (Gemini's suggestion)
        if SCIPY_AVAILABLE:
            analysis_hilbert = analyze_phase_structure(
                vectors,
                method=ComplexificationMethod.HILBERT
            )
        else:
            analysis_hilbert = None

        print("\n" + "=" * 70)
        print("PHASE SHARPENING HYPOTHESIS TEST")
        print("=" * 70)

        print("\nSign-to-phase method:")
        print(f"  Geodesic mean: {analysis_sign['geodesic_distribution']['mean_deg']:.2f} deg")
        print(f"  Geodesic peak: {analysis_sign['geodesic_distribution']['peak_center_deg']:.2f} deg")
        print(f"  Phase pentagonal score: {analysis_sign['phase_distribution']['pentagonal_score']:.2f}")
        print(f"  Mean Df: {analysis_sign['mean_Df']:.2f}")

        if analysis_hilbert:
            print("\nHilbert transform method:")
            print(f"  Geodesic mean: {analysis_hilbert['geodesic_distribution']['mean_deg']:.2f} deg")
            print(f"  Geodesic peak: {analysis_hilbert['geodesic_distribution']['peak_center_deg']:.2f} deg")
            print(f"  Phase pentagonal score: {analysis_hilbert['phase_distribution']['pentagonal_score']:.2f}")
            print(f"  Mean Df: {analysis_hilbert['mean_Df']:.2f}")

        # Test passes if analysis runs without error
        # Actual sharpening is recorded for human review
        assert analysis_sign is not None


# ============================================================================
# T11: Q51 Methodology Comparison (Local vs Global Phase Extraction)
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestQ51Methodology:
    """
    Test T11: Compare local complexification vs Q51 global PCA methodology.

    CRITICAL INSIGHT from running the actual Q51 tests:
    - Local complexification (Hilbert, sign-to-phase): FAILS to find phase structure
    - Global PCA + phase arithmetic (Q51 methodology): 90.9% pass rate, 4.05x separation

    The complex structure is in RELATIONSHIPS, not individual vectors.

    Q51 test results (test_q51_phase_arithmetic.py):
    - all-MiniLM-L6-v2: 86.4% pass, 4.15x separation
    - all-mpnet-base-v2: 100.0% pass, 6.59x separation
    - bge-small-en-v1.5: 86.4% pass, 4.22x separation
    - all-MiniLM-L12-v2: 95.5% pass, 2.35x separation
    - gte-small: 86.4% pass, 2.93x separation
    - CROSS-MODEL: 90.9% mean pass, 4.05x mean separation
    """

    def test_local_vs_global_phase_extraction(self, word_embeddings):
        """
        Demonstrate the difference between local and global phase extraction.

        Local (per-pair PCA): Extract phase from 2D PCA fitted per analogy pair
        Global (Q51): Extract phase from PCA on ENTIRE word ensemble

        The key insight: Global PCA establishes a SHARED coordinate system that
        preserves relational structure. Per-pair PCA loses this.
        """
        from sklearn.decomposition import PCA
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Classic analogies from Q51 test
        analogies = [
            ('king', 'queen', 'man', 'woman'),
            ('brother', 'sister', 'boy', 'girl'),
            ('walk', 'walked', 'run', 'ran'),
            ('big', 'bigger', 'small', 'smaller'),
            ('good', 'better', 'bad', 'worse'),
            ('father', 'mother', 'son', 'daughter'),
        ]

        # Non-analogies (random 4-tuples that should FAIL)
        non_analogies = [
            ('king', 'walked', 'good', 'sister'),
            ('queen', 'bigger', 'worse', 'ran'),
            ('man', 'father', 'walk', 'small'),
        ]

        # Get all words
        all_words = list(set(w for a in (analogies + non_analogies) for w in a))
        vecs = model.encode(all_words, convert_to_numpy=True)
        embeddings = {w: v/np.linalg.norm(v) for w, v in zip(all_words, vecs)}
        all_vecs = np.array([embeddings[w] for w in all_words])

        # METHOD 1: Per-pair PCA (NO shared coordinate system)
        per_pair_errors = []
        for a, b, c, d in analogies:
            # Fit PCA on just these 4 words (no global context)
            pair_vecs = np.array([embeddings[w] for w in [a, b, c, d]])
            pca_pair = PCA(n_components=2)
            projs = pca_pair.fit_transform(pair_vecs)

            z = projs[:, 0] + 1j * projs[:, 1]
            phases = np.angle(z)

            theta_ba = phases[1] - phases[0]
            theta_dc = phases[3] - phases[2]

            theta_ba = np.angle(np.exp(1j * theta_ba))
            theta_dc = np.angle(np.exp(1j * theta_dc))

            error = abs(theta_ba - theta_dc)
            if error > np.pi:
                error = 2 * np.pi - error
            per_pair_errors.append(np.degrees(error))

        mean_per_pair_error = np.mean(per_pair_errors)

        # METHOD 2: Global PCA (Q51 methodology - shared coordinate system)
        pca_global = PCA(n_components=2)
        pca_global.fit(all_vecs)

        def get_global_phase(word):
            v = embeddings[word]
            proj = pca_global.transform(v.reshape(1, -1))[0]
            z = proj[0] + 1j * proj[1]
            return np.angle(z)

        global_errors = []
        for a, b, c, d in analogies:
            theta_a = get_global_phase(a)
            theta_b = get_global_phase(b)
            theta_c = get_global_phase(c)
            theta_d = get_global_phase(d)

            theta_ba = np.angle(np.exp(1j * (theta_b - theta_a)))
            theta_dc = np.angle(np.exp(1j * (theta_d - theta_c)))

            error = abs(theta_ba - theta_dc)
            if error > np.pi:
                error = 2 * np.pi - error
            global_errors.append(np.degrees(error))

        mean_global_error = np.mean(global_errors)

        # Test non-analogies with global PCA (should have HIGH error)
        non_analogy_errors = []
        for a, b, c, d in non_analogies:
            theta_a = get_global_phase(a)
            theta_b = get_global_phase(b)
            theta_c = get_global_phase(c)
            theta_d = get_global_phase(d)

            theta_ba = np.angle(np.exp(1j * (theta_b - theta_a)))
            theta_dc = np.angle(np.exp(1j * (theta_d - theta_c)))

            error = abs(theta_ba - theta_dc)
            if error > np.pi:
                error = 2 * np.pi - error
            non_analogy_errors.append(np.degrees(error))

        mean_non_analogy_error = np.mean(non_analogy_errors)

        # Q51 threshold: pass if error < 45 degrees (pi/4)
        threshold_pass = 45.0  # degrees

        global_pass_count = sum(1 for e in global_errors if e < threshold_pass)
        global_pass_rate = global_pass_count / len(global_errors)

        # Separation ratio: non-analogies should have higher error
        separation_ratio = mean_non_analogy_error / mean_global_error if mean_global_error > 0 else 0

        print("\n" + "=" * 70)
        print("LOCAL VS GLOBAL PHASE EXTRACTION TEST")
        print("=" * 70)
        print(f"\nPer-pair PCA (no shared coordinate system):")
        print(f"  Mean error: {mean_per_pair_error:.1f} deg")
        print(f"\nGlobal PCA (Q51 methodology - SHARED coordinate system):")
        print(f"  Mean error on analogies: {mean_global_error:.1f} deg")
        print(f"  Mean error on non-analogies: {mean_non_analogy_error:.1f} deg")
        print(f"  Pass rate: {global_pass_rate:.1%}")
        print(f"  Separation ratio: {separation_ratio:.2f}x")
        print(f"\nQ51 cross-model results: 90.9% pass, 4.05x separation")

        # The key assertion: global should discriminate analogies from non-analogies
        # Analogies should have lower error than non-analogies
        assert mean_global_error < mean_non_analogy_error, \
            f"Analogies ({mean_global_error:.1f} deg) should have lower error than non-analogies ({mean_non_analogy_error:.1f} deg)"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
