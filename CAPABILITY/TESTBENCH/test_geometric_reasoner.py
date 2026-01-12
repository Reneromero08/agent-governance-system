"""
Tests for Geometric Reasoner (Q43/Q44/Q45 Validation)

These tests verify that the geometric operations work correctly
and that the quantum-semantic properties are preserved.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add CAPABILITY to path
PRIMITIVES_PATH = Path(__file__).parent.parent / "PRIMITIVES"
sys.path.insert(0, str(PRIMITIVES_PATH))

from geometric_reasoner import (
    GeometricState,
    GeometricOperations,
    GeometricReasoner,
    renormalize_periodically,
    trim_history
)


# ============================================================================
# GeometricState Tests (Q43)
# ============================================================================

class TestGeometricState:
    """Tests for Q43 quantum state properties"""

    def test_normalization(self):
        """State should be normalized to unit sphere"""
        v = np.array([3.0, 4.0, 0.0])
        state = GeometricState(vector=v)
        norm = np.linalg.norm(state.vector)
        assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_Df_participation_ratio(self):
        """Df should measure how spread out the state is"""
        # Concentrated state (low Df)
        concentrated = GeometricState(vector=np.array([1.0, 0.0, 0.0, 0.0]))
        # Will be normalized, so [1,0,0,0]
        assert concentrated.Df == pytest.approx(1.0, abs=0.1)

        # Spread state (high Df)
        spread = GeometricState(vector=np.array([1.0, 1.0, 1.0, 1.0]))
        # After normalization: [0.5, 0.5, 0.5, 0.5]
        assert spread.Df == pytest.approx(4.0, abs=0.1)

    def test_E_with_self(self):
        """E with self should be 1.0 (normalized state)"""
        state = GeometricState(vector=np.random.randn(100))
        E = state.E_with(state)
        assert E == pytest.approx(1.0, abs=1e-6)

    def test_E_with_orthogonal(self):
        """E with orthogonal state should be 0.0"""
        state1 = GeometricState(vector=np.array([1.0, 0.0, 0.0]))
        state2 = GeometricState(vector=np.array([0.0, 1.0, 0.0]))
        E = state1.E_with(state2)
        assert E == pytest.approx(0.0, abs=1e-6)

    def test_distance_to_self(self):
        """Distance to self should be 0"""
        state = GeometricState(vector=np.random.randn(100))
        d = state.distance_to(state)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_distance_to_orthogonal(self):
        """Distance to orthogonal state should be pi/2"""
        state1 = GeometricState(vector=np.array([1.0, 0.0, 0.0]))
        state2 = GeometricState(vector=np.array([0.0, 1.0, 0.0]))
        d = state1.distance_to(state2)
        assert d == pytest.approx(np.pi / 2, abs=1e-6)

    def test_receipt_has_required_fields(self):
        """Receipt should have all required fields"""
        state = GeometricState(vector=np.random.randn(100))
        receipt = state.receipt()
        assert 'vector_hash' in receipt
        assert 'Df' in receipt
        assert 'dim' in receipt
        assert 'operations' in receipt


# ============================================================================
# GeometricOperations Tests (Q45)
# ============================================================================

class TestGeometricOperations:
    """Tests for Q45 geometric operations"""

    def test_add_produces_normalized_result(self):
        """Addition should produce normalized state"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))
        result = GeometricOperations.add(s1, s2)
        norm = np.linalg.norm(result.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_subtract_produces_normalized_result(self):
        """Subtraction should produce normalized state"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))
        result = GeometricOperations.subtract(s1, s2)
        norm = np.linalg.norm(result.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_superpose_produces_normalized_result(self):
        """Superposition should produce normalized state"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))
        result = GeometricOperations.superpose(s1, s2)
        norm = np.linalg.norm(result.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_entangle_produces_normalized_result(self):
        """Entanglement should produce normalized state"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))
        result = GeometricOperations.entangle(s1, s2)
        norm = np.linalg.norm(result.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_entangle_disentangle_recovery(self):
        """Disentangle should approximately recover original"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))

        bound = GeometricOperations.entangle(s1, s2)
        recovered = GeometricOperations.disentangle(bound, s2)

        # Should have high similarity with original
        similarity = recovered.E_with(s1)
        # Note: Recovery is approximate, not exact
        assert similarity > 0.5, f"Expected >0.5 similarity, got {similarity}"

    def test_interpolate_endpoints(self):
        """Interpolation at t=0 and t=1 should give endpoints"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))

        at_0 = GeometricOperations.interpolate(s1, s2, 0.0)
        at_1 = GeometricOperations.interpolate(s1, s2, 1.0)

        assert at_0.E_with(s1) > 0.99
        assert at_1.E_with(s2) > 0.99

    def test_interpolate_midpoint(self):
        """Midpoint should be equidistant from both endpoints"""
        s1 = GeometricState(vector=np.array([1.0, 0.0, 0.0]))
        s2 = GeometricState(vector=np.array([0.0, 1.0, 0.0]))

        midpoint = GeometricOperations.interpolate(s1, s2, 0.5)

        d1 = midpoint.distance_to(s1)
        d2 = midpoint.distance_to(s2)

        assert abs(d1 - d2) < 1e-6, f"Distances should be equal: {d1} vs {d2}"

    def test_project_onto_single_context(self):
        """Projection onto single context should align with it"""
        query = GeometricState(vector=np.random.randn(100))
        context = GeometricState(vector=np.random.randn(100))

        projected = GeometricOperations.project(query, [context])

        # Should have high similarity with context
        similarity = projected.E_with(context)
        assert similarity > 0.5

    def test_operations_have_receipts(self):
        """All operations should record receipts"""
        s1 = GeometricState(vector=np.random.randn(100))
        s2 = GeometricState(vector=np.random.randn(100))

        result = GeometricOperations.add(s1, s2)
        assert len(result.operation_history) > 0
        assert result.operation_history[-1]['op'] == 'add'


# ============================================================================
# GeometricReasoner Tests
# ============================================================================

@pytest.fixture
def reasoner():
    """Create a reasoner for testing"""
    try:
        return GeometricReasoner()
    except ImportError:
        pytest.skip("sentence-transformers not installed")


class TestGeometricReasoner:
    """Tests for the GeometricReasoner interface"""

    def test_initialize_produces_state(self, reasoner):
        """Initialize should produce a GeometricState"""
        state = reasoner.initialize("test text")
        assert isinstance(state, GeometricState)
        assert len(state.vector) == reasoner.dim

    def test_initialize_is_deterministic(self, reasoner):
        """Same text should produce same state"""
        state1 = reasoner.initialize("test text")
        state2 = reasoner.initialize("test text")

        similarity = state1.E_with(state2)
        assert similarity > 0.999

    def test_different_text_produces_different_state(self, reasoner):
        """Different text should produce different states"""
        state1 = reasoner.initialize("cat")
        state2 = reasoner.initialize("quantum mechanics")

        similarity = state1.E_with(state2)
        assert similarity < 0.9  # Should be different

    def test_stats_tracking(self, reasoner):
        """Stats should track operations"""
        initial_stats = reasoner.get_stats()
        assert initial_stats['initializations'] == 0

        reasoner.initialize("test")
        reasoner.initialize("another test")

        stats = reasoner.get_stats()
        assert stats['initializations'] == 2

    def test_geometric_ops_increment_counter(self, reasoner):
        """Geometric operations should increment counter"""
        s1 = reasoner.initialize("cat")
        s2 = reasoner.initialize("dog")

        initial = reasoner.stats['geometric_operations']

        reasoner.add(s1, s2)
        reasoner.superpose(s1, s2)
        reasoner.entangle(s1, s2)

        assert reasoner.stats['geometric_operations'] == initial + 3


# ============================================================================
# Q45 Semantic Operation Tests
# ============================================================================

class TestQ45SemanticOperations:
    """Tests validating Q45 semantic geometry claims"""

    def test_analogy_direction(self, reasoner):
        """Analogy should find semantically related concepts"""
        corpus = ["woman", "girl", "female", "person", "chair", "table"]

        # king - queen + man should be closer to woman than to chair
        results = reasoner.analogy("king", "queen", "man", corpus, k=6)

        # Get positions
        result_texts = [r[0] for r in results]

        # "woman" or "female" should rank higher than "chair"
        woman_idx = result_texts.index("woman") if "woman" in result_texts else 99
        female_idx = result_texts.index("female") if "female" in result_texts else 99
        chair_idx = result_texts.index("chair") if "chair" in result_texts else 99

        best_semantic = min(woman_idx, female_idx)
        assert best_semantic < chair_idx, f"Semantic result should rank higher than 'chair'"

    def test_blend_finds_hypernym(self, reasoner):
        """Blending should find common hypernyms"""
        corpus = ["pet", "animal", "mammal", "furniture", "building"]

        results = reasoner.blend("cat", "dog", corpus, k=5)
        result_texts = [r[0] for r in results]

        # "pet" or "animal" or "mammal" should rank high
        semantic_matches = [t for t in result_texts[:3] if t in ["pet", "animal", "mammal"]]
        assert len(semantic_matches) > 0, f"Expected semantic match in top 3, got {result_texts[:3]}"

    def test_navigate_produces_path(self, reasoner):
        """Navigate should produce intermediate points"""
        corpus = ["hot", "warm", "lukewarm", "cool", "cold"]

        path = reasoner.navigate("hot", "cold", steps=2, corpus=corpus, k=3)

        assert len(path) == 3  # start, middle, end
        assert path[0]['t'] == 0.0
        assert path[-1]['t'] == 1.0

    def test_gate_discriminates(self, reasoner):
        """Gate should open for related queries, close for unrelated"""
        context = ["Python programming", "web development", "Django framework"]

        # Related query
        related = reasoner.gate("How do I build a website?", context, threshold=0.3)

        # Unrelated query
        unrelated = reasoner.gate("What is the capital of France?", context, threshold=0.3)

        # Related should have higher E
        assert related['E'] > unrelated['E'], f"Related E ({related['E']}) should exceed unrelated E ({unrelated['E']})"


# ============================================================================
# Drift and Stability Tests
# ============================================================================

class TestDriftAndStability:
    """Tests for numerical stability over many operations"""

    def test_normalization_after_many_ops(self):
        """State should remain normalized after many operations"""
        state = GeometricState(vector=np.random.randn(100))

        for _ in range(100):
            other = GeometricState(vector=np.random.randn(100))
            state = GeometricOperations.entangle(state, other)

        norm = np.linalg.norm(state.vector)
        assert abs(norm - 1.0) < 1e-5, f"Norm drifted to {norm} after 100 ops"

    def test_renormalize_utility(self):
        """Renormalize utility should fix drift"""
        state = GeometricState(vector=np.random.randn(100))

        # Manually introduce drift
        state.vector = state.vector * 1.5

        state = renormalize_periodically(state, every_n_ops=1)
        # Force trigger by adding to history
        state.operation_history = [{}] * 100
        state = renormalize_periodically(state, every_n_ops=100)

        norm = np.linalg.norm(state.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_trim_history_utility(self):
        """Trim history should cap operation history"""
        state = GeometricState(vector=np.random.randn(100))
        state.operation_history = [{}] * 500

        state = trim_history(state, max_history=100)

        assert len(state.operation_history) == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""

    def test_full_reasoning_chain(self, reasoner):
        """Complete reasoning chain should work"""
        # Initialize concepts
        king = reasoner.initialize("king")
        queen = reasoner.initialize("queen")
        man = reasoner.initialize("man")

        # Perform analogy calculation geometrically
        diff = reasoner.subtract(queen, king)
        result = reasoner.add(man, diff)

        # Should be close to "woman"
        woman = reasoner.initialize("woman")
        similarity = result.E_with(woman)

        # Not expecting perfect match, but should be reasonable
        assert similarity > 0.3, f"Expected >0.3 similarity to 'woman', got {similarity}"

    def test_embedding_reduction(self, reasoner):
        """Geometric ops should dominate over boundary ops"""
        # Do a chain of operations
        state = reasoner.initialize("start")
        for _ in range(10):
            other = reasoner.initialize("other")  # 1 boundary op
            state = reasoner.entangle(state, other)  # 1 geometric op
            state = reasoner.superpose(state, other)  # 1 geometric op

        stats = reasoner.get_stats()

        # Should have more geometric ops than boundary ops
        # 11 initializations vs 20 geometric ops
        assert stats['geometric_operations'] >= stats['initializations']


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
