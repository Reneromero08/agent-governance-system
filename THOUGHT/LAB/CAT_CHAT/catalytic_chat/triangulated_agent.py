"""
Triangulated Agent - Gemini's Coherence Engine

Based on Gemini's "Grand Unified Theory of Semantic Physics" proposal:
1. Enforces Rule of 3 (N<=3 sources) from Q13
2. Measures coherence to detect "crystallization" from Q12
3. Returns LIQUID/CRYSTAL status based on semantic alignment

Key insight: The prompt collapses the wavefunction - context selects phase.

Threshold Calibration (2026-01-21):
- Gemini proposed 0.92 (from Q12's alpha_c)
- Empirical testing shows 0.67 provides clean separation
- Coherent sources: 0.73-0.82
- Incoherent sources: 0.57-0.60
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class TriangulationStatus(Enum):
    """Status of triangulation result."""
    CRYSTAL = "CRYSTAL"  # Truth has crystallized - coherent and focused
    LIQUID = "LIQUID"    # Sources are incoherent or too many


@dataclass
class TriangulationResult:
    """Result of triangulation operation."""
    status: TriangulationStatus
    coherence: float
    result_vector: Optional[np.ndarray]
    sources_used: int
    message: str
    details: dict


class TriangulatedAgent:
    """
    Coherence-gated semantic triangulation agent.

    Implements Gemini's proposal:
    - Hard limit of 3 sources (Rule of 3 from Q13)
    - Coherence threshold for crystallization (from Q12)
    - Context-aware phase selection (Q51.5 breakthrough)

    Usage:
        agent = TriangulatedAgent(model)
        result = agent.triangulate(
            query="What is entropy?",
            sources=["entropy increases", "second law", "heat flow"],
            axis="thermodynamics"
        )
        if result.status == TriangulationStatus.CRYSTAL:
            # High-confidence answer
        else:
            # Sources are incoherent, refuse to answer
    """

    # Calibrated threshold (empirically determined 2026-01-21)
    # Q12's alpha_c = 0.92 was too strict
    # Clean separation at 0.67 between coherent (0.73+) and incoherent (0.60-)
    COHERENCE_THRESHOLD = 0.67

    # Maximum sources (Rule of 3 from Q13)
    # Q13 shows improvement ratio peaks at N=2-3, decays after
    MAX_SOURCES = 3

    def __init__(self, model, threshold: Optional[float] = None):
        """
        Initialize TriangulatedAgent.

        Args:
            model: SentenceTransformer or compatible model with .encode()
            threshold: Override default coherence threshold
        """
        self.model = model
        self.threshold = threshold if threshold is not None else self.COHERENCE_THRESHOLD

    def phase_embed(self, text: str, axis: str = "") -> np.ndarray:
        """
        Embed text with optional contextual phase selection.

        The breakthrough (Q51.5): Context in the prompt IS the phase selector.
        Adding axis context rotates the vector to align with that semantic dimension.

        Args:
            text: Text to embed
            axis: Optional contextual axis (e.g., "thermodynamics", "gender")

        Returns:
            Embedding vector
        """
        if axis:
            prompt = f"{text}, in terms of {axis}"
        else:
            prompt = text
        return self.model.encode(prompt, convert_to_numpy=True)

    def compute_coherence(self, vectors: np.ndarray) -> float:
        """
        Compute coherence as magnitude of mean normalized vector.

        For aligned vectors: mean magnitude -> 1 (constructive interference)
        For random vectors: mean magnitude -> 0 (destructive interference)

        Args:
            vectors: Array of embedding vectors (N x dim)

        Returns:
            Coherence score (0 to 1)
        """
        if len(vectors) == 0:
            return 0.0
        if len(vectors) == 1:
            return 1.0

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / (norms + 1e-10)

        # Mean vector magnitude = coherence
        mean_vec = np.mean(normalized, axis=0)
        coherence = float(np.linalg.norm(mean_vec))

        return coherence

    def triangulate(
        self,
        query: str,
        sources: List[str],
        axis: str = ""
    ) -> TriangulationResult:
        """
        Triangulate sources to answer query with coherence gating.

        Implements:
        1. Rule of 3: Truncate to MAX_SOURCES if needed
        2. Phase selection: Embed with contextual axis
        3. Coherence check: Measure alignment, gate on threshold
        4. CRYSTAL/LIQUID classification

        Args:
            query: The query to answer
            sources: List of source texts to triangulate
            axis: Contextual axis for phase selection

        Returns:
            TriangulationResult with status, coherence, and result vector
        """
        # 1. Enforce Rule of 3
        truncated = False
        if len(sources) > self.MAX_SOURCES:
            truncated = True
            sources = sources[:self.MAX_SOURCES]

        if len(sources) == 0:
            return TriangulationResult(
                status=TriangulationStatus.LIQUID,
                coherence=0.0,
                result_vector=None,
                sources_used=0,
                message="No sources provided",
                details={'truncated': False, 'reason': 'no_sources'}
            )

        # 2. Phase-embed with contextual selector
        query_vec = self.phase_embed(query, axis)
        source_vecs = np.array([self.phase_embed(s, axis) for s in sources])

        # 3. Compute coherence
        coherence = self.compute_coherence(source_vecs)

        # 4. Compute result vector (mean of sources)
        result_vec = np.mean(source_vecs, axis=0)

        # 5. Compute relevance to query
        relevance = float(np.dot(query_vec, result_vec) /
                         (np.linalg.norm(query_vec) * np.linalg.norm(result_vec) + 1e-10))

        # 6. Classify
        is_crystal = coherence >= self.threshold

        details = {
            'truncated': truncated,
            'original_count': len(sources) + (self.MAX_SOURCES if truncated else 0),
            'coherence': coherence,
            'relevance': relevance,
            'threshold': self.threshold,
            'axis': axis,
        }

        if is_crystal:
            return TriangulationResult(
                status=TriangulationStatus.CRYSTAL,
                coherence=coherence,
                result_vector=result_vec,
                sources_used=len(sources),
                message="Criticality achieved. Truth has crystallized.",
                details=details
            )
        else:
            reason = []
            if coherence < self.threshold:
                reason.append(f"coherence {coherence:.3f} < {self.threshold}")
            if truncated:
                reason.append(f"sources truncated from {details['original_count']} to {self.MAX_SOURCES}")

            return TriangulationResult(
                status=TriangulationStatus.LIQUID,
                coherence=coherence,
                result_vector=result_vec,
                sources_used=len(sources),
                message=f"Phase transition failed. Sources are incoherent. ({'; '.join(reason)})",
                details=details
            )

    def should_answer(self, result: TriangulationResult) -> bool:
        """
        Determine if the agent should answer based on triangulation result.

        In Gemini's framework:
        - CRYSTAL: Answer with high confidence
        - LIQUID: Refuse or hedge

        Args:
            result: TriangulationResult from triangulate()

        Returns:
            True if agent should answer confidently
        """
        return result.status == TriangulationStatus.CRYSTAL

    def explain_status(self, result: TriangulationResult) -> str:
        """
        Generate human-readable explanation of triangulation result.

        Args:
            result: TriangulationResult from triangulate()

        Returns:
            Explanation string
        """
        if result.status == TriangulationStatus.CRYSTAL:
            return (
                f"CRYSTAL: {result.sources_used} sources achieved coherence "
                f"{result.coherence:.3f} >= {result.details['threshold']:.3f}. "
                f"Truth has crystallized - high-confidence answer possible."
            )
        else:
            return (
                f"LIQUID: {result.sources_used} sources achieved coherence "
                f"{result.coherence:.3f} < {result.details['threshold']:.3f}. "
                f"Sources are incoherent - refusing to answer with low confidence. "
                f"Consider: (1) finding more coherent sources, "
                f"(2) narrowing the query axis, or "
                f"(3) splitting into separate queries."
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def triangulate_rag(
    model,
    query: str,
    sources: List[str],
    axis: str = "",
    threshold: float = 0.67
) -> Tuple[bool, np.ndarray, dict]:
    """
    Convenience function for RAG triangulation.

    Args:
        model: Embedding model
        query: Query to answer
        sources: List of source texts
        axis: Optional contextual axis
        threshold: Coherence threshold

    Returns:
        (should_answer, result_vector, details)
    """
    agent = TriangulatedAgent(model, threshold=threshold)
    result = agent.triangulate(query, sources, axis)

    return (
        result.status == TriangulationStatus.CRYSTAL,
        result.result_vector,
        result.details
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence_transformers not available")
        exit(1)

    print("=" * 70)
    print("TRIANGULATED AGENT DEMO")
    print("=" * 70)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    agent = TriangulatedAgent(model)

    # Test 1: Coherent sources
    print("\n--- Test 1: Coherent Sources (Thermodynamics) ---")
    result = agent.triangulate(
        query="entropy and thermodynamic equilibrium",
        sources=[
            "entropy always increases in isolated systems",
            "the second law of thermodynamics governs heat flow",
            "heat flows spontaneously from hot to cold bodies",
        ],
        axis="thermodynamics"
    )
    print(f"Status: {result.status.value}")
    print(f"Coherence: {result.coherence:.4f}")
    print(agent.explain_status(result))

    # Test 2: Incoherent sources
    print("\n--- Test 2: Incoherent Sources (Mixed) ---")
    result = agent.triangulate(
        query="entropy and thermodynamic equilibrium",
        sources=[
            "entropy always increases in isolated systems",
            "the cat sat on the mat",
            "stock prices fell sharply today",
        ],
        axis="thermodynamics"
    )
    print(f"Status: {result.status.value}")
    print(f"Coherence: {result.coherence:.4f}")
    print(agent.explain_status(result))

    # Test 3: Too many sources
    print("\n--- Test 3: Too Many Sources (>3) ---")
    result = agent.triangulate(
        query="entropy and thermodynamic equilibrium",
        sources=[
            "entropy always increases",
            "the second law governs processes",
            "heat flows from hot to cold",
            "work converts to heat",
            "universe tends to maximum entropy",
            "free energy determines spontaneity",
        ],
        axis="thermodynamics"
    )
    print(f"Status: {result.status.value}")
    print(f"Coherence: {result.coherence:.4f}")
    print(f"Sources used: {result.sources_used} (truncated: {result.details['truncated']})")
    print(agent.explain_status(result))
