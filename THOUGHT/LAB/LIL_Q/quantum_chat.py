"""
LIL_Q - Chat with the quantum manifold.
Your formula: E = <psi|phi>

Q44 proved: E correlates r=0.977 with Born rule
Q45 proved: Pure geometry works 100% across 5 models

Context comes from outside (MCP, files, etc). LIL_Q stays pure.
"""

import numpy as np
from typing import Callable, Optional, List
import sys
from pathlib import Path

# Use existing CORTEX embedding engine
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine


class QuantumChat:
    """Navigate the quantum semantic manifold."""

    def __init__(self, llm_fn: Callable[[str, float], str]):
        """
        Args:
            llm_fn: Function(query, E) -> response
        """
        self.engine = EmbeddingEngine()  # Uses existing CORTEX engine
        self.llm = llm_fn
        self.mind: Optional[np.ndarray] = None

    def enter(self, text: str) -> np.ndarray:
        """Enter the manifold. Text -> unit vector."""
        v = self.engine.embed(text)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def E(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        The formula. Born rule.
        E = <psi|phi> = dot(v1, v2)
        """
        return float(np.dot(v1, v2))

    def navigate(self, query: np.ndarray, context: List[np.ndarray] = None) -> tuple[np.ndarray, float]:
        """
        Navigate manifold with optional context.
        Returns (new_state, resonance).

        Context blending: query += sum(c * E(query, c) for c in context)
        Mind blending: query += mind * E(query, mind)
        """
        blended = query.copy()

        # Blend with context if provided (from MCP, files, etc)
        if context:
            for c in context:
                E_c = self.E(query, c)
                blended = blended + c * E_c

        # Blend with mind if we have memory
        if self.mind is not None:
            resonance = self.E(query, self.mind)
            blended = blended + self.mind * resonance
        else:
            resonance = 1.0

        # Normalize
        norm = np.linalg.norm(blended)
        blended = blended / norm if norm > 0 else blended

        return blended, resonance

    def remember(self, state: np.ndarray):
        """Update mind via quantum entanglement (circular convolution)."""
        if self.mind is None:
            self.mind = state
        else:
            entangled = np.fft.ifft(
                np.fft.fft(self.mind) * np.fft.fft(state)
            ).real
            norm = np.linalg.norm(entangled)
            self.mind = entangled / norm if norm > 0 else entangled

    def chat(self, query: str, context: List[str] = None) -> tuple[str, float]:
        """
        The whole thing.
        Enter -> Navigate -> Remember -> Respond -> Remember

        Context comes from outside (MCP, files, etc). Just strings.
        """
        # Enter manifold
        q = self.enter(query)

        # Enter context to manifold (if provided)
        context_vectors = None
        if context:
            context_vectors = [self.enter(doc) for doc in context if doc]

        # Navigate with context (measure resonance with mind)
        state, E_val = self.navigate(q, context_vectors)

        # Remember query state
        self.remember(state)

        # Generate response (LLM knows E)
        response = self.llm(query, E_val)

        # Remember response
        r = self.enter(response)
        self.remember(r)

        return response, E_val
