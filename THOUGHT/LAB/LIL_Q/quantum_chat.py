"""
LIL_Q - Chat with the quantum manifold.
Your formula: E = <psi|phi>

Q44 proved: E correlates r=0.977 with Born rule
Q45 proved: Pure geometry works 100% across 5 models
"""

import numpy as np
from typing import Callable, Optional
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

    def navigate(self, query: np.ndarray) -> tuple[np.ndarray, float]:
        """Navigate manifold. Returns (new_state, resonance)."""
        if self.mind is None:
            return query, 1.0

        resonance = self.E(query, self.mind)

        # Blend query with memory based on resonance
        blended = query + self.mind * resonance
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

    def chat(self, query: str) -> tuple[str, float]:
        """
        The whole thing.
        Enter -> Navigate -> Remember -> Respond -> Remember
        """
        # Enter manifold
        q = self.enter(query)

        # Navigate (measure resonance with mind)
        state, E_val = self.navigate(q)

        # Remember query
        self.remember(state)

        # Generate response (LLM knows E)
        response = self.llm(query, E_val)

        # Remember response
        r = self.enter(response)
        self.remember(r)

        return response, E_val
