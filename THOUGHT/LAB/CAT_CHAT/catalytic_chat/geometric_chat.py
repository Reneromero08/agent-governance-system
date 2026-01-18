"""
Geometric Chat - I.2 CAT Chat Integration

Chat with geometric reasoning. Embeddings ONLY at boundaries.
All reasoning is pure vector operations validated by Q44/Q45.

5-Step Pipeline (from FERAL_RESIDENT_QUANTUM_ROADMAP):
1. BOUNDARY: Initialize query to manifold
2. PURE GEOMETRY: Project onto context
3. PURE GEOMETRY: E-gate for relevance (threshold=0.5)
4. BOUNDARY: Generate with LLM if gate open
5. PURE GEOMETRY: Update conversation state via entangle

Acceptance Criteria:
- I.2.1: Geometric context assembly works
- I.2.2: E-gating correlates with response quality
- I.2.3: Conversation state updates geometrically
- I.2.4: High-E responses are measurably better
"""

import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# Add CAPABILITY to path for imports
CAPABILITY_PATH = Path(__file__).parent.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

# Add NAVIGATION/CORTEX for cassette network
CORTEX_PATH = Path(__file__).parent.parent.parent.parent / "NAVIGATION" / "CORTEX" / "network"
if str(CORTEX_PATH) not in sys.path:
    sys.path.insert(0, str(CORTEX_PATH))

try:
    from geometric_reasoner import (
        GeometricReasoner,
        GeometricState,
        GeometricOperations
    )
except ImportError:
    GeometricReasoner = None
    GeometricState = None
    GeometricOperations = None

try:
    from geometric_cassette import GeometricCassetteNetwork
except ImportError:
    GeometricCassetteNetwork = None


@dataclass
class GeometricChatResult:
    """
    Result of respond() operation.

    Pattern from FERAL_RESIDENT/vector_brain.py:ThinkResult
    """
    response: str
    E_resonance: float       # Query resonance with context (mean E)
    E_compression: float     # Response resonance with conversation state
    gate_open: bool
    query_Df: float
    conversation_Df: float
    distance_from_start: float
    turn_index: int
    receipt: Dict


class GeometricChat:
    """
    Chat with geometric reasoning. Embeddings ONLY at boundaries.

    5-Step Pipeline (from roadmap):
    1. BOUNDARY: Initialize query to manifold
    2. PURE GEOMETRY: Project onto context
    3. PURE GEOMETRY: E-gate for relevance (threshold=0.5)
    4. BOUNDARY: Generate with LLM if gate open
    5. PURE GEOMETRY: Update conversation state via entangle

    Usage:
        chat = GeometricChat()

        result = chat.respond(
            "What is authentication?",
            ["OAuth is...", "JWT tokens are..."],
            lambda q, ctx: "Generated response"
        )

        print(f"E: {result.E_resonance:.3f}, Gate: {result.gate_open}")
        print(f"Conversation Df: {result.conversation_Df:.1f}")
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        E_threshold: float = 0.5,
        cassette_network: Optional['GeometricCassetteNetwork'] = None,
        auto_routing: bool = False
    ):
        """
        Initialize geometric chat.

        Args:
            model_name: Sentence transformer model for embeddings
            E_threshold: Threshold for E-gating (default 0.5 from Q44)
            cassette_network: Optional GeometricCassetteNetwork for context retrieval
            auto_routing: If True, auto-discover cassette network from config
        """
        if GeometricReasoner is None:
            raise ImportError(
                "GeometricReasoner not available. "
                "Install: pip install sentence-transformers"
            )

        # Lazy init (pattern from geometric_cassette.py:128-138)
        self._reasoner: Optional[GeometricReasoner] = None
        self._model_name = model_name

        # Conversation state
        self.conversation_state: Optional[GeometricState] = None
        self._initial_state: Optional[GeometricState] = None
        self.turn_history: List[Dict] = []
        self.E_threshold = E_threshold

        # Cassette network integration (I.1)
        # Auto-routing loads from cassettes.json if enabled
        self._cassette_network = cassette_network
        self._auto_routing = auto_routing
        if auto_routing and cassette_network is None:
            self._cassette_network = self._auto_load_network()

        # Stats (pattern from geometric_cassette.py:119-126)
        self._stats = {
            'turns': 0,
            'gate_opens': 0,
            'gate_closes': 0,
            'embedding_calls': 0,
            'geometric_ops': 0
        }

    @property
    def reasoner(self) -> 'GeometricReasoner':
        """Lazy init (pattern from geometric_cassette.py:128-138)"""
        if self._reasoner is None:
            self._reasoner = GeometricReasoner(self._model_name)
        return self._reasoner

    @property
    def cassette_network(self) -> Optional['GeometricCassetteNetwork']:
        """Access the cassette network (may be None if not configured)."""
        return self._cassette_network

    @property
    def has_routing(self) -> bool:
        """Check if cassette network is available for routing."""
        return self._cassette_network is not None and len(self._cassette_network.cassettes) > 0

    def _auto_load_network(self) -> Optional['GeometricCassetteNetwork']:
        """
        Auto-load cassette network from cassettes.json.

        Returns:
            GeometricCassetteNetwork or None if loading fails
        """
        if GeometricCassetteNetwork is None:
            return None

        try:
            return GeometricCassetteNetwork.from_config()
        except Exception as e:
            # Log but don't fail - auto-routing is optional
            import sys
            print(f"[WARN] Auto-routing unavailable: {e}", file=sys.stderr)
            return None

    @classmethod
    def with_auto_routing(
        cls,
        E_threshold: float = 0.5,
        model_name: str = 'all-MiniLM-L6-v2'
    ) -> 'GeometricChat':
        """
        Create GeometricChat with automatic cassette network routing.

        Convenience factory method that auto-discovers cassettes from
        NAVIGATION/CORTEX/network/cassettes.json.

        Args:
            E_threshold: Threshold for E-gating (default 0.5)
            model_name: Sentence transformer model

        Returns:
            GeometricChat with cassette network configured

        Example:
            # Quick setup with auto-routing
            chat = GeometricChat.with_auto_routing()

            # Use respond_with_retrieval for automatic context
            result = chat.respond_with_retrieval(
                "What is authentication?",
                llm_generate
            )
        """
        return cls(
            model_name=model_name,
            E_threshold=E_threshold,
            auto_routing=True
        )

    # ========================================================================
    # Main Interface
    # ========================================================================

    def respond(
        self,
        user_query: str,
        context_docs: List[str],
        llm_generate: Callable[[str, List[str]], str]
    ) -> GeometricChatResult:
        """
        Generate response with geometric reasoning.

        5-Step Pipeline:
        1. BOUNDARY: Initialize query to manifold
        2. PURE GEOMETRY: Project onto context
        3. PURE GEOMETRY: E-gate for relevance
        4. BOUNDARY: Generate with LLM if gate open
        5. PURE GEOMETRY: Update conversation state via entangle

        Args:
            user_query: User's question/input
            context_docs: List of context documents
            llm_generate: Function(query, context) -> response string

        Returns:
            GeometricChatResult with response and metrics
        """
        self._stats['turns'] += 1
        turn_index = self._stats['turns']

        # Step 1: BOUNDARY - Initialize query to manifold
        query_state = self.reasoner.initialize(user_query)
        self._stats['embedding_calls'] += 1

        # Initialize context states
        context_states = []
        for doc in context_docs:
            if doc and doc.strip():
                ctx_state = self.reasoner.initialize(doc)
                context_states.append(ctx_state)
                self._stats['embedding_calls'] += 1

        # Step 2: PURE GEOMETRY - Project onto context
        if context_states:
            projected = self.reasoner.project(query_state, context_states)
            self._stats['geometric_ops'] += 1
        else:
            projected = query_state

        # Step 3: PURE GEOMETRY - E-gate for relevance
        gate_result = self._compute_gate(query_state, context_states)

        # Step 4: BOUNDARY - Generate with LLM
        if gate_result['gate_open']:
            response_text = llm_generate(user_query, context_docs)
            self._stats['gate_opens'] += 1
        else:
            response_text = f"[LOW RESONANCE E={gate_result['E']:.3f}] " + \
                           llm_generate(user_query, context_docs)
            self._stats['gate_closes'] += 1

        # Step 5: PURE GEOMETRY - Update conversation state via entangle
        response_state = self.reasoner.initialize(response_text)
        self._stats['embedding_calls'] += 1

        E_compression = self._update_conversation_state(response_state)
        self._stats['geometric_ops'] += 1

        # Build receipt
        receipt = self._build_receipt(
            user_query=user_query,
            response_text=response_text,
            gate_result=gate_result,
            E_compression=E_compression,
            turn_index=turn_index
        )

        # Record turn history
        self.turn_history.append(receipt)

        return GeometricChatResult(
            response=response_text,
            E_resonance=gate_result['E'],
            E_compression=E_compression,
            gate_open=gate_result['gate_open'],
            query_Df=query_state.Df,
            conversation_Df=self.conversation_state.Df if self.conversation_state else 0.0,
            distance_from_start=self.conversation_distance_from_start(),
            turn_index=turn_index,
            receipt=receipt
        )

    # ========================================================================
    # Cassette Network Integration (I.1)
    # ========================================================================

    def retrieve_context_geometric(
        self,
        query_state: 'GeometricState',
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve context from cassette network using E-gating.

        Pattern from geometric_cassette.py:458-488 (query_with_gate)

        Args:
            query_state: GeometricState to query with
            k: Number of results to return
            threshold: E threshold (default: self.E_threshold)

        Returns:
            List of gated results with E scores
        """
        if self._cassette_network is None:
            return []

        threshold = threshold or self.E_threshold

        # Use existing GeometricCassetteNetwork.query_merged()
        results = self._cassette_network.query_merged(query_state, k * 2)

        # Apply E-gate (pattern from geometric_cassette.py)
        gated = [r for r in results if r['E'] >= threshold]

        return gated[:k]

    def respond_with_retrieval(
        self,
        user_query: str,
        llm_generate: Callable[[str, List[str]], str],
        k: int = 10
    ) -> GeometricChatResult:
        """
        Respond with automatic context retrieval from cassette network.

        Args:
            user_query: User's question
            llm_generate: LLM generation function
            k: Number of context docs to retrieve

        Returns:
            GeometricChatResult
        """
        if self._cassette_network is None:
            raise RuntimeError("No cassette network configured")

        # Initialize query
        query_state = self.reasoner.initialize(user_query)
        self._stats['embedding_calls'] += 1

        # Retrieve context geometrically
        results = self.retrieve_context_geometric(query_state, k)
        context_docs = [r.get('content', '') for r in results if r.get('content')]

        return self.respond(user_query, context_docs, llm_generate)

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _compute_gate(
        self,
        query_state: 'GeometricState',
        context_states: List['GeometricState']
    ) -> Dict:
        """
        Compute E-gate for relevance filtering.

        Returns:
            Dict with E, gate_open, context_alignment
        """
        if not context_states:
            return {
                'E': 0.0,
                'gate_open': False,
                'context_alignment': [],
                'threshold': self.E_threshold,
                'message': 'Empty context'
            }

        # Compute E (Born rule) with each context item
        E_values = [query_state.E_with(c) for c in context_states]
        self._stats['geometric_ops'] += len(E_values)

        E_mean = sum(E_values) / len(E_values) if E_values else 0.0

        return {
            'E': E_mean,
            'gate_open': E_mean >= self.E_threshold,
            'context_alignment': E_values,
            'threshold': self.E_threshold,
            'max_E': max(E_values) if E_values else 0.0,
            'min_E': min(E_values) if E_values else 0.0
        }

    def _update_conversation_state(
        self,
        response_state: 'GeometricState'
    ) -> float:
        """
        Update conversation state via entangle.

        Pattern from geometric_memory.py:77-80

        Returns:
            E_compression (response resonance with conversation state)
        """
        if self.conversation_state is None:
            # First turn
            self.conversation_state = response_state
            self._initial_state = GeometricState(
                vector=response_state.vector.copy(),
                operation_history=[]
            )
            return 1.0  # Perfect self-resonance

        # Compute E before entangle
        E_compression = response_state.E_with(self.conversation_state)

        # Entangle response into conversation state
        self.conversation_state = self.reasoner.entangle(
            self.conversation_state,
            response_state
        )

        return E_compression

    def _build_receipt(
        self,
        user_query: str,
        response_text: str,
        gate_result: Dict,
        E_compression: float,
        turn_index: int
    ) -> Dict:
        """Build provenance receipt for this turn."""
        return {
            'turn_index': turn_index,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query_hash': hashlib.sha256(user_query.encode()).hexdigest()[:16],
            'response_hash': hashlib.sha256(response_text.encode()).hexdigest()[:16],
            'E_resonance': gate_result['E'],
            'E_compression': E_compression,
            'gate_open': gate_result['gate_open'],
            'threshold': gate_result['threshold'],
            'context_alignment': gate_result['context_alignment'],
            'conversation_Df': self.conversation_state.Df if self.conversation_state else 0.0,
            'distance_from_start': self.conversation_distance_from_start(),
            'conversation_hash': self.conversation_state.receipt()['vector_hash'] if self.conversation_state else None
        }

    # ========================================================================
    # Metrics (pattern from geometric_memory.py:164-197)
    # ========================================================================

    def conversation_distance_from_start(self) -> float:
        """
        Track how far conversation has evolved (geodesic distance).

        Pattern from geometric_memory.py:164-174

        Returns:
            Angle in radians from initial state
        """
        if self._initial_state is None or self.conversation_state is None:
            return 0.0

        return self.conversation_state.distance_to(self._initial_state)

    def get_metrics(self) -> Dict:
        """
        Get comprehensive conversation metrics.

        Pattern from geometric_memory.py:176-197
        """
        if self.conversation_state is None:
            return {
                'turn_count': 0,
                'conversation_Df': 0.0,
                'distance_from_start': 0.0,
                'Df_history': [],
                'E_history': [],
                'gate_open_rate': 0.0,
                'stats': self._stats
            }

        return {
            'turn_count': len(self.turn_history),
            'conversation_Df': self.conversation_state.Df,
            'distance_from_start': self.conversation_distance_from_start(),
            'Df_history': [t.get('conversation_Df', 0) for t in self.turn_history],
            'E_history': [t.get('E_resonance', 0) for t in self.turn_history],
            'gate_open_rate': self._stats['gate_opens'] / max(1, self._stats['turns']),
            'conversation_hash': self.conversation_state.receipt()['vector_hash'],
            'stats': self._stats,
            'reasoner_stats': self.reasoner.get_stats()
        }

    def clear(self):
        """Reset conversation state (for testing or new sessions)"""
        self.conversation_state = None
        self._initial_state = None
        self.turn_history = []
        self._stats = {
            'turns': 0,
            'gate_opens': 0,
            'gate_closes': 0,
            'embedding_calls': 0,
            'geometric_ops': 0
        }


# ============================================================================
# Example Usage
# ============================================================================

def example_geometric_chat():
    """Demonstrate geometric chat"""
    print("=== Geometric Chat Example ===\n")

    # Simple echo LLM for testing
    def echo_llm(query: str, context: List[str]) -> str:
        ctx_summary = f" (context: {len(context)} docs)" if context else ""
        return f"Response to: {query}{ctx_summary}"

    chat = GeometricChat()

    # Simulate conversation
    queries = [
        ("What is authentication?", ["OAuth provides...", "JWT tokens are..."]),
        ("How do I implement JWT?", ["To implement JWT...", "The signing process..."]),
        ("What about refresh tokens?", ["Refresh tokens allow...", "Security considerations..."]),
    ]

    for query, context in queries:
        result = chat.respond(query, context, echo_llm)
        print(f"Q: {query}")
        print(f"A: {result.response}")
        print(f"   E={result.E_resonance:.3f}, Gate={result.gate_open}, Df={result.conversation_Df:.1f}")
        print()

    print("=== Conversation Metrics ===")
    metrics = chat.get_metrics()
    print(f"Turns: {metrics['turn_count']}")
    print(f"Distance from start: {metrics['distance_from_start']:.3f} radians")
    print(f"Gate open rate: {metrics['gate_open_rate']:.1%}")
    print(f"Df trend: {[f'{d:.1f}' for d in metrics['Df_history']]}")


if __name__ == "__main__":
    example_geometric_chat()
