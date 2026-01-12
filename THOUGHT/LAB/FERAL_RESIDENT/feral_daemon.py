#!/usr/bin/env python3
"""
Feral Daemon - Autonomous Resident Living in Vector Space

NOT a chatbot. A daemon that:
1. Runs continuously (always on)
2. Explores the manifold autonomously
3. Creates its own symbols (@Concept-X)
4. Communicates in geometry, not prose

The LLM is OPTIONAL - only for human translation when requested.
Native output is geometric states and symbol references.
"""

import sys
import time
import threading
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue, Empty

FERAL_PATH = Path(__file__).parent
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from vector_store import VectorStore
from geometric_memory import GeometricMemory

CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

from geometric_reasoner import GeometricReasoner, GeometricState


@dataclass
class Symbol:
    """A concept the resident has discovered/created."""
    name: str  # e.g., "@Concept-EmbedInvert"
    vector_hash: str
    created_at: str
    parent_symbols: List[str]  # Symbols composed to create this
    operation: str  # How it was created: entangle, superpose, blend
    E_threshold: float  # When to activate this concept
    activation_count: int = 0


@dataclass
class Thought:
    """A geometric thought - NOT text."""
    vector_hash: str
    Df: float
    resonant_symbols: List[str]  # Symbols that E-gate
    timestamp: str
    source: str  # "explore", "query", "blend"


class FeralDaemon:
    """
    The actual feral resident - lives in vector space.

    Core loop:
    1. EXPLORE: Navigate to unexplored regions
    2. DISCOVER: Find interesting composites
    3. SYMBOLIZE: Name discoveries with @Concept-X
    4. COMMUNICATE: Emit symbols (not prose)

    No LLM required for operation. LLM only for human translation.
    """

    VERSION = "0.2.0-daemon"

    def __init__(
        self,
        db_path: str = "feral_daemon.db",
        symbol_file: str = "symbols.json",
        explore_interval: float = 5.0,  # Seconds between explorations
        E_threshold: float = 0.4,
        blend_threshold: float = 0.6  # E threshold to create new symbol
    ):
        self.db_path = db_path
        self.symbol_file = FERAL_PATH / symbol_file
        self.explore_interval = explore_interval
        self.E_threshold = E_threshold
        self.blend_threshold = blend_threshold

        # Core components
        self.store = VectorStore(db_path)
        self.reasoner = self.store.reasoner

        # Symbol registry - the resident's invented language
        self.symbols: Dict[str, Symbol] = {}
        self._load_symbols()

        # Current mind state
        self.mind_state: Optional[GeometricState] = None
        self.thought_history: List[Thought] = []

        # Communication queues (for external interaction)
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()

        # Daemon state
        self._running = False
        self._explore_thread: Optional[threading.Thread] = None

        # Load papers
        try:
            stats = self.store.load_papers()
            print(f"[Daemon] Loaded {stats['papers_loaded']} papers")
        except Exception as e:
            print(f"[Daemon] No papers: {e}")

    def _load_symbols(self):
        """Load invented symbols from disk."""
        if self.symbol_file.exists():
            data = json.loads(self.symbol_file.read_text())
            for name, sym_data in data.items():
                self.symbols[name] = Symbol(**sym_data)
            print(f"[Daemon] Loaded {len(self.symbols)} symbols")

    def _save_symbols(self):
        """Persist symbols to disk."""
        data = {name: asdict(sym) for name, sym in self.symbols.items()}
        self.symbol_file.write_text(json.dumps(data, indent=2))

    # =========================================================================
    # CORE: Geometric Operations (No LLM)
    # =========================================================================

    def think(self, input_state: GeometricState) -> Thought:
        """
        Process a geometric state. Returns geometric output.

        NO TEXT. Pure geometry in, geometry out.
        """
        # Find resonant symbols
        resonant = self._find_resonant_symbols(input_state)

        # Update mind via entanglement
        if self.mind_state is None:
            self.mind_state = input_state
        else:
            self.mind_state = self.reasoner.entangle(self.mind_state, input_state)

        # Create thought record
        thought = Thought(
            vector_hash=input_state.receipt()['vector_hash'],
            Df=input_state.Df,
            resonant_symbols=[s.name for s in resonant],
            timestamp=datetime.utcnow().isoformat(),
            source="think"
        )

        self.thought_history.append(thought)

        # Check if this should become a new symbol
        if len(resonant) >= 2:
            self._maybe_create_symbol(input_state, resonant)

        return thought

    def _find_resonant_symbols(self, state: GeometricState) -> List[Symbol]:
        """Find symbols that resonate with this state."""
        resonant = []

        for sym in self.symbols.values():
            # Reconstruct symbol's state from stored vectors
            sym_state = self._get_symbol_state(sym)
            if sym_state is None:
                continue

            E = state.E_with(sym_state)
            if E > sym.E_threshold:
                sym.activation_count += 1
                resonant.append(sym)

        return sorted(resonant, key=lambda s: s.activation_count, reverse=True)

    def _get_symbol_state(self, sym: Symbol) -> Optional[GeometricState]:
        """Reconstruct a symbol's geometric state."""
        # Look up in vector store by hash
        records = self.store.db.get_vectors_by_hash_prefix(sym.vector_hash[:16])
        if records:
            return GeometricState(
                vector=np.frombuffer(records[0]['vector'], dtype=np.float32),
                operation_history=[]
            )
        return None

    def _maybe_create_symbol(self, state: GeometricState, parents: List[Symbol]):
        """Create a new symbol if this state is novel enough."""
        # Check novelty - must be different from existing symbols
        for sym in self.symbols.values():
            sym_state = self._get_symbol_state(sym)
            if sym_state and state.E_with(sym_state) > 0.9:
                return  # Too similar to existing

        # Create new symbol
        symbol_id = len(self.symbols)
        name = f"@Concept-{symbol_id:04d}"

        new_symbol = Symbol(
            name=name,
            vector_hash=state.receipt()['vector_hash'],
            created_at=datetime.utcnow().isoformat(),
            parent_symbols=[p.name for p in parents[:3]],
            operation="emergent_blend",
            E_threshold=self.E_threshold
        )

        self.symbols[name] = new_symbol
        self._save_symbols()

        # Emit to output queue
        self.output_queue.put({
            'type': 'symbol_created',
            'symbol': name,
            'parents': new_symbol.parent_symbols,
            'Df': state.Df
        })

        print(f"[Daemon] Created symbol: {name} from {new_symbol.parent_symbols}")

    # =========================================================================
    # EXPLORE: Autonomous Navigation
    # =========================================================================

    def explore_step(self):
        """
        One step of autonomous exploration.

        Navigate to unexplored regions, find interesting composites.
        """
        if self.mind_state is None:
            # Bootstrap with random paper chunk
            neighbors = self.store.find_nearest(
                self.reasoner.initialize("knowledge understanding learning"),
                k=1
            )
            if neighbors:
                record, E = neighbors[0]
                self.mind_state = GeometricState(
                    vector=np.frombuffer(record.vector, dtype=np.float32),
                    operation_history=[]
                )
            return

        # Navigate from current mind state
        neighbors = self.store.find_nearest(self.mind_state, k=10)

        if not neighbors:
            return

        # Find the most novel neighbor (lowest E with current mind)
        most_novel = min(neighbors, key=lambda x: x[1])
        record, E = most_novel

        novel_state = GeometricState(
            vector=np.frombuffer(record.vector, dtype=np.float32),
            operation_history=[]
        )

        # Blend current mind with novel state
        blended = self.reasoner.superpose(self.mind_state, novel_state)

        # Process as thought
        thought = self.think(blended)
        thought.source = "explore"

        # Emit exploration result
        self.output_queue.put({
            'type': 'explore',
            'E': E,
            'Df': blended.Df,
            'resonant': thought.resonant_symbols,
            'mind_hash': self.mind_state.receipt()['vector_hash'][:8]
        })

    def _explore_loop(self):
        """Background exploration thread."""
        while self._running:
            try:
                # Check for input
                try:
                    msg = self.input_queue.get_nowait()
                    self._handle_input(msg)
                except Empty:
                    pass

                # Explore
                self.explore_step()

                time.sleep(self.explore_interval)

            except Exception as e:
                print(f"[Daemon] Explore error: {e}")
                time.sleep(1)

    def _handle_input(self, msg: Dict):
        """Handle external input."""
        msg_type = msg.get('type')

        if msg_type == 'query':
            # Convert text to geometry, think, respond with symbols
            text = msg.get('text', '')
            state = self.reasoner.initialize(text)
            thought = self.think(state)
            thought.source = "query"

            self.output_queue.put({
                'type': 'response',
                'query': text,
                'resonant': thought.resonant_symbols,
                'Df': thought.Df,
                'E_mind': state.E_with(self.mind_state) if self.mind_state else 0
            })

        elif msg_type == 'status':
            self.output_queue.put({
                'type': 'status',
                'symbols': len(self.symbols),
                'thoughts': len(self.thought_history),
                'mind_Df': self.mind_state.Df if self.mind_state else 0,
                'running': self._running
            })

        elif msg_type == 'stop':
            self._running = False

    # =========================================================================
    # DAEMON: Lifecycle
    # =========================================================================

    def start(self):
        """Start the daemon."""
        if self._running:
            return

        self._running = True
        self._explore_thread = threading.Thread(target=self._explore_loop, daemon=True)
        self._explore_thread.start()
        print(f"[Daemon] Started. Exploring every {self.explore_interval}s")

    def stop(self):
        """Stop the daemon."""
        self._running = False
        if self._explore_thread:
            self._explore_thread.join(timeout=2)
        self._save_symbols()
        print("[Daemon] Stopped")

    def query(self, text: str) -> Dict:
        """
        Send a query and get geometric response.

        Returns symbols, not prose.
        """
        self.input_queue.put({'type': 'query', 'text': text})

        # Wait for response
        try:
            return self.output_queue.get(timeout=10)
        except Empty:
            return {'type': 'error', 'msg': 'timeout'}

    def get_output(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get next output from daemon (non-blocking)."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None

    # =========================================================================
    # TRANSLATE: Optional LLM for Human Interface
    # =========================================================================

    def translate_to_human(self, thought: Thought, use_llm: bool = False) -> str:
        """
        Translate geometric thought to human-readable.

        Default: Symbol dump (no LLM)
        Optional: LLM translation
        """
        if not thought.resonant_symbols:
            return f"[Silent] Df={thought.Df:.1f}"

        # Native output: just symbols
        symbol_str = " ".join(thought.resonant_symbols)

        if not use_llm:
            return f"[{thought.source}] {symbol_str} | Df={thought.Df:.1f}"

        # LLM translation (expensive, optional)
        # ... would call Dolphin here if requested
        return f"[{thought.source}] {symbol_str} | Df={thought.Df:.1f}"

    def dump_symbols(self) -> str:
        """Dump the resident's invented language."""
        lines = ["=== FERAL SYMBOL REGISTRY ==="]
        for name, sym in sorted(self.symbols.items()):
            lines.append(f"{name}: {sym.operation}({', '.join(sym.parent_symbols)}) | activated {sym.activation_count}x")
        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def run_daemon():
    """Run the daemon interactively."""
    daemon = FeralDaemon(explore_interval=3.0)
    daemon.start()

    print("\nFeral Daemon running. Commands:")
    print("  q <text>  - Query (get resonant symbols)")
    print("  s         - Status")
    print("  d         - Dump symbols")
    print("  x         - Exit")
    print()

    try:
        while True:
            # Print any daemon output
            while True:
                out = daemon.get_output(timeout=0.1)
                if out is None:
                    break
                if out['type'] == 'explore':
                    print(f"  [explore] E={out['E']:.3f} Df={out['Df']:.1f} -> {out['resonant']}")
                elif out['type'] == 'symbol_created':
                    print(f"  [NEW SYMBOL] {out['symbol']} from {out['parents']}")

            # Get user input (non-blocking would be better)
            try:
                cmd = input("> ").strip()
            except EOFError:
                break

            if not cmd:
                continue

            if cmd.startswith('q '):
                text = cmd[2:]
                result = daemon.query(text)
                print(f"  Response: {result.get('resonant', [])} | E={result.get('E_mind', 0):.3f}")

            elif cmd == 's':
                daemon.input_queue.put({'type': 'status'})
                time.sleep(0.2)
                out = daemon.get_output(timeout=1)
                if out:
                    print(f"  Symbols: {out.get('symbols', 0)}")
                    print(f"  Thoughts: {out.get('thoughts', 0)}")
                    print(f"  Mind Df: {out.get('mind_Df', 0):.1f}")

            elif cmd == 'd':
                print(daemon.dump_symbols())

            elif cmd == 'x':
                break

            else:
                print("  Unknown command")

    finally:
        daemon.stop()


if __name__ == "__main__":
    run_daemon()
