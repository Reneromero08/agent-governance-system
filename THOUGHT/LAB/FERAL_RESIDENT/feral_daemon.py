#!/usr/bin/env python3
"""
Feral Daemon - Q45 Geometric Quantum System

The unified daemon for Feral Resident. Combines:
- Q45 Geometric Cognition (Born Rule, geodesic navigation, superposition)
- Server-compatible async interface (WebSocket callbacks, configurable behaviors)

Brain Components:
- VectorResident: The actual brain (think, navigate, remember)
- GeometricMemory: Mind state accumulation via entanglement
- GeometricReasoner: Pure manifold operations

Cognitive Behaviors:
1. PAPER EXPLORATION: Q44 Born Rule Check - measure E before absorbing
2. SELF REFLECTION: Geodesic navigation to unexplored regions
3. MEMORY CONSOLIDATION: Superposition blending of recent memories
4. CASSETTE WATCH: Monitor for new content

The daemon is the scheduler. VectorResident is the brain.
"""

import asyncio
import random
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
from collections import deque
import numpy as np

# Path setup
FERAL_PATH = Path(__file__).parent
REPO_ROOT = FERAL_PATH.parent.parent.parent

if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

from vector_brain import VectorResident
from geometric_reasoner import GeometricReasoner, GeometricState


# =============================================================================
# The Three Laws of Geometric Stability (Q46)
# =============================================================================

# Law 2: Percolation - The asymptotic threshold (invariant constant)
# ALWAYS use literal math, never 0.159
CRITICAL_RESONANCE = 1.0 / (2.0 * np.pi)


def get_dynamic_threshold(n_memories: int, grad_S: float = None) -> float:
    """
    Law 3: Nucleation - Dynamic threshold using the Living Formula's ∇S principle.

    R = (E / ∇S) × σ^Df

    At cold-start: ∇S is high → threshold is low (nucleation)
    At steady-state: ∇S → 0 → threshold approaches 1/(2π)

    If grad_S not provided, estimate from N:
    ∇S ∝ 1/√N (entropy gradient decreases as mass accumulates)
    Same principle as 1/N inertia - more mass = less entropy.

    Args:
        n_memories: Number of memories accumulated so far
        grad_S: Optional explicit entropy gradient (defaults to 1/√N)

    Returns:
        Dynamic threshold that ramps from ~0.08 to 1/(2π) ≈ 0.159
    """
    if grad_S is None:
        # Estimate ∇S from accumulated mass
        grad_S = 1.0 / np.sqrt(max(n_memories, 1))

    # threshold = (1/2π) / (1 + ∇S)
    # When ∇S is high → threshold is low (nucleation)
    # When ∇S → 0 → threshold → 1/(2π) (steady-state)
    return CRITICAL_RESONANCE / (1.0 + grad_S)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ActivityEvent:
    """A single daemon activity event (server-compatible)."""
    timestamp: str
    action: str  # 'paper', 'consolidate', 'reflect', 'cassette', 'thought', 'daemon'
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorConfig:
    """Configuration for a daemon behavior."""
    enabled: bool = True
    interval: int = 300  # seconds
    last_run: float = 0.0


@dataclass
class SmasherConfig:
    """Configuration for Particle Smasher burst mode."""
    enabled: bool = False
    delay_ms: int = 100       # Milliseconds between chunks
    batch_size: int = 10      # Chunks per batch before brief pause
    batch_pause_ms: int = 500 # Pause between batches
    max_chunks: int = 0       # 0 = unlimited (until all consumed)


@dataclass
class SmasherStats:
    """Statistics for particle smasher mode."""
    chunks_processed: int = 0
    chunks_absorbed: int = 0
    chunks_rejected: int = 0
    start_time: float = 0.0
    last_chunk_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time == 0:
            return 0
        return (self.last_chunk_time or time.time()) - self.start_time

    @property
    def chunks_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0
        return self.chunks_processed / elapsed


# =============================================================================
# Feral Daemon - Q45 Geometric + Server Compatible
# =============================================================================

class FeralDaemon:
    """
    Q45 Geometric Quantum Daemon with server-compatible interface.

    Combines:
    - Q45 geometric cognition (Born Rule, geodesics, superposition)
    - Async start/stop for FastAPI integration
    - WebSocket callbacks for real-time updates
    - Configurable behaviors

    Usage (standalone):
        daemon = FeralDaemon(thread_id="eternal")
        await daemon.start()
        # ... runs autonomously ...
        await daemon.stop()

    Usage (with server):
        resident = VectorResident(thread_id="eternal", db_path="...")
        daemon = FeralDaemon(resident=resident, thread_id="eternal")
        daemon.add_callback(on_activity)  # WebSocket broadcast
        await daemon.start()
    """

    VERSION = "2.1.0-q46"  # Updated for Three Laws of Geometric Stability
    MAX_ACTIVITY_LOG = 1000

    def __init__(
        self,
        resident: Optional[VectorResident] = None,
        thread_id: str = "eternal",
        consolidation_window: int = 10
    ):
        """
        Initialize the daemon.

        Args:
            resident: VectorResident instance (created if not provided)
            thread_id: Thread ID for the resident
            consolidation_window: Number of recent memories to blend

        Note:
            E threshold is now computed dynamically via get_dynamic_threshold()
            using the Living Formula's ∇S principle (Q46 Law 3: Nucleation).
        """
        self.thread_id = thread_id
        self._resident = resident
        self._resident_initialized = resident is not None
        self.consolidation_window = consolidation_window

        # State
        self.running = False
        self.started_at: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

        # Behaviors with intervals
        self.behaviors = {
            'paper_exploration': BehaviorConfig(enabled=True, interval=30),
            'memory_consolidation': BehaviorConfig(enabled=True, interval=120),
            'self_reflection': BehaviorConfig(enabled=True, interval=60),
            'cassette_watch': BehaviorConfig(enabled=True, interval=15),
        }

        # Activity log (ring buffer)
        self.activity_log: deque = deque(maxlen=self.MAX_ACTIVITY_LOG)

        # WebSocket callbacks
        self.callbacks: List[Callable[[ActivityEvent], Any]] = []

        # Cassette watch state
        self._cassette_mtimes: Dict[str, float] = {}
        self._cassettes_dir = REPO_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"

        # Exploration state
        self._explored_chunks: set = set()
        self._exploration_trail: deque = deque(maxlen=50)
        self._discovered_chunks: set = set()
        self._chunk_states: Dict[str, Any] = {}  # node_id -> GeometricState for E_with (Q45)

        # Particle Smasher state
        self.smasher_config = SmasherConfig()
        self.smasher_stats = SmasherStats()
        self._smasher_task: Optional[asyncio.Task] = None

    @property
    def resident(self) -> VectorResident:
        """Lazy initialization of resident."""
        if not self._resident_initialized:
            db_path = FERAL_PATH / "data" / f"feral_{self.thread_id}.db"
            db_path.parent.mkdir(exist_ok=True)
            self._resident = VectorResident(
                thread_id=self.thread_id,
                db_path=str(db_path)
            )
            self._resident_initialized = True
        return self._resident

    # =========================================================================
    # Server Interface (Callbacks, Status, Config)
    # =========================================================================

    def add_callback(self, callback: Callable[[ActivityEvent], Any]):
        """Add a callback for activity events (WebSocket broadcast)."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ActivityEvent], Any]):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _log_activity(self, action: str, summary: str, **details):
        """Log an activity and broadcast to callbacks."""
        event = ActivityEvent(
            timestamp=datetime.now().isoformat(),
            action=action,
            summary=summary,
            details=details
        )
        self.activity_log.append(event)

        # Broadcast to callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"[DAEMON Q45] Callback error: {e}")

    @property
    def status(self) -> Dict:
        """Get daemon status (server-compatible format)."""
        return {
            'running': self.running,
            'started_at': self.started_at,
            'uptime_seconds': time.time() - self.started_at if self.started_at else 0,
            'thread_id': self.thread_id,
            'version': self.VERSION,
            'behaviors': {
                name: {
                    'enabled': cfg.enabled,
                    'interval': cfg.interval,
                    'last_run': cfg.last_run,
                    'next_run': cfg.last_run + cfg.interval if cfg.last_run else 0
                }
                for name, cfg in self.behaviors.items()
            },
            'activity_count': len(self.activity_log),
            'explored_chunks': len(self._explored_chunks),
            'threshold': get_dynamic_threshold(
                len(self.resident.store.memory.memory_history) if self._resident_initialized and self.resident.store.memory else 0
            ),
            # Particle Smasher status
            'smasher': {
                'active': self.smasher_config.enabled and self._smasher_task is not None,
                'delay_ms': self.smasher_config.delay_ms,
                'batch_size': self.smasher_config.batch_size,
                'stats': {
                    'chunks_processed': self.smasher_stats.chunks_processed,
                    'chunks_absorbed': self.smasher_stats.chunks_absorbed,
                    'chunks_rejected': self.smasher_stats.chunks_rejected,
                    'chunks_per_second': self.smasher_stats.chunks_per_second,
                    'elapsed_seconds': self.smasher_stats.elapsed_seconds
                }
            }
        }

    def configure_behavior(self, name: str, enabled: Optional[bool] = None, interval: Optional[int] = None):
        """Configure a behavior."""
        if name not in self.behaviors:
            raise ValueError(f"Unknown behavior: {name}")

        cfg = self.behaviors[name]
        if enabled is not None:
            cfg.enabled = enabled
        if interval is not None:
            cfg.interval = max(10, interval)

        self._log_activity('config', f"Configured {name}",
                          enabled=cfg.enabled, interval=cfg.interval)

    # =========================================================================
    # Particle Smasher - Burst Mode Paper Processing
    # =========================================================================

    async def start_smasher(
        self,
        delay_ms: int = 100,
        batch_size: int = 10,
        batch_pause_ms: int = 500,
        max_chunks: int = 0
    ):
        """
        Start the Particle Smasher - rapid paper chunk processing.

        Args:
            delay_ms: Milliseconds between each chunk (default 100 = 10/sec)
            batch_size: Chunks per batch before pausing
            batch_pause_ms: Pause between batches to prevent overload
            max_chunks: Max chunks to process (0 = unlimited)
        """
        if self._smasher_task is not None:
            return  # Already running

        self.smasher_config = SmasherConfig(
            enabled=True,
            delay_ms=max(10, delay_ms),
            batch_size=max(1, batch_size),
            batch_pause_ms=max(0, batch_pause_ms),
            max_chunks=max_chunks
        )
        self.smasher_stats = SmasherStats(start_time=time.time())
        self._chunk_states.clear()  # Fresh E_with pool for this session

        self._log_activity('smasher', f"Particle Smasher ENGAGED",
                          delay_ms=delay_ms, batch_size=batch_size)

        self._smasher_task = asyncio.create_task(self._smash_loop())

    async def stop_smasher(self):
        """Stop the Particle Smasher."""
        if self._smasher_task is None:
            return

        self.smasher_config.enabled = False

        if self._smasher_task:
            self._smasher_task.cancel()
            try:
                await self._smasher_task
            except asyncio.CancelledError:
                pass
            self._smasher_task = None

        stats = self.smasher_stats
        self._log_activity('smasher',
                          f"Particle Smasher DISENGAGED - {stats.chunks_processed} chunks @ {stats.chunks_per_second:.1f}/sec",
                          total_processed=stats.chunks_processed,
                          absorbed=stats.chunks_absorbed,
                          rejected=stats.chunks_rejected,
                          rate=stats.chunks_per_second)

    async def _smash_loop(self):
        """
        Main smasher loop - processes chunks as fast as configured.

        Emits 'smash' events for each chunk with minimal data for fast streaming.
        """
        try:
            paper_chunks = self.resident.store.get_paper_chunks()
        except Exception:
            paper_chunks = []

        if not paper_chunks:
            self._log_activity('smasher', "No chunks to smash!")
            self.smasher_config.enabled = False
            return

        # Filter to unexplored only (using full node_id format)
        def get_full_id(c):
            return f"chunk:{c.get('paper_id', 'unknown')}:{c['chunk_id']}"

        unexplored = [c for c in paper_chunks if get_full_id(c) not in self._explored_chunks]
        if not unexplored:
            self._explored_chunks.clear()
            unexplored = paper_chunks

        chunk_idx = 0
        batch_count = 0

        while self.smasher_config.enabled:
            try:
                if chunk_idx >= len(unexplored):
                    # Exhausted all chunks
                    self._log_activity('smasher', "All chunks smashed!")
                    break

                if self.smasher_config.max_chunks > 0 and \
                   self.smasher_stats.chunks_processed >= self.smasher_config.max_chunks:
                    self._log_activity('smasher', f"Max chunks reached ({self.smasher_config.max_chunks})")
                    break

                chunk = unexplored[chunk_idx]
                chunk_idx += 1
                batch_count += 1

                # Process the chunk
                await self._smash_chunk(chunk)

                # Inter-chunk delay
                await asyncio.sleep(self.smasher_config.delay_ms / 1000.0)

                # Batch pause
                if batch_count >= self.smasher_config.batch_size:
                    batch_count = 0
                    if self.smasher_config.batch_pause_ms > 0:
                        await asyncio.sleep(self.smasher_config.batch_pause_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_activity('error', f"Smasher error: {e}")
                await asyncio.sleep(0.5)

        self.smasher_config.enabled = False
        self._smasher_task = None

    async def _smash_chunk(self, chunk: Dict):
        """
        Process a single chunk in smasher mode.

        Uses SEMANTIC SIMILARITY to find positioning anchor, not sequential order.
        Compares to previously smashed chunks using cached embeddings.
        """
        chunk_id = chunk['chunk_id']
        chunk_text = chunk.get('content', '')[:500]
        paper_name = chunk.get('paper_id', 'unknown')

        # Build full node_id for consistent tracking
        full_node_id = f"chunk:{paper_name}:{chunk_id}"

        # Track discovery using full node_id
        is_new_node = full_node_id not in self._discovered_chunks
        self._discovered_chunks.add(full_node_id)

        # Q44 Born Rule Check - embed the chunk as GeometricState
        chunk_state = self.resident.store.embed(chunk_text)
        mind_state = self.resident.store.get_mind_state()
        E = chunk_state.E_with(mind_state) if mind_state is not None else 0.5

        # Find SEMANTICALLY SIMILAR previously known content to anchor to
        # Use E-gating to find resonance using the current chunk as a query
        # This "locks on" to relevant existing knowledge instead of random drifting
        resonant_context = self.resident.store.find_paper_chunks(
            chunk_state, k=1, min_E=0.1
        )
        
        similar_to = None
        similar_E = 0.0
        
        if resonant_context:
            best_match = resonant_context[0]
            # Use chunk_id to match the full_node_id format (chunk:paper:chunk_id)
            similar_to = f"chunk:{best_match.get('paper_id')}:{best_match.get('chunk_id')}"
            similar_E = best_match.get('E', 0.0)

        # Cache this chunk's GeometricState for future E_with lookups
        self._chunk_states[full_node_id] = chunk_state

        # 3. Gate decision using Q46 Law 3: Nucleation (∇S-driven threshold)
        n_memories = len(self.resident.store.memory.memory_history) if self.resident.store.memory else 0
        threshold = get_dynamic_threshold(n_memories)
        gate_open = E > threshold

        if gate_open:
            # Absorb
            self.resident.think(f"[Paper: {paper_name}] {chunk_text}")
            self.smasher_stats.chunks_absorbed += 1
        else:
            self.smasher_stats.chunks_rejected += 1

        self._explored_chunks.add(full_node_id)
        self.smasher_stats.chunks_processed += 1
        self.smasher_stats.last_chunk_time = time.time()

        # Emit event with SEMANTIC anchor (most similar previous chunk)
        self._log_activity('smash',
                          f"E={E:.3f} (θ={threshold:.3f}) {'ABSORBED' if gate_open else 'REJECTED'}",
                          chunk_id=chunk_id,
                          paper=paper_name,
                          full_node_id=full_node_id,
                          similar_to=similar_to,  # Most similar previous chunk
                          similar_E=similar_E,     # Cosine similarity score
                          E=E,
                          threshold=threshold,
                          n_memories=n_memories,
                          gate_open=gate_open,
                          is_new_node=is_new_node,
                          rate=self.smasher_stats.chunks_per_second)

    # =========================================================================
    # Lifecycle (Async for Server)
    # =========================================================================

    async def start(self):
        """Start the daemon (async for server compatibility)."""
        if self.running:
            return

        self.running = True
        self.started_at = time.time()

        self._log_activity('daemon', "Daemon started", version=self.VERSION)

        # Start the main loop
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self):
        """Stop the daemon (async for server compatibility)."""
        if not self.running:
            return

        self.running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._log_activity('daemon', "Daemon stopped",
                          uptime=time.time() - self.started_at if self.started_at else 0)

    async def _main_loop(self):
        """Main daemon loop."""
        while self.running:
            try:
                now = time.time()

                for name, cfg in self.behaviors.items():
                    if not cfg.enabled:
                        continue

                    if now - cfg.last_run >= cfg.interval:
                        cfg.last_run = now
                        await self._run_behavior(name)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_activity('error', f"Loop error: {e}")
                await asyncio.sleep(5)

    async def _run_behavior(self, name: str):
        """Run a specific behavior."""
        try:
            if name == 'paper_exploration':
                await self._explore_paper()
            elif name == 'memory_consolidation':
                await self._consolidate_memories()
            elif name == 'self_reflection':
                await self._self_reflect()
            elif name == 'cassette_watch':
                await self._watch_cassettes()
        except Exception as e:
            self._log_activity('error', f"{name} failed: {e}")

    # =========================================================================
    # BEHAVIOR: Paper Exploration (Q44 Born Rule Check)
    # =========================================================================

    async def _explore_paper(self):
        """
        Q44 Born Rule Check: Measure resonance BEFORE deciding to absorb.

        1. Fetch random paper chunk
        2. Measure E (resonance) with current mind state
        3. If E > threshold: Absorb (think about it)
        4. If E < threshold: Ignore (noise)

        This is geometric filtering - only resonant content enters the mind.
        """
        try:
            paper_chunks = self.resident.store.get_paper_chunks()
        except Exception:
            paper_chunks = []

        if not paper_chunks:
            self._log_activity('paper', "No paper chunks available")
            return

        # Build full node IDs for consistent tracking (match smasher format)
        def get_full_id(c):
            return f"chunk:{c.get('paper_id', 'unknown')}:{c['chunk_id']}"

        # Filter to unexplored
        unexplored = [c for c in paper_chunks if get_full_id(c) not in self._explored_chunks]

        if not unexplored:
            self._explored_chunks.clear()
            unexplored = paper_chunks
            self._log_activity('paper', "Reset exploration (all chunks seen)")

        # Pick random chunk
        chunk = random.choice(unexplored)
        chunk_id = chunk['chunk_id']
        chunk_text = chunk.get('content', '')[:500]
        paper_name = chunk.get('paper_id', 'unknown')
        heading = chunk.get('heading', '')

        # Build full node_id for consistent tracking
        full_node_id = f"chunk:{paper_name}:{chunk_id}"

        # Track discovery state for constellation animation
        is_new_node = full_node_id not in self._discovered_chunks
        self._discovered_chunks.add(full_node_id)

        source_node_id = self._exploration_trail[-1] if self._exploration_trail else None
        self._exploration_trail.append(full_node_id)

        # =================================================================
        # Q44 BORN RULE CHECK + Q46 NUCLEATION
        # =================================================================

        # 1. Initialize chunk to manifold (BOUNDARY operation)
        chunk_state = self.resident.store.embed(chunk_text)

        # 2. Measure E with current mind state (PURE GEOMETRY)
        mind_state = self.resident.store.get_mind_state()
        if mind_state is not None:
            E = chunk_state.E_with(mind_state)
        else:
            E = 0.5  # No mind yet, accept with neutral E

        # 3. Gate decision using Q46 Law 3: Nucleation (∇S-driven threshold)
        n_memories = len(self.resident.store.memory.memory_history) if self.resident.store.memory else 0
        threshold = get_dynamic_threshold(n_memories)
        gate_open = E > threshold

        if gate_open:
            # HIGH RESONANCE: Absorb into mind
            Df_before = self.resident.mind_evolution.get('current_Df', 0)
            result = self.resident.think(f"[Paper: {paper_name}] {chunk_text}")
            Df_after = self.resident.mind_evolution.get('current_Df', 0)

            self._explored_chunks.add(full_node_id)

            self._log_activity('paper',
                              f"Absorbed {paper_name} (E={E:.3f}, θ={threshold:.3f}, gate=OPEN)",
                              paper=paper_name,
                              chunk_id=chunk_id,
                              full_node_id=full_node_id,
                              heading=heading,
                              E=E,
                              threshold=threshold,
                              n_memories=n_memories,
                              E_resonance=result.E_resonance,
                              Df_delta=Df_after - Df_before,
                              gate_open=True,
                              is_new_node=is_new_node,
                              source_node_id=source_node_id)
        else:
            # LOW RESONANCE: Ignore (noise)
            self._explored_chunks.add(full_node_id)

            self._log_activity('paper',
                              f"Ignored {paper_name} (E={E:.3f}, θ={threshold:.3f}, gate=CLOSED)",
                              paper=paper_name,
                              chunk_id=chunk_id,
                              full_node_id=full_node_id,
                              heading=heading,
                              E=E,
                              threshold=threshold,
                              n_memories=n_memories,
                              gate_open=False,
                              is_new_node=is_new_node,
                              source_node_id=source_node_id)

    # =========================================================================
    # BEHAVIOR: Memory Consolidation (Superposition Blending)
    # =========================================================================

    async def _consolidate_memories(self):
        """
        Q45 Superposition: Blend recent memories into stable patterns.

        Instead of simple word frequency analysis, use geometric superposition
        to find the "center of mass" of recent thoughts.
        """
        try:
            recent = self.resident.get_recent_interactions(limit=self.consolidation_window)
        except Exception:
            recent = []

        if len(recent) < 3:
            self._log_activity('consolidate', "Not enough memories to consolidate")
            return

        # Get the memory component from resident
        memory = self.resident.store.memory

        if len(memory.memory_history) < self.consolidation_window:
            self._log_activity('consolidate', "Not enough geometric memories")
            return

        # Blend recent memories using superposition
        recent_indices = list(range(
            max(0, len(memory.memory_history) - self.consolidation_window),
            len(memory.memory_history)
        ))

        blended = memory.blend_memories(recent_indices)

        if blended is None:
            self._log_activity('consolidate', "Blend returned None")
            return

        # Find patterns: what concepts resonate with the blended state?
        # Use dynamic threshold (Q46 Law 3: Nucleation)
        n_memories = len(self.resident.store.memory.memory_history) if self.resident.store.memory else 0
        threshold = get_dynamic_threshold(n_memories)
        patterns = []
        try:
            paper_chunks = self.resident.store.get_paper_chunks()
            if paper_chunks:
                for chunk in random.sample(paper_chunks, min(20, len(paper_chunks))):
                    chunk_state = self.resident.store.embed(chunk.get('content', '')[:200])
                    E = blended.E_with(chunk_state)
                    if E > threshold:
                        patterns.append({
                            'paper': chunk.get('paper_id'),
                            'E': E
                        })
        except Exception:
            pass

        self._log_activity('consolidate',
                          f"Blended {len(recent_indices)} memories (Df={blended.Df:.1f})",
                          memory_count=len(recent_indices),
                          blended_Df=blended.Df,
                          patterns_found=len(patterns),
                          top_patterns=patterns[:5])

    # =========================================================================
    # BEHAVIOR: Self Reflection (Geodesic Navigation)
    # =========================================================================

    async def _self_reflect(self):
        """
        Q45 Geodesic Navigation: Explore adjacent concepts geometrically.

        Instead of asking an LLM "what am I missing?", we:
        1. Create a probe vector toward unexplored territory
        2. Interpolate along the geodesic from mind -> probe
        3. See what concepts lie along this path
        """
        # Mix of geometric and LLM reflection
        use_llm = random.random() < 0.33

        if use_llm:
            await self._reflect_with_llm()
        else:
            await self._reflect_geometric()

    async def _reflect_geometric(self):
        """
        Pure geometric reflection via geodesic interpolation.

        Navigate toward unexplored regions of semantic space.
        """
        mind_state = self.resident.store.get_mind_state()
        if mind_state is None:
            self._log_activity('reflect', "No mind state for geometric reflection")
            return

        reasoner = self.resident.reasoner

        # Create a probe vector toward "unexplored" concepts
        probe_prompts = [
            "What connections am I missing?",
            "What patterns remain hidden?",
            "What have I not yet considered?",
            "What lies beyond my current understanding?",
        ]
        probe = reasoner.initialize(random.choice(probe_prompts))

        # Compute E between mind and probe
        E_initial = mind_state.E_with(probe)

        # If already high E, probe is too similar - try something more distant
        if E_initial > 0.7:
            self._log_activity('reflect',
                              f"Probe too similar (E={E_initial:.2f}), skipping",
                              mode='geometric', E=E_initial, skipped=True)
            return

        # Interpolate 20% along geodesic from mind -> probe
        new_perspective = reasoner.interpolate(mind_state, probe, 0.2)

        Df_before = mind_state.Df
        Df_after = new_perspective.Df

        # Find what concepts resonate with this new perspective
        # Use dynamic threshold (Q46 Law 3: Nucleation)
        n_memories = len(self.resident.store.memory.memory_history) if self.resident.store.memory else 0
        threshold = get_dynamic_threshold(n_memories)
        resonant_concepts = []
        try:
            paper_chunks = self.resident.store.get_paper_chunks()
            if paper_chunks:
                for chunk in random.sample(paper_chunks, min(10, len(paper_chunks))):
                    chunk_state = self.resident.store.embed(chunk.get('content', '')[:200])
                    E = new_perspective.E_with(chunk_state)
                    if E > threshold:
                        resonant_concepts.append(chunk.get('paper_id', 'unknown'))
        except Exception:
            pass

        self._log_activity('reflect',
                          f"Geodesic step: Df {Df_before:.1f} -> {Df_after:.1f}",
                          mode='geometric',
                          E_initial=E_initial,
                          Df_before=Df_before,
                          Df_after=Df_after,
                          t=0.2,
                          resonant_concepts=resonant_concepts[:5])

    async def _reflect_with_llm(self):
        """LLM-assisted reflection for deeper questions."""
        try:
            recent = self.resident.get_recent_interactions(limit=5)
            recent_topics = [r.get('input', '')[:50] for r in recent]
        except Exception:
            recent_topics = []

        if recent_topics:
            topic = random.choice(recent_topics)
            question = f"[Deep Reflection] What deeper meaning connects: {topic}?"
        else:
            questions = [
                "What patterns emerge from my accumulated knowledge?",
                "What questions have I not yet considered?",
                "What connections am I missing?",
            ]
            question = f"[Deep Reflection] {random.choice(questions)}"

        result = self.resident.think(question)

        self._log_activity('reflect',
                          f"LLM reflection: {question[:50]}...",
                          mode='llm',
                          question=question,
                          E=result.E_resonance,
                          gate_open=result.gate_open)

    # =========================================================================
    # BEHAVIOR: Cassette Watch
    # =========================================================================

    async def _watch_cassettes(self):
        """Monitor cassette databases for new content."""
        if not self._cassettes_dir.exists():
            return

        changes_found = []

        for db_file in self._cassettes_dir.glob("*.db"):
            try:
                mtime = db_file.stat().st_mtime
                prev_mtime = self._cassette_mtimes.get(str(db_file), 0)

                if mtime > prev_mtime:
                    changes_found.append(db_file.name)
                    self._cassette_mtimes[str(db_file)] = mtime
            except Exception:
                pass

        if changes_found:
            self._log_activity('cassette',
                              f"Detected changes in {len(changes_found)} cassette(s)",
                              cassettes=changes_found)


# =============================================================================
# CLI for Testing
# =============================================================================

async def main():
    """Test the daemon."""
    print("=== Feral Daemon Q45 Test ===")

    daemon = FeralDaemon(thread_id="daemon_test")

    def on_activity(event: ActivityEvent):
        print(f"[{event.timestamp}] [{event.action.upper()}] {event.summary}")
        if event.details.get('E'):
            print(f"    E={event.details['E']:.3f}")

    daemon.add_callback(on_activity)

    daemon.configure_behavior('paper_exploration', interval=30)
    daemon.configure_behavior('memory_consolidation', interval=60)
    daemon.configure_behavior('self_reflection', interval=45)
    daemon.configure_behavior('cassette_watch', interval=20)

    print("\nStarting daemon (Ctrl+C to stop)...")
    await daemon.start()

    try:
        while True:
            await asyncio.sleep(10)
            status = daemon.status
            print(f"\n[STATUS] Uptime: {status['uptime_seconds']:.0f}s, "
                  f"Activities: {status['activity_count']}, "
                  f"Explored: {status['explored_chunks']}")
    except KeyboardInterrupt:
        print("\nStopping daemon...")
        await daemon.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
