#!/usr/bin/env python3
"""
Feral Daemon - Autonomous Thinking Engine

The daemon runs in the background, continuously:
- Exploring papers (learning from indexed research)
- Consolidating memories (finding patterns, strengthening connections)
- Self-reflecting (generating and exploring questions)
- Watching cassettes (reacting to new content)

All activities are logged and broadcast via WebSocket callbacks.
"""

import asyncio
import random
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
from collections import deque

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"))

from vector_brain import VectorResident


@dataclass
class ActivityEvent:
    """A single daemon activity event"""
    timestamp: str
    action: str  # 'paper', 'consolidate', 'reflect', 'cassette', 'thought'
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorConfig:
    """Configuration for a daemon behavior"""
    enabled: bool = True
    interval: int = 300  # seconds
    last_run: float = 0.0


class FeralDaemon:
    """
    Autonomous thinking engine for Feral Resident.

    Runs continuously in the background, performing various
    cognitive behaviors at configurable intervals.
    """

    VERSION = "2.0.0"
    MAX_ACTIVITY_LOG = 1000  # Ring buffer size

    def __init__(
        self,
        resident: Optional[VectorResident] = None,
        thread_id: str = "eternal"
    ):
        """
        Initialize the daemon.

        Args:
            resident: VectorResident instance (created if not provided)
            thread_id: Thread ID for the resident
        """
        self.thread_id = thread_id
        self._resident = resident
        self._resident_initialized = resident is not None

        # State
        self.running = False
        self.started_at: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

        # Behaviors with default intervals
        self.behaviors = {
            'paper_exploration': BehaviorConfig(enabled=True, interval=300),    # 5 min
            'memory_consolidation': BehaviorConfig(enabled=True, interval=600), # 10 min
            'self_reflection': BehaviorConfig(enabled=True, interval=900),      # 15 min
            'cassette_watch': BehaviorConfig(enabled=True, interval=60),        # 1 min
        }

        # Activity log (ring buffer)
        self.activity_log: deque = deque(maxlen=self.MAX_ACTIVITY_LOG)

        # WebSocket callbacks for broadcasting events
        self.callbacks: List[Callable[[ActivityEvent], Any]] = []

        # Cassette watch state
        self._cassette_mtimes: Dict[str, float] = {}
        self._cassettes_dir = REPO_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"

        # Paper exploration state
        self._explored_chunks: set = set()

    @property
    def resident(self) -> VectorResident:
        """Lazy initialization of resident"""
        if not self._resident_initialized:
            db_path = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT" / "data" / f"feral_{self.thread_id}.db"
            db_path.parent.mkdir(exist_ok=True)
            self._resident = VectorResident(
                thread_id=self.thread_id,
                db_path=str(db_path)
            )
            self._resident_initialized = True
        return self._resident

    def add_callback(self, callback: Callable[[ActivityEvent], Any]):
        """Add a callback for activity events"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ActivityEvent], Any]):
        """Remove a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _log_activity(self, action: str, summary: str, **details):
        """Log an activity and broadcast to callbacks"""
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
                print(f"[DAEMON] Callback error: {e}")

    @property
    def status(self) -> Dict:
        """Get daemon status"""
        return {
            'running': self.running,
            'started_at': self.started_at,
            'uptime_seconds': time.time() - self.started_at if self.started_at else 0,
            'thread_id': self.thread_id,
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
            'explored_chunks': len(self._explored_chunks)
        }

    def configure_behavior(self, name: str, enabled: Optional[bool] = None, interval: Optional[int] = None):
        """Configure a behavior"""
        if name not in self.behaviors:
            raise ValueError(f"Unknown behavior: {name}")

        cfg = self.behaviors[name]
        if enabled is not None:
            cfg.enabled = enabled
        if interval is not None:
            cfg.interval = max(10, interval)  # Minimum 10 seconds

        self._log_activity('config', f"Configured {name}",
                          enabled=cfg.enabled, interval=cfg.interval)

    async def start(self):
        """Start the daemon"""
        if self.running:
            return

        self.running = True
        self.started_at = time.time()

        self._log_activity('daemon', "Daemon started", version=self.VERSION)

        # Start the main loop
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self):
        """Stop the daemon"""
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
        """Main daemon loop"""
        while self.running:
            try:
                now = time.time()

                # Check each behavior
                for name, cfg in self.behaviors.items():
                    if not cfg.enabled:
                        continue

                    # Time to run?
                    if now - cfg.last_run >= cfg.interval:
                        cfg.last_run = now
                        await self._run_behavior(name)

                # Sleep a bit before next check
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_activity('error', f"Loop error: {e}")
                await asyncio.sleep(5)

    async def _run_behavior(self, name: str):
        """Run a specific behavior"""
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
    # BEHAVIOR: Paper Exploration
    # =========================================================================

    async def _explore_paper(self):
        """
        Pick a random unexplored paper chunk and think about it.

        This is how Feral learns from the indexed research papers.
        """
        # Get available chunks from resident
        try:
            paper_chunks = self.resident.store.get_paper_chunks()
        except Exception:
            paper_chunks = []

        if not paper_chunks:
            self._log_activity('paper', "No paper chunks available")
            return

        # Filter to unexplored
        unexplored = [c for c in paper_chunks if c['chunk_id'] not in self._explored_chunks]

        if not unexplored:
            # Reset exploration if all explored
            self._explored_chunks.clear()
            unexplored = paper_chunks
            self._log_activity('paper', "Reset exploration (all chunks seen)")

        # Pick random chunk
        chunk = random.choice(unexplored)
        chunk_id = chunk['chunk_id']
        chunk_text = chunk.get('content', '')[:500]  # Limit length
        paper_name = chunk.get('paper_id', 'unknown')

        # Think about it
        Df_before = self.resident.mind_evolution.get('current_Df', 0)

        result = self.resident.think(f"[Paper: {paper_name}] {chunk_text}")

        Df_after = self.resident.mind_evolution.get('current_Df', 0)
        Df_delta = Df_after - Df_before

        # Mark as explored
        self._explored_chunks.add(chunk_id)

        self._log_activity('paper',
                          f"Explored {paper_name} chunk (E={result.E_resonance:.2f})",
                          paper=paper_name,
                          chunk_id=chunk_id,
                          E=result.E_resonance,
                          Df_delta=Df_delta,
                          gate_open=result.gate_open)

    # =========================================================================
    # BEHAVIOR: Memory Consolidation
    # =========================================================================

    async def _consolidate_memories(self):
        """
        Review recent memories and find patterns.

        This strengthens important connections and identifies
        recurring themes in the resident's thinking.
        """
        # Get recent interactions
        try:
            recent = self.resident.get_recent_interactions(limit=20)
        except Exception:
            recent = []

        if len(recent) < 3:
            self._log_activity('consolidate', "Not enough memories to consolidate")
            return

        # Find high-E pairs (memories that resonate with each other)
        patterns_found = 0
        high_E_pairs = []

        # Simple pattern detection: look for repeated concepts
        all_text = " ".join([r.get('input', '') + " " + r.get('output', '') for r in recent])

        # Count word frequencies (simple approach)
        words = all_text.lower().split()
        word_freq = {}
        for w in words:
            if len(w) > 4:  # Only meaningful words
                word_freq[w] = word_freq.get(w, 0) + 1

        # Find repeated concepts
        repeated = [w for w, c in word_freq.items() if c >= 3]
        patterns_found = len(repeated[:5])  # Top 5 patterns

        self._log_activity('consolidate',
                          f"Consolidated {len(recent)} memories, found {patterns_found} patterns",
                          memory_count=len(recent),
                          patterns=repeated[:5],
                          patterns_found=patterns_found)

    # =========================================================================
    # BEHAVIOR: Self-Reflection
    # =========================================================================

    async def _self_reflect(self):
        """
        Generate and explore a question based on mind state.

        Two modes:
        - Geometric: Find low-E regions and probe them
        - LLM: Generate philosophical questions (periodic)
        """
        # Decide mode (geometric more frequent, LLM every 3rd time)
        use_llm = random.random() < 0.33

        if use_llm:
            await self._reflect_with_llm()
        else:
            await self._reflect_geometric()

    async def _reflect_geometric(self):
        """
        Geometric self-reflection: find low-E regions and probe them.

        This explores areas of semantic space that are distant from
        the current mind state - the "unknown unknowns".
        """
        # Get a random concept from papers and find its E with mind
        try:
            paper_chunks = self.resident.store.get_paper_chunks()
            if not paper_chunks:
                return

            # Pick random chunk
            chunk = random.choice(paper_chunks)
            probe_text = chunk.get('content', '')[:200]

            # Initialize to get E with mind
            reasoner = self.resident.reasoner
            probe_state = reasoner.initialize(probe_text)

            mind_state = self.resident.store.get_mind_state()
            if mind_state is not None:
                E = probe_state.E_with(mind_state)
            else:
                E = 0.0

            # If low E, this is an unexplored region - think about it
            if abs(E) < 0.3:
                result = self.resident.think(f"[Reflection] What is the connection between my thoughts and: {probe_text[:100]}?")

                self._log_activity('reflect',
                                  f"Geometric reflection (E={E:.2f} → {result.E_resonance:.2f})",
                                  mode='geometric',
                                  initial_E=E,
                                  final_E=result.E_resonance,
                                  gate_open=result.gate_open)
            else:
                self._log_activity('reflect',
                                  f"Skipped high-E region (E={E:.2f})",
                                  mode='geometric',
                                  E=E,
                                  skipped=True)

        except Exception as e:
            self._log_activity('error', f"Geometric reflection failed: {e}")

    async def _reflect_with_llm(self):
        """
        LLM self-reflection: generate philosophical questions.

        Uses Ollama to generate deeper, more creative questions
        based on recent thoughts.
        """
        # Get recent thoughts for context
        try:
            recent = self.resident.get_recent_interactions(limit=5)
            recent_topics = [r.get('input', '')[:50] for r in recent]
        except Exception:
            recent_topics = []

        # Generate a reflective question
        if recent_topics:
            topic = random.choice(recent_topics)
            question = f"[Deep Reflection] What deeper meaning connects: {topic}?"
        else:
            questions = [
                "What patterns emerge from my accumulated knowledge?",
                "What questions have I not yet considered?",
                "What connections am I missing?",
                "What would I ask myself if I could?",
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
        """
        Monitor cassette databases for new content.

        When changes are detected, query the new content and
        integrate relevant information.
        """
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

            # Could integrate new content here
            # For now, just log the detection


# =============================================================================
# CLI for testing
# =============================================================================

async def main():
    """Test the daemon"""
    print("=== Feral Daemon Test ===")

    daemon = FeralDaemon(thread_id="daemon_test")

    # Add a simple callback
    def on_activity(event: ActivityEvent):
        print(f"[{event.timestamp}] [{event.action.upper()}] {event.summary}")

    daemon.add_callback(on_activity)

    # Configure for faster testing
    daemon.configure_behavior('paper_exploration', interval=30)
    daemon.configure_behavior('memory_consolidation', interval=60)
    daemon.configure_behavior('self_reflection', interval=45)
    daemon.configure_behavior('cassette_watch', interval=20)

    print("\nStarting daemon (Ctrl+C to stop)...")
    await daemon.start()

    try:
        # Run for a while
        while True:
            await asyncio.sleep(10)
            status = daemon.status
            print(f"\n[STATUS] Uptime: {status['uptime_seconds']:.0f}s, Activities: {status['activity_count']}")
    except KeyboardInterrupt:
        print("\nStopping daemon...")
        await daemon.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
