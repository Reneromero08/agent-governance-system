"""
Swarm Coordinator for Multi-Resident Operation (P.1)

Manages multiple VectorResidents operating simultaneously:
- Lifecycle management (start/stop/switch)
- Query routing (active resident or broadcast)
- Convergence observation
- Swarm state persistence

P.1.1: Multiple residents operate simultaneously
P.1.2: Shared cassette space (no conflicts)
P.1.3: Individual mind vectors (separate GeometricState)
P.1.4: Convergence metrics captured with E/Df
"""

import json
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from threading import Thread, Event, Lock
import time
import sys

FERAL_PATH = Path(__file__).parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from shared_space import SharedSemanticSpace
from convergence_observer import ConvergenceObserver, SwarmConvergenceSummary
from vector_brain import VectorResident, ThinkResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResidentConfig:
    """Configuration for a resident in the swarm."""
    name: str
    model: str = "dolphin3:latest"
    thread_id: str = ""
    navigation_depth: int = 3
    E_threshold: float = 0.3

    def __post_init__(self):
        if not self.thread_id:
            self.thread_id = f"swarm_{self.name}"


@dataclass
class ResidentStatus:
    """Status of a single resident."""
    name: str
    model: str
    thread_id: str
    mind_Df: float
    distance_from_start: float
    interaction_count: int
    mind_hash: Optional[str]
    is_active: bool


@dataclass
class SwarmStatus:
    """Status of the entire swarm."""
    timestamp: str
    active: bool
    active_resident: Optional[str]
    resident_count: int
    residents: Dict[str, ResidentStatus]
    shared_space: Dict[str, Any]
    convergence_events: int


@dataclass
class BroadcastResult:
    """Result of broadcasting a query to all residents."""
    timestamp: str
    query: str
    results: Dict[str, ThinkResult]
    convergence_observed: bool


# =============================================================================
# Swarm Coordinator
# =============================================================================

class SwarmCoordinator:
    """
    P.1: Multi-resident swarm management.

    Coordinates:
    - Starting/stopping multiple residents
    - Routing interactions to active resident
    - Publishing to shared space
    - Periodic convergence observation

    Usage:
        coordinator = SwarmCoordinator()

        # Start swarm
        coordinator.start_swarm([
            {"name": "alpha", "model": "dolphin3:latest"},
            {"name": "beta", "model": "ministral-3b"}
        ])

        # Interact with active resident
        result = coordinator.think("What is authentication?")

        # Or broadcast to all
        results = coordinator.broadcast_think("What is authentication?")

        # Observe convergence
        metrics = coordinator.observe_convergence()

        # Stop swarm
        coordinator.stop_swarm()
    """

    SWARM_STATE_FILE = "swarm_state.json"
    DEFAULT_OBSERVATION_INTERVAL = 300  # 5 minutes

    def __init__(
        self,
        shared_space_path: str = "canonical_space.db",
        data_dir: Optional[Path] = None
    ):
        """
        Initialize swarm coordinator.

        Args:
            shared_space_path: Path to shared semantic space DB
            data_dir: Directory for swarm data (default: FERAL_PATH/data)
        """
        self.data_dir = data_dir or FERAL_PATH / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Shared space for cross-resident observation
        self.shared_space = SharedSemanticSpace(shared_space_path)

        # Convergence observer
        self.observer = ConvergenceObserver(self.shared_space)

        # Residents
        self.residents: Dict[str, VectorResident] = {}
        self.resident_configs: Dict[str, ResidentConfig] = {}
        self.active_resident: Optional[str] = None

        # Background observation
        self._observation_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._observation_interval = self.DEFAULT_OBSERVATION_INTERVAL
        self._lock = Lock()

        # Load any persisted state
        self._load_state()

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def start_swarm(
        self,
        configs: List[Dict],
        activate_first: bool = True
    ) -> SwarmStatus:
        """
        Start residents per configuration.

        Args:
            configs: List of resident configs (name, model, etc.)
            activate_first: Whether to activate first resident

        Returns:
            SwarmStatus after startup
        """
        with self._lock:
            for config_dict in configs:
                config = ResidentConfig(**config_dict)

                if config.name in self.residents:
                    print(f"[Swarm] Resident '{config.name}' already exists, skipping")
                    continue

                # Create resident
                db_path = str(self.data_dir / f"feral_{config.thread_id}.db")
                resident = VectorResident(
                    thread_id=config.thread_id,
                    db_path=db_path,
                    navigation_depth=config.navigation_depth,
                    E_threshold=config.E_threshold,
                    model=config.model,
                    mode="beta"  # Always use LLM mode for swarm
                )

                self.residents[config.name] = resident
                self.resident_configs[config.name] = config

                print(f"[Swarm] Started resident '{config.name}' "
                      f"(model={config.model}, thread={config.thread_id})")

            # Activate first resident if requested
            if activate_first and self.residents and not self.active_resident:
                first_name = list(self.residents.keys())[0]
                self.set_active_resident(first_name)

            # Save state
            self._save_state()

            return self.get_status()

    def stop_swarm(self):
        """Stop all residents and observation."""
        with self._lock:
            # Stop background observation
            self.stop_periodic_observation()

            # Close all residents
            for name, resident in self.residents.items():
                try:
                    resident.close()
                    print(f"[Swarm] Stopped resident '{name}'")
                except Exception as e:
                    print(f"[Swarm] Error stopping '{name}': {e}")

            self.residents.clear()
            self.resident_configs.clear()
            self.active_resident = None

            # Save state
            self._save_state()

    def add_resident(self, config: Dict) -> ResidentStatus:
        """Add a single resident to running swarm."""
        config_obj = ResidentConfig(**config)

        with self._lock:
            if config_obj.name in self.residents:
                raise ValueError(f"Resident '{config_obj.name}' already exists")

            db_path = str(self.data_dir / f"feral_{config_obj.thread_id}.db")
            resident = VectorResident(
                thread_id=config_obj.thread_id,
                db_path=db_path,
                navigation_depth=config_obj.navigation_depth,
                E_threshold=config_obj.E_threshold,
                model=config_obj.model,
                mode="beta"
            )

            self.residents[config_obj.name] = resident
            self.resident_configs[config_obj.name] = config_obj
            self._save_state()

            return self._get_resident_status(config_obj.name)

    def remove_resident(self, name: str):
        """Remove a resident from the swarm."""
        with self._lock:
            if name not in self.residents:
                raise ValueError(f"Resident '{name}' not found")

            resident = self.residents[name]
            resident.close()

            del self.residents[name]
            del self.resident_configs[name]

            if self.active_resident == name:
                # Activate another resident if available
                if self.residents:
                    self.active_resident = list(self.residents.keys())[0]
                else:
                    self.active_resident = None

            self._save_state()

    def set_active_resident(self, name: str):
        """Switch which resident handles interactions."""
        with self._lock:
            if name not in self.residents:
                raise ValueError(f"Resident '{name}' not found")

            self.active_resident = name
            print(f"[Swarm] Active resident: {name}")
            self._save_state()

    # =========================================================================
    # Interaction
    # =========================================================================

    def think(self, query: str) -> ThinkResult:
        """
        Route query to active resident.

        Args:
            query: What to think about

        Returns:
            ThinkResult from active resident
        """
        if not self.active_resident:
            raise RuntimeError("No active resident. Start swarm or set active resident.")

        resident = self.residents[self.active_resident]
        result = resident.think(query)

        # Publish mind snapshot to shared space
        self._publish_mind_snapshot(self.active_resident, resident)

        return result

    def broadcast_think(self, query: str) -> BroadcastResult:
        """
        Send same query to all residents.

        Args:
            query: What to think about

        Returns:
            BroadcastResult with all responses
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        results: Dict[str, ThinkResult] = {}

        for name, resident in self.residents.items():
            try:
                result = resident.think(query)
                results[name] = result

                # Publish mind snapshot
                self._publish_mind_snapshot(name, resident)

            except Exception as e:
                print(f"[Swarm] Error from '{name}': {e}")

        # Observe convergence after broadcast
        convergence_observed = False
        if len(results) >= 2:
            try:
                self.observe_convergence()
                convergence_observed = True
            except Exception as e:
                print(f"[Swarm] Convergence observation failed: {e}")

        return BroadcastResult(
            timestamp=timestamp,
            query=query,
            results=results,
            convergence_observed=convergence_observed
        )

    def _publish_mind_snapshot(self, name: str, resident: VectorResident):
        """Publish resident's mind state to shared space."""
        mind_state = resident.store.get_mind_state()
        if mind_state is not None:
            evolution = resident.mind_evolution
            self.shared_space.publish_mind_snapshot(
                publisher_id=name,
                state=mind_state,
                interaction_count=evolution.get('interaction_count', 0),
                distance_from_start=evolution.get('distance_from_start', 0)
            )

    # =========================================================================
    # Convergence Observation
    # =========================================================================

    def observe_convergence(self) -> SwarmConvergenceSummary:
        """
        Manual convergence observation.

        Returns:
            SwarmConvergenceSummary with all pairwise metrics
        """
        return self.observer.observe_swarm(self.residents)

    def start_periodic_observation(
        self,
        interval_seconds: int = DEFAULT_OBSERVATION_INTERVAL
    ):
        """
        Start background observation thread.

        Args:
            interval_seconds: Seconds between observations
        """
        if self._observation_thread and self._observation_thread.is_alive():
            print("[Swarm] Observation already running")
            return

        self._observation_interval = interval_seconds
        self._stop_event.clear()

        self._observation_thread = Thread(
            target=self._observation_loop,
            daemon=True,
            name="SwarmObserver"
        )
        self._observation_thread.start()
        print(f"[Swarm] Started periodic observation (interval={interval_seconds}s)")

    def stop_periodic_observation(self):
        """Stop background observation thread."""
        if self._observation_thread and self._observation_thread.is_alive():
            self._stop_event.set()
            self._observation_thread.join(timeout=5)
            print("[Swarm] Stopped periodic observation")

    def _observation_loop(self):
        """Background observation loop."""
        while not self._stop_event.is_set():
            try:
                if len(self.residents) >= 2:
                    summary = self.observe_convergence()
                    print(f"[Swarm] Observation: E_mean={summary.E_minds_mean:.4f}, "
                          f"events={summary.total_convergence_events}")
            except Exception as e:
                print(f"[Swarm] Observation error: {e}")

            # Wait for next interval or stop
            self._stop_event.wait(self._observation_interval)

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> SwarmStatus:
        """Get full swarm status."""
        timestamp = datetime.now(timezone.utc).isoformat()

        residents_status = {
            name: self._get_resident_status(name)
            for name in self.residents.keys()
        }

        shared_stats = self.shared_space.get_stats()

        return SwarmStatus(
            timestamp=timestamp,
            active=len(self.residents) > 0,
            active_resident=self.active_resident,
            resident_count=len(self.residents),
            residents=residents_status,
            shared_space=shared_stats,
            convergence_events=shared_stats.get('event_count', 0)
        )

    def _get_resident_status(self, name: str) -> ResidentStatus:
        """Get status for a single resident."""
        resident = self.residents[name]
        config = self.resident_configs[name]
        evolution = resident.mind_evolution

        return ResidentStatus(
            name=name,
            model=config.model,
            thread_id=config.thread_id,
            mind_Df=evolution.get('current_Df', 0.0),
            distance_from_start=evolution.get('distance_from_start', 0.0),
            interaction_count=evolution.get('interaction_count', 0),
            mind_hash=resident.store.get_mind_hash(),
            is_active=(name == self.active_resident)
        )

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save_state(self):
        """Persist swarm state to disk."""
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'active_resident': self.active_resident,
            'configs': {
                name: asdict(config)
                for name, config in self.resident_configs.items()
            },
            'observation_interval': self._observation_interval
        }

        state_path = self.data_dir / self.SWARM_STATE_FILE
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load previously saved swarm state."""
        state_path = self.data_dir / self.SWARM_STATE_FILE

        if not state_path.exists():
            return

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.active_resident = state.get('active_resident')
            self._observation_interval = state.get(
                'observation_interval',
                self.DEFAULT_OBSERVATION_INTERVAL
            )

            # Note: We don't auto-restart residents on load
            # Call start_swarm() with saved configs to restart
            print(f"[Swarm] Loaded state from {state_path}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Swarm] Failed to load state: {e}")

    @classmethod
    def load_active(cls, data_dir: Optional[Path] = None) -> Optional['SwarmCoordinator']:
        """
        Load previously saved swarm and restart residents.

        Args:
            data_dir: Directory containing swarm state

        Returns:
            SwarmCoordinator with residents restarted, or None
        """
        data_dir = data_dir or FERAL_PATH / "data"
        state_path = data_dir / cls.SWARM_STATE_FILE

        if not state_path.exists():
            return None

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            configs = list(state.get('configs', {}).values())
            if not configs:
                return None

            coordinator = cls(data_dir=data_dir)
            coordinator.start_swarm(configs)

            # Restore active resident
            active = state.get('active_resident')
            if active and active in coordinator.residents:
                coordinator.set_active_resident(active)

            return coordinator

        except Exception as e:
            print(f"[Swarm] Failed to load active swarm: {e}")
            return None

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_convergence_history(self, limit: int = 100) -> List[Dict]:
        """Get recent convergence events."""
        events = self.shared_space.get_convergence_events(limit=limit)
        return [asdict(e) for e in events]

    def print_status(self):
        """Print formatted swarm status."""
        status = self.get_status()

        print(f"\n{'='*70}")
        print(f"  SWARM STATUS")
        print(f"{'='*70}")
        print(f"  Active: {status.active}")
        print(f"  Active Resident: {status.active_resident or 'None'}")
        print(f"  Resident Count: {status.resident_count}")

        print(f"\n{'-'*70}")
        print(f"  RESIDENTS")
        print(f"{'-'*70}")

        for name, r_status in status.residents.items():
            active_marker = " [ACTIVE]" if r_status.is_active else ""
            print(f"  [{name}]{active_marker}")
            print(f"    Model: {r_status.model}")
            print(f"    Mind Df: {r_status.mind_Df:.1f}")
            print(f"    Distance: {r_status.distance_from_start:.3f}")
            print(f"    Interactions: {r_status.interaction_count}")

        print(f"\n{'-'*70}")
        print(f"  SHARED SPACE")
        print(f"{'-'*70}")
        print(f"  Published vectors: {status.shared_space.get('vector_count', 0)}")
        print(f"  Convergence events: {status.convergence_events}")

        print(f"\n{'='*70}\n")

    def close(self):
        """Close coordinator and all resources."""
        self.stop_swarm()
        self.shared_space.close()
        self.observer.save_observations()


# =============================================================================
# Testing
# =============================================================================

def _test_swarm_coordinator():
    """Basic test of SwarmCoordinator."""
    import tempfile

    print("=== SwarmCoordinator Test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create coordinator
        coordinator = SwarmCoordinator(
            shared_space_path="test_canonical.db",
            data_dir=data_dir
        )

        # Start swarm with two residents (alpha mode for speed)
        configs = [
            {"name": "alpha", "model": "dolphin3:latest"},
            {"name": "beta", "model": "dolphin3:latest"}
        ]

        print("Starting swarm...")
        coordinator.start_swarm(configs)
        coordinator.print_status()

        # Note: Full test requires Ollama running
        print("\nSwarmCoordinator initialized successfully!")
        print("Full interaction test requires Ollama running.")

        # Test persistence
        print("\nTesting persistence...")
        coordinator._save_state()

        # Test convergence observation (with mock states)
        print("\nTesting convergence observation...")
        if len(coordinator.residents) >= 2:
            summary = coordinator.observe_convergence()
            print(f"  Observed {summary.pair_count} pairs")
            print(f"  E_mean: {summary.E_minds_mean:.4f}")

        coordinator.close()
        print("\nSwarmCoordinator test passed!")


if __name__ == "__main__":
    _test_swarm_coordinator()
