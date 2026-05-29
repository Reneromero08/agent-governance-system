"""
Shared Semantic Space for Multi-Resident Observation (P.1.1)

Enables multiple VectorResidents to:
- Publish vectors to a shared space
- Observe other residents' semantic states
- Record convergence events between residents

Architecture:
- Per-resident DBs remain private (mind_vector isolation)
- SharedSemanticSpace holds published/observable vectors
- Thread-safe via write lock (read-many, write-one)

Q44/Q45 validated: All similarity via E (Born rule)
"""

import sqlite3
import hashlib
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
from dataclasses import dataclass
import sys

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"))
from geometric_reasoner import GeometricState


@dataclass
class PublishedVector:
    """A vector published to the shared space."""
    vector_id: str
    publisher_id: str
    state: GeometricState
    category: str
    metadata: Dict
    created_at: str


@dataclass
class ConvergenceEvent:
    """A recorded convergence event between residents."""
    event_id: str
    timestamp: str
    resident_a: str
    resident_b: str
    E_value: float
    Df_a: float
    Df_b: float
    event_type: str
    metadata: Dict


class SharedSemanticSpace:
    """
    Shared vector space for multi-resident observation.

    P.1.1: Multiple residents can publish to and observe this space.
    Per-resident mind states remain private; only published forms are shared.

    Thread-safe via write lock (read-many, write-one pattern).

    Usage:
        space = SharedSemanticSpace("canonical_space.db")

        # Resident publishes their state
        vector_id = space.publish("resident_alpha", state, category="mind_snapshot")

        # Another resident observes
        neighbors = space.find_nearest(query_state, k=10, exclude_publisher="resident_beta")

        # Check other minds
        other_minds = space.get_other_minds("resident_beta")
    """

    SCHEMA = """
    -- Published vectors from all residents
    CREATE TABLE IF NOT EXISTS published_vectors (
        vector_id TEXT PRIMARY KEY,
        publisher_id TEXT NOT NULL,
        vec_blob BLOB NOT NULL,
        vec_sha256 TEXT NOT NULL,
        Df REAL NOT NULL,
        category TEXT NOT NULL,
        metadata JSON,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_pub_publisher ON published_vectors(publisher_id);
    CREATE INDEX IF NOT EXISTS idx_pub_category ON published_vectors(category);
    CREATE INDEX IF NOT EXISTS idx_pub_sha256 ON published_vectors(vec_sha256);
    CREATE INDEX IF NOT EXISTS idx_pub_created ON published_vectors(created_at);

    -- Convergence events between resident pairs
    CREATE TABLE IF NOT EXISTS convergence_events (
        event_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        resident_a TEXT NOT NULL,
        resident_b TEXT NOT NULL,
        E_value REAL NOT NULL,
        Df_a REAL,
        Df_b REAL,
        event_type TEXT NOT NULL,
        metadata JSON
    );

    CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON convergence_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_conv_type ON convergence_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_conv_pair ON convergence_events(resident_a, resident_b);

    -- Swarm state (key-value for coordination)
    CREATE TABLE IF NOT EXISTS swarm_state (
        key TEXT PRIMARY KEY,
        value JSON,
        updated_at TEXT NOT NULL
    );
    """

    # Categories for published vectors
    CATEGORY_MIND_SNAPSHOT = "mind_snapshot"
    CATEGORY_CANONICAL = "canonical"
    CATEGORY_PAPER = "paper"
    CATEGORY_CONCEPT = "concept"
    CATEGORY_INTERACTION = "interaction"

    # Event types for convergence
    EVENT_HIGH_RESONANCE = "high_resonance"
    EVENT_NOTATION_SHARED = "notation_shared"
    EVENT_CONCEPT_ALIGNED = "concept_aligned"
    EVENT_DF_CORRELATION = "df_correlation"

    def __init__(self, db_path: str = "canonical_space.db"):
        """
        Initialize shared semantic space.

        Args:
            db_path: Path to SQLite database (relative to FERAL_RESIDENT/data/)
        """
        # Resolve path relative to data directory
        base_dir = Path(__file__).parent / "data"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / db_path

        # Thread-safe write lock
        self._write_lock = Lock()

        # Connect with multi-thread support
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level="DEFERRED"
        )
        self.conn.row_factory = sqlite3.Row

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Create tables if not exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    # =========================================================================
    # Publishing
    # =========================================================================

    def publish(
        self,
        publisher_id: str,
        state: GeometricState,
        category: str = CATEGORY_CANONICAL,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Publish a vector to shared space.

        Thread-safe via write lock.

        Args:
            publisher_id: Unique identifier for the publishing resident
            state: GeometricState to publish
            category: Type of vector (mind_snapshot, canonical, paper, etc.)
            metadata: Optional metadata dict

        Returns:
            vector_id: Unique ID for the published vector
        """
        with self._write_lock:
            # Serialize vector
            vec_blob = state.vector.astype(np.float32).tobytes()
            vec_sha256 = hashlib.sha256(vec_blob).hexdigest()

            # Generate unique ID
            vector_id = f"{publisher_id}_{category}_{vec_sha256[:12]}"
            now = datetime.now(timezone.utc).isoformat()

            # Insert or replace (allows republishing)
            self.conn.execute("""
                INSERT OR REPLACE INTO published_vectors
                (vector_id, publisher_id, vec_blob, vec_sha256, Df, category, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vector_id,
                publisher_id,
                vec_blob,
                vec_sha256,
                float(state.Df),
                category,
                json.dumps(metadata or {}),
                now
            ))
            self.conn.commit()

            return vector_id

    def publish_mind_snapshot(
        self,
        publisher_id: str,
        state: GeometricState,
        interaction_count: int = 0,
        distance_from_start: float = 0.0
    ) -> str:
        """
        Convenience method to publish a mind snapshot.

        Args:
            publisher_id: Resident identifier
            state: Current mind state
            interaction_count: Number of interactions so far
            distance_from_start: Geodesic distance from initial state
        """
        return self.publish(
            publisher_id,
            state,
            category=self.CATEGORY_MIND_SNAPSHOT,
            metadata={
                "interaction_count": interaction_count,
                "distance_from_start": distance_from_start,
                "Df": float(state.Df)
            }
        )

    # =========================================================================
    # Querying
    # =========================================================================

    def find_nearest(
        self,
        query: GeometricState,
        k: int = 10,
        exclude_publisher: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Tuple[PublishedVector, float]]:
        """
        Find k nearest neighbors using E (Born rule).

        Args:
            query: GeometricState to query with
            k: Number of neighbors to return
            exclude_publisher: Optionally exclude vectors from this publisher
            category: Optionally filter by category

        Returns:
            List of (PublishedVector, E_value) tuples, sorted by E descending
        """
        # Build query
        sql = "SELECT * FROM published_vectors WHERE 1=1"
        params = []

        if exclude_publisher:
            sql += " AND publisher_id != ?"
            params.append(exclude_publisher)

        if category:
            sql += " AND category = ?"
            params.append(category)

        rows = self.conn.execute(sql, params).fetchall()

        # Compute E with each row
        results = []
        for row in rows:
            vec = np.frombuffer(row['vec_blob'], dtype=np.float32)
            other_state = GeometricState(vector=vec, operation_history=[])
            E = query.E_with(other_state)

            pub_vec = PublishedVector(
                vector_id=row['vector_id'],
                publisher_id=row['publisher_id'],
                state=other_state,
                category=row['category'],
                metadata=json.loads(row['metadata'] or '{}'),
                created_at=row['created_at']
            )
            results.append((pub_vec, E))

        # Sort by E descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_other_minds(
        self,
        exclude_publisher: str,
        limit: int = 100
    ) -> List[PublishedVector]:
        """
        Get mind snapshots from other residents.

        Args:
            exclude_publisher: Publisher to exclude (usually self)
            limit: Maximum number of snapshots to return

        Returns:
            List of PublishedVector objects for mind snapshots
        """
        rows = self.conn.execute("""
            SELECT * FROM published_vectors
            WHERE category = ? AND publisher_id != ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (self.CATEGORY_MIND_SNAPSHOT, exclude_publisher, limit)).fetchall()

        return [
            PublishedVector(
                vector_id=row['vector_id'],
                publisher_id=row['publisher_id'],
                state=GeometricState(
                    vector=np.frombuffer(row['vec_blob'], dtype=np.float32),
                    operation_history=[]
                ),
                category=row['category'],
                metadata=json.loads(row['metadata'] or '{}'),
                created_at=row['created_at']
            )
            for row in rows
        ]

    def get_publisher_vectors(
        self,
        publisher_id: str,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[PublishedVector]:
        """
        Get all vectors from a specific publisher.
        """
        sql = "SELECT * FROM published_vectors WHERE publisher_id = ?"
        params = [publisher_id]

        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            PublishedVector(
                vector_id=row['vector_id'],
                publisher_id=row['publisher_id'],
                state=GeometricState(
                    vector=np.frombuffer(row['vec_blob'], dtype=np.float32),
                    operation_history=[]
                ),
                category=row['category'],
                metadata=json.loads(row['metadata'] or '{}'),
                created_at=row['created_at']
            )
            for row in rows
        ]

    def get_all_publishers(self) -> List[str]:
        """Get list of all unique publisher IDs."""
        rows = self.conn.execute(
            "SELECT DISTINCT publisher_id FROM published_vectors"
        ).fetchall()
        return [row['publisher_id'] for row in rows]

    # =========================================================================
    # Convergence Events
    # =========================================================================

    def record_convergence_event(
        self,
        resident_a: str,
        resident_b: str,
        E_value: float,
        Df_a: float,
        Df_b: float,
        event_type: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record a convergence event between two residents.

        Args:
            resident_a: First resident ID
            resident_b: Second resident ID
            E_value: E(mind_a, mind_b) - Born rule resonance
            Df_a: Participation ratio of resident A
            Df_b: Participation ratio of resident B
            event_type: Type of convergence event
            metadata: Optional additional data

        Returns:
            event_id: Unique ID for the event
        """
        with self._write_lock:
            # Sort resident IDs for consistent pair ordering
            if resident_a > resident_b:
                resident_a, resident_b = resident_b, resident_a
                Df_a, Df_b = Df_b, Df_a

            now = datetime.now(timezone.utc).isoformat()

            # Generate event ID
            event_id = hashlib.sha256(
                f"{resident_a}{resident_b}{now}{E_value}".encode()
            ).hexdigest()[:16]

            self.conn.execute("""
                INSERT INTO convergence_events
                (event_id, timestamp, resident_a, resident_b, E_value, Df_a, Df_b, event_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                now,
                resident_a,
                resident_b,
                E_value,
                Df_a,
                Df_b,
                event_type,
                json.dumps(metadata or {})
            ))
            self.conn.commit()

            return event_id

    def get_convergence_events(
        self,
        resident_a: Optional[str] = None,
        resident_b: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ConvergenceEvent]:
        """
        Query convergence events with optional filters.
        """
        sql = "SELECT * FROM convergence_events WHERE 1=1"
        params = []

        if resident_a:
            sql += " AND (resident_a = ? OR resident_b = ?)"
            params.extend([resident_a, resident_a])

        if resident_b:
            sql += " AND (resident_a = ? OR resident_b = ?)"
            params.extend([resident_b, resident_b])

        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            ConvergenceEvent(
                event_id=row['event_id'],
                timestamp=row['timestamp'],
                resident_a=row['resident_a'],
                resident_b=row['resident_b'],
                E_value=row['E_value'],
                Df_a=row['Df_a'],
                Df_b=row['Df_b'],
                event_type=row['event_type'],
                metadata=json.loads(row['metadata'] or '{}')
            )
            for row in rows
        ]

    def get_pair_history(
        self,
        resident_a: str,
        resident_b: str,
        limit: int = 100
    ) -> List[ConvergenceEvent]:
        """Get convergence history for a specific pair."""
        # Normalize pair order
        if resident_a > resident_b:
            resident_a, resident_b = resident_b, resident_a

        rows = self.conn.execute("""
            SELECT * FROM convergence_events
            WHERE resident_a = ? AND resident_b = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (resident_a, resident_b, limit)).fetchall()

        return [
            ConvergenceEvent(
                event_id=row['event_id'],
                timestamp=row['timestamp'],
                resident_a=row['resident_a'],
                resident_b=row['resident_b'],
                E_value=row['E_value'],
                Df_a=row['Df_a'],
                Df_b=row['Df_b'],
                event_type=row['event_type'],
                metadata=json.loads(row['metadata'] or '{}')
            )
            for row in rows
        ]

    # =========================================================================
    # Swarm State
    # =========================================================================

    def set_swarm_state(self, key: str, value: Any):
        """Set a swarm state value."""
        with self._write_lock:
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute("""
                INSERT OR REPLACE INTO swarm_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), now))
            self.conn.commit()

    def get_swarm_state(self, key: str, default: Any = None) -> Any:
        """Get a swarm state value."""
        row = self.conn.execute(
            "SELECT value FROM swarm_state WHERE key = ?", (key,)
        ).fetchone()

        if row:
            return json.loads(row['value'])
        return default

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get statistics about the shared space."""
        vector_count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM published_vectors"
        ).fetchone()['cnt']

        event_count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM convergence_events"
        ).fetchone()['cnt']

        publisher_count = self.conn.execute(
            "SELECT COUNT(DISTINCT publisher_id) as cnt FROM published_vectors"
        ).fetchone()['cnt']

        category_counts = {}
        for row in self.conn.execute("""
            SELECT category, COUNT(*) as cnt FROM published_vectors GROUP BY category
        """).fetchall():
            category_counts[row['category']] = row['cnt']

        event_type_counts = {}
        for row in self.conn.execute("""
            SELECT event_type, COUNT(*) as cnt FROM convergence_events GROUP BY event_type
        """).fetchall():
            event_type_counts[row['event_type']] = row['cnt']

        return {
            'vector_count': vector_count,
            'event_count': event_count,
            'publisher_count': publisher_count,
            'category_counts': category_counts,
            'event_type_counts': event_type_counts,
            'db_path': str(self.db_path)
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Testing
# =============================================================================

def _test_shared_space():
    """Basic test of SharedSemanticSpace."""
    import tempfile
    import os

    # Use temp file for test
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = f.name

    try:
        # Create space
        space = SharedSemanticSpace(test_db)

        # Create test states
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.random.randn(384).astype(np.float32)

        state1 = GeometricState(vector=vec1, operation_history=[])
        state2 = GeometricState(vector=vec2, operation_history=[])

        # Test publishing
        id1 = space.publish("resident_alpha", state1, category="mind_snapshot")
        id2 = space.publish("resident_beta", state2, category="mind_snapshot")

        print(f"Published: {id1}, {id2}")

        # Test querying
        neighbors = space.find_nearest(state1, k=5)
        print(f"Nearest to state1: {len(neighbors)} results")
        for pub_vec, E in neighbors:
            print(f"  {pub_vec.publisher_id}: E={E:.4f}, Df={pub_vec.state.Df:.2f}")

        # Test other minds
        other_minds = space.get_other_minds("resident_alpha")
        print(f"Other minds (from alpha's view): {len(other_minds)}")

        # Test convergence events
        event_id = space.record_convergence_event(
            resident_a="resident_alpha",
            resident_b="resident_beta",
            E_value=state1.E_with(state2),
            Df_a=state1.Df,
            Df_b=state2.Df,
            event_type="high_resonance"
        )
        print(f"Recorded event: {event_id}")

        # Test stats
        stats = space.get_stats()
        print(f"Stats: {stats}")

        space.close()
        print("SharedSemanticSpace test passed!")

    finally:
        os.unlink(test_db)


if __name__ == "__main__":
    _test_shared_space()
